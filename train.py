import numpy as np
import os
from typing import Optional, Tuple
from diffusers.pipelines.latent_consistency_models import LatentConsistencyModelPipeline
import torchvision
from tqdm import tqdm
import torch
from dataclasses import dataclass, asdict
import wandb
import bitsandbytes as bnb
from accelerate import Accelerator

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from diffusers import (
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    LCMScheduler,
    AutoencoderKL,
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as LoRAAttnProcessor
from diffusers.loaders.utils import AttnProcsLayers
from artcritic.patched_lcm_call import lcm_patched_call
from artcritic.patched_sd_call import sd_patched_call
from artcritic import patched_sdxl_call

from artcritic.prompts import DiffusionDBPromptQA, DiffusionDB
from artcritic.reward.cfn import CFNReward
from artcritic.reward.dummy import DummyReward



@dataclass
class TrainingArgs:
    reward_type: str = "llava"

    use_mixed_precision: bool = True
    precision: str = "fp16"
    device:str = "cuda"
    seed: int = 42

    grad_checkpoint: bool = True

    # learning rate.
    learning_rate: float = 3e-4
    # Adam beta1.
    adam_beta1: float = 0.9
    # Adam beta2.
    adam_beta2: float = 0.999
    # Adam weight decay.
    adam_weight_decay: float = 1e-4
    # Adam epsilon.
    adam_epsilon: float = 1e-8
    # maximum gradient norm for gradient clipping.
    max_grad_norm: float = 1.0

    enable_paged_adamw_32bit: bool = True
    enable_paged_adamw_8bit: bool = False
    enable_adamw_8bit: bool = False

    lora_rank: int = 1

    max_n_batches: int = 10000
    batch_size: int = 1
    gradient_accumulation_steps: int = 4

    # this only works with cfn loss
    accum_freq: int = 1

    # since accum batches are run with no grad, you can usually fit higher batch sizes
    accum_batch_size: int = 12

    resume_from: str = ""

    log_images_every: int = 8

    save_every: int = 500

    eval_every: int = 8

    eval_batch_size: int = 4

    image_height: int = 512
    image_width:int = 512


@dataclass
class ModelArgs:
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    model_name_or_url: str = "runwayml/stable-diffusion-v1-5"

    vae_model_name_or_url: str = ""#"madebyollin/sdxl-vae-fp16-fix"

    # name of lora model if any
    adapter_name_or_url: Optional[str] = "latent-consistency/lcm-lora-sdv1-5"

    adapter_name_or_url_2: Optional[str] = None

    is_lcm: bool = True

    variant: str = "fp16"

    model_steps: int = 4

    sd_guidance_scale: float = 0.0

    torch_compile: bool = False

    def load_model(self) -> DiffusionPipeline:
        model_args = self
        # load scheduler, tokenizer and models.
        pipeline = DiffusionPipeline.from_pretrained(
            model_args.model_name_or_url, variant=model_args.variant, torch_dtype=torch.float16,
        )
        if model_args.is_lcm:
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

        if model_args.vae_model_name_or_url:
            vae = AutoencoderKL.from_pretrained(model_args.vae_model_name_or_url)
            pipeline.vae = vae

        if model_args.adapter_name_or_url:
            pipeline.load_lora_weights(model_args.adapter_name_or_url)
            pipeline.fuse_lora()
            pipeline.unload_lora_weights()

        if model_args.adapter_name_or_url_2:
            pipeline.load_lora_weights(model_args.adapter_name_or_url_2)
            pipeline.fuse_lora()

        # freeze parameters of models to save more memory
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.unet.requires_grad_(False)

        if model_args.torch_compile:
            pipeline.unet = torch.compile(pipeline.unet)
            pipeline.vae = torch.compile(pipeline.vae)

        pipeline.enable_vae_tiling()
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()

        # doesnt work with lora
        #pipeline.enable_xformers_memory_efficient_attention()
        # pipeline.enable_model_cpu_offload()

        # disable safety checker
        pipeline.safety_checker = None

        # make the progress bar nicer
        pipeline.set_progress_bar_config(
            position=1,
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        return pipeline


def main(
    train_args: TrainingArgs = TrainingArgs(),
    model_args: ModelArgs = ModelArgs(),
):

    do_accum = train_args.accum_freq > 1

    assert train_args.log_images_every % train_args.accum_freq == 0

    config_d = {"train_" + k: v for k, v in asdict(train_args).items()}
    config_d.update({"model_" + k: v for k, v in asdict(model_args).items()})

    wandb.init(project='artcritic', config = config_d)

    print(f"\n{config_d}")

    pipeline = model_args.load_model()

    if isinstance(pipeline, LatentConsistencyModelPipeline):
        patched_call = lcm_patched_call
    elif isinstance(pipeline, StableDiffusionPipeline):
        patched_call = sd_patched_call
    elif isinstance(pipeline, StableDiffusionXLPipeline):
        patched_call = patched_sdxl_call.patched_call
    else:
        raise ValueError(f"unrecognized pipeline class! {type(pipeline)}")

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    if train_args.precision == "fp16":
        inference_dtype = torch.float16
    elif train_args.precision == "bf16":
        inference_dtype = torch.bfloat16
    else:
        raise ValueError(train_args.precision)

    lora_config = LoraConfig(
        r=train_args.lora_rank,
        lora_alpha=train_args.lora_rank*2,
        target_modules=[
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            #"conv1",
            #"conv2",
            #"conv_shortcut",
            #"downsamplers.0.conv",
            #"upsamplers.0.conv",
            "time_emb_proj",
        ],
    )
    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    lora_parameters = [p for p in pipeline.unet.parameters() if p.requires_grad]

    print("UNET  ", end="")

    pipeline.unet.print_trainable_parameters()

    pipeline = pipeline.to(train_args.device)

    adam_args = dict(
        params=lora_parameters,
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )
    if train_args.enable_paged_adamw_32bit:
        optimizer = bnb.optim.PagedAdam32bit(**adam_args)
    elif train_args.enable_paged_adamw_8bit:
        optimizer = bnb.optim.PagedAdam8bit(**adam_args)
    elif train_args.enable_adamw_8bit:
        optimizer = bnb.optim.Adam8bit(**adam_args)
    else:
        optimizer = torch.optim.AdamW(**adam_args)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = train_args.max_n_batches // train_args.gradient_accumulation_steps)

    train_prompter = DiffusionDB(seed=train_args.seed, split="train")
    eval_prompter = DiffusionDB(seed=train_args.seed + 1, split="test")

    if train_args.reward_type == "llava":
        from artcritic.reward.llava import LlavaReward, LlavaRewardSimpleRater, LlavaQA
        rewarder = LlavaReward(
            inference_dtype=inference_dtype, device=train_args.device
        )
    elif train_args.reward_type == "llava-qa":
        from artcritic.reward.llava import LlavaReward, LlavaRewardSimpleRater, LlavaQA
        rewarder = LlavaQA(
            inference_dtype=inference_dtype, device=train_args.device
        )
    elif train_args.reward_type == "llava-rater":
        from artcritic.reward.llava import LlavaReward, LlavaRewardSimpleRater, LlavaQA
        rewarder = LlavaRewardSimpleRater(
            inference_dtype=inference_dtype, device=train_args.device
        )
    elif train_args.reward_type == "dummy":
        rewarder = DummyReward(
            inference_dtype=inference_dtype, device=train_args.device
        )
    elif train_args.reward_type == "hps":
        rewarder = CFNReward(inference_dtype=inference_dtype, device=train_args.device, base_model_name="adams-story/HPSv2-hf", peft_model_url=None)
    elif train_args.reward_type == "cfn":
        rewarder = CFNReward(inference_dtype=inference_dtype, device=train_args.device,)
    else:
        raise NotImplementedError

    if train_args.use_mixed_precision:
        _acc_precision = train_args.precision
    else:
        _acc_precision = "no"

    accelerator = Accelerator(mixed_precision=_acc_precision, gradient_accumulation_steps=train_args.gradient_accumulation_steps)

    pipeline = pipeline.to(accelerator.device).to(inference_dtype)
    if train_args.use_mixed_precision:
        pipeline.unet = pipeline.unet.float()

    pipeline.unet, optimizer, lr_scheduler = accelerator.prepare(pipeline.unet, optimizer, lr_scheduler)


    first_epoch = 0

    losses_to_log = []

    pipeline.unet.train()

    eval_prompt_batch = [eval_prompter.ds[i] for i in range(train_args.eval_batch_size)]
    eval_prompts = [x["prompt"] for x in eval_prompt_batch]
    eval_prompts = [p.strip().replace('"', "") for p in eval_prompts]


    accum_texts = []
    accum_image_emb = []
    accum_text_emb = []
    accum_randn_noise = []

    num_channels_latents = pipeline.unet.config.in_channels

    generator = torch.Generator(device=accelerator.device).manual_seed(train_args.seed)

    #################### TRAINING ####################
    for i in tqdm(range(first_epoch, train_args.max_n_batches)):
        with accelerator.accumulate(pipeline.unet):
            # uses accum_batch_size
            if do_accum:
                train_prompt_batch = [train_prompter() for _ in range(train_args.accum_batch_size)]
            else:
                train_prompt_batch = [train_prompter() for _ in range(train_args.batch_size)]

            train_prompts = [x["prompt"] for x in train_prompt_batch]

            if not do_accum:
                with accelerator.autocast():
                    ims = patched_call(
                        pipeline,
                        train_prompts,
                        output_type="pt",
                        guidance_scale=model_args.sd_guidance_scale,
                        num_inference_steps=model_args.model_steps,
                        use_gradient_checkpointing=train_args.grad_checkpoint,
                        generator=generator,
                        height = train_args.image_height,
                        width = train_args.image_width,
                    ).images
                    ims = ims.to(inference_dtype)
                    loss, reward = rewarder(ims, train_prompt_batch)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                losses_to_log.append(loss.item())

                print(f"loss {loss.item():.4E}")

            else:
                # First, cache the features without any gradient tracking.
                # TODO this only works for CFG == 0
                randn_latents = torch.randn((train_args.accum_batch_size, num_channels_latents, train_args.image_height//pipeline.vae_scale_factor, train_args.image_width // pipeline.vae_scale_factor), device=accelerator.device, dtype=inference_dtype, generator=generator)
                accum_randn_noise.append(randn_latents)

                with torch.no_grad():
                    with accelerator.autocast():
                        ims = patched_call(
                            pipeline,
                            train_prompts,
                            output_type="pt",
                            guidance_scale=model_args.sd_guidance_scale,
                            num_inference_steps=model_args.model_steps,
                            use_gradient_checkpointing=train_args.grad_checkpoint,
                            latents=randn_latents,
                            height = train_args.image_height,
                            width = train_args.image_width,
                        ).images
                        ims = ims.to(inference_dtype)
                        im_embeds, text_embeds = rewarder.get_embeds(ims, train_prompt_batch)
                accum_image_emb.append(im_embeds)
                accum_text_emb.append(text_embeds)
                accum_texts.append(train_prompts)
                
                if ((i + 1) % train_args.accum_freq) == 0:
                    # Now, ready to take gradients for the last accum_freq batches.
                    # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
                    # Call backwards each time, but only step optimizer at the end.

                    print("taking loss on accum batch")
                    for j in range(train_args.accum_freq):

                        n_minibatches = len(accum_texts[j]) // train_args.batch_size

                        for k in range(n_minibatches):
                            start_i = k * train_args.batch_size
                            end_i = (k+1) * train_args.batch_size

                            curr_texts = accum_texts[j][start_i:end_i]
                            curr_latents = accum_randn_noise[j][start_i:end_i]

                            with accelerator.autocast():
                                ims = patched_call(
                                    pipeline,
                                    curr_texts,
                                    latents=curr_latents,
                                    output_type="pt",
                                    guidance_scale=model_args.sd_guidance_scale,
                                    num_inference_steps=model_args.model_steps,
                                    use_gradient_checkpointing=train_args.grad_checkpoint,
                                    height = train_args.image_height,
                                    width = train_args.image_width,
                                ).images

                                im_embeds, text_embeds = rewarder.get_embeds(ims, [{'prompt': x} for x in curr_texts])

                                past_accum_im_embeds = accum_image_emb[:j]
                                future_accum_im_embeds = accum_image_emb[j+1:]
                                mini_past_im_embeds = accum_image_emb[j][:start_i]
                                mini_future_im_embeds = accum_image_emb[j][end_i:]

                                all_im_embeds = torch.cat(past_accum_im_embeds + [mini_past_im_embeds] + [im_embeds] + [mini_future_im_embeds] + future_accum_im_embeds)

                                past_accum_txt_embeds = accum_text_emb[:j]
                                future_accum_txt_embeds = accum_text_emb[j+1:]
                                mini_past_txt_embeds = accum_text_emb[j][:start_i]
                                mini_future_txt_embeds = accum_text_emb[j][end_i:]
                                all_text_embeds = torch.cat(past_accum_txt_embeds + [mini_past_txt_embeds] + [text_embeds] + [mini_future_txt_embeds] + future_accum_txt_embeds)

                            loss = rewarder.get_loss(all_im_embeds, all_text_embeds)
                            print(f"accum loss {loss.item():.4f}")
                            reward = -loss
                            # hopefully doesn't underflow with all the division
                            accelerator.backward(loss / (train_args.accum_freq * n_minibatches))
                            losses_to_log.append(loss.item())

                    accum_texts, accum_image_emb, accum_text_emb = [], [], []
                    print("done taking loss on accum batch")

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(lora_parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()



        if (i + 1) % train_args.log_images_every == 0:
            train_images = []
            for j, image in enumerate(ims):
                image = image.clamp(0, 1).cpu().detach()
                pil = torchvision.transforms.ToPILImage()(image)
                train_images.append(
                    wandb.Image(
                        pil,
                        caption=f"{curr_texts[j]} | {reward.item():.2f}",
                    )
                )

            wandb.log(
                {
                    "train": {
                        "images": train_images,
                        "loss": np.array(losses_to_log).mean(),
                    }
                },
                step=i,
            )

            losses_to_log = []

        if (
            ((i + 1) % train_args.eval_every == 0)
            or (i == 0)
            or (i == train_args.max_n_batches - 1)
        ):
            print("running eval...")

            with torch.inference_mode():
                eval_generator = torch.Generator(device=accelerator.device).manual_seed(420)
                randn_latents = torch.randn((train_args.accum_batch_size, num_channels_latents, train_args.image_height//pipeline.vae_scale_factor, train_args.image_width // pipeline.vae_scale_factor), device=accelerator.device, dtype=inference_dtype, generator=eval_generator)
                pipeline.unet.eval()
                with accelerator.autocast():
                    ims = patched_call(
                        pipeline,
                        eval_prompts,
                        output_type="pt",
                        guidance_scale=model_args.sd_guidance_scale,
                        num_inference_steps=model_args.model_steps,
                        latents = randn_latents,
                    ).images
                    loss, reward = rewarder(ims, eval_prompt_batch)
                pipeline.unet.train()
            train_images = []
            for j, image in enumerate(ims):
                image = image.clamp(0, 1).cpu().detach()
                pil = torchvision.transforms.ToPILImage()(image)
                train_images.append(
                    wandb.Image(
                        pil,
                        caption=f"{eval_prompt_batch[j]['prompt']} | {reward.item():.2f}",
                    )
                )

            wandb.log(
                {"test": {"images": train_images, "loss": loss}},
                step=i,
            )

        if i % train_args.save_every == 0 and i > 0:
            print("saving model...")
            pipeline.unet.save_pretrained("./out/")
            lora_state_dict = get_peft_model_state_dict(pipeline.unet)
            StableDiffusionPipeline.save_lora_weights(os.path.join("./out/", "unet_lora/"), lora_state_dict)


    pipeline.unet.save_pretrained("./out/")
    lora_state_dict = get_peft_model_state_dict(pipeline.unet)
    StableDiffusionPipeline.save_lora_weights(os.path.join("./out/", "unet_lora/"), lora_state_dict)


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(main)
