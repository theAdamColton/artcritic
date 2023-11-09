import contextlib
from typing import Tuple
import torchvision
from tqdm import tqdm
import torch
from accelerate import Accelerator
import torch.utils.checkpoint as checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataclasses import dataclass, asdict
import wandb
import random

from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from artcritic.prompts import DiffusionDBPromptUpscaled
from artcritic.reward.dummy import DummyReward
from artcritic.reward.llava import LlavaReward

@dataclass
class TrainingArgs:
    save_freq:int = 100

    reward_type:str = "llava"
    
    precision:str  = "fp16"
    # number of checkpoints to keep before overwriting old ones.
    num_checkpoint_limit:int = 10
    # random seed for reproducibility.
    seed:int = 42    

    truncated_backprop:bool = False
    truncated_backprop_rand:bool = False
    truncated_backprop_minmax:Tuple[int,int] = (35,45)
    trunc_backprop_timestep:int = 100
    
    grad_checkpoint:bool = True
    same_evaluation:bool = True
    
    loss_coeff:float = 1.0
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    use_8bit_adam:bool = False
    # learning rate.
    learning_rate:float = 3e-4
    # Adam beta1.
    adam_beta1:float = 0.9
    # Adam beta2.
    adam_beta2:float = 0.999
    # Adam weight decay.
    adam_weight_decay:float = 1e-4
    # Adam epsilon.
    adam_epsilon:float = 1e-8 
    # maximum gradient norm for gradient clipping.
    max_grad_norm:float = 1.0    
    grad_scale:int = 1
    lora_rank:int = 2

    max_n_batches:int = 1000
    batch_size:int = 1
    gradient_accumulation_steps:int = 1

    resume_from:str = ""

    log_images_every: int = 4


@dataclass
class ModelArgs:
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    model_name_or_url:str = "runwayml/stable-diffusion-v1-5"
    # revision of the model to load.
    revision:str = "main"

    model_steps:int = 20

    sd_guidance_scale:float = 7.5

def main(train_args: TrainingArgs=TrainingArgs(),
         model_args: ModelArgs=ModelArgs(),
         ):
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=train_args.precision,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
    )

    config_d = {"train_" + k: v for k,v in asdict(train_args).items()}
    config_d.update({"model_" + k: v for k,v in asdict(model_args).items()})
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="align-prop", config=config_d,
        )

    logger = get_logger(__name__, log_level="INFO")

    
    logger.info(f"\n{config_d}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(train_args.seed, device_specific=True)
    
    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(model_args.model_name_or_url, revision=model_args.revision)
    
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)


    # disable safety checker
    pipeline.safety_checker = None    
    
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.scheduler.set_timesteps(model_args.model_steps)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.    
    if train_args.precision == "fp16":
        inference_dtype = torch.float16
    elif train_args.precision == "bf16":
        inference_dtype = torch.bfloat16
    else:
        raise ValueError(train_args.precision)

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    pipeline.unet.to(accelerator.device, dtype=inference_dtype)    
    # Set correct lora layers
    lora_attn_procs = {}
    for name in pipeline.unet.attn_processors.keys():
        cross_attention_dim = (
            None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = pipeline.unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = pipeline.unet.config.block_out_channels[block_id]
        else:
            raise ValueError(f"{name} not recognized")

        lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=train_args.lora_rank)
    pipeline.unet.set_attn_processor(lora_attn_procs)

    # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
    # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
    # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
    class _Wrapper(AttnProcsLayers):
        def forward(self, *args, **kwargs):
            return pipeline.unet(*args, **kwargs)

    unet = _Wrapper(pipeline.unet.attn_processors)        

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        output_splits = output_dir.split("/")
        output_splits[1] = wandb.run.name
        output_dir = "/".join(output_splits)
        assert len(models) == 1
        if isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    accelerator.register_save_state_pre_hook(save_model_hook)


    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=train_args.learning_rate,
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        weight_decay=train_args.adam_weight_decay,
        eps=train_args.adam_epsilon,
    )

    train_prompter = DiffusionDBPromptUpscaled(seed=train_args.seed, split='train')

    # TODO no val dataset
    eval_prompter = DiffusionDBPromptUpscaled(seed=train_args.seed+1, split='train')

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]

    train_neg_prompt_embeds = neg_prompt_embed.repeat(train_args.batch_size, 1, 1)

    autocast = contextlib.nullcontext
    
    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)
    
    if train_args.reward_type=='llava':
        reward_fn = LlavaReward(inference_dtype=inference_dtype, device=accelerator.device)
    elif train_args.reward_type == "dummy":
        reward_fn = DummyReward(inference_dtype=inference_dtype, device=accelerator.device)
    else:
        raise NotImplementedError

    timesteps = pipeline.scheduler.timesteps

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                model_args.model_name_or_url, revision=model_args.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_load_state_pre_hook(load_model_hook)

    
    if train_args.resume_from:
        logger.info(f"Resuming from {train_args.resume_from}")
        accelerator.load_state(train_args.resume_from)

    first_epoch = 0 
    global_step = 0

    #################### TRAINING ####################        
    for i in list(range(first_epoch, train_args.max_n_batches)):
        unet.train()
        
        latent = torch.randn((train_args.batch_size, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)    

        if accelerator.is_main_process:
            logger.info(f"{wandb.run.name} train_batch {i}: training")

        prompt_batch = [
                train_prompter() for _ in range(train_args.batch_size)
            ]

        prompts = [x[0]['prompt_original'] for x in prompt_batch]
        prompts = [p.strip().replace("\"","") for p in prompts]
        prompts_upscaled = [x[0]['prompt_upscaled'] for x in prompt_batch]
        prompts_upscaled = [p.strip().replace("\"","") for p in prompts_upscaled]

        prompt_ids = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)   

        pipeline.scheduler.alphas_cumprod = pipeline.scheduler.alphas_cumprod.to(accelerator.device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]         
        
        with accelerator.accumulate(unet):
            with autocast():
                with torch.enable_grad(): # important b/c don't have on by default in module                        
                    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                        t = torch.tensor([t],
                                            dtype=inference_dtype,
                                            device=latent.device)
                        t = t.repeat(train_args.batch_size)
                        
                        if train_args.grad_checkpoint:
                            noise_pred_uncond = checkpoint.checkpoint(unet, latent, t, train_neg_prompt_embeds, use_reentrant=False).sample
                            noise_pred_cond = checkpoint.checkpoint(unet, latent, t, prompt_embeds, use_reentrant=False).sample
                        else:
                            noise_pred_uncond = unet(latent, t, train_neg_prompt_embeds).sample
                            noise_pred_cond = unet(latent, t, prompt_embeds).sample
                                                        
                        if train_args.truncated_backprop:
                            if train_args.truncated_backprop_rand:
                                timestep = random.randint(train_args.truncated_backprop_minmax[0],train_args.truncated_backprop_minmax[1])
                                if i < timestep:
                                    noise_pred_uncond = noise_pred_uncond.detach()
                                    noise_pred_cond = noise_pred_cond.detach()
                            else:
                                if i < train_args.trunc_backprop_timestep:
                                    noise_pred_uncond = noise_pred_uncond.detach()
                                    noise_pred_cond = noise_pred_cond.detach()

                        grad = (noise_pred_cond - noise_pred_uncond)
                        noise_pred = noise_pred_uncond + model_args.sd_guidance_scale * grad                
                        latent = pipeline.scheduler.step(noise_pred, t[0].long(), latent).prev_sample
                                            
                    ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype) / 0.18215).sample
                    
                    loss, rewards = reward_fn(ims, prompts_upscaled)
                    
                    logger.info(f"loss {loss.item():.4f}")

                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), train_args.max_grad_norm)

                    _s = 0.0
                    for p in unet.parameters():
                        if p.requires_grad:
                            _s += p.grad.sum() * 1000
                    print("grad sum", _s)

                    optimizer.step()
                    optimizer.zero_grad()                        

                    if (i+1) % train_args.log_images_every == 0:
                        images = []
                        for i, image in enumerate(ims):
                            image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = torchvision.transforms.ToPILImage()(image)
                            images.append(wandb.Image(pil, caption=f"{prompts[i]} | {prompts_upscaled[i]} | {rewards:.2f}"))
                        
                        accelerator.log(
                            {"images": images, "loss":loss},
                            step=global_step,
                        )

                global_step += 1

        
        if (i+1) % train_args.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state("out/")


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
