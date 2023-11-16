import contextlib
from typing import Optional
from diffusers.pipelines.latent_consistency_models import LatentConsistencyModelPipeline
import torchvision
from tqdm import tqdm
import torch
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataclasses import dataclass, asdict
import wandb
import bitsandbytes as bnb

from diffusers import DiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline, LCMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from artcritic.patched_lcm_call import lcm_patched_call
from artcritic.patched_sd_call import sd_patched_call

from artcritic.prompts import DiffusionDBPromptUpscaled
from artcritic.reward.dummy import DummyReward
from artcritic.reward.hps import HPSReward
from artcritic.reward.llava import LlavaReward, LlavaRewardSimpleRater


logger = get_logger(__name__, log_level="INFO")

@dataclass
class TrainingArgs:
    save_freq:int = 500

    reward_type:str = "llava"
    
    precision:str  = "fp16"
    seed:int = 42    
    
    grad_checkpoint:bool = True
    
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

    enable_paged_adamw_32bit:bool=True
    enable_paged_adamw_8bit:bool=False
    enable_adamw_8bit:bool=False

    lora_rank:int = 1

    max_n_batches:int = 10000
    batch_size:int = 1
    gradient_accumulation_steps:int = 4

    resume_from:str = ""

    log_images_every: int = 8

    eval_every:int = 8

    eval_batch_size: int = 4


@dataclass
class ModelArgs:
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    #model_name_or_url:str = "Lykon/dreamshaper-7"
    model_name_or_url:str = "runwayml/stable-diffusion-v1-5"

    # name of lora model if any
    adapter_name_or_url: Optional[str] = "latent-consistency/lcm-lora-sdv1-5"

    adapter_name_or_url_2: Optional[str] = None

    is_lcm:bool = True

    # revision of the model to load.
    revision:str = "main"

    model_steps:int = 4

    sd_guidance_scale:float = 0.0

    torch_compile:bool = False

    def load_model(self) -> DiffusionPipeline:
        model_args = self
        # load scheduler, tokenizer and models.
        pipeline= DiffusionPipeline.from_pretrained(model_args.model_name_or_url, revision=model_args.revision)
        if model_args.is_lcm:
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)

        if model_args.adapter_name_or_url is not None:
            pipeline.load_lora_weights(model_args.adapter_name_or_url)
            pipeline.fuse_lora()

        if model_args.adapter_name_or_url_2 is not None:
            pipeline.load_lora_weights(model_args.adapter_name_or_url_2)
            pipeline.fuse_lora()
        
        # freeze parameters of models to save more memory
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.unet.requires_grad_(False)

        if model_args.torch_compile:
            pipeline.unet = torch.compile(pipeline.unet)

        pipeline.enable_xformers_memory_efficient_attention()
        #pipeline.enable_model_cpu_offload()

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

def main(train_args: TrainingArgs=TrainingArgs(),
         model_args: ModelArgs=ModelArgs(),
         ):

    generator = torch.Generator().manual_seed(train_args.seed)

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

    
    logger.info(f"\n{config_d}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(train_args.seed, device_specific=True)
    
    pipeline = model_args.load_model()


    if isinstance(pipeline, LatentConsistencyModelPipeline):
        patched_call = lcm_patched_call
    elif isinstance(pipeline, StableDiffusionPipeline):
        patched_call = sd_patched_call
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

    adam_args = dict(
            params=unet.parameters(),
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


    train_prompter = DiffusionDBPromptUpscaled(seed=train_args.seed, split='train')

    eval_prompter = DiffusionDBPromptUpscaled(seed=train_args.seed+1, split='test')

    autocast = contextlib.nullcontext
    
    unet, optimizer = accelerator.prepare(unet, optimizer)
    
    if train_args.reward_type=='llava':
        reward_fn = LlavaReward(inference_dtype=inference_dtype, device=accelerator.device)
    if train_args.reward_type=='llava-rater':
        reward_fn = LlavaRewardSimpleRater(inference_dtype=inference_dtype, device=accelerator.device)
    elif train_args.reward_type == "dummy":
        reward_fn = DummyReward(inference_dtype=inference_dtype, device=accelerator.device)
    elif train_args.reward_type == "hps":
        reward_fn = HPSReward(inference_dtype=inference_dtype, device=accelerator.device)
    else:
        raise NotImplementedError

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

    losses_to_log = 0.0

    unet.train()

    #################### TRAINING ####################        
    for i in tqdm(range(first_epoch, train_args.max_n_batches)):
        
        if accelerator.is_main_process:
            logger.info(f"{wandb.run.name} train_batch {i}: training")

        train_prompt_batch = [
                train_prompter() for _ in range(train_args.batch_size)
            ]

        train_prompts = [x[0]['prompt_original'] for x in train_prompt_batch]
        train_prompts = [p.strip().replace("\"","") for p in train_prompts]
        train_prompts_upscaled = [x[0]['prompt_upscaled'] for x in train_prompt_batch]
        train_prompts_upscaled = [p.strip().replace("\"","") for p in train_prompts_upscaled]

        with accelerator.accumulate(unet):
            with autocast():
                with torch.enable_grad(): # important b/c don't have on by default in module                        
                    ims = patched_call(pipeline, train_prompts, output_type="pt", guidance_scale=model_args.sd_guidance_scale, num_inference_steps=model_args.model_steps, use_gradient_checkpointing=train_args.grad_checkpoint, generator=generator).images

                    loss, reward = reward_fn(ims, train_prompts, train_prompts_upscaled)

                    losses_to_log += loss.item()
                    
                    logger.info(f"loss {loss.item():.4f}")

                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), train_args.max_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()                        

        if (i+1) % train_args.log_images_every == 0:
            train_images = []
            for j, image in enumerate(ims):
                image = image.clamp(0,1).cpu().detach()
                pil = torchvision.transforms.ToPILImage()(image)
                train_images.append(wandb.Image(pil, caption=f"{train_prompts[j]} | {train_prompts_upscaled[j]} | {reward.item():.2f}"))
            
            accelerator.log(
                    {"train":{"images": train_images, "loss":losses_to_log / train_args.log_images_every}},
                step=i,
            )

            losses_to_log = 0.0
        
        if (i+1) % train_args.save_freq == 0 and accelerator.is_main_process:
            accelerator.save_state("out/")

        if ((i+1) % train_args.eval_every == 0) or (i==0) or (i==train_args.max_n_batches-1):
            print("running eval...")
            eval_prompt_batch = eval_prompter.ds[:train_args.eval_batch_size]
            eval_prompts = eval_prompt_batch['prompt_original']
            eval_prompts = [p.strip().replace("\"","") for p in eval_prompts]
            eval_prompts_upscaled = eval_prompt_batch['prompt_upscaled']
            eval_prompts_upscaled = [p.strip().replace("\"","") for p in eval_prompts_upscaled]

            with torch.inference_mode():
                eval_generator = torch.Generator().manual_seed(420)
                ims = pipeline(eval_prompts,
                            output_type="pt",
                               guidance_scale=model_args.sd_guidance_scale,
                               num_inference_steps=model_args.model_steps,
                               use_gradient_checkpointing=train_args.grad_checkpoint,
                               generator=eval_generator,
                               ).images
                loss, reward = reward_fn(ims, eval_prompts, eval_prompts_upscaled)
            train_images = []
            for j, image in enumerate(ims):
                image = image.clamp(0,1).cpu().detach()
                pil = torchvision.transforms.ToPILImage()(image)
                train_images.append(wandb.Image(pil, caption=f"{eval_prompts[j]} | {eval_prompts_upscaled[j]} | {reward.item():.2f}"))
            
            accelerator.log(
                    {"test":{"images": train_images, "loss": loss}},
                step=i,
            )


if __name__ == "__main__":
    import jsonargparse
    jsonargparse.CLI(main)
