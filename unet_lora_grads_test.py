import torch
from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict

from artcritic.patched_sd_call import sd_patched_call
from train import ModelArgs

#pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to('cuda')
#pipe.safety_checker=None
#pipe.text_encoder.requires_grad_(False)
#pipe.unet.requires_grad_(False)
#pipe.vae.requires_grad_(False)
#unet=pipe.unet

ma = ModelArgs()
pipe = ma.load_model()
unet=pipe.unet

lora_config = LoraConfig(
    r=1,
    target_modules=[
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_in",
        "proj_out",
        "ff.net.0.proj",
        "ff.net.2",
        "conv1",
        "conv2",
        "conv_shortcut",
        "downsamplers.0.conv",
        "upsamplers.0.conv",
        "time_emb_proj",
    ],
)
unet = get_peft_model(unet, lora_config)


im=sd_patched_call(pipe,"hello", num_inference_steps=2, output_type='pt')
#pe,_=pipe.encode_prompt('hello', 'cuda', 1, False)
#x = unet(torch.randn(1,4,64,64).to(torch.float16).to('cuda'), 999, pe)
