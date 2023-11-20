import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import clip_loss
from artcritic.reward.reward import Reward
from peft import PeftModel
from datasets import load_dataset


class CFNReward(nn.Module):
    def __init__(self, inference_dtype=None, device=None, peft_model_url = "adams-story/cfn-dalle3", base_model_name = "openai/clip-vit-base-patch32"):
        super().__init__()
        print("loading cfn clip")
        model = CLIPModel.from_pretrained(base_model_name)
        if peft_model_url:
            model = PeftModel.from_pretrained(model,peft_model_url)
            model = model.merge_and_unload()

        self.processor = CLIPProcessor.from_pretrained(base_model_name)
        model = model.to(device, dtype=inference_dtype)
        model = torch.compile(model)
        #model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        self.model = model
        self.device = device
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        device=self.device

    def get_embeds(self, im_pix, batched_prompt_d):
        to_h = self.processor.image_processor.crop_size['height']
        to_w = self.processor.image_processor.crop_size['width']
        x_var = F.interpolate(im_pix, (to_h, to_w), antialias=True, mode="bilinear")
        x_var = self.normalize(x_var).to(im_pix.dtype)

        prompts = [ d['prompt'] for d in batched_prompt_d]

        with torch.no_grad():
            def _emb_prompts(prompts):
                text_inputs = self.processor(
                    text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                text_embeds = self.model.get_text_features(**text_inputs)
                text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
                return text_embeds
            text_embeds = _emb_prompts(prompts)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        image_embeds = self.model.get_image_features(x_var)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        return image_embeds, text_embeds

    def forward(self, im_pix, batched_prompt_d):
        image_embeds, text_embeds = self.get_embeds(im_pix, batched_prompt_d)
        loss = self.get_loss(image_embeds, text_embeds)
        return loss, -loss

    def get_loss(self, image_embeds, text_embeds):
        # contrastive clip reward score
        loss = clip_loss(text_embeds @ image_embeds.T * self.model.logit_scale)
        return loss

