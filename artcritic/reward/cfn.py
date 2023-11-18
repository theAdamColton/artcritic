import torch
import torch.nn.functional as F
import torchvision
from transformers import CLIPModel, CLIPProcessor
from artcritic.reward.reward import Reward


class CFNReward(Reward):
    def __init__(self, inference_dtype=None, device=None):
        print("loading cfn clip")
        model: CLIPModel = CLIPModel.from_pretrained("adams-story/cfn-dalle3")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
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

    def __call__(self, im_pix, batched_prompt_d):
        to_h = self.processor.image_processor.crop_size['height']
        to_w = self.processor.image_processor.crop_size['width']
        x_var = F.interpolate(im_pix, (to_h, to_w), antialias=False, mode="nearest")
        x_var = self.normalize(x_var).to(im_pix.dtype)
        prompts = [ d['prompt_upscaled'] for d in batched_prompt_d]
        text_inputs = self.processor(
            text=prompts, return_tensors="pt", padding="max_length", truncation=True
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embeds = self.model.get_image_features(x_var)
        text_embeds = self.model.get_text_features(**text_inputs)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # simply increases the non contrastive hps reward score
        scores = (image_embeds * text_embeds).sum(-1) * self.model.logit_scale

        score = scores.mean()

        return -score, score

