import torch
from transformers import CLIPModel, CLIPProcessor
import torchvision

from artcritic.reward.reward import Reward

class HPSReward(Reward):
    def __init__(self,
                 inference_dtype=None, device=None):
        model:CLIPModel=CLIPModel.from_pretrained("adams-story/HPSv2-hf")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = model.to(device, dtype=inference_dtype)
        model.eval()

        self.model = model
        self.target_size =  224
        self.device = device
        self.normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                    std=[0.26862954, 0.26130258, 0.27577711])
        
    def __call__(self, im_pix, prompts):    
        im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        x_var = torchvision.transforms.Resize(self.target_size)(im_pix)
        x_var = self.normalize(x_var).to(im_pix.dtype)        
        text_inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k:v.to(self.device) for k,v in text_inputs.items()}
        outputs = self.model(pixel_values=x_var, **text_inputs)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
