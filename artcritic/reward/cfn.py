import torchvision
from transformers import CLIPModel, CLIPProcessor
from artcritic.reward.hps import HPSReward


class CFNReward(HPSReward):
    def __init__(self, inference_dtype=None, device=None):
        print("loading cfn clip")
        model: CLIPModel = CLIPModel.from_pretrained("adams-story/cfn-dalle3")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        model = model.to(device, dtype=inference_dtype)
        #model.gradient_checkpointing_enable()
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()

        self.model = model
        self.target_size = 224
        self.device = device
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

