import torch
from train import ModelArgs
import matplotlib.pyplot as plt


def main(
        model_args:ModelArgs = ModelArgs()
        ):

    pipeline= model_args.load_model().to(torch.float16).to('cuda')

    im = pipeline("Ken Follett Pillars of the Earth Cathedral Construction, dusty English Summer, Middle Ages, 4K landscape photograph, highly detailed", guidance_scale=model_args.sd_guidance_scale, num_inference_steps=model_args.model_steps).images[0]

    plt.imshow(im)
    plt.savefig("outfig.jpg")

if __name__=="__main__":
    import jsonargparse
    jsonargparse.CLI(main)
