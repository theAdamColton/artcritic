"""
use a reward function, iterate over a wds of images,
rank them, and then display the top k and bottom k
"""

import pandas as pd
import torch
from artcritic.reward.llava import LlavaRewardSimpleRater
import webdataset as wds

def main(
        wds_path: str = "/home/figes/datasets/improved_aesthetics_6plus/laion-high-resolution/00{000..200}.tar",
        n:int = 100,
        ):
    #reward_fn = LlavaRewardSimpleRater()
    ds = wds.WebDataset(wds_path).decode("torchrgb").with_length(n)

    for data in ds:
        import bpdb
        bpdb.set_trace()


if __name__ =="__main__":
    main()

