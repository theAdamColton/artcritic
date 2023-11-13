from typing import Tuple
import torch

class Reward:
    def __call__(self, images, prompts, prompts_detailed) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
