from typing import Tuple
import torch

class Reward:
    def __call__(self, image, prompt) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
