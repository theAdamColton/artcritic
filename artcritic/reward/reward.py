from typing import Dict, List, Tuple
import torch

class Reward:
    def __call__(self, images, batched_prompt_d: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
