"""基于随机量化的随机平滑推理 (RQ-Smoothing)。"""

import torch

from .base_smoothing import BaseSmoothing
from .randomized_quantization import RandomizedQuantizationAugModule


class RQSmoothing(BaseSmoothing):
    def __init__(
        self,
        model,
        n_bins: int = 8,
        n_samples: int = 50,
        num_classes: int = 10,
        device: str | torch.device = "cuda",
    ):
        super().__init__(
            model=model, n_samples=n_samples, num_classes=num_classes, device=device
        )
        self.n_bins = n_bins
        self.rq = RandomizedQuantizationAugModule(region_num=n_bins).to(self.device)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.rq(x)
