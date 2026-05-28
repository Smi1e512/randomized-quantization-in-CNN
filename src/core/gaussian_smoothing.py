# 加性高斯噪声的随机平滑推理

import torch

from .base_smoothing import BaseSmoothing


class GaussianSmoothing(BaseSmoothing):
    def __init__(
        self,
        model,
        std: float = 0.1,
        n_samples: int = 50,
        num_classes: int = 10,
        device: str | torch.device = "cuda",
    ):
        super().__init__(
            model=model, n_samples=n_samples, num_classes=num_classes, device=device
        )
        self.std = std

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return torch.clamp(x + noise, 0.0, 1.0)
