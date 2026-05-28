"""BPDA (Athalye et al. ICML 2018): 前向走真 transform, 反向把 transform 视为 identity, 让攻击者绕过不可导防御求梯度; 与 EOTWrapper 正交可叠加。"""

import torch
import torch.nn as nn


class BPDAWrapper(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = self.transform(x)
        return x + (y - x).detach()  # 前向 = y, 反向 ∂out/∂x = I
