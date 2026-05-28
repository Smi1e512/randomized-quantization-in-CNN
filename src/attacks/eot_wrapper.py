"""EOT (Expectation over Transformations) 包装: 对随机变换采样 k 次平均 logit, 让攻击者梯度对随机性鲁棒, 避免朴素 PGD 高估鲁棒性。"""

import torch
import torch.nn as nn


class EOTWrapper(nn.Module):
    def __init__(self, model: nn.Module, transform, k: int = 10):
        super().__init__()
        self.model = model
        self.transform = transform
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k <= 1:
            return self.model(self.transform(x))
        logits = 0
        for _ in range(self.k):
            logits = logits + self.model(self.transform(x))
        return logits / self.k
