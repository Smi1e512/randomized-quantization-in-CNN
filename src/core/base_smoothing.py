"""随机平滑推理基类: 对单图采 N 次随机视图 -> softmax 平均 -> argmax (软投票); 子类只需实现 transform(x)。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSmoothing:
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        num_classes: int = 10,
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.n_samples = n_samples
        self.num_classes = num_classes
        self.device = torch.device(device) if isinstance(device, str) else device

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """子类实现: 对输入做一次随机变换。"""
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        batch_size = x.size(0)
        total_probs = torch.zeros(batch_size, self.num_classes, device=self.device)

        for _ in range(self.n_samples):
            x_view = self.transform(x)
            logits = self.model(x_view)
            total_probs = total_probs + F.softmax(logits, dim=1)

        return (total_probs / self.n_samples).argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """同 predict, 但返回平均概率, 用于可视化置信度。"""
        self.model.eval()
        batch_size = x.size(0)
        total_probs = torch.zeros(batch_size, self.num_classes, device=self.device)

        for _ in range(self.n_samples):
            x_view = self.transform(x)
            logits = self.model(x_view)
            total_probs = total_probs + F.softmax(logits, dim=1)

        return total_probs / self.n_samples
