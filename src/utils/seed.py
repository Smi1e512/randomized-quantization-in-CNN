# 统一固定 Python / NumPy / PyTorch (CPU+CUDA) 的随机种子

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    # deterministic=True 启用 cuDNN 确定性模式, 严格复现但部分算子会变慢
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
