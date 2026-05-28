"""适配 CIFAR-10 32x32 的 ResNet-50。"""

import torch.nn as nn
from torchvision.models import resnet50


def get_resnet50_cifar10(num_classes=10):
    """ResNet-50 在 32x32 小图上的标准改动: conv1 改 3x3 stride 1, 去掉首层 maxpool。"""
    model = resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(2048, num_classes)
    return model
