"""CIFAR-10 DataLoader 工厂: 像素保持 [0,1] 仅 ToTensor 不做 mean/std 归一化, PGD 的 eps=8/255 直接对应任务书扰动幅度。"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


TEST_TRANSFORM = transforms.Compose([transforms.ToTensor()])

TRAIN_TRANSFORM_PLAIN = transforms.Compose([transforms.ToTensor()])

TRAIN_TRANSFORM_AUG = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)


def _make_loader(dataset, batch_size, shuffle, num_workers, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        persistent_workers=(num_workers > 0),
    )


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = True,
    seed: int | None = None,
):
    """构造训练+测试 DataLoader (供 train.py 用); augment=True 时启用 RandomCrop+HFlip。"""
    train_transform = TRAIN_TRANSFORM_AUG if augment else TRAIN_TRANSFORM_PLAIN
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=TEST_TRANSFORM
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    train_loader = _make_loader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers,
        generator=generator,
    )
    test_loader = _make_loader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, test_loader


def get_cifar10_test_loader(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    num_eval: int | None = None,
):
    """只构造测试集 DataLoader (供 evaluate.py / sweep_*.py); num_eval 截取前 N 张用于 smoke test。"""
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=TEST_TRANSFORM
    )
    if num_eval is not None:
        test_dataset = Subset(test_dataset, range(num_eval))
    return _make_loader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )
