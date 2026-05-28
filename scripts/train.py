"""ResNet-50 + CIFAR-10 训练入口; --mode 取值 baseline/gaussian/rq, 可用 --resume 从 last_<mode>.pth 续训。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core import RandomizedQuantizationAugModule
from src.data_utils import get_cifar10_loaders
from src.models import get_resnet50_cifar10
from src.utils import CSVLogger, set_seed

import argparse
import os
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-50 on CIFAR-10")
    parser.add_argument("--mode", type=str, default="baseline", choices=["baseline", "gaussian", "rq"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_augment", action="store_true", help="关掉 RandomCrop+HFlip (消融)")
    parser.add_argument("--n_bins", type=int, default=8, help="RQ 分箱数 (仅 rq 模式)")
    parser.add_argument("--gaussian_std", type=float, default=0.1, help="高斯噪声标准差 (仅 gaussian 模式)")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./results/logs")
    parser.add_argument("--resume", type=str, default=None, help="从 last_<mode>.pth 续训")
    return parser.parse_args()


def train_one_epoch(
    model, train_loader, criterion, optimizer, device, args, rq_module=None, scaler=None
):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        if args.mode == "gaussian":
            noise = torch.randn_like(images) * args.gaussian_std
            images = torch.clamp(images + noise, 0, 1)
        elif args.mode == "rq":
            images = rq_module(images)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def evaluate(model, test_loader, criterion, device, args, rq_module=None):
    """训练时的轻量监控评测, 用与训练一致的输入变换保证 BN 统计量匹配; 真正的鲁棒/clean 评测由 evaluate.py 完成。"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            if args.mode == "gaussian":
                noise = torch.randn_like(images) * args.gaussian_std
                images = torch.clamp(images + noise, 0, 1)
            elif args.mode == "rq" and rq_module is not None:
                images = rq_module(images)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def _mode_tag(args):
    """rq 且 n_bins != 8 时加后缀 rq_n<k>, 否则原样 (n=8 为默认值保持向后兼容)。"""
    if args.mode == "rq" and args.n_bins != 8:
        return f"rq_n{args.n_bins}"
    return args.mode


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = _mode_tag(args)
    print(f"Using device: {device}")
    print(f"Training mode: {args.mode}, tag: {tag}, n_bins: {args.n_bins}, seed: {args.seed}, augment: {not args.no_augment}")

    os.makedirs(args.save_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, f"train_log_{tag}.csv")
    fieldnames = [
        "epoch", "train_loss", "train_acc", "test_loss", "test_acc",
        "lr", "time_sec", "is_best",
    ]
    if args.resume and os.path.exists(log_path):
        logger = CSVLogger.append(log_path, fieldnames)
        print(f"训练日志将追加到: {log_path}")
    else:
        logger = CSVLogger(log_path, fieldnames)
        print(f"训练日志将保存到: {log_path}")

    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=(not args.no_augment),
        seed=args.seed,
    )

    model = get_resnet50_cifar10(num_classes=10).to(device)

    rq_module = None
    if args.mode == "rq":
        rq_module = RandomizedQuantizationAugModule(region_num=args.n_bins).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_acc = 0.0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume 路径不存在: {args.resume}")
        print(f"从 {args.resume} 续训...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"  -> 从 epoch {start_epoch} 开始, best_acc={best_acc:.2f}%")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args, rq_module, scaler
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, args, rq_module
        )
        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch [{epoch}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  "
            f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%  "
            f"LR: {current_lr:.6f}  Time: {elapsed:.1f}s"
        )

        is_best = 0
        if test_acc > best_acc:
            best_acc = test_acc
            best_path = os.path.join(args.save_dir, f"best_{tag}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best model saved to {best_path} (acc: {best_acc:.2f}%)")
            is_best = 1

        last_path = os.path.join(args.save_dir, f"last_{tag}.pth")
        torch.save(
            {
                "epoch": epoch,
                "best_acc": best_acc,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "args": vars(args),
            },
            last_path,
        )

        logger.log(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{train_acc:.4f}",
                f"{test_loss:.6f}",
                f"{test_acc:.4f}",
                f"{current_lr:.8f}",
                f"{elapsed:.2f}",
                is_best,
            ]
        )

    print(f"\nTraining complete. Best test accuracy: {best_acc:.2f}%")
    print(f"训练日志已保存到: {log_path}")


if __name__ == "__main__":
    main()
