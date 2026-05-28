# 非自适应攻击评测

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from src.attacks import build_attack
from src.core import GaussianSmoothing, RQSmoothing
from src.data_utils import get_cifar10_test_loader
from src.models import get_resnet50_cifar10
from src.utils import CSVLogger, set_seed


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--n_bins", type=int, default=8)
    parser.add_argument("--gaussian_std", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--attack", type=str, default="pgd", choices=["none", "fgsm", "pgd"],
        help="自适应攻击用 sweep_adaptive.py",
    )
    parser.add_argument("--epsilon", type=float, default=8.0, help="单 ε, 单位 1/255")
    parser.add_argument("--epsilon_list", type=float, nargs="*", default=None, help="多 ε 扫描, 单位 1/255")
    parser.add_argument("--pgd_steps", type=int, default=20)
    parser.add_argument("--pgd_alpha", type=float, default=2 / 255)

    parser.add_argument("--modes", nargs="+", default=["baseline", "gaussian", "rq"])
    parser.add_argument(
        "--defenses", nargs="+", default=["no_defense", "gaussian", "rq"],
        choices=["no_defense", "gaussian", "rq"],
    )

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_eval", type=int, default=None, help="None = 全测试集 10000 张")
    parser.add_argument("--out_csv", type=str, default="./results/tables/robustness.csv")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = get_resnet50_cifar10(num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def make_attack(args, model, eps, defense):
    del defense  # 非自适应攻击不依赖 defense, 占位保持接口一致

    name = args.attack
    if name in ("fgsm", "pgd"):
        return build_attack(
            model, name=name, eps=eps,
            alpha=args.pgd_alpha, steps=args.pgd_steps,
        )

    raise ValueError(
        f"Unknown attack: {name!r} (本脚本仅支持 fgsm/pgd; 自适应攻击请用 sweep_adaptive.py)"
    )


def make_predictor(defense, model, args, device):
    if defense == "no_defense":
        @torch.no_grad()
        def _pred(x):
            return model(x).argmax(dim=1)
        return _pred
    if defense == "gaussian":
        gs = GaussianSmoothing(model, std=args.gaussian_std, n_samples=args.n_samples, device=device)
        return gs.predict
    if defense == "rq":
        rqs = RQSmoothing(model, n_bins=args.n_bins, n_samples=args.n_samples, device=device)
        return rqs.predict
    raise ValueError(f"Unknown defense: {defense!r}")


def evaluate_clean(model, loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Clean", leave=False):
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def evaluate_clean_with_defenses(args, model, loader, device):
    correct = {d: 0 for d in args.defenses}
    total = 0
    predictors = {d: make_predictor(d, model, args, device) for d in args.defenses}

    for images, labels in tqdm(loader, desc="Clean (per-defense)", leave=False):
        images, labels = images.to(device), labels.to(device)
        for defense in args.defenses:
            preds = predictors[defense](images)
            correct[defense] += preds.eq(labels).sum().item()
        total += labels.size(0)
    return {d: 100.0 * correct[d] / total for d in args.defenses}


def evaluate_under_attack(args, model, loader, device, eps):
    correct = {d: 0 for d in args.defenses}
    total = 0

    attacks = {d: make_attack(args, model, eps, d) for d in args.defenses}
    predictors = {d: make_predictor(d, model, args, device) for d in args.defenses}

    cache_attack_for_no_defense = (

        args.attack in ("fgsm", "pgd") and "no_defense" in args.defenses
    )

    for images, labels in tqdm(loader, desc=f"eps={eps * 255:.1f}/255", leave=False):
        images, labels = images.to(device), labels.to(device)

        adv_shared = attacks["no_defense"](images, labels) if cache_attack_for_no_defense else None

        for defense in args.defenses:
            if args.attack in ("fgsm", "pgd") and adv_shared is not None:
                adv = adv_shared
            else:
                adv = attacks[defense](images, labels)
            preds = predictors[defense](adv)
            correct[defense] += preds.eq(labels).sum().item()
        total += labels.size(0)

    return {d: 100.0 * correct[d] / total for d in args.defenses}


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = get_cifar10_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_eval=args.num_eval,
    )

    eps_list = [e / 255.0 for e in args.epsilon_list] if args.epsilon_list else [args.epsilon / 255.0]

    n_eval = args.num_eval if args.num_eval is not None else 10000
    print(f"\n{'='*70}")
    print(f"Eval samples : {n_eval}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Attack       : {args.attack}")
    print(f"Eps (×255)   : {[round(e * 255, 2) for e in eps_list]}")
    print(f"n_bins       : {args.n_bins}")
    print(f"n_samples    : {args.n_samples}")
    print(f"Modes        : {args.modes}")
    print(f"Defenses     : {args.defenses}")
    print(f"{'='*70}\n")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["model", "attack", "eps_x255", "n_bins", "n_samples", "defense", "accuracy"]
    logger = (
        CSVLogger.append(args.out_csv, fieldnames)
        if os.path.exists(args.out_csv)
        else CSVLogger(args.out_csv, fieldnames)
    )
    print(f"结果写入: {args.out_csv}\n")

    for mode in args.modes:
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{mode}.pth")
        if not os.path.exists(ckpt_path):
            print(f"\n[跳过] {ckpt_path} 不存在")
            continue

        print(f"\n{'=' * 60}")
        print(f"评估模型: {mode}  |  ckpt: {ckpt_path}")
        print(f"{'=' * 60}")

        model = load_model(ckpt_path, device)

        if args.attack == "none":
            results = evaluate_clean_with_defenses(args, model, test_loader, device)
            for defense, acc in results.items():
                print(f"  Clean (defense={defense:>11s}) acc={acc:.2f}%")
                logger.log([mode, "none", 0, args.n_bins, args.n_samples, defense, f"{acc:.4f}"])
            continue

        clean_acc = evaluate_clean(model, test_loader, device)
        print(f"  Clean Accuracy: {clean_acc:.2f}%")
        logger.log([mode, "none", 0, args.n_bins, args.n_samples, "no_defense", f"{clean_acc:.4f}"])

        for eps in eps_list:
            results = evaluate_under_attack(args, model, test_loader, device, eps)
            eps_x255 = round(eps * 255, 2)
            for defense, acc in results.items():
                print(f"  eps={eps_x255:>5.2f}/255  defense={defense:>11s}  acc={acc:.2f}%")
                logger.log([mode, args.attack, eps_x255, args.n_bins, args.n_samples, defense, f"{acc:.4f}"])

            if "no_defense" in results and "rq" in results:
                asr_no = 100 - results["no_defense"]
                asr_rq = 100 - results["rq"]
                print(f"  -> ASR no_defense={asr_no:.2f}%, ASR rq={asr_rq:.2f}%, reduction={asr_no - asr_rq:.2f}%")

    print(f"\n完成。CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
