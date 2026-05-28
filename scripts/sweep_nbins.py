# n_bins 扫描脚本

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from src.attacks import BPDAWrapper, EOTWrapper, build_attack
from src.core import GaussianSmoothing, RQSmoothing
from src.data_utils import get_cifar10_test_loader
from src.models import get_resnet50_cifar10
from src.utils import CSVLogger, set_seed


def load_model(checkpoint_path, device):
    model = get_resnet50_cifar10(num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _build_adaptive_wrapper(args, model, defense, eot_k: int):
    if defense == "rq":
        from src.core import RandomizedQuantizationAugModule

        device = next(model.parameters()).device
        rq = RandomizedQuantizationAugModule(region_num=args.n_bins).to(device)
        transform = BPDAWrapper(rq)
        return EOTWrapper(model, transform, k=eot_k)

    if defense == "gaussian":
        std = args.gaussian_std

        def _gauss(x):
            return torch.clamp(x + torch.randn_like(x) * std, 0.0, 1.0)

        return EOTWrapper(model, _gauss, k=eot_k)

    return model


def make_attack(args, model, eps, defense):
    name = args.attack
    n_classes = 10

    if name in ("fgsm", "pgd"):
        return build_attack(model, name=name, eps=eps, alpha=args.pgd_alpha, steps=args.pgd_steps)

    if name == "eot_pgd":
        wrapped = _build_adaptive_wrapper(args, model, defense, eot_k=args.eot_k)
        return build_attack(wrapped, name="pgd", eps=eps, alpha=args.pgd_alpha, steps=args.pgd_steps)

    if name == "autoattack":
        version = args.aa_version
        if defense == "no_defense":
            wrapped = model
            if version == "rand":
                version = "standard"
        else:
            wrapped = _build_adaptive_wrapper(args, model, defense, eot_k=1)
        return build_attack(
            wrapped, name="autoattack", eps=eps,
            aa_version=version, n_classes=n_classes, seed=args.seed,
        )

    raise ValueError(f"Unknown attack: {name!r}")


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


def evaluate_clean_with_defenses(args, model, loader, device):
    correct = {d: 0 for d in args.defenses}
    total = 0
    predictors = {d: make_predictor(d, model, args, device) for d in args.defenses}

    for images, labels in tqdm(loader, desc="Clean", leave=False):
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

        adv_shared = (
            attacks["no_defense"](images, labels) if cache_attack_for_no_defense else None
        )

        for defense in args.defenses:
            if args.attack in ("fgsm", "pgd") and adv_shared is not None:
                adv = adv_shared
            else:
                adv = attacks[defense](images, labels)
            preds = predictors[defense](adv)
            correct[defense] += preds.eq(labels).sum().item()
        total += labels.size(0)

    return {d: 100.0 * correct[d] / total for d in args.defenses}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--gaussian_std", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument(
        "--attack", type=str, default="pgd",
        choices=["fgsm", "pgd", "eot_pgd", "autoattack"],
    )
    parser.add_argument("--epsilon", type=float, default=8.0, help="单 ε, 单位 1/255")
    parser.add_argument("--pgd_steps", type=int, default=20)
    parser.add_argument("--pgd_alpha", type=float, default=2 / 255)
    parser.add_argument("--eot_k", type=int, default=10)
    parser.add_argument("--aa_version", type=str, default="rand", choices=["standard", "rand", "plus"])

    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument(
        "--defenses", nargs="+", default=["rq"],
        choices=["no_defense", "gaussian", "rq"],
    )
    parser.add_argument("--n_bins_list", type=int, nargs="+", required=True)
    parser.add_argument(
        "--paired", action="store_true",
    )

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _eot_k_for(args):
    return args.eot_k if args.attack == "eot_pgd" else 0


def _aa_version_for(args):
    return args.aa_version if args.attack == "autoattack" else "-"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.paired and len(args.modes) != len(args.n_bins_list):
        raise ValueError(
            f"--paired 要求 len(modes)={len(args.modes)} 与 "
            f"len(n_bins_list)={len(args.n_bins_list)} 相等"
        )

    pairs = (
        list(zip(args.modes, args.n_bins_list))
        if args.paired
        else [(m, n) for m in args.modes for n in args.n_bins_list]
    )

    test_loader = get_cifar10_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_eval=args.num_eval,
    )
    eps = args.epsilon / 255.0
    eps_x255 = round(eps * 255, 2)

    n_eval = args.num_eval if args.num_eval is not None else 10000
    print(f"\n{'=' * 70}")
    print(
        f"Eval samples : {n_eval} (全测试集)"
        if args.num_eval is None
        else f"Eval samples : {n_eval} (子集 smoke test)"
    )
    print(f"Attack       : {args.attack}  (ε={eps_x255}/255)")
    print(f"Defenses     : {args.defenses}")
    print(f"Modes        : {args.modes}")
    print(f"n_bins_list  : {args.n_bins_list}")
    print(f"Paired mode  : {args.paired}")
    print(f"Total runs   : {len(pairs)}  (每个 run 跑 clean + robust)")
    print(f"{'=' * 70}\n")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = [
        "model", "attack", "eps_x255", "n_bins", "n_samples",
        "eot_k", "aa_version", "defense", "accuracy",
    ]
    logger = (
        CSVLogger.append(args.out_csv, fieldnames)
        if os.path.exists(args.out_csv)
        else CSVLogger(args.out_csv, fieldnames)
    )
    print(f"结果将写到: {args.out_csv}\n")

    model_cache = {}

    for idx, (mode, n_bins) in enumerate(pairs, 1):
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{mode}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[{idx}/{len(pairs)}] [跳过] {ckpt_path} 不存在")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(pairs)}] mode={mode}  n_bins={n_bins}")
        print(f"{'=' * 60}")

        if mode not in model_cache:
            model_cache[mode] = load_model(ckpt_path, device)
        model = model_cache[mode]

        args.n_bins = n_bins


        clean_results = evaluate_clean_with_defenses(args, model, test_loader, device)
        for defense, acc in clean_results.items():
            print(f"  CLEAN  defense={defense:>11s}  acc={acc:.2f}%")
            logger.log([mode, "none", 0, n_bins, args.n_samples, 0, "-", defense, f"{acc:.4f}"])

        robust_results = evaluate_under_attack(args, model, test_loader, device, eps)
        for defense, acc in robust_results.items():
            print(f"  ROBUST defense={defense:>11s}  acc={acc:.2f}%  (ε={eps_x255}/255, {args.attack})")
            logger.log([
                mode, args.attack, eps_x255, n_bins, args.n_samples,
                _eot_k_for(args), _aa_version_for(args), defense, f"{acc:.4f}",
            ])

        for d in args.defenses:
            if d in clean_results and d in robust_results:
                drop = clean_results[d] - robust_results[d]
                print(
                    f"  -> defense={d:>11s}: clean={clean_results[d]:.2f}%  "
                    f"robust={robust_results[d]:.2f}%  (absolute drop={drop:.2f}%)"
                )

    print(f"\n完成。CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
