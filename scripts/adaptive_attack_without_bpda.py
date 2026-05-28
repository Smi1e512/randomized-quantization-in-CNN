"""无 BPDA 实验: 主表 3×3 在"无 BPDA 自适应攻击" (EOT-PGD-no-BPDA / AA-rand-no-BPDA) 下的鲁棒率, RQ 列若仍接近 clean 即为梯度遮蔽典型信号; --rq_only 仅测 RQ 列。"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm

from src.attacks import EOTWrapper, build_attack
from src.core import GaussianSmoothing, RQSmoothing, RandomizedQuantizationAugModule
from src.data_utils import get_cifar10_test_loader
from src.models import get_resnet50_cifar10
from src.utils import CSVLogger, set_seed


# 主表 3×3 单元 (mode, defense, clean_known, pgd_known); clean/pgd 来自 clean_per_defense.csv / robust_pgd.csv (n=8, N=100)
_CELLS = [
    ("baseline", "no_defense", 95.43,  0.00),
    ("baseline", "gaussian",   25.08, 15.21),
    ("baseline", "rq",         84.87, 19.29),
    ("gaussian", "no_defense", 89.91,  7.52),
    ("gaussian", "gaussian",   91.87, 16.87),
    ("gaussian", "rq",         90.58, 20.74),
    ("rq",       "no_defense", 92.32,  0.03),
    ("rq",       "gaussian",   77.97, 33.85),
    ("rq",       "rq",         92.69,  2.97),
]


# --- 模型 / 攻击构造 ---

def load_model(checkpoint_path, device):
    model = get_resnet50_cifar10(num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _build_no_bpda_wrapper(args, model, defense, eot_k: int):
    """无 BPDA 的自适应包装: 对比 sweep_adaptive 的 BPDAWrapper(rq) 反向走 identity, 本脚本 transform=rq 反向 ≈ 0 (梯度遮蔽源头)。"""
    device = next(model.parameters()).device

    if defense == "rq":
        rq = RandomizedQuantizationAugModule(region_num=args.n_bins).to(device)
        return EOTWrapper(model, rq, k=eot_k)

    if defense == "gaussian":
        std = args.gaussian_std  # gaussian 本身可导, 不需要 BPDA, 等价于常规 EOT-PGD

        def _gauss(x):
            return torch.clamp(x + torch.randn_like(x) * std, 0.0, 1.0)

        return EOTWrapper(model, _gauss, k=eot_k)

    return model  # no_defense: 直接攻击基础分类器


def make_attack(args, model, eps, defense, attack_name):
    if attack_name == "eot_pgd_no_bpda":
        wrapped = _build_no_bpda_wrapper(args, model, defense, eot_k=args.eot_k)
        return build_attack(
            wrapped, name="pgd", eps=eps,
            alpha=args.pgd_alpha, steps=args.pgd_steps,
        )

    if attack_name == "aa_rand_no_bpda":
        if defense == "no_defense":
            wrapped = model
            version = "standard"
        else:
            wrapped = _build_no_bpda_wrapper(args, model, defense, eot_k=1)  # AA 内部带 EOT, 外层不套 BPDA → 对 RQ 反向梯度仍为 0
            version = args.aa_version
        return build_attack(
            wrapped, name="autoattack", eps=eps,
            aa_version=version, n_classes=10, seed=args.seed,
        )

    raise ValueError(f"未知攻击 {attack_name!r}")


def make_predictor(defense, model, args, device):
    """评测端仍用完整软投票 (与主表一致), 只换攻击端协议为无 BPDA 版本。"""
    if defense == "no_defense":
        @torch.no_grad()
        def _pred(x):
            return model(x).argmax(dim=1)
        return _pred
    if defense == "gaussian":
        gs = GaussianSmoothing(model, std=args.gaussian_std,
                               n_samples=args.n_samples, device=device)
        return gs.predict
    if defense == "rq":
        rqs = RQSmoothing(model, n_bins=args.n_bins,
                          n_samples=args.n_samples, device=device)
        return rqs.predict
    raise ValueError(f"Unknown defense {defense!r}")


# --- 评测循环 ---

def evaluate_attack(defense, attack_name, model, loader, eps, args, device):
    attack = make_attack(args, model, eps, defense, attack_name)
    predictor = make_predictor(defense, model, args, device)
    correct, total = 0, 0
    desc = f"{attack_name}[{defense}] ε={eps * 255:.1f}/255"
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        adv = attack(images, labels)
        preds = predictor(adv)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# --- CLI ---

def parse_args():
    p = argparse.ArgumentParser(
        description="主表 3×3 在'无 BPDA 自适应攻击'下的鲁棒率 (无 BPDA 实验)"
    )

    p.add_argument("--rq_only", action="store_true",
                   help="仅评测 RQ 推理列 (3 单元), 节省时间")
    p.add_argument(
        "--attacks", type=str, nargs="+",
        default=["eot_pgd_no_bpda", "aa_rand_no_bpda"],
        choices=["eot_pgd_no_bpda", "aa_rand_no_bpda"],
    )

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_samples", type=int, default=100, help="评测端软投票次数")
    p.add_argument("--gaussian_std", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--n_bins", type=int, default=8, help="主表统一 n=8")
    p.add_argument("--epsilon", type=float, default=8.0, help="单位 1/255")
    p.add_argument("--pgd_steps", type=int, default=20)
    p.add_argument("--pgd_alpha", type=float, default=2 / 255)
    p.add_argument("--eot_k", type=int, default=10)
    p.add_argument("--aa_version", type=str, default="rand",
                   choices=["standard", "rand", "plus"])

    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    p.add_argument("--num_eval", type=int, default=None)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# --- 主流程 ---

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cells = [c for c in _CELLS if (c[1] == "rq" if args.rq_only else True)]
    print(f"评测单元: {len(cells)} (rq_only={args.rq_only})")

    test_loader = get_cifar10_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_eval=args.num_eval,
    )
    eps = args.epsilon / 255.0
    eps_x255 = round(eps * 255, 2)

    n_eval = args.num_eval if args.num_eval is not None else 10000
    n_total_runs = len(cells) * len(args.attacks)

    print(f"\n{'=' * 72}")
    print(f"Eval samples : {n_eval}")
    print(f"Cells        : {len(cells)} 个 (mode × defense)")
    print(f"Attacks      : {args.attacks}")
    print(f"n_bins       : {args.n_bins}")
    print(f"ε            : {eps_x255}/255")
    print(f"n_samples    : {args.n_samples}")
    print(f"eot_k        : {args.eot_k} (EOT 包装在变换外, 但 *不* 套 BPDA)")
    print(f"Total runs   : {n_total_runs}")
    print(f"Out CSV      : {args.out_csv}")
    print(f"{'=' * 72}\n")

    fieldnames = [
        "model", "defense", "n_bins", "n_samples", "eot_k", "aa_version",
        "attack", "eps_x255", "accuracy",
        "clean_known", "pgd_known", "gap_vs_clean",
    ]
    logger = (
        CSVLogger.append(args.out_csv, fieldnames)
        if os.path.exists(args.out_csv)
        else CSVLogger(args.out_csv, fieldnames)
    )
    print(f"结果写入: {args.out_csv}\n")

    model_cache = {}
    run_idx = 0
    for cell_idx, (mode, defense, clean_known, pgd_known) in enumerate(cells, 1):
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{mode}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[{cell_idx}/{len(cells)}] [跳过] {ckpt_path} 不存在")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{cell_idx}/{len(cells)}] {mode} + {defense} (n={args.n_bins})")
        print(f"  主表已知: clean={clean_known:.2f}%  PGD-20={pgd_known:.2f}%")
        print(f"{'=' * 60}")

        if mode not in model_cache:
            model_cache[mode] = load_model(ckpt_path, device)
        model = model_cache[mode]

        for attack_name in args.attacks:
            run_idx += 1
            print(f"\n  --> [run {run_idx}/{n_total_runs}] attack={attack_name}")

            acc = evaluate_attack(defense, attack_name, model, test_loader, eps, args, device)
            csv_eot_k = args.eot_k if attack_name == "eot_pgd_no_bpda" else 0
            csv_aa_ver = args.aa_version if attack_name == "aa_rand_no_bpda" else "-"
            gap = acc - clean_known

            print(f"      acc = {acc:.2f}%   (clean={clean_known:.2f}, "
                  f"gap_vs_clean={gap:+.2f}, PGD-20={pgd_known:.2f})")

            logger.log([
                mode, defense, args.n_bins, args.n_samples,
                csv_eot_k, csv_aa_ver,
                attack_name, eps_x255,
                f"{acc:.4f}",
                f"{clean_known:.2f}",
                f"{pgd_known:.2f}",
                f"{gap:+.4f}",
            ])

    print(f"\n{'=' * 72}")
    print(f"全部完成。CSV: {args.out_csv}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
