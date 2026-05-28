# 自适应攻击

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


# 配置列表 (config_label, mode, defense, n_bins, sweep_origin, pgd20_known)
_CONFIGS = [
    ("A_gaussian_n3",   "gaussian", "rq",       3,  "A",      39.22),  # 扫描 A 全局最优
    ("A_baseline_n5",   "baseline", "rq",       5,  "A",      35.90),  # 扫描 A baseline 最优 (梯度遮蔽检验)
    ("B_optimal_n8",    "rq",       "gaussian", 8,  "B",      40.27),  # 扫描 B 最优 (train_n=8)
    ("B_secondary_5",   "rq_n5",    "gaussian", 8,  "B",      39.96),  # 扫描 B 次优 (train_n=5)
    ("B_secondary_15",  "rq_n15",   "gaussian", 8,  "B",      38.85),  # 扫描 B 次优 (train_n=15)
    ("C_dual_8_3",      "rq",       "rq",       3,  "C",      39.02),  # 扫描 C 双指标最优 (8,3)
    ("C_raw_5_2",       "rq_n5",    "rq",       2,  "C",      53.13),  # 扫描 C 裸鲁棒最高 (5,2)
    ("C_hetero_15_3",   "rq_n15",   "rq",       3,  "C",      45.35),  # 扫描 C 异构强势点 (15,3)
    ("Diag_2_2",        "rq_n2",    "rq",       2,  "C-diag", 42.07),  # 同源对角线 (2,2)
    ("Diag_8_8",        "rq",       "rq",       8,  "C-diag",  6.40),  # 同源对角线 (8,8) = 旧 rq+RQ
]


def _filter_by_origins(origins):
    return [c for c in _CONFIGS if c[4] in origins]


_PRESETS = {
    "main":      _filter_by_origins({"A", "B", "C"}),
    "main+diag": _filter_by_origins({"A", "B", "C", "C-diag"}),
    "A":         _filter_by_origins({"A"}),
    "B":         _filter_by_origins({"B"}),
    "C":         _filter_by_origins({"C"}),
    "diag":      _filter_by_origins({"C-diag"}),
}


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


def make_attack(args, model, eps, defense, attack_name):
    n_classes = 10

    if attack_name in ("fgsm", "pgd"):
        return build_attack(
            model, name=attack_name, eps=eps,
            alpha=args.pgd_alpha, steps=args.pgd_steps,
        )

    if attack_name == "eot_pgd":
        wrapped = _build_adaptive_wrapper(args, model, defense, eot_k=args.eot_k)
        return build_attack(
            wrapped, name="pgd", eps=eps,
            alpha=args.pgd_alpha, steps=args.pgd_steps,
        )

    if attack_name == "autoattack":
        version = args.aa_version
        if defense == "no_defense":
            wrapped = model  # 无随机防御时 'rand' 无意义, 退化到 standard
            if version == "rand":
                version = "standard"
        else:
            wrapped = _build_adaptive_wrapper(args, model, defense, eot_k=1)
        return build_attack(
            wrapped, name="autoattack", eps=eps,
            aa_version=version, n_classes=n_classes, seed=args.seed,
        )

    raise ValueError(f"Unknown attack: {attack_name!r}")


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


def evaluate_clean_single(defense, model, loader, args, device):
    predictor = make_predictor(defense, model, args, device)
    correct, total = 0, 0
    for images, labels in tqdm(loader, desc=f"clean[{defense}]", leave=False):
        images, labels = images.to(device), labels.to(device)
        preds = predictor(images)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def evaluate_attack_single(defense, attack_name, model, loader, eps, args, device):
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


def parse_custom_configs(spec_list):
    configs = []
    for spec in spec_list:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"--configs 项 {spec!r} 格式错误, 期望 mode:defense:n_bins")
        mode, defense, n_bins_str = parts
        try:
            n_bins = int(n_bins_str)
        except ValueError:
            raise ValueError(f"n_bins 必须为整数, 得到 {n_bins_str!r}")
        if defense not in ("no_defense", "gaussian", "rq"):
            raise ValueError(f"defense 必须为 no_defense/gaussian/rq, 得到 {defense!r}")
        label = f"custom_{mode}_{defense}_n{n_bins}"
        configs.append((label, mode, defense, n_bins, "custom", None))
    return configs


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preset", type=str, default=None, choices=list(_PRESETS.keys()))
    parser.add_argument("--configs", type=str, nargs="+", default=None, help="自定义配置")
    parser.add_argument(
        "--attacks", type=str, nargs="+",
        default=["clean", "pgd", "eot_pgd", "autoattack"],
        choices=["clean", "fgsm", "pgd", "eot_pgd", "autoattack"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--gaussian_std", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--pgd_steps", type=int, default=20)
    parser.add_argument("--pgd_alpha", type=float, default=2 / 255)
    parser.add_argument("--eot_k", type=int, default=10)
    parser.add_argument("--aa_version", type=str, default="rand", choices=["standard", "rand", "plus"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--num_eval", type=int, default=None)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def _eot_k_for(attack_name, args):
    return args.eot_k if attack_name == "eot_pgd" else 0


def _aa_version_for(attack_name, args):
    return args.aa_version if attack_name == "autoattack" else "-"


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.preset is not None:
        configs = list(_PRESETS[args.preset])
        print(f"使用预设: {args.preset}  ({len(configs)} 个配置)")
    elif args.configs is not None:
        configs = parse_custom_configs(args.configs)
        print(f"使用自定义配置: {len(configs)} 个")
    else:
        raise SystemExit("必须指定 --preset 或 --configs (至少其一)")

    test_loader = get_cifar10_test_loader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_eval=args.num_eval,
    )
    eps = args.epsilon / 255.0
    eps_x255 = round(eps * 255, 2)

    n_eval = args.num_eval if args.num_eval is not None else 10000
    n_robust_attacks = sum(1 for a in args.attacks if a != "clean")
    n_total_runs = len(configs) * (
        (1 if "clean" in args.attacks else 0) + n_robust_attacks
    )

    print(f"\n{'=' * 72}")
    print(f"Eval samples : {n_eval} (全测试集)" if args.num_eval is None
          else f"Eval samples : {n_eval} (子集 smoke test)")
    print(f"Batch size   : {args.batch_size}")
    print(f"Configs      : {len(configs)}")
    print(f"Attacks      : {args.attacks}")
    print(f"ε            : {eps_x255}/255")
    print(f"n_samples    : {args.n_samples}  (软投票次数)")
    if "eot_pgd" in args.attacks:
        print(f"eot_k        : {args.eot_k}  -> 攻击端 / 样本 ≈ "
              f"{args.pgd_steps * args.eot_k} 次前向")
    if "autoattack" in args.attacks:
        print(f"aa_version   : {args.aa_version}")
    print(f"Total runs   : {n_total_runs}  ({len(configs)} configs × {len(args.attacks)} attacks)")
    print(f"Out CSV      : {args.out_csv}")
    print(f"{'=' * 72}\n")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = [
        "config_label", "sweep_origin", "model", "defense", "attack", "eps_x255",
        "n_bins", "n_samples", "eot_k", "aa_version", "accuracy", "pgd20_known",
    ]
    logger = (
        CSVLogger.append(args.out_csv, fieldnames)
        if os.path.exists(args.out_csv)
        else CSVLogger(args.out_csv, fieldnames)
    )
    print(f"结果写入: {args.out_csv}\n")

    model_cache = {}
    run_idx = 0
    for cfg_idx, (label, mode, defense, n_bins, sweep_origin, pgd20_known) in enumerate(configs, 1):
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{mode}.pth")
        if not os.path.exists(ckpt_path):
            print(f"[{cfg_idx}/{len(configs)}] [跳过] {ckpt_path} 不存在")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{cfg_idx}/{len(configs)}] config={label}")
        print(f"  mode={mode}  defense={defense}  n_bins={n_bins}  origin={sweep_origin}")
        if pgd20_known is not None:
            print(f"  (sweep_nbins 已测 PGD-20 = {pgd20_known:.2f}%)")
        print(f"{'=' * 60}")

        if mode not in model_cache:
            model_cache[mode] = load_model(ckpt_path, device)
        model = model_cache[mode]

        args.n_bins = n_bins  # 透传给下游 make_attack/make_predictor

        for attack_name in args.attacks:
            run_idx += 1
            print(f"\n  --> [run {run_idx}/{n_total_runs}] attack={attack_name}")

            if attack_name == "clean":
                acc = evaluate_clean_single(defense, model, test_loader, args, device)
                csv_attack = "none"
                csv_eps = 0
            else:
                acc = evaluate_attack_single(defense, attack_name, model, test_loader, eps, args, device)
                csv_attack = attack_name
                csv_eps = eps_x255

            print(f"      acc = {acc:.2f}%")

            logger.log([
                label, sweep_origin, mode, defense, csv_attack, csv_eps,
                n_bins, args.n_samples,
                _eot_k_for(attack_name, args), _aa_version_for(attack_name, args),
                f"{acc:.4f}",
                f"{pgd20_known:.2f}" if pgd20_known is not None else "-",
            ])

    print(f"\n{'=' * 72}")
    print(f"全部完成。CSV: {args.out_csv}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
