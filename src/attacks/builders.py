# 对抗攻击工厂

import torchattacks


def build_attack(
    model,
    name: str = "pgd",
    eps: float = 8 / 255,
    alpha: float = 2 / 255,
    steps: int = 20,
    aa_version: str = "standard",
    n_classes: int = 10,
    seed: int = 42,
):
    # 构造 torchattacks 攻击对象
    name = name.lower()
    if name == "fgsm":
        return torchattacks.FGSM(model, eps=eps)
    if name == "pgd":
        return torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    if name in ("autoattack", "aa"):
        return torchattacks.AutoAttack(
            model,
            norm="Linf",
            eps=eps,
            version=aa_version,
            n_classes=n_classes,
            seed=seed,
            verbose=False,
        )
    raise ValueError(
        f"Unknown attack: {name!r}, expected one of ['fgsm', 'pgd', 'autoattack']"
    )
