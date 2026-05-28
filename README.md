# 基于随机量化的图像分类对抗鲁棒性防御研究

## 1. 目录速览

```
Code/
├── src/                算法核心（被脚本 import）
├── scripts/            可执行入口（5 个 .py，对应 5 类实验）
├── checkpoints/        训练得到的 .pth 权重
├── results/
│   ├── logs/           训练日志（每模型一份 per-epoch）
│   └── tables/         评测结果（论文图表的原始数据）
└── requirements.txt
```

---

## 2. 源码文件作用（`src/`）

### 2.1 `src/core/`　防御算法

| 文件 | 作用 |
|---|---|
| `randomized_quantization.py` | 随机量化算子。 |
| `base_smoothing.py` | 随机平滑推理基类：采样 N 次随机变换 → softmax 平均 → argmax 软投票。 |
| `rq_smoothing.py` | RQ 软投票推理（**本课题主防御**）。 |
| `gaussian_smoothing.py` | 加性高斯软投票，σ 默认 0.1。 |

### 2.2 `src/attacks/`　攻击与攻击包装

| 文件 | 作用 |
|---|---|
| `bpda_wrapper.py` | BPDA 包装：前向走真 transform，反向视为 identity，让攻击绕过不可导防御求梯度。 |
| `eot_wrapper.py` | EOT 包装：对随机变换采样 k 次 logit 取均值，让梯度对随机性鲁棒。 |
| `builders.py` | FGSM / PGD / AutoAttack 攻击工厂。 |

### 2.3 其他

| 文件 | 作用 |
|---|---|
| `src/models/resnet.py` | ResNet-50 适配 CIFAR-10：`conv1` 改 3×3 stride 1、去首层 maxpool，避免 32×32 过度下采样。 |
| `src/data_utils/cifar10.py` | DataLoader 工厂。像素保持 [0,1] 不做 mean/std 归一化。 |
| `src/utils/logging.py` | 轻量 CSV 日志器，支持新建 / 追加两种模式。 |
| `src/utils/seed.py` | 统一固定 Python / NumPy / PyTorch 随机种子。 |

---

## 3. 可执行脚本（`scripts/`）

5 个脚本对应论文 5 类实验，按依赖顺序：

| 脚本 | 作用 |
|---|---|
| `train.py` | 模型的统一训练入口 |
| `evaluate.py` | 评测入口 |
| `sweep_nbins.py` | 分箱数三类扫描 |
| `adaptive.py` | 自适应攻击评测主入口。 |
| `adaptive_attack_without_bpda.py` | 无 BPDA 攻击评测。 |

---

## 4. 训练日志（`results/logs/`）

每份 `train_log_<tag>.csv` 是单模型 per-epoch 训练日志，共 12 份。

```text
epoch          当前 epoch 编号 (1..200)
train_loss     训练集平均 CE 损失
train_acc      训练集 top-1 准确率 (%)
test_loss      测试集平均 CE 损失（按训练时输入变换评测，仅作收敛监控）
test_acc       测试集 top-1 准确率 (%)
lr             当前学习率（余弦退火）
time_sec       该 epoch 训练 + 监控评测总耗时（秒）
is_best        是否在此 epoch 刷新最优测试准确率 (0/1)
```

| 文件名 | 对应权重 | 训练侧设置 |
|---|---|---|
| `train_log_baseline.csv` | `best_baseline.pth` | 无任何输入扰动 |
| `train_log_gaussian.csv` | `best_gaussian.pth` | 训练时叠加高斯变换 |
| `train_log_rq.csv` | `best_rq.pth` | RQ 训练增广，n = 8 |
| `train_log_rq_n{k}.csv` | `best_rq_n{k}.pth` | RQ 训练增广，不同分箱数 |

---

## 5. 评测结果（`results/tables/`）

| 文件 | 含义 |
|---|---|
| `clean_per_defense.csv` | 3 训练模式 × 3 推理防御 的干净准确率（9 行）。 |
| `robust_pgd.csv` | 3 训练模式 × 3 推理防御 在 PGD-20 下的鲁棒准确率（主表，13 行）。 |
| `adaptive_attack_without_bpda.csv` | 3 训练模式 × 3 推理防御在 无 BPDA 的 EOT-PGD / AutoAttack-rand 下的鲁棒准确率（19 行）。 |
| `eot-pgd.csv` | 3 训练模式 × 3 推理防御 在 BPDA + EOT-PGD（EOT-k=10）下的鲁棒准确率（9 行）。 |
| `sweep_nbins_infer.csv` | 固定训练侧、扫描推理侧分箱数 `n_infer ∈ {2,3,5,8,10,15,20,25,30,40}` 的 clean / PGD-20 结果（扫描 A，37 行）。 |
| `sweep_nbins_train.csv` | 固定推理侧 `n_infer = 8`、扫描 9 个 `rq_n<k>` 训练模型的 clean / PGD-20 结果（扫描 B，41 行）。 |
| `sweep_nbins_both.csv` | 9 训练侧 × 9 推理侧 = 81 组合的 clean / PGD-20 结果（扫描 C，163 行）。 |
| `sweep_adaptive.csv` | 10 候选配置的自适应攻击评测结果（41 行）。 |
