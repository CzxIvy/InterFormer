# InterFormer

PyTorch implementation for CTR prediction with heterogeneous feature interaction.

## 项目简介

该仓库实现了 InterFormer 的核心训练流程，支持：

- 交互模块切换：`dhen` / `mha`
- 交错方式消融：`sole` / `sep` / `n2s` / `s2n` / `int`
- 序列分支可选 RoPE
- 真实 CTR 数据训练、合成数据自检、批量消融实验

## 目录说明

- `models/interformer.py`: 模型主体
- `model.py`: 模型构建与默认配置
- `data/ctr_dataset.py`: CSV/JSONL 数据读取与 batch 拼接
- `utils/metrics.py`: `AUC / gAUC / LogLoss`
- `scripts/train_ctr.py`: 真实数据训练入口
- `scripts/train_synthetic.py`: 合成数据冒烟测试
- `scripts/run_ablation.py`: 一键消融实验
- `configs/datasets/*.json`: 数据集字段模板
- `configs/train_ctr.local.json`: 本地调试推荐配置
- `configs/train_ctr.repro.cosine.json`: 复现模板（Cosine LR）
- `configs/train_ctr.repro.step.json`: 复现模板（Step LR）
- `configs/ablation.local.json`: 消融运行配置

## 训练配置文件说明

`train_ctr.py` 支持“文件配置优先”，默认读取：

- `configs/train_ctr.local.json`

推荐流程：

1. 先改 `train_path / val_path / test_path`
2. 再调 `batch_size / lr / epochs`
3. 最后按需调 `scheduler / warmup / grad_clip / early_stop`

关键稳定性参数：

- `warmup_epochs`: 前 N 个 epoch 线性 warmup 到目标学习率
- `grad_clip_norm`: 梯度裁剪阈值（0 表示关闭）
- `nan_guard`: 检测到非有限 loss/grad 时立即停止并提示 batch 位置

## 快速开始

### 1) 本地调试训练

```bash
python scripts/train_ctr.py --run-config configs/train_ctr.local.json
```

### 2) 复现实验（Cosine）

```bash
python scripts/train_ctr.py --run-config configs/train_ctr.repro.cosine.json
```

### 3) 复现实验（Step）

```bash
python scripts/train_ctr.py --run-config configs/train_ctr.repro.step.json
```

### 4) CLI 局部覆盖（覆盖配置文件同名字段）

```bash
python scripts/train_ctr.py --run-config configs/train_ctr.local.json --epochs 10 --interleave-mode sep
```

## 一键消融

先编辑 `configs/ablation.local.json`，然后运行：

```bash
python scripts/run_ablation.py
```

只跑部分模式：

```bash
python scripts/run_ablation.py --modes sep int
```

输出：

- `results/ablation/<mode>.json`
- `results/ablation/summary.csv`

## 数据格式

支持：

- CSV
- JSONL

字段由 `configs/datasets/*.json` 定义：

- `label_col`, `user_id_col`
- `dense_cols`
- `sparse_cols`, `sparse_vocab_sizes`
- `seq_cols`, `seq_vocab_sizes`
- `max_seq_len`, `seq_delim`

提示：

- 若序列分隔符与真实数据不一致（例如你数据是 `^`，配置写成空格），会触发解析异常。
- `user_id` 可以放进 `sparse_cols`（用于个性化建模），同时 `user_id_col` 继续用于 gAUC 分组。

## 常见问题

### 1) lr 提高后 loss 变 NaN

优先排查并调整：

- 降低 `lr`（例如 `0.01 -> 0.003/0.001`）
- 增加 `warmup_epochs`（1~3）
- 打开 `grad_clip_norm`（如 `1.0`）
- 保持 `nan_guard=true`

### 2) 为什么出现 zero-element tensor 警告

通常是 `dense_cols` 为空导致 `Linear(0, d_model)`，可运行但建议确认这是预期。

### 3) 如何开启 wandb

在配置里设置：

- `use_wandb: true`
- `wandb_project`
- `wandb_run_name`

并确保本地已登录：`wandb login`
