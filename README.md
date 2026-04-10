# InterFormer

PyTorch implementation for CTR prediction with heterogeneous feature interaction.

## 1. 项目简介

该仓库实现了 InterFormer 的核心训练流程，支持：

- 交互模块切换：dhen / mha
- 交错方式消融：sole / sep / n2s / s2n / int
- 序列分支可选 RoPE
- 真实 CTR 数据训练、合成数据快速自检、批量消融实验

## 2. 目录说明

- models/interformer.py: 模型主体
- model.py: 模型构建与默认配置
- data/ctr_dataset.py: CSV/JSONL 数据读取与拼 batch
- utils/metrics.py: AUC / gAUC / LogLoss
- scripts/train_ctr.py: 真实数据训练入口
- scripts/train_synthetic.py: 合成数据冒烟测试
- scripts/run_ablation.py: 一键消融实验
- configs/datasets/*.json: 数据集字段模板
- configs/train_ctr.local.json: 训练运行配置
- configs/train_ctr.repro.cosine.json: 论文复现实验模板（Cosine LR）
- configs/train_ctr.repro.step.json: 论文复现实验模板（Step LR）
- configs/ablation.local.json: 消融运行配置

## 3. 环境准备

建议 Python 3.10+，并安装 PyTorch。

如果你使用 wandb，可额外安装：

```bash
pip install wandb
```

## 4. 快速开始


### 4.1 先修改运行配置

编辑 configs/train_ctr.local.json，至少确认：

- dataset_config
- train_path
- val_path
- batch_size / epochs / lr

推荐配置文件字段（含进度与早停参数）：

```json
{
	"dataset_config": "configs/datasets/amazon_electronics.json",
	"train_path": "data/AmazonElectronics_x1/train.csv",
	"val_path": "data/AmazonElectronics_x1/test.csv",
	"test_path": "",
	"epochs": 5,
	"batch_size": 2048,
	"lr": 0.001,
	"num_workers": 0,
	"interaction_arch": "dhen",
	"interleave_mode": "int",
	"use_rope": false,
	"rope_base": 10000.0,
	"use_wandb": true,
	"wandb_project": "interformer-repro",
	"wandb_run_name": "",
	"wandb_entity": "",
	"wandb_log_interval": 50,
	"progress_interval": 10,
	"progress_bar": true,
	"early_stop_patience": 0,
	"early_stop_min_delta": 0.0001,
	"early_stop_metric": "val_logloss"
}
```

说明：
- progress_interval：每多少个 batch 打印一次进度（默认 10）
- progress_bar：是否用单行动态进度条（true）
- early_stop_patience：早停耐心轮数，0 表示关闭
- early_stop_min_delta：早停时指标最小改进量
- early_stop_metric：监控哪个指标（val_logloss/val_auc/val_gauc）

### 4.2 启动训练

```bash
python scripts/train_ctr.py
```

默认会读取：

```text
--run-config configs/train_ctr.local.json
```

### 4.3 进度显示

动态单行进度条（默认）：

```bash
python scripts/train_ctr.py --progress-interval 10
```

普通多行日志：

```bash
python scripts/train_ctr.py --progress-interval 10 --no-progress-bar
```

## 5. 常用参数

训练脚本 scripts/train_ctr.py 常用参数：

- --run-config: 运行配置文件路径
- --dataset-config: 数据集字段配置路径
- --train-path / --val-path / --test-path: 数据文件路径
- --interaction-arch: dhen 或 mha
- --interleave-mode: sole / sep / n2s / s2n / int
- --use-rope --rope-base: 是否开启 RoPE 及其 base
- --scheduler: 学习率调度器（none / cosine / step）
- --scheduler-t-max: cosine 调度器的 T_max
- --scheduler-min-lr: cosine 调度器的最小学习率
- --scheduler-step-size: step 调度器的 step_size
- --scheduler-gamma: step 调度器的 gamma
- --progress-interval: 多少个 batch 打印一次进度
- --no-progress-bar: 关闭动态进度条，改为多行日志
- --early-stop-patience: 早停耐心轮数，0 表示关闭早停
- --early-stop-min-delta: 指标最小改进阈值
- --early-stop-metric: 早停监控指标（val_logloss / val_auc / val_gauc）
- --use-wandb 及相关参数: 开启实验追踪
- --save-metrics-path: 将最终指标写入 JSON

示例：

```bash
python scripts/train_ctr.py \
	--run-config configs/train_ctr.local.json \
	--epochs 5 \
	--interleave-mode int \
	--interaction-arch dhen \
	--early-stop-patience 3 \
	--early-stop-metric val_logloss \
	--early-stop-min-delta 1e-4 \
	--save-metrics-path results/one_run.json
```

论文复现实验模板：

```bash
python scripts/train_ctr.py --run-config configs/train_ctr.repro.cosine.json
python scripts/train_ctr.py --run-config configs/train_ctr.repro.step.json
```

## 6. 配置优先级

参数来源优先级（高到低）：

1. CLI 显式参数
2. run-config 文件
3. 代码默认值

说明：

- --use-rope、--use-wandb、--no-progress-bar 这类布尔开关，只在命令行显式传入时覆盖。

## 7. 数据格式

支持文件类型：

- CSV
- JSONL

数据字段由 configs/datasets/*.json 定义，核心字段包括：

- label_col, user_id_col
- dense_cols
- sparse_cols, sparse_vocab_sizes
- seq_cols, seq_vocab_sizes
- max_seq_len, seq_delim

注意：

- seq_delim 必须与真实序列列分隔符一致（例如空格、^ 等）。
- 如果分隔符不一致，可能出现字符串转 int 失败（如 12617^28604^9017）。

## 8. 消融实验

先编辑 configs/ablation.local.json，再运行：

```bash
python scripts/run_ablation.py
```

只跑部分模式：

```bash
python scripts/run_ablation.py --modes sep int
```

输出：

- results/ablation/<mode>.json
- results/ablation/summary.csv

## 9. 合成数据快速自检

```bash
python scripts/train_synthetic.py --epochs 1 --interaction-arch dhen --interleave-mode int
```

用于快速检查训练链路是否通畅，不代表真实业务效果。

## 10. 常见问题

### 10.1 UserWarning: Initializing zero-element tensors is a no-op

常见于 dense_cols 为空时，模型会构造 Linear(0, d_model)。通常是警告，不一定导致训练中断。

### 10.2 ValueError: invalid literal for int()

多见于序列列分隔符配置错误。请检查数据集配置中的 seq_delim 是否与数据一致。

### 10.3 如何开启早停

示例 1：监控 val_logloss（越小越好）

```bash
python scripts/train_ctr.py \
	--early-stop-patience 3 \
	--early-stop-metric val_logloss \
	--early-stop-min-delta 1e-4
```

示例 2：监控 val_auc（越大越好）

```bash
python scripts/train_ctr.py \
	--early-stop-patience 3 \
	--early-stop-metric val_auc \
	--early-stop-min-delta 1e-4
```

说明：

- 当连续 patience 个 epoch 未达到 min_delta 的有效改进时，训练提前结束。
