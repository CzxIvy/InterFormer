# InterFormer (paper reproduction)

This repository provides a PyTorch reproduction of:

`InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction` (Zeng et al., 2025).

## What is implemented

- Interleaving heterogeneous learning with bidirectional information exchange.
- `InteractionArch` switchable as `dhen` or `mha`.
- `SequenceArch` with `PFFN + MHA`, optional RoPE.
- `CrossArch` with non-sequence and sequence summarization (`CLS + PMA + recent`).
- Ablation interleave modes: `sole / sep / n2s / s2n / int`.

## Project layout

- `models/interformer.py`: InterFormer model.
- `model.py`: model builder + default config.
- `data/ctr_dataset.py`: real CTR dataset loader + collate.
- `utils/metrics.py`: `AUC / gAUC / LogLoss`.
- `scripts/train_ctr.py`: training on real CSV/JSONL CTR data.
- `scripts/run_ablation.py`: one-click ablation runner.
- `scripts/train_synthetic.py`: synthetic smoke test.
- `configs/datasets/*.json`: dataset templates for Amazon/Taobao/KuaiVideo.
- `configs/train_ctr.local.json`: run config template.
- `configs/ablation.local.json`: ablation config template.

## Recommended: file-based config

Edit `configs/train_ctr.local.json` once, then run:

```bash
python scripts/train_ctr.py
```

By default, `train_ctr.py` reads:

- `--run-config configs/train_ctr.local.json`

You can still override single fields via CLI, e.g.:

```bash
python scripts/train_ctr.py --epochs 10 --interleave-mode sep
```

## One-click ablation

Edit `configs/ablation.local.json`, then run:

```bash
python scripts/run_ablation.py
```

Run selected modes only:

```bash
python scripts/run_ablation.py --modes sep int
```

Outputs:

- Per-mode metrics JSON: `results/ablation/<mode>.json`
- Summary table: `results/ablation/summary.csv`

## Real CTR training

```bash
python scripts/train_ctr.py --run-config configs/train_ctr.local.json
```

Optional CLI overrides:

```bash
--dataset-config configs/datasets/amazon_electronics.json
--train-path /path/to/train.csv
--val-path /path/to/val.csv
--test-path /path/to/test.csv
--interaction-arch dhen
--interleave-mode int
--use-rope --rope-base 10000
--use-wandb --wandb-project interformer-repro --wandb-run-name exp1
--save-metrics-path results/one_run.json
```

## Input schema (via dataset config)

Each config defines:

- `label_col`, `user_id_col`
- `dense_cols` (float features)
- `sparse_cols` + `sparse_vocab_sizes`
- `seq_cols` + `seq_vocab_sizes` (sequence columns, delim-joined ids)
- `max_seq_len`, `seq_delim`

`train_ctr.py` expects CSV or JSONL.

## Ablation switches

- Interaction block: `dhen|mha`
- Interleave mode: `sole|sep|n2s|s2n|int`
- Sequence position encoding: `use_rope`

## Metrics

Training script reports:

- `AUC`
- `gAUC` (user-group weighted AUC)
- `LogLoss`

## Synthetic quick check

```bash
python scripts/train_synthetic.py --epochs 1 --interaction-arch dhen --interleave-mode int
```
