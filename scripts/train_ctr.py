from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ctr_dataset import CTRDatasetSpec, build_ctr_dataloader
from model import build_interformer, default_interformer_config
from utils.metrics import auc_score, gauc_score, logloss_score


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _default_best_checkpoint_path(run_cfg: Dict[str, Any]) -> str:
    metrics_path = str(run_cfg.get("save_metrics_path", "") or "").strip()
    if metrics_path:
        path = Path(metrics_path)
        return str(path.with_suffix(".best.pt"))
    run_name = str(run_cfg.get("wandb_run_name", "") or "").strip() or "interformer"
    return str(Path("results") / f"{run_name}.best.pt")


def maybe_init_wandb(cfg_cli: Dict[str, Any], cfg_model):
    if not cfg_cli.get("use_wandb", False):
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[warn] use_wandb=true, but `wandb` is not installed. Continue without wandb logging.")
        return None

    run = wandb.init(
        project=cfg_cli.get("wandb_project", "interformer-repro"),
        entity=cfg_cli.get("wandb_entity") or None,
        name=cfg_cli.get("wandb_run_name") or None,
        config={
            **cfg_cli,
            "model_config": asdict(cfg_model),
        },
    )
    return run


def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_users = []
    with torch.no_grad():
        for batch in loader:
            dense = batch["dense"].to(device)
            sparse = batch["sparse"].to(device)
            seq = batch["seq"].to(device)
            label = batch["label"].to(device)
            user_ids = batch["user_id"]

            logits = model(dense, sparse, seq)
            prob = torch.sigmoid(logits)

            all_labels.append(label.detach().cpu())
            all_probs.append(prob.detach().cpu())
            all_users.extend(user_ids)

    y = torch.cat(all_labels)
    p = torch.cat(all_probs)
    return {
        "auc": auc_score(y, p),
        "gauc": gauc_score(all_users, y, p),
        "logloss": logloss_score(y, p),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train InterFormer on CTR dataset. Prefer --run-config JSON and use CLI for overrides."
    )
    parser.add_argument("--run-config", type=str, default="configs/train_ctr.local.json")

    parser.add_argument("--dataset-config", type=str, default=None)
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--interaction-arch", type=str, choices=["dhen", "mha"], default=None)
    parser.add_argument("--interleave-mode", type=str, choices=["sole", "sep", "n2s", "s2n", "int"], default=None)
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--rope-base", type=float, default=None)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "step"], default=None)
    parser.add_argument("--scheduler-t-max", type=int, default=None)
    parser.add_argument("--scheduler-min-lr", type=float, default=None)
    parser.add_argument("--scheduler-step-size", type=int, default=None)
    parser.add_argument("--scheduler-gamma", type=float, default=None)

    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--no-nan-guard", action="store_true")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        choices=["val_logloss", "val_auc", "val_gauc"],
        default=None,
    )

    parser.add_argument("--save-metrics-path", type=str, default=None)
    parser.add_argument("--save-best-checkpoint-path", type=str, default=None)
    return parser.parse_args()


def _build_runtime_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    defaults = {
        "dataset_config": "configs/datasets/amazon_electronics.json",
        "train_path": "",
        "val_path": "",
        "test_path": "",
        "epochs": 5,
        "batch_size": 512,
        "lr": 1e-3,
        "num_workers": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "interaction_arch": "dhen",
        "interleave_mode": "int",
        "use_rope": False,
        "rope_base": 10000.0,
        "scheduler": "none",
        "scheduler_t_max": 100,
        "scheduler_min_lr": 0.0,
        "scheduler_step_size": 10,
        "scheduler_gamma": 0.9,
        "warmup_epochs": 0,
        "grad_clip_norm": 0.0,
        "nan_guard": True,
        "use_wandb": False,
        "wandb_project": "interformer-repro",
        "wandb_run_name": "",
        "wandb_entity": "",
        "wandb_log_interval": 50,
        "progress_interval": 50,
        "progress_bar": True,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "early_stop_metric": "val_logloss",
        "save_metrics_path": "",
        "save_best_checkpoint_path": "",
    }

    file_cfg: Dict[str, Any] = {}
    if args.run_config and Path(args.run_config).exists():
        file_cfg = load_json(args.run_config)

    cli_cfg = {
        "dataset_config": args.dataset_config,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "test_path": args.test_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "device": args.device,
        "interaction_arch": args.interaction_arch,
        "interleave_mode": args.interleave_mode,
        "rope_base": args.rope_base,
        "scheduler": args.scheduler,
        "scheduler_t_max": args.scheduler_t_max,
        "scheduler_min_lr": args.scheduler_min_lr,
        "scheduler_step_size": args.scheduler_step_size,
        "scheduler_gamma": args.scheduler_gamma,
        "warmup_epochs": args.warmup_epochs,
        "grad_clip_norm": args.grad_clip_norm,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_entity": args.wandb_entity,
        "wandb_log_interval": args.wandb_log_interval,
        "progress_interval": args.progress_interval,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_metric": args.early_stop_metric,
        "save_metrics_path": args.save_metrics_path,
        "save_best_checkpoint_path": args.save_best_checkpoint_path,
    }

    merged = {**defaults, **file_cfg}
    for k, v in cli_cfg.items():
        if v is not None:
            merged[k] = v

    if args.use_rope:
        merged["use_rope"] = True
    if args.use_wandb:
        merged["use_wandb"] = True
    if args.no_progress_bar:
        merged["progress_bar"] = False
    if args.no_nan_guard:
        merged["nan_guard"] = False

    missing = [k for k in ["train_path", "val_path"] if not merged.get(k)]
    if missing:
        raise ValueError(f"Missing required runtime config fields: {missing}. Set them in --run-config or CLI.")

    return merged


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def main() -> None:
    args = _parse_args()
    run_cfg = _build_runtime_cfg(args)

    ds_cfg = load_json(run_cfg["dataset_config"])
    spec = CTRDatasetSpec.from_dict(ds_cfg["data"])

    cfg = default_interformer_config()
    cfg.num_dense = len(spec.dense_cols)
    cfg.sparse_vocab_sizes = list(spec.sparse_vocab_sizes)
    cfg.seq_vocab_sizes = list(spec.seq_vocab_sizes)
    cfg.max_seq_len = spec.max_seq_len

    for k, v in ds_cfg.get("model", {}).items():
        setattr(cfg, k, v)

    cfg.interaction_arch = run_cfg["interaction_arch"]
    cfg.interleave_mode = run_cfg["interleave_mode"]
    cfg.use_rope = run_cfg["use_rope"]
    cfg.rope_base = run_cfg["rope_base"]

    model = build_interformer(asdict(cfg)).to(run_cfg["device"])

    train_loader = build_ctr_dataloader(
        file_path=run_cfg["train_path"],
        spec=spec,
        batch_size=run_cfg["batch_size"],
        shuffle=True,
        num_workers=run_cfg["num_workers"],
    )
    val_loader = build_ctr_dataloader(
        file_path=run_cfg["val_path"],
        spec=spec,
        batch_size=run_cfg["batch_size"],
        shuffle=False,
        num_workers=run_cfg["num_workers"],
    )
    test_loader = None
    if run_cfg["test_path"]:
        test_loader = build_ctr_dataloader(
            file_path=run_cfg["test_path"],
            spec=spec,
            batch_size=run_cfg["batch_size"],
            shuffle=False,
            num_workers=run_cfg["num_workers"],
        )

    base_lr = float(run_cfg["lr"])
    opt = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)
    scheduler_name = str(run_cfg.get("scheduler", "none")).lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(run_cfg.get("scheduler_t_max", run_cfg["epochs"])),
            eta_min=float(run_cfg.get("scheduler_min_lr", 0.0)),
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=int(run_cfg.get("scheduler_step_size", 10)),
            gamma=float(run_cfg.get("scheduler_gamma", 0.9)),
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}. Use one of ['none', 'cosine', 'step']")

    warmup_epochs = max(0, int(run_cfg.get("warmup_epochs", 0)))
    total_warmup_steps = warmup_epochs * max(len(train_loader), 1)
    grad_clip_norm = float(run_cfg.get("grad_clip_norm", 0.0))
    nan_guard = bool(run_cfg.get("nan_guard", True))

    wandb_run = maybe_init_wandb(run_cfg, cfg)
    best_ckpt_path = str(run_cfg.get("save_best_checkpoint_path", "") or _default_best_checkpoint_path(run_cfg))

    last_train_logloss = 0.0
    last_val_metrics = {"auc": 0.0, "gauc": 0.0, "logloss": 0.0}
    best_val_metrics = {"auc": 0.0, "gauc": 0.0, "logloss": 0.0}

    early_stop_metric = str(run_cfg.get("early_stop_metric", "val_logloss"))
    early_stop_patience = int(run_cfg.get("early_stop_patience", 0))
    early_stop_min_delta = float(run_cfg.get("early_stop_min_delta", 0.0))

    metric_mode = "min" if early_stop_metric == "val_logloss" else "max"
    best_metric = float("inf") if metric_mode == "min" else float("-inf")
    best_epoch = 0
    bad_epochs = 0

    global_step = 0
    stop_on_error = False

    for epoch in range(1, run_cfg["epochs"] + 1):
        model.train()
        total_train_loss = 0.0
        total_samples = 0
        n_batches = len(train_loader)
        epoch_start = time.time()
        progress_interval = max(1, int(run_cfg.get("progress_interval", 50)))
        use_progress_bar = bool(run_cfg.get("progress_bar", True))

        for batch_idx, batch in enumerate(train_loader, start=1):
            if total_warmup_steps > 0 and global_step < total_warmup_steps:
                warmup_factor = float(global_step + 1) / float(total_warmup_steps)
                _set_lr(opt, base_lr * warmup_factor)

            dense = batch["dense"].to(run_cfg["device"])
            sparse = batch["sparse"].to(run_cfg["device"])
            seq = batch["seq"].to(run_cfg["device"])
            label = batch["label"].to(run_cfg["device"])

            logits = model(dense, sparse, seq)
            loss = F.binary_cross_entropy_with_logits(logits, label)

            if nan_guard and (not _all_finite(logits) or not _all_finite(loss)):
                print(
                    f"[nan-guard] Non-finite detected before backward: epoch={epoch} batch={batch_idx} "
                    f"lr={opt.param_groups[0]['lr']:.8f}"
                )
                stop_on_error = True
                break

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if nan_guard:
                bad_grad = False
                for p in model.parameters():
                    if p.grad is not None and (not _all_finite(p.grad)):
                        bad_grad = True
                        break
                if bad_grad:
                    print(
                        f"[nan-guard] Non-finite gradient detected: epoch={epoch} batch={batch_idx} "
                        f"lr={opt.param_groups[0]['lr']:.8f}"
                    )
                    stop_on_error = True
                    break

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            opt.step()

            bsz = dense.size(0)
            total_train_loss += loss.item() * bsz
            total_samples += bsz
            global_step += 1

            is_last_batch = batch_idx == n_batches
            if batch_idx == 1 or batch_idx % progress_interval == 0 or is_last_batch:
                elapsed = time.time() - epoch_start
                sec_per_batch = elapsed / max(batch_idx, 1)
                eta_sec = sec_per_batch * max(n_batches - batch_idx, 0)
                pct = 100.0 * batch_idx / max(n_batches, 1)
                avg_loss = total_train_loss / max(total_samples, 1)
                filled = int((pct / 100.0) * 24)
                bar = "=" * filled + "." * (24 - filled)
                line = (
                    f"[train] e{epoch}/{run_cfg['epochs']} "
                    f"[{bar}] {pct:5.1f}% "
                    f"b{batch_idx}/{n_batches} "
                    f"loss={loss.item():.5f} avg={avg_loss:.5f} "
                    f"lr={opt.param_groups[0]['lr']:.8f} "
                    f"{sec_per_batch:.3f}s/b eta={_format_seconds(eta_sec)}"
                )
                if use_progress_bar:
                    print(line.ljust(160), end="\n" if is_last_batch else "\r", flush=True)
                else:
                    print(line)

            if wandb_run is not None and global_step % run_cfg["wandb_log_interval"] == 0:
                wandb_run.log({"train/batch_logloss": loss.item(), "train/epoch": epoch}, step=global_step)

        if stop_on_error:
            break

        train_logloss = total_train_loss / max(total_samples, 1)
        val_metrics = evaluate(model, val_loader, run_cfg["device"])

        last_train_logloss = train_logloss
        last_val_metrics = val_metrics

        print(
            f"epoch={epoch} "
            f"mode={run_cfg['interleave_mode']} "
            f"interaction_arch={run_cfg['interaction_arch']} "
            f"train_logloss={train_logloss:.5f} "
            f"val_auc={val_metrics['auc']:.5f} "
            f"val_gauc={val_metrics['gauc']:.5f} "
            f"val_logloss={val_metrics['logloss']:.5f} "
            f"lr={opt.param_groups[0]['lr']:.8f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_logloss": train_logloss,
                    "val/auc": val_metrics["auc"],
                    "val/gauc": val_metrics["gauc"],
                    "val/logloss": val_metrics["logloss"],
                    "train/lr": opt.param_groups[0]["lr"],
                    "train/epoch": epoch,
                },
                step=global_step,
            )

        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()

        monitor_values = {
            "val_logloss": val_metrics["logloss"],
            "val_auc": val_metrics["auc"],
            "val_gauc": val_metrics["gauc"],
        }
        current_metric = monitor_values[early_stop_metric]

        if metric_mode == "min":
            improved = current_metric < (best_metric - early_stop_min_delta)
        else:
            improved = current_metric > (best_metric + early_stop_min_delta)

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            bad_epochs = 0
            ckpt_path = Path(best_ckpt_path)
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "monitor": early_stop_metric,
                    "run_config": run_cfg,
                    "model_config": asdict(cfg),
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(
                f"[checkpoint] saved best model: epoch={epoch} "
                f"monitor={early_stop_metric} value={current_metric:.5f} path={ckpt_path}"
            )
        else:
            bad_epochs += 1

        if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
            print(
                f"[early-stop] stop at epoch={epoch} "
                f"monitor={early_stop_metric} current={current_metric:.5f} "
                f"best={best_metric:.5f} best_epoch={best_epoch} "
                f"patience={early_stop_patience} min_delta={early_stop_min_delta}"
            )
            break

    test_metrics = None
    if best_epoch > 0 and Path(best_ckpt_path).exists():
        best_checkpoint = torch.load(best_ckpt_path, map_location=run_cfg["device"])
        model.load_state_dict(best_checkpoint["model_state_dict"])
        print(
            f"[checkpoint] loaded best model: epoch={best_epoch} "
            f"monitor={early_stop_metric} best={best_metric:.5f} path={best_ckpt_path}"
        )
        last_val_metrics = dict(best_val_metrics)

    if (not stop_on_error) and test_loader is not None:
        test_metrics = evaluate(model, test_loader, run_cfg["device"])
        print(
            f"test_auc={test_metrics['auc']:.5f} "
            f"test_gauc={test_metrics['gauc']:.5f} "
            f"test_logloss={test_metrics['logloss']:.5f}"
        )

    if run_cfg.get("save_metrics_path"):
        out_path = Path(run_cfg["save_metrics_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "interleave_mode": run_cfg["interleave_mode"],
            "interaction_arch": run_cfg["interaction_arch"],
            "use_rope": run_cfg["use_rope"],
            "epochs": run_cfg["epochs"],
            "train_logloss": last_train_logloss,
            "val": last_val_metrics,
            "test": test_metrics,
            "stopped_on_nan": stop_on_error,
            "early_stop": {
                "metric": early_stop_metric,
                "patience": early_stop_patience,
                "min_delta": early_stop_min_delta,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
            },
            "best_checkpoint_path": best_ckpt_path,
            "run_config": run_cfg,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
