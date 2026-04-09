from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import build_interformer, default_interformer_config
from utils.metrics import auc_score, logloss_score


class SyntheticCTRDataset(Dataset):
    def __init__(self, n_samples: int, cfg) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.cfg = cfg

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        dense = torch.randn(self.cfg.num_dense)
        sparse = torch.tensor(
            [torch.randint(0, v, size=(1,)).item() for v in self.cfg.sparse_vocab_sizes], dtype=torch.long
        )
        seq = torch.stack(
            [torch.randint(0, v, size=(self.cfg.max_seq_len,)) for v in self.cfg.seq_vocab_sizes], dim=0
        )

        score = dense[:3].sum() * 0.4 + (sparse[0] % 7).float() * 0.1 + (seq[0, -5:].float().mean() % 5) * 0.08
        prob = torch.sigmoid(score / 3.0)
        label = torch.bernoulli(prob).float()
        return dense, sparse, seq, label


def maybe_init_wandb(args, cfg):
    if not args.use_wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("[warn] --use-wandb enabled, but `wandb` is not installed. Continue without wandb logging.")
        return None

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config={
            **vars(args),
            "model_config": asdict(cfg),
        },
    )
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interaction-arch", type=str, default="dhen", choices=["dhen", "mha"])
    parser.add_argument("--interleave-mode", type=str, default="int", choices=["sole", "sep", "n2s", "s2n", "int"])
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--rope-base", type=float, default=10000.0)

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="interformer-repro")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-log-interval", type=int, default=20)

    args = parser.parse_args()

    cfg = default_interformer_config()
    cfg.interaction_arch = args.interaction_arch
    cfg.interleave_mode = args.interleave_mode
    cfg.use_rope = args.use_rope
    cfg.rope_base = args.rope_base

    model = build_interformer(asdict(cfg)).to(args.device)

    train_ds = SyntheticCTRDataset(args.train_size, cfg)
    val_ds = SyntheticCTRDataset(args.val_size, cfg)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    wandb_run = maybe_init_wandb(args, cfg)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for dense, sparse, seq, y in train_loader:
            dense = dense.to(args.device)
            sparse = sparse.to(args.device)
            seq = seq.to(args.device)
            y = y.to(args.device)

            logits = model(dense, sparse, seq)
            loss = F.binary_cross_entropy_with_logits(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item() * dense.size(0)
            global_step += 1

            if wandb_run is not None and global_step % args.wandb_log_interval == 0:
                wandb_run.log({"train/batch_logloss": loss.item(), "train/epoch": epoch}, step=global_step)

        model.eval()
        ys = []
        ps = []
        with torch.no_grad():
            for dense, sparse, seq, y in val_loader:
                dense = dense.to(args.device)
                sparse = sparse.to(args.device)
                seq = seq.to(args.device)
                y = y.to(args.device)

                prob = torch.sigmoid(model(dense, sparse, seq))
                ys.append(y.detach().cpu())
                ps.append(prob.detach().cpu())

        y_true = torch.cat(ys)
        y_score = torch.cat(ps)
        train_logloss = total_loss / len(train_ds)
        val_logloss = logloss_score(y_true, y_score)
        val_auc = auc_score(y_true, y_score)

        print(
            f"epoch={epoch} "
            f"mode={args.interleave_mode} "
            f"interaction_arch={args.interaction_arch} "
            f"use_rope={args.use_rope} "
            f"train_logloss={train_logloss:.4f} "
            f"val_logloss={val_logloss:.4f} "
            f"val_auc={val_auc:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch_logloss": train_logloss,
                    "val/logloss": val_logloss,
                    "val/auc": val_auc,
                    "train/epoch": epoch,
                },
                step=global_step,
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
