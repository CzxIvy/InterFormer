from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class CTRDatasetSpec:
    label_col: str
    user_id_col: str
    dense_cols: Sequence[str]
    sparse_cols: Sequence[str]
    seq_cols: Sequence[str]
    sparse_vocab_sizes: Sequence[int]
    seq_vocab_sizes: Sequence[int]
    max_seq_len: int = 100
    seq_delim: str = " "

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CTRDatasetSpec":
        return cls(
            label_col=data["label_col"],
            user_id_col=data["user_id_col"],
            dense_cols=data.get("dense_cols", []),
            sparse_cols=data.get("sparse_cols", []),
            seq_cols=data.get("seq_cols", []),
            sparse_vocab_sizes=data.get("sparse_vocab_sizes", []),
            seq_vocab_sizes=data.get("seq_vocab_sizes", []),
            max_seq_len=int(data.get("max_seq_len", 100)),
            seq_delim=data.get("seq_delim", " "),
        )


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    if suffix in {".jsonl", ".json"}:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported file type: {path}. Use .csv or .jsonl")


def _parse_seq(value: Any, max_seq_len: int, seq_delim: str) -> List[int]:
    if value is None:
        seq: List[int] = []
    elif isinstance(value, list):
        seq = [int(x) for x in value]
    else:
        s = str(value).strip()
        if not s:
            seq = []
        else:
            seq = [int(x) for x in s.split(seq_delim) if x != ""]

    if len(seq) >= max_seq_len:
        seq = seq[-max_seq_len:]
    else:
        seq = seq + [0] * (max_seq_len - len(seq))
    return seq


def _safe_bucket(v: int, vocab_size: int) -> int:
    if vocab_size <= 0:
        return max(v, 0)
    return int(v) % vocab_size


class CTRDataset(Dataset):
    def __init__(self, file_path: str | Path, spec: CTRDatasetSpec) -> None:
        super().__init__()
        self.path = Path(file_path)
        self.spec = spec
        self.rows = _load_rows(self.path)

        if len(spec.sparse_cols) != len(spec.sparse_vocab_sizes):
            raise ValueError("sparse_cols and sparse_vocab_sizes length mismatch")
        if len(spec.seq_cols) != len(spec.seq_vocab_sizes):
            raise ValueError("seq_cols and seq_vocab_sizes length mismatch")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        dense = torch.tensor([float(row.get(c, 0.0)) for c in self.spec.dense_cols], dtype=torch.float)

        sparse_vals = []
        for c, vocab in zip(self.spec.sparse_cols, self.spec.sparse_vocab_sizes):
            sparse_vals.append(_safe_bucket(int(row.get(c, 0)), vocab))
        sparse = torch.tensor(sparse_vals, dtype=torch.long)

        seq_fields = []
        for c, vocab in zip(self.spec.seq_cols, self.spec.seq_vocab_sizes):
            seq = _parse_seq(row.get(c, ""), self.spec.max_seq_len, self.spec.seq_delim)
            seq_fields.append([_safe_bucket(v, vocab) for v in seq])
        seq = torch.tensor(seq_fields, dtype=torch.long)

        label = torch.tensor(float(row.get(self.spec.label_col, 0.0)), dtype=torch.float)
        user_id = str(row.get(self.spec.user_id_col, ""))
        return dense, sparse, seq, label, user_id


def ctr_collate_fn(batch: Iterable):
    dense, sparse, seq, label, user_id = zip(*batch)
    return {
        "dense": torch.stack(dense, dim=0),
        "sparse": torch.stack(sparse, dim=0),
        "seq": torch.stack(seq, dim=0),
        "label": torch.stack(label, dim=0),
        "user_id": list(user_id),
    }


def build_ctr_dataloader(
    file_path: str | Path,
    spec: CTRDatasetSpec,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    ds = CTRDataset(file_path=file_path, spec=spec)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ctr_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
