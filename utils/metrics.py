from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import torch


def _to_tensor(x: Iterable | torch.Tensor, dtype=torch.float) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(dtype)
    return torch.tensor(list(x), dtype=dtype)


def auc_score(y_true: Sequence[float] | torch.Tensor, y_score: Sequence[float] | torch.Tensor) -> float:
    y_true_t = _to_tensor(y_true, dtype=torch.float)
    y_score_t = _to_tensor(y_score, dtype=torch.float)

    pos = y_true_t == 1
    neg = y_true_t == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = torch.argsort(y_score_t)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(y_score_t) + 1, dtype=torch.float)
    sum_pos_ranks = ranks[pos].sum().item()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def logloss_score(y_true: Sequence[float] | torch.Tensor, y_prob: Sequence[float] | torch.Tensor, eps: float = 1e-8) -> float:
    y_true_t = _to_tensor(y_true, dtype=torch.float)
    y_prob_t = _to_tensor(y_prob, dtype=torch.float).clamp(min=eps, max=1.0 - eps)
    loss = -(y_true_t * torch.log(y_prob_t) + (1.0 - y_true_t) * torch.log(1.0 - y_prob_t))
    return float(loss.mean().item())


def gauc_score(
    user_ids: Sequence[str],
    y_true: Sequence[float] | torch.Tensor,
    y_score: Sequence[float] | torch.Tensor,
) -> float:
    y_true_t = _to_tensor(y_true, dtype=torch.float)
    y_score_t = _to_tensor(y_score, dtype=torch.float)

    group_idx = defaultdict(list)
    for i, uid in enumerate(user_ids):
        group_idx[uid].append(i)

    weighted_auc = 0.0
    total_weight = 0
    for _, idxs in group_idx.items():
        yt = y_true_t[idxs]
        ys = y_score_t[idxs]

        has_pos = bool((yt == 1).any().item())
        has_neg = bool((yt == 0).any().item())
        if not (has_pos and has_neg):
            continue

        weight = len(idxs)
        weighted_auc += auc_score(yt, ys) * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5
    return float(weighted_auc / total_weight)
