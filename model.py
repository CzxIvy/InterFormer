from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from models.interformer import InterFormer, InterFormerConfig


def default_interformer_config() -> InterFormerConfig:
    # Default toy config for quick experiments.
    return InterFormerConfig(
        num_dense=8,
        sparse_vocab_sizes=[10000, 5000, 2000, 3000],
        seq_vocab_sizes=[10000, 2000],
        d_model=64,
        n_layers=3,
        n_heads=4,
        dropout=0.1,
        n_nonseq_summary=4,
        n_pma_tokens=2,
        n_recent_tokens=2,
        max_seq_len=100,
        interaction_arch="dhen",
        use_rope=False,
        rope_base=10000.0,
        interleave_mode="int",
    )


def build_interformer(config_dict: Dict[str, Any] | None = None) -> InterFormer:
    cfg = default_interformer_config()
    if config_dict:
        cfg = InterFormerConfig(**{**asdict(cfg), **config_dict})
    return InterFormer(cfg)
