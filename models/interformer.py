from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class InterFormerConfig:
    num_dense: int
    sparse_vocab_sizes: Sequence[int]
    seq_vocab_sizes: Sequence[int]
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    n_nonseq_summary: int = 4
    n_pma_tokens: int = 2
    n_recent_tokens: int = 2
    max_seq_len: int = 100
    interaction_arch: str = "dhen"
    use_rope: bool = False
    rope_base: float = 10000.0
    interleave_mode: str = "int"  # one of: sole, sep, n2s, s2n, int

    @property
    def num_sparse(self) -> int:
        return len(self.sparse_vocab_sizes)

    @property
    def num_seq_fields(self) -> int:
        return len(self.seq_vocab_sizes)

    @property
    def n_nonseq_tokens(self) -> int:
        # 1 dense token + num_sparse sparse tokens
        return 1 + self.num_sparse


class MLP(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.0, activation: str = "swish") -> None:
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if activation == "swish":
                    layers.append(nn.SiLU())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenLCE(nn.Module):
    """Linear compression over token axis: [B, N, D] -> [B, M, D]."""

    def __init__(self, in_tokens: int, out_tokens: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_tokens, out_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.proj(x)  # [B, D, M]
        return x.transpose(1, 2)  # [B, M, D]


class Gating(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.gate(x))


class SequencePreprocessor(nn.Module):
    """MaskNet-style preprocessing: self-mask then LCE compression."""

    def __init__(self, cfg: InterFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.seq_embeddings = nn.ModuleList([nn.Embedding(v, cfg.d_model) for v in cfg.seq_vocab_sizes])
        seq_in_dim = cfg.num_seq_fields * cfg.d_model
        self.mask_mlp = nn.Sequential(nn.Linear(seq_in_dim, seq_in_dim), nn.SiLU(), nn.Linear(seq_in_dim, seq_in_dim))
        self.lce_mlp = nn.Linear(seq_in_dim, cfg.d_model)

    def forward(self, seq_x: torch.Tensor) -> torch.Tensor:
        # seq_x: [B, K, T]
        embeds = []
        for i, emb in enumerate(self.seq_embeddings):
            embeds.append(emb(seq_x[:, i, :]))  # [B, T, D]
        x = torch.cat(embeds, dim=-1)  # [B, T, K*D]
        x = x * torch.sigmoid(self.mask_mlp(x))
        return self.lce_mlp(x)


class PersonalizedFFN(nn.Module):
    """W_pffn = X_sum^T W, then apply to sequence tokens."""

    def __init__(self, n_nonseq_summary: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_nonseq_summary, d_model) * 0.02)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, xsum: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        # xsum: [B, Nsum, D], seq: [B, T, D]
        wpffn = torch.einsum("bdn,nk->bdk", xsum.transpose(1, 2), self.weight)  # [B, D, D]
        seq = torch.einsum("bti,bij->btj", seq, wpffn)
        return self.out_proj(seq)


class DotProductInteraction(nn.Module):
    """DOT interaction module in DHEN-style ensemble."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        attn_logits = torch.matmul(x, x.transpose(1, 2)) * self.scale  # [B, N, N]
        attn = torch.softmax(attn_logits, dim=-1)
        out = torch.matmul(attn, x)
        return self.dropout(out)


class DCNTokenInteraction(nn.Module):
    """A DCN-like cross interaction over token dimension."""

    def __init__(self, n_tokens: int, dropout: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_tokens) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_tokens))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x0, x: [B, N, D]
        x_t = x.transpose(1, 2)  # [B, D, N]
        x0_t = x0.transpose(1, 2)  # [B, D, N]
        cross_scalar = torch.einsum("bdn,n->bd", x_t, self.w).unsqueeze(-1)  # [B, D, 1]
        out_t = x0_t * (cross_scalar + self.b.view(1, 1, -1)) + x_t
        out = out_t.transpose(1, 2)
        return self.dropout(out)


class InteractionArchBlockDHEN(nn.Module):
    """DHEN-style interaction block: Ensemble(DOT, DCN) + shortcut + norm."""

    def __init__(self, d_model: int, n_tokens: int, dropout: float) -> None:
        super().__init__()
        self.dot_inter = DotProductInteraction(d_model=d_model, dropout=dropout)
        self.dcn_inter = DCNTokenInteraction(n_tokens=n_tokens, dropout=dropout)

        self.ensemble_proj = nn.Linear(d_model * 2, d_model)
        self.shortcut = MLP([d_model, d_model], dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.post_mlp = MLP([d_model, d_model * 2, d_model], dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_nonseq: torch.Tensor, ssum: torch.Tensor) -> torch.Tensor:
        n_nonseq = x_nonseq.size(1)
        z = torch.cat([x_nonseq, ssum], dim=1)  # [B, N_total, D]

        dot_out = self.dot_inter(z)
        dcn_out = self.dcn_inter(z, z)

        ensemble = self.ensemble_proj(torch.cat([dot_out, dcn_out], dim=-1))
        z = self.norm(ensemble + self.shortcut(z))
        z = z + self.drop(self.post_mlp(z))

        return z[:, :n_nonseq, :]


class InteractionArchBlockMHA(nn.Module):
    """Behavior-aware interaction on [X || S_sum] with MHA."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = MLP([d_model, d_model * 2, d_model], dropout=dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x_nonseq: torch.Tensor, ssum: torch.Tensor) -> torch.Tensor:
        n_nonseq = x_nonseq.size(1)
        z = torch.cat([x_nonseq, ssum], dim=1)
        a, _ = self.attn(self.norm1(z), self.norm1(z), self.norm1(z), need_weights=False)
        z = z + self.drop(a)
        z = z + self.drop(self.ffn(self.norm2(z)))
        return z[:, :n_nonseq, :]


class RoPEMultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, rope_base: float = 10000.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for RoPE attention")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.rope_base = rope_base
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _rope_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.head_dim // 2
        theta = 1.0 / (self.rope_base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, theta)  # [T, head_dim/2]
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype=dtype)  # [T, head_dim]
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype=dtype)  # [T, head_dim]
        return cos[None, None, :, :], sin[None, None, :, :]

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return x * cos + self._rotate_half(x) * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # [B, H, T, Dh]
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        cos, sin = self._rope_cos_sin(seq_len, x.device, x.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # [B, H, T, Dh]
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class SequenceArchBlock(nn.Module):
    def __init__(self, cfg: InterFormerConfig) -> None:
        super().__init__()
        self.use_rope = cfg.use_rope
        self.pffn = PersonalizedFFN(cfg.n_nonseq_summary, cfg.d_model)
        self.norm1 = nn.LayerNorm(cfg.d_model)

        if self.use_rope:
            self.mha_rope = RoPEMultiheadSelfAttention(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                dropout=cfg.dropout,
                rope_base=cfg.rope_base,
            )
            self.mha = None
        else:
            self.mha = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            self.mha_rope = None

        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MLP([cfg.d_model, cfg.d_model * 2, cfg.d_model], dropout=cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, xsum: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        z = seq + self.drop(self.pffn(xsum, seq))

        if self.use_rope:
            a = self.mha_rope(self.norm1(z))
        else:
            a, _ = self.mha(self.norm1(z), self.norm1(z), self.norm1(z), need_weights=False)

        z = z + self.drop(a)
        z = z + self.drop(self.ffn(self.norm2(z)))
        return z


class CrossArch(nn.Module):
    def __init__(self, cfg: InterFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.nonseq_lce = TokenLCE(cfg.n_nonseq_tokens, cfg.n_nonseq_summary)
        self.nonseq_gate = Gating(cfg.d_model)

        self.pma_query = nn.Parameter(torch.randn(cfg.n_pma_tokens, cfg.d_model) * 0.02)
        self.pma = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.seq_gate = Gating(cfg.d_model)

    def summarize_nonseq(self, x_nonseq: torch.Tensor) -> torch.Tensor:
        xsum = self.nonseq_lce(x_nonseq)
        return self.nonseq_gate(xsum)

    def summarize_seq(self, seq: torch.Tensor) -> torch.Tensor:
        bsz = seq.size(0)
        cls_part = seq[:, : self.cfg.n_nonseq_summary, :]

        q = self.pma_query.unsqueeze(0).expand(bsz, -1, -1)
        pma_part, _ = self.pma(q, seq, seq, need_weights=False)

        recent = seq[:, -self.cfg.n_recent_tokens :, :]
        ssum = torch.cat([cls_part, pma_part, recent], dim=1)
        return self.seq_gate(ssum)


class InterFormer(nn.Module):
    def __init__(self, cfg: InterFormerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        valid_modes = {"sole", "sep", "n2s", "s2n", "int"}
        if cfg.interleave_mode not in valid_modes:
            raise ValueError(f"Unsupported interleave_mode: {cfg.interleave_mode}. Use one of {sorted(valid_modes)}")

        self.dense_proj = nn.Linear(cfg.num_dense, cfg.d_model)
        self.sparse_embeddings = nn.ModuleList([nn.Embedding(v, cfg.d_model) for v in cfg.sparse_vocab_sizes])
        self.seq_pre = SequencePreprocessor(cfg)

        self.cross = CrossArch(cfg)
        self.cls_tokens = nn.Parameter(torch.randn(cfg.n_nonseq_summary, cfg.d_model) * 0.02)

        self.n_seq_summary = cfg.n_nonseq_summary + cfg.n_pma_tokens + cfg.n_recent_tokens
        n_interaction_tokens = cfg.n_nonseq_tokens + self.n_seq_summary

        interaction_arch = cfg.interaction_arch.lower()
        if interaction_arch == "dhen":
            self.interaction_blocks = nn.ModuleList(
                [
                    InteractionArchBlockDHEN(
                        d_model=cfg.d_model,
                        n_tokens=n_interaction_tokens,
                        dropout=cfg.dropout,
                    )
                    for _ in range(cfg.n_layers)
                ]
            )
        elif interaction_arch == "mha":
            self.interaction_blocks = nn.ModuleList(
                [
                    InteractionArchBlockMHA(
                        d_model=cfg.d_model,
                        n_heads=cfg.n_heads,
                        dropout=cfg.dropout,
                    )
                    for _ in range(cfg.n_layers)
                ]
            )
        else:
            raise ValueError(f"Unsupported interaction_arch: {cfg.interaction_arch}. Use 'dhen' or 'mha'.")

        self.sequence_blocks = nn.ModuleList([SequenceArchBlock(cfg) for _ in range(cfg.n_layers)])
        self.head = MLP([cfg.d_model * (cfg.n_nonseq_summary + self.n_seq_summary), 256, 64, 1], dropout=cfg.dropout)

    def preprocess_nonseq(self, dense_x: torch.Tensor, sparse_x: torch.Tensor) -> torch.Tensor:
        dense_token = self.dense_proj(dense_x).unsqueeze(1)  # [B, 1, D]
        sparse_tokens = [emb(sparse_x[:, i]).unsqueeze(1) for i, emb in enumerate(self.sparse_embeddings)]
        return torch.cat([dense_token] + sparse_tokens, dim=1)

    def _blank_seq_summary(self, seq: torch.Tensor) -> torch.Tensor:
        bsz = seq.size(0)
        return torch.zeros(bsz, self.n_seq_summary, self.cfg.d_model, device=seq.device, dtype=seq.dtype)

    def _blank_nonseq_summary(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        return torch.zeros(bsz, self.cfg.n_nonseq_summary, self.cfg.d_model, device=x.device, dtype=x.dtype)

    def forward(self, dense_x: torch.Tensor, sparse_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        # X(1), S(1)
        x = self.preprocess_nonseq(dense_x, sparse_x)
        s = self.seq_pre(seq_x)

        # first-layer prepend: only allow non-seq->seq when mode in {int, n2s}
        use_nonseq_to_seq = self.cfg.interleave_mode in {"int", "n2s"}
        if use_nonseq_to_seq:
            xsum_init = self.cross.summarize_nonseq(x)
        else:
            xsum_init = self.cls_tokens.unsqueeze(0).expand(x.size(0), -1, -1)
        s = torch.cat([xsum_init, s], dim=1)

        if self.cfg.interleave_mode == "sole":
            # naive early sequence summarization, then interaction-only stacking
            seq_mean = s[:, self.cfg.n_nonseq_summary :, :].mean(dim=1, keepdim=True)
            ssum_sole = seq_mean.expand(-1, self.n_seq_summary, -1)
            for inter_block in self.interaction_blocks:
                x = inter_block(x, ssum_sole)
            xsum = self.cross.summarize_nonseq(x)
            ssum = ssum_sole
        else:
            for inter_block, seq_block in zip(self.interaction_blocks, self.sequence_blocks):
                xsum = self.cross.summarize_nonseq(x)
                ssum = self.cross.summarize_seq(s)
                zero_xsum = self._blank_nonseq_summary(x)
                zero_ssum = self._blank_seq_summary(s)

                if self.cfg.interleave_mode == "int":
                    x = inter_block(x, ssum)
                    s = seq_block(xsum, s)
                elif self.cfg.interleave_mode == "sep":
                    x = inter_block(x, zero_ssum)
                    s = seq_block(zero_xsum, s)
                elif self.cfg.interleave_mode == "n2s":
                    x = inter_block(x, zero_ssum)
                    s = seq_block(xsum, s)
                elif self.cfg.interleave_mode == "s2n":
                    x = inter_block(x, ssum)
                    s = seq_block(zero_xsum, s)

            xsum = self.cross.summarize_nonseq(x)
            ssum = self.cross.summarize_seq(s)

        features = torch.cat([xsum.flatten(1), ssum.flatten(1)], dim=1)
        logit = self.head(features).squeeze(-1)
        return logit

    def predict_proba(self, dense_x: torch.Tensor, sparse_x: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(dense_x, sparse_x, seq_x))
