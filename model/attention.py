"""
Quantum — Multi-Head Self-Attention with RoPE + GQA + KV Cache.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kv_cache import KVCache
from .rope import RoPE


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPE(self.d_head, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        offset = cache.current_len if cache is not None else 0
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update(layer_idx, k, v)

        # GQA — expand K and V to match n_heads
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        T_full = k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        causal_mask = torch.triu(
            torch.ones(T, T_full, device=x.device), diagonal=T_full - T + 1
        ).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)
