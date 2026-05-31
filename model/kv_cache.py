"""
Quantum — KV Cache for fast inference.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class KVCache:
    max_batch_size: int
    max_seq_len: int
    n_layers: int
    n_kv_heads: int          # GQA — fewer KV heads than Q heads
    d_head: int
    device: torch.device

    k_cache: list[torch.Tensor] = field(init=False)
    v_cache: list[torch.Tensor] = field(init=False)
    current_len: int = field(init=False, default=0)

    def __post_init__(self):
        shape = (self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.d_head)
        self.k_cache = [torch.zeros(shape, device=self.device) for _ in range(self.n_layers)]
        self.v_cache = [torch.zeros(shape, device=self.device) for _ in range(self.n_layers)]

    def update(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, H, T, D = k.shape
        self.k_cache[layer][:B, :, self.current_len:self.current_len + T, :] = k
        self.v_cache[layer][:B, :, self.current_len:self.current_len + T, :] = v
        k_full = self.k_cache[layer][:B, :, :self.current_len + T, :]
        v_full = self.v_cache[layer][:B, :, :self.current_len + T, :]
        return k_full, v_full

    def advance(self, n: int = 1):
        self.current_len += n

    def reset(self):
        self.current_len = 0
        for i in range(self.n_layers):
            self.k_cache[i].zero_()
            self.v_cache[i].zero_()