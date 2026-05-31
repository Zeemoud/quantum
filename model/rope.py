"""
Quantum — Rotary Position Embeddings (RoPE).
Encodes position directly into Q and K via rotation matrices.
Better than learned position embeddings for long sequences.
Used by LLaMA, Mistral, Qwen, and most modern LLMs.
"""

import torch
import torch.nn as nn


def precompute_freqs(d_head: int, max_seq_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin rotation frequencies."""
    # Frequencies: θ_i = 1 / (base ^ (2i / d_head))
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, theta)  # (max_seq_len, d_head/2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0) -> torch.Tensor:
    T = x.shape[2]
    cos = cos[offset:offset + T].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + T].unsqueeze(0).unsqueeze(0)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    x_rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)

    return x_rotated.flatten(-2)


class RoPE(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        cos, sin = precompute_freqs(d_head, max_seq_len, base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        return apply_rope(x, self.cos, self.sin, offset)