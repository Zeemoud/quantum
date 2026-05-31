"""
Quantum — RMSNorm (Root Mean Square Layer Normalization).
Simpler and faster than LayerNorm — no mean subtraction, no bias.
Used by LLaMA, Mistral, and most modern LLMs.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x²))
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight