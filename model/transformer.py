"""
Quantum — Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention
from .config import QuantumConfig
from .norm import RMSNorm
from .kv_cache import KVCache


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: QuantumConfig):
        super().__init__()
        self.attn = MultiHeadAttention(config.d_model, config.n_heads, config.n_kv_heads, config.max_seq_len, config.dropout)
        self.ff = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: "KVCache | None" = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask, cache, layer_idx)
        x = x + self.ff(self.norm2(x))
        return x


class QuantumModel(nn.Module):
    def __init__(self, config: QuantumConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying — share embeddings with output head
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Scaled init — divise par sqrt(2 * n_layers) pour stabiliser
                # les résidus qui s'accumulent couche après couche
                std = 0.02
                if "out_proj" in name or "down" in name:
                    std /= (2 * self.config.n_layers) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                # Zero out padding token embedding
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, f"Sequence too long: {T} > {self.config.max_seq_len}"

        x = self.dropout(self.token_emb(input_ids))

        for i, block in enumerate(self.blocks):
            x = block(x, mask, cache, layer_idx=i)

        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        B = input_ids.shape[0]
        device = input_ids.device

        cache = KVCache(
            max_batch_size=B,
            max_seq_len=self.config.max_seq_len,
            n_layers=self.config.n_layers,
            n_kv_heads=self.config.n_kv_heads,
            d_head=self.config.d_model // self.config.n_heads,
            device=device,
        )

        # Prefill
        logits = self(input_ids, cache=cache)
        cache.advance(input_ids.shape[1])
        generated = input_ids
        generated_list = input_ids[0].tolist()  # Liste Python pour la penalty

        for _ in range(max_new_tokens):
            last_logits = logits[:, -1, :].clone()

            # Repetition penalty — vectorisé
            if repetition_penalty != 1.0:
                unique_ids = list(set(generated_list))
                penalty_mask = last_logits[0, unique_ids] > 0
                last_logits[0, unique_ids] = torch.where(
                    penalty_mask,
                    last_logits[0, unique_ids] / repetition_penalty,
                    last_logits[0, unique_ids] * repetition_penalty,
                )

            last_logits = last_logits / temperature

            if top_k > 0:
                values, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                last_logits = torch.scatter(last_logits, 1, sorted_indices, sorted_logits)

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            generated = torch.cat([generated, next_token], dim=1)
            generated_list.append(token_id)
            if token_id == self.config.eos_token_id:
                break
            logits = self(next_token, cache=cache)
            cache.advance(1)

        return generated

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())