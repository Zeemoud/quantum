"""
Quantum — Evaluation utilities.
Perplexity is the main metric: lower = better.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_tokens = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        # Reshape for loss: (B*T, vocab_size) vs (B*T,)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    model.train()
    return {"loss": avg_loss, "perplexity": perplexity}