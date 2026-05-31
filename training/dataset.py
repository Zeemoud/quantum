"""
Quantum — Dataset and DataLoader utilities.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = chunk[:-1]  # Input
        y = chunk[1:]  # Target (shifted by 1)
        return x, y


def load_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def build_dataloader(
    token_ids: list[int],
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TextDataset(token_ids, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )
