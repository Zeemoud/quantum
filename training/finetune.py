"""
Quantum — Conversational fine-tuning.
Fine-tunes the base model on chat data using the [USER]/[ASSISTANT] template.

Usage:
    python -m training.finetune
    python -m training.finetune --checkpoint checkpoints/step_50000.pt
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import math

from model import QuantumModel, CONFIG
from model.tokenizer import QuantumTokenizer
from model.config import QuantumConfig
from training.dataset import build_dataloader, load_text_file
from training.evaluate import evaluate


# Chat special tokens
USER_TOKEN = "[USER]"
ASSISTANT_TOKEN = "[ASSISTANT]"
END_TOKEN = "[END]"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(step: int, max_steps: int, learning_rate: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return learning_rate * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(model: nn.Module, step: int, loss: float, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "loss": loss,
        "config": model.config,
        "type": "finetune",
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


def finetune():
    device = get_device()
    use_amp = device.type == "cuda"
    print(f"Fine-tuning on: {device}")
    print(f"Mixed precision: {'enabled' if use_amp else 'disabled'}")

    # Fine-tuning hyperparams — smaller lr, fewer steps
    FT_LR = 5e-5           # 6x smaller than pretraining
    FT_STEPS = 2000        # Short fine-tuning
    FT_WARMUP = 100
    FT_BATCH = 8           # Smaller batch
    ACCUMULATION = 4
    DROPOUT = 0.2          # Higher dropout to prevent overfitting

    # --- Tokenizer ---
    tokenizer_path = "checkpoints/tokenizer.json"
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError("Tokenizer not found. Run training first.")
    tokenizer = QuantumTokenizer.load(tokenizer_path)

    # Add chat special tokens to tokenizer
    for token in [USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN]:
        if token not in tokenizer.token_to_id:
            idx = len(tokenizer.token_to_id)
            tokenizer.token_to_id[token] = idx
            tokenizer.id_to_token[idx] = token
            print(f"  Added token: {token} → id {idx}")

    # --- Dataset ---
    chat_file = Path("data/corpus_chat.txt")
    if not chat_file.exists():
        raise FileNotFoundError("Chat data not found. Run prepare_chat_data first.")

    print("Building chat dataset...")
    text = load_text_file(str(chat_file))
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"  ✓ Total tokens: {len(all_ids):,}")

    split = int(0.9 * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    train_loader = build_dataloader(train_ids, CONFIG.max_seq_len, FT_BATCH)
    val_loader = build_dataloader(
        val_ids, CONFIG.max_seq_len, FT_BATCH, shuffle=False
    )

    # --- Model — load from checkpoint ---
    # Find best base checkpoint
    checkpoint_arg = "checkpoints/step_50000.pt"
    candidates = sorted(
        Path("checkpoints").glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    ckpt_path = Path(checkpoint_arg)
    if not ckpt_path.exists() and candidates:
        ckpt_path = candidates[-1]
        print(f"  Using checkpoint: {ckpt_path}")

    print(f"Loading base model from {ckpt_path.name}...")
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Update config with higher dropout for fine-tuning
    ft_config = QuantumConfig(**{
        **CONFIG.__dict__,
        "dropout": DROPOUT,
    })

    model = QuantumModel(ft_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  ✓ Parameters: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FT_LR,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.pad_token_id)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Fine-tuning loop ---
    print(f"\nStarting fine-tuning ({FT_STEPS} steps)...\n")
    step = 0
    best_val_loss = float("inf")

    for epoch in range(1, 1000):
        model.train()
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            lr = get_lr(step, FT_STEPS, FT_LR, FT_WARMUP)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / ACCUMULATION

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

                pbar.set_postfix(
                    step=step,
                    loss=f"{loss.item() * ACCUMULATION:.4f}",
                    lr=f"{lr:.2e}",
                )

                if step % 500 == 0:
                    save_checkpoint(
                        model, step, loss.item(),
                        f"checkpoints/ft_step_{step}.pt",
                    )

                if step >= FT_STEPS:
                    break

        # Validation
        if len(val_ids) > CONFIG.max_seq_len:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"\nFT Epoch {epoch} — "
                f"val loss: {val_metrics['loss']:.4f} | "
                f"ppl: {val_metrics['perplexity']:.2f}"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    model, step, val_metrics["loss"],
                    "checkpoints/ft_best.pt",
                )

        if step >= FT_STEPS:
            print("\nFine-tuning complete!")
            # Save final tokenizer with new tokens
            tokenizer.save("checkpoints/ft_tokenizer.json")
            print("  ✓ Tokenizer saved → checkpoints/ft_tokenizer.json")
            break


if __name__ == "__main__":
    finetune()