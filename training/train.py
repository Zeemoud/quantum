"""
Quantum — Training loop with mixed precision and gradient accumulation.
Usage: python -m training.train
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from model import QuantumModel, CONFIG
from model.tokenizer import QuantumTokenizer
from model.config import QuantumConfig
from training.dataset import load_text_file, build_dataloader
from training.evaluate import evaluate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(step: int, config: QuantumConfig) -> float:
    """Linear warmup then cosine decay."""
    import math
    if step < config.warmup_steps:
        return config.learning_rate * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(model: nn.Module, step: int, loss: float, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "model_state": model.state_dict(),
        "loss": loss,
        "config": model.config,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


def train():
    device = get_device()
    use_amp = device.type == "cuda"  # Mixed precision only on CUDA
    print(f"Training on : {device}")
    print(f"Mixed precision : {'enabled' if use_amp else 'disabled'}")

    # --- Tokenizer ---
    tokenizer_path = "checkpoints/tokenizer.json"
    if Path(tokenizer_path).exists():
        print("Loading tokenizer...")
        tokenizer = QuantumTokenizer.load(tokenizer_path)
    else:
        print("Training tokenizer...")
        texts = [load_text_file(p) for p in Path("data").glob("*.txt")]
        if not texts:
            raise FileNotFoundError("No .txt files found in data/")
        tokenizer = QuantumTokenizer(vocab_size=CONFIG.vocab_size)
        tokenizer.train(texts)
        tokenizer.save(tokenizer_path)
        print(f"  ✓ Vocabulary size: {len(tokenizer)}")

    # --- Dataset ---
    print("Building dataset...")
    texts = [load_text_file(p) for p in Path("data").glob("*.txt")]
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    print(f"  ✓ Total tokens: {len(all_ids):,}")

    split = int(0.9 * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    train_loader = build_dataloader(train_ids, CONFIG.max_seq_len, CONFIG.batch_size)
    val_loader = build_dataloader(val_ids, CONFIG.max_seq_len, CONFIG.batch_size, shuffle=False)

    # --- Model ---
    model = QuantumModel(CONFIG).to(device)
    print(f"  ✓ Parameters: {model.num_parameters():,}")

    # Resume from checkpoint if available
    step = 0
    best_val_loss = float("inf")
    resume_path = sorted(Path("checkpoints").glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if resume_path:
        latest = resume_path[-1]
        ckpt = torch.load(str(latest), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        step = ckpt["step"]
        best_val_loss = ckpt.get("loss", float("inf"))
        print(f"  ✓ Resumed from {latest.name} (step {step})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG.learning_rate,
        weight_decay=CONFIG.weight_decay,
        betas=(0.9, 0.95),  # Better for LLMs than default (0.9, 0.999)
    )
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.pad_token_id)

    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Gradient accumulation — simulate larger batch size
    # Effective batch size = batch_size × accumulation_steps
    accumulation_steps = 4

    # --- Training loop ---
    print(f"\nStarting training (effective batch size: {CONFIG.batch_size * accumulation_steps})...\n")

    for epoch in range(1, 1000):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # Learning rate schedule
            lr = get_lr(step, CONFIG)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / accumulation_steps  # Normalize for accumulation

            # Backward pass
            scaler.scale(loss).backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step += 1

                pbar.set_postfix(
                    loss=f"{loss.item() * accumulation_steps:.4f}",
                    lr=f"{lr:.2e}",
                )

                # Save checkpoint every 1000 steps
                if step % 1000 == 0:
                    save_checkpoint(model, step, loss.item(), f"checkpoints/step_{step}.pt")

                if step >= CONFIG.max_steps:
                    break

            epoch_loss += loss.item() * accumulation_steps

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch} — train: {avg_train_loss:.4f} | val: {val_metrics['loss']:.4f} | ppl: {val_metrics['perplexity']:.2f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, step, val_metrics["loss"], "checkpoints/best.pt")

        if step >= CONFIG.max_steps:
            print("\nTraining complete.")
            break


if __name__ == "__main__":
    train()