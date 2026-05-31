"""
Quantum — Loss visualization.
Reads loss values from checkpoint files and plots the training curve.
Usage: python -m scripts.plot_loss
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.style as style
from pathlib import Path


def load_checkpoints() -> list[tuple[int, float]]:
    """Load step and loss from all checkpoints."""
    checkpoints = sorted(Path("checkpoints").glob("step_*.pt"))
    data = []
    for ckpt in checkpoints:
        try:
            step = int(ckpt.stem.split("_")[1])
            info = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            loss = info.get("loss", None)
            if loss is not None:
                data.append((step, loss))
                print(f"  step {step:>6} — loss {loss:.4f}")
        except Exception as e:
            print(f"  Skipping {ckpt.name}: {e}")
    return sorted(data)


def plot(data: list[tuple[int, float]]):
    steps = [d[0] for d in data]
    losses = [d[1] for d in data]

    style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")

    # Plot line
    ax.plot(steps, losses, color="#7c6aff", linewidth=2, label="Train loss")

    # Scatter points
    ax.scatter(steps, losses, color="#a78bfa", s=40, zorder=5)

    # Annotate last point
    ax.annotate(
        f"  loss={losses[-1]:.4f}",
        xy=(steps[-1], losses[-1]),
        color="#a78bfa",
        fontsize=10,
    )

    ax.set_xlabel("Steps", color="#888")
    ax.set_ylabel("Loss", color="#888")
    ax.set_title("Quantum — Training Loss", color="#ececec", fontsize=14, pad=15)
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#2e2e2e")
    ax.spines["left"].set_color("#2e2e2e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#2e2e2e", linestyle="--", linewidth=0.5)
    ax.legend(facecolor="#242424", edgecolor="#2e2e2e", labelcolor="#ececec")

    plt.tight_layout()
    output = "checkpoints/loss_curve.png"
    plt.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    print(f"\n  ✓ Saved → {output}")
    plt.show()


def main():
    print("Loading checkpoints...\n")
    data = load_checkpoints()
    if not data:
        print("No checkpoints found!")
        return
    print(f"\n{len(data)} checkpoints loaded.")
    plot(data)


if __name__ == "__main__":
    main()