"""
Quantum — Benchmark script.
Measures generation speed, latency and VRAM usage.
Usage: python -m scripts.benchmark
       python -m scripts.benchmark --checkpoint checkpoints/step_21000.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import CONFIG, QuantumModel
from model.tokenizer import QuantumTokenizer


def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = QuantumTokenizer.load("checkpoints/tokenizer.json")
    ckpt = sorted(Path("checkpoints").glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    path = Path(checkpoint_path)
    if not path.exists() and ckpt:
        path = ckpt[-1]
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    model = QuantumModel(CONFIG).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, device


def vram_used(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 ** 2
    return 0.0


def benchmark_generation(model, tokenizer, device, prompt: str, n_tokens: int = 100) -> dict:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.token_to_id[tokenizer.bos_token]] + input_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Warmup
    with torch.no_grad():
        _ = model.generate(input_tensor, max_new_tokens=5)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # Measure latency (time to first token)
    start_latency = time.perf_counter()
    with torch.no_grad():
        idx = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(idx)
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        _ = torch.multinomial(probs, num_samples=1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = (time.perf_counter() - start_latency) * 1000

    # Measure throughput with KV cache
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=n_tokens)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    actual_tokens = output.shape[1] - len(input_ids)
    tokens_per_sec = actual_tokens / elapsed

    vram = 0.0
    if device.type == "cuda":
        vram = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    return {
        "latency_ms": latency,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": actual_tokens,
        "elapsed_sec": elapsed,
        "vram_mb": vram,
    }


def main():
    parser = argparse.ArgumentParser(description="Quantum Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--tokens", type=int, default=100)
    args = parser.parse_args()

    print("\n🔬 Quantum Benchmark")
    print("Loading model...", end=" ", flush=True)
    model, tokenizer, device = load_model(args.checkpoint)
    print(f"✓ ({model.num_parameters():,} params on {device})\n")

    prompts = [
        "Bonjour, comment",
        "Il était une fois",
        "The captain looked at",
        "Dans la nuit sombre",
    ]

    results = []
    for prompt in prompts:
        print(f"  Prompt: '{prompt}'")
        result = benchmark_generation(model, tokenizer, device, prompt, args.tokens)
        results.append(result)
        print(f"  ├─ Latency       : {result['latency_ms']:.1f} ms")
        print(f"  ├─ Tokens/sec    : {result['tokens_per_sec']:.1f}")
        print(f"  ├─ Tokens generated: {result['total_tokens']}")
        if result["vram_mb"] > 0:
            print(f"  └─ VRAM peak     : {result['vram_mb']:.0f} MB")
        else:
            print("  └─ VRAM          : N/A (CPU)")
        print()

    # Summary
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_lat = sum(r["latency_ms"] for r in results) / len(results)
    avg_vram = sum(r["vram_mb"] for r in results) / len(results)

    print("─" * 40)
    print(f"  Avg tokens/sec  : {avg_tps:.1f}")
    print(f"  Avg latency     : {avg_lat:.1f} ms")
    if avg_vram > 0:
        print(f"  Avg VRAM peak   : {avg_vram:.0f} MB")
    print(f"  Parameters      : {model.num_parameters():,}")
    print(f"  Device          : {device}")
    print()


if __name__ == "__main__":
    main()