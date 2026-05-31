"""
Quantum CLI — Test the model directly in the terminal.
Usage: python -m scripts.chat
       python -m scripts.chat --checkpoint checkpoints/step_6000.pt
       python -m scripts.chat --temp 0.9 --top-p 0.95
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from model import CONFIG, QuantumModel
from model.tokenizer import QuantumTokenizer


def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_path = Path("checkpoints/tokenizer.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError("Tokenizer not found at checkpoints/tokenizer.json")

    tokenizer = QuantumTokenizer.load(str(tokenizer_path))

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        candidates = sorted(Path("checkpoints").glob("step_*.pt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found. Train the model first.")
        ckpt = candidates[-1]
        print(f"  Using checkpoint: {ckpt}")

    checkpoint = torch.load(str(ckpt), map_location=device, weights_only=False)
    model = QuantumModel(CONFIG).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer, device


def generate(model, tokenizer, device, prompt, args):
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.token_to_id[tokenizer.bos_token]] + input_ids

    with torch.no_grad():
        generated = list(input_ids)
        new_ids = []

        for _ in range(args.max_tokens):
            idx = torch.tensor([generated], dtype=torch.long, device=device)
            if idx.shape[1] > CONFIG.max_seq_len:
                idx = idx[:, -CONFIG.max_seq_len:]
            logits = model(idx)
            last_logits = logits[:, -1, :].clone()

            for token_id in generated:
                if last_logits[0, token_id] > 0:
                    last_logits[0, token_id] /= args.rep_penalty
                else:
                    last_logits[0, token_id] *= args.rep_penalty

            last_logits = last_logits / args.temp
            values, _ = torch.topk(last_logits, min(args.top_k, last_logits.size(-1)))
            last_logits[last_logits < values[:, -1:]] = float("-inf")
            sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > args.top_p
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            last_logits = torch.scatter(last_logits, 1, sorted_indices, sorted_logits)
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            generated.append(token_id)
            new_ids.append(token_id)

            if token_id == CONFIG.eos_token_id:
                break

    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Quantum CLI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--rep-penalty", type=float, default=1.2)
    args = parser.parse_args()

    print("\n🔮 Quantum CLI")
    print("Loading model...", end=" ", flush=True)
    model, tokenizer, device = load_model(args.checkpoint)
    print(f"✓ ({model.num_parameters():,} params on {device})")
    print("Type your prompt and press Enter. Ctrl+C to exit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue

            print("Quantum: ", end="", flush=True)
            response = generate(model, tokenizer, device, prompt, args)
            print(response)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break


if __name__ == "__main__":
    main()