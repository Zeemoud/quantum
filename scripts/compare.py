"""
Quantum — Compare two checkpoints.
Usage: python -m scripts.compare --a checkpoints/step_21000.pt --b checkpoints/step_26000.pt
"""

import argparse

import torch
import torch.nn.functional as F

from model import CONFIG, QuantumModel
from model.tokenizer import QuantumTokenizer


def load(path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = QuantumTokenizer.load("checkpoints/tokenizer.json")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = QuantumModel(CONFIG).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, device, ckpt.get("loss", "?"), ckpt.get("step", "?")


def generate(model, tokenizer, device, prompt: str, max_tokens: int = 80) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.token_to_id[tokenizer.bos_token]] + input_ids

    with torch.no_grad():
        generated = list(input_ids)
        new_ids = []
        for _ in range(max_tokens):
            idx = torch.tensor([generated], dtype=torch.long, device=device)
            if idx.shape[1] > CONFIG.max_seq_len:
                idx = idx[:, -CONFIG.max_seq_len :]
            logits = model(idx)
            last_logits = logits[:, -1, :] / 0.8
            values, _ = torch.topk(last_logits, 50)
            last_logits[last_logits < values[:, -1:]] = float("-inf")
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            generated.append(token_id)
            new_ids.append(token_id)
            if token_id == CONFIG.eos_token_id:
                break
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Compare two Quantum checkpoints")
    parser.add_argument("--a", required=True, help="First checkpoint")
    parser.add_argument("--b", required=True, help="Second checkpoint")
    parser.add_argument("--tokens", type=int, default=80)
    args = parser.parse_args()

    prompts = [
        "Bonjour, je voudrais",
        "Il était une fois",
        "The captain said",
    ]

    print("\n🔍 Quantum — Checkpoint Comparison\n")

    print("Loading checkpoints...")
    model_a, tok_a, dev_a, loss_a, step_a = load(args.a)
    model_b, tok_b, dev_b, loss_b, step_b = load(args.b)

    loss_a_str = f"{loss_a:.4f}" if isinstance(loss_a, float) else str(loss_a)
    loss_b_str = f"{loss_b:.4f}" if isinstance(loss_b, float) else str(loss_b)
    print(f"  A: step={step_a} loss={loss_a_str}")
    print(f"  B: step={step_b} loss={loss_b_str}\n")

    for prompt in prompts:
        print(f"{'─' * 60}")
        print(f'Prompt: "{prompt}"\n')
        out_a = generate(model_a, tok_a, dev_a, prompt, args.tokens)
        out_b = generate(model_b, tok_b, dev_b, prompt, args.tokens)
        print(f"[A step={step_a}]\n{out_a}\n")
        print(f"[B step={step_b}]\n{out_b}\n")


if __name__ == "__main__":
    main()
