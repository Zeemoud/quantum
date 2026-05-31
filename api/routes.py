"""
Quantum API routes.
POST /api/chat         — generate a response (standard)
POST /api/chat/stream  — stream tokens one by one (SSE)
"""

import asyncio
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from model import CONFIG, QuantumModel
from model.kv_cache import KVCache
from model.tokenizer import QuantumTokenizer

router = APIRouter()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_model() -> tuple[QuantumModel, QuantumTokenizer] | None:
    tokenizer_path = Path("checkpoints/tokenizer.json")
    # Use best.pt if available, otherwise use the latest step checkpoint
    checkpoint_path = Path("checkpoints/best.pt")
    if not checkpoint_path.exists():
        candidates = sorted(Path("checkpoints").glob("step_*.pt"))
        if candidates:
            checkpoint_path = candidates[-1]

    if not tokenizer_path.exists() or not checkpoint_path.exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = QuantumTokenizer.load(str(tokenizer_path))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = QuantumModel(CONFIG).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2


class ChatResponse(BaseModel):
    response: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------


def generate_tokens(
    model: QuantumModel,
    tokenizer: QuantumTokenizer,
    message: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
):
    """Generator that yields decoded tokens one by one."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(message, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    cache = KVCache(
        max_batch_size=1,
        max_seq_len=CONFIG.max_seq_len,
        n_layers=CONFIG.n_layers,
        n_kv_heads=CONFIG.n_kv_heads,
        d_head=CONFIG.d_model // CONFIG.n_heads,
        device=device,
    )

    with torch.no_grad():
        logits = model(input_tensor, cache=cache)
        generated = list(input_ids)

        for _ in range(max_new_tokens):
            last_logits = logits[:, -1, :].clone()

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in generated:
                    if last_logits[0, token_id] > 0:
                        last_logits[0, token_id] /= repetition_penalty
                    else:
                        last_logits[0, token_id] *= repetition_penalty

            last_logits = last_logits / temperature

            # Top-k
            if top_k > 0:
                values, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < values[:, -1:]] = float("-inf")

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_indices_to_remove] = float("-inf")
                last_logits = torch.scatter(last_logits, 1, sorted_indices, sorted_logits)

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()
            generated.append(token_id)

            if token_id == CONFIG.eos_token_id:
                break

            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            yield token_text

            logits = model(next_token, cache=cache)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = get_model()
    if result is None:
        return ChatResponse(
            response="Quantum is not trained yet. Run `python -m training.train` first.",
            model_loaded=False,
        )
    model, tokenizer = result
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        tokens = list(
            generate_tokens(
                model,
                tokenizer,
                request.message,
                request.max_new_tokens,
                request.temperature,
                request.top_k,
                request.top_p,
                request.repetition_penalty,
            )
        )
        return ChatResponse(response="".join(tokens), model_loaded=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    result = get_model()

    if result is None:

        async def not_trained():
            yield "data: Quantum is not trained yet.\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(not_trained(), media_type="text/event-stream")

    model, tokenizer = result

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    async def stream():
        try:
            for token in generate_tokens(
                model,
                tokenizer,
                request.message,
                request.max_new_tokens,
                request.temperature,
                request.top_k,
                request.top_p,
                request.repetition_penalty,
            ):
                yield f"data: {token}\n\n"
                await asyncio.sleep(0)  # Yield control to event loop
        except Exception as e:
            yield f"data: [ERROR] {e}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/status")
async def status():
    result = get_model()
    if result is None:
        return {"status": "not_trained", "message": "No checkpoint found."}
    model, tokenizer = result
    return {
        "status": "ready",
        "parameters": model.num_parameters(),
        "vocab_size": len(tokenizer),
    }
