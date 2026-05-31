# Quantum

[![CI](https://github.com/Zeemoud/quantum/actions/workflows/ci.yml/badge.svg)](https://github.com/Zeemoud/quantum/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Parameters](https://img.shields.io/badge/parameters-34M-purple.svg)]()

An open-source conversational AI built entirely from scratch — no pretrained models, no external AI dependencies.

Every component is hand-crafted: tokenizer, transformer architecture, training loop, and inference engine.

## Architecture

Quantum is a decoder-only transformer with modern improvements:

- **Tokenizer** — Byte Pair Encoding (BPE) trained from scratch
- **RoPE** — Rotary Position Embeddings for better long-range understanding
- **GQA** — Grouped Query Attention (like LLaMA 3) for efficient inference
- **RMSNorm** — Faster and more stable than LayerNorm
- **SwiGLU** — Better feed-forward activation (used by LLaMA, Mistral)
- **KV Cache** — Fast autoregressive generation
- **Mixed precision** — FP16 training on CUDA GPUs

Current model size: **34M parameters**

## Project structure

```
quantum/
├── model/ # Core AI: tokenizer, transformer, attention, RoPE, GQA
├── training/ # Training loop, dataset, evaluation
├── api/ # FastAPI backend
├── web/ # React + TypeScript chat interface
├── scripts/ # CLI, benchmark, loss visualization, checkpoint comparison
├── data/ # Training data (not tracked)
└── checkpoints/ # Model weights (not tracked)
```

## Quick start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Node.js 18+
- CUDA GPU recommended (runs on CPU too)

### Setup

```bash
git clone https://github.com/your-username/quantum.git
cd quantum

# Python
uv sync --extra dev
cp .env.example .env

# Frontend
cd web && npm install && cd ..
```

### Prepare training data

```bash
python -m training.prepare_data --max-articles 2000 --lang both
```

### Train

```bash
python -m training.train
```

Training on a single RTX 5060 Ti (16GB): ~4h for 50,000 steps.

### Run

```bash
# Terminal 1 — API
uvicorn api.server:app --reload --host 0.0.0.0

# Terminal 2 — Web interface
cd web && npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### Docker

```bash
docker-compose up --build
```

## Scripts

```bash
# Interactive CLI
python -m scripts.chat --checkpoint checkpoints/step_50000.pt

# Benchmark (tokens/sec, latency, VRAM)
python -m scripts.benchmark

# Loss curve
python -m scripts.plot_loss

# Compare two checkpoints
python -m scripts.compare --a checkpoints/step_10000.pt --b checkpoints/step_50000.pt
```

## Training results

| Step  | Loss      |
| ----- | --------- |
| 1000  | 3.43      |
| 5000  | 0.52      |
| 10000 | 0.28      |
| 20000 | 0.15      |
| 30000 | 0.09      |
| 40000 | 0.07      |
| 50000 | **0.067** |

## Benchmark (RTX 5060 Ti 16GB)

| Metric     | Value  |
| ---------- | ------ |
| Tokens/sec | ~14    |
| Latency    | ~50ms  |
| VRAM       | 172 MB |
| Parameters | 34M    |

## Roadmap

- [x] BPE tokenizer from scratch
- [x] Transformer architecture (RoPE, GQA, RMSNorm, SwiGLU)
- [x] Training loop (mixed precision, gradient accumulation, cosine decay)
- [x] FastAPI backend
- [x] React chat interface with streaming
- [x] Docker deployment
- [ ] Conversational fine-tuning
- [ ] Early stopping
- [ ] Larger training corpus (10x)
- [ ] KV cache optimization

## Philosophy

Quantum is built for learning and full technical ownership. No pretrained weights, no API wrappers — every line of code is written and understood.

## License

MIT
