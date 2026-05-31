"""
Quantum model configuration.
All hyperparameters live here — change this file to experiment.
"""

from dataclasses import dataclass


@dataclass
class QuantumConfig:
    # Vocabulary
    vocab_size: int = 8000          # Number of tokens in the vocabulary
    pad_token_id: int = 0
    bos_token_id: int = 1           # Beginning of sequence
    eos_token_id: int = 2           # End of sequence

    # Architecture
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    n_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1

    # Training
    batch_size: int = 16        # Réduit de 64 à 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 50_000

    # Generation
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    max_new_tokens: int = 256


# Default config — import this everywhere
CONFIG = QuantumConfig()