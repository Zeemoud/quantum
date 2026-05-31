"""Tests for the Quantum model components."""

import torch
import pytest
from model.config import QuantumConfig
from model.transformer import QuantumModel
from model.tokenizer import QuantumTokenizer
from model.norm import RMSNorm
from model.rope import RoPE, apply_rope, precompute_freqs


def test_config_defaults():
    config = QuantumConfig()
    assert config.vocab_size == 8000
    assert config.n_heads == 8
    assert config.n_layers == 8
    assert config.d_model % config.n_heads == 0


def test_config_custom():
    config = QuantumConfig(d_model=512, n_heads=16)
    assert config.d_model == 512
    assert config.n_heads == 16


def test_rmsnorm():
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    out = norm(x)
    assert out.shape == x.shape


def test_rope_shape():
    cos, sin = precompute_freqs(d_head=32, max_seq_len=512)
    assert cos.shape == (512, 16)
    assert sin.shape == (512, 16)


def test_rope_apply():
    x = torch.randn(2, 8, 16, 32)  # (B, n_heads, T, d_head)
    cos, sin = precompute_freqs(d_head=32, max_seq_len=512)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_model_forward():
    config = QuantumConfig(vocab_size=100, d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=128)
    model = QuantumModel(config)
    x = torch.randint(0, 100, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 100)


def test_model_generate():
    config = QuantumConfig(vocab_size=100, d_model=64, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=128)
    model = QuantumModel(config)
    input_ids = torch.randint(0, 100, (1, 8))
    output = model.generate(input_ids, max_new_tokens=10)
    assert output.shape[0] == 1
    assert output.shape[1] >= 8


def test_model_parameters():
    config = QuantumConfig()
    model = QuantumModel(config)
    assert model.num_parameters() > 0


def test_tokenizer_encode_decode():
    tok = QuantumTokenizer(vocab_size=200)
    texts = ["hello world", "this is a test", "quantum ai model"]
    tok.train(texts)
    ids = tok.encode("hello world")
    decoded = tok.decode(ids)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert isinstance(decoded, str)


def test_tokenizer_special_tokens():
    tok = QuantumTokenizer(vocab_size=200)
    tok.train(["hello world"])
    ids = tok.encode("hello", add_special_tokens=True)
    assert ids[0] == tok.token_to_id[tok.bos_token]
    assert ids[-1] == tok.token_to_id[tok.eos_token]


def test_tokenizer_save_load(tmp_path):
    tok = QuantumTokenizer(vocab_size=200)
    tok.train(["hello world quantum"])
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    tok2 = QuantumTokenizer.load(path)
    assert tok.token_to_id == tok2.token_to_id
    assert tok.merges == tok2.merges