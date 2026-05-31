"""
Quantum tokenizer — Byte Pair Encoding (BPE) from scratch.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


class QuantumTokenizer:
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: list[tuple[str, str]] = []

        # Special tokens
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self._add_special_tokens()

    def _add_special_tokens(self):
        for tok in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
            self._add_token(tok)

    def _add_token(self, token: str):
        if token not in self.token_to_id:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts: list[str]):
        """Train BPE on a list of texts."""
        # Start from character-level vocabulary
        vocab = self._build_char_vocab(texts)

        # Run BPE merges until vocab_size is reached
        while len(self.token_to_id) < self.vocab_size:
            pairs = self._get_pair_counts(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self._merge_pair(vocab, best)
            self.merges.append(best)
            self._add_token("".join(best))

    def _build_char_vocab(self, texts: list[str]) -> dict[tuple, int]:
        vocab: dict[tuple, int] = defaultdict(int)
        for text in texts:
            for word in text.strip().split():
                chars = tuple(list(word) + ["</w>"])
                vocab[chars] += 1
        # Add all characters to token vocabulary
        for word in vocab:
            for ch in word:
                self._add_token(ch)
        return dict(vocab)

    def _get_pair_counts(self, vocab: dict) -> dict[tuple, int]:
        pairs: dict[tuple, int] = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return dict(pairs)

    def _merge_pair(self, vocab: dict, pair: tuple) -> dict:
        new_vocab = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word, freq in vocab.items():
            word_str = " ".join(word)
            word_str = pattern.sub("".join(pair), word_str)
            new_vocab[tuple(word_str.split())] = freq
        return new_vocab

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = self._tokenize(text)
        ids = [self.token_to_id.get(t, self.token_to_id[self.unk_token]) for t in tokens]
        if add_special_tokens:
            ids = [self.token_to_id[self.bos_token]] + ids + [self.token_to_id[self.eos_token]]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = {
            self.token_to_id[self.pad_token],
            self.token_to_id[self.bos_token],
            self.token_to_id[self.eos_token],
            self.token_to_id[self.unk_token],
        }

        result = ""
        for i in ids:
            if skip_special_tokens and i in special_ids:
                continue
            tok = self.id_to_token.get(i, self.unk_token)
            if tok.endswith("</w>"):
                result += tok[:-4] + " "
            else:
                result += tok

        return result.strip()

    def _tokenize(self, text: str) -> list[str]:
        tokens = []
        for word in text.strip().split():
            word_tokens = list(word) + ["</w>"]
            # Apply merges in order
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i + 1] == merge[1]:
                        word_tokens = word_tokens[:i] + ["".join(merge)] + word_tokens[i + 2:]
                    else:
                        i += 1
            tokens.extend(word_tokens)
        return tokens

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": self.merges,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "QuantumTokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(vocab_size=data["vocab_size"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {v: k for k, v in data["token_to_id"].items()}
        tok.merges = [tuple(m) for m in data["merges"]]
        return tok

    def __len__(self) -> int:
        return len(self.token_to_id)