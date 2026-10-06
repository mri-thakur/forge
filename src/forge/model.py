"""Decoder-only transformer: tied embeddings, RMSNorm, RoPE, SwiGLU, GQA.

NumPy is an executable oracle, including an independent training path. The
optional TorchModel uses identical weights and operations on CPU/CUDA/MPS.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from forge.autograd import Tensor, softmax


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    dim: int = 64
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    hidden: int = 176
    context: int = 512
    seed: int = 17

    def __post_init__(self):
        if (
            min(
                self.vocab_size,
                self.dim,
                self.layers,
                self.heads,
                self.kv_heads,
                self.hidden,
                self.context,
            )
            < 1
        ):
            raise ValueError("model dimensions must be positive")
        if self.dim % self.heads or self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("heads must divide dim, GQA must divide heads, head_dim must be even")

    @property
    def head_dim(self):
        return self.dim // self.heads


def initial_weights(config):
    c, rng = config, np.random.default_rng(config.seed)
    weights = {
        "embed": (rng.normal(size=(c.vocab_size, c.dim)) * 0.02).astype(np.float32),
        "norm": np.ones(c.dim, dtype=np.float32),
    }
    for i in range(c.layers):
        prefix = f"layers.{i}."
        for name in ("attn_norm", "ffn_norm"):
            weights[prefix + name] = np.ones(c.dim, dtype=np.float32)
        for name, shape in {
            "q": (c.dim, c.dim),
            "k": (c.dim, c.kv_heads * c.head_dim),
            "v": (c.dim, c.kv_heads * c.head_dim),
            "o": (c.dim, c.dim),
            "gate": (c.dim, c.hidden),
            "up": (c.dim, c.hidden),
            "down": (c.hidden, c.dim),
        }.items():
            scale = 0.02 / np.sqrt(2 * c.layers) if name in ("o", "down") else 0.02
            weights[prefix + name] = (rng.normal(size=shape) * scale).astype(np.float32)
    return weights


class NumpyModel:
    backend = "numpy"
    device = "cpu"

    def __init__(self, config=ModelConfig(), weights=None):
        self.config = config
        self.weight_origin = "random initialization" if weights is None else "supplied weights"
        self.weights = initial_weights(config) if weights is None else weights
        expected = initial_weights(config)
        if set(expected) != set(self.weights):
            raise ValueError("checkpoint has missing or unexpected weights")
        if any(self.weights[k].shape != v.shape for k, v in expected.items()):
            raise ValueError("checkpoint weight shapes do not match configuration")

    @property
    def parameter_count(self):
        return sum(w.size for w in self.weights.values())

    @staticmethod
    def norm(x, weight):
        return x * (x.power(2).mean(axis=-1, keepdims=True) + 1e-6).power(-0.5) * weight

    def rope(self, x, positions):
        d = self.config.head_dim
        angle = positions[..., None, None] * (10000 ** (-np.arange(0, d, 2) / d))
        cos, sin = np.cos(angle).astype(x.data.dtype), np.sin(angle).astype(x.data.dtype)
        # Interleaved pairs; concat is expressed through a tiny custom operation.
        even, odd = x[..., ::2], x[..., 1::2]
        return interleave(even * cos - odd * sin, even * sin + odd * cos)

    def forward(self, tokens, positions=None, pool=None, requests=None, grad=False):
        tokens = np.asarray(tokens, dtype=np.int64)
        if tokens.ndim != 2 or tokens.size == 0:
            raise ValueError("tokens must be a nonempty [batch, time] array")
        b, t = tokens.shape
        c = self.config
        positions = (
            np.broadcast_to(np.arange(t), (b, t)) if positions is None else np.asarray(positions)
        )
        if positions.shape != tokens.shape or positions.max() >= c.context:
            raise ValueError("positions exceed model context or have wrong shape")
        w = {key: Tensor(value, grad) for key, value in self.weights.items()}
        x = w["embed"][tokens]
        for layer in range(c.layers):
            p = f"layers.{layer}."
            normalized = self.norm(x, w[p + "attn_norm"])
            q = self.rope(
                (normalized @ w[p + "q"]).reshape(b, t, c.heads, c.head_dim), positions
            ).transpose(0, 2, 1, 3)
            k = self.rope(
                (normalized @ w[p + "k"]).reshape(b, t, c.kv_heads, c.head_dim), positions
            )
            v = (normalized @ w[p + "v"]).reshape(b, t, c.kv_heads, c.head_dim)
            if pool is not None:
                if grad:
                    raise ValueError("training through a serving cache is unsupported")
                pool.write(layer, requests, positions, k.data, v.data)
                key, value, lengths = pool.read(layer, requests)
                k, v = Tensor(key), Tensor(value)
                mask = (
                    np.arange(key.shape[1])[None, None, None, :] <= positions[:, None, :, None]
                ) & (
                    np.arange(key.shape[1])[None, None, None, :]
                    < np.asarray(lengths)[:, None, None, None]
                )
            else:
                mask = np.arange(t)[None, None, None, :] <= positions[:, None, :, None]
            # Repeat KV heads via indexing so backward sums the repeated heads.
            indices = np.arange(c.heads) // (c.heads // c.kv_heads)
            k = k[:, :, indices, :].transpose(0, 2, 3, 1)
            v = v[:, :, indices, :].transpose(0, 2, 1, 3)
            attention = softmax((q @ k) * (c.head_dim**-0.5), mask)
            output = (attention @ v).transpose(0, 2, 1, 3).reshape(b, t, c.dim)
            x = x + output @ w[p + "o"]
            normalized = self.norm(x, w[p + "ffn_norm"])
            gate = normalized @ w[p + "gate"]
            x = x + ((gate * gate.sigmoid()) * (normalized @ w[p + "up"])) @ w[p + "down"]
        logits = self.norm(x, w["norm"]) @ w["embed"].transpose(1, 0)
        if grad:
            return logits, w
        return logits.data

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, config=json.dumps(asdict(self.config)), **self.weights)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as checkpoint:
            config = ModelConfig(**json.loads(str(checkpoint["config"])))
            weights = {key: checkpoint[key].copy() for key in checkpoint.files if key != "config"}
        model = cls(config, weights)
        model.weight_origin = f"checkpoint:{Path(path).name}"
        return model


def interleave(a, b):
    data = np.empty(a.data.shape[:-1] + (2 * a.data.shape[-1],), dtype=a.data.dtype)
    data[..., ::2], data[..., 1::2] = a.data, b.data

    def backward(g):
        a.accumulate(g[..., ::2])
        b.accumulate(g[..., 1::2])

    return Tensor(data, a.requires_grad or b.requires_grad, (a, b), backward)


class ByteTokenizer:
    """Explicitly a byte tokenizer, not BPE; all 256 byte values are representable."""

    @staticmethod
    def encode(text):
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens):
        return bytes(tokens).decode("utf-8", errors="replace")
