"""DeepSeek-V4.1 Engram for mlx-lm, transcribed from the pinned reference source.

mlx-lm's ``deepseek_v41`` refuses checkpoints with Engram layers. This adds the
n-gram hash state, the FP8 hashed-embedding lookup and the gated residual write,
following ``inference/model.py`` and ``inference/engram.py`` at the pinned
revision. The table is read through a ``RowSource`` so the ~98 GB per-layer
tables can stay on disk; only the rows a token hashes to are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn
import numpy as np

DEAD = -1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def _next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


@dataclass(frozen=True)
class EngramLayout:
    """Prime-sized bucket ranges per (layer, n-gram size, head), as the source draws them."""

    max_ngram_size: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]
    primes: tuple[tuple[tuple[int, ...], ...], ...]
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config: dict) -> EngramLayout:
        layer_ids = tuple(config["engram_layer_ids"])
        max_ngram, heads = config["engram_max_ngram_size"], config["engram_n_heads"]
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(max_ngram - 1):
                sizes, current = [], config["engram_vocab_size"] - 1
                for _ in range(heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        return cls(
            max_ngram_size=max_ngram,
            layer_ids=layer_ids,
            num_embeddings=tuple(config["engram_num_embeddings"]),
            primes=tuple(primes),
            n_heads=heads,
            head_dim=config["engram_head_dim"],
        )


def hash_multipliers(layer_ids, max_ngram_size: int, vocab_size: int) -> np.ndarray:
    """One odd multiplier per (layer, lookback) from a per-layer RNG, bounded against overflow."""
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        values = np.random.default_rng(10007 * layer_id).integers(
            0, bound, size=(max_ngram_size,), dtype=np.int64
        )
        rows.append(values * 2 + 1)
    return np.stack(rows)


class NgramHashState:
    """Maps each position to the hash ids of the n-grams ending there, across prefill and decode.

    Hashing uses int64 on the host: products reach 2^62, beyond MLX's 32-bit-safe integer paths.
    """

    def __init__(self, layout: EngramLayout, token_map: np.ndarray, compressed_vocab: int, pad_id: int):
        self.layout = layout
        self.token_map = token_map.astype(np.int64)
        self.pad_id = int(self.token_map[pad_id])
        flat = [[p for per in layer for p in per] for layer in layout.primes]
        self.offsets = np.array([np.cumsum([0, *sizes[:-1]]) for sizes in flat], dtype=np.int64)
        self.primes = np.array(layout.primes, dtype=np.int64)
        self.multipliers = hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_vocab)
        self.history: list[int] = []

    def reset(self) -> None:
        self.history = []

    def __call__(self, input_ids: np.ndarray, start_pos: int) -> np.ndarray:
        """input_ids: [L] for one sequence. Returns [L, n_layers, n_hash_cols] int64."""
        del self.history[start_pos:]
        if len(self.history) != start_pos:
            raise ValueError("engram history does not reach start_pos")
        self.history.extend(int(v) for v in self.token_map[np.asarray(input_ids, dtype=np.int64)])
        cache = np.array(self.history, dtype=np.int64)
        positions = np.arange(start_pos, start_pos + len(input_ids))
        tokens, blocked = [], np.zeros(len(positions), dtype=bool)
        for shift in range(self.layout.max_ngram_size):
            source = cache[np.maximum(positions - shift, 0)]
            blocked = blocked | (positions < shift) | (source == DEAD)
            tokens.append(np.where(blocked, self.pad_id, source))
        tokens = np.stack(tokens, axis=-1)  # [L, max_ngram]
        products = tokens[:, None, :] * self.multipliers[None]  # [L, layers, max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = np.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling[..., None] % self.primes[None, :, i - 1])
        return np.concatenate(hashes, axis=-1) + self.offsets[None]


class RowSource(Protocol):
    """Returns FP8 E4M3 codes [n, dim] and E8M0 scale codes [n, dim/32] for table rows."""

    def rows(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def decode_rows(codes: np.ndarray, scales: np.ndarray) -> mx.array:
    """FP8 E4M3 rows times their E8M0 block scales, rounded to BF16 as the source does."""
    values = mx.from_fp8(mx.array(codes), mx.float32)
    exponent = mx.array(scales.astype(np.int32)) - 127
    scale = mx.power(mx.array(2.0, mx.float32), exponent.astype(mx.float32))
    blocks = values.reshape(*values.shape[:-1], -1, 32) * scale[..., None]
    return blocks.reshape(values.shape).astype(mx.bfloat16)


class Engram(nn.Module):
    """Writes an n-gram lookup into the hyper-connection streams, gated per stream copy."""

    def __init__(self, hidden_size: int, hc_mult: int, layout: EngramLayout, eps: float, linear: nn.Module):
        super().__init__()
        self.dim, self.hc_mult, self.eps = hidden_size, hc_mult, eps
        self.n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = linear
        self.q_weight = mx.ones((hc_mult, hidden_size))
        self.k_weight = mx.ones((hc_mult, hidden_size))
        self.source: RowSource | None = None

    def __call__(self, streams: mx.array, hash_ids: np.ndarray) -> mx.array:
        """streams: [B, L, hc, dim]; hash_ids: [L, n_hash_cols] for B == 1."""
        if self.source is None:
            raise RuntimeError("engram layer has no row source")
        codes, scales = self.source.rows(hash_ids.reshape(-1))
        rows = decode_rows(codes, scales).reshape(1, hash_ids.shape[0], -1)
        kv = self.wkv(rows)
        key = kv[..., : self.hc_mult * self.dim].astype(mx.float32)
        key = key.reshape(*key.shape[:-1], self.hc_mult, self.dim)
        value = kv[..., self.hc_mult * self.dim :].astype(mx.float32)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        h = streams.astype(mx.float32)
        rstd = mx.rsqrt(mx.mean(h * h, axis=-1) + self.eps) * mx.rsqrt(mx.mean(key * key, axis=-1) + self.eps)
        dot = mx.sum(h * weight * key, axis=-1) * rstd * self.dim**-0.5
        signed = mx.where(dot < 0, -1.0, 1.0) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6))
        gate = mx.sigmoid(signed)
        return (h + gate[..., None] * value[..., None, :]).astype(streams.dtype)
