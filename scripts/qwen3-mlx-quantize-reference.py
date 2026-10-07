#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "mlx==0.32.2",
#   "numpy==2.5.3",
# ]
# ///
"""Fingerprint MLX's affine quantization of a BF16 Qwen3 checkpoint.

Every projection and the token embedding is quantized with ``mx.quantize``
(group size 64, ``--bits`` bits), the transform mlx-lm applies when
converting. MLX is pinned to the release mlx-sys bundles for the Rust side,
so both run the same quantizer. For
each resulting packed-weight, scale and bias tensor this prints two wrapping
64-bit sums over its raw bit patterns: a plain sum, and a sum weighted by
element position, so a reordering also changes it. A native implementation
that quantizes at load must reproduce every pair.

Published mlx-community checkpoints were converted by older MLX releases
whose rounding differs, so this is the reference rather than their files.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
MASK = (1 << 64) - 1


def quantizable(name: str, tied: bool) -> bool:
    if not name.endswith(".weight"):
        return False
    stem = name.removesuffix(".weight")
    if stem == "lm_head":
        return not tied
    return stem == "model.embed_tokens" or stem.split(".")[-1] in PROJECTIONS


def fingerprint(array: mx.array) -> dict[str, int]:
    bits = mx.uint32 if array.dtype == mx.uint32 else mx.uint16
    values = np.array(array.view(bits)).astype(np.uint64).ravel()
    positions = np.arange(1, values.size + 1, dtype=np.uint64)
    return {
        "count": int(values.size),
        "sum": int(values.sum(dtype=np.uint64)) & MASK,
        "weighted": int((values * positions).sum(dtype=np.uint64)) & MASK,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="BF16 checkpoint directory"
    )
    parser.add_argument("--bits", type=int, choices=(4, 6, 8), required=True)
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    tied = bool(config.get("tie_word_embeddings", False))
    tensors: dict[str, mx.array] = {}
    for shard in sorted(args.model.glob("*.safetensors")):
        tensors.update(mx.load(str(shard)))
    fingerprints = {}
    for name in sorted(tensors):
        if not quantizable(name, tied):
            continue
        stem = name.removesuffix(".weight")
        packed, scales, biases = mx.quantize(
            tensors[name], group_size=64, bits=args.bits
        )
        for suffix, array in (
            (".weight", packed),
            (".scales", scales),
            (".biases", biases),
        ):
            fingerprints[stem + suffix] = fingerprint(array)
    print(
        json.dumps(
            {
                "oracle": f"mlx.core.quantize, group_size 64, bits {args.bits}",
                "bits": args.bits,
                "mlx": importlib.metadata.version("mlx"),
                "fingerprint": "wrapping u64 sums of raw bit patterns: plain, and weighted by 1-based position",
                "tensors": fingerprints,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
