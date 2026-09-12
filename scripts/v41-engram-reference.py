#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""Capture a synthetic, source-pinned reference for V4.1 Engram hash state.

This deliberately extracts only ``NgramHashState.forward`` from the retained
upstream source.  The synthetic compressed token map, prime buckets, offsets,
and multipliers below are explicit test inputs: this is not a tokenizer
normalization or NumPy-RNG compatibility test.
"""

from __future__ import annotations

import ast
import hashlib
import json
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
SOURCE_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "v41-engram-pinned.py"
)
CACHE_POISON = -777


@dataclass(frozen=True)
class ExplicitTables:
    """Fixed synthetic state, intentionally independent of tokenizer/RNG code."""

    token_map: tuple[int, ...] = (7, 3, 11, 1, 9, 42, 6, 4)
    # [layer][ngram (2..4)][head], globally distinct but each layer's offsets reset at 0.
    primes: tuple[tuple[tuple[int, ...], ...], ...] = (
        ((17, 19), (23, 29), (31, 37)),
        ((41, 43), (47, 53), (59, 61)),
    )
    # The upstream constructor flattens each layer's n-gram/head bucket offsets.
    offsets: tuple[tuple[int, ...], ...] = (
        (0, 17, 36, 59, 88, 119),
        (0, 41, 84, 131, 184, 243),
    )
    multipliers: tuple[tuple[int, ...], ...] = ((3, 5, 7, 9), (11, 13, 15, 17))
    layer_ids: tuple[int, ...] = (1, 14)
    max_ngram_size: int = 4
    raw_pad_id: int = 5


def load_pinned_forward() -> Callable[..., torch.Tensor]:
    """AST-extract only the checked upstream method, retaining its decorator."""
    source = SOURCE_PATH.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"refusing source SHA {actual}; expected pinned {SOURCE_SHA256} at {SOURCE_PATH}"
        )
    module = ast.parse(source, filename=str(SOURCE_PATH))
    class_node = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == "NgramHashState"
        ),
        None,
    )
    if class_node is None:
        raise RuntimeError("pinned NgramHashState class missing")
    forward = next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "forward"
        ),
        None,
    )
    if not isinstance(forward, ast.FunctionDef):
        raise TypeError("pinned NgramHashState.forward missing or async")
    extracted = ast.Module(body=[forward], type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace: dict[str, Any] = {"torch": torch}
    exec(compile(extracted, str(SOURCE_PATH), "exec"), namespace)  # noqa: S102 -- hash-gated AST node above.
    result = namespace.get("forward")
    if not callable(result):
        raise TypeError("AST extraction did not produce callable forward")
    return result


def holder(tables: ExplicitTables, batches: int, capacity: int) -> SimpleNamespace:
    """Minimal receiver for the extracted method; no upstream constructors execute."""
    if batches <= 0 or capacity <= 0:
        raise ValueError("batches and capacity must be positive")
    if tables.raw_pad_id >= len(tables.token_map):
        raise ValueError("raw pad ID must index explicit compressed token map")
    cache = torch.full((batches, capacity), CACHE_POISON, dtype=torch.int64)
    return SimpleNamespace(
        layout=SimpleNamespace(max_ngram_size=tables.max_ngram_size),
        token_map=torch.tensor(tables.token_map, dtype=torch.int64),
        primes=torch.tensor(tables.primes, dtype=torch.int64),
        offsets=torch.tensor(tables.offsets, dtype=torch.int64),
        multipliers=torch.tensor(tables.multipliers, dtype=torch.int64),
        pad_id=tables.token_map[tables.raw_pad_id],
        DEAD=-1,
        cache=cache,
    )


def run_chunks(
    forward: Callable[..., torch.Tensor],
    tables: ExplicitTables,
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    chunks: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run consecutive chunks against a poisoned cache and return output/cache clones."""
    if input_ids.ndim != 2 or token_mask.shape != input_ids.shape:
        raise ValueError("input IDs and token mask must both be [batch, tokens]")
    if sum(chunks) != input_ids.shape[1] or any(chunk <= 0 for chunk in chunks):
        raise ValueError("chunks must partition token width into positive spans")
    state = holder(tables, input_ids.shape[0], input_ids.shape[1])
    outputs: list[torch.Tensor] = []
    start = 0
    for width in chunks:
        stop = start + width
        outputs.append(
            forward(state, input_ids[:, start:stop], start, token_mask[:, start:stop])
        )
        start = stop
    if torch.any(state.cache == CACHE_POISON):
        raise AssertionError("written sequence left poisoned cache locations")
    return torch.cat(outputs, dim=1).clone(), state.cache.clone()


def tensor_record(name: str, value: torch.Tensor) -> dict[str, Any]:
    """JSON-safe tensor value plus an exact CPU byte digest."""
    contiguous = value.detach().cpu().contiguous()
    flat = contiguous.reshape(-1).tolist()
    if contiguous.dtype == torch.int64:
        raw = struct.pack(f"<{len(flat)}q", *flat)
    elif contiguous.dtype == torch.bool:
        raw = bytes(int(item) for item in flat)
    else:
        raise TypeError(f"unsupported fixture digest dtype {contiguous.dtype}")
    return {
        "name": name,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256_le_bytes": hashlib.sha256(raw).hexdigest(),
        "values": contiguous.tolist(),
    }


def main() -> None:
    forward = load_pinned_forward()
    tables = ExplicitTables()
    # B=2 proves independent histories; false marks a DEAD/image-like span.
    input_ids = torch.tensor(
        ((0, 1, 2, 3, 4, 6), (6, 5, 4, 3, 2, 1)), dtype=torch.int64
    )
    token_mask = torch.tensor(
        ((True, True, False, True, True, True), (True, False, True, True, True, True))
    )

    one_shot, one_cache = run_chunks(forward, tables, input_ids, token_mask, (6,))
    split, split_cache = run_chunks(forward, tables, input_ids, token_mask, (3, 3))
    token_by_token, token_by_token_cache = run_chunks(
        forward, tables, input_ids, token_mask, (1, 1, 1, 1, 1, 1)
    )
    dirty_state = holder(tables, input_ids.shape[0], input_ids.shape[1])
    _ = forward(dirty_state, input_ids.flip(1), 0, token_mask)
    reset, reset_cache = (
        forward(dirty_state, input_ids, 0, token_mask).clone(),
        dirty_state.cache.clone(),
    )
    before_dead_changed = input_ids.clone()
    before_dead_changed[:, :2] = torch.tensor(((4, 6), (0, 7)), dtype=torch.int64)
    dead_boundary, _ = run_chunks(
        forward, tables, before_dead_changed, token_mask, (6,)
    )
    if not torch.equal(one_shot, split):
        raise AssertionError(
            "split prefill/decode hash output differs from one-shot output"
        )
    if not torch.equal(one_shot, token_by_token) or not torch.equal(
        one_cache, token_by_token_cache
    ):
        raise AssertionError(
            "token-by-token decode differs from one-shot output or cache"
        )
    if not torch.equal(one_shot, reset) or not torch.equal(one_cache, reset_cache):
        raise AssertionError("fresh start_pos=0 reset is not repeatable")
    for batch in range(input_ids.shape[0]):
        solo, _ = run_chunks(
            forward,
            tables,
            input_ids[batch : batch + 1],
            token_mask[batch : batch + 1],
            (6,),
        )
        if not torch.equal(one_shot[batch : batch + 1], solo):
            raise AssertionError(
                f"batch {batch} history leaked across independent cache rows"
            )
    # At position 3, a DEAD at position 2 (row 0) or 1 (row 1) blocks the
    # changed earlier history. Row 1's live position 2 must still contribute.
    if not torch.equal(one_shot[:, 3], dead_boundary[:, 3]):
        raise AssertionError("DEAD boundary did not block all earlier history")
    if tables.token_map[tables.raw_pad_id] == tables.raw_pad_id:
        raise AssertionError(
            "fixture must demonstrate compressed pad differs from raw pad ID"
        )

    payload = {
        "schema_version": 1,
        "source": {
            "revision": REVISION,
            "sha256": SOURCE_SHA256,
            "path": "artifacts/v41-engram-pinned.py",
            "url": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/dba1be0a40aa45a94ad051997016db3960a90277/inference/engram.py",
            "symbol": "NgramHashState.forward",
        },
        "scope": {
            "included": "hash-state forward with explicit synthetic tensors",
            "excluded": [
                "tokenizer normalization",
                "compressed-token-map construction",
                "NumPy multiplier generation",
                "Engram embedding rows and gated residual",
                "checkpoint loading",
                "CED and Metal execution",
            ],
        },
        "input": {
            "raw_token_ids": input_ids.tolist(),
            "token_mask": token_mask.tolist(),
            "chunkings": {
                "one_shot": [6],
                "split": [3, 3],
                "token_by_token": [1, 1, 1, 1, 1, 1],
            },
            "cache_poison": CACHE_POISON,
        },
        "explicit_state": {
            "layer_ids": list(tables.layer_ids),
            "raw_pad_id": tables.raw_pad_id,
            "compressed_pad_id": tables.token_map[tables.raw_pad_id],
        },
        "tensors": [
            tensor_record(
                "token_map", torch.tensor(tables.token_map, dtype=torch.int64)
            ),
            tensor_record("primes", torch.tensor(tables.primes, dtype=torch.int64)),
            tensor_record("offsets", torch.tensor(tables.offsets, dtype=torch.int64)),
            tensor_record(
                "multipliers", torch.tensor(tables.multipliers, dtype=torch.int64)
            ),
            tensor_record("one_shot_hashes", one_shot),
            tensor_record("split_hashes", split),
            tensor_record("token_by_token_hashes", token_by_token),
            tensor_record("dead_boundary_hashes", dead_boundary),
            tensor_record("one_shot_cache", one_cache),
            tensor_record("split_cache", split_cache),
            tensor_record("token_by_token_cache", token_by_token_cache),
            tensor_record("dirty_state_reset_hashes", reset),
            tensor_record("dirty_state_reset_cache", reset_cache),
        ],
        "assertions": [
            "one-shot output equals split prefill/decode output",
            "token-by-token decode equals one-shot output and cache",
            "start_pos=0 overwrites dirty state and repeats output and cache",
            "each B=2 row equals its independent B=1 run",
            "changing tokens before a DEAD span cannot change the next position's hashes",
            "no written cache value retains poison",
            "fixture compressed pad differs from raw pad ID",
        ],
        "receipt": {"device": "cpu", "torch_version": torch.__version__},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
