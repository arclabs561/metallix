#!/usr/bin/env python3
"""Capture a mathematical FP32-input sparse-attention fixture from pinned V4.1 provenance.

The pinned TileLang source is verified for provenance only. This script does
not execute it or import TileLang: it independently evaluates dense gathers,
FP64-stable softmax, and a denominator-only sink on small synthetic arrays.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import struct
from pathlib import Path

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
SOURCE_URL = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/"
    f"{REVISION}/inference/kernel.py"
)


def product(values: list[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def values_for(size: int, seed: int) -> list[float]:
    return [f32(((index * 29 + seed) % 47 - 23) / 11) for index in range(size)]


def f32(value: float) -> float:
    """Round through IEEE-754 binary32 before the independent FP64 math."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reference(case: dict[str, object]) -> list[float]:
    batches = int(case["batches"])
    queries = int(case["query_positions"])
    heads = int(case["heads"])
    dimensions = int(case["dimensions"])
    keys = int(case["key_positions"])
    slots = int(case["sparse_slots"])
    query = list(case["query"])
    kv = list(case["shared_kv"])
    sink = list(case["attn_sink"])
    indices = list(case["indices"])
    scale = float(case["scale"])
    output = [0.0] * product([batches, queries, heads, dimensions])
    for batch in range(batches):
        for query_position in range(queries):
            index_base = (batch * queries + query_position) * slots
            gathered = indices[index_base : index_base + slots]
            for head in range(heads):
                query_base = (
                    (batch * queries + query_position) * heads + head
                ) * dimensions
                scores: list[tuple[int, float]] = []
                for key_index in gathered:
                    if key_index == -1:
                        continue
                    key_base = (batch * keys + key_index) * dimensions
                    score = (
                        sum(
                            float(query[query_base + dimension])
                            * float(kv[key_base + dimension])
                            for dimension in range(dimensions)
                        )
                        * scale
                    )
                    scores.append((key_index, score))
                if not scores:
                    continue
                maximum = max(float(sink[head]), *(score for _, score in scores))
                weights = [(key, math.exp(score - maximum)) for key, score in scores]
                denominator = math.exp(float(sink[head]) - maximum) + sum(
                    weight for _, weight in weights
                )
                output_base = query_base
                for dimension in range(dimensions):
                    numerator = sum(
                        weight
                        * float(kv[(batch * keys + key) * dimensions + dimension])
                        for key, weight in weights
                    )
                    output[output_base + dimension] = numerator / denominator
    return output


def make_case(
    name: str,
    batches: int,
    queries: int,
    heads: int,
    dimensions: int,
    keys: int,
    slots: int,
    indices: list[int],
    sink: list[float],
    seed: int,
) -> dict[str, object]:
    case: dict[str, object] = {
        "name": name,
        "batches": batches,
        "query_positions": queries,
        "heads": heads,
        "dimensions": dimensions,
        "key_positions": keys,
        "sparse_slots": slots,
        "scale": f32(1 / math.sqrt(dimensions)),
        "query": values_for(batches * queries * heads * dimensions, seed),
        "shared_kv": values_for(batches * keys * dimensions, seed + 9),
        "attn_sink": [f32(value) for value in sink],
        "indices": indices,
    }
    case["expected_output"] = reference(case)
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="pinned inference/kernel.py"
    )
    args = parser.parse_args()
    source = args.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        parser.error("source SHA256 differs from the pinned official implementation")
    tree = ast.parse(source)
    sparse_kernels = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sparse_attn_kernel"
    ]
    if len(sparse_kernels) != 1:
        parser.error("expected exactly one sparse_attn_kernel function")

    cases = [
        make_case(
            "multi_batch_head_dense_gather",
            2,
            2,
            3,
            4,
            5,
            3,
            [0, 2, -1, 4, 1, 3, 3, -1, 0, 2, 4, 1],
            [-0.75, 0.0, 0.5],
            3,
        ),
        make_case(
            "duplicate_slots_are_not_deduplicated",
            1,
            1,
            2,
            3,
            3,
            4,
            [1, 1, 2, -1],
            [-0.25, 0.75],
            11,
        ),
        make_case(
            "masked_and_all_masked_rows",
            1,
            2,
            1,
            2,
            3,
            3,
            [2, -1, 0, -1, -1, -1],
            [1.25],
            19,
        ),
        make_case(
            "sink_dominates_denominator",
            1,
            1,
            2,
            2,
            2,
            2,
            [0, 1],
            [18.0, 24.0],
            27,
        ),
    ]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "url": SOURCE_URL,
                    "revision": REVISION,
                    "sha256": SOURCE_SHA256,
                    "symbol": "sparse_attn_kernel",
                    "license": "MIT",
                },
                "reference": {
                    "implementation": "independent dense gather plus stable FP64 softmax",
                    "input_dtype": "IEEE-754 float32-rounded JSON scalars",
                    "output_dtype": "float64 JSON scalars; Rust narrows to float32",
                },
                "scope": (
                    "Synthetic sparse-attention semantics only. Duplicate slots count independently; "
                    "-1 slots contribute nothing; a per-head sink contributes only to the denominator; "
                    "all-masked rows return zero. No causal masking policy, BF16 input rounding, BF16 "
                    "probability rounding, block-64 online softmax, TileLang/CUDA execution, projections, "
                    "KV cache, RoPE, model weights, or Metal execution."
                ),
                "cases": cases,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
