#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""Capture pinned Compressor.forward with explicit projection stubs on CPU.

Exercises real source pooling, partial-group state and RMSNorm. Identity wkv
and reversed/scaled wgate stand in for learned projections. No checkpoint,
RoPE, quantization or shared-cache publication is qualified here.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
SOURCE_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
SOURCE = Path(__file__).resolve().parent.parent / "artifacts/v41-reference-model.py"


def methods():
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("pinned model source hash mismatch")
    tree = ast.parse(source)
    extracted = []
    for class_name in ("Compressor", "RMSNorm"):
        cls = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "forward"
        )
        method.name = class_name.lower()
        extracted.append(method)
    namespace = {"torch": torch}
    exec(  # noqa: S102 -- only inspected methods from SHA-checked source
        compile(
            ast.fix_missing_locations(ast.Module(body=extracted, type_ignores=[])),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["compressor"], namespace["rmsnorm"]


def bits(tensor):
    return [
        int(v) & 0xFFFF
        for v in tensor.contiguous().view(torch.int16).flatten().tolist()
    ]


def main():
    if torch.__version__.split("+")[0] != "2.13.0":
        raise ValueError("fixture requires Torch 2.13.0")
    torch.set_num_threads(1)
    forward, norm_forward = methods()
    norm = SimpleNamespace(
        eps=1e-6, weight=torch.tensor([1.0, 0.75, 1.25, 0.5], dtype=torch.bfloat16)
    )

    def receiver(ratio):
        return SimpleNamespace(
            compress_ratio=ratio,
            wkv=lambda x: x.clone(),
            wgate=lambda x: x.flip(-1) * 0.75,
            norm=lambda x: norm_forward(norm, x),
            kv_state=torch.zeros((2, ratio, 4), dtype=torch.float32),
            score_state=torch.full((2, ratio, 4), -torch.inf, dtype=torch.float32),
        )

    x = torch.tensor(
        [((i * 13 + 5) % 37 - 18) / 8 for i in range(2 * 9 * 4)], dtype=torch.bfloat16
    ).reshape(2, 9, 4)
    cases = []
    for ratio in (1, 2, 3):
        expected = forward(receiver(ratio), x, 0)
        for prefix in (1, 4, 7):
            state = receiver(ratio)
            calls = []
            pieces = []
            for start, length in [(0, prefix), *[(i, 1) for i in range(prefix, 9)]]:
                chunk = x[:, start : start + length]
                output = forward(state, chunk, start)
                if output is not None:
                    pieces.append(output)
                calls.append(
                    {
                        "start": start,
                        "positions": length,
                        "input_bf16": bits(chunk),
                        "output_bf16": None if output is None else bits(output),
                        "output_shape": None if output is None else list(output.shape),
                    }
                )
            combined = torch.cat(pieces, dim=1)
            if not torch.equal(combined, expected):
                raise AssertionError(
                    "prefill plus singleton decode differs from monolithic source"
                )
            # Reuse a dirty state for a short reset; sequential writes must replace
            # every relevant slot before pooling a completed new group.
            reset = forward(state, x[:, :1], 0)
            fresh = receiver(ratio)
            fresh_reset = forward(fresh, x[:, :1], 0)
            if (reset is None) != (fresh_reset is None):
                raise AssertionError("reset output presence differs")
            for start in range(1, ratio):
                reset = forward(state, x[:, start : start + 1], start)
                fresh_reset = forward(fresh, x[:, start : start + 1], start)
            if reset is None or not torch.equal(reset, fresh_reset):
                raise AssertionError(
                    "dirty reset leaks earlier sequence into completed group"
                )
            cases.append(
                {
                    "ratio": ratio,
                    "prefix": prefix,
                    "calls": calls,
                    "full_output_bf16": bits(expected),
                }
            )
    print(
        json.dumps(
            {
                "schema_version": 1,
                "source": {
                    "revision": REVISION,
                    "sha256": SOURCE_SHA256,
                    "symbols": ["Compressor.forward", "RMSNorm.forward"],
                },
                "scope": "source pooling and normalization with identity wkv and reversed-scaled wgate stubs; not full attention",
                "receipt": {"torch": str(torch.__version__), "device": "cpu"},
                "batches": 2,
                "width": 4,
                "epsilon": 1e-6,
                "norm_weight_bf16": bits(norm.weight),
                "cases": cases,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
