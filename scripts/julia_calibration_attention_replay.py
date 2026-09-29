#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Calibration-only layer-zero softmax and value-reduction replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

POSITIONS, HEADS, HEAD_DIM, WIDTH = 7, 6, 64, 384
QKV_SHAPE = (POSITIONS, 3 * WIDTH)
SCORE_SHAPE = (HEADS, POSITIONS, POSITIONS)
ATTENDED_SHAPE = (POSITIONS, WIDTH)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor(value: object, shape: tuple[int, ...], label: str) -> torch.Tensor:
    result = torch.tensor(value, dtype=torch.float32)
    if tuple(result.shape) != shape:
        raise ValueError(f"{label} shape {tuple(result.shape)} does not equal {shape}")
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


def metric(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    difference = (
        (actual.to(torch.float64) - reference.to(torch.float64)).abs().flatten()
    )
    maximum, index = difference.max(dim=0)
    return {
        "max_abs": maximum.item(),
        "flat_index": index.item(),
        "count": difference.numel(),
    }


def validate_native(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 2:
        raise ValueError("native trace schema must be 2")
    if payload.get("case") != "cal_len7":
        raise ValueError("native trace must be the frozen cal_len7 case")
    if payload.get("attention_layout") != "head_query_key":
        raise ValueError("native attention layout must be head/query/key")


def capture(payload: dict[str, Any], label: str) -> tuple[torch.Tensor, ...]:
    qkv = tensor(payload.get("qkv"), QKV_SHAPE, f"{label} QKV")
    logits = tensor(payload.get("logits"), SCORE_SHAPE, f"{label} logits")
    probabilities = tensor(
        payload.get("probabilities"), SCORE_SHAPE, f"{label} probabilities"
    )
    attended = tensor(payload.get("attended"), ATTENDED_SHAPE, f"{label} attended")
    return qkv, logits, probabilities, attended


def replay(payload: dict[str, Any], label: str) -> dict[str, Any]:
    qkv, logits, probabilities, attended = capture(payload, label)
    values = qkv.reshape(POSITIONS, 3, HEADS, HEAD_DIM)[:, 2]
    torch_probabilities = logits.softmax(dim=-1, dtype=torch.float32)
    tensor_attended = torch.einsum("hqk,khd->qhd", probabilities, values).reshape(
        ATTENDED_SHAPE
    )
    torch_attended = torch.einsum("hqk,khd->qhd", torch_probabilities, values).reshape(
        ATTENDED_SHAPE
    )
    return {
        "probability_sum_error": (probabilities.sum(dim=-1) - 1.0).abs().max().item(),
        "probabilities_vs_torch_f32_softmax": metric(
            probabilities, torch_probabilities
        ),
        "actual_attended_vs_tensor_same_probabilities": metric(
            attended, tensor_attended
        ),
        "actual_attended_vs_tensor_torch_softmax": metric(attended, torch_attended),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native_bytes = args.native_trace.read_bytes()
    source_bytes = args.source_trace.read_bytes()
    native = json.loads(native_bytes)
    source = json.loads(source_bytes)
    if not isinstance(native, dict) or not isinstance(source, dict):
        raise TypeError("trace payloads must be JSON objects")
    validate_native(native)
    if source.get("case") != "cal_len7":
        raise ValueError("source trace must be the frozen cal_len7 case")
    source_f32 = source.get("source_f32")
    if not isinstance(source_f32, dict):
        raise TypeError("source trace lacks source_f32 capture")
    receipt = {
        "schema_version": 1,
        "scope": "cal_len7 calibration layer-zero softmax and value reduction only",
        "held_out_accessed": False,
        "identities": {
            "native_trace_sha256": hashlib.sha256(native_bytes).hexdigest(),
            "source_trace_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "diagnostic_sha256": sha256(Path(__file__)),
        },
        "native": replay(native, "native"),
        "source": replay(source_f32, "source"),
        "source_attended": "actual pinned source module capture; source logits/probabilities are explicit QKV reconstruction",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
