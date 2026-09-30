#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Calibration-only same-input Wqkv projection attribution for cal_len7."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
POSITIONS, WIDTH = 7, 384
QKV_SHAPE = (POSITIONS, 3 * WIDTH)
EMBEDDING_SHAPE = (POSITIONS, WIDTH)
REPORT_MAX_BYTES = 128 * 1024 * 1024
SOURCE_TRACE_MAX_BYTES = 2 * 1024 * 1024


def load_module(name: str, filename: str) -> ModuleType:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_bounded(path: Path, limit: int, label: str) -> bytes:
    with path.open("rb") as source:
        payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return payload


def json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise TypeError(f"{label} must be a JSON object")
    return decoded


def tensor(value: object, shape: tuple[int, ...], label: str) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32)
    if tuple(result.shape) != shape:
        raise ValueError(f"{label} shape {tuple(result.shape)} does not equal {shape}")
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


def signed_metric(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    difference = (left.to(torch.float64) - right.to(torch.float64)).flatten()
    absolute = difference.abs()
    maximum, index = absolute.max(dim=0)
    return {
        "max_abs": maximum.item(),
        "signed_at_max_abs": difference[index].item(),
        "flat_index": index.item(),
        "rms": difference.square().mean().sqrt().item(),
        "mean_signed": difference.mean().item(),
        "count": difference.numel(),
    }


def calibration_case(report: dict[str, Any]) -> dict[str, Any]:
    cases = report.get("cases")
    if not isinstance(cases, list):
        raise TypeError("calibration report cases must be a list")
    if any(
        isinstance(case, dict) and case.get("split") == "held_out" for case in cases
    ):
        raise ValueError("calibration report must not contain held-out cases")
    matches = [
        case
        for case in cases
        if isinstance(case, dict)
        and case.get("name") == "cal_len7"
        and case.get("split") == "calibration"
    ]
    if len(matches) != 1:
        raise ValueError("calibration report requires exactly one calibration cal_len7")
    return matches[0]


def embeddings_from_report(report: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    case = calibration_case(report)
    native = case.get("native_f32_vs_f64")
    source_boundaries = case.get("source_f32_boundaries")
    if not isinstance(native, dict) or not isinstance(source_boundaries, dict):
        raise TypeError("cal_len7 report lacks native or source boundaries")
    native_boundaries = native.get("boundaries")
    source_embedding = source_boundaries.get("embedding")
    if not isinstance(native_boundaries, list) or not native_boundaries:
        raise ValueError("cal_len7 report native embedding boundary is missing")
    if not isinstance(source_embedding, dict):
        raise TypeError("cal_len7 report source embedding boundary is invalid")
    return (
        tensor(native_boundaries[0], EMBEDDING_SHAPE, "native embedding"),
        tensor(source_embedding.get("value"), EMBEDDING_SHAPE, "source embedding"),
    )


def source_qkv_from_trace(payload: dict[str, Any]) -> torch.Tensor:
    if payload.get("case") != "cal_len7":
        raise ValueError("source trace must be the calibration cal_len7 case")
    source_f32 = payload.get("source_f32")
    if not isinstance(source_f32, dict):
        raise TypeError("source trace lacks source_f32 tensors")
    return tensor(source_f32.get("qkv"), QKV_SHAPE, "source QKV")


def write_exclusive(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    global torch
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--native-trace", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_bytes = read_bounded(
        args.calibration_report, REPORT_MAX_BYTES, "calibration report"
    )
    source_trace_bytes = read_bounded(
        args.source_trace, SOURCE_TRACE_MAX_BYTES, "source trace"
    )
    report = json_object(report_bytes, "calibration report")
    source_trace = json_object(source_trace_bytes, "source trace")
    native_oracle = load_module(
        "julia_calibration_source_oracle", "julia_calibration_source_oracle.py"
    )
    native_trace, native_trace_sha256 = native_oracle.load_native_trace(
        torch, args.native_trace
    )
    native_qkv = tensor(native_trace.get("qkv"), QKV_SHAPE, "native QKV")
    source_qkv = source_qkv_from_trace(source_trace)
    native_embedding, source_embedding = embeddings_from_report(report)
    reference = load_module("julia_accuracy_reference", "julia_accuracy_reference.py")
    if report.get("manifest_sha256") != reference.MANIFEST_SHA256:
        raise ValueError(
            "calibration report manifest identity does not match frozen pin"
        )
    if report.get("weight_f32_sha256") != reference.weights_f32_sha256():
        raise ValueError(
            "calibration report weight identity does not match generated F32"
        )
    weight = reference.FULL.layer_weights(0)["wqkv"]
    if tuple(weight.shape) != (3 * WIDTH, WIDTH) or weight.dtype != torch.float32:
        raise ValueError("generated layer-zero Wqkv weight shape or dtype is invalid")

    source_on_native = torch.nn.functional.linear(native_embedding, weight)
    source_on_source = torch.nn.functional.linear(source_embedding, weight)
    source_identity = signed_metric(source_on_source, source_qkv)
    if source_identity["max_abs"] != 0.0:
        raise RuntimeError(
            "F.linear(source embedding) does not exactly reproduce captured source QKV"
        )
    projection_residual = native_qkv.to(torch.float64) - source_on_native.to(
        torch.float64
    )
    embedding_input_component = source_on_native.to(
        torch.float64
    ) - source_on_source.to(torch.float64)
    identity_component = source_on_source.to(torch.float64) - source_qkv.to(
        torch.float64
    )
    overall = native_qkv.to(torch.float64) - source_qkv.to(torch.float64)
    closure = overall - (
        projection_residual + embedding_input_component + identity_component
    )
    closure_max_abs = closure.abs().max().item()
    if closure_max_abs != 0.0:
        raise RuntimeError("projection attribution components do not close exactly")
    weight_sha256 = sha256(weight.contiguous().numpy().tobytes())
    receipt = {
        "schema_version": 1,
        "scope": "cal_len7 calibration same-input F32 Wqkv projection attribution only",
        "held_out_accessed": False,
        "identities": {
            "calibration_report_sha256": sha256(report_bytes),
            "native_trace_sha256": native_trace_sha256,
            "source_trace_sha256": sha256(source_trace_bytes),
            "manifest_sha256": report["manifest_sha256"],
            "weight_f32_sha256": report["weight_f32_sha256"],
            "layer0_wqkv_sha256": weight_sha256,
            "diagnostic_sha256": sha256(Path(__file__).read_bytes()),
        },
        "shapes": {"embedding": list(EMBEDDING_SHAPE), "qkv": list(QKV_SHAPE)},
        "definitions": {
            "projection_residual": "native captured QKV minus F.linear(native embedding, pinned Wqkv)",
            "embedding_input_component": "F.linear(native embedding, pinned Wqkv) minus F.linear(source embedding, pinned Wqkv)",
            "source_identity_component": "F.linear(source embedding, pinned Wqkv) minus captured source QKV",
            "overall": "native captured QKV minus captured source QKV",
            "maxima_and_norms": "component-wise signed differences add exactly; max-absolute and RMS summaries do not add",
            "interpretation": "conditional decomposition: input propagation includes every embedding-path difference; projection residual is not proof of a GEMM cause; algebraic closure checks bookkeeping only",
        },
        "source_linear_identity": source_identity,
        "components": {
            "projection_residual": signed_metric(native_qkv, source_on_native),
            "embedding_input_propagation": signed_metric(
                source_on_native, source_on_source
            ),
            "source_identity": source_identity,
            "overall": signed_metric(native_qkv, source_qkv),
            "additive_closure_max_abs": closure_max_abs,
        },
    }
    write_exclusive(args.output, receipt)


if __name__ == "__main__":
    main()
