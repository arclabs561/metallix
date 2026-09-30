#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.5.3", "torch==2.13.0"]
# ///
"""Replay frozen calibration embedding normalization with a balanced F32 tree.

The default replays fixed ``cal_len7``; ``--all-calibration`` applies the
predeclared no-worse/strict-improvement gate to every frozen calibration case.
This sensitivity control consumes an existing calibration report and does not
execute the pinned encoder. Keep output receipts ignored. Passing its
falsifier does not qualify a runtime change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "fixtures/julia-1/accuracy-cases.json"
WIDTH = 384
CASE_IDS = [2, 4, 6, 1, 3, 5, 7]
CASE_MASK = [True, True, True, True, True, True, True]
REPORT_MAX_BYTES = 128 * 1024 * 1024
MANIFEST_MAX_BYTES = 1024 * 1024
# The already-frozen cal_len7 native-versus-source embedding maximum.  This is
# a diagnostic falsifier, not an acceptance tolerance or a runtime policy.
SERIAL_SOURCE_MAX_ABS = 5.960464477539062e-7


def load_module(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_bounded(path: Path, limit: int, label: str) -> bytes:
    with path.open("rb") as source:
        payload = source.read(limit + 1)
    if len(payload) > limit:
        raise ValueError(f"{label} exceeds {limit} bytes")
    return payload


def validate_frozen_case(case: object) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise TypeError("frozen cal_len7 case must be an object")
    if case.get("name") != "cal_len7" or case.get("split") != "calibration":
        raise ValueError("frozen case is not calibration cal_len7")
    input_ids = case.get("input_ids")
    if (
        not isinstance(input_ids, list)
        or any(type(token_id) is not int for token_id in input_ids)
        or input_ids != CASE_IDS
    ):
        raise ValueError("calibration cal_len7 input IDs do not match the frozen case")
    attention_mask = case.get("attention_mask")
    if (
        not isinstance(attention_mask, list)
        or any(type(value) is not bool for value in attention_mask)
        or attention_mask != CASE_MASK
    ):
        raise ValueError("calibration cal_len7 mask does not match the frozen case")
    return case


def validate_frozen_input(case: object) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise TypeError("frozen calibration case must be an object")
    input_ids = case.get("input_ids")
    attention_mask = case.get("attention_mask")
    if (
        not isinstance(input_ids, list)
        or not input_ids
        or any(type(token_id) is not int for token_id in input_ids)
    ):
        raise ValueError("frozen calibration input IDs must be nonempty integers")
    if (
        not isinstance(attention_mask, list)
        or len(attention_mask) != len(input_ids)
        or any(type(value) is not bool for value in attention_mask)
    ):
        raise ValueError("frozen calibration attention mask is malformed")
    return case


def frozen_calibration_cases(expected_manifest_sha256: str) -> list[dict[str, Any]]:
    payload = read_bounded(MANIFEST, MANIFEST_MAX_BYTES, "accuracy manifest")
    if hashlib.sha256(payload).hexdigest() != expected_manifest_sha256:
        raise ValueError("accuracy manifest identity does not match calibration report")
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError("accuracy manifest is not valid JSON") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise TypeError("accuracy manifest cases must be a list")
    matches = [
        case
        for case in manifest["cases"]
        if isinstance(case, dict) and case.get("split") == "calibration"
    ]
    names = [case.get("name") for case in matches]
    if not matches or any(type(name) is not str for name in names):
        raise ValueError("accuracy manifest calibration cases have invalid names")
    if len(names) != len(set(names)):
        raise ValueError("accuracy manifest calibration case names are duplicated")
    return [validate_frozen_input(case) for case in matches]


def calibration_report_cases(
    report: dict[str, Any], expected_names: set[str]
) -> dict[str, dict[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise TypeError("calibration report cases must be objects")
    if any(case.get("split") == "held_out" for case in cases):
        raise ValueError("calibration report must not contain held-out cases")
    calibration = [case for case in cases if case.get("split") == "calibration"]
    names = [case.get("name") for case in calibration]
    if (
        any(type(name) is not str for name in names)
        or len(names) != len(expected_names)
        or set(names) != expected_names
    ):
        raise ValueError("calibration report cases do not exactly match frozen names")
    return {case["name"]: case for case in calibration}


def tensor(value: object, shape: tuple[int, ...], label: str) -> Any:
    result = torch.tensor(value, dtype=torch.float32)
    if tuple(result.shape) != shape:
        raise ValueError(f"{label} shape {tuple(result.shape)} does not equal {shape}")
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} contains a non-finite value")
    return result


def metric(left: Any, right: Any) -> dict[str, Any]:
    difference = (left.to(torch.float64) - right.to(torch.float64)).flatten()
    absolute = difference.abs()
    maximum, index = absolute.max(dim=0)
    return {
        "max_abs": maximum.item(),
        "signed_at_max_abs": difference[index].item(),
        "flat_index": index.item(),
        "rms": difference.square().mean().sqrt().item(),
        "count": difference.numel(),
    }


def pairwise_sum(values: Any) -> Any:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("pairwise reduction requires one nonempty vector")
    if values.dtype != torch.float32:
        raise TypeError("pairwise reduction requires F32 values")
    level = values
    while level.numel() > 1:
        pairs = level.numel() // 2
        reduced = level[: pairs * 2].reshape(pairs, 2).sum(dim=1, dtype=torch.float32)
        level = torch.cat((reduced, level[pairs * 2 :]))
    return level[0]


def serial_two_pass(rows: Any, weight: Any) -> Any:
    output = torch.empty_like(rows)
    width = torch.tensor(float(WIDTH), dtype=torch.float32)
    epsilon = torch.tensor(1e-5, dtype=torch.float32)
    for index, row in enumerate(rows):
        total = torch.tensor(0.0, dtype=torch.float32)
        for value in row:
            total = total + value
        mean = total / width
        total = torch.tensor(0.0, dtype=torch.float32)
        for value in row:
            delta = value - mean
            total = total + delta * delta
        output[index] = (
            (row - mean) * (total / width + epsilon).sqrt().reciprocal() * weight
        )
    return output


def balanced_two_pass(rows: Any, weight: Any) -> Any:
    output = torch.empty_like(rows)
    width = torch.tensor(float(WIDTH), dtype=torch.float32)
    epsilon = torch.tensor(1e-5, dtype=torch.float32)
    for index, row in enumerate(rows):
        mean = pairwise_sum(row) / width
        deviations = row - mean
        variance = pairwise_sum(deviations * deviations) / width
        output[index] = deviations * (variance + epsilon).sqrt().reciprocal() * weight
    return output


def write_exclusive(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(receipt, indent=2) + "\n")


def case_result(
    frozen: dict[str, Any], report_case: dict[str, Any], reference: ModuleType
) -> dict[str, Any]:
    positions = len(frozen["input_ids"])
    embedding_shape = (positions, WIDTH)
    native = report_case.get("native_f32_vs_f64")
    source_boundaries = report_case.get("source_f32_boundaries")
    if not isinstance(native, dict) or not isinstance(source_boundaries, dict):
        raise TypeError(f"{frozen['name']} report lacks native or source boundaries")
    native_boundaries = native.get("boundaries")
    source_embedding = source_boundaries.get("embedding")
    if not isinstance(native_boundaries, list) or not native_boundaries:
        raise ValueError(f"{frozen['name']} native embedding boundary is missing")
    if not isinstance(source_embedding, dict):
        raise TypeError(f"{frozen['name']} source embedding boundary is invalid")
    native_embedding = tensor(
        native_boundaries[0], embedding_shape, f"{frozen['name']} native embedding"
    )
    source = tensor(
        source_embedding.get("value"),
        embedding_shape,
        f"{frozen['name']} source embedding",
    )
    ids = torch.tensor(frozen["input_ids"], dtype=torch.int64)
    rows = reference.FULL.ENCODER.values((reference.FULL.VOCAB, WIDTH), 200)[ids]
    weight = reference.FULL.near_one(201)
    serial = serial_two_pass(rows, weight)
    balanced = balanced_two_pass(rows, weight)
    source_norm = torch.nn.functional.layer_norm(rows, (WIDTH,), weight, None, 1e-5)
    serial_native = metric(serial, native_embedding)
    source_identity = metric(source_norm, source)
    serial_source = metric(serial, source)
    if serial_native["max_abs"] != 0.0:
        raise RuntimeError(
            f"serial two-pass replay does not exactly reproduce {frozen['name']} native embedding"
        )
    if source_identity["max_abs"] != 0.0:
        raise RuntimeError(
            f"torch LayerNorm does not exactly reproduce {frozen['name']} source embedding"
        )
    balanced_source = metric(balanced, source)
    return {
        "name": frozen["name"],
        "positions": positions,
        "controls": {
            "serial_two_pass_vs_native": serial_native,
            "serial_two_pass_vs_source": serial_source,
            "torch_layer_norm_vs_source": source_identity,
        },
        "candidate": {
            "balanced_two_pass_vs_source": balanced_source,
            "balanced_two_pass_vs_native": metric(balanced, native_embedding),
            "no_worse_than_serial_source_max_abs": balanced_source["max_abs"]
            <= serial_source["max_abs"],
            "strictly_better_than_serial_source_max_abs": balanced_source["max_abs"]
            < serial_source["max_abs"],
        },
    }


def predeclared_gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the all-calibration falsifier to already-validated case results."""
    if not results:
        raise ValueError("predeclared calibration gate requires at least one case")
    controls_exact = all(
        result["controls"]["serial_two_pass_vs_native"]["max_abs"] == 0.0
        and result["controls"]["torch_layer_norm_vs_source"]["max_abs"] == 0.0
        for result in results
    )
    no_worse = all(
        result["candidate"]["no_worse_than_serial_source_max_abs"] for result in results
    )
    strict_improvement = any(
        result["candidate"]["strictly_better_than_serial_source_max_abs"]
        for result in results
    )
    return {
        "serial_native_and_torch_source_identity_exact": controls_exact,
        "balanced_no_worse_each_case": no_worse,
        "balanced_strict_improvement_at_least_one_case": strict_improvement,
        "passes": controls_exact and no_worse and strict_improvement,
        "limitation": "this gate is not full encoder acceptance and does not alter runtime arithmetic or frozen accuracy bounds",
    }


def main() -> None:
    global torch
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-calibration", action="store_true")
    args = parser.parse_args()
    report_bytes = read_bounded(
        args.calibration_report, REPORT_MAX_BYTES, "calibration report"
    )
    try:
        report = json.loads(report_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("calibration report is not valid JSON") from error
    if not isinstance(report, dict):
        raise TypeError("calibration report must be a JSON object")
    reference = load_module("julia_accuracy_reference", "julia_accuracy_reference.py")
    if report.get("manifest_sha256") != reference.MANIFEST_SHA256:
        raise ValueError(
            "calibration report manifest identity does not match frozen pin"
        )
    if report.get("weight_f32_sha256") != reference.weights_f32_sha256():
        raise ValueError(
            "calibration report weight identity does not match generated F32"
        )
    frozen_cases = frozen_calibration_cases(report["manifest_sha256"])
    report_cases = calibration_report_cases(
        report, {case["name"] for case in frozen_cases}
    )
    selected = (
        frozen_cases
        if args.all_calibration
        else [
            validate_frozen_case(
                next(case for case in frozen_cases if case["name"] == "cal_len7")
            )
        ]
    )
    results = [
        case_result(case, report_cases[case["name"]], reference) for case in selected
    ]
    if not args.all_calibration:
        result = results[0]
        serial_source = result["controls"]["serial_two_pass_vs_source"]
        if serial_source["max_abs"] != SERIAL_SOURCE_MAX_ABS:
            raise RuntimeError(
                "calibration report does not match the frozen serial source gap"
            )
        balanced_source = result["candidate"]["balanced_two_pass_vs_source"]
        receipt = {
            "schema_version": 1,
            "scope": "cal_len7 calibration balanced F32 embedding normalization sensitivity only",
            "held_out_accessed": False,
            "identities": {
                "calibration_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "manifest_sha256": report["manifest_sha256"],
                "weight_f32_sha256": report["weight_f32_sha256"],
                "diagnostic_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            "controls": {
                **result["controls"],
            },
            "candidate": {
                "topology": "fixed balanced pairwise F32 sum for mean and variance after F32 deviations",
                "balanced_two_pass_vs_source": balanced_source,
                "balanced_two_pass_vs_native": result["candidate"][
                    "balanced_two_pass_vs_native"
                ],
                "falsifier": {
                    "strictly_better_than_serial_source_max_abs": balanced_source[
                        "max_abs"
                    ]
                    < SERIAL_SOURCE_MAX_ABS,
                    "serial_source_max_abs": SERIAL_SOURCE_MAX_ABS,
                    "candidate_source_max_abs": balanced_source["max_abs"],
                    "meaning": "failure rejects this reduction tree as a runtime experiment candidate; it does not alter any acceptance bound",
                },
                "limitation": "the source CPU LayerNorm uses its own vectorized reduction; this fixed tree is a sensitivity control, not a claim of source-kernel equivalence",
            },
        }
    else:
        receipt = {
            "schema_version": 1,
            "scope": "all frozen calibration cases balanced F32 embedding normalization sensitivity only",
            "held_out_accessed": False,
            "identities": {
                "calibration_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "manifest_sha256": report["manifest_sha256"],
                "weight_f32_sha256": report["weight_f32_sha256"],
                "diagnostic_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
            "candidate": {
                "topology": "fixed balanced pairwise F32 sum for mean and variance after F32 deviations",
                "predeclared_gate": predeclared_gate(results),
            },
            "cases": results,
        }
    write_exclusive(args.output, receipt)


if __name__ == "__main__":
    main()
