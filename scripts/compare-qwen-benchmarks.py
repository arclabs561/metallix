#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Compare two completed, otherwise-identical Qwen benchmark receipts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def load_benchmark_module() -> ModuleType:
    path = Path(__file__).with_name("benchmark-qwen.py")
    specification = importlib.util.spec_from_file_location("benchmark_qwen", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


benchmark_qwen = load_benchmark_module()


def read_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read receipt: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("receipt must be a JSON object")  # noqa: TRY004
    return payload


def equal_field(baseline: dict[str, Any], candidate: dict[str, Any], field: str) -> Any:
    left = baseline.get(field)
    right = candidate.get(field)
    if left != right:
        raise ValueError(f"receipts differ in {field}")
    return left


def positive_number(value: object, name: str, *, zero_ok: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if result < 0 or (result == 0 and not zero_ok):
        raise ValueError(f"{name} must be positive")
    return result


def validate_reported_summary(
    summary: object, recomputed: dict[str, object], label: str
) -> None:
    if not isinstance(summary, dict):
        raise ValueError(f"{label}.warm_decode must be an object")  # noqa: TRY004
    expected_count = recomputed["sample_count"]
    if (
        type(summary.get("sample_count")) is not int
        or summary["sample_count"] != expected_count
    ):
        raise ValueError(f"{label}.warm_decode sample_count is inconsistent")
    for field in ("median_ms", "mean_ms"):
        actual = positive_number(summary.get(field), f"{label}.warm_decode.{field}")
        if actual != recomputed[field]:
            raise ValueError(f"{label}.warm_decode.{field} is inconsistent")
    stdev = positive_number(
        summary.get("sample_stdev_ms"),
        f"{label}.warm_decode.sample_stdev_ms",
        zero_ok=True,
    )
    if stdev != recomputed["sample_stdev_ms"]:
        raise ValueError(f"{label}.warm_decode.sample_stdev_ms is inconsistent")
    medians = summary.get("per_run_median_ms")
    if not isinstance(medians, list) or any(
        type(value) not in (int, float) or not math.isfinite(float(value)) or value <= 0
        for value in medians
    ):
        raise ValueError(f"{label}.warm_decode.per_run_median_ms is invalid")
    if medians != recomputed["per_run_median_ms"]:
        raise ValueError(f"{label}.warm_decode.per_run_median_ms is inconsistent")
    if summary.get("generated_ids") != list(recomputed["generated_ids"]):
        raise ValueError(f"{label}.warm_decode.generated_ids is inconsistent")
    if summary.get("backend") != recomputed["backend"]:
        raise ValueError(f"{label}.warm_decode.backend is inconsistent")


def validate_receipt(receipt: dict[str, Any], label: str) -> dict[str, Any]:
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise ValueError(f"{label} has unsupported schema_version")
    if receipt.get("status") != "completed":
        raise ValueError(f"{label} is not completed")
    workload = receipt.get("workload")
    if not isinstance(workload, dict):
        raise ValueError(f"{label}.workload must be an object")  # noqa: TRY004
    input_ids = workload.get("input_ids")
    if not isinstance(input_ids, list) or not input_ids:
        raise ValueError(f"{label}.workload.input_ids is invalid")
    if any(type(token) is not int or token < 0 for token in input_ids):
        raise ValueError(f"{label}.workload.input_ids is invalid")
    for field in ("max_tokens", "runs"):
        if type(workload.get(field)) is not int or workload[field] <= 0:
            raise ValueError(f"{label}.workload.{field} is invalid")
    if (
        type(workload.get("discard_decode")) is not int
        or workload["discard_decode"] < 0
    ):
        raise ValueError(f"{label}.workload.discard_decode is invalid")
    host = receipt.get("host")
    if not isinstance(host, dict) or any(
        not isinstance(host.get(field), str) or not host[field]
        for field in ("system", "release", "machine")
    ):
        raise ValueError(f"{label}.host is invalid")
    apple_hardware = receipt.get("apple_hardware")
    if (
        not isinstance(apple_hardware, dict)
        or not isinstance(apple_hardware.get("cpu"), str)
        or not apple_hardware["cpu"]
        or type(apple_hardware.get("memory_bytes")) is not int
        or apple_hardware["memory_bytes"] <= 0
    ):
        raise ValueError(f"{label}.apple_hardware is invalid")
    if not isinstance(receipt.get("scope"), str) or not receipt["scope"]:
        raise ValueError(f"{label}.scope is invalid")
    hashes = receipt.get("sha256")
    if not isinstance(hashes, dict):
        raise ValueError(f"{label}.sha256 is invalid")  # noqa: TRY004
    for field in ("binary", "config", "weights"):
        value = hashes.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label}.sha256.{field} is invalid")
    raw_runs = receipt.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != workload["runs"]:
        raise ValueError(f"{label}.runs count is inconsistent")
    if any(not isinstance(run, dict) for run in raw_runs):
        raise ValueError(f"{label}.runs must contain objects")
    try:
        # Receipts serialize ``Run`` with dataclasses.asdict(), whereas
        # parse_run consumes the generator's wire envelope. Reconstruct that
        # envelope here so validation and warm-stat calculation remain owned by
        # benchmark-qwen.py.
        runs = [
            benchmark_qwen.parse_run(
                {
                    **run,
                    "schema_version": 1,
                    "operation": "qwen3_greedy_cached_generation",
                    "input_ids": input_ids,
                    "finish_reason": "length",
                    "cache_comparisons": [],
                },
                input_ids,
                workload["max_tokens"],
            )
            for run in raw_runs
        ]
        recomputed = benchmark_qwen.summarize_runs(runs, workload["discard_decode"])
    except ValueError as error:
        raise ValueError(f"{label}.runs are invalid: {error}") from error
    validate_reported_summary(receipt.get("warm_decode"), recomputed, label)
    return {"workload": workload, "runs": runs, "warm_decode": recomputed}


def ratio_and_change(baseline: float, candidate: float) -> dict[str, float]:
    return {
        "candidate_over_baseline": candidate / baseline,
        "percent_change": (candidate / baseline - 1) * 100,
    }


def comparison(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, object]:
    left = validate_receipt(baseline, "baseline")
    right = validate_receipt(candidate, "candidate")
    for field in ("workload",):
        if left[field] != right[field]:
            raise ValueError("receipts differ in workload")
    for field in ("host", "apple_hardware", "scope"):
        equal_field(baseline, candidate, field)
    left_summary = left["warm_decode"]
    right_summary = right["warm_decode"]
    if left_summary["backend"] != right_summary["backend"]:
        raise ValueError("receipts differ in backend")
    if left_summary["generated_ids"] != right_summary["generated_ids"]:
        raise ValueError("receipts differ in generated_ids")
    if baseline["sha256"]["config"] != candidate["sha256"]["config"]:
        raise ValueError("receipts differ in sha256.config")
    if baseline["sha256"]["weights"] != candidate["sha256"]["weights"]:
        raise ValueError("receipts differ in sha256.weights")

    baseline_medians = left_summary["per_run_median_ms"]
    candidate_medians = right_summary["per_run_median_ms"]
    return {
        "schema_version": 1,
        "status": "comparable",
        "scope": "Comparable warm-decode timing evidence only; no correctness, causality, statistical-significance, or automatic winner claim.",
        "workload": left["workload"],
        "backend": left_summary["backend"],
        "generated_ids": list(left_summary["generated_ids"]),
        "sha256": {
            "config": baseline["sha256"]["config"],
            "weights": baseline["sha256"]["weights"],
            "binary_hashes_equal": baseline["sha256"]["binary"]
            == candidate["sha256"]["binary"],
            "baseline_binary": baseline["sha256"]["binary"],
            "candidate_binary": candidate["sha256"]["binary"],
        },
        "baseline": {
            "median_ms": left_summary["median_ms"],
            "per_run_median_ms": baseline_medians,
            "per_run_median_variance_ms2": statistics.variance(baseline_medians),
        },
        "candidate": {
            "median_ms": right_summary["median_ms"],
            "per_run_median_ms": candidate_medians,
            "per_run_median_variance_ms2": statistics.variance(candidate_medians),
        },
        "median": ratio_and_change(
            left_summary["median_ms"], right_summary["median_ms"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = comparison(read_receipt(args.baseline), read_receipt(args.candidate))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
