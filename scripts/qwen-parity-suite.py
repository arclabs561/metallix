#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.13.0",
#   "transformers==5.12.1",
# ]
# ///
"""Run Qwen3 raw-token numerical parity cases; this is not a benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

CASES: tuple[tuple[str, tuple[int, ...]], ...] = (
    # config.vocab_size is 151,936, so this exercises the last embedding row.
    ("length_1_high_vocab_last_row", (151_935,)),
    # Qwen3 tokenizer IDs for "Hello, world" (without a chat template).
    ("length_3_ordinary_text", (9_707, 11, 1_879)),
    ("length_17_repeated_token", (11_504,) * 17),
    (
        "length_64_ordinary_text_cycle",
        (785, 3_974, 13_876, 38_835, 34_208, 916, 279, 15_678, 5_562, 13) * 6
        + (9_707, 11, 1_879, 0),
    ),
)


def as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def has_nonfinite_value(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(has_nonfinite_value(item) for item in value.values())
    if isinstance(value, list):
        return any(has_nonfinite_value(item) for item in value)
    return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_process(command: list[str], timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "passed": False,
            "error": f"timed out after {timeout_seconds:g}s",
            "elapsed_ms": (time.monotonic() - started) * 1_000,
            "stdout": as_text(error.stdout),
            "stderr": as_text(error.stderr),
        }
    return {
        "passed": completed.returncode == 0,
        "exit_status": completed.returncode,
        "elapsed_ms": (time.monotonic() - started) * 1_000,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def parse_json_output(result: dict[str, Any], label: str) -> dict[str, Any] | None:
    if not result["passed"]:
        return None
    try:
        payload = json.loads(result["stdout"])
    except json.JSONDecodeError as error:
        result["passed"] = False
        result["error"] = f"{label} did not emit JSON: {error.msg}"
        return None
    if not isinstance(payload, dict):
        result["passed"] = False
        result["error"] = f"{label} JSON root must be an object"
        return None
    if has_nonfinite_value(payload):
        result["passed"] = False
        result["error"] = f"{label} JSON contains a non-finite number"
        return None
    return payload


def validate_reference(
    payload: dict[str, Any], input_ids: tuple[int, ...], sidecar: Path
) -> str | None:
    if payload.get("input_ids") != list(input_ids):
        return "reference JSON input IDs do not match the requested case"
    metadata = payload.get("last_token_logits_f32le")
    if not isinstance(metadata, dict):
        return "reference JSON lacks full-logit sidecar metadata"
    if (
        metadata.get("element_count") != 151_936
        or metadata.get("byte_count") != 151_936 * 4
    ):
        return "reference sidecar metadata has an unexpected vocabulary size"
    if not sidecar.is_file():
        return "reference capture did not create full-logit sidecar"
    if metadata.get("sha256") != sha256_file(sidecar):
        return "reference sidecar SHA256 differs from its JSON metadata"
    return None


def compact_process_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        key: value for key, value in result.items() if key not in {"stdout", "stderr"}
    }
    if result.get("error") is None and not result["passed"]:
        compact["error"] = result["stderr"].strip() or "process returned failure"
    return compact


def run_case(
    name: str,
    input_ids: tuple[int, ...],
    *,
    python: str,
    reference_script: Path,
    binary: Path,
    model: Path,
    timeout_seconds: float,
    temp_dir: Path,
) -> dict[str, Any]:
    reference_path = temp_dir / f"{name}.f32"
    reference_manifest_path = temp_dir / f"{name}.json"
    raw_ids = ",".join(str(token) for token in input_ids)
    reference_result = run_process(
        [
            python,
            str(reference_script),
            "--model",
            str(model),
            "--input-ids",
            raw_ids,
            "--logits-output",
            str(reference_path),
        ],
        timeout_seconds,
    )
    reference_json = parse_json_output(reference_result, "reference capture")
    metallix_result: dict[str, Any] | None = None
    metallix_json: dict[str, Any] | None = None
    if reference_json is not None:
        reference_error = validate_reference(reference_json, input_ids, reference_path)
        if reference_error is not None:
            reference_result["passed"] = False
            reference_result["error"] = reference_error
            reference_json = None
    if reference_json is not None:
        # Preserve the reference process's JSON verbatim so Rust validates the
        # same provenance and sidecar hash that this harness inspected.
        reference_manifest_path.write_text(reference_result["stdout"], encoding="utf-8")
        metallix_result = run_process(
            [
                str(binary),
                "forward-qwen-metal",
                "--model",
                str(model),
                "--input-ids",
                raw_ids,
                "--reference",
                str(reference_path),
                "--reference-manifest",
                str(reference_manifest_path),
                # One measured repeat is sufficient for correctness; the suite
                # deliberately makes no performance comparison.
                "--repeats",
                "1",
            ],
            timeout_seconds,
        )
        metallix_json = parse_json_output(metallix_result, "Metallix forward")
    parity = metallix_json.get("parity") if metallix_json is not None else None
    passed = bool(
        reference_result["passed"]
        and metallix_result is not None
        and metallix_result["passed"]
        and parity is not None
        and parity.get("passed") is True
    )
    error = None
    if not passed:
        for result in (reference_result, metallix_result):
            if result is not None and result.get("error"):
                error = result["error"]
                break
        if error is None and parity is not None:
            error = "full-logit comparison exceeded its configured tolerance"
        if error is None:
            error = "parity did not complete"

    return {
        "name": name,
        "input_ids": list(input_ids),
        "input_length": len(input_ids),
        "passed": passed,
        "error": error,
        "reference": compact_process_result(reference_result),
        "metallix": None
        if metallix_result is None
        else compact_process_result(metallix_result),
        "parity": parity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="local Qwen3 model directory"
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/debug/metallix"),
        help="Metallix binary",
    )
    parser.add_argument(
        "--reference-script",
        type=Path,
        default=Path("scripts/qwen-reference.py"),
        help="CPU reference capture script",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for the reference script",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=300.0, help="per-process timeout"
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if not args.model.is_dir():
        parser.error(f"--model is not a directory: {args.model}")
    if not args.binary.is_file():
        parser.error(f"--binary is not a file: {args.binary}")
    if not args.reference_script.is_file():
        parser.error(f"--reference-script is not a file: {args.reference_script}")

    with tempfile.TemporaryDirectory(prefix="metallix-qwen-parity-") as temporary:
        cases = [
            run_case(
                name,
                input_ids,
                python=args.python,
                reference_script=args.reference_script,
                binary=args.binary,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                temp_dir=Path(temporary),
            )
            for name, input_ids in CASES
        ]
    passed = all(case["passed"] for case in cases)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "operation": "qwen3_raw_token_numerical_parity_suite",
                "scope": "correctness only; this suite does not generate a benchmark",
                "model": str(args.model),
                "case_count": len(cases),
                "passed": passed,
                "cases": cases,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
