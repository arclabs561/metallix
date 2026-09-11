#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Contract tests for the Qwen benchmark receipt comparator."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("compare-qwen-benchmarks.py")
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def generation_run(decode_ms: list[float]) -> dict[str, object]:
    return {
        "generated_ids": [3, 4, 5, 6],
        "decode_ms": decode_ms,
        "load_ms": 10.0,
        "prefill_ms": 5.0,
        "backend": "metal",
        "cached_tokens": 5,
        "logical_kv_bytes": 160,
        "logical_weight_bytes": 1024,
    }


def receipt(*, binary: str, offset: float = 0.0) -> dict[str, object]:
    # This is the serialized schema emitted by benchmark-qwen.py, including its
    # generated Run dataclass form rather than a hand-waved comparison format.
    runs = [
        generation_run([100.0, 2.0 + offset, 4.0 + offset]),
        generation_run([100.0, 4.0 + offset, 6.0 + offset]),
        generation_run([100.0, 6.0 + offset, 8.0 + offset]),
    ]
    per_run = [3.0 + offset, 5.0 + offset, 7.0 + offset]
    return {
        "schema_version": 1,
        "status": "completed",
        "host": {"system": "Darwin", "release": "24.0", "machine": "arm64"},
        "apple_hardware": {"cpu": "Apple M3 Max", "memory_bytes": 128},
        "workload": {
            "input_ids": [1, 2],
            "max_tokens": 4,
            "runs": 3,
            "discard_decode": 1,
        },
        "scope": "fixed workload",
        "sha256": {"binary": binary, "config": HASH_B, "weights": HASH_C},
        "runs": runs,
        "warm_decode": {
            "sample_count": 6,
            "median_ms": 5.0 + offset,
            "mean_ms": 5.0 + offset,
            "sample_stdev_ms": 2.0976176963403033,
            "per_run_median_ms": per_run,
            "generated_ids": [3, 4, 5, 6],
            "backend": "metal",
        },
    }


def invoke(
    baseline: pathlib.Path, candidate: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


class ComparisonCliTests(unittest.TestCase):
    def write(
        self, directory: pathlib.Path, name: str, value: dict[str, object]
    ) -> pathlib.Path:
        path = directory / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_comparison_recomputes_stats_and_allows_changed_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            baseline = self.write(directory, "baseline.json", receipt(binary=HASH_A))
            candidate = self.write(
                directory, "candidate.json", receipt(binary="d" * 64, offset=1.0)
            )

            result = invoke(baseline, candidate)

            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual(output["status"], "comparable")
            self.assertFalse(output["sha256"]["binary_hashes_equal"])
            self.assertEqual(output["baseline"]["median_ms"], 5.0)
            self.assertEqual(output["candidate"]["median_ms"], 6.0)
            self.assertEqual(output["median"]["candidate_over_baseline"], 1.2)
            self.assertAlmostEqual(output["median"]["percent_change"], 20.0)
            self.assertEqual(output["baseline"]["per_run_median_variance_ms2"], 4.0)
            self.assertNotIn("winner", output)

    def test_refuses_mismatched_workload_failed_receipt_and_output(self) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []
        changed_workload = receipt(binary=HASH_A)
        changed_workload["workload"] = {
            **changed_workload["workload"],
            "input_ids": [9],
        }  # type: ignore[arg-type]
        changed_workload["runs"] = [
            {**run, "cached_tokens": 4}
            for run in changed_workload["runs"]  # type: ignore[index]
        ]
        cases.append(("workload", changed_workload, "differ in workload"))
        failed = receipt(binary=HASH_A)
        failed["status"] = "failed"
        cases.append(("failed", failed, "not completed"))
        output_drift = receipt(binary=HASH_A)
        output_drift["runs"] = [
            {**run, "generated_ids": [3, 4, 5, 7]}
            for run in output_drift["runs"]  # type: ignore[index]
        ]
        cases.append(("output", output_drift, "generated_ids"))
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            baseline = self.write(directory, "baseline.json", receipt(binary=HASH_A))
            for name, candidate_receipt, message in cases:
                with self.subTest(name=name):
                    candidate = self.write(directory, f"{name}.json", candidate_receipt)
                    result = invoke(baseline, candidate)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)

    def test_refuses_aggregate_that_does_not_match_recomputed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            baseline = self.write(directory, "baseline.json", receipt(binary=HASH_A))
            candidate_receipt = receipt(binary=HASH_A)
            candidate_receipt["warm_decode"] = {
                **candidate_receipt["warm_decode"],  # type: ignore[arg-type]
                "median_ms": 999.0,
            }
            candidate = self.write(directory, "candidate.json", candidate_receipt)

            result = invoke(baseline, candidate)

            self.assertEqual(result.returncode, 2)
            self.assertIn("median_ms is inconsistent", result.stderr)


if __name__ == "__main__":
    unittest.main()
