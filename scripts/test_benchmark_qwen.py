# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""CPU-only contract tests for the Qwen benchmark result parser."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import unittest

SCRIPT = pathlib.Path(__file__).with_name("benchmark-qwen.py")
SPEC = importlib.util.spec_from_file_location("benchmark_qwen", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
benchmark_qwen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark_qwen
SPEC.loader.exec_module(benchmark_qwen)


def fixture() -> dict[object, object]:
    return {
        "schema_version": 1,
        "operation": "qwen3_greedy_cached_generation",
        "input_ids": [1, 2],
        "generated_ids": [3, 4, 5, 6],
        "finish_reason": "length",
        "decode_ms": [100.0, 2.0, 4.0],
        "load_ms": 10.0,
        "prefill_ms": 5.0,
        "backend": "test",
        "cache_comparisons": [],
        "cached_tokens": 5,
        "logical_kv_bytes": 160,
        "logical_weight_bytes": 1024,
    }


def run(
    decode_ms: tuple[float, ...],
    *,
    generated_ids: tuple[int, ...] = (3, 4, 5, 6),
    backend: str = "test",
):
    return benchmark_qwen.Run(
        generated_ids=generated_ids,
        decode_ms=decode_ms,
        load_ms=10.0,
        prefill_ms=5.0,
        backend=backend,
        cached_tokens=5,
        logical_kv_bytes=160,
        logical_weight_bytes=1024,
    )


def make_model(root: pathlib.Path) -> pathlib.Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"tiny test weights")
    return model


def make_binary(root: pathlib.Path, body: str) -> pathlib.Path:
    binary = root / "fake-metallix"
    binary.write_text(
        "#!" + sys.executable + "\n" + textwrap.dedent(body), encoding="utf-8"
    )
    binary.chmod(0o755)
    return binary


def run_benchmark(
    *args: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ | (environment or {})
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=5,
        env=env,
        check=False,
    )


class ParseRunTests(unittest.TestCase):
    def test_rejects_non_object_payload(self) -> None:
        for payload in (None, [], "not JSON object"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                benchmark_qwen.parse_run(payload, [1, 2], 4)

    def test_parses_exact_contract(self) -> None:
        parsed = benchmark_qwen.parse_run(fixture(), [1, 2], 4)

        self.assertEqual(parsed.generated_ids, (3, 4, 5, 6))
        self.assertEqual(parsed.decode_ms, (100.0, 2.0, 4.0))
        self.assertEqual(parsed.load_ms, 10.0)
        self.assertEqual(parsed.prefill_ms, 5.0)
        self.assertEqual(parsed.backend, "test")
        self.assertEqual(parsed.cached_tokens, 5)
        self.assertEqual(parsed.logical_kv_bytes, 160)
        self.assertEqual(parsed.logical_weight_bytes, 1024)

    def test_rejects_wrong_prompt(self) -> None:
        with self.assertRaises(ValueError):
            benchmark_qwen.parse_run(fixture(), [1, 9], 4)

    def test_rejects_early_eos(self) -> None:
        payload = fixture()
        payload["generated_ids"] = [3, 4]
        payload["decode_ms"] = [100.0]
        payload["finish_reason"] = "eos"
        with self.assertRaises(ValueError):
            benchmark_qwen.parse_run(payload, [1, 2], 4)

    def test_accepts_eos_only_at_requested_length(self) -> None:
        payload = fixture()
        payload["finish_reason"] = "eos"
        self.assertEqual(
            benchmark_qwen.parse_run(payload, [1, 2], 4).generated_ids, (3, 4, 5, 6)
        )

    def test_rejects_schema_operation_and_cache_comparisons_drift(self) -> None:
        for key, value in (
            ("schema_version", True),
            ("schema_version", 2),
            ("operation", "other"),
            ("cache_comparisons", [{"equal": True}]),
        ):
            with self.subTest(key=key, value=value):
                payload = fixture()
                payload[key] = value
                with self.assertRaises(ValueError):
                    benchmark_qwen.parse_run(payload, [1, 2], 4)

    def test_rejects_bad_numbers_and_truncated_decode(self) -> None:
        for value in (math.nan, math.inf, True, -1.0, 10**10_000):
            with self.subTest(decode=value):
                payload = fixture()
                payload["decode_ms"] = [100.0, value, 4.0]
                with self.assertRaises(ValueError):
                    benchmark_qwen.parse_run(payload, [1, 2], 4)

        for key, value in (("load_ms", -1.0), ("prefill_ms", True)):
            with self.subTest(field=key):
                payload = fixture()
                payload[key] = value
                with self.assertRaises(ValueError):
                    benchmark_qwen.parse_run(payload, [1, 2], 4)

        payload = fixture()
        payload["decode_ms"] = [100.0, 2.0]
        with self.assertRaises(ValueError):
            benchmark_qwen.parse_run(payload, [1, 2], 4)

    def test_rejects_invalid_ids_cache_counts_and_empty_backend(self) -> None:
        for key, value in (
            ("input_ids", [1, True]),
            ("generated_ids", [3, 4, 5, -1]),
            ("cached_tokens", 4),
            ("logical_kv_bytes", 0),
            ("logical_weight_bytes", True),
            ("backend", ""),
        ):
            with self.subTest(key=key):
                payload = fixture()
                payload[key] = value
                with self.assertRaises(ValueError):
                    benchmark_qwen.parse_run(payload, [1, 2], 4)


class SummarizeRunsTests(unittest.TestCase):
    def test_discards_per_run_outlier_and_reports_sample_statistics(self) -> None:
        summary = benchmark_qwen.summarize_runs(
            [run((100.0, 2.0, 4.0)), run((100.0, 4.0, 6.0)), run((100.0, 6.0, 8.0))],
            discard_decode=1,
        )

        self.assertEqual(summary["sample_count"], 6)
        self.assertEqual(summary["median_ms"], 5.0)
        self.assertEqual(summary["mean_ms"], 5.0)
        self.assertAlmostEqual(summary["sample_stdev_ms"], math.sqrt(4.4))
        self.assertEqual(summary["per_run_median_ms"], [3.0, 5.0, 7.0])
        self.assertEqual(tuple(summary["generated_ids"]), (3, 4, 5, 6))
        self.assertEqual(summary["backend"], "test")

    def test_rejects_mismatched_runs_and_short_windows(self) -> None:
        baseline = run((1.0, 2.0, 3.0))
        with self.assertRaises(ValueError):
            benchmark_qwen.summarize_runs(
                [baseline, run((1.0, 2.0, 3.0), generated_ids=(3, 4, 5, 7)), baseline],
                0,
            )
        with self.assertRaises(ValueError):
            benchmark_qwen.summarize_runs(
                [baseline, run((1.0, 2.0, 3.0), backend="other"), baseline], 0
            )
        with self.assertRaises(ValueError):
            benchmark_qwen.summarize_runs([baseline, baseline], 0)
        with self.assertRaises(ValueError):
            benchmark_qwen.summarize_runs([baseline, baseline, baseline], -1)
        with self.assertRaises(ValueError):
            benchmark_qwen.summarize_runs(
                [run((1.0, 2.0)), run((1.0, 2.0)), run((1.0, 2.0))], 1
            )


class ReceiptCliTests(unittest.TestCase):
    def test_process_failure_writes_terminal_failed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = make_model(root)
            binary = make_binary(root, "import sys\nsys.exit(7)\n")
            output = root / "failed.json"

            result = run_benchmark(
                "--binary",
                str(binary),
                "--model",
                str(model),
                "--output",
                str(output),
                "--input-ids",
                "1,2",
                "--max-tokens",
                "4",
                "--runs",
                "3",
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["runs"], [])
            self.assertIn("generation run 1 exited 7", receipt["error"])

    def test_successful_runs_write_hashes_and_warm_decode_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = make_model(root)
            payload = json.dumps(fixture())
            binary = make_binary(root, f"import sys\nprint({payload!r})\n")
            output = root / "success.json"

            result = run_benchmark(
                "--binary",
                str(binary),
                "--model",
                str(model),
                "--output",
                str(output),
                "--input-ids",
                "1,2",
                "--max-tokens",
                "4",
                "--runs",
                "3",
                "--discard-decode",
                "1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(len(receipt["runs"]), 3)
            self.assertEqual(receipt["warm_decode"]["sample_count"], 6)
            self.assertEqual(receipt["warm_decode"]["median_ms"], 3.0)
            self.assertEqual(receipt["warm_decode"]["mean_ms"], 3.0)
            self.assertEqual(
                receipt["warm_decode"]["per_run_median_ms"], [3.0, 3.0, 3.0]
            )
            self.assertEqual(
                receipt["sha256"]["binary"],
                hashlib.sha256(binary.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                receipt["sha256"]["config"],
                hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
            )
            self.assertEqual(
                receipt["sha256"]["weights"],
                hashlib.sha256((model / "model.safetensors").read_bytes()).hexdigest(),
            )

    def test_refuses_to_overwrite_existing_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = make_model(root)
            binary = make_binary(root, "import sys\nsys.exit(7)\n")
            output = root / "existing.json"
            output.write_text("sentinel", encoding="utf-8")

            result = run_benchmark(
                "--binary",
                str(binary),
                "--model",
                str(model),
                "--output",
                str(output),
                "--input-ids",
                "1,2",
                "--max-tokens",
                "4",
                "--runs",
                "3",
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")

    def test_timeout_writes_failed_receipt_and_reaps_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = make_model(root)
            pid_path = root / "child.pid"
            binary = make_binary(
                root,
                """
                import os
                import pathlib
                import time

                pathlib.Path(os.environ["BENCHMARK_TEST_CHILD_PID"]).write_text(str(os.getpid()))
                time.sleep(30)
                """,
            )
            output = root / "timeout.json"

            result = run_benchmark(
                "--binary",
                str(binary),
                "--model",
                str(model),
                "--output",
                str(output),
                "--input-ids",
                "1,2",
                "--max-tokens",
                "4",
                "--runs",
                "3",
                "--timeout-seconds",
                "0.5",
                environment={"BENCHMARK_TEST_CHILD_PID": str(pid_path)},
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertIn("TimeoutExpired", receipt["error"])
            child_pid = int(pid_path.read_text(encoding="utf-8"))
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)


if __name__ == "__main__":
    unittest.main()
