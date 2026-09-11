"""CPU-only subprocess tests for the checkpoint random-read probe."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("benchmark-checkpoint-io.py")
MIB = 1024 * 1024


def make_checkpoint(root: pathlib.Path, *, size: int = 2 * MIB) -> pathlib.Path:
    path = root / "model.safetensors"
    path.write_bytes(bytes(range(256)) * (size // 256))
    return path


def run_probe(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


class CheckpointIoCliTests(unittest.TestCase):
    def test_records_deterministic_aligned_actual_reads_without_changing_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = make_checkpoint(root)
            before = checkpoint.stat()
            first_output = root / "first.json"
            second_output = root / "second.json"
            arguments = ("--file", str(checkpoint), "--reads", "2", "--seed", "7")

            first = run_probe(*arguments, "--output", str(first_output))
            second = run_probe(*arguments, "--output", str(second_output))

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            receipt = json.loads(first_output.read_text(encoding="utf-8"))
            repeat = json.loads(second_output.read_text(encoding="utf-8"))
            after = checkpoint.stat()
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(
                receipt["workload"]["cache_mode"],
                "cached mode; OS cache state is uncontrolled",
            )
            self.assertNotIn("file", receipt["workload"])
            self.assertEqual(
                receipt["workload"]["artifact_basename"], "model.safetensors"
            )
            self.assertEqual(len(receipt["environment"]["probe_sha256"]), 64)
            self.assertEqual(
                receipt["result"]["file_metadata"]["path_before"],
                receipt["result"]["file_metadata"]["opened_descriptor"],
            )
            self.assertEqual(
                receipt["result"]["file_metadata"]["path_before"],
                receipt["result"]["file_metadata"]["path_after"],
            )
            self.assertEqual(len(receipt["result"]["samples"]), 8)
            self.assertEqual(len(receipt["result"]["curve"]), 4)
            self.assertEqual(
                [
                    (sample["read_size_bytes"], sample["offset"], sample["bytes"])
                    for sample in receipt["result"]["samples"]
                ],
                [
                    (sample["read_size_bytes"], sample["offset"], sample["bytes"])
                    for sample in repeat["result"]["samples"]
                ],
            )
            self.assertEqual(
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            )
            for sample in receipt["result"]["samples"]:
                self.assertEqual(sample["bytes"], sample["read_size_bytes"])
                self.assertEqual(sample["offset"] % sample["read_size_bytes"], 0)
                self.assertLessEqual(sample["offset"] + sample["bytes"], before.st_size)
                self.assertGreaterEqual(sample["latency_ms"], 0)
            for summary in receipt["result"]["curve"]:
                self.assertEqual(summary["sample_count"], 2)
                self.assertEqual(summary["total_bytes"], summary["read_size_bytes"] * 2)
                self.assertGreater(summary["throughput_bytes_per_second"], 0)

    def test_missing_or_small_file_writes_failed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            missing_output = root / "missing.json"
            missing = run_probe(
                "--file",
                str(root / "missing.safetensors"),
                "--output",
                str(missing_output),
            )
            self.assertEqual(missing.returncode, 1)
            missing_receipt = json.loads(missing_output.read_text(encoding="utf-8"))
            self.assertEqual(missing_receipt["status"], "failed")
            self.assertIn("FileNotFoundError", missing_receipt["error"])

            small = make_checkpoint(root, size=4 * 1024)
            small_output = root / "small.json"
            result = run_probe(
                "--file", str(small), "--output", str(small_output), "--reads", "1"
            )
            self.assertEqual(result.returncode, 1)
            receipt = json.loads(small_output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertIn("at least", receipt["error"])

    def test_timeout_writes_terminal_failed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = make_checkpoint(root)
            output = root / "timeout.json"

            result = run_probe(
                "--file",
                str(checkpoint),
                "--output",
                str(output),
                "--reads",
                "1",
                "--timeout-seconds",
                "0.0000001",
            )

            self.assertEqual(result.returncode, 1)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "failed")
            self.assertIn("TimeoutError", receipt["error"])

    def test_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            checkpoint = make_checkpoint(root)
            output = root / "existing.json"
            output.write_text("sentinel", encoding="utf-8")

            result = run_probe(
                "--file", str(checkpoint), "--output", str(output), "--reads", "1"
            )

            self.assertEqual(result.returncode, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")


if __name__ == "__main__":
    unittest.main()
