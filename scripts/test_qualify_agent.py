"""Task-level qualification must reject a successful process with an unfinished task."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("qualify-agent.py")
SPEC = importlib.util.spec_from_file_location("qualify_agent", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class QualificationTests(unittest.TestCase):
    def test_launch_failure_is_preserved_and_fake_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "not-executable"
            binary.write_text("not a program")
            for snapshot in (root / "revision", root / "snapshots" / "revision"):
                snapshot.mkdir(parents=True)
                output = root / snapshot.parent.name
                output = output.with_name(output.name + "-output")
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "--run",
                        "--binary",
                        str(binary),
                        "--model",
                        str(snapshot),
                        "--expected-model-revision",
                        "revision",
                        "--output",
                        str(output),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                if snapshot.parent.name == "snapshots":
                    receipt = json.loads((output / "receipt.json").read_text())
                    self.assertEqual(receipt["status"], "failed")
                    self.assertEqual(receipt["runs"], [])
                else:
                    self.assertFalse(output.exists())

    def test_recorded_two_file_failure_is_not_process_success(self):
        result = module.assess(
            module.CASES[1],
            0,
            "The answer is in detail.txt. Read that file next.\n",
            "tool: read_file\n",
        )
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["process_success"])
        self.assertFalse(result["checks"]["answer_present"])
        self.assertFalse(result["checks"]["required_tools_observed"])

    def test_answer_alone_does_not_prove_tools_were_used(self):
        self.assertFalse(module.assess(module.CASES[1], 0, "ORCHID-728", "")["passed"])
        self.assertTrue(
            module.assess(
                module.CASES[1],
                0,
                "The code is ORCHID-728.",
                "tool: read_file\ntool: read_file\n",
            )["passed"]
        )

    def test_failure_or_timeout_never_passes_with_matching_partial_output(self):
        for status in (None, 1, -9):
            result = module.assess(
                module.CASES[0], status, "indigo", "tool: search_file\n"
            )
            self.assertFalse(result["passed"])

    def test_actual_child_timeout_preserves_partial_output(self):
        status, stdout, _ = module.execute(
            [
                sys.executable,
                "-c",
                "import time; print('partial', flush=True); time.sleep(30)",
            ],
            1,
        )
        self.assertIsNone(status)
        self.assertEqual(stdout.strip(), "partial")

    def test_dry_run_does_not_need_a_model_or_start_a_process(self):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--model",
                "/missing-model",
                "--expected-model-revision",
                "test",
                "--output",
                "/missing-output",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
