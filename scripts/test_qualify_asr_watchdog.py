"""Exercise watchdog completion and cap failure with an owned fake CPU child."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "watchdog", Path(__file__).with_name("qualify-asr-watchdog.py")
)
assert spec is not None and spec.loader is not None
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


@unittest.skipUnless(hasattr(os, "waitid"), "requires unreaped-child waitid support")
class WatchdogTests(unittest.TestCase):
    def run_child(self, body, probes, timeout=1800):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "transcribe"
            binary.write_text(f"#!{sys.executable}\n" + body)
            binary.chmod(0o700)
            output = root / "result"
            argv = [
                "watchdog",
                "--binary",
                str(binary),
                "--model",
                str(root),
                "--output",
                str(output),
                "--abort-gpu-gib",
                "1",
                "--timeout-seconds",
                str(timeout),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(watchdog, "probe", side_effect=probes),
            ):
                code = watchdog.main()
            return code, json.loads((output / "summary.json").read_text())

    def test_exit_zero_without_completion_is_failure(self):
        code, receipt = self.run_child("pass\n", lambda _: {"gpu_in_use_bytes": 0})
        self.assertEqual(code, 1)
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["exit"], 0)

    def test_gpu_cap_terminates_owned_child(self):
        code, receipt = self.run_child(
            "import time; time.sleep(30)\n",
            [
                {"gpu_in_use_bytes": 0},
                {"gpu_in_use_bytes": 2**30},
            ],
        )
        self.assertEqual(code, 1)
        self.assertEqual(receipt["reason"], "GPU memory cap reached")
        self.assertLess(receipt["exit"], 0)

    def test_completed_long_clip_passes(self):
        body = (
            "import json, sys\nfrom pathlib import Path\n"
            "Path(sys.argv[3]).write_text(json.dumps({'id': 'silence-1200', "
            "'audio_seconds': 1200, 'stopped': True}) + '\\n')\n"
        )
        code, receipt = self.run_child(body, lambda _: {"gpu_in_use_bytes": 0})
        self.assertEqual(code, 0)
        self.assertTrue(receipt["passed"])

    def test_leader_exit_cleans_up_descendants(self):
        body = "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        code, receipt = self.run_child(body, lambda _: {"gpu_in_use_bytes": 0})
        self.assertEqual(code, 1)
        self.assertEqual(receipt["reason"], "orphaned descendants")

    def test_timeout_cleans_up(self):
        code, receipt = self.run_child(
            "import time; time.sleep(30)\n",
            lambda _: {"gpu_in_use_bytes": 0},
            timeout=1,
        )
        self.assertEqual(code, 1)
        self.assertEqual(receipt["reason"], "wall timeout reached")
        self.assertLess(receipt["exit"], 0)

    def test_probe_error_cleans_up(self):
        code, receipt = self.run_child(
            "import time; time.sleep(30)\n",
            [
                {"gpu_in_use_bytes": 0},
                OSError("probe failed"),
            ],
        )
        self.assertEqual(code, 1)
        self.assertEqual(receipt["reason"], "guard failure: OSError")
        self.assertLess(receipt["exit"], 0)


if __name__ == "__main__":
    unittest.main()
