"""CPU-only contract tests for the dry-run chat qualification runner."""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

SCRIPT = pathlib.Path(__file__).with_name("qualify-chat.py")
SPEC = importlib.util.spec_from_file_location("qualify_chat", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def cli_payload(
    *, ids: list[int] | None = None, metrics: dict[str, int] | None = None
) -> str:
    return json.dumps(
        {
            "text": "answer",
            "generated_token_ids": [1, 2] if ids is None else ids,
            "finish_reason": "length",
            "metrics": {
                "generated_tokens": 2,
                "context_tokens": 2048,
                "prompt_tokens": 1983,
            }
            if metrics is None
            else metrics,
        }
    )


class QualifyChatTests(unittest.TestCase):
    def test_budget_contract_rejects_wrong_arithmetic(self) -> None:
        self.assertEqual(
            module.parse_budget_error(
                {"error": {"message": "received 1983 + 256 = 2239"}}, 256
            ),
            1983,
        )
        with self.assertRaises(ValueError):
            module.parse_budget_error({"error": {"message": "received 1 + 2 = 9"}}, 2)

    def test_cli_requires_length_exact_count_and_rss(self) -> None:
        parsed = module.parse_cli(
            cli_payload(), "123 maximum resident set size", 2, 2048, 1983
        )
        self.assertEqual(parsed.ids, (1, 2))
        with self.assertRaises(ValueError):
            module.parse_cli(cli_payload(), "", 2, 2048, 1983)
        with self.assertRaises(ValueError):
            module.parse_cli(
                cli_payload().replace('"length"', '"eos"'),
                "1 maximum resident set size",
                2,
                2048,
                1983,
            )
        with self.assertRaises(ValueError):
            module.parse_cli(
                cli_payload(ids=[1, -2]),
                "1 maximum resident set size",
                2,
                2048,
                1983,
            )
        with self.assertRaises(ValueError):
            module.parse_cli(
                cli_payload(
                    metrics={
                        "generated_tokens": 2,
                        "context_tokens": 512,
                        "prompt_tokens": 1983,
                    }
                ),
                "1 maximum resident set size",
                2,
                2048,
                1983,
            )

    def test_cli_trials_passes_requested_kv_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            args = argparse.Namespace(
                binary="target/release/mx",
                model="/models/checkpoint",
                runs=3,
                max_tokens=2,
                context_tokens=2048,
                kv_budget_mib=1024,
                timeout_seconds=120,
            )
            completed = subprocess.CompletedProcess(
                [], 0, cli_payload(), "123 maximum resident set size"
            )
            with patch.object(module, "command_run", return_value=completed) as run:
                module.cli_trials(args, output, 1983)
            for call in run.call_args_list:
                command = call.args[0]
                budget_index = command.index("--kv-budget-mib")
                self.assertEqual(command[budget_index + 1], "1024")

    def test_kv_budget_is_positive_and_recorded_in_dry_run(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.main(["--kv-budget-mib", "0"])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(module.main([]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["kv_budget_mib"], 512)
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(module.main(["--kv-budget-mib", "1024"]), 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["status"], "dry_run")
        self.assertEqual(plan["kv_budget_mib"], 1024)

    def test_http_requires_explicit_incomplete_usage_and_stable_hash(self) -> None:
        sample = {
            "terminal_status": "incomplete",
            "received_completed": False,
            "completion_tokens": 2,
            "output_text_sha256": "a" * 64,
        }
        parsed = module.parse_http(
            json.dumps({"status": "incomplete", "samples": [sample, sample, sample]}),
            2,
            3,
        )
        self.assertEqual(parsed["samples"], 3)
        sample["completion_tokens"] = None
        with self.assertRaises(ValueError):
            module.parse_http(
                json.dumps({"status": "incomplete", "samples": [sample]}), 2, 1
            )
        with self.assertRaises(TypeError):
            module.parse_http("[]", 2, 3)
        with self.assertRaises(ValueError):
            module.parse_http(json.dumps({"status": "incomplete", "samples": []}), 2, 3)

    def test_timeout_kills_owned_process_group_and_preserves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            grandchild_pid: int | None = None
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                module.command_run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import subprocess, sys, time; "
                            "child = subprocess.Popen([sys.executable, '-c', "
                            "'import time; time.sleep(30)'], "
                            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                            "print(child.pid, flush=True); time.sleep(30)"
                        ),
                    ],
                    output,
                    "child-timeout",
                    1,
                )
            try:
                grandchild_pid = int(
                    (output / "child-timeout.stdout.log").read_text().strip()
                )
                for _ in range(10):
                    status = subprocess.run(
                        ["ps", "-p", str(grandchild_pid), "-o", "stat="],
                        check=False,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    if not status or status.startswith("Z"):
                        break
                    time.sleep(0.1)
                self.assertTrue(
                    not status or status.startswith("Z"),
                    f"timed-out process group left grandchild {grandchild_pid} in {status!r}",
                )
                self.assertTrue((output / "child-timeout.stderr.log").exists())
            finally:
                if grandchild_pid is not None:
                    status = subprocess.run(
                        ["ps", "-p", str(grandchild_pid), "-o", "stat="],
                        check=False,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    if status and not status.startswith("Z"):
                        os.kill(grandchild_pid, 9)


if __name__ == "__main__":
    unittest.main()
