"""Contract tests for the dry-run-first native Codex qualification runner."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).with_name("qualify-codex.py")
SPEC = importlib.util.spec_from_file_location("qualify_codex", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def events(*items: dict, completed: bool = True) -> str:
    values = [json.dumps({"type": "thread.started", "thread_id": "thread"})]
    values.extend(
        json.dumps({"type": "item.completed", "item": item}) for item in items
    )
    if completed:
        values.append(json.dumps({"type": "turn.completed"}))
    return "\n".join(values)


class QualifyCodexTests(unittest.TestCase):
    value = "FACT-abc123"
    marker = "QUALIFIED:FACT-abc123"

    def successful_stdout(self) -> str:
        return events(
            {
                "id": "command",
                "type": "command_execution",
                "exit_code": 0,
                "aggregated_output": f"qualification_value={self.value}\n",
            },
            {"id": "message", "type": "agent_message", "text": self.marker},
        )

    def test_success_requires_command_evidence_and_exact_final_marker(self) -> None:
        result = module.assess(
            0, False, self.successful_stdout(), self.value, self.marker
        )
        self.assertTrue(result["passed"])

    def test_answer_without_execution_does_not_pass(self) -> None:
        stdout = events({"id": "message", "type": "agent_message", "text": self.marker})
        result = module.assess(0, False, stdout, self.value, self.marker)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["successful_command_execution"])

    def test_answer_before_execution_does_not_pass(self) -> None:
        stdout = events(
            {"id": "message", "type": "agent_message", "text": self.marker},
            {
                "id": "command",
                "type": "command_execution",
                "exit_code": 0,
                "aggregated_output": f"qualification_value={self.value}\n",
            },
        )
        result = module.assess(0, False, stdout, self.value, self.marker)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["command_precedes_final_marker"])

    def test_nonzero_and_timeout_do_not_pass(self) -> None:
        for exit_code, timed_out in ((1, False), (None, True)):
            with self.subTest(exit_code=exit_code, timed_out=timed_out):
                self.assertFalse(
                    module.assess(
                        exit_code,
                        timed_out,
                        self.successful_stdout(),
                        self.value,
                        self.marker,
                    )["passed"]
                )

    def test_evidence_must_precede_answer_for_every_command_order(self) -> None:
        items = {
            "unrelated": {
                "type": "command_execution",
                "exit_code": 0,
                "aggregated_output": "unrelated output",
            },
            "evidence": {
                "type": "command_execution",
                "exit_code": 0,
                "aggregated_output": f"qualification_value={self.value}\n",
            },
            "answer": {"type": "agent_message", "text": self.marker},
        }
        for order in itertools.permutations(items):
            with self.subTest(order=order):
                result = module.assess(
                    0,
                    False,
                    events(*(items[name] for name in order)),
                    self.value,
                    self.marker,
                )
                self.assertEqual(
                    result["passed"], order.index("evidence") < order.index("answer")
                )

    def test_malformed_json_and_unknown_completed_shape_fail(self) -> None:
        malformed = module.assess(0, False, "not json", self.value, self.marker)
        self.assertFalse(malformed["passed"])
        self.assertFalse(malformed["checks"]["jsonl_valid"])
        unknown = module.assess(
            0,
            False,
            events({"id": "unknown", "type": "other"}),
            self.value,
            self.marker,
        )
        self.assertFalse(unknown["passed"])
        self.assertFalse(unknown["checks"]["supported_event_shape"])
        unknown_event = module.assess(
            0,
            False,
            json.dumps({"type": "surprise"}),
            self.value,
            self.marker,
        )
        self.assertFalse(unknown_event["checks"]["supported_event_shape"])

    def test_bool_command_exit_code_and_unsuccessful_turn_events_fail(self) -> None:
        bool_exit = events(
            {
                "id": "command",
                "type": "command_execution",
                "exit_code": False,
                "aggregated_output": f"qualification_value={self.value}\n",
            },
            {"id": "message", "type": "agent_message", "text": self.marker},
        )
        result = module.assess(0, False, bool_exit, self.value, self.marker)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["supported_event_shape"])

        for terminal in (
            {"type": "turn.failed", "error": {"message": "failed"}},
            {"type": "error", "message": "failed"},
        ):
            with self.subTest(terminal=terminal["type"]):
                stdout = self.successful_stdout() + "\n" + json.dumps(terminal)
                result = module.assess(0, False, stdout, self.value, self.marker)
                self.assertFalse(result["passed"])
                self.assertFalse(result["checks"]["supported_event_shape"])

    def test_missing_turn_completed_does_not_pass(self) -> None:
        stdout = events(
            {
                "id": "command",
                "type": "command_execution",
                "exit_code": 0,
                "aggregated_output": f"qualification_value={self.value}\n",
            },
            {"id": "message", "type": "agent_message", "text": self.marker},
            completed=False,
        )
        result = module.assess(0, False, stdout, self.value, self.marker)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["turn_completed"])

    def test_turn_completed_before_final_message_does_not_pass(self) -> None:
        stdout = "\n".join(
            (
                json.dumps({"type": "thread.started", "thread_id": "thread"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "command",
                            "type": "command_execution",
                            "exit_code": 0,
                            "aggregated_output": f"qualification_value={self.value}\n",
                        },
                    }
                ),
                json.dumps({"type": "turn.completed"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "message",
                            "type": "agent_message",
                            "text": self.marker,
                        },
                    }
                ),
            )
        )
        result = module.assess(0, False, stdout, self.value, self.marker)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["turn_completed"])

    def test_unsafe_urls_are_rejected(self) -> None:
        for url in (
            "https://127.0.0.1:8321/v1",
            "http://example.com:8321/v1",
            "http://localhost:8321/v1",
            "http://127.0.0.1/v1",
            "http://user@127.0.0.1:8321/v1",
        ):
            with self.subTest(url=url), self.assertRaises(argparse.ArgumentTypeError):
                module.loopback_http_url(url)
        self.assertEqual(
            module.loopback_http_url("http://127.0.0.1:8321/v1/"),
            "http://127.0.0.1:8321/v1",
        )

    def test_dirty_output_is_rejected_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            (output / "prior.txt").write_text("prior")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run",
                    "--url",
                    "http://127.0.0.1:8321/v1",
                    "--model-id",
                    "metallix-qwen3",
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((output / "prior.txt").read_text(), "prior")

    def test_timeout_kills_process_group_and_retains_partial_output(self) -> None:
        status, stdout, _, timed_out, error = module.execute(
            [
                sys.executable,
                "-c",
                "import time; print('partial', flush=True); time.sleep(30)",
            ],
            1,
        )
        self.assertIsNone(status)
        self.assertTrue(timed_out)
        self.assertIsNone(error)
        self.assertEqual(stdout.strip(), "partial")

    def test_interrupt_kills_and_reaps_before_reraising(self) -> None:
        process = Mock(pid=4321)
        process.communicate.side_effect = [KeyboardInterrupt(), ("", "")]
        with (
            patch.object(module.subprocess, "Popen", return_value=process),
            patch.object(module.os, "killpg") as killpg,
            self.assertRaises(KeyboardInterrupt),
        ):
            module.execute(["ignored"], 1)
        killpg.assert_called_once_with(4321, module.signal.SIGKILL)
        self.assertEqual(process.communicate.call_count, 2)

    def test_workspace_snapshot_handles_non_utf8_and_unexpected_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "binary").write_bytes(b"\xff")
            snapshot, error = module.workspace_snapshot(workspace)
            self.assertEqual(
                snapshot, {"binary": module.hashlib.sha256(b"\xff").hexdigest()}
            )
            self.assertIsNone(error)
            (workspace / "unexpected").mkdir()
            snapshot, error = module.workspace_snapshot(workspace)
            self.assertIsNone(snapshot)
            self.assertIn("non-file", error)

    def test_workspace_snapshot_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "target"
            target.write_text("target")
            (workspace / "link").symlink_to(target)
            snapshot, error = module.workspace_snapshot(workspace)
            self.assertIsNone(snapshot)
            self.assertIn("non-file", error)

    def test_dry_run_never_needs_binary_or_output(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--url",
                "http://127.0.0.1:8321/v1",
                "--model-id",
                "metallix-qwen3",
                "--output",
                "/definitely/not/created",
                "--codex-binary",
                "/missing/codex",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout)["status"], "dry_run")

    def test_command_uses_only_ephemeral_per_process_settings(self) -> None:
        workspace = Path("/tmp/qualification-workspace")
        command = module.codex_command(
            "codex",
            "http://127.0.0.1:8321/v1",
            "metallix-qwen3",
            workspace,
            workspace / "instructions.md",
            "read facts.txt",
        )
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--ephemeral", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertIn("read-only", command)
        self.assertIn("model_context_window=16384", command)
        self.assertIn("project_doc_max_bytes=0", command)
        self.assertIn('web_search="disabled"', command)
        self.assertTrue(any('wire_api="responses"' in value for value in command))

    def test_prompt_does_not_reveal_the_fixture_value(self) -> None:
        fixture_value = "FACT-unique-hidden-value"
        prompt = module.qualification_prompt()
        self.assertIn("facts.txt", prompt)
        self.assertIn("QUALIFIED:<the qualification_value you read>", prompt)
        self.assertNotIn(fixture_value, prompt)


if __name__ == "__main__":
    unittest.main()
