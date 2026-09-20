"""Structured agent qualification must distinguish a finished task from exit success."""

from __future__ import annotations

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


def completed_receipt(case: dict, *, final_text: str | None = None) -> dict:
    metrics = {
        "context_tokens": 32,
        "planned_kv_bytes": 1,
        "session_load_ms": 10.0,
        "render_ms": 1.0,
        "prefill_ms": 2.0,
        "time_to_first_token_ms": None,
        "decode_ms": [],
        "decode_total_ms": 0.0,
        "prompt_tokens": 4,
        "generated_tokens": 1,
    }
    text = case["answer"] if final_text is None else final_text
    digest, byte_count = module.text_identity(text)
    return {
        "schema_version": 1,
        "status": "completed",
        "final_text": text,
        "turns": [
            {
                "turn_index": 0,
                "finish_reason": "eos",
                "metrics": metrics,
                "generated_text_sha256": "0" * 64,
                "generated_text_utf8_bytes": 0,
                "calls": [
                    {**call, "outcome": "ok"} for call in module.expected_calls(case)
                ],
            },
            {
                "turn_index": 1,
                "finish_reason": "eos",
                "metrics": metrics.copy(),
                "generated_text_sha256": digest,
                "generated_text_utf8_bytes": byte_count,
                "calls": [],
            },
        ],
    }


class QualificationTests(unittest.TestCase):
    def assess_receipt(self, case: dict, receipt: dict, status: int | None = 0) -> dict:
        return module.assess(case, status, json.dumps(receipt), "agent diagnostics\n")

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
        case = module.CASES[1]
        receipt = completed_receipt(
            case,
            final_text="The answer is in detail.txt. Read that file next.",
        )
        receipt["turns"][0]["calls"] = [
            {**module.expected_calls(case)[0], "outcome": "ok"}
        ]
        result = self.assess_receipt(case, receipt)
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["process_success"])
        self.assertTrue(result["checks"]["execution_completed"])
        self.assertFalse(result["checks"]["answer_present"])
        self.assertFalse(result["checks"]["required_executions_observed"])
        self.assertIsNone(result["extra_execution_count"])

    def test_successful_process_with_unfinished_execution_evidence_fails(self):
        case = module.CASES[1]
        receipt = completed_receipt(case)
        receipt["turns"][0]["calls"] = [
            {**module.expected_calls(case)[0], "outcome": "ok"}
        ]
        result = self.assess_receipt(case, receipt)
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["process_success"])
        self.assertTrue(result["checks"]["answer_present"])
        self.assertTrue(result["checks"]["execution_completed"])
        self.assertFalse(result["checks"]["required_executions_observed"])

    def test_wrong_tool_path_or_arguments_never_proves_execution(self):
        case = module.CASES[0]
        receipt = completed_receipt(case)
        receipt["turns"][0]["calls"][0]["relative_path"] = "detail.txt"
        self.assertFalse(self.assess_receipt(case, receipt)["passed"])

        receipt = completed_receipt(case)
        receipt["turns"][0]["calls"][0]["arguments_sha256"] = "0" * 64
        result = self.assess_receipt(case, receipt)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["required_executions_observed"])

    def test_extra_successful_exploration_is_recorded_but_not_task_failure(self):
        case = module.CASES[-1]
        receipt = completed_receipt(case)
        extra_search = {
            "name": "search_file",
            "relative_path": "cedar.txt",
            "arguments_sha256": module.arguments_sha256(
                {"path": "cedar.txt", "query": "CEDAR"}
            ),
            "outcome": "ok",
        }
        receipt["turns"][0]["calls"] = [
            extra_search,
            receipt["turns"][0]["calls"][0],
            extra_search,
            receipt["turns"][0]["calls"][1],
        ]
        result = self.assess_receipt(case, receipt)
        self.assertTrue(result["passed"])
        self.assertEqual(result["extra_execution_count"], 2)

    def test_failed_extra_execution_never_passes(self):
        receipt = completed_receipt(module.CASES[0])
        receipt["turns"][0]["calls"].append(
            {
                "name": "search_file",
                "relative_path": "inventory.txt",
                "arguments_sha256": "0" * 64,
                "outcome": "error",
            }
        )
        result = self.assess_receipt(module.CASES[0], receipt)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["all_executions_succeeded"])

    def test_failed_receipt_is_distinct_from_answer_and_process_exit(self):
        case = module.CASES[0]
        receipt = completed_receipt(case)
        receipt["status"] = "failed"
        receipt["error"] = "execution_failed"
        result = self.assess_receipt(case, receipt)
        self.assertFalse(result["passed"])
        self.assertTrue(result["checks"]["process_success"])
        self.assertTrue(result["checks"]["answer_present"])
        self.assertFalse(result["checks"]["execution_completed"])

    def test_missing_optional_failed_fields_do_not_crash_assessment(self):
        receipt = {
            "schema_version": 1,
            "status": "failed",
            "turns": [],
        }
        result = self.assess_receipt(module.CASES[0], receipt)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["answer_present"])
        self.assertFalse(result["checks"]["required_executions_observed"])

    def test_malformed_completed_receipt_with_answer_and_calls_is_rejected(self):
        mutations = {
            "completed_error": lambda receipt: receipt.update(
                {"error": "execution_failed"}
            ),
            "missing_metrics": lambda receipt: receipt["turns"][0].pop("metrics"),
            "invalid_metric": lambda receipt: receipt["turns"][0]["metrics"].update(
                {"context_tokens": 0}
            ),
            "bad_count_relation": lambda receipt: receipt["turns"][0]["metrics"].update(
                {"generated_tokens": 2}
            ),
            "nonterminal_finish": lambda receipt: receipt["turns"][-1].update(
                {"finish_reason": "length"}
            ),
            "terminal_call": lambda receipt: receipt["turns"][-1]["calls"].append(
                {**module.expected_calls(module.CASES[0])[0], "outcome": "ok"}
            ),
            "earlier_length_with_call": lambda receipt: receipt["turns"][0].update(
                {"finish_reason": "length"}
            ),
            "earlier_no_call": lambda receipt: receipt["turns"][0].update(
                {"calls": []}
            ),
            "final_text_mismatch": lambda receipt: receipt.update(
                {"final_text": "other answer"}
            ),
            "final_digest_mismatch": lambda receipt: receipt["turns"][-1].update(
                {"generated_text_sha256": "0" * 64}
            ),
            "final_byte_count_mismatch": lambda receipt: receipt["turns"][-1].update(
                {"generated_text_utf8_bytes": 0}
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                receipt = json.loads(json.dumps(completed_receipt(module.CASES[0])))
                mutate(receipt)
                result = self.assess_receipt(module.CASES[0], receipt)
                self.assertFalse(result["passed"])
                self.assertFalse(result["checks"]["structured_receipt_valid"])

    def test_failed_receipt_rejects_calls_after_a_length_turn(self):
        receipt = completed_receipt(module.CASES[0])
        receipt["status"] = "failed"
        receipt["error"] = "execution_failed"
        receipt["turns"][0]["finish_reason"] = "length"
        result = self.assess_receipt(module.CASES[0], receipt)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["structured_receipt_valid"])

    def test_phase_costs_reject_nonfinite_or_boolean_metrics(self):
        receipt = completed_receipt(module.CASES[0])
        receipt["turns"][0]["metrics"]["prefill_ms"] = float("nan")
        self.assertIsNone(module.phase_costs(receipt))
        receipt = completed_receipt(module.CASES[0])
        receipt["turns"][0]["metrics"]["generated_tokens"] = True
        self.assertIsNone(module.phase_costs(receipt))

    def test_canonical_argument_hash_ignores_object_key_order(self):
        self.assertEqual(
            module.arguments_sha256({"path": "inventory.txt", "query": "beta"}),
            module.arguments_sha256({"query": "beta", "path": "inventory.txt"}),
        )

    def test_failure_or_timeout_never_passes_with_matching_receipt(self):
        receipt = completed_receipt(module.CASES[0])
        for status in (None, 1, -9):
            self.assertFalse(
                self.assess_receipt(module.CASES[0], receipt, status)["passed"]
            )

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
        plan = json.loads(result.stdout)
        self.assertEqual(plan["status"], "dry_run")
        self.assertEqual(plan["cases"][-1]["name"], "held_out_factual_join")

    def test_dry_run_records_configured_resident_limits(self):
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
                "--context-tokens",
                "4096",
                "--kv-budget-mib",
                "1024",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        plan = json.loads(result.stdout)
        self.assertEqual(plan["context_tokens"], 4096)
        self.assertEqual(plan["kv_budget_mib"], 1024)


if __name__ == "__main__":
    unittest.main()
