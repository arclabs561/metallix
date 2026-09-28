#!/usr/bin/env python3
"""Unit tests for the independent typed-decision qualification oracle."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "qualify_decisions", Path(__file__).with_name("qualify-decisions.py")
)
qualifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualifier)


def request() -> dict:
    return {
        "state": "the supplied state",
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Choose one.",
                "criteria": {"alpha": "first", "beta": "second"},
            },
            "score": {
                "type": "score",
                "instructions": "Rate it.",
                "criteria": ["low", "high"],
            },
            "noul": {
                "type": "noul",
                "instructions": "Is it valid?",
                "criteria": {"false": "not valid", "true": "valid"},
            },
        },
    }


def option_tokens(options: list) -> list[dict]:
    return [
        {"option": option, "label": qualifier.OPTION_LABELS[index], "token_id": index}
        for index, option in enumerate(options)
    ]


def receipt() -> dict:
    return {
        "schema_version": 1,
        "operation": qualifier.OPERATION,
        "model": "local-test-model",
        "backend": "local-test-backend",
        "session_load_ms": 3.0,
        "calibration": {
            "status": qualifier.CALIBRATION_STATUS,
            "method": qualifier.CALIBRATION_METHOD,
            "temperature": 1.0,
            "note": "not calibrated",
        },
        "provenance": {"model": "local-test"},
        "usage": {"input_tokens": 12, "output_tokens": 0},
        "answers": {
            "choice": {
                "type": "choice",
                "option_order": ["alpha", "beta"],
                "option_tokens": option_tokens(["alpha", "beta"]),
                "probabilities": {"alpha": 0.2, "beta": 0.8},
                "choice": "beta",
                "prompt_tokens": 4,
                "render_ms": 1.0,
                "prefill_ms": 2.0,
            },
            "score": {
                "type": "score",
                "option_order": ["0", "1"],
                "option_tokens": option_tokens(["0", "1"]),
                "probabilities": {"0": 0.7, "1": 0.3},
                "score": 0.3,
                "prompt_tokens": 4,
                "render_ms": 1.0,
                "prefill_ms": 2.0,
            },
            "noul": {
                "type": "noul",
                "option_order": ["false", "true"],
                "option_tokens": option_tokens(["false", "true"]),
                "probabilities": {"false": 0.4, "true": 0.6},
                "noul": 0.6,
                "prompt_tokens": 4,
                "render_ms": 1.0,
                "prefill_ms": 2.0,
            },
        },
    }


class DecisionQualificationTests(unittest.TestCase):
    def test_valid_choice_score_and_explicit_boolean_noul(self):
        qualifier.check_receipt(receipt(), request())

    def test_probability_contract_rejects_bad_mapping_overflow_and_wrong_argmax(self):
        for mutate in (
            lambda value: value["answers"]["choice"].update(
                option_order=["beta", "alpha"]
            ),
            lambda value: value["answers"]["score"]["probabilities"].update({"0": 1.1}),
            lambda value: value["answers"]["choice"].update(choice="alpha"),
            lambda value: value["answers"]["score"].update(score=0.0),
            lambda value: value["answers"]["noul"].update(noul=0.5),
            lambda value: value["answers"]["choice"].update(
                option_tokens=list(
                    reversed(value["answers"]["choice"]["option_tokens"])
                )
            ),
            lambda value: value["answers"]["choice"]["option_tokens"][1].update(
                token_id=0
            ),
            lambda value: value.update(schema_version=2),
            lambda value: value.update(session_load_ms=-1.0),
            lambda value: value["calibration"].update(temperature=0.0),
            lambda value: value["usage"].update(input_tokens=11),
        ):
            with self.subTest(mutate=mutate):
                invalid = copy.deepcopy(receipt())
                mutate(invalid)
                with self.assertRaises(ValueError):
                    qualifier.check_receipt(invalid, request())

    def test_task_bridge_maps_score_index_and_noul_labels_without_fitting(self):
        score_task = {
            "id": "score-id",
            "state": {"incident": "minor"},
            "labels": ["0", "1"],
            "expected": 1,
            "question": {
                "type": "score",
                "instructions": "rate",
                "criteria": ["minor", "major"],
            },
        }
        request_value, bridge = qualifier.task_to_request(score_task)
        self.assertEqual(
            request_value["questions"]["score-id"]["criteria"], ["minor", "major"]
        )
        self.assertEqual(bridge["expected_index"], 1)
        noul_task = {
            "id": "noul-id",
            "state": "answer",
            "labels": ["no", "yes"],
            "expected": "yes",
            "question": {
                "type": "noul",
                "instructions": "valid?",
                "criteria": {"false": "bad", "true": "good"},
            },
        }
        _, noul_bridge = qualifier.task_to_request(noul_task)
        self.assertEqual(
            (noul_bridge["false_label"], noul_bridge["true_label"]), ("no", "yes")
        )

    def test_task_bridge_refuses_ambiguous_noul_and_missing_choice_criteria(self):
        ambiguous = {
            "id": "n",
            "state": "x",
            "labels": ["left", "right"],
            "expected": "left",
            "question": {
                "type": "noul",
                "instructions": "x",
                "criteria": {"false": "no", "true": "yes"},
            },
        }
        with self.assertRaises(ValueError):
            qualifier.task_to_request(ambiguous)
        missing = {
            "id": "c",
            "state": "x",
            "labels": ["a", "b"],
            "expected": "a",
            "question": {
                "type": "choice",
                "instructions": "x",
                "criteria": {"a": "only"},
            },
        }
        with self.assertRaises(ValueError):
            qualifier.task_to_request(missing)

    def test_public_bridge_allows_unmeasured_noul_default_and_refuses_private(self):
        unmeasured = {
            "id": "unmeasured",
            "split": "public",
            "state": "x",
            "labels": ["no", "yes"],
            "expected": None,
            "question": {"type": "noul", "instructions": "x"},
        }
        projected, bridge = qualifier.task_to_request(unmeasured)
        self.assertFalse(bridge["eligible"])
        self.assertEqual(
            projected["questions"]["unmeasured"]["criteria"],
            {"false": "false", "true": "true"},
        )
        private = copy.deepcopy(unmeasured)
        private["split"] = "private"
        with self.assertRaises(ValueError):
            qualifier.task_to_request(private)

    def test_bridge_matches_runtime_state_boundary(self):
        task = {
            "id": "state",
            "state": "valid",
            "labels": ["a", "b"],
            "expected": "a",
            "question": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"a": "first", "b": "second"},
            },
        }
        for scalar in (None, True, 1, 1.5):
            with self.subTest(scalar=scalar):
                invalid = copy.deepcopy(task)
                invalid["state"] = scalar
                with self.assertRaises((TypeError, ValueError)):
                    qualifier.task_to_request(invalid)

    def test_result_metrics_keep_accuracy_and_probability_separate(self):
        task = {
            "id": "score-id",
            "state": "x",
            "labels": ["0", "1"],
            "expected": 0,
            "question": {
                "type": "score",
                "instructions": "rate",
                "criteria": ["low", "high"],
            },
        }
        decision_request, bridge = qualifier.task_to_request(task)
        report = receipt()
        report["answers"] = {"score-id": report["answers"]["score"]}
        report["usage"]["input_tokens"] = 4
        decision_request["questions"] = {
            "score-id": decision_request["questions"]["score-id"]
        }
        qualifier.check_receipt(report, decision_request)
        outcome = qualifier.result_for_task(task, bridge, report)
        self.assertTrue(outcome["correct"])
        self.assertEqual(outcome["expected_probability"], 0.7)

    def test_replay_comparison_excludes_timing_but_preserves_decision_content(self):
        first = receipt()
        second = copy.deepcopy(first)
        second["session_load_ms"] = 999
        second["answers"]["choice"]["render_ms"] = 999
        second["answers"]["choice"]["prefill_ms"] = 999
        self.assertTrue(qualifier.same_decision(first, second))
        second["usage"]["input_tokens"] = 999
        self.assertFalse(qualifier.same_decision(first, second))
        second = copy.deepcopy(first)
        second["answers"]["choice"]["choice"] = "alpha"
        self.assertFalse(qualifier.same_decision(first, second))

    def test_summary_breaks_out_type_family_probability_and_latency(self):
        successful = {
            "id": "choice-1",
            "family": "routing",
            "type": "choice",
            "eligible": True,
            "expected": "alpha",
            "selected": "alpha",
            "correct": True,
            "expected_probability": 0.8,
            "probabilities": {"alpha": 0.8, "beta": 0.2},
        }
        zero_support = {
            "id": "choice-2",
            "family": "routing",
            "type": "choice",
            "eligible": True,
            "expected": "alpha",
            "selected": "beta",
            "correct": False,
            "expected_probability": 0.0,
            "probabilities": {"alpha": 0.0, "beta": 1.0},
        }
        summary = qualifier.summarize_qualification(
            {
                "tasks": [
                    {"id": "choice-1", "elapsed_ms": 10.0, "result": successful},
                    {"id": "choice-2", "elapsed_ms": 20.0, "result": zero_support},
                ]
            }
        )
        routing = summary["by_family"]["routing"]
        self.assertEqual(routing["total"], 2)
        self.assertEqual(routing["correct"], 1)
        self.assertEqual(routing["probability_quality"]["multiclass_brier_mean"], 1.04)
        self.assertIsNone(routing["probability_quality"]["mean_nll"])
        self.assertEqual(
            routing["probability_quality"]["zero_expected_probability_count"], 1
        )
        self.assertEqual(routing["latency_ms"]["process_elapsed_ms"]["median_ms"], 15.0)
        self.assertEqual(summary["by_type"]["choice"]["accuracy"], 0.5)

    def test_command_exposes_bounded_context_and_kv_configuration(self):
        command = qualifier._command(
            Path("bin"), Path("model"), Path("request.json"), 2048, 1024
        )
        self.assertEqual(command[command.index("--context-tokens") + 1], "2048")
        self.assertEqual(command[command.index("--kv-budget-mib") + 1], "1024")

    def test_labelled_unsupported_source_task_keeps_accuracy_denominator(self):
        unsupported = {
            "id": "unsupported",
            "family": "policy",
            "state": "x",
            "labels": ["a", "b"],
            "expected": "a",
            "provenance": {"exclude_reason": None},
            "question": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"a": "only"},
            },
        }
        record = qualifier._record_metadata(
            {"id": "unsupported"}, {"unsupported": unsupported}
        )
        record["latency"] = {
            "process_elapsed_ms": None,
            "session_load_ms": None,
            "render_ms": None,
            "prefill_ms": None,
        }
        bucket = qualifier._bucket([record])
        self.assertTrue(record["eligible"])
        self.assertEqual(bucket["accuracy"], 0.0)
        self.assertEqual(bucket["coverage"], 0.0)

    def test_verified_source_recomputes_and_invalid_receipt_cannot_use_cached_result(
        self,
    ):
        source_task = {
            "id": "choice",
            "family": "routing",
            "state": "the supplied state",
            "labels": ["alpha", "beta"],
            "expected": "beta",
            "question": {
                "type": "choice",
                "instructions": "Choose one.",
                "criteria": {"alpha": "first", "beta": "second"},
            },
        }
        projected, _ = qualifier.task_to_request(source_task)
        report = receipt()
        report["answers"] = {"choice": report["answers"]["choice"]}
        report["usage"]["input_tokens"] = 4
        entry = {
            "id": "choice",
            "request": projected,
            "report": report,
            "result": {"correct": False, "type": "choice", "eligible": True},
        }
        recomputed = qualifier._record_metadata(entry, {"choice": source_task})
        self.assertTrue(recomputed["result"]["correct"])
        self.assertEqual(
            recomputed["metrics_source"], "recomputed_from_verified_dataset"
        )
        report["usage"]["output_tokens"] = 1
        invalid = qualifier._record_metadata(entry, {"choice": source_task})
        self.assertIsNone(invalid["result"])
        self.assertIn("verification_error", invalid)

    def test_source_hash_mismatch_fails_closed(self):
        task = {
            "id": "x",
            "family": "policy",
            "state": "x",
            "labels": ["a", "b"],
            "expected": "a",
            "question": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"a": "first", "b": "second"},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tasks.jsonl"
            source.write_text(json.dumps(task) + "\n", encoding="utf-8")
            summary = qualifier.summarize_qualification(
                {
                    "source": str(source),
                    "source_sha256": "not-the-real-hash",
                    "tasks": [],
                }
            )
        self.assertEqual(
            summary["source_dataset_verification"], "sha256_mismatch_fail_closed"
        )


if __name__ == "__main__":
    unittest.main()
