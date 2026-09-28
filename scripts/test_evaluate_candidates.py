#!/usr/bin/env python3
"""CPU-only contracts for the frozen candidate-control evaluator."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "evaluate_candidates", Path(__file__).with_name("evaluate-candidates.py")
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class CandidateEvaluationTests(unittest.TestCase):
    def test_confirmation_suite_is_frozen_disjoint_and_feasible(self) -> None:
        tasks = module.frozen_tasks("confirmation-v1")
        self.assertEqual(
            module.task_hash(128, 90_000, "confirmation-v1"),
            "513ecc4fcf05570ae5063e23042750e692790ad8cf9fcc9a580aea261ab27a0c",
        )
        self.assertTrue(
            set(module.suite_seeds()).isdisjoint(module.suite_seeds("confirmation-v1"))
        )
        self.assertTrue(
            {task.prompt for task in tasks}.isdisjoint(
                task.prompt for task in module.frozen_tasks()
            )
        )
        for task in tasks:
            cursor = task.window[0]
            intervals = []
            for duration in task.durations:
                intervals.append({"start": cursor, "end": cursor + duration})
                cursor += duration
            self.assertTrue(module.task_adherent({"intervals": intervals}, task))
            self.assertLessEqual(cursor, task.window[1])
            self.assertLessEqual(task.window[1], 24)
            self.assertEqual(task.count, len(task.durations))
        with self.assertRaises(ValueError):
            module.frozen_tasks("unregistered")

    def test_checkpoint_template_changes_both_arms_without_changing_tasks(self) -> None:
        manifests = []
        for flags in ([], ["--chat-template"]):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    module.main(["--binary", "mx", "--model", "model", *flags]),
                    0,
                )
            manifests.append(json.loads(stdout.getvalue()))
        plain, chat = manifests
        self.assertEqual(plain["prompt_format"], "plain")
        self.assertEqual(chat["prompt_format"], "checkpoint_chat")
        self.assertEqual(plain["task_hash"], chat["task_hash"])
        for before, after in zip(plain["plans"], chat["plans"], strict=True):
            command = after["command"].copy()
            self.assertEqual(command.count("--chat-template"), 1)
            command.remove("--chat-template")
            self.assertEqual(command, before["command"])

    def test_task_hash_and_plan_are_fixed_before_execution(self) -> None:
        args = type(
            "Args",
            (),
            {
                "binary": Path("target/release/mx"),
                "model": Path("/models/qwen"),
                "max_tokens": 128,
                "max_candidate_ms": 90_000,
                "task_aware": False,
                "output": None,
            },
        )()
        plans = module.plan(args)
        self.assertEqual(len(plans), 16)
        self.assertEqual(
            module.task_hash(128, 90_000),
            "87f4e94e312389756108e582095caa1a64c87711989052a2df680c46d77745e8",
        )
        self.assertNotEqual(
            module.task_hash(127, 90_000), module.task_hash(128, 90_000)
        )
        first_baseline, first_candidate = plans[0], plans[1]
        self.assertEqual(first_baseline.task, first_candidate.task)
        self.assertEqual(first_baseline.seed, first_candidate.seed)
        self.assertNotIn("--verify-schedule", first_baseline.command)
        self.assertIn("--verify-schedule", first_candidate.command)
        self.assertEqual(
            first_baseline.command,
            first_candidate.command[: len(first_baseline.command)],
        )
        self.assertIsNone(first_candidate.requirements)

    def test_task_aware_candidate_receives_exact_structured_requirement_file(
        self,
    ) -> None:
        args = type(
            "Args",
            (),
            {
                "binary": Path("target/release/mx"),
                "model": Path("/models/qwen"),
                "max_tokens": 128,
                "max_candidate_ms": 90_000,
                "task_aware": True,
                "output": Path("receipt"),
            },
        )()
        baseline, candidate = module.plan(args)[:2]
        self.assertIsNone(baseline.requirements)
        self.assertEqual(
            candidate.requirements,
            {"durations": [2, 2], "window": {"start": 0, "end": 8}},
        )
        self.assertNotIn("prompt", candidate.requirements)
        self.assertEqual(
            candidate.requirements_path,
            Path("receipt/requirements/candidate-two_short-seed17.json"),
        )
        requirement_flag = candidate.command.index("--schedule-requirements")
        self.assertEqual(
            candidate.command[requirement_flag + 1], str(candidate.requirements_path)
        )

    def test_task_aware_dry_run_plans_relative_paths_without_writing(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(
                module.main(
                    [
                        "--binary",
                        "mx",
                        "--model",
                        "model",
                        "--task-aware",
                    ]
                ),
                0,
            )
        manifest = json.loads(stdout.getvalue())
        candidate = next(
            item for item in manifest["plans"] if item["arm"] == "candidate"
        )
        self.assertEqual(
            candidate["schedule_requirements"]["path"],
            "requirements/candidate-two_short-seed17.json",
        )
        self.assertEqual(
            candidate["schedule_requirements"]["payload"],
            {"durations": [2, 2], "window": {"start": 0, "end": 8}},
        )

    def test_execute_requirement_files_are_owned_by_output_directory(self) -> None:
        args = type(
            "Args",
            (),
            {
                "binary": Path("target/release/mx"),
                "model": Path("/models/qwen"),
                "max_tokens": 128,
                "max_candidate_ms": 90_000,
                "task_aware": True,
                "output": None,
            },
        )()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args.output = output
            plans = module.plan(args)
            module.write_requirements(plans, output)
            files = sorted((output / "requirements").glob("*.json"))
            self.assertEqual(len(files), 8)
            first = json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(set(first), {"durations", "window"})

    def test_independent_semantic_and_task_scores_do_not_conflate(self) -> None:
        task = module.frozen_tasks()[1]
        correct = {
            "intervals": [
                {"start": 0, "end": 1},
                {"start": 2, "end": 4},
                {"start": 5, "end": 8},
            ]
        }
        semantic_only = {"intervals": [{"start": 0, "end": 1}, {"start": 2, "end": 3}]}
        overlapping = {"intervals": [{"start": 0, "end": 3}, {"start": 1, "end": 2}]}
        self.assertTrue(module.semantic_valid(correct))
        self.assertTrue(module.task_adherent(correct, task))
        self.assertTrue(module.semantic_valid(semantic_only))
        self.assertFalse(module.task_adherent(semantic_only, task))
        self.assertFalse(module.semantic_valid(overlapping))

    def test_candidate_score_keeps_work_and_elapsed_separate(self) -> None:
        item = module.RunPlan("candidate", module.frozen_tasks()[0], 17, ("mx",))
        report = {
            "constraint": {
                "generated_text": '{"intervals":[{"start":0,"end":2},{"start":2,"end":4}]}'
            },
            "candidate_verification": {
                "status": "accepted",
                "total_generated_tokens": 23,
                "costs_ms": {
                    "load": 1.0,
                    "prefill": 2.0,
                    "constraint_setup": 3.0,
                    "fork_total": 4.0,
                    "decode_total": 5.0,
                    "verifier_total": 6.0,
                    "request_total": 456.5,
                },
            },
        }
        scored = module.score(report, item, 0, 789.0)
        self.assertEqual(scored["generated_work_tokens"], 23)
        self.assertEqual(scored["receipt_elapsed_ms"], 456.5)
        self.assertEqual(scored["wall_elapsed_ms"], 789.0)
        strict_item = module.RunPlan(
            "candidate",
            module.frozen_tasks()[0],
            17,
            ("mx",),
            requirements={"durations": [2, 2], "window": {"start": 0, "end": 8}},
        )
        wrong_duration = copy.deepcopy(report)
        wrong_duration["constraint"]["generated_text"] = (
            '{"intervals":[{"start":0,"end":1},{"start":2,"end":3}]}'
        )
        # The weak verifier may accept this; the explicit contract must not.
        self.assertFalse(module.score(wrong_duration, item, 0, 1.0)["task_adherent"])
        with self.assertRaisesRegex(ValueError, "explicit task requirements"):
            module.score(wrong_duration, strict_item, 0, 1.0)
        bad = copy.deepcopy(report)
        bad["candidate_verification"]["status"] = "exhausted"
        with self.assertRaises(ValueError):
            module.score(bad, item, 0, 1.0)

    def test_error_rows_preserve_the_denominator_without_fake_costs(self) -> None:
        item = module.RunPlan("baseline", module.frozen_tasks()[0], 17, ("mx",))
        row = module.error_row(item, ValueError("malformed receipt"), 20.0)
        summary = module.aggregate([row], "baseline")
        self.assertEqual(summary["runs"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["generated_work_measured_runs"], 0)
        self.assertEqual(summary["receipt_elapsed_measured_runs"], 0)
        self.assertEqual(summary["generated_work_tokens"], 0)
        self.assertEqual(summary["receipt_elapsed_ms"], 0)


if __name__ == "__main__":
    unittest.main()
