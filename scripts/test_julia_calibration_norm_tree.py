#!/usr/bin/env python3
"""Dependency-free input boundaries for the Julia normalization tree diagnostic."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "julia_calibration_norm_tree", ROOT / "scripts/julia_calibration_norm_tree.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Julia normalization-tree diagnostic")
TREE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TREE)


class NormalizationTreeInputTest(unittest.TestCase):
    def test_fixed_cal_len7_input_rejects_nonfrozen_values(self) -> None:
        case = {"name": "cal_len7", "split": "calibration"}
        frozen = {
            **case,
            "input_ids": TREE.CASE_IDS,
            "attention_mask": TREE.CASE_MASK,
        }
        self.assertEqual(TREE.validate_frozen_case(frozen)["input_ids"], TREE.CASE_IDS)
        with self.assertRaises(ValueError):
            TREE.validate_frozen_case({**frozen, "input_ids": [*TREE.CASE_IDS[:-1], 0]})
        with self.assertRaises(ValueError):
            TREE.validate_frozen_case(
                {**frozen, "input_ids": [True, *TREE.CASE_IDS[1:]]}
            )
        with self.assertRaises(ValueError):
            TREE.validate_frozen_case(
                {**frozen, "input_ids": [2.0, *TREE.CASE_IDS[1:]]}
            )
        with self.assertRaises(ValueError):
            TREE.validate_frozen_case(
                {**frozen, "attention_mask": [False, *TREE.CASE_MASK[1:]]}
            )
        with self.assertRaises(ValueError):
            TREE.validate_frozen_case(
                {**frozen, "attention_mask": [1, *TREE.CASE_MASK[1:]]}
            )

    def test_bounded_read_and_exclusive_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_bytes(b"x" * 2)
            original_limit = TREE.REPORT_MAX_BYTES
            TREE.REPORT_MAX_BYTES = 1
            try:
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    TREE.read_bounded(path, TREE.REPORT_MAX_BYTES, "report")
            finally:
                TREE.REPORT_MAX_BYTES = original_limit
            output = Path(directory) / "receipt.json"
            TREE.write_exclusive(output, {"ok": True})
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            with self.assertRaises(FileExistsError):
                TREE.write_exclusive(output, {"ok": False})

    def test_all_calibration_report_cases_require_exact_names(self) -> None:
        expected = {"cal_a", "cal_b"}
        report = {
            "cases": [
                {"name": "legacy", "split": "legacy_diagnostic"},
                {"name": "cal_a", "split": "calibration"},
                {"name": "cal_b", "split": "calibration"},
            ]
        }
        self.assertEqual(set(TREE.calibration_report_cases(report, expected)), expected)
        with self.assertRaises(ValueError):
            TREE.calibration_report_cases({"cases": report["cases"][:-1]}, expected)
        with self.assertRaises(ValueError):
            TREE.calibration_report_cases(
                {
                    "cases": [
                        *report["cases"][:-1],
                        {"name": "cal_a", "split": "calibration"},
                    ]
                },
                expected,
            )
        with self.assertRaises(ValueError):
            TREE.calibration_report_cases(
                {
                    "cases": [
                        *report["cases"],
                        {"name": "hold", "split": "held_out"},
                    ]
                },
                expected,
            )

    def test_predeclared_gate_rejects_a_worsened_case(self) -> None:
        def result(*, no_worse: bool, strict: bool) -> dict[str, object]:
            return {
                "controls": {
                    "serial_two_pass_vs_native": {"max_abs": 0.0},
                    "torch_layer_norm_vs_source": {"max_abs": 0.0},
                },
                "candidate": {
                    "no_worse_than_serial_source_max_abs": no_worse,
                    "strictly_better_than_serial_source_max_abs": strict,
                },
            }

        passing = TREE.predeclared_gate(
            [result(no_worse=True, strict=True), result(no_worse=True, strict=False)]
        )
        self.assertTrue(passing["passes"])
        worsened = TREE.predeclared_gate(
            [result(no_worse=True, strict=True), result(no_worse=False, strict=False)]
        )
        self.assertFalse(worsened["balanced_no_worse_each_case"])
        self.assertFalse(worsened["passes"])

    def test_propagation_gate_rejects_a_worsened_boundary(self) -> None:
        def case(*, no_worse: bool, strict: bool) -> dict[str, object]:
            return {
                "controls": {
                    "scalar_boundaries_match_saved_native_bits": True,
                    "balanced_embedding_matches_python_control": True,
                },
                "boundaries": [
                    {
                        "no_worse_than_scalar_source_max_abs": no_worse,
                        "strictly_better_than_scalar_source_max_abs": strict,
                    }
                ],
            }

        passing = TREE.propagation_gate(
            [case(no_worse=True, strict=True), case(no_worse=True, strict=False)]
        )
        self.assertTrue(passing["passes"])
        worsened = TREE.propagation_gate(
            [case(no_worse=True, strict=True), case(no_worse=False, strict=False)]
        )
        self.assertFalse(worsened["balanced_no_worse_every_calibration_boundary"])
        self.assertFalse(worsened["passes"])

    def test_propagation_receipt_requires_exact_calibration_names(self) -> None:
        payload = {
            "schema_version": 1,
            "protocol_schema": 1,
            "cases": [{"name": "one"}, {"name": "two"}],
        }
        self.assertEqual(
            set(TREE.propagation_native_cases(payload, {"one", "two"})),
            {"one", "two"},
        )
        with self.assertRaises(ValueError):
            TREE.propagation_native_cases(
                {**payload, "cases": [{"name": "one"}, {"name": "one"}]},
                {"one", "two"},
            )


if __name__ == "__main__":
    unittest.main()
