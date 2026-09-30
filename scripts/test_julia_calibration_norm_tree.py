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
    def test_calibration_case_rejects_held_out_and_duplicates(self) -> None:
        case = {"name": "cal_len7", "split": "calibration"}
        report = {"cases": [case]}
        self.assertEqual(TREE.calibration_case(report)["name"], "cal_len7")
        with self.assertRaises(ValueError):
            TREE.calibration_case(
                {
                    "cases": [
                        case,
                        {"name": "next", "split": "held_out"},
                    ]
                }
            )
        with self.assertRaises(ValueError):
            TREE.calibration_case(
                {
                    "cases": [
                        case,
                        case,
                    ]
                }
            )
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


if __name__ == "__main__":
    unittest.main()
