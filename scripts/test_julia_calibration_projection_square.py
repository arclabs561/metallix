#!/usr/bin/env python3
"""Dependency-free boundary checks for the Julia projection-square receipt."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "julia_calibration_projection_square",
    ROOT / "scripts/julia_calibration_projection_square.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Julia projection-square diagnostic")
SQUARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SQUARE)


class ProjectionSquareInputTest(unittest.TestCase):
    def test_requires_one_calibration_case_and_rejects_held_out(self) -> None:
        report = {"cases": [{"name": "cal_len7", "split": "calibration"}]}
        self.assertEqual(SQUARE.calibration_case(report)["name"], "cal_len7")
        with self.assertRaises(ValueError):
            SQUARE.calibration_case(
                {
                    "cases": [
                        {"name": "cal_len7", "split": "calibration"},
                        {"name": "held", "split": "held_out"},
                    ]
                }
            )
        with self.assertRaises(ValueError):
            SQUARE.calibration_case(
                {
                    "cases": [
                        {"name": "cal_len7", "split": "calibration"},
                        {"name": "cal_len7", "split": "calibration"},
                    ]
                }
            )

    def test_bounded_read_and_exclusive_write(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_bytes(b"x" * 2)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                SQUARE.read_bounded(path, 1, "receipt")
            output = Path(directory) / "output.json"
            SQUARE.write_exclusive(output, {"ok": True})
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            with self.assertRaises(FileExistsError):
                SQUARE.write_exclusive(output, {"ok": False})


if __name__ == "__main__":
    unittest.main()
