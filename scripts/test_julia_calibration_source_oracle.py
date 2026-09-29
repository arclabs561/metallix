#!/usr/bin/env python3
"""Dependency-free argument and calibration-case checks for the source oracle."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "julia_calibration_source_oracle",
    ROOT / "scripts/julia_calibration_source_oracle.py",
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Julia source oracle")
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


class SourceOracleInputTest(unittest.TestCase):
    def test_fixed_case_and_argument_boundaries(self) -> None:
        ORACLE.validate_case(ORACLE.CASE)
        ORACLE.validate_arguments(True, False, None)
        ORACLE.validate_arguments(False, True, Path("receipt.json"))
        with self.assertRaises(ValueError):
            ORACLE.validate_case({**ORACLE.CASE, "input_ids": [1]})
        with self.assertRaises(ValueError):
            ORACLE.validate_arguments(True, True, None)
        with self.assertRaises(ValueError):
            ORACLE.validate_arguments(False, True, None)

    def test_receipt_write_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            ORACLE.write_exclusive(path, {"ok": True})
            with self.assertRaises(FileExistsError):
                ORACLE.write_exclusive(path, {"ok": False})


if __name__ == "__main__":
    unittest.main()
