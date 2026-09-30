#!/usr/bin/env python3
"""Dependency-free argument and calibration-case checks for the source oracle."""

from __future__ import annotations

import importlib.util
import json
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

    def test_native_trace_rejects_wrong_calibration_and_excess_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.json"
            for trace in (
                {"schema_version": 1},
                {"schema_version": 2, "case": "held_out"},
                {
                    "schema_version": 2,
                    "case": "cal_len7",
                    "attention_layout": "unknown",
                },
            ):
                path.write_text(json.dumps(trace))
                # Admission must reject this metadata before consulting Torch.
                with self.assertRaises(ValueError):
                    ORACLE.load_native_trace(None, path)
            path.write_bytes(b" " * (ORACLE.NATIVE_TRACE_MAX_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                ORACLE.load_native_trace(None, path)

    def test_native_embedding_trace_rejects_wrong_case_and_excess_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embedding.json"
            for trace in (
                {"schema_version": 2},
                {"schema_version": 1, "case": "held_out"},
                {
                    "schema_version": 1,
                    "case": "cal_len7",
                    "input_ids": [1],
                },
            ):
                path.write_text(json.dumps(trace))
                with self.assertRaises(ValueError):
                    ORACLE.load_native_embedding_trace(None, path)
            path.write_bytes(b" " * (ORACLE.NATIVE_TRACE_MAX_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                ORACLE.load_native_embedding_trace(None, path)


if __name__ == "__main__":
    unittest.main()
