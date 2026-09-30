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

    def test_len2_intervention_admission_rejects_mixed_modes(self) -> None:
        native = Path("native.json")
        report = Path("calibration.json")
        output = Path("receipt.json")
        ORACLE.validate_len2_intervention_arguments(
            True, output, native, report, False, False, None, None
        )
        with self.assertRaises(ValueError):
            ORACLE.validate_len2_intervention_arguments(
                True, output, native, report, True, False, None, None
            )
        with self.assertRaises(ValueError):
            ORACLE.validate_len2_intervention_arguments(
                True, output, native, report, False, True, None, None
            )
        with self.assertRaises(ValueError):
            ORACLE.validate_len2_intervention_arguments(
                True, output, native, report, False, False, Path("legacy.json"), None
            )
        with self.assertRaisesRegex(ValueError, "require --run-len2-intervention"):
            ORACLE.validate_len2_intervention_arguments(
                False, output, native, report, False, True, None, None
            )

    def test_len2_intervention_trace_rejects_nonfrozen_case_before_torch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.json"
            path.write_text(json.dumps({"schema_version": 1, "case": "held_out"}))
            with self.assertRaises(ValueError):
                ORACLE.load_len2_intervention_trace(None, path)
            path.write_bytes(b" " * (ORACLE.NATIVE_TRACE_MAX_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                ORACLE.load_len2_intervention_trace(None, path)

    def test_len2_intervention_rejects_duplicate_cached_case_before_torch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            case = {"name": "cal_len2", "split": "calibration"}
            path.write_text(
                json.dumps(
                    {
                        "manifest_sha256": ORACLE.MANIFEST_SHA256,
                        "weight_f32_sha256": ORACLE.WEIGHT_SHA256,
                        "cases": [case, case],
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "requires one cal_len2"):
                ORACLE.load_len2_cached_calibration_case(None, path)

    def test_qkv_override_gate_rejects_a_worsened_downstream_stage(self) -> None:
        def comparisons(
            *,
            candidate_attended: float,
            candidate_residual: float,
            candidate_output: float,
        ) -> dict[str, dict[str, dict[str, float]]]:
            return {
                stage: {
                    "scalar_native_vs_same_input_source": {"max_abs": baseline},
                    "scalar_native_vs_qkv_override_source": {"max_abs": candidate},
                }
                for stage, baseline, candidate in (
                    ("attended", 2.0, candidate_attended),
                    ("post_wo_residual", 3.0, candidate_residual),
                    ("layer_0_output", 4.0, candidate_output),
                )
            }

        passing = ORACLE.qkv_override_gate(
            comparisons(
                candidate_attended=1.0, candidate_residual=3.0, candidate_output=4.0
            )
        )
        self.assertTrue(passing["passes"])
        worsened = ORACLE.qkv_override_gate(
            comparisons(
                candidate_attended=1.0, candidate_residual=3.1, candidate_output=3.0
            )
        )
        self.assertFalse(worsened["no_worse_each_downstream_stage"])
        self.assertFalse(worsened["passes"])


if __name__ == "__main__":
    unittest.main()
