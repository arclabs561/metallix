#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Opt-in fail-closed checks for Julia accuracy receipt validation helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "julia_accuracy_reference", ROOT / "scripts" / "julia_accuracy_reference.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Julia accuracy reference")
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)


class ReceiptValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.weight_sha = REFERENCE.weights_f32_sha256()

    def payload(self, cases: list[dict[str, object]]) -> dict[str, object]:
        return {
            "protocol_schema": 1,
            "manifest_sha256": REFERENCE.MANIFEST_SHA256,
            "weight_f32_sha256": self.weight_sha,
            "cases": cases,
        }

    def test_receipt_rejects_duplicate_missing_and_hash_mismatch(self) -> None:
        expected = {"one", "two"}
        with self.assertRaises(ValueError):
            REFERENCE.validate_native_receipt(self.payload([{"name": "one"}]), expected)
        with self.assertRaises(ValueError):
            REFERENCE.validate_native_receipt(
                self.payload([{"name": "one"}, {"name": "one"}]), expected
            )
        payload = self.payload([{"name": "one"}, {"name": "two"}])
        payload["manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            REFERENCE.validate_native_receipt(payload, expected)

    def test_shape_nonfinite_and_invalid_marker_reject(self) -> None:
        with self.assertRaises(ValueError):
            REFERENCE.check_shape_finite(torch.zeros((1, 2)), (1, 3), "shape")
        with self.assertRaises(ValueError):
            REFERENCE.check_shape_finite(
                torch.tensor([[float("nan")]]), (1, 1), "nonfinite"
            )
        with self.assertRaises(ValueError):
            REFERENCE.check_scores(
                torch.tensor([-9999.0]), torch.tensor([False]), "mask"
            )

    def test_boundary_metrics_retains_f64_reference(self) -> None:
        source = torch.tensor([1.0], dtype=torch.float32)
        reference = torch.tensor([1.0 + 1e-10], dtype=torch.float64)
        measured = REFERENCE.boundary_metrics(source, reference, 1.0)
        self.assertGreater(measured["max_abs"], 9e-11)
        self.assertLess(measured["max_abs"], 2e-10)
        with self.assertRaises(TypeError):
            REFERENCE.boundary_metrics(source, reference.float(), 1.0)


if __name__ == "__main__":
    unittest.main()
