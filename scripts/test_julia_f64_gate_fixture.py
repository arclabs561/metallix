#!/usr/bin/env python3
"""Structural controls for the float64 Julia accuracy gate fixture."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "fixtures/julia-1/f64-gate-reference.json"
MANIFEST = ROOT / "fixtures/julia-1/accuracy-cases.json"


class JuliaF64GateFixtureTests(unittest.TestCase):
    def test_covers_every_split_with_finite_reference_values(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(fixture["gate_ratio"], 4.0)
        self.assertEqual(fixture["score_max_abs"], 1e-5)
        cases = fixture["cases"]
        splits = [case["split"] for case in cases]
        self.assertEqual(splits.count("legacy"), 5)
        self.assertEqual(splits.count("calibration"), 8)
        self.assertEqual(splits.count("held_out"), 8)
        manifest = json.loads(MANIFEST.read_text())["cases"]
        self.assertEqual(
            {case["name"] for case in cases if case["split"] != "legacy"},
            {case["name"] for case in manifest},
        )
        for case in cases:
            rows = case["f64_hidden"]
            self.assertEqual(len(rows), len(case["input_ids"]))
            self.assertTrue(all(len(row) == 384 for row in rows))
            self.assertTrue(all(math.isfinite(v) for row in rows for v in row))
            self.assertGreater(case["source_f32_hidden_max_abs"], 0.0)
            self.assertLess(case["source_f32_hidden_max_abs"], 1e-4)


if __name__ == "__main__":
    unittest.main()
