#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""Tests for the pinned synthetic Julia decision-head numerical fixture."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

import torch

SCRIPT_PATH = Path(__file__).with_name("julia_head_reference.py")
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures/julia-1/head-reference.json"
SPEC = importlib.util.spec_from_file_location("julia_head_reference", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Julia head reference")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


class JuliaHeadReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text())

    def test_frozen_fixture_matches_source_and_transparent_reference(self) -> None:
        reference.verify(self.fixture)

    def test_fixture_requires_padding_permutation_and_invalid_marker_cases(
        self,
    ) -> None:
        cases = {case["name"]: case for case in self.fixture["cases"]}
        self.assertEqual(
            set(cases),
            {
                "base",
                "padding_perturbation",
                "option_permutation",
                "unmasked_padding_control",
                "masked_marker",
            },
        )
        base = torch.tensor(cases["base"]["expected_scores"])
        padding = torch.tensor(cases["padding_perturbation"]["expected_scores"])
        permutation = torch.tensor(cases["option_permutation"]["expected_scores"])
        unmasked = torch.tensor(cases["unmasked_padding_control"]["expected_scores"])
        masked = torch.tensor(cases["masked_marker"]["expected_scores"])
        self.assertTrue(torch.allclose(base, padding, rtol=1e-5, atol=1e-5))
        self.assertTrue(torch.allclose(base, permutation.flip(0), rtol=1e-5, atol=1e-5))
        self.assertFalse(torch.allclose(base, unmasked, rtol=1e-5, atol=1e-5))
        self.assertEqual(float(masked[-1]), -10000.0)
        self.assertTrue(torch.allclose(base, masked[:2], rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
