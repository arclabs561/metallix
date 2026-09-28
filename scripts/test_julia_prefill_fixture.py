#!/usr/bin/env python3
"""Structural controls for the frozen Julia prefill source fixture."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parent.parent / "fixtures/julia-1/prefill-reference.json"


class JuliaPrefillFixtureTests(unittest.TestCase):
    def test_composes_the_required_prefix_and_head_controls(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(
            fixture["sources"]["julia_model"],
            {
                "revision": "a85b127321d580d65176c89ced8273f305745d85",
                "sha256": "ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81",
            },
        )
        self.assertEqual(
            fixture["sources"]["modernbert"],
            {
                "revision": "08810b1e278938278c50153ee1edfd7a20a759da",
                "sha256": "83875f54a029339c62a8f5061801873d41e134b3e9abb8308b8e9b0f9f57b5dc",
            },
        )
        self.assertEqual(fixture["operator_config"]["encoder_layers"], [0, 1])
        self.assertEqual(fixture["operator_config"]["head_layers"], 2)
        self.assertEqual(fixture["operator_config"]["positions"], 6)
        self.assertEqual(fixture["operator_config"]["width"], 384)
        self.assertEqual(fixture["operator_config"]["tolerance"], 1e-5)
        self.assertEqual(
            fixture["input_generation"],
            "arange(6*384), ((x*7)%29-14)/20; optional padded rows 4..5 add ((x*11)%31-15)/3",
        )
        self.assertEqual(
            fixture["weight_generation"],
            "encoder ordinals layer0=0..5, layer1=6..11; Julia head uses pinned named-parameter ordinals from head-reference",
        )
        cases = {case["name"]: case for case in fixture["cases"]}
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
        self.assertEqual(
            cases["base"]["expected_scores"],
            cases["padding_perturbation"]["expected_scores"],
        )
        self.assertEqual(
            cases["base"]["expected_scores"],
            list(reversed(cases["option_permutation"]["expected_scores"])),
        )
        self.assertEqual(
            len(cases["base"]["expected_scores"]),
            len(cases["unmasked_padding_control"]["expected_scores"]),
        )
        delta = max(
            abs(left - right)
            for left, right in zip(
                cases["base"]["expected_scores"],
                cases["unmasked_padding_control"]["expected_scores"],
            )
        )
        self.assertGreater(delta, fixture["operator_config"]["tolerance"])
        self.assertEqual(cases["masked_marker"]["expected_scores"][-1], -10000.0)


if __name__ == "__main__":
    unittest.main()
