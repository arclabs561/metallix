#!/usr/bin/env python3
"""Structural controls for the frozen full Julia encoder prefill fixture."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parent.parent / "fixtures/julia-1/full-prefill-reference.json"


class JuliaFullPrefillFixtureTests(unittest.TestCase):
    def test_pins_sources_configuration_and_padding_controls(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(
            fixture["sources"]["julia_model"]["revision"],
            "a85b127321d580d65176c89ced8273f305745d85",
        )
        self.assertEqual(
            fixture["sources"]["julia_model"]["sha256"],
            "ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81",
        )
        self.assertEqual(
            fixture["sources"]["modernbert"]["revision"],
            "08810b1e278938278c50153ee1edfd7a20a759da",
        )
        self.assertEqual(
            fixture["sources"]["modernbert"]["sha256"],
            "83875f54a029339c62a8f5061801873d41e134b3e9abb8308b8e9b0f9f57b5dc",
        )
        self.assertEqual(
            fixture["sources"]["modernbert"]["attention_implementation"], "sdpa"
        )
        self.assertEqual(
            fixture["operator_config"],
            {
                "layers": 22,
                "positions_max": 8,
                "vocab_rows": 8,
                "norm_eps": 1e-5,
                "norm_bias": False,
                "tolerance": 1e-5,
            },
        )
        cases = {case["name"]: case for case in fixture["cases"]}
        self.assertEqual(
            set(cases),
            {
                "padded_base",
                "padded_changed_rows",
                "unmasked_control",
                "no_padding",
                "masked_marker",
            },
        )
        self.assertEqual(
            cases["padded_base"]["expected_hidden"][:4],
            cases["padded_changed_rows"]["expected_hidden"][:4],
        )
        delta = max(
            abs(left - right)
            for left, right in zip(
                cases["padded_base"]["expected_scores"],
                cases["unmasked_control"]["expected_scores"],
            )
        )
        self.assertGreater(delta, fixture["operator_config"]["tolerance"])
        self.assertEqual(cases["masked_marker"]["expected_scores"][-1], -10000.0)


if __name__ == "__main__":
    unittest.main()
