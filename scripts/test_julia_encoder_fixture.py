#!/usr/bin/env python3
"""Structural regression checks for the Julia ModernBERT source fixture."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

FIXTURE = Path(__file__).parent.parent / "fixtures/julia-1/encoder-reference.json"


class JuliaEncoderFixtureTests(unittest.TestCase):
    def test_global_padding_local_window_and_negative_control_are_present(self) -> None:
        fixture = json.loads(FIXTURE.read_text())
        self.assertEqual(
            fixture["source"]["revision"], "08810b1e278938278c50153ee1edfd7a20a759da"
        )
        self.assertEqual(
            fixture["source"]["sha256"],
            "83875f54a029339c62a8f5061801873d41e134b3e9abb8308b8e9b0f9f57b5dc",
        )
        self.assertEqual(fixture["operator_config"]["local_radius"], 64)
        cases = {item["name"]: item for item in fixture["cases"]}
        self.assertEqual(
            set(cases),
            {
                "global_padding",
                "global_padding_perturbed",
                "global_unmasked_control",
                "local_window_crossing",
                "global_window_control",
                "local_distant_perturbation",
                "global_distant_perturbation",
                "local_all_masked_query",
            },
        )
        self.assertEqual(
            cases["global_padding"]["output_sha256"],
            cases["global_padding_perturbed"]["output_sha256"],
        )
        self.assertNotEqual(
            cases["global_padding"]["output_sha256"],
            cases["global_unmasked_control"]["output_sha256"],
        )
        self.assertEqual(cases["local_window_crossing"]["positions"], 66)
        self.assertEqual(cases["local_window_crossing"]["layer"], 1)
        self.assertNotEqual(
            cases["local_window_crossing"]["output_sha256"],
            cases["global_window_control"]["output_sha256"],
        )
        self.assertEqual(
            cases["local_window_crossing"]["expected_output"][0],
            cases["local_distant_perturbation"]["expected_output"][0],
        )
        self.assertNotEqual(
            cases["global_window_control"]["expected_output"][0],
            cases["global_distant_perturbation"]["expected_output"][0],
        )
        self.assertEqual(
            cases["local_all_masked_query"]["attention_mask"].count(True), 1
        )


if __name__ == "__main__":
    unittest.main()
