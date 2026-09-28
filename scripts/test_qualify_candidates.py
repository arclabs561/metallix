#!/usr/bin/env python3
"""Independent schedule oracle and false-positive receipt checks."""

import copy
import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "qualify_candidates", Path(__file__).with_name("qualify-candidates.py")
)
qualifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qualifier)


class CandidateQualificationTests(unittest.TestCase):
    def test_pairwise_oracle_rejects_nested_overlap_and_bad_intervals(self):
        for spans, expected in (
            ([(0, 1), (1, 2)], True),
            ([(2, 3), (0, 1)], True),
            ([(0, 3), (1, 2)], False),
            ([(0, 0)], False),
            ([], False),
            ([(True, 2)], False),
            ([(0, float("nan"))], False),
        ):
            with self.subTest(spans=spans):
                value = {"intervals": [{"start": a, "end": b} for a, b in spans]}
                self.assertEqual(qualifier.non_overlapping(value), expected)

    def test_exhaustion_cannot_hide_accepted_output_or_repeat_retry_seed(self):
        report = {
            "candidate_verification": {
                "status": "exhausted",
                "attempts": [
                    {"seed": 1, "generated_tokens": 8, "status": "rejected"},
                    {"seed": 2, "generated_tokens": 8, "status": "rejected"},
                ],
                "max_attempts": 2,
                "max_total_generated_tokens": 32,
                "total_generated_tokens": 16,
            },
            "generated_ids": [],
            "input_ids": [1, 2],
            "cached_tokens": 2,
        }
        qualifier.check_receipt(report, 1, False)
        leaked = copy.deepcopy(report)
        leaked["constraint"] = {"generated_text": "{}"}
        with self.assertRaises(AssertionError):
            qualifier.check_receipt(leaked, 1, False)
        repeated = copy.deepcopy(report)
        repeated["candidate_verification"]["attempts"][1]["seed"] = 1
        with self.assertRaises(AssertionError):
            qualifier.check_receipt(repeated, 1, False)
        omitted_cost = copy.deepcopy(report)
        omitted_cost["candidate_verification"]["total_generated_tokens"] = 8
        with self.assertRaises(AssertionError):
            qualifier.check_receipt(omitted_cost, 1, False)


if __name__ == "__main__":
    unittest.main()
