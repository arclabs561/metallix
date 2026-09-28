#!/usr/bin/env python3
"""Independent vectors for the pinned Julia-1 encoding reference.

These hand-derived vectors exercise serialization only.  They do not download
the published tokenizer or weights and do not demonstrate native Julia support.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "julia_encoding_contract", Path(__file__).with_name("julia_encoding_contract.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Julia encoding contract")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def tokenizer() -> object:
    return contract.StubTokenizer(
        {
            "choice question: Pick": [10, 11],
            "choice question: Hi  ": [12, 13],
            "noul question: Is": [14],
            " A": [20],
            " B": [21],
            " false": [22],
            " true": [23],
            " does not satisfy the safety criterion": [25, 26],
            " satisfies the safety criterion": [27, 28],
            " A ": [24],
            "state": [30, 31],
            "boolean-state": [32],
            "state-state-state-state-state": [33, 34, 35, 36, 37],
            "choice question: long": [40, 41, 42, 43, 44, 45, 46],
            " " + "many": list(range(100, 149)),
        }
    )


class JuliaEncodingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = tokenizer()

    def test_hand_derived_choice_ids_marker_positions_and_padding(self) -> None:
        row = {"state": "state", "question": "Pick", "options": ["A", "B"]}
        encoded = contract.sequence(self.tokenizer, row, max_length=40, head_length=32)
        self.assertEqual(
            encoded,
            {
                "ids": [1, 10, 11, 1, 4, 20, 4, 21, 1, 30, 31, 1],
                "markers": [4, 6],
                "qtype": 0,
                "truncated": False,
            },
        )
        batch = contract.collate(self.tokenizer, [row], max_length=40, head_length=32)
        self.assertEqual(
            batch["input_ids"], [[1, 10, 11, 1, 4, 20, 4, 21, 1, 30, 31, 1, 0, 0, 0, 0]]
        )
        self.assertEqual(batch["attention_mask"], [[1] * 12 + [0] * 4])
        self.assertEqual(batch["marker_pos"], [[4, 6]])
        self.assertEqual(batch["marker_mask"], [[True, True]])

    def test_option_permutation_changes_marker_contents_not_marker_order(self) -> None:
        encoded = contract.sequence(
            self.tokenizer,
            {"state": "state", "question": "Pick", "options": ["B", "A"]},
            max_length=40,
            head_length=32,
        )
        self.assertEqual(encoded["ids"], [1, 10, 11, 1, 4, 21, 4, 20, 1, 30, 31, 1])
        self.assertEqual(encoded["markers"], [4, 6])

    def test_noul_keeps_explicit_false_true_order_and_type_id(self) -> None:
        encoded = contract.sequence(
            self.tokenizer,
            {
                "state": "boolean-state",
                "question": "Is",
                "type": "noul",
                "options": ["false", "true"],
            },
            max_length=40,
            head_length=32,
        )
        self.assertEqual(encoded["ids"], [1, 14, 1, 4, 22, 4, 23, 1, 32, 1])
        self.assertEqual(encoded["markers"], [3, 5])
        self.assertEqual(encoded["qtype"], 2)
        with self.assertRaisesRegex(contract.ContractError, "options must contain"):
            contract.validate_row(
                {
                    "state": "boolean-state",
                    "question": "Is",
                    "type": "noul",
                    "options": ["only-one"],
                }
            )

    def test_noul_preserves_descriptive_false_true_criteria_in_caller_order(
        self,
    ) -> None:
        row = {
            "state": "boolean-state",
            "question": "Is",
            "type": "noul",
            "options": [
                "does not satisfy the safety criterion",
                "satisfies the safety criterion",
            ],
        }
        self.assertIs(contract.validate_row(row), row)
        encoded = contract.sequence(
            self.tokenizer, row, max_length=40, head_length=32, strict=True
        )
        self.assertEqual(
            encoded,
            {
                "ids": [1, 14, 1, 4, 25, 26, 4, 27, 28, 1, 32, 1],
                "markers": [3, 6],
                "qtype": 2,
                "truncated": False,
                "option_tokens": [2, 2],
            },
        )
        self.assertNotIn(22, encoded["ids"])
        self.assertNotIn(23, encoded["ids"])

    def test_noul_rejects_missing_or_invalid_descriptive_options(self) -> None:
        base = {"state": "boolean-state", "question": "Is", "type": "noul"}
        invalid = (
            base,
            {**base, "options": None},
            {**base, "options": ["only false"]},
            {**base, "options": ["false description", ""]},
            {**base, "options": ["false description", 2]},
        )
        for row in invalid:
            with (
                self.subTest(row=row),
                self.assertRaisesRegex(contract.ContractError, "options"),
            ):
                contract.validate_row(row)

    def test_mask_text_is_sanitized_before_tokenization(self) -> None:
        encoded = contract.sequence(
            self.tokenizer,
            {"state": "state", "question": "Hi [MASK]", "options": ["A[MASK]", "B"]},
            max_length=40,
            head_length=32,
        )
        self.assertEqual(encoded["ids"], [1, 12, 13, 1, 4, 24, 4, 21, 1, 30, 31, 1])
        with self.assertRaisesRegex(contract.ContractError, "Reserved model marker"):
            contract.sequence(
                self.tokenizer,
                {"state": "state", "question": "Hi [MASK]", "options": ["A", "B"]},
                max_length=40,
                head_length=32,
                strict=True,
            )

    def test_strict_mode_rejects_each_lossy_boundary(self) -> None:
        with self.assertRaisesRegex(contract.ContractError, "Question/options"):
            contract.sequence(
                self.tokenizer,
                {"state": "state", "question": "long", "options": ["A", "B"]},
                max_length=20,
                head_length=10,
                strict=True,
            )
        with self.assertRaisesRegex(contract.ContractError, "Option exceeds"):
            contract.sequence(
                self.tokenizer,
                {"state": "state", "question": "Pick", "options": ["many", "B"]},
                strict=True,
            )
        with self.assertRaisesRegex(contract.ContractError, "Game state"):
            contract.sequence(
                self.tokenizer,
                {
                    "state": "state-state-state-state-state",
                    "question": "Pick",
                    "options": ["A", "B"],
                },
                max_length=14,
                head_length=9,
                strict=True,
            )

    def test_non_strict_state_overflow_reports_truncation(self) -> None:
        encoded = contract.sequence(
            self.tokenizer,
            {
                "state": "state-state-state-state-state",
                "question": "Pick",
                "options": ["A", "B"],
            },
            max_length=14,
            head_length=9,
        )
        self.assertEqual(
            encoded["ids"], [1, 10, 11, 1, 4, 20, 4, 21, 1, 33, 34, 35, 36, 1]
        )
        self.assertTrue(encoded["truncated"])

    def test_collator_pads_marker_columns_and_rounds_to_eight(self) -> None:
        rows = [
            {
                "state": "ignored",
                "question": "ignored",
                "options": ["A", "B"],
                "target": 1,
                "_encoded": {"ids": [1, 10, 1], "markers": [1], "qtype": 0},
            },
            {
                "state": "ignored",
                "question": "ignored",
                "options": ["A", "B", "C"],
                "target": 2,
                "_encoded": {
                    "ids": [1, 14, 1, 4, 22, 4, 23, 1, 32],
                    "markers": [3, 5, 7],
                    "qtype": 2,
                },
            },
        ]
        batch = contract.collate(self.tokenizer, rows, max_length=16)
        self.assertEqual(len(batch["input_ids"][0]), 16)
        self.assertEqual(batch["marker_pos"], [[1, 0, 0], [3, 5, 7]])
        self.assertEqual(
            batch["marker_mask"], [[True, False, False], [True, True, True]]
        )
        self.assertEqual(batch["qtype"], [0, 2])
        self.assertEqual(batch["labels"], [1, 2])


if __name__ == "__main__":
    unittest.main()
