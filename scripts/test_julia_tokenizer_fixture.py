#!/usr/bin/env python3
"""Pure-stdlib checks for pinned Julia tokenizer sequence vectors."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

CONTRACT_PATH = Path(__file__).with_name("julia_encoding_contract.py")
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures/julia-1/tokenizer-sequence.json"
SPEC = importlib.util.spec_from_file_location("julia_encoding_contract", CONTRACT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Julia encoding contract")
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


class JuliaTokenizerFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE_PATH.read_text())
        tokenizer = cls.fixture["tokenizer"]
        special = tokenizer["special_ids"]
        cls.tokenizer = contract.StubTokenizer(
            cls.fixture["lexical_ids"],
            pad_token_id=special["pad"],
            cls_token_id=special["cls"],
            sep_token_id=special["sep"],
            mask_token_id=special["mask"],
            mask_token=tokenizer["mask_token"],
        )

    def test_fixture_has_pinned_source_and_tokenizer_provenance(self) -> None:
        source = self.fixture["source"]
        tokenizer = self.fixture["tokenizer"]
        self.assertEqual(source["revision"], contract.SOURCE_REVISION)
        self.assertEqual(
            source["sha256"],
            "e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e",
        )
        self.assertEqual(source["ast_symbols"], ["QTYPES", "sequence"])
        self.assertEqual(
            tokenizer["sha256"],
            "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
        )
        self.assertEqual(tokenizer["tokenizers_version"], "0.23.2")
        self.assertEqual(len(tokenizer["sha256"]), hashlib.sha256().digest_size * 2)

    def test_exact_vectors_cover_strict_mask_and_truncation(self) -> None:
        for case in self.fixture["cases"]:
            parameters = case["parameters"]
            with self.subTest(case=case["name"]):
                if "expected_error" in case:
                    with self.assertRaisesRegex(
                        contract.ContractError, case["expected_error"]
                    ):
                        contract.sequence(self.tokenizer, case["row"], **parameters)
                else:
                    self.assertEqual(
                        contract.sequence(self.tokenizer, case["row"], **parameters),
                        case["expected"],
                    )

    def test_reverse_options_only_swaps_marker_aligned_tokens(self) -> None:
        cases = {case["name"]: case for case in self.fixture["cases"]}
        forward = cases["choice_strict"]["expected"]
        reverse = cases["choice_reverse_options"]["expected"]
        self.assertEqual(forward["markers"], reverse["markers"])
        self.assertEqual(forward["qtype"], reverse["qtype"])
        self.assertEqual(forward["truncated"], reverse["truncated"])
        self.assertEqual(forward["option_tokens"], reverse["option_tokens"])
        self.assertEqual(forward["ids"][:6], reverse["ids"][:6])
        self.assertEqual(forward["ids"][10:], reverse["ids"][10:])
        self.assertEqual(forward["ids"][6:10], [4, 586, 4, 599])
        self.assertEqual(reverse["ids"][6:10], [4, 599, 4, 586])


if __name__ == "__main__":
    unittest.main()
