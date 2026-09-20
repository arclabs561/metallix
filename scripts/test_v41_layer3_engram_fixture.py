#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.13.0",
#   "numpy==2.5.3",
#   "sympy==1.14.0",
#   "tokenizers==0.23.2",
# ]
# ///
"""Source-backed integrity and rejection tests for the layer-three Engram fixture."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
FIXTURE = SCRIPTS.parent / "fixtures/deepseek-v41/layer3-engram-reference.json"


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerThreeEngramFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_module("v41-forward-reference.py", "v41_engram_runner")
        cls.exporter = load_module(
            "v41_layer3_engram_capture.py", "v41_engram_exporter"
        )
        cls.receipt = cls.runner.run_capture()
        cls.fixture = cls.exporter.engram_fixture(cls.receipt)

    def test_committed_fixture_matches_current_source_capture(self) -> None:
        self.assertEqual(json.loads(FIXTURE.read_text()), self.fixture)

    def test_wkv_split_and_block_entry_are_exact(self) -> None:
        for case in self.fixture["cases"]:
            wkv = bytes.fromhex(case["wkv_output"]["storage_hex"])
            key = bytes.fromhex(case["key"]["storage_hex"])
            value = bytes.fromhex(case["value"]["storage_hex"])
            rebuilt = b"".join(
                key[offset : offset + 512] + value[index * 256 : (index + 1) * 256]
                for index, offset in enumerate(range(0, len(key), 512))
            )
            self.assertEqual(rebuilt, wkv)
            self.assertEqual(
                case["output"]["storage_sha256"],
                case["block_entry"]["storage_sha256"],
            )

    def test_missing_engram_projection_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        del receipt["steps"][0]["intermediates"]["layers.3.engram.wkv"]
        with self.assertRaisesRegex(TypeError, "WKV"):
            self.exporter.engram_fixture(receipt)

    def test_changed_engram_output_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["steps"][0]["intermediates"]["layers.3.engram"]["storage_sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(RuntimeError, "exact storage"):
            self.exporter.engram_fixture(receipt)

    def test_changed_hash_state_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["engram"]["hash_state"]["primes"]["finite"] = False
        with self.assertRaisesRegex(RuntimeError, "invalid finite"):
            self.exporter.engram_fixture(receipt)


if __name__ == "__main__":
    unittest.main()
