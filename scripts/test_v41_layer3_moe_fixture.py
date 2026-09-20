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
"""Source-backed rejection tests for the layer-three block-entry fixture."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


def load_runner():
    path = Path(__file__).with_name("v41-forward-reference.py")
    spec = importlib.util.spec_from_file_location("v41_layer3_moe_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("source runner import unavailable")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


class LayerThreeMoeFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_runner()
        cls.receipt = cls.runner.run_capture()

    def test_terminal_state_is_the_layer_four_entry(self) -> None:
        fixture = self.runner.moe_fixture(self.receipt, layer=3)
        self.assertEqual([case["start_pos"] for case in fixture["cases"]], [0, 5, 6])
        for case in fixture["cases"]:
            entry = case["next_block_entry"]
            self.assertEqual(
                case["block_output"]["storage_sha256"],
                entry["residual"]["storage_sha256"],
            )
            self.assertEqual(
                case["block_next_pre"]["storage_sha256"],
                entry["incoming_pre"]["storage_sha256"],
            )

    def test_missing_layer_four_entry_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        del receipt["steps"][0]["intermediates"]["layers.4.block_input"]
        with self.assertRaisesRegex(TypeError, "layer-four block-entry"):
            self.runner.moe_fixture(receipt, layer=3)

    def test_changed_terminal_storage_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["steps"][0]["intermediates"]["layers.3"][0]["storage_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "terminal state"):
            self.runner.moe_fixture(receipt, layer=3)


if __name__ == "__main__":
    unittest.main()
