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
"""Integrity tests for the same-trace layer-zero to layer-one fixture."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
FIXTURE = SCRIPTS.parent / "fixtures/deepseek-v41/layer0-to-layer1-reference.json"


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerZeroToLayerOneFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load("v41-forward-reference.py", "v41_layer_zero_bridge_runner")
        cls.exporter = load(
            "v41_layer0_to_layer1_capture.py", "v41_layer_zero_bridge_exporter"
        )
        cls.receipt = cls.runner.run_capture()
        cls.fixture = cls.exporter.layer0_to_layer1_fixture(cls.receipt)

    def test_committed_fixture_matches_current_source_capture(self) -> None:
        committed = json.loads(FIXTURE.read_text())
        self.assertEqual(
            {key: value for key, value in committed.items() if key != "source"},
            {key: value for key, value in self.fixture.items() if key != "source"},
        )
        changing = {
            "runner_sha256",
            "forward_observers_sha256",
            "complete_capture_sha256",
            "extractor_sha256",
        }
        self.assertEqual(
            {
                key: value
                for key, value in committed["source"].items()
                if key not in changing
            },
            {
                key: value
                for key, value in self.fixture["source"].items()
                if key not in changing
            },
        )

    def test_every_layer_zero_terminal_residual_is_the_layer_one_stream(self) -> None:
        self.assertEqual(
            [case["start_pos"] for case in self.fixture["cases"]], [0, 5, 6]
        )
        for case in self.fixture["cases"]:
            self.assertEqual(
                case["block_output"]["storage_sha256"],
                case["layer_one_engram_stream"]["storage_sha256"],
            )
            sequence = 5 if case["start_pos"] == 0 else 1
            self.assertEqual(case["block_output"]["shape"], [1, sequence, 2, 128])
            self.assertEqual(case["block_next_pre"]["shape"], [1, sequence, 2])
            self.assertEqual(set(case["hc"]), {"attention", "ffn"})
            self.assertEqual(case["gate_weights"]["shape"], [sequence, 2])
            self.assertEqual(case["gate_indices"]["shape"], [sequence, 2])
            startup = case["startup"]
            self.assertEqual(startup["input_ids"]["shape"], [1, sequence])
            self.assertEqual(startup["embedding"]["shape"], [1, sequence, 128])
            self.assertEqual(startup["embedding"]["dtype"], "torch.bfloat16")
            attention = case["attention"]
            self.assertEqual(attention["frequencies"]["shape"], [8, 16])
            self.assertEqual(
                attention["window"]["prepared"]["shape"], [1, sequence, 64]
            )
            self.assertEqual(
                attention["window"]["read"]["shape"],
                [1, sequence if case["start_pos"] == 0 else 6, 64],
            )
            self.assertEqual(
                attention["window"]["indices"]["shape"],
                [1, sequence, sequence if case["start_pos"] == 0 else 6],
            )
            self.assertEqual(
                attention["sparse"]["kv"]["storage_sha256"],
                attention["window"]["read"]["storage_sha256"],
            )
            self.assertEqual(
                attention["sparse"]["indices"]["storage_sha256"],
                attention["window"]["indices"]["storage_sha256"],
            )
        self.assertEqual(self.fixture["model"]["n_routed_experts"], 4)
        self.assertEqual(self.fixture["model"]["o_groups"], 2)
        self.assertEqual(self.fixture["model"]["vocab_size"], 8)
        self.assertEqual(self.fixture["parameters"]["embed.weight"]["shape"], [8, 128])
        self.assertEqual(
            self.fixture["parameters"]["layers.0.ffn.gate.weight"]["shape"], [4, 128]
        )

    def test_rejects_spliced_layer_one_stream(self) -> None:
        mutated = copy.deepcopy(self.receipt)
        stream = mutated["steps"][0]["intermediates"]["layers.1.engram_input"]["stream"]
        stream["storage_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            RuntimeError, "invalid layer-one Engram stream storage"
        ):
            self.exporter.layer0_to_layer1_fixture(mutated)

    def test_rejects_missing_layer_zero_ffn_hc_call(self) -> None:
        mutated = copy.deepcopy(self.receipt)
        calls = mutated["steps"][1]["hyper_connection_mixes"]
        mutated["steps"][1]["hyper_connection_mixes"] = [
            call
            for call in calls
            if not (call["layer_id"] == 0 and call["sublayer"] == "ffn")
        ]
        with self.assertRaisesRegex(RuntimeError, "requires both HC calls"):
            self.exporter.layer0_to_layer1_fixture(mutated)


if __name__ == "__main__":
    unittest.main()
