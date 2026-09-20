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
"""Source-backed integrity and rejection tests for the layer-two FFN tail."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPTS = Path(__file__).parent
FIXTURE = SCRIPTS.parent / "fixtures/deepseek-v41/layer2-ffn-reference.json"


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerTwoFfnFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load("v41-forward-reference.py", "v41_layer2_ffn_runner")
        cls.exporter = load("v41_layer2_ffn_capture.py", "v41_layer2_ffn_exporter")
        cls.receipt = cls.runner.run_capture()
        cls.fixture = cls.exporter.layer2_ffn_fixture(cls.receipt)

    def test_committed_fixture_matches_current_source_capture(self) -> None:
        """Additive observer hooks cannot rewrite this historical numerical oracle."""
        committed = json.loads(FIXTURE.read_text())
        self.assertEqual(
            {key: value for key, value in committed.items() if key != "source"},
            {key: value for key, value in self.fixture.items() if key != "source"},
        )
        self.assertEqual(
            {
                key: value
                for key, value in committed["source"].items()
                if key not in {"forward_observers_sha256", "complete_capture_sha256"}
            },
            {
                key: value
                for key, value in self.fixture["source"].items()
                if key not in {"forward_observers_sha256", "complete_capture_sha256"}
            },
        )
        for key in ("forward_observers_sha256", "complete_capture_sha256"):
            self.assertRegex(self.fixture["source"][key], r"^[0-9a-f]{64}$")

    def test_terminal_tail_has_both_layer_three_seams(self) -> None:
        for case in self.fixture["cases"]:
            self.assertEqual(
                case["output"]["storage_sha256"],
                case["engram_stream"]["storage_sha256"],
            )
            self.assertEqual(
                case["next_pre"]["storage_sha256"],
                case["layer_three_incoming_pre"]["storage_sha256"],
            )

    def test_missing_post_attention_residual_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        del receipt["steps"][0]["intermediates"]["layers.2.after_attention_residual"]
        with self.assertRaisesRegex(TypeError, "post-attention"):
            self.exporter.layer2_ffn_fixture(receipt)

    def test_changed_terminal_residual_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["steps"][0]["intermediates"]["layers.2"][0]["storage_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exact storage"):
            self.exporter.layer2_ffn_fixture(receipt)

    def test_changed_returned_pre_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["steps"][0]["intermediates"]["layers.2"][1]["storage_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exact storage"):
            self.exporter.layer2_ffn_fixture(receipt)

    def test_consistently_hashed_nonfinite_parameter_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        tensor = receipt["encoded_parameters"]["layers.2.ffn.shared_experts.w1.weight"]
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[-1] = 0x7F
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertTrue(tensor["finite"])
        with self.assertRaisesRegex(RuntimeError, "nonfinite storage"):
            self.exporter.layer2_ffn_fixture(receipt)

    def test_layer_two_hc_binding_is_restored_after_failure(self) -> None:
        runner = self.runner
        graph = runner.source_loader.load_text_graph(runner.kernels)
        graph.shared_attn = graph.SharedAttentionRuntime()
        with graph.set_dtype(torch.bfloat16):
            args, tokenizer = runner.executable_args(
                graph, runner.forward_manifest.manifest()
            )
            model = graph.Transformer(args, tokenizer).eval()
            runner.initialize_parameters(model)
            runner.initialize_runtime_buffers(model)
            existed = "hc_mixes" in model.layers[2].__dict__
            original = model.layers[2].__dict__.get("hc_mixes")
            with (
                patch.object(
                    model.layers[2].ffn,
                    "forward",
                    side_effect=RuntimeError("injected layer-two FFN failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected layer-two FFN failure"),
                runner.v41_forward_observers.hooks_for(
                    model,
                    graph,
                    runner.tensor_record,
                    runner.object_record,
                    runner.MAX_HOOK_RECORDS,
                ),
            ):
                model(torch.tensor([runner.TRACE_INPUT_IDS[0][:5]]), start_pos=0)
            self.assertEqual("hc_mixes" in model.layers[2].__dict__, existed)
            self.assertIs(model.layers[2].__dict__.get("hc_mixes"), original)


if __name__ == "__main__":
    unittest.main()
