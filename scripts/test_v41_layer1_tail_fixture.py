#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "numpy==2.5.3", "sympy==1.14.0", "tokenizers==0.23.2"]
# ///
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
FIXTURE = SCRIPTS.parent / "fixtures/deepseek-v41/layer1-tail-reference.json"


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerOneTailFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load("v41-forward-reference.py", "v41_layer1_tail_runner")
        cls.exporter = load("v41_layer1_tail_capture.py", "v41_layer1_tail_exporter")
        cls.receipt = cls.runner.run_capture()
        cls.fixture = json.loads(
            json.dumps(cls.exporter.layer1_tail_fixture(cls.receipt))
        )

    def test_committed_fixture_matches_current_source_capture(self):
        self.assertEqual(json.loads(FIXTURE.read_text()), self.fixture)

    def test_missing_input_and_terminal_mutation_are_rejected(self):
        capture = copy.deepcopy(self.receipt)
        del capture["steps"][0]["intermediates"]["layers.1.ffn"]
        with self.assertRaisesRegex(TypeError, "MoE output"):
            self.exporter.layer1_tail_fixture(capture)
        capture = copy.deepcopy(self.receipt)
        record = capture["steps"][0]["intermediates"]["layers.1"][0]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] ^= 1
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "does not feed"):
            self.exporter.layer1_tail_fixture(capture)

    def test_nonfinite_parameter_is_rejected(self):
        capture = copy.deepcopy(self.receipt)
        record = capture["encoded_parameters"]["layers.1.hc_attn_fn"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[:4] = bytes.fromhex("0000807f")
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            self.exporter.layer1_tail_fixture(capture)

    def test_source_hc_calls_are_layer_one(self):
        for step, case in zip(
            self.receipt["steps"], self.fixture["cases"], strict=True
        ):
            selected = [
                call
                for call in step["hyper_connection_mixes"]
                if call["layer_id"] == 1 and call["sublayer"] in {"attention", "ffn"}
            ]
            self.assertEqual(
                {call["sublayer"] for call in selected}, {"attention", "ffn"}
            )
            self.assertEqual(len(selected), 2)
            for call in selected:
                kind = call["sublayer"]
                self.assertEqual(case[f"{kind}_hc_mixes"], call["inputs"]["mixes"])
                self.assertEqual(case[f"{kind}_coefficients"], call["outputs"])

    def test_terminal_output_feeds_layer_two_entry(self):
        for case in self.fixture["cases"]:
            self.assertEqual(
                case["output"]["storage_sha256"],
                case["layer_two_residual"]["storage_sha256"],
            )
            self.assertEqual(
                case["next_pre"]["storage_sha256"],
                case["layer_two_incoming_pre"]["storage_sha256"],
            )

    def test_layer_one_hc_binding_is_restored_after_failure(self) -> None:
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
            existed = "hc_mixes" in model.layers[1].__dict__
            original = model.layers[1].__dict__.get("hc_mixes")
            with (
                patch.object(
                    model.layers[1].ffn,
                    "forward",
                    side_effect=RuntimeError("injected layer-one FFN failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected layer-one FFN failure"),
                runner.v41_forward_observers.hooks_for(
                    model,
                    graph,
                    runner.tensor_record,
                    runner.object_record,
                    runner.MAX_HOOK_RECORDS,
                ),
            ):
                model(torch.tensor([runner.TRACE_INPUT_IDS[0][:5]]), start_pos=0)
            self.assertEqual("hc_mixes" in model.layers[1].__dict__, existed)
            self.assertIs(model.layers[1].__dict__.get("hc_mixes"), original)


if __name__ == "__main__":
    unittest.main()
