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
"""Source-backed integrity and rejection tests for the layer-one owner fixture."""

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
FIXTURE = SCRIPTS.parent / "fixtures/deepseek-v41/layer1-ratio2-owner-reference.json"


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerOneOwnerFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load("v41-forward-reference.py", "v41_layer1_owner_runner")
        cls.exporter = load("v41_layer1_owner_capture.py", "v41_layer1_owner_exporter")
        cls.receipt = cls.runner.run_capture()
        cls.fixture = cls.exporter.layer1_owner_fixture(cls.receipt)

    def test_committed_fixture_matches_current_source_capture(self) -> None:
        self.assertEqual(json.loads(FIXTURE.read_text()), self.fixture)

    def test_owner_schedule_preserves_ratio_two_partial_publication(self) -> None:
        self.assertEqual(
            [case["start_pos"] for case in self.fixture["cases"]], [0, 5, 6]
        )
        self.assertEqual(
            [case["compressed_prefix"] for case in self.fixture["cases"]], [2, 3, 3]
        )
        self.assertEqual(
            [case["group_frequency_positions"] for case in self.fixture["cases"]],
            [[0, 2], [4], []],
        )
        self.assertIsNone(self.fixture["cases"][2]["latent"])
        self.assertNotIn(
            "k_after_rope_fp4", self.fixture["cases"][2]["index_operations"]
        )

    def test_owner_has_no_candidate_mask_semantics(self) -> None:
        self.assertEqual(self.fixture["model"]["owner_layer"], 1)
        self.assertEqual(self.fixture["model"]["candidate_source_layer"], 3)
        serialized = json.dumps(self.fixture, sort_keys=True)
        self.assertNotIn("candidate_mask", serialized)
        for case in self.fixture["cases"]:
            self.assertEqual(
                case["selected_indices"]["storage_sha256"],
                case["compressed_indices"]["storage_sha256"],
            )

    def test_partial_decode_keeps_owner_publication_separate_from_score_operand(
        self,
    ) -> None:
        completed, partial = self.fixture["cases"][1:]
        self.assertEqual(
            completed["index_key_prefix"]["storage_sha256"],
            completed["index_score_key_prefix"]["storage_sha256"],
        )
        self.assertEqual(
            completed["index_key_prefix"]["storage_sha256"],
            partial["index_key_prefix"]["storage_sha256"],
        )
        self.assertNotEqual(
            partial["index_key_prefix"]["storage_sha256"],
            partial["index_score_key_prefix"]["storage_sha256"],
        )

    def test_source_partial_score_operand_is_current_layer_three_publication(
        self,
    ) -> None:
        partial = self.receipt["steps"][2]["intermediates"]
        completed = self.receipt["steps"][1]["intermediates"]
        owner_score_prefix = partial["layers.1.attn.indexer_observation"]["inputs"][
            "shared_index_k_prefix"
        ]
        layer_three_score_prefix = completed["layers.3.attn.indexer_observation"][
            "inputs"
        ]["shared_index_k_prefix"]
        self.assertEqual(owner_score_prefix["shape"], [1, 3, 64])
        self.assertEqual(layer_three_score_prefix["shape"], [1, 6, 64])
        self.assertEqual(owner_score_prefix["dtype"], "torch.bfloat16")
        self.assertEqual(layer_three_score_prefix["dtype"], "torch.bfloat16")
        self.assertEqual(
            bytes.fromhex(owner_score_prefix["storage_hex"]),
            bytes.fromhex(layer_three_score_prefix["storage_hex"])[: 3 * 64 * 2],
        )

    def test_missing_gate_projection_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        del receipt["steps"][0]["intermediates"]["layers.1.attn.compressor.wgate"]
        with self.assertRaisesRegex(TypeError, "gate projection"):
            self.exporter.layer1_owner_fixture(receipt)

    def test_pinned_source_revision_and_model_are_required(self) -> None:
        for field in ("revision", "model_sha256"):
            with self.subTest(field=field):
                receipt = copy.deepcopy(self.receipt)
                receipt["source"][field] = "0" * 64
                with self.assertRaisesRegex(RuntimeError, f"source {field}"):
                    self.exporter.layer1_owner_fixture(receipt)

    def test_partial_decode_cannot_publish_a_latent(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["steps"][2]["intermediates"]["layers.1.attn.compressor"] = receipt[
            "steps"
        ][1]["intermediates"]["layers.1.attn.compressor"]
        with self.assertRaisesRegex(RuntimeError, "partial.*published"):
            self.exporter.layer1_owner_fixture(receipt)

    def test_partial_decode_preserves_both_published_prefixes(self) -> None:
        paths = (
            ("layers.1.attn.indexer_observation", "inputs", "owner_key_prefix"),
            ("layers.1.attn.compressed", "borrowed_kv"),
        )
        for path in paths:
            with self.subTest(path=path):
                receipt = copy.deepcopy(self.receipt)
                value: object = receipt["steps"][2]["intermediates"]
                for key in path:
                    value = value[key]  # type: ignore[index]
                tensor = value  # type: ignore[assignment]
                raw = bytearray.fromhex(tensor["storage_hex"])
                raw[0:2] = b"\x00\x00"
                tensor["storage_hex"] = raw.hex()
                tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                with self.assertRaisesRegex(RuntimeError, "partial decode changed"):
                    self.exporter.layer1_owner_fixture(receipt)

    def test_changed_selected_index_storage_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        record = receipt["steps"][1]["intermediates"]["layers.1.attn.compressed"][
            "indices"
        ]
        record["storage_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exact storage"):
            self.exporter.layer1_owner_fixture(receipt)

    def test_consistently_hashed_nonfinite_fp32_parameter_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        tensor = receipt["encoded_parameters"]["layers.1.attn.compressor.wkv.weight"]
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[-4:] = bytes.fromhex("0000807f")
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertTrue(tensor["finite"])
        with self.assertRaisesRegex(RuntimeError, "nonfinite FP32"):
            self.exporter.layer1_owner_fixture(receipt)

    def test_layer_one_wrappers_restore_after_source_failure(self) -> None:
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
            attention = model.layers[1].attn
            indexer = attention.indexer
            self.assertIsNotNone(indexer)
            bindings = {
                (attention, "_window_kv"): attention.__dict__.get("_window_kv"),
                (attention, "_compress_kv"): attention.__dict__.get("_compress_kv"),
                (indexer, "forward"): indexer.__dict__.get("forward"),
            }
            existed = {
                (module, name): name in module.__dict__ for module, name in bindings
            }
            with (
                patch.object(
                    attention.compressor.wgate,
                    "forward",
                    side_effect=RuntimeError("injected layer-one gate failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "injected layer-one gate failure"),
                runner.v41_forward_observers.hooks_for(
                    model,
                    graph,
                    runner.tensor_record,
                    runner.object_record,
                    runner.MAX_HOOK_RECORDS,
                ),
            ):
                model(torch.tensor([runner.TRACE_INPUT_IDS[0][:5]]), start_pos=0)
            for (module, name), original in bindings.items():
                self.assertEqual(name in module.__dict__, existed[(module, name)])
                self.assertIs(module.__dict__.get(name), original)


if __name__ == "__main__":
    unittest.main()
