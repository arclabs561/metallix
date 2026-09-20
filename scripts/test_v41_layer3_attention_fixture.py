"""Integrity checks for the distinct layer-three source-attention fixture."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import v41_layer3_attention_capture as capture

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures/deepseek-v41/forward-layer3-attention-reference.json"
)


class LayerThreeAttentionFixtureTest(unittest.TestCase):
    def routing_receipt(self) -> dict:
        # Routing sentinels test aliases only; the committed fixture supplies
        # all numerical oracles. These are deliberately not tensor records.
        boundaries = (
            "attention_input",
            "attn.wq_a",
            "attn.q_norm",
            "attn.wq_b",
            "attn.window",
            "attn.compressed",
            "attn.indexer_observation",
            "attn.wo_b_input",
            "attn",
        )
        return {
            "encoded_parameters": {
                f"layers.{layer}.attn.{name}": f"layer-{layer}-{name}"
                for layer in (3, 4)
                for name in capture.REQUIRED_PARAMETERS
            },
            "attention_static": {
                "layer_3_freqs_cis": "three",
                "layer_4_freqs_cis": "four",
            },
            "steps": [
                {
                    "intermediates": {
                        f"layers.{layer}.{name}": f"layer-{layer}-{name}"
                        for layer in (3, 4)
                        for name in boundaries
                    },
                    "sparse_attention_calls": [
                        {"layer_id": 3, "origin": "three"},
                        {"layer_id": 4, "origin": "four"},
                    ],
                }
            ],
        }

    def test_projection_routes_only_layer_three_operands_without_mutating_input(self):
        source = self.routing_receipt()
        mapped = capture._receipt_for_generic_projector(source)
        self.assertEqual(mapped["attention_static"]["layer_4_freqs_cis"], "three")
        self.assertEqual(source["attention_static"]["layer_4_freqs_cis"], "four")
        for name in capture.REQUIRED_PARAMETERS:
            self.assertEqual(
                mapped["encoded_parameters"][f"layers.4.attn.{name}"], f"layer-3-{name}"
            )
        self.assertEqual(
            mapped["steps"][0]["sparse_attention_calls"],
            [{"layer_id": 4, "origin": "three"}],
        )
        self.assertEqual(
            mapped["steps"][0]["intermediates"]["layers.4.attn"], "layer-3-attn"
        )

    def test_missing_producer_operands_never_fall_back_to_consumer(self):
        for kind in ("frequency", "parameter", "boundary", "duplicate_call"):
            source = self.routing_receipt()
            if kind == "frequency":
                del source["attention_static"]["layer_3_freqs_cis"]
            elif kind == "parameter":
                del source["encoded_parameters"]["layers.3.attn.wq_a.weight"]
            elif kind == "boundary":
                del source["steps"][0]["intermediates"]["layers.3.attn"]
            else:
                source["steps"][0]["sparse_attention_calls"].append({"layer_id": 3})
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                capture._receipt_for_generic_projector(source)

    def fixture(self) -> dict[str, object]:
        value = json.loads(FIXTURE.read_text())
        self.assertEqual(value["schema_version"], 1)
        self.assertIn("layer-three source attention", value["scope"])
        source = value["source"]
        self.assertEqual(
            source["complete_capture_sha256"],
            "87201aa0c04f6887bd8329f1f0fe7206d29cf54f60e192962c2ad0ca8dcdd740",
        )
        self.assertEqual(
            source["frequency_source"], "attention_static.layer_3_freqs_cis"
        )
        self.assertIn("layer-three", value["frequency_scope"])
        self.assertFalse(
            any("layers.4." in name for name in value["encoded_parameters"])
        )
        self.assertEqual(
            source["forward_observers_sha256"],
            "27462b9190a1cf7a9771a68a87a9b3e0af0d4c87cab677b6d94219b032f42d60",
        )
        return value

    def tensor(self, record: dict[str, object], label: str) -> None:
        self.assertIn(record["dtype"], {"torch.bfloat16", "torch.int32"}, label)
        raw = bytes.fromhex(record["storage_hex"])
        width = 2 if record["dtype"] == "torch.bfloat16" else 4
        self.assertEqual(len(raw), record["numel"] * width, label)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), record["storage_sha256"], label
        )

    def test_source_boundaries_are_complete(self) -> None:
        fixture = self.fixture()
        parameters = fixture["encoded_parameters"]
        for suffix in (
            "wq_a.weight",
            "q_norm.weight",
            "wq_b.weight",
            "wkv.weight",
            "kv_norm.weight",
            "wo_a.weight",
            "wo_b.weight",
        ):
            self.assertIn(f"layers.3.attn.{suffix}", parameters)
        self.assertEqual([case["start_pos"] for case in fixture["cases"]], [0, 5, 6])
        for case in fixture["cases"]:
            for name in (
                "input",
                "prepared_window_kv",
                "window_kv",
                "compressed_kv",
                "sparse_output_pre_inverse_rope",
                "wo_b_input",
                "output",
                "window_indices",
                "compressed_indices",
            ):
                self.tensor(case[name], name)

    def test_changed_source_storage_is_rejected(self) -> None:
        fixture = self.fixture()
        record = dict(fixture["cases"][0]["output"])
        record["storage_hex"] = "00" + record["storage_hex"][2:]
        with self.assertRaises(AssertionError):
            self.tensor(record, "mutated output")


if __name__ == "__main__":
    unittest.main()
