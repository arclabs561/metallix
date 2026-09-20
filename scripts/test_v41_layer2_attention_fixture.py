#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "numpy==2.5.3", "sympy==1.14.0", "tokenizers==0.23.2"]
# ///
"""Source-backed checks for layer-two attention's borrowed owner publication."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerTwoAttentionFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load("v41-forward-reference.py", "v41_layer2_attention_runner")
        cls.exporter = load(
            "v41_layer2_attention_capture.py", "v41_layer2_attention_exporter"
        )
        cls.receipt = cls.runner.run_capture()
        projected = cls.exporter.attention_fixture(
            cls.receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
        )
        cls.fixture = json.loads(json.dumps(projected, sort_keys=True, allow_nan=False))

    def test_committed_fixture_matches_current_source_capture(self) -> None:
        committed = json.loads(
            (
                SCRIPTS.parent / "fixtures/deepseek-v41/layer2-attention-reference.json"
            ).read_text()
        )
        self.assertEqual(
            {key: value for key, value in committed.items() if key != "source"},
            {key: value for key, value in self.fixture.items() if key != "source"},
        )
        changing = {"forward_observers_sha256", "complete_capture_sha256"}
        for field in changing:
            self.assertRegex(self.fixture["source"][field], r"^[0-9a-f]{64}$")
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

    def test_layer_two_borrows_exact_current_layer_one_publication(self) -> None:
        for case in self.fixture["cases"]:
            self.assertEqual(
                case["compressed_kv"]["storage_sha256"],
                case["layer_one_published_kv"]["storage_sha256"],
            )
            self.assertEqual(
                case["compressed_indices"]["storage_sha256"],
                case["layer_one_published_indices"]["storage_sha256"],
            )
            self.assertNotIn("indexer", case)

    def test_missing_owner_publication_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        del receipt["steps"][0]["intermediates"]["layers.1.attn.compressed"]
        with self.assertRaisesRegex(TypeError, "layer-one KV/index publication"):
            self.exporter.attention_fixture(
                receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
            )

    def test_wrong_sparse_layer_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        for call in receipt["steps"][0]["sparse_attention_calls"]:
            if call["layer_id"] == 2:
                call["layer_id"] = 99
        with self.assertRaisesRegex(RuntimeError, "layer-two sparse"):
            self.exporter.attention_fixture(
                receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
            )

    def test_layer_two_handoff_matches_fixed_ffn_oracle(self) -> None:
        ffn = json.loads(
            (
                SCRIPTS.parent / "fixtures/deepseek-v41/layer2-ffn-reference.json"
            ).read_text()
        )
        for attention, tail in zip(self.fixture["cases"], ffn["cases"], strict=True):
            self.assertEqual(
                attention["after_attention_residual"]["storage_sha256"],
                tail["after_attention_residual"]["storage_sha256"],
            )
            self.assertEqual(
                attention["attention_pre"]["storage_sha256"],
                tail["attention_pre"]["storage_sha256"],
            )

    def test_wrong_but_hashed_owner_publication_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        record = receipt["steps"][0]["intermediates"]["layers.2.attn.compressed"][
            "borrowed_kv"
        ]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] ^= 1
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "sparse KV"):
            self.exporter.attention_fixture(
                receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
            )

    def test_wrong_revision_and_forged_finite_are_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["source"]["revision"] = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "unexpected source revision"):
            self.exporter.attention_fixture(
                receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
            )
        receipt = copy.deepcopy(self.receipt)
        record = receipt["steps"][0]["intermediates"]["layers.2.attn.wq_a"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[:2] = bytes.fromhex("807f")
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        record["finite"] = True
        with self.assertRaisesRegex(RuntimeError, "nonfinite storage"):
            self.exporter.attention_fixture(
                receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
            )

    def test_paired_forged_owner_publications_are_rejected_at_sparse_boundary(
        self,
    ) -> None:
        receipt = copy.deepcopy(self.receipt)
        for path in (
            ("layers.1.attn.compressed", "borrowed_kv"),
            ("layers.2.attn.compressed", "borrowed_kv"),
        ):
            record = receipt["steps"][0]["intermediates"][path[0]][path[1]]
            raw = bytearray.fromhex(record["storage_hex"])
            raw[0] ^= 1
            record["storage_hex"] = raw.hex()
            record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "sparse KV"):
            self.exporter.attention_fixture(
                receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
            )

    def test_source_schedule_and_boolean_dimensions_are_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["model_args"]["index_source_layers"] = [1, 2, 4]
        with self.assertRaisesRegex(RuntimeError, "Indexer schedule"):
            self.exporter.attention_fixture(
                receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
            )
        receipt = copy.deepcopy(self.receipt)
        record = receipt["steps"][0]["intermediates"]["layers.2.attn.wq_a"]
        record["shape"][0] = True
        with self.assertRaisesRegex(TypeError, "invalid tensor metadata"):
            self.exporter.attention_fixture(
                receipt, helper_path=SCRIPTS / "v41_layer2_attention_capture.py"
            )


if __name__ == "__main__":
    unittest.main()
