#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "numpy==2.5.3", "sympy==1.14.0", "tokenizers==0.23.2"]
# ///
"""Source-backed integrity checks for layer-one ratio-two attention."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
FIXTURE = SCRIPTS.parent / "fixtures/deepseek-v41/layer1-attention-reference.json"


def load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerOneAttentionFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load("v41-forward-reference.py", "v41_layer1_attention_runner")
        cls.exporter = load(
            "v41_layer1_attention_capture.py", "v41_layer1_attention_exporter"
        )
        cls.capture = cls.runner.run_capture()
        projected = cls.exporter.attention_fixture(
            cls.capture, helper_path=SCRIPTS / "v41_layer1_attention_capture.py"
        )
        cls.fixture = json.loads(json.dumps(projected, sort_keys=True, allow_nan=False))

    def test_committed_fixture_matches_current_source_capture(self) -> None:
        self.assertEqual(json.loads(FIXTURE.read_text()), self.fixture)

    def test_owner_selected_ids_reach_compressed_attention(self) -> None:
        for case in self.fixture["cases"]:
            self.assertEqual(
                case["compressed_indices"]["storage_sha256"],
                case["indexer"]["output_indices"]["storage_sha256"],
            )

    def test_changed_owner_ids_are_rejected(self) -> None:
        capture = copy.deepcopy(self.capture)
        record = capture["steps"][0]["intermediates"]["layers.1.attn.compressed"][
            "indices"
        ]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] ^= 1
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "selected indices"):
            self.exporter.attention_fixture(
                capture, helper_path=SCRIPTS / "v41_layer1_attention_capture.py"
            )

    def test_changed_sparse_kv_is_rejected(self) -> None:
        capture = copy.deepcopy(self.capture)
        call = next(
            call
            for call in capture["steps"][0]["sparse_attention_calls"]
            if call["layer_id"] == 1
        )
        record = call["inputs"]["kv"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] ^= 1
        record["storage_hex"] = raw.hex()
        with self.assertRaisesRegex(RuntimeError, "exact storage"):
            self.exporter.attention_fixture(
                capture, helper_path=SCRIPTS / "v41_layer1_attention_capture.py"
            )

    def test_changed_parameter_and_nonfinite_payload_are_rejected(self) -> None:
        capture = copy.deepcopy(self.capture)
        record = capture["encoded_parameters"]["layers.1.attn.wq_a.weight"]
        record["storage_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exact storage"):
            self.exporter.attention_fixture(
                capture, helper_path=SCRIPTS / "v41_layer1_attention_capture.py"
            )
        capture = copy.deepcopy(self.capture)
        record = capture["steps"][0]["intermediates"]["layers.1.attn.wq_a"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[:2] = bytes.fromhex("807f")
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        record["finite"] = True
        with self.assertRaisesRegex(RuntimeError, "nonfinite storage"):
            self.exporter.attention_fixture(
                capture, helper_path=SCRIPTS / "v41_layer1_attention_capture.py"
            )

    def test_boolean_shape_metadata_is_rejected(self) -> None:
        capture = copy.deepcopy(self.capture)
        record = capture["steps"][0]["intermediates"]["layers.1.attn.wq_a"]
        record["shape"][0] = True
        with self.assertRaisesRegex(TypeError, "invalid tensor metadata"):
            self.exporter.attention_fixture(
                capture, helper_path=SCRIPTS / "v41_layer1_attention_capture.py"
            )


if __name__ == "__main__":
    unittest.main()
