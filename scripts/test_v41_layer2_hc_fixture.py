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

S = Path(__file__).parent
F = S.parent / "fixtures/deepseek-v41/layer2-hc-reference.json"


def load(n, x):
    s = importlib.util.spec_from_file_location(n, S / x)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


class T(unittest.TestCase):
    @classmethod
    def setUpClass(c):
        c.r = load("r", "v41-forward-reference.py")
        c.e = load("e", "v41_layer2_hc_capture.py")
        c.receipt = c.r.run_capture()
        c.fixture = json.loads(json.dumps(c.e.layer2_hc_fixture(c.receipt)))

    def test_fixture(self):
        self.assertEqual(json.loads(F.read_text()), self.fixture)

    def test_two_copy_contract_is_pinned(self):
        self.assertEqual(self.fixture["block_config"]["copies"], 2)
        for case in self.fixture["cases"]:
            sequence = 5 if case["start_pos"] == 0 else 1
            self.assertEqual(case["residual"]["shape"], [1, sequence, 2, 128])

    def test_matches_fixed_attention_input(self):
        attention = json.loads(
            (
                S.parent / "fixtures/deepseek-v41/layer2-attention-reference.json"
            ).read_text()
        )
        for hc, source_attention in zip(
            self.fixture["cases"], attention["cases"], strict=True
        ):
            self.assertEqual(
                hc["attention_input"]["storage_sha256"],
                source_attention["input"]["storage_sha256"],
            )

    def test_matches_ffn_handoff(self):
        f = json.loads(
            (S.parent / "fixtures/deepseek-v41/layer2-ffn-reference.json").read_text()
        )
        for a, b in zip(self.fixture["cases"], f["cases"], strict=True):
            self.assertEqual(
                a["after_attention_residual"]["storage_sha256"],
                b["after_attention_residual"]["storage_sha256"],
            )
            self.assertEqual(
                a["attention_pre"]["storage_sha256"],
                b["attention_pre"]["storage_sha256"],
            )

    def test_bad_seam(self):
        x = copy.deepcopy(self.receipt)
        r = x["steps"][0]["intermediates"]["layers.2.block_input"]["residual"]
        raw = bytearray.fromhex(r["storage_hex"])
        raw[0] ^= 1
        r["storage_hex"] = raw.hex()
        r["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "does not feed"):
            self.e.layer2_hc_fixture(x)

    def test_runtime_metadata_hc_parameter_and_configuration_are_rejected(self):
        x = copy.deepcopy(self.receipt)
        x["runtime"]["storage_byteorder"] = "big"
        with self.assertRaisesRegex(RuntimeError, "little-endian"):
            self.e.layer2_hc_fixture(x)
        x = copy.deepcopy(self.receipt)
        x["steps"][0]["intermediates"]["layers.1"][0]["shape"][0] = True
        with self.assertRaisesRegex(TypeError, "invalid tensor metadata"):
            self.e.layer2_hc_fixture(x)
        x = copy.deepcopy(self.receipt)
        call = next(
            call
            for call in x["steps"][0]["hyper_connection_mixes"]
            if call["layer_id"] == 2 and call["sublayer"] == "attention"
        )
        call["inputs"]["hc_scale"]["storage_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "exact storage"):
            self.e.layer2_hc_fixture(x)
        x = copy.deepcopy(self.receipt)
        call = next(
            call
            for call in x["steps"][0]["hyper_connection_mixes"]
            if call["layer_id"] == 2 and call["sublayer"] == "attention"
        )
        call["inputs"]["hc_mult"] = 3
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            self.e.layer2_hc_fixture(x)

    def test_bad_source(self):
        x = copy.deepcopy(self.receipt)
        x["source"]["revision"] = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "unexpected source revision"):
            self.e.layer2_hc_fixture(x)


if __name__ == "__main__":
    unittest.main()
