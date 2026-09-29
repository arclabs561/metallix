"""Persisted controls for alternate L1-to-L2 and L2-to-Engram boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

import v41_partition_layer2_capture as capture
from v41_partition_owner_capture import CaptureError


class PartitionLayerTwoCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures = Path(__file__).resolve().parent.parent / "fixtures/deepseek-v41"
        cls.layer_one = json.loads(
            (fixtures / "partition-layer1-reference.json").read_text()
        )
        cls.fixture = json.loads(
            (fixtures / "partition-layer2-reference.json").read_text()
        )

    @staticmethod
    def _mutate_tensor(tensor: dict[str, Any]) -> None:
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[0] ^= 1
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()

    def test_validates_persisted_l1_l2_schedule(self) -> None:
        capture.validate_layer2_join(self.layer_one, self.fixture)
        for projection in (
            self.fixture["hc"],
            self.fixture["attention"],
            self.fixture["ffn"],
        ):
            self.assertEqual(
                [case["start_pos"] for case in projection["cases"]], [0, 4, 5, 6]
            )
        self.assertEqual(
            self.fixture["attention"]["runtime"], {"storage_byteorder": "little"}
        )

    def test_rejects_detached_l1_tail_to_l2_hc(self) -> None:
        for field in ("output", "next_pre"):
            with self.subTest(field=field):
                layer_one = copy.deepcopy(self.layer_one)
                self._mutate_tensor(layer_one["tail"]["cases"][1][field])
                with self.assertRaises(CaptureError):
                    capture.validate_layer2_join(layer_one, self.fixture)

    def test_rejects_detached_borrowed_l1_publication(self) -> None:
        for field in (
            "layer_one_published_kv",
            "layer_one_published_indices",
            "compressed_kv",
            "compressed_indices",
        ):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                self._mutate_tensor(fixture["attention"]["cases"][2][field])
                with self.assertRaises(CaptureError):
                    capture.validate_layer2_join(self.layer_one, fixture)

    def test_rejects_detached_l2_hc_attention_and_ffn_handoffs(self) -> None:
        mutations = (
            ("attention", "cases", 1, "input"),
            ("attention", "cases", 1, "output"),
            ("hc", "cases", 1, "after_attention_residual"),
            ("ffn", "cases", 1, "attention_pre"),
        )
        for section, cases, index, field in mutations:
            with self.subTest(section=section, field=field):
                fixture = copy.deepcopy(self.fixture)
                self._mutate_tensor(fixture[section][cases][index][field])
                with self.assertRaises(CaptureError):
                    capture.validate_layer2_join(self.layer_one, fixture)

    def test_rejects_detached_l2_terminal_to_l3_entry(self) -> None:
        for field in ("output", "next_pre"):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                self._mutate_tensor(fixture["ffn"]["cases"][3][field])
                with self.assertRaises(CaptureError):
                    capture.validate_layer2_join(self.layer_one, fixture)

    def test_rejects_detached_or_absent_provenance(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["attention"]["source"]["revision"] = "0" * 40
        with self.assertRaises(CaptureError):
            capture.validate_layer2_join(self.layer_one, fixture)

        fixture = copy.deepcopy(self.fixture)
        del fixture["source_receipt_sha256"]
        for child in (fixture["hc"], fixture["attention"], fixture["ffn"]):
            del child["source_receipt_sha256"]
        with self.assertRaises(CaptureError):
            capture.validate_layer2_join(self.layer_one, fixture)

        layer_one = copy.deepcopy(self.layer_one)
        del layer_one["capture_identity"]
        with self.assertRaises(CaptureError):
            capture.validate_layer2_join(layer_one, self.fixture)


if __name__ == "__main__":
    unittest.main()
