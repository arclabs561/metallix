"""Persisted controls for alternate L2-to-L3 Engram source handoffs."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

from v41_partition_layer3_engram_capture import (
    join_alternate_calls,
    validate_layer3_engram_join,
)
from v41_partition_owner_capture import CaptureError


class LayerThreeEngramPartitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parent.parent / "fixtures/deepseek-v41"
        cls.layer_two = json.loads(
            (root / "partition-layer2-reference.json").read_text()
        )
        cls.fixture = json.loads(
            (root / "partition-layer3-engram-reference.json").read_text()
        )

    @staticmethod
    def change(tensor: dict) -> None:
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[0] ^= 1
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()

    def test_persisted_upstream_and_block_boundaries(self) -> None:
        validate_layer3_engram_join(self.layer_two, self.fixture)
        self.assertEqual(
            [case["start_pos"] for case in self.fixture["cases"]], [0, 4, 5, 6]
        )
        self.assertEqual(
            [case["input_ids"]["shape"] for case in self.fixture["cases"]],
            [[1, 4], [1, 1], [1, 1], [1, 1]],
        )

    def test_rejects_changed_native_upstream_or_stream_or_block(self) -> None:
        for field in ("stream", "output", "block_entry"):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                self.change(fixture["cases"][1][field])
                with self.assertRaises(CaptureError):
                    validate_layer3_engram_join(self.layer_two, fixture)
        upstream = copy.deepcopy(self.layer_two)
        self.change(upstream["ffn"]["cases"][2]["output"])
        with self.assertRaises(CaptureError):
            validate_layer3_engram_join(upstream, self.fixture)

    def test_rejects_missing_or_detached_provenance(self) -> None:
        for field in ("source", "source_receipt_sha256", "capture_identity"):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                del fixture[field]
                with self.assertRaises(CaptureError):
                    validate_layer3_engram_join(self.layer_two, fixture)
        fixture = copy.deepcopy(self.fixture)
        fixture["source"]["revision"] = "0" * 40
        with self.assertRaises(CaptureError):
            validate_layer3_engram_join(self.layer_two, fixture)

    def test_rejects_missing_and_reordered_calls(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        fixture["cases"].pop()
        with self.assertRaises(CaptureError):
            validate_layer3_engram_join(self.layer_two, fixture)
        fixture = copy.deepcopy(self.fixture)
        fixture["cases"][1], fixture["cases"][2] = (
            fixture["cases"][2],
            fixture["cases"][1],
        )
        with self.assertRaises(CaptureError):
            validate_layer3_engram_join(self.layer_two, fixture)

    def test_observed_ids_join_only_matching_capture_coordinates(self) -> None:
        observed = [
            {
                "start_pos": case["start_pos"],
                "token_count": case["input_ids"]["shape"][1],
                "input_ids": case["input_ids"],
            }
            for case in self.fixture["cases"]
        ]
        captured = [
            {key: value for key, value in call.items() if key != "input_ids"}
            for call in observed
        ]
        alternate = {"calls": observed, "alternate_capture": {"calls": captured}}
        joined = join_alternate_calls(alternate)
        self.assertEqual(
            [case["input_ids"] for case in joined],
            [case["input_ids"] for case in observed],
        )
        for field in ("start_pos", "token_count"):
            changed = copy.deepcopy(alternate)
            changed["alternate_capture"]["calls"][1][field] += 1
            with self.assertRaises(CaptureError):
                join_alternate_calls(changed)
        changed = copy.deepcopy(alternate)
        del changed["calls"][1]["input_ids"]
        with self.assertRaises(CaptureError):
            join_alternate_calls(changed)


if __name__ == "__main__":
    unittest.main()
