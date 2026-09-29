"""Persisted controls for alternate L1 owner-to-attention boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

import v41_partition_layer1_capture as capture
from v41_partition_owner_capture import CaptureError


class PartitionLayerOneCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "fixtures/deepseek-v41/partition-layer1-reference.json"
            ).read_text()
        )

    @staticmethod
    def _mutate_tensor(tensor: dict[str, Any]) -> None:
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[0] ^= 1
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()

    def test_validates_observed_owner_attention_schedule(self) -> None:
        capture.validate_layer1_join(self.fixture)
        self.assertEqual(
            [(case["start_pos"], case["sequence"]) for case in self.fixture["cases"]],
            [(0, 4), (4, 1), (5, 1), (6, 1)],
        )
        self.assertEqual(
            [case["compressed_prefix"] for case in self.fixture["cases"]],
            [2, 2, 3, 3],
        )
        self.assertEqual(
            [case["offset"] for case in self.fixture["cases"]],
            [4, 6, 6, 6],
        )

    def test_rejects_detached_owner_attention_boundaries(self) -> None:
        paths = (
            ("input",),
            ("compressed_kv",),
            ("compressed_indices",),
            ("indexer", "output_indices"),
        )
        for path in paths:
            with self.subTest(path=path):
                fixture = copy.deepcopy(self.fixture)
                tensor = fixture["attention"]["cases"][1]
                for key in path:
                    tensor = tensor[key]
                self._mutate_tensor(tensor)
                with self.assertRaises(CaptureError):
                    capture.validate_layer1_join(fixture)

    def test_rejects_detached_join_provenance(self) -> None:
        mutations = (
            ("source", "revision", "0" * 40),
            ("source_receipt_sha256", None, "0" * 64),
            ("capture_identity", "sha256", "0" * 64),
            ("runtime", "storage_byteorder", "big"),
        )
        for section, key, replacement in mutations:
            with self.subTest(section=section, key=key):
                fixture = copy.deepcopy(self.fixture)
                target = fixture["attention"]
                if key is None:
                    target[section] = replacement
                else:
                    target[section][key] = replacement
                with self.assertRaises(CaptureError):
                    capture.validate_layer1_join(fixture)

    def test_rejects_matching_absent_provenance(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        del fixture["source_receipt_sha256"]
        del fixture["attention"]["source_receipt_sha256"]
        with self.assertRaises(CaptureError):
            capture.validate_layer1_join(fixture)

    def test_rejects_changed_partial_prefix_retention(self) -> None:
        for case_index, field in ((1, "index_key_prefix"), (3, "compressed_kv_prefix")):
            with self.subTest(case_index=case_index, field=field):
                fixture = copy.deepcopy(self.fixture)
                self._mutate_tensor(fixture["cases"][case_index][field])
                with self.assertRaises(CaptureError):
                    capture.validate_layer1_join(fixture)

    def test_distinguishes_own_and_prior_score_prefixes(self) -> None:
        capture.validate_layer1_join(self.fixture)
        own = (0, 2)
        prior = (1, 3)
        for index in own:
            case = self.fixture["cases"][index]
            self.assertEqual(case["index_score_key_prefix"], case["index_key_prefix"])
        for index in prior:
            case = self.fixture["cases"][index]
            self.assertNotEqual(
                case["index_score_key_prefix"], case["index_key_prefix"]
            )

        fixture = copy.deepcopy(self.fixture)
        fixture["cases"][1]["index_score_key_prefix"] = copy.deepcopy(
            fixture["cases"][1]["index_key_prefix"]
        )
        with self.assertRaises(CaptureError):
            capture.validate_layer1_join(fixture)

        fixture = copy.deepcopy(self.fixture)
        fixture["cases"][0]["index_score_key_prefix"] = copy.deepcopy(
            fixture["cases"][2]["index_key_prefix"]
        )
        with self.assertRaises(CaptureError):
            capture.validate_layer1_join(fixture)


if __name__ == "__main__":
    unittest.main()
