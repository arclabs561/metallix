#!/usr/bin/env python3
"""Counterfactual controls for the observed four-call native-owner operands."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

import v41_partition_owner_capture as capture


class PartitionOwnerCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "fixtures/deepseek-v41/partition-owner-reference.json"
            ).read_text()
        )

    def test_observed_owner_schedule_and_partial_consumers(self) -> None:
        capture.validate_fixture(self.fixture)
        self.assertEqual(
            [(c["start_pos"], c["token_count"]) for c in self.fixture["cases"]],
            [(0, 4), (4, 1), (5, 1), (6, 1)],
        )
        self.assertEqual(
            [c["index_key_prefix"]["shape"] for c in self.fixture["cases"]],
            [[1, n, 64] for n in (4, 5, 6, 7)],
        )
        for case, count in zip(self.fixture["cases"], (2, None, 3, None), strict=True):
            prefix = case["next_layer1_score_prefix"]
            if count is None:
                self.assertIsNone(prefix)
            else:
                self.assertEqual(prefix["shape"], [1, count, 64])
                self.assertTrue(
                    case["index_key_prefix"]["storage_hex"].startswith(
                        prefix["storage_hex"]
                    )
                )

    def test_rejects_malformed_geometry_and_storage(self) -> None:
        for defect in ("shape", "hash", "nonfinite", "schedule", "count", "source"):
            with self.subTest(defect=defect):
                fixture = copy.deepcopy(self.fixture)
                tensor = fixture["cases"][1]["input"]
                if defect == "shape":
                    tensor["shape"] = [128]
                elif defect == "hash":
                    tensor["storage_sha256"] = "0" * 64
                elif defect == "nonfinite":
                    raw = b"\x80\x7f" + bytes.fromhex(tensor["storage_hex"])[2:]
                    tensor["storage_hex"] = raw.hex()
                    tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                elif defect == "schedule":
                    fixture["cases"][1]["start_pos"] = 5
                elif defect == "count":
                    fixture["cases"][1]["token_count"] = True
                else:
                    fixture["source"]["model_sha256"] = "0" * 64
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    capture.validate_fixture(fixture)

    def test_rejects_substituted_partial_consumer(self) -> None:
        fixture = copy.deepcopy(self.fixture)
        tensor = fixture["cases"][0]["next_layer1_score_prefix"]
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[0] ^= 1
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises((ValueError, RuntimeError)):
            capture.validate_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
