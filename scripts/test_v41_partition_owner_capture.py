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

    def test_rejects_relabelled_capture_identity(self) -> None:
        mutations = (
            ("source", "probe_sha256", "0" * 64),
            ("source", "cpu_backend_sha256", "0" * 64),
            ("source", "observer_sha256", "0" * 64),
            ("capture_identity", "probe_sha256", "0" * 64),
            ("capture_identity", "sha256", "0" * 64),
            ("capture_identity", "schedule", [4, True, 1, 1]),
            ("model", "batches", True),
        )
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                fixture = copy.deepcopy(self.fixture)
                fixture[section][key] = value
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)
        for key, value in (
            ("source_receipt_sha256", "g" * 64),
            ("schema_version", True),
        ):
            with self.subTest(key=key):
                fixture = copy.deepcopy(self.fixture)
                fixture[key] = value
                with self.assertRaises(capture.CaptureError):
                    capture.validate_fixture(fixture)

    def test_selection_geometry_and_causal_boundary(self) -> None:
        capture.validate_fixture(self.fixture)
        self.assertEqual(
            [case["selection"]["offset"] for case in self.fixture["cases"]],
            [4, 6, 6, 6],
        )
        for case in self.fixture["cases"]:
            selection = case["selection"]
            self.assertEqual(selection["candidate_mask"]["dtype"], "torch.bool")
            self.assertEqual(selection["indices"]["shape"], [1, case["token_count"], 1])
            if case["start_pos"] == 0:
                self.assertFalse(selection["causal_scores"]["finite"])
            else:
                self.assertIsNone(selection["causal_scores"])

    def test_rejects_malformed_selection_operands(self) -> None:
        for defect in (
            "offset",
            "mask",
            "fp8",
            "causal_nan",
            "query_shape",
            "model",
            "model_unit",
        ):
            with self.subTest(defect=defect):
                fixture = copy.deepcopy(self.fixture)
                selection = fixture["cases"][0]["selection"]
                if defect == "offset":
                    selection["offset"] = 5
                elif defect == "query_shape":
                    selection["qr"]["shape"] = [1, 5, 32]
                elif defect == "model":
                    fixture["selection_model"]["index_heads"] = True
                elif defect == "model_unit":
                    fixture["selection_model"]["candidate_block_size"] = True
                else:
                    if defect == "mask":
                        tensor, replacement = selection["candidate_mask"], b"\x02"
                    elif defect == "fp8":
                        tensor, replacement = (
                            fixture["selection_weights"]["wq_a_codes"],
                            b"\x7f",
                        )
                    else:
                        tensor, replacement = selection["causal_scores"], b"\xc0\x7f"
                    raw = (
                        replacement
                        + bytes.fromhex(tensor["storage_hex"])[len(replacement) :]
                    )
                    tensor["storage_hex"] = raw.hex()
                    tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    capture.validate_fixture(fixture)

    def test_attention_window_and_publication_boundaries(self) -> None:
        capture.validate_fixture(self.fixture)
        for case in self.fixture["cases"]:
            attention = case["attention"]
            self.assertEqual(attention["start_pos"], case["start_pos"])
            self.assertEqual(
                attention["output"]["shape"], [1, case["token_count"], 128]
            )
            self.assertEqual(attention["window_ring_after"]["shape"], [1, 6, 64])
            self.assertEqual(
                attention["compressed_indices"]["storage_sha256"],
                case["selection"]["indices"]["storage_sha256"],
            )
            self.assertEqual(
                attention["compressed_kv"]["storage_sha256"],
                case["compressed_kv_prefix"]["storage_sha256"],
            )

    def test_rejects_detached_attention_inputs_and_publications(self) -> None:
        for field in ("input", "compressed_kv", "compressed_indices"):
            with self.subTest(field=field):
                fixture = copy.deepcopy(self.fixture)
                tensor = fixture["cases"][1]["attention"][field]
                raw = bytearray.fromhex(tensor["storage_hex"])
                raw[0] ^= 1
                tensor["storage_hex"] = raw.hex()
                tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
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
