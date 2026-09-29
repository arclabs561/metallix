#!/usr/bin/env python3
"""Structural controls for the compact source partition bridge receipt."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path

import v41_partition_boundaries as boundaries

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "fixtures"
    / "deepseek-v41"
    / "partition-bridge-reference.json"
)


class PartitionBoundariesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text())

    def fresh(self) -> dict:
        return copy.deepcopy(self.fixture)

    @staticmethod
    def tensor(capture: dict, call: int, path: tuple[str, ...]) -> dict:
        value: object = capture["calls"][call]
        for key in path:
            value = value[key]
        assert isinstance(value, dict)
        return value

    @staticmethod
    def flip_byte(tensor: dict) -> None:
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[0] ^= 1
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()

    def assert_rejected(self, capture: dict) -> None:
        with self.assertRaises(ValueError):
            boundaries.validate_bridges(capture)

    def test_actual_compact_bridge_receipt_is_healthy(self) -> None:
        result = boundaries.validate_bridges(self.fresh())
        self.assertEqual(result["call_count"], 4)
        self.assertEqual(
            [item["start_pos"] for item in result["layer1_to_layer2"]],
            [0, 4, 5, 6],
        )

    def test_each_named_bridge_rejects_changed_source_bytes(self) -> None:
        paths = {
            "layer1_kv": (
                0,
                (
                    "intermediates",
                    "layers.2.attn.compressed",
                    "borrowed_kv",
                ),
            ),
            "layer1_indices": (
                1,
                (
                    "intermediates",
                    "layers.2.attn.compressed",
                    "indices",
                ),
            ),
            "candidate_mask": (
                2,
                (
                    "intermediates",
                    "layers.4.attn.indexer_observation",
                    "inputs",
                    "candidate_mask",
                ),
            ),
            "shared_index_keys": (
                3,
                (
                    "intermediates",
                    "layers.4.attn.indexer_observation",
                    "inputs",
                    "shared_index_k_prefix",
                ),
            ),
            "prior_layer3_prefix": (
                3,
                (
                    "intermediates",
                    "layers.1.attn.indexer_observation",
                    "inputs",
                    "shared_index_k_prefix",
                ),
            ),
        }
        for name, (call, path) in paths.items():
            with self.subTest(name=name):
                capture = self.fresh()
                self.flip_byte(self.tensor(capture, call, path))
                self.assert_rejected(capture)

    def test_malformed_tensor_metadata_and_nonfinite_flag_lie_are_rejected(
        self,
    ) -> None:
        path = (
            "intermediates",
            "layers.2.attn.compressed",
            "borrowed_kv",
        )
        for defect in ("storage", "hash", "shape", "dtype", "nonfinite"):
            with self.subTest(defect=defect):
                capture = self.fresh()
                tensor = self.tensor(capture, 0, path)
                if defect == "storage":
                    tensor["storage_hex"] = ""
                elif defect == "hash":
                    tensor["storage_sha256"] = "0" * 64
                elif defect == "shape":
                    tensor["shape"] = [1]
                elif defect == "dtype":
                    tensor["dtype"] = "torch.float32"
                else:
                    raw = bytearray.fromhex(tensor["storage_hex"])
                    raw[:2] = b"\x80\xff"
                    tensor["storage_hex"] = raw.hex()
                    tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
                    tensor["finite"] = True
                self.assert_rejected(capture)

    def test_schedule_rejects_boolean_and_float_values(self) -> None:
        for call, field, value in ((0, "start_pos", True), (1, "token_count", 1.0)):
            with self.subTest(field=field, value=value):
                capture = self.fresh()
                capture["calls"][call][field] = value
                self.assert_rejected(capture)

    def test_layer_four_selected_ids_are_not_aliases_of_layer_three(self) -> None:
        capture = self.fresh()
        for call in capture["calls"][:3]:
            l3 = call["intermediates"]["layers.3.attn.compressed"]["indices"]
            l4 = call["intermediates"]["layers.4.attn.compressed"]["indices"]
            self.assertNotEqual(
                l3["storage_sha256"],
                l4["storage_sha256"],
                "source capture confirms L4 performs its own selection",
            )
        # A bridge validator must accept this healthy source receipt without
        # imposing L3's producer-selected IDs as an L4 consumer oracle.
        self.assertEqual(boundaries.validate_bridges(capture)["call_count"], 4)

    def test_projection_requires_completed_observer_control_and_is_reproducible(
        self,
    ) -> None:
        receipt = {
            "status": "completed_source_partition_experiment",
            "observer_noninterference": {
                "exact_noninterference": True,
                "per_call": [{"exact_noninterference": True} for _ in range(4)],
            },
            "alternate": {"alternate_capture": self.fresh()},
            "source": self.fixture["source"],
            "baseline_oracle": self.fixture["baseline_oracle"],
        }
        raw = json.dumps(receipt).encode()
        projected = boundaries.project_bridges(raw)
        self.assertEqual(
            projected["source_receipt_sha256"], hashlib.sha256(raw).hexdigest()
        )
        self.assertEqual(projected["calls"], self.fixture["calls"])
        self.assertEqual(boundaries.validate_bridges(projected)["call_count"], 4)

        receipt["status"] = "source_execution_failed"
        with self.assertRaises(ValueError):
            boundaries.project_bridges(json.dumps(receipt).encode())
        receipt["status"] = "completed_source_partition_experiment"
        receipt["observer_noninterference"]["per_call"][2]["exact_noninterference"] = (
            False
        )
        with self.assertRaises(ValueError):
            boundaries.project_bridges(json.dumps(receipt).encode())


if __name__ == "__main__":
    unittest.main()
