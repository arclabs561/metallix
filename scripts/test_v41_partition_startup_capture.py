#!/usr/bin/env python3
"""Controls for joining alternate source calls to startup token records."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import unittest
from pathlib import Path
from typing import Any

import v41_partition_startup_capture as startup
from v41_partition_owner_capture import CaptureError


class PartitionStartupCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "fixtures/deepseek-v41/partition-startup-reference.json"
            ).read_text()
        )

    def alternate(self) -> tuple[dict[str, Any], list[int]]:
        observed = []
        captured = []
        token_ids = []
        for ordinal, case in enumerate(self.fixture["cases"]):
            ids = copy.deepcopy(case["startup"]["input_ids"])
            count = ids["numel"]
            token_ids.extend(
                value
                for (value,) in struct.iter_unpack(
                    "<q", bytes.fromhex(ids["storage_hex"])
                )
            )
            observed.append(
                {
                    "start_pos": case["start_pos"],
                    "token_count": count,
                    "input_ids": ids,
                }
            )
            captured.append(
                {
                    "start_pos": case["start_pos"],
                    "token_count": count,
                    "capture_ordinal": ordinal,
                }
            )
        return {"calls": observed, "alternate_capture": {"calls": captured}}, token_ids

    def test_joins_fixture_startup_ids_to_observed_capture_calls(self) -> None:
        alternate, token_ids = self.alternate()

        joined = startup.join_startup_calls(alternate, token_ids)

        self.assertEqual([call["capture_ordinal"] for call in joined], [0, 1, 2, 3])
        self.assertEqual(
            [call["input_ids"]["storage_sha256"] for call in joined],
            [
                case["startup"]["input_ids"]["storage_sha256"]
                for case in self.fixture["cases"]
            ],
        )
        self.assertEqual(token_ids, list(range(7)))

    def test_rejects_swapped_capture_calls(self) -> None:
        alternate, token_ids = self.alternate()
        (
            alternate["alternate_capture"]["calls"][1],
            alternate["alternate_capture"]["calls"][2],
        ) = (
            alternate["alternate_capture"]["calls"][2],
            alternate["alternate_capture"]["calls"][1],
        )

        with self.assertRaises(CaptureError):
            startup.join_startup_calls(alternate, token_ids)

    def test_rejects_mismatched_or_boolean_call_counts(self) -> None:
        for section, replacement in (("calls", 2), ("alternate_capture", True)):
            with self.subTest(section=section, replacement=replacement):
                alternate, token_ids = self.alternate()
                if section == "calls":
                    alternate["calls"][1]["token_count"] = replacement
                else:
                    alternate["alternate_capture"]["calls"][1]["token_count"] = (
                        replacement
                    )

                with self.assertRaises(CaptureError):
                    startup.join_startup_calls(alternate, token_ids)

    def test_rejects_changed_startup_token_storage_even_with_recomputed_hash(
        self,
    ) -> None:
        alternate, token_ids = self.alternate()
        ids = alternate["calls"][1]["input_ids"]
        raw = bytearray.fromhex(ids["storage_hex"])
        raw[0] ^= 1
        ids["storage_hex"] = raw.hex()
        ids["storage_sha256"] = hashlib.sha256(raw).hexdigest()

        with self.assertRaises(CaptureError):
            startup.join_startup_calls(alternate, token_ids)

    def test_rejects_corrupted_startup_token_hash(self) -> None:
        alternate, token_ids = self.alternate()
        alternate["calls"][1]["input_ids"]["storage_sha256"] = "0" * 64

        with self.assertRaises(CaptureError):
            startup.join_startup_calls(alternate, token_ids)

    def test_rejects_omitted_observed_call(self) -> None:
        alternate, token_ids = self.alternate()
        alternate["calls"].pop()

        with self.assertRaises(CaptureError):
            startup.join_startup_calls(alternate, token_ids)


if __name__ == "__main__":
    unittest.main()
