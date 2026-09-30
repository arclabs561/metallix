#!/usr/bin/env python3
"""Check partition comparison against frozen source tensors and corruptions."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import v41_partition_probe as probe


class PartitionProbeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bundle = json.loads(
            (
                Path(__file__).resolve().parent.parent
                / "fixtures/deepseek-v41/reduced-runner-reference.json"
            ).read_text()
        )
        head = bundle["projections"]["head"]["cases"][-1]
        raw = struct.pack(
            f"<{len(head['logits_fp32_bits'])}I", *head["logits_fp32_bits"]
        )
        cls.baseline = {
            "terminal_logits": {
                "dtype": "torch.float32",
                "shape": head["logits_shape"],
                "numel": len(head["logits_fp32_bits"]),
                "storage_hex": raw.hex(),
                "storage_sha256": hashlib.sha256(raw).hexdigest(),
                "finite": True,
            },
            "cache_after": {
                "layer_3.compressed_kv": bundle["projections"]["layer3_attention"][
                    "cases"
                ][-1]["compressed_kv"],
            },
        }

    @staticmethod
    def replace_word(run: dict, word: int) -> None:
        tensor = run["terminal_logits"]
        raw = struct.pack("<I", word) + bytes.fromhex(tensor["storage_hex"])[4:]
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()

    def test_frozen_tensor_identity_and_changed_logit(self) -> None:
        comparison = probe.compare_runs(self.baseline, copy.deepcopy(self.baseline))
        self.assertTrue(comparison["terminal"]["exact_bits"])
        self.assertEqual(comparison["terminal"]["max_abs_f32"], 0)
        changed = copy.deepcopy(self.baseline)
        first = struct.unpack(
            "<I", bytes.fromhex(changed["terminal_logits"]["storage_hex"])[:4]
        )[0]
        self.replace_word(changed, first + 1)
        comparison = probe.compare_runs(self.baseline, changed)
        self.assertFalse(comparison["terminal"]["exact_bits"])
        self.assertGreater(comparison["terminal"]["max_abs_f32"], 0)

    def test_signed_zero_is_not_bit_parity(self) -> None:
        positive, negative = copy.deepcopy(self.baseline), copy.deepcopy(self.baseline)
        self.replace_word(positive, 0)
        self.replace_word(negative, 0x80000000)
        comparison = probe.compare_runs(positive, negative)
        self.assertFalse(comparison["terminal"]["exact_bits"])
        self.assertEqual(comparison["terminal"]["max_abs_f32"], 0)
        self.assertEqual(comparison["terminal"]["signed_zero_bit_changes"], 1)

    def test_invalid_terminal_cannot_report_parity(self) -> None:
        for defect in ("absent", "nonfinite", "shape", "storage", "hash"):
            with self.subTest(defect=defect):
                changed = copy.deepcopy(self.baseline)
                tensor = changed["terminal_logits"]
                if defect == "absent":
                    del changed["terminal_logits"]
                elif defect == "nonfinite":
                    self.replace_word(changed, 0x7F800000)
                elif defect == "shape":
                    tensor["shape"] = [tensor["numel"]]
                elif defect == "storage":
                    tensor["storage_hex"] = ""
                elif defect == "hash":
                    tensor["storage_sha256"] = "0" * 64
                with self.assertRaises(probe.ProbeError):
                    probe.compare_runs(self.baseline, changed)

    def test_cache_changes_are_separate_from_logit_parity(self) -> None:
        changed = copy.deepcopy(self.baseline)
        tensor = changed["cache_after"]["layer_3.compressed_kv"]
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[0] ^= 1
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        comparison = probe.compare_runs(self.baseline, changed)
        self.assertTrue(comparison["terminal"]["exact_bits"])
        self.assertEqual(
            comparison["cache"]["changed_fields"], ["layer_3.compressed_kv"]
        )

    def test_common_endpoints_align_by_tokens_not_call_index(self) -> None:
        def run(counts):
            start = 0
            calls = []
            for count in counts:
                calls.append(
                    {
                        "start_pos": start,
                        "token_count": count,
                        "logits": self.baseline["terminal_logits"],
                        "cache_after": self.baseline["cache_after"],
                    }
                )
                start += count
            return {"calls": calls}

        comparisons = probe.compare_common_endpoints(run((5, 1, 1)), run((4, 1, 1, 1)))
        self.assertEqual([item["tokens_processed"] for item in comparisons], [5, 6, 7])
        self.assertTrue(all(item["terminal"]["exact_bits"] for item in comparisons))

    def test_observer_noninterference_requires_each_call_to_match(self) -> None:
        def run() -> dict:
            return {
                "calls": [
                    {
                        "start_pos": start,
                        "token_count": count,
                        "logits": copy.deepcopy(self.baseline["terminal_logits"]),
                        "cache_after": copy.deepcopy(self.baseline["cache_after"]),
                    }
                    for start, count in ((0, 4), (4, 1), (5, 1), (6, 1))
                ]
            }

        observed, control = run(), run()
        result = probe.observer_noninterference(observed, control)
        self.assertTrue(result["exact_noninterference"])
        self.assertEqual(len(result["per_call"]), 4)

        changed_logit = run()
        self.replace_word({"terminal_logits": changed_logit["calls"][2]["logits"]}, 0)
        result = probe.observer_noninterference(changed_logit, control)
        self.assertFalse(result["exact_noninterference"])
        self.assertFalse(result["per_call"][2]["comparison"]["terminal"]["exact_bits"])

        changed_cache = run()
        tensor = changed_cache["calls"][1]["cache_after"]["layer_3.compressed_kv"]
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[0] ^= 1
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        result = probe.observer_noninterference(changed_cache, control)
        self.assertFalse(result["exact_noninterference"])
        self.assertEqual(
            result["per_call"][1]["comparison"]["cache"]["changed_fields"],
            ["layer_3.compressed_kv"],
        )

    def test_observer_noninterference_rejects_different_geometry(self) -> None:
        observed = {
            "calls": [
                {
                    "start_pos": 0,
                    "token_count": 4,
                    "logits": self.baseline["terminal_logits"],
                    "cache_after": self.baseline["cache_after"],
                }
            ]
        }
        control = copy.deepcopy(observed)
        control["calls"][0]["token_count"] = 5
        with self.assertRaises(probe.ProbeError):
            probe.observer_noninterference(observed, control)

    def test_failed_execution_is_not_success_and_receipts_are_not_overwritten(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            failed = {
                "status": "source_execution_failed",
                "failed_schedule": "alternate",
            }
            with patch.object(probe, "build_probe", return_value=failed):
                self.assertEqual(probe.main(["--run", "--output", str(path)]), 1)
            self.assertEqual(json.loads(path.read_text()), failed)
            original = path.read_bytes()
            with patch.object(
                probe,
                "build_probe",
                return_value={"status": "completed_source_partition_experiment"},
            ):
                self.assertEqual(probe.main(["--run", "--output", str(path)]), 1)
            self.assertEqual(path.read_bytes(), original)

    def test_partition_is_positive_integer_exact_cover(self) -> None:
        self.assertEqual(probe.validate_schedule((5, 1, 1)), (5, 1, 1))
        self.assertEqual(probe.validate_schedule((4, 1, 1, 1)), (4, 1, 1, 1))
        for invalid in ((), (6,), (8,), (0, 7), (-1, 8), (True, 6), (4.0, 1, 1, 1)):
            with self.subTest(schedule=invalid), self.assertRaises(probe.ProbeError):
                probe.validate_schedule(invalid)

    def test_alternate_prefill_sweep_is_bounded_and_default_compatible(self) -> None:
        self.assertEqual(probe.alternate_schedule(), probe.ALTERNATE)
        self.assertEqual(probe.alternate_schedule(2), (2, 1, 1, 1, 1, 1))
        self.assertEqual(probe.alternate_schedule(7), (7,))
        for invalid in (True, False, 1, 8, 2.0, "4", None):
            with (
                self.subTest(prefill_tokens=invalid),
                self.assertRaises(probe.ProbeError),
            ):
                probe.alternate_schedule(invalid)

    def test_cli_forwards_alternate_prefill_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            with patch.object(
                probe,
                "build_probe",
                return_value={"status": "completed_source_partition_experiment"},
            ) as build:
                self.assertEqual(
                    probe.main(
                        [
                            "--run",
                            "--output",
                            str(path),
                            "--prefill-tokens",
                            "3",
                        ]
                    ),
                    0,
                )
            build.assert_called_once_with(capture_alternate=False, prefill_tokens=3)


if __name__ == "__main__":
    unittest.main()
