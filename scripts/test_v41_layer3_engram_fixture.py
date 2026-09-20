#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.13.0",
#   "numpy==2.5.3",
#   "sympy==1.14.0",
#   "tokenizers==0.23.2",
# ]
# ///
"""Source-backed integrity and rejection tests for the layer-three Engram fixture."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
FIXTURE = SCRIPTS.parent / "fixtures/deepseek-v41/layer3-engram-reference.json"


def load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerThreeEngramFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = load_module("v41-forward-reference.py", "v41_engram_runner")
        cls.exporter = load_module(
            "v41_layer3_engram_capture.py", "v41_engram_exporter"
        )
        cls.receipt = cls.runner.run_capture()
        cls.fixture = cls.exporter.engram_fixture(cls.receipt)

    def test_committed_fixture_matches_current_source_values(self) -> None:
        """The historical fixture pins values, not later additive hook hashes."""
        committed = json.loads(FIXTURE.read_text())
        self.assertEqual(
            {key: value for key, value in committed.items() if key != "source"},
            {key: value for key, value in self.fixture.items() if key != "source"},
        )
        stable = (
            "revision",
            "model_sha256",
            "engram_sha256",
            "kernel_source_sha256",
            "cpu_backend_sha256",
            "loader_sha256",
            "runner_sha256",
            "storage_byteorder",
        )
        self.assertEqual(
            {key: committed["source"][key] for key in stable},
            {key: self.fixture["source"][key] for key in stable},
        )
        # Layer-two instrumentation intentionally changes both these receipt
        # identities while preserving the historical Engram numerical oracle.
        for key in ("forward_observers_sha256", "complete_capture_sha256"):
            self.assertRegex(self.fixture["source"][key], r"^[0-9a-f]{64}$")
            self.assertNotEqual(committed["source"][key], self.fixture["source"][key])

    def test_wkv_split_and_block_entry_are_exact(self) -> None:
        for case in self.fixture["cases"]:
            wkv = bytes.fromhex(case["wkv_output"]["storage_hex"])
            key = bytes.fromhex(case["key"]["storage_hex"])
            value = bytes.fromhex(case["value"]["storage_hex"])
            rebuilt = b"".join(
                key[offset : offset + 512] + value[index * 256 : (index + 1) * 256]
                for index, offset in enumerate(range(0, len(key), 512))
            )
            self.assertEqual(rebuilt, wkv)
            self.assertEqual(
                case["output"]["storage_sha256"],
                case["block_entry"]["storage_sha256"],
            )

    def test_missing_engram_projection_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        del receipt["steps"][0]["intermediates"]["layers.3.engram.wkv"]
        with self.assertRaisesRegex(TypeError, "WKV"):
            self.exporter.engram_fixture(receipt)

    def test_changed_engram_output_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["steps"][0]["intermediates"]["layers.3.engram"]["storage_sha256"] = (
            "0" * 64
        )
        with self.assertRaisesRegex(RuntimeError, "exact storage"):
            self.exporter.engram_fixture(receipt)

    def test_changed_hash_state_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["engram"]["hash_state"]["primes"]["finite"] = False
        with self.assertRaisesRegex(RuntimeError, "invalid finite"):
            self.exporter.engram_fixture(receipt)

    def test_consistently_hashed_nonfinite_parameter_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        tensor = receipt["encoded_parameters"]["layers.3.engram.embed.weight"]
        raw = bytearray.fromhex(tensor["storage_hex"])
        raw[-1] = 0x7F
        tensor["storage_hex"] = raw.hex()
        tensor["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        self.assertTrue(tensor["finite"])
        with self.assertRaisesRegex(RuntimeError, "nonfinite storage"):
            self.exporter.engram_fixture(receipt)

    def test_numeric_storage_rejects_nonfinite_encodings(self) -> None:
        for dtype, invalid, valid in (
            ("torch.float8_e4m3fn", ["7f", "ff"], ["00", "80", "01", "7e", "fe"]),
            ("torch.float8_e8m0fnu", ["ff"], ["00", "01", "fe"]),
            (
                "torch.bfloat16",
                ["807f", "80ff", "c07f"],
                ["0000", "0080", "0100", "7f7f"],
            ),
            (
                "torch.float32",
                ["0000807f", "000080ff", "0100807f"],
                ["00000000", "00000080", "01000000", "ffff7f7f"],
            ),
        ):
            for storage in invalid + valid:
                with self.subTest(dtype=dtype, storage=storage):
                    raw = bytes.fromhex(storage)
                    tensor = {
                        "dtype": dtype,
                        "shape": [1],
                        "numel": 1,
                        "finite": True,
                        "storage_hex": storage,
                        "storage_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                    if storage in invalid:
                        with self.assertRaisesRegex(RuntimeError, "nonfinite storage"):
                            self.exporter._require_tensor(
                                tensor, "control", dtype=dtype, shape=[1]
                            )
                    else:
                        self.assertEqual(
                            self.exporter._require_tensor(
                                tensor, "control", dtype=dtype, shape=[1]
                            ),
                            tensor,
                        )


if __name__ == "__main__":
    unittest.main()
