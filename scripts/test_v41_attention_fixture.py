"""Dependency-free structural checks for the source-derived attention fixture."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "deepseek-v41" / "forward-attention-reference.json"
CASE_TENSORS = {
    "input": ("torch.bfloat16", 2),
    "wq_a_output": ("torch.bfloat16", 2),
    "q_norm_output": ("torch.bfloat16", 2),
    "wq_b_pre_rope": ("torch.bfloat16", 2),
    "q_after_rope": ("torch.bfloat16", 2),
    "window_kv": ("torch.bfloat16", 2),
    "prepared_window_kv": ("torch.bfloat16", 2),
    "window_indices": ("torch.int32", 4),
    "window_ring_after": ("torch.bfloat16", 2),
    "compressed_kv": ("torch.bfloat16", 2),
    "compressed_indices": ("torch.int32", 4),
    "sparse_output_pre_inverse_rope": ("torch.bfloat16", 2),
    "wo_b_input": ("torch.bfloat16", 2),
    "output": ("torch.bfloat16", 2),
}


class AttentionFixtureTest(unittest.TestCase):
    def read_fixture(self) -> dict[str, object]:
        self.assertTrue(FIXTURE.is_file(), f"missing generated fixture: {FIXTURE}")
        result = json.loads(FIXTURE.read_text())
        self.assertIsInstance(result, dict)
        return result

    def assert_tensor(
        self, record: object, label: str, expected_dtype: str, byte_width: int
    ) -> None:
        self.assertIsInstance(record, dict, label)
        self.assertEqual(record.get("dtype"), expected_dtype, label)
        shape = record.get("shape")
        numel = record.get("numel")
        storage_hex = record.get("storage_hex")
        self.assertIsInstance(shape, list, label)
        self.assertTrue(
            all(isinstance(width, int) and width > 0 for width in shape), label
        )
        self.assertIsInstance(numel, int, label)
        self.assertEqual(numel, _product(shape), label)
        self.assertIsInstance(storage_hex, str, label)
        raw = bytes.fromhex(storage_hex)
        self.assertEqual(len(raw), numel * byte_width, label)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), record.get("storage_sha256"), label
        )

    def test_layer_four_source_boundaries_are_complete(self) -> None:
        fixture = self.read_fixture()
        self.assertEqual(fixture.get("schema_version"), 1)
        self.assertIn("source attention boundaries", fixture.get("scope", ""))
        source = fixture.get("source")
        self.assertIsInstance(source, dict)
        self.assertEqual(
            source.get("revision"), "dba1be0a40aa45a94ad051997016db3960a90277"
        )
        for name in (
            "model_sha256",
            "kernel_source_sha256",
            "cpu_backend_sha256",
            "loader_sha256",
            "runner_sha256",
            "attention_helper_sha256",
            "complete_capture_sha256",
            "manifest_canonical_sha256",
        ):
            self.assertRegex(source.get(name, ""), r"^[0-9a-f]{64}$", name)
        self.assertEqual(source.get("storage_byteorder"), "little")

        model = fixture.get("model")
        self.assertIsInstance(model, dict)
        for name in ("o_groups", "o_lora_rank", "window_size", "compress_ratios"):
            self.assertIn(name, model)
        self.assertEqual(model["o_groups"], model["n_heads"])

        frequencies = fixture.get("frequencies")
        self.assertIsInstance(frequencies, dict)
        self.assertEqual(frequencies.get("complex_dtype"), "torch.complex64")
        self.assertEqual(
            _product(frequencies.get("shape")), len(frequencies.get("fp32_pairs", []))
        )
        for pair in frequencies["fp32_pairs"]:
            self.assertEqual(len(pair), 2)
            self.assertTrue(
                all(isinstance(bits, int) and 0 <= bits < 2**32 for bits in pair)
            )

        parameters = fixture.get("encoded_parameters")
        self.assertIsInstance(parameters, dict)
        for name in (
            "layers.4.attn.attn_sink",
            "layers.4.attn.wq_a.weight",
            "layers.4.attn.wq_a.scale",
            "layers.4.attn.wq_b.weight",
            "layers.4.attn.wq_b.scale",
            "layers.4.attn.wkv.weight",
            "layers.4.attn.wkv.scale",
            "layers.4.attn.wo_a.weight",
            "layers.4.attn.wo_b.weight",
            "layers.4.attn.wo_b.scale",
        ):
            self.assertIn(name, parameters)
        byte_widths = {
            "torch.bfloat16": 2,
            "torch.float32": 4,
            "torch.float8_e4m3fn": 1,
            "torch.float8_e8m0fnu": 1,
        }
        for name, record in parameters.items():
            dtype = record.get("dtype")
            self.assertIn(dtype, byte_widths, name)
            self.assert_tensor(record, name, dtype, byte_widths[dtype])

        cases = fixture.get("cases")
        self.assertIsInstance(cases, list)
        self.assertEqual([case.get("start_pos") for case in cases], [0, 5, 6])
        for case in cases:
            for name, (dtype, width) in CASE_TENSORS.items():
                self.assert_tensor(case.get(name), name, dtype, width)

    def test_changed_tensor_bytes_fail_integrity_check(self) -> None:
        fixture = self.read_fixture()
        record = dict(fixture["cases"][0]["input"])
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] ^= 1
        record["storage_hex"] = raw.hex()
        with self.assertRaises(AssertionError):
            self.assert_tensor(record, "corrupted input", "torch.bfloat16", 2)


def _product(shape: object) -> int:
    if not isinstance(shape, list):
        return -1
    result = 1
    for width in shape:
        if not isinstance(width, int):
            return -1
        result *= width
    return result


if __name__ == "__main__":
    unittest.main()
