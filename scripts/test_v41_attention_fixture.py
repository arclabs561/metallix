"""Dependency-free structural checks for the source-derived attention fixture."""

from __future__ import annotations

import hashlib
import json
import math
import struct
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
            "forward_observers_sha256",
            "attention_helper_sha256",
            "complete_capture_sha256",
            "manifest_canonical_sha256",
        ):
            self.assertRegex(source.get(name, ""), r"^[0-9a-f]{64}$", name)
        self.assertEqual(
            source["forward_observers_sha256"],
            hashlib.sha256(
                (ROOT / "scripts" / "v41_forward_observers.py").read_bytes()
            ).hexdigest(),
        )
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

    def test_indexer_boundaries_retain_exact_source_storage(self) -> None:
        fixture = self.read_fixture()
        cases = fixture["cases"]
        for case in cases:
            start_pos = case["start_pos"]
            indexer = case.get("indexer")
            self.assertIsInstance(indexer, dict, start_pos)
            inputs = indexer.get("inputs")
            operations = indexer.get("operations")
            self.assertIsInstance(inputs, dict, start_pos)
            self.assertIsInstance(operations, dict, start_pos)
            self.assertIsNone(inputs.get("latent"), start_pos)
            self.assertEqual(inputs.get("start_pos"), start_pos)
            self.assert_tensor(inputs.get("x"), "indexer x", "torch.bfloat16", 2)
            self.assert_tensor(inputs.get("qr"), "indexer qr", "torch.bfloat16", 2)
            self.assert_tensor(
                inputs.get("shared_index_k_prefix"),
                "source-published index K prefix",
                "torch.bfloat16",
                2,
            )
            self.assert_tensor(
                inputs.get("candidate_mask"),
                "source candidate mask",
                "torch.bool",
                1,
            )
            prefix_shape = inputs["shared_index_k_prefix"]["shape"]
            self.assertEqual(prefix_shape[-1], 64)
            candidate_shape = inputs["candidate_mask"]["shape"]
            self.assertEqual(candidate_shape[0], 1)
            self.assertEqual(candidate_shape[-1], prefix_shape[1])
            for name in (
                "weights_proj_output",
                "scaled_weights",
                "q_after_rope_fp4",
                "scores_einsum",
                "scores_after_relu",
                "scores_weighted_per_head",
                "scores_after_head_sum",
                "scores_after_candidate_mask",
            ):
                self.assert_tensor(operations.get(name), name, "torch.bfloat16", 2)
            if start_pos == 0:
                self.assert_tensor(
                    operations.get("scores_after_causal_mask"),
                    "scores_after_causal_mask",
                    "torch.bfloat16",
                    2,
                )
            else:
                self.assertNotIn("scores_after_causal_mask", operations)
            self.assertFalse(operations["scores_after_candidate_mask"]["finite"])
            self.assert_tensor(
                indexer.get("output_indices"),
                "source Indexer output indices",
                "torch.int32",
                4,
            )

    def test_masked_score_previews_match_exact_bf16_storage(self) -> None:
        self.assert_masked_previews(self.read_fixture())

    def assert_masked_previews(self, fixture: dict[str, object]) -> None:
        saw_masked_value = False
        for case in fixture["cases"]:
            record = case["indexer"]["operations"]["scores_after_candidate_mask"]
            raw = bytes.fromhex(record["storage_hex"])
            previews = record["sample_f32"]
            self.assertEqual(len(previews), min(record["numel"], 8))
            for position, preview in enumerate(previews):
                word = raw[2 * position : 2 * position + 2]
                value = struct.unpack("<f", b"\x00\x00" + word)[0]
                if math.isfinite(value):
                    self.assertEqual(preview, value)
                else:
                    # The pinned masks contain negative infinity, not NaN or
                    # positive infinity. Its JSON preview must retain that sign.
                    self.assertEqual(value, -math.inf)
                    self.assertEqual(preview, "-inf")
                    saw_masked_value = True
            json.dumps(record, allow_nan=False)
        self.assertTrue(saw_masked_value, "exercise a real source-masked score")

    def test_nonfinite_numeric_and_wrong_sign_previews_are_rejected(self) -> None:
        fixture = self.read_fixture()
        previews = fixture["cases"][0]["indexer"]["operations"][
            "scores_after_candidate_mask"
        ]["sample_f32"]
        masked_position = previews.index("-inf")
        for replacement in (-math.inf, "inf"):
            with self.subTest(replacement=replacement):
                previews[masked_position] = replacement
                with self.assertRaises(AssertionError):
                    self.assert_masked_previews(fixture)

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
