"""Dependency-free structural tests for the candidate-producer fixture exporter."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from v41_candidate_capture import candidate_fixture


def _tensor(
    dtype: str, shape: list[int], byte_width: int, *, fill: int = 0, finite: bool = True
) -> dict[str, object]:
    numel = 1
    for width in shape:
        numel *= width
    raw = bytes([fill]) * (numel * byte_width)
    return {
        "dtype": dtype,
        "finite": finite,
        "numel": numel,
        "shape": shape,
        "storage_hex": raw.hex(),
        "storage_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _negative_infinity_bf16(shape: list[int]) -> dict[str, object]:
    numel = 1
    for width in shape:
        numel *= width
    raw = b"\x80\xff" * numel
    return {
        "dtype": "torch.bfloat16",
        "finite": False,
        "numel": numel,
        "shape": shape,
        "storage_hex": raw.hex(),
        "storage_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _receipt() -> dict[str, object]:
    source_hash = "a" * 64
    parameters = {
        "layers.3.attn.wq_a.weight": _tensor("torch.float8_e4m3fn", [32, 128], 1),
        "layers.3.attn.wq_a.scale": _tensor("torch.float8_e8m0fnu", [1, 4], 1),
        "layers.3.attn.q_norm.weight": _tensor("torch.bfloat16", [32], 2),
        "layers.3.attn.indexer.wq_b.weight": _tensor(
            "torch.float8_e4m3fn", [128, 32], 1
        ),
        "layers.3.attn.indexer.wq_b.scale": _tensor("torch.float8_e8m0fnu", [4, 1], 1),
        "layers.3.attn.indexer.weights_proj.weight": _tensor(
            "torch.bfloat16", [2, 128], 2
        ),
    }
    steps = []
    for start_pos, sequence, offset in ((0, 5, 5), (5, 1, 6), (6, 1, 6)):
        end_pos = start_pos + sequence
        input_x = _tensor("torch.bfloat16", [1, sequence, 128], 2, fill=start_pos + 1)
        qr = _tensor("torch.bfloat16", [1, sequence, 32], 2, fill=start_pos + 2)
        latent = _tensor("torch.bfloat16", [1, sequence, 64], 2, fill=start_pos + 3)
        score_shape = [1, sequence, 2, end_pos]
        reduced_shape = [1, sequence, end_pos]
        operations = {
            "q_after_rope_fp4": _tensor("torch.bfloat16", [1, sequence, 2, 64], 2),
            "k_after_rope_fp4": _tensor("torch.bfloat16", [1, sequence, 64], 2),
            "weights_proj_output": _tensor("torch.bfloat16", [1, sequence, 2], 2),
            "scaled_weights": _tensor("torch.bfloat16", [1, sequence, 2], 2),
            "scores_einsum": _tensor("torch.bfloat16", score_shape, 2),
            "scores_after_relu": _tensor("torch.bfloat16", score_shape, 2),
            "scores_weighted_per_head": _tensor("torch.bfloat16", score_shape, 2),
            "scores_after_head_sum": _tensor("torch.bfloat16", reduced_shape, 2),
        }
        if start_pos == 0:
            operations["scores_after_causal_mask"] = _negative_infinity_bf16(
                reduced_shape
            )
        steps.append(
            {
                "start_pos": start_pos,
                "intermediates": {
                    "layers.3.attention_input": input_x,
                    "layers.3.attn.wq_a": _tensor(
                        "torch.bfloat16", [1, sequence, 32], 2, fill=start_pos + 4
                    ),
                    "layers.3.attn.q_norm": qr,
                    "layers.3.attn.indexer_observation": {
                        "inputs": {
                            "x": input_x,
                            "qr": qr,
                            "latent": latent,
                            "start_pos": start_pos,
                            "offset": offset,
                            "shared_index_k_prefix": _tensor(
                                "torch.bfloat16", [1, end_pos, 64], 2
                            ),
                        },
                        "operations": operations,
                        "candidate_mask_after": _tensor(
                            "torch.bool", reduced_shape, 1, fill=1
                        ),
                        "output_indices": _tensor("torch.int32", [1, sequence, 1], 4),
                    },
                },
            }
        )
    return {
        "capture_status": "completed synthetic source-forward capture; no parity claim",
        "coverage_status": {"pending": []},
        "source": {
            "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
            "model_sha256": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
            "kernel_source_sha256": source_hash,
            "runner_sha256": source_hash,
            "forward_observers_sha256": source_hash,
        },
        "runtime": {"storage_byteorder": "little"},
        "manifest_canonical_sha256": "b" * 64,
        "model_args": {
            "max_batch_size": 1,
            "max_seq_len": 8,
            "dim": 128,
            "head_dim": 64,
            "q_lora_rank": 32,
            "rope_head_dim": 32,
            "index_n_heads": 2,
            "index_head_dim": 64,
            "candidate_source_layer": 3,
            "candidate_topk_blocks": 2,
            "candidate_block_size": 1,
            "index_topk": 1,
            "window_size": 6,
            "compress_ratios": [0, 2, 2, 1, 1],
            "index_source_layers": [1, 3, 4],
            "kv_source_layers": [1, 3],
            "norm_eps": 1e-20,
        },
        "attention_static": {
            "layer_4_freqs_cis": _tensor("torch.complex64", [8, 16], 8)
        },
        "encoded_parameters": parameters,
        "steps": steps,
    }


class CandidateCaptureTest(unittest.TestCase):
    def test_checked_in_fixture_identity_and_storage(self) -> None:
        root = Path(__file__).resolve().parent.parent
        fixture = json.loads(
            (
                root / "fixtures/deepseek-v41/forward-candidate-reference.json"
            ).read_text()
        )
        self.assertEqual(
            fixture["source"]["complete_capture_sha256"],
            "7c5cc8541da338fa3426d63e32b9a66e9132e07ab68ee26d86fbf9e29f62f48d",
        )
        self.assertEqual(
            fixture["source"]["forward_observers_sha256"],
            hashlib.sha256(
                (root / "scripts/v41_forward_observers.py").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            [case["inputs"]["offset"] for case in fixture["cases"]], [5, 6, 6]
        )
        self.assert_storage_tree(fixture)

    def assert_storage_tree(self, value: object) -> None:
        if isinstance(value, dict):
            if "storage_hex" in value:
                raw = bytes.fromhex(value["storage_hex"])
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), value["storage_sha256"]
                )
                width = {
                    "torch.bfloat16": 2,
                    "torch.float8_e4m3fn": 1,
                    "torch.float8_e8m0fnu": 1,
                    "torch.int32": 4,
                    "torch.complex64": 8,
                    "torch.bool": 1,
                }[value["dtype"]]
                self.assertEqual(len(raw), value["numel"] * width)
            for child in value.values():
                self.assert_storage_tree(child)
        elif isinstance(value, list):
            for child in value:
                self.assert_storage_tree(child)

    def test_extracts_pinned_candidate_boundaries(self) -> None:
        fixture = candidate_fixture(_receipt())
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(fixture["model"]["owner_layer"], 3)
        self.assertEqual(fixture["frequencies"]["shape"], [8, 16])
        self.assertEqual(
            list(fixture["encoded_parameters"]),
            [
                "layers.3.attn.wq_a.weight",
                "layers.3.attn.wq_a.scale",
                "layers.3.attn.q_norm.weight",
                "layers.3.attn.indexer.wq_b.weight",
                "layers.3.attn.indexer.wq_b.scale",
                "layers.3.attn.indexer.weights_proj.weight",
            ],
        )
        self.assertEqual([case["start_pos"] for case in fixture["cases"]], [0, 5, 6])
        self.assertIn("scores_after_causal_mask", fixture["cases"][0]["operations"])
        self.assertNotIn("scores_after_causal_mask", fixture["cases"][1]["operations"])
        self.assertEqual(
            [case["candidate_mask"]["shape"] for case in fixture["cases"]],
            [[1, 5, 5], [1, 1, 6], [1, 1, 7]],
        )

    def test_rejects_wrong_boundary_missing_observed_mask(self) -> None:
        receipt = _receipt()
        del receipt["steps"][0]["intermediates"]["layers.3.attn.indexer_observation"][
            "candidate_mask_after"
        ]
        with self.assertRaisesRegex(TypeError, "produced candidate mask"):
            candidate_fixture(receipt)

    def test_rejects_nonfinite_score_and_invalid_boolean_storage(self) -> None:
        receipt = _receipt()
        record = receipt["steps"][1]["intermediates"][
            "layers.3.attn.indexer_observation"
        ]["operations"]["scores_einsum"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[:2] = b"\x80\x7f"
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "finite BF16"):
            candidate_fixture(receipt)
        receipt = _receipt()
        record = receipt["steps"][1]["intermediates"][
            "layers.3.attn.indexer_observation"
        ]["candidate_mask_after"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] = 2
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "boolean 0/1"):
            candidate_fixture(receipt)

    def test_rejects_reserved_fp8_nan_codes(self) -> None:
        receipt = _receipt()
        record = receipt["encoded_parameters"]["layers.3.attn.wq_a.weight"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] = 0x7F
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "finite E4M3FN"):
            candidate_fixture(receipt)
        receipt = _receipt()
        record = receipt["encoded_parameters"]["layers.3.attn.wq_a.scale"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] = 0xFF
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "finite E8M0FNU"):
            candidate_fixture(receipt)

    def test_rejects_causal_stage_labeled_nonfinite_without_negative_infinity(
        self,
    ) -> None:
        receipt = _receipt()
        record = receipt["steps"][0]["intermediates"][
            "layers.3.attn.indexer_observation"
        ]["operations"]["scores_after_causal_mask"]
        raw = b"\x00\x00" * record["numel"]
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "negative infinity"):
            candidate_fixture(receipt)

    def test_rejects_valid_but_wrong_provenance_and_offsets(self) -> None:
        receipt = _receipt()
        receipt["source"]["revision"] = "f" * 40
        with self.assertRaisesRegex(RuntimeError, "pinned source revision"):
            candidate_fixture(receipt)
        receipt = _receipt()
        receipt["steps"][2]["intermediates"]["layers.3.attn.indexer_observation"][
            "inputs"
        ]["offset"] = 5
        with self.assertRaisesRegex(RuntimeError, "invalid call offsets"):
            candidate_fixture(receipt)


if __name__ == "__main__":
    unittest.main()
