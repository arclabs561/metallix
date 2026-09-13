"""Dependency-free parser checks for the layer-three index-key fixture exporter."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from v41_index_key_capture import compressor_fixture, index_key_fixture


def _tensor(
    dtype: str, shape: list[int], byte_width: int, fill: int = 0
) -> dict[str, object]:
    numel = 1
    for width in shape:
        numel *= width
    raw = bytes([fill]) * (numel * byte_width)
    return {
        "dtype": dtype,
        "finite": True,
        "numel": numel,
        "shape": shape,
        "storage_hex": raw.hex(),
        "storage_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _receipt() -> dict[str, object]:
    source_hash = "a" * 64
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
        "manifest_canonical_sha256": "c" * 64,
        "model_args": {
            "max_batch_size": 1,
            "max_seq_len": 8,
            "dim": 128,
            "head_dim": 64,
            "index_head_dim": 64,
            "rope_head_dim": 32,
            "candidate_source_layer": 3,
            "compress_ratios": [0, 2, 2, 1, 1],
            "index_source_layers": [1, 3, 4],
            "kv_source_layers": [1, 3],
            "norm_eps": 1e-20,
        },
        "attention_static": {
            "layer_4_freqs_cis": _tensor("torch.complex64", [8, 16], 8)
        },
        "encoded_parameters": {
            "layers.3.attn.indexer.wk.weight": _tensor("torch.bfloat16", [64, 64], 2),
            "layers.3.attn.indexer.k_norm.weight": _tensor("torch.bfloat16", [64], 2),
            "layers.3.attn.compressor.wkv.weight": _tensor(
                "torch.bfloat16", [64, 128], 2
            ),
            "layers.3.attn.compressor.norm.weight": _tensor("torch.bfloat16", [64], 2),
        },
        "steps": [
            {
                "start_pos": start_pos,
                "intermediates": {
                    "layers.3.attention_input": _tensor(
                        "torch.bfloat16", [1, sequence, 128], 2, start_pos + 3
                    ),
                    "layers.3.attn.compressor.wkv": _tensor(
                        "torch.bfloat16", [1, sequence, 64], 2, start_pos + 4
                    ),
                    "layers.3.attn.compressor": _tensor(
                        "torch.bfloat16", [1, sequence, 64], 2, start_pos + 1
                    ),
                },
                "caches_after": {
                    "layer_3.index_k": _tensor(
                        "torch.bfloat16", [1, 8, 64], 2, start_pos + 2
                    )
                },
            }
            for start_pos, sequence in ((0, 5), (5, 1), (6, 1))
        ],
    }


class IndexKeyCaptureTest(unittest.TestCase):
    def test_current_compressor_fixture_pins_live_observer(self) -> None:
        root = Path(__file__).resolve().parent.parent
        fixture = json.loads(
            (
                root / "fixtures/deepseek-v41/forward-compressor-reference.json"
            ).read_text()
        )
        self.assertEqual(
            fixture["source"]["forward_observers_sha256"],
            hashlib.sha256(
                (root / "scripts/v41_forward_observers.py").read_bytes()
            ).hexdigest(),
        )

    def test_extracts_pinned_prefixes_and_provenance(self) -> None:
        fixture = index_key_fixture(_receipt())
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(
            fixture["source"]["frequency_source"], "attention_static.layer_4_freqs_cis"
        )
        self.assertEqual(fixture["model"]["expected_start_positions"], [0, 5, 6])
        self.assertEqual(fixture["weights"]["wk"]["shape"], [64, 64])
        self.assertEqual(fixture["frequencies"]["shape"], [8, 16])
        self.assertEqual([case["start_pos"] for case in fixture["cases"]], [0, 5, 6])
        self.assertEqual(
            [case["index_cache_after"]["shape"] for case in fixture["cases"]],
            [[1, 5, 64], [1, 6, 64], [1, 7, 64]],
        )
        for case in fixture["cases"]:
            tensor = case["index_cache_after"]
            raw = bytes.fromhex(tensor["storage_hex"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), tensor["storage_sha256"])

    def test_rejects_corrupt_tensor_storage(self) -> None:
        receipt = _receipt()
        record = receipt["encoded_parameters"]["layers.3.attn.indexer.wk.weight"]
        record["storage_hex"] = "01" + record["storage_hex"][2:]
        with self.assertRaisesRegex(RuntimeError, "storage hash"):
            index_key_fixture(receipt)

    def test_rejects_nonfinite_bf16_even_when_metadata_claims_finite(self) -> None:
        receipt = _receipt()
        record = receipt["encoded_parameters"]["layers.3.attn.indexer.wk.weight"]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[:2] = b"\x80\x7f"  # +infinity in little-endian BF16 storage.
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(RuntimeError, "finite BF16 storage"):
            index_key_fixture(receipt)

    def test_rejects_unproven_frequency_reuse(self) -> None:
        receipt = _receipt()
        receipt["model_args"]["compress_ratios"][4] = 2
        with self.assertRaisesRegex(RuntimeError, "ratio-one"):
            index_key_fixture(receipt)

    def test_rejects_wrong_parameter_boundary(self) -> None:
        receipt = _receipt()
        receipt["encoded_parameters"] = {
            "layers.3.attn.compressor.wkv.weight": _tensor(
                "torch.bfloat16", [64, 128], 2
            ),
            "layers.3.attn.compressor.norm.weight": _tensor("torch.bfloat16", [64], 2),
        }
        with self.assertRaisesRegex(TypeError, "layers.3.attn.indexer.wk.weight"):
            index_key_fixture(receipt)

    def test_rejects_wrong_source_provenance_and_layout(self) -> None:
        receipt = _receipt()
        receipt["source"]["revision"] = "not-the-pinned-revision"
        with self.assertRaisesRegex(RuntimeError, "pinned source revision"):
            index_key_fixture(receipt)
        receipt = _receipt()
        receipt["source"]["revision"] = "f" * 40
        with self.assertRaisesRegex(RuntimeError, "pinned source revision"):
            index_key_fixture(receipt)
        receipt = _receipt()
        receipt["steps"][1]["intermediates"]["layers.3.attn.compressor"]["shape"] = [
            1,
            2,
            64,
        ]
        with self.assertRaisesRegex(RuntimeError, "shape"):
            index_key_fixture(receipt)

    def test_extracts_compressor_stages_with_shared_provenance(self) -> None:
        fixture = compressor_fixture(_receipt())
        self.assertEqual(fixture["schema_version"], 1)
        self.assertEqual(fixture["model"]["input_dimension"], 128)
        self.assertEqual(fixture["model"]["latent_dimension"], 64)
        self.assertEqual(fixture["model"]["compression_ratio"], 1)
        self.assertEqual(fixture["weights"]["wkv"]["shape"], [64, 128])
        self.assertEqual(fixture["weights"]["norm"]["shape"], [64])
        self.assertEqual([case["start_pos"] for case in fixture["cases"]], [0, 5, 6])
        self.assertEqual(
            [case["attention_input"]["shape"] for case in fixture["cases"]],
            [[1, 5, 128], [1, 1, 128], [1, 1, 128]],
        )
        for case in fixture["cases"]:
            for name in ("attention_input", "projected", "latent"):
                tensor = case[name]
                raw = bytes.fromhex(tensor["storage_hex"])
                self.assertEqual(
                    hashlib.sha256(raw).hexdigest(), tensor["storage_sha256"]
                )

    def test_compressor_rejects_wrong_weight_boundary(self) -> None:
        receipt = _receipt()
        del receipt["encoded_parameters"]["layers.3.attn.compressor.wkv.weight"]
        receipt["encoded_parameters"]["layers.3.attn.compressor.wk.weight"] = _tensor(
            "torch.bfloat16", [64, 128], 2
        )
        with self.assertRaisesRegex(TypeError, "compressor.wkv.weight"):
            compressor_fixture(receipt)

    def test_compressor_rejects_malformed_attention_input(self) -> None:
        receipt = _receipt()
        record = receipt["steps"][1]["intermediates"]["layers.3.attention_input"]
        record["shape"] = [1, 2, 128]
        with self.assertRaisesRegex(RuntimeError, "attention input.*shape"):
            compressor_fixture(receipt)


if __name__ == "__main__":
    unittest.main()
