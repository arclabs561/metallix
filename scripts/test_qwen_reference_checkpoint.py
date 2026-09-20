"""Stdlib checks for Qwen reference checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from qwen_reference_checkpoint import (
    CheckpointIndexError,
    checkpoint_weight_provenance,
    is_safe_shard_filename,
    required_shards,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QwenReferenceCheckpointTest(unittest.TestCase):
    def write_index(self, root: Path, payload: object) -> Path:
        path = root / "model.safetensors.index.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_single_file_receipt_stays_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "model.safetensors"
            weights.write_bytes(b"single")
            self.assertEqual(
                checkpoint_weight_provenance(root, sha256_file),
                {"weights_sha256": sha256_file(weights)},
            )

    def test_indexed_receipt_hashes_index_and_exact_unique_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "model-00001-of-00002.safetensors"
            second = root / "model-00002-of-00002.safetensors"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            index = self.write_index(
                root,
                {
                    "metadata": {"total_size": 11},
                    "weight_map": {
                        "model.layers.0.weight": second.name,
                        "model.embed_tokens.weight": first.name,
                        "model.layers.1.weight": second.name,
                    },
                },
            )
            self.assertEqual(
                checkpoint_weight_provenance(root, sha256_file),
                {
                    "weights_index_sha256": sha256_file(index),
                    "weight_shards": [
                        {"filename": first.name, "sha256": sha256_file(first)},
                        {"filename": second.name, "sha256": sha256_file(second)},
                    ],
                },
            )

    def test_invalid_index_shapes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for payload in (
                [],
                {},
                {"weight_map": []},
                {"weight_map": {"": "a.safetensors"}},
            ):
                index = self.write_index(root, payload)
                with self.assertRaises(CheckpointIndexError):
                    required_shards(index)

    def test_unsafe_index_paths_and_missing_required_shards_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for filename in (
                "../outside.safetensors",
                "nested/a.safetensors",
                "a.bin",
                ".safetensors",
            ):
                index = self.write_index(root, {"weight_map": {"tensor": filename}})
                with self.assertRaises(CheckpointIndexError):
                    required_shards(index)
            self.assertTrue(is_safe_shard_filename("model-00001-of-00001.safetensors"))
            self.assertFalse(is_safe_shard_filename("../model.safetensors"))
            self.write_index(
                root, {"weight_map": {"tensor": "model-00001-of-00001.safetensors"}}
            )
            with self.assertRaises(FileNotFoundError):
                checkpoint_weight_provenance(root, sha256_file)


if __name__ == "__main__":
    unittest.main()
