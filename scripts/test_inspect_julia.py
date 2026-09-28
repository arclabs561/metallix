#!/usr/bin/env python3
"""Header-only tests for the pinned Julia-1 artifact inspector."""

from __future__ import annotations

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "inspect_julia", Path(__file__).with_name("inspect-julia.py")
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Julia inspector")
inspector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspector)
PUBLISHED_HEADER = (
    Path(__file__).parent.parent
    / "fixtures/julia-1/published-header.safetensors-header"
)


def header_for(tensors: dict[str, tuple[int, ...]]) -> tuple[bytes, int]:
    """Create a complete prefix/header and its hypothetical complete size."""
    offset = 0
    body = {"__metadata__": {"format": "pt", "family": "julia"}}
    for name, shape in tensors.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        end = offset + elements * 4
        body[name] = {
            "dtype": "F32",
            "shape": list(shape),
            "data_offsets": [offset, end],
        }
        offset = end
    encoded = json.dumps(body, separators=(",", ":")).encode()
    return struct.pack("<Q", len(encoded)) + encoded, 8 + len(encoded) + offset


def known_header() -> tuple[bytes, int]:
    return header_for(inspector.KNOWN_JULIA_TENSORS | {"encoder.unknown": (1,)})


class JuliaHeaderTests(unittest.TestCase):
    def test_committed_published_header_passes_default_strict_contract(self) -> None:
        prefix_and_header = PUBLISHED_HEADER.read_bytes()
        tensors = inspector.parse_safetensors_header(prefix_and_header, 577189056)
        inspector.validate_known_julia_tensors(tensors)
        inspector.validate_published_header(prefix_and_header, 577189056, tensors)
        self.assertEqual(len(tensors), 170)

    def test_same_shape_encoder_key_mutation_rejects_published_header(self) -> None:
        prefix_and_header = PUBLISHED_HEADER.read_bytes()
        header = json.loads(prefix_and_header[8:])
        header["encoder.embeddings.tok_embeddings.weight"] = header.pop(
            "encoder.embeddings.tok_embeddings.weight"
        )
        header["encoder.embeddings.tok_embeddings.renamed.weight"] = header.pop(
            "encoder.embeddings.tok_embeddings.weight"
        )
        encoded = json.dumps(header, separators=(",", ":")).encode()
        mutated = struct.pack("<Q", len(encoded)) + encoded
        file_bytes = 8 + len(encoded) + 577189056 - len(prefix_and_header)
        tensors = inspector.parse_safetensors_header(mutated, file_bytes)
        with self.assertRaisesRegex(inspector.InspectionError, "pinned header"):
            inspector.validate_published_header(mutated, file_bytes, tensors)

    def test_hand_built_header_accepts_f32_offsets_and_known_julia_shapes(self) -> None:
        header, file_bytes = known_header()
        tensors = inspector.parse_safetensors_header(header, file_bytes)
        inspector.validate_known_julia_tensors(tensors)
        self.assertEqual(tensors["temperature"]["shape"], (3,))
        self.assertEqual(tensors["scorer.1.weight"]["shape"], (384, 384))
        self.assertEqual(
            tensors["encoder.unknown"]["offsets"][1], file_bytes - len(header)
        )

    def test_header_rejects_wrong_dtype_and_payload_holes(self) -> None:
        header, file_bytes = known_header()
        body = json.loads(header[8:])
        body["temperature"]["dtype"] = "F16"
        encoded = json.dumps(body).encode()
        malformed = struct.pack("<Q", len(encoded)) + encoded
        with self.assertRaisesRegex(inspector.InspectionError, "expected F32"):
            inspector.parse_safetensors_header(malformed, file_bytes)

        body = json.loads(header[8:])
        body["temperature"]["data_offsets"] = [4, 16]
        encoded = json.dumps(body).encode()
        malformed = struct.pack("<Q", len(encoded)) + encoded
        with self.assertRaisesRegex(inspector.InspectionError, "hole"):
            inspector.parse_safetensors_header(
                malformed, 8 + len(encoded) + file_bytes - len(header)
            )

    def test_known_tensor_shape_is_checked_without_claiming_a_strict_keyset(
        self,
    ) -> None:
        header, file_bytes = known_header()
        tensors = inspector.parse_safetensors_header(header, file_bytes)
        tensors["type_emb.weight"]["shape"] = (3, 385)
        with self.assertRaisesRegex(inspector.InspectionError, "type_emb.weight"):
            inspector.validate_known_julia_tensors(tensors)

    def test_header_only_artifact_checks_metadata_and_reports_remaining_boundary(
        self,
    ) -> None:
        header, file_bytes = known_header()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "encoder").mkdir()
            (root / "tokenizer").mkdir()
            (root / "config.json").write_text(json.dumps(inspector.TOP_LEVEL_CONFIG))
            (root / "julia_config.json").write_text(json.dumps(inspector.JULIA_CONFIG))
            (root / "encoder" / "config.json").write_text(
                json.dumps(inspector.ENCODER_CONFIG)
            )
            (root / "model.safetensors").write_bytes(header)
            report = inspector.inspect_artifact(
                root, declared_model_bytes=file_bytes, verify_published_header=False
            )
            with self.assertRaisesRegex(inspector.InspectionError, "pinned header"):
                inspector.inspect_artifact(root, declared_model_bytes=file_bytes)
        self.assertEqual(report["source_revision"], inspector.SOURCE_REVISION)
        self.assertEqual(report["native_support"], "unsupported")
        self.assertEqual(
            report["strict_tensor_keyset"],
            "unverified: published-header verification was explicitly disabled",
        )

    def test_metadata_rejects_a_different_modernbert_contract(self) -> None:
        header, file_bytes = known_header()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "encoder").mkdir()
            (root / "tokenizer").mkdir()
            (root / "config.json").write_text(json.dumps(inspector.TOP_LEVEL_CONFIG))
            (root / "julia_config.json").write_text(json.dumps(inspector.JULIA_CONFIG))
            bad_encoder = dict(inspector.ENCODER_CONFIG, hidden_size=385)
            (root / "encoder" / "config.json").write_text(json.dumps(bad_encoder))
            (root / "model.safetensors").write_bytes(header)
            with self.assertRaisesRegex(inspector.InspectionError, "hidden_size"):
                inspector.inspect_artifact(
                    root, declared_model_bytes=file_bytes, verify_published_header=False
                )


if __name__ == "__main__":
    unittest.main()
