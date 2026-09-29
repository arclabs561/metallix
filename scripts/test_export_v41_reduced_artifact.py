#!/usr/bin/env python3
"""Dependency-free regression checks for the reduced artifact exporter."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "fixtures/deepseek-v41/reduced-runner-reference.json"
EXPORTER = ROOT / "scripts/export_v41_reduced_artifact.py"
SPEC = importlib.util.spec_from_file_location("reduced_exporter", EXPORTER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReducedArtifactExportTests(unittest.TestCase):
    def test_deterministic_oracle_free_export(self) -> None:
        source = json.loads(SOURCE.read_text())
        first = MODULE.export(source)
        second = MODULE.export(source)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {"schema_version", "format", "config", "tensors"})
        self.assertEqual(first["schema_version"], 1)
        self.assertEqual(first["format"], "metallix.deepseek.reduced")
        self.assertNotIn("source_sha256", first["config"])
        self.assertEqual(first["tensors"]["rotary.startup"]["shape"], [8, 16, 2])
        self.assertNotEqual(
            first["tensors"]["rotary.startup"], first["tensors"]["rotary.shared"]
        )
        self.assertIn("layers.3.attn.indexer.wq_b.weight", first["tensors"])
        self.assertIn("layers.3.attn.indexer.weights_proj.weight", first["tensors"])
        for record in first["tensors"].values():
            raw = bytes.fromhex(record["storage_hex"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["storage_sha256"])

    def test_stdout_is_only_the_artifact_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(EXPORTER), "--source", str(SOURCE), "--output", "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stderr, "")
        artifact = json.loads(completed.stdout)
        self.assertEqual(
            set(artifact), {"schema_version", "format", "config", "tensors"}
        )

    def test_refuses_corrupt_or_conflicting_source_operands(self) -> None:
        source = json.loads(SOURCE.read_text())
        corrupt = copy.deepcopy(source)
        corrupt["projections"]["layer0_to_layer1"]["parameters"]["embed.weight"][
            "storage_sha256"
        ] = "0" * 64
        with self.assertRaises(MODULE.ExportError):
            MODULE.export(corrupt)
        conflict = copy.deepcopy(source)
        record = conflict["projections"]["layer3_candidate"]["encoded_parameters"][
            "layers.3.attn.q_norm.weight"
        ]
        raw = bytearray.fromhex(record["storage_hex"])
        raw[0] ^= 1
        record["storage_hex"] = raw.hex()
        record["storage_sha256"] = hashlib.sha256(raw).hexdigest()
        with self.assertRaises(MODULE.ExportError):
            MODULE.export(conflict)

    def test_refuses_inconsistent_layer_configuration(self) -> None:
        source = json.loads(SOURCE.read_text())
        source["projections"]["layer3_attention"]["model"]["norm_eps"] = 1e-8
        with self.assertRaises(MODULE.ExportError):
            MODULE.export(source)


if __name__ == "__main__":
    unittest.main()
