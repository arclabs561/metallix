"""Tests for the operator-only DeepSeek artifact reproducibility receipt."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("v41_native_receipt.py")


class NativeReceiptTests(unittest.TestCase):
    def run_receipt(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

    def test_records_index_and_selected_files_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            index = root / "model.safetensors.index.json"
            config = root / "config.json"
            index.write_bytes(b'{"weight_map":{}}\n')
            config.write_bytes(b'{"model_type":"deepseek"}\n')
            before = (index.read_bytes(), config.read_bytes())
            result = self.run_receipt(
                "--artifact-root",
                str(root),
                "--index",
                str(index),
                "--file",
                "config.json",
                "--gate",
                "kv_rotary_tail=e51de85f62270c8e",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(result.stdout)
            self.assertEqual(receipt["schema_version"], 1)
            self.assertTrue(receipt["operator_only"])
            self.assertIn("model execution", receipt["not_a_claim"])
            self.assertEqual(receipt["artifact"]["index"]["path"], index.name)
            self.assertEqual(
                receipt["gate_checksums"]["kv_rotary_tail"], "e51de85f62270c8e"
            )
            records = {entry["path"]: entry for entry in receipt["artifact"]["files"]}
            self.assertEqual(set(records), {index.name, config.name})
            self.assertEqual(
                records[index.name]["sha256"], hashlib.sha256(before[0]).hexdigest()
            )
            self.assertEqual((index.read_bytes(), config.read_bytes()), before)

    def test_rejects_index_outside_root_and_bad_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory) / "artifact"
            root.mkdir()
            outside = pathlib.Path(directory) / "index.json"
            outside.write_text("{}\n", encoding="utf-8")
            outside_result = self.run_receipt(
                "--artifact-root",
                str(root),
                "--index",
                str(outside),
            )
            self.assertEqual(outside_result.returncode, 1)
            self.assertIn("inside artifact root", outside_result.stderr)
            inside = root / "index.json"
            inside.write_text("{}\n", encoding="utf-8")
            bad_gate = self.run_receipt(
                "--artifact-root",
                str(root),
                "--index",
                str(inside),
                "--gate",
                "checksum=not-hex",
            )
            self.assertEqual(bad_gate.returncode, 1)
            self.assertIn("NAME=hex-checksum", bad_gate.stderr)

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            index = root / "index.json"
            index.write_text("{}\n", encoding="utf-8")
            output = root / "receipt.json"
            output.write_text("sentinel", encoding="utf-8")
            result = self.run_receipt(
                "--artifact-root",
                str(root),
                "--index",
                str(index),
                "--output",
                str(output),
            )
            self.assertEqual(result.returncode, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel")


if __name__ == "__main__":
    unittest.main()
