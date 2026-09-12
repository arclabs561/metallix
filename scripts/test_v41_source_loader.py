"""Provenance and source-body preservation checks; no tensor dependencies."""

import ast
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import v41_source_loader as loader


class SourceLoaderTests(unittest.TestCase):
    def test_changed_bytes_rejected_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_text("not valid python!", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                loader.checked_tree(path, "0" * 64)

    def test_checked_tree_preserves_code(self):
        source = b"def answer():\n    return 42\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_bytes(source)
            tree = loader.checked_tree(path, hashlib.sha256(source).hexdigest())
            self.assertEqual(ast.dump(tree), ast.dump(ast.parse(source)))

    @unittest.skipUnless(
        (loader.ROOT / "artifacts/v41-reference-model.py").is_file(),
        "retained pinned model source required for source-body comparison",
    )
    def test_actual_source_bodies_and_main_guard_are_unchanged(self):
        path = loader.ROOT / "artifacts/v41-reference-model.py"
        original = loader.checked_tree(path, loader.MODEL_SHA256)
        adapted = loader.text_graph_tree(path)
        expected = [
            node
            for node in original.body
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module in loader.REPLACED_IMPORTS
            )
        ]
        self.assertEqual(len(original.body) - len(adapted.body), 4)
        self.assertEqual(
            [ast.dump(node) for node in adapted.body],
            [ast.dump(node) for node in expected],
        )
        self.assertIn("class Transformer", ast.unparse(adapted))
        compile(adapted, str(path), "exec")

    @unittest.skipUnless(
        (loader.ROOT / "artifacts/v41-engram-pinned.py").is_file()
        and (loader.ROOT / "artifacts/v41-reference-model.py").is_file(),
        "retained model and Engram sources required",
    )
    def test_missing_kernels_rejected_before_torch_import(self):
        with self.assertRaisesRegex(
            TypeError, "missing CPU numerical kernel: act_quant"
        ):
            loader.load_text_graph(ModuleType("empty_kernels"))


if __name__ == "__main__":
    unittest.main()
