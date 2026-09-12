"""Hash-gated loading of the unmodified V4.1 text graph with explicit kernels.

No download and no numerical implementation lives here. The CPU runner supplies
the six kernel boundaries; vision imports are omitted for text-only execution.
Class/function bodies are preserved, including Engram and Transformer.forward.
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
MODEL_SHA256 = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
ENGRAM_SHA256 = "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
KERNEL_NAMES = (
    "act_quant",
    "fp4_act_quant",
    "fp4_gemm",
    "fp8_gemm",
    "hc_split_sinkhorn",
    "sparse_attn",
)
REPLACED_IMPORTS = frozenset(("engram", "kernel", "vision", "image_processor"))


def checked_tree(path: Path, expected_sha256: str) -> ast.Module:
    """Reject changed bytes before parsing or executing any upstream code."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError(f"pinned source hash mismatch: {path.name}")
    return ast.parse(source, filename=str(path))


def text_graph_tree(path: Path) -> ast.Module:
    """Remove only dependency imports; preserve every source operation body."""
    tree = checked_tree(path, MODEL_SHA256)
    removed = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module in REPLACED_IMPORTS
    }
    if removed != REPLACED_IMPORTS:
        raise ValueError("pinned dependency boundary differs from loader contract")
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module in REPLACED_IMPORTS)
    ]
    return tree


def _execute_module(tree: ast.Module, path: Path, module: ModuleType) -> ModuleType:
    """Register temporarily for dataclass introspection, restoring prior state."""
    previous = sys.modules.get(module.__name__)
    sys.modules[module.__name__] = module
    module.__file__ = str(path)
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102 -- hash-checked source only
    finally:
        if previous is None:
            del sys.modules[module.__name__]
        else:
            sys.modules[module.__name__] = previous
    return module


def load_text_graph(kernels: ModuleType, source_dir: Path | None = None) -> ModuleType:
    """Load pinned graph, with numerical substitutions explicit at call site.

    Requires Torch, NumPy and SymPy in the caller environment. Tokenizer
    normalization additionally requires tokenizers when constructing Engram.
    Callers must use vision_n_layers=0 and images/token_types=None. This loader
    does not certify kernel accuracy or a completed forward capture.
    """
    source_dir = source_dir or ROOT / "artifacts"
    model_path = source_dir / "v41-reference-model.py"
    engram_path = source_dir / "v41-engram-pinned.py"
    # Check both inputs and all replacement boundaries before executing either.
    model_tree = text_graph_tree(model_path)
    engram_tree = checked_tree(engram_path, ENGRAM_SHA256)
    for name in KERNEL_NAMES:
        if not callable(getattr(kernels, name, None)):
            raise TypeError(f"missing CPU numerical kernel: {name}")
    engram = _execute_module(
        engram_tree, engram_path, ModuleType("_metallix_v41_engram")
    )
    model = ModuleType("_metallix_v41_model")
    model.EngramLayout = engram.EngramLayout
    model.NgramHashState = engram.NgramHashState
    for name in KERNEL_NAMES:
        setattr(model, name, getattr(kernels, name))
    return _execute_module(model_tree, model_path, model)
