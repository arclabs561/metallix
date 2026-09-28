#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers==0.23.2"]
# ///
"""Check pinned Julia `sequence` AST against local contract and tokenizer.

Requires local source and tokenizer assets. The source hash is checked before
extracting and executing only `QTYPES` and `sequence` from its AST. It never
imports the remote module, starts a model, or contacts a network service.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
DEFAULT_FIXTURE = ROOT / "fixtures/julia-1/tokenizer-sequence.json"
MAX_SOURCE_BYTES = 64 * 1024
MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
PINNED_SOURCE_SHA256 = (
    "e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e"
)


def read_hashed_regular(path: Path, maximum: int, expected_hash: str) -> bytes:
    with path.open("rb") as asset:
        if not stat.S_ISREG(os.fstat(asset.fileno()).st_mode):
            raise ValueError(f"asset must be a regular file: {path}")
        size = os.fstat(asset.fileno()).st_size
        if size > maximum:
            raise ValueError(f"asset exceeds {maximum} byte limit: {path}")
        data = asset.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"asset exceeds {maximum} byte limit: {path}")
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"asset SHA-256 mismatch: {path}")
    return data


def extract_sequence(source: bytes) -> Callable[..., dict[str, Any]]:
    """Compile only the two inspected pure source nodes after hash validation."""
    module = ast.parse(source, filename="pinned-julia-data.py", mode="exec")
    qtypes = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "QTYPES"
                for target in node.targets
            )
        ),
        None,
    )
    sequence = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "sequence"
        ),
        None,
    )
    if qtypes is None or sequence is None or sequence.decorator_list:
        raise ValueError(
            "pinned source lacks plain QTYPES assignment or sequence function"
        )
    isolated = ast.fix_missing_locations(
        ast.Module(body=[qtypes, sequence], type_ignores=[])
    )
    namespace: dict[str, Any] = {"json": json}
    exec(compile(isolated, "pinned-julia-data.py", "exec"), namespace)  # noqa: S102
    extracted = namespace.get("sequence")
    if not callable(extracted):
        raise TypeError("pinned source extraction did not define sequence")
    return extracted


def load_contract() -> Any:
    path = Path(__file__).with_name("julia_encoding_contract.py")
    spec = importlib.util.spec_from_file_location("julia_encoding_contract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load local Julia encoding contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LocalTokenizer:
    """Tiny adapter with only the attributes `sequence` reads."""

    def __init__(self, tokenizer: Any, fixture: dict[str, Any]) -> None:
        special = fixture["tokenizer"]["special_ids"]
        self.raw = tokenizer
        self.pad_token_id = special["pad"]
        self.cls_token_id = special["cls"]
        self.sep_token_id = special["sep"]
        self.mask_token_id = special["mask"]
        self.mask_token = fixture["tokenizer"]["mask_token"]

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        return {
            "input_ids": self.raw.encode(
                text, add_special_tokens=add_special_tokens
            ).ids
        }


def run_case(
    source_sequence: Callable[..., dict[str, Any]],
    contract: Any,
    tokenizer: LocalTokenizer,
    case: dict[str, Any],
) -> dict[str, Any]:
    parameters = case["parameters"]
    if "expected_error" in case:
        expected = case["expected_error"]
        source_error = contract_error(
            source_sequence, tokenizer, case["row"], parameters
        )
        local_error = contract_error(
            contract.sequence, tokenizer, case["row"], parameters
        )
        if source_error != expected or local_error != expected:
            raise ValueError(f"case {case['name']} error does not match fixture")
        return {"name": case["name"], "status": "expected_error"}
    source_result = source_sequence(tokenizer, case["row"], **parameters)
    local_result = contract.sequence(tokenizer, case["row"], **parameters)
    if source_result != case["expected"] or local_result != case["expected"]:
        raise ValueError(f"case {case['name']} output does not match fixture")
    return {"name": case["name"], "status": "matched"}


def contract_error(
    sequence: Callable[..., dict[str, Any]],
    tokenizer: LocalTokenizer,
    row: dict[str, Any],
    parameters: dict[str, Any],
) -> str:
    try:
        sequence(tokenizer, row, **parameters)
    except ValueError as error:
        return str(error)
    raise ValueError("expected sequence to reject fixture case")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args(argv)
    fixture = json.loads(args.fixture.read_text())
    if fixture["source"]["sha256"] != PINNED_SOURCE_SHA256:
        raise ValueError("fixture does not name the required pinned Julia source")
    source = read_hashed_regular(args.source, MAX_SOURCE_BYTES, PINNED_SOURCE_SHA256)
    read_hashed_regular(
        args.tokenizer, MAX_TOKENIZER_BYTES, fixture["tokenizer"]["sha256"]
    )
    required_version = fixture["tokenizer"]["tokenizers_version"]
    if importlib.metadata.version("tokenizers") != required_version:
        raise ValueError(f"tokenizers must be exactly {required_version}")
    from tokenizers import Tokenizer

    raw_tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer = LocalTokenizer(raw_tokenizer, fixture)
    for text, expected in fixture["lexical_ids"].items():
        actual = tokenizer(text, add_special_tokens=False)["input_ids"]
        if actual != expected:
            raise ValueError("published tokenizer lexical IDs do not match fixture")
    source_sequence = extract_sequence(source)
    contract = load_contract()
    results = [
        run_case(source_sequence, contract, tokenizer, case)
        for case in fixture["cases"]
    ]
    print(
        json.dumps(
            {
                "status": "passed",
                "source_sha256": PINNED_SOURCE_SHA256,
                "tokenizer_sha256": fixture["tokenizer"]["sha256"],
                "cases": results,
                "scope": "local tokenizer and audited AST-extracted pinned sequence parity; executes only those pinned source nodes, with no remote module import, model startup, or native Julia execution",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
