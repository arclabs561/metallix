#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Create a reproducibility receipt for a local DeepSeek MLX artifact.

This is an operator-only inventory of bytes and bounded native-gate observations.
It does not load a model, execute inference, contact the network, or prove
serving readiness.  Paths are recorded for operator traceability; credentials
and environment variables are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_HEX = re.compile(r"^[0-9a-fA-F]+$")


class ReceiptError(ValueError):
    """The artifact receipt input is invalid or outside the operator scope."""


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ReceiptError(f"{label} must be a regular file: {path}")
    return resolved


def _inside(path: Path, root: Path, label: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ReceiptError(f"{label} must be inside artifact root") from error
    return relative.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate(value: str) -> tuple[str, str]:
    name, separator, checksum = value.partition("=")
    if not separator or not name or not _HEX.fullmatch(checksum):
        raise ReceiptError("--gate must be NAME=hex-checksum")
    if len(checksum) not in (16, 32, 64):
        raise ReceiptError("--gate checksum must contain 16, 32, or 64 hex digits")
    return name, checksum.lower()


def build_receipt(
    root_arg: str,
    index_arg: str,
    files: list[str],
    gates: list[str],
) -> dict[str, Any]:
    root = Path(root_arg).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ReceiptError(f"artifact root must be a directory: {root_arg}")
    index = _regular_file(Path(index_arg), "index")
    index_relative = _inside(index, root, "index")

    selected: dict[str, Path] = {index_relative: index}
    for file_arg in files:
        file = _regular_file(
            Path(file_arg) if Path(file_arg).is_absolute() else root / file_arg, "file"
        )
        relative = _inside(file, root, "file")
        selected[relative] = file

    file_records = [
        {
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for relative, path in sorted(selected.items())
    ]
    gate_records: dict[str, str] = {}
    for raw_gate in gates:
        name, checksum = _gate(raw_gate)
        if name in gate_records:
            raise ReceiptError(f"duplicate gate name: {name}")
        gate_records[name] = checksum

    return {
        "schema_version": 1,
        "status": "completed",
        "scope": "operator-only local DeepSeek MLX artifact reproducibility receipt",
        "operator_only": True,
        "claims": [
            "artifact path and selected byte identities are recorded",
            "selected files are hashed without modifying them",
            "gate checksums are observations supplied by the operator",
        ],
        "not_a_claim": [
            "model execution",
            "generation or tokenizer parity",
            "Codex readiness",
            "license permission beyond operator-provided provenance",
        ],
        "artifact": {
            "root": str(root),
            "index": {
                "path": index_relative,
                "size_bytes": index.stat().st_size,
                "sha256": _sha256(index),
            },
            "files": file_records,
        },
        "gate_checksums": gate_records,
    }


def _write_output(path_arg: str | None, receipt: dict[str, Any]) -> None:
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if path_arg is None:
        sys.stdout.write(encoded)
        return
    output = Path(path_arg).expanduser()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(output, flags, 0o644)
    except FileExistsError as error:
        raise ReceiptError(f"refusing to overwrite receipt: {output}") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
        destination.write(encoded)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root", required=True, help="local MLX artifact directory"
    )
    parser.add_argument(
        "--index", required=True, help="weight-map/index file inside artifact root"
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="additional file relative to root (repeatable)",
    )
    parser.add_argument(
        "--gate",
        action="append",
        default=[],
        help="operator observation NAME=hex-checksum (repeatable)",
    )
    parser.add_argument(
        "--output", help="write receipt to a new file instead of stdout"
    )
    args = parser.parse_args(argv)
    try:
        receipt = build_receipt(args.artifact_root, args.index, args.file, args.gate)
        _write_output(args.output, receipt)
    except (OSError, ReceiptError) as error:
        print(f"receipt error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
