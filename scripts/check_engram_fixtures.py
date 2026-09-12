"""Verify the two checked-in DeepSeek V4.1 Engram reference fixtures.

This is deliberately a static, dependency-free integrity check. It validates
the fixture tensor records and their little-endian byte digests; it does not
execute the captured Python/Torch reference programs or any Metal code.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HASH_FIXTURE = ROOT / "fixtures/deepseek-v41/engram-hash-reference.json"
GATE_FIXTURE = ROOT / "fixtures/deepseek-v41/engram-gate-reference.json"

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
UINT16_MAX = (1 << 16) - 1
UINT32_MAX = (1 << 32) - 1
SHA256_HEX_LENGTH = 64

GATE_ENCODINGS = {
    "torch.bfloat16": "bf16_bits_u16",
    "torch.int64": "i64",
    "torch.float32": "f32_bits_u32",
    "torch.bool": "bool",
}


class FixtureError(ValueError):
    """A checked-in Engram fixture does not meet its static wire contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FixtureError(message)


def flatten(values: object, *, name: str) -> list[object]:
    if isinstance(values, list):
        flattened: list[object] = []
        for value in values:
            flattened.extend(flatten(value, name=name))
        return flattened
    return [values]


def validate_hash_nesting(values: object, dimensions: list[int], *, name: str) -> None:
    """Require the nested JSON layout that Rust's shaped tensor reader uses."""

    def visit(value: object, depth: int) -> None:
        if depth == len(dimensions):
            require(
                not isinstance(value, list), f"{name}: hash values have extra nesting"
            )
            return
        require(
            isinstance(value, list), f"{name}: hash values must match shape nesting"
        )
        require(
            len(value) == dimensions[depth],
            f"{name}: hash values must match shape nesting",
        )
        for item in value:
            visit(item, depth + 1)

    visit(values, 0)


def tensor_shape(record: dict[str, object], *, name: str) -> tuple[list[int], int]:
    shape = record.get("shape")
    require(isinstance(shape, list) and shape, f"{name}: shape must be a nonempty list")
    product = 1
    dimensions: list[int] = []
    for dimension in shape:
        require(
            isinstance(dimension, int)
            and not isinstance(dimension, bool)
            and dimension > 0,
            f"{name}: shape dimensions must be positive integers",
        )
        product *= dimension
        dimensions.append(dimension)
    return dimensions, product


def integer_values(
    values: list[object], *, minimum: int, maximum: int, name: str
) -> list[int]:
    integers: list[int] = []
    for value in values:
        require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{name}: values must be integers",
        )
        require(minimum <= value <= maximum, f"{name}: value is outside its bit range")
        integers.append(value)
    return integers


def bool_values(values: list[object], *, name: str) -> list[bool]:
    for value in values:
        require(isinstance(value, bool), f"{name}: values must be JSON booleans")
    return values  # type: ignore[return-value]


def digest_values(dtype: str, values: list[object], *, name: str) -> str:
    if dtype == "torch.int64":
        integers = integer_values(
            values, minimum=INT64_MIN, maximum=INT64_MAX, name=name
        )
        raw = struct.pack(f"<{len(integers)}q", *integers)
    elif dtype == "torch.bfloat16":
        integers = integer_values(values, minimum=0, maximum=UINT16_MAX, name=name)
        raw = struct.pack(f"<{len(integers)}H", *integers)
    elif dtype == "torch.float32":
        integers = integer_values(values, minimum=0, maximum=UINT32_MAX, name=name)
        raw = struct.pack(f"<{len(integers)}I", *integers)
    elif dtype == "torch.bool":
        raw = bytes(bool_values(values, name=name))
    else:
        raise FixtureError(f"{name}: unsupported dtype {dtype!r}")
    return hashlib.sha256(raw).hexdigest()


def validate_tensor(record: object, *, fixture_kind: str) -> None:
    require(isinstance(record, dict), "tensor record must be an object")
    name = record.get("name")
    require(isinstance(name, str) and name, "tensor record requires a nonempty name")
    dtype = record.get("dtype")
    require(isinstance(dtype, str), f"{name}: dtype must be a string")
    if fixture_kind == "hash":
        require(dtype == "torch.int64", f"{name}: hash fixture requires torch.int64")
        require(
            "encoding" not in record, f"{name}: hash fixture must not declare encoding"
        )
    else:
        expected_encoding = GATE_ENCODINGS.get(dtype)
        require(
            expected_encoding is not None, f"{name}: unsupported gate dtype {dtype!r}"
        )
        require(
            record.get("encoding") == expected_encoding,
            f"{name}: requires encoding {expected_encoding!r}",
        )
    dimensions, product = tensor_shape(record, name=name)
    values = record.get("values")
    require(isinstance(values, list), f"{name}: values must be a list")
    if fixture_kind == "hash":
        validate_hash_nesting(values, dimensions, name=name)
    else:
        require(
            all(not isinstance(value, list) for value in values),
            f"{name}: gate values must be a flat scalar array",
        )
    flat_values = flatten(values, name=name)
    require(len(flat_values) == product, f"{name}: values length does not match shape")
    digest = record.get("sha256_le_bytes")
    require(
        isinstance(digest, str)
        and len(digest) == SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in digest),
        f"{name}: sha256_le_bytes must be lowercase hexadecimal",
    )
    actual_digest = digest_values(dtype, flat_values, name=name)
    require(actual_digest == digest, f"{name}: sha256_le_bytes does not match values")


def validate_fixture(payload: object, *, fixture_kind: str) -> None:
    """Validate one known Engram fixture object without executing a model."""
    require(fixture_kind in {"hash", "gate"}, "unknown Engram fixture kind")
    require(isinstance(payload, dict), "fixture root must be an object")
    schema_version = payload.get("schema_version")
    require(
        isinstance(schema_version, int)
        and not isinstance(schema_version, bool)
        and schema_version == 1,
        "fixture requires integer schema_version 1",
    )
    tensors = payload.get("tensors")
    require(isinstance(tensors, list) and tensors, "fixture requires nonempty tensors")
    names: set[str] = set()
    for tensor in tensors:
        validate_tensor(tensor, fixture_kind=fixture_kind)
        name = tensor["name"]
        require(name not in names, f"duplicate tensor name {name!r}")
        names.add(name)


def load_fixture(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FixtureError(f"could not read {path.name}: {error}") from error


def main() -> int:
    try:
        validate_fixture(load_fixture(HASH_FIXTURE), fixture_kind="hash")
        validate_fixture(load_fixture(GATE_FIXTURE), fixture_kind="gate")
    except FixtureError as error:
        print(f"Engram fixture check failed: {error}", file=sys.stderr)
        return 1
    print(
        "Engram fixtures verified: hash and gate tensor records match little-endian SHA-256"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
