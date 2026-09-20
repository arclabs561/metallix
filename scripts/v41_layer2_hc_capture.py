"""Project the source layer-one to layer-two attention HC bridge exactly."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).parent
SOURCE_FIELDS = (
    "revision",
    "model_sha256",
    "engram_sha256",
    "kernel_source_sha256",
    "cpu_backend_sha256",
    "loader_sha256",
    "runner_sha256",
    "forward_observers_sha256",
)
PINNED = {
    "revision": "dba1be0a40aa45a94ad051997016db3960a90277",
    "model_sha256": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "engram_sha256": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "kernel_source_sha256": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
}


def _load_ffn():
    spec = importlib.util.spec_from_file_location(
        "v41_l2_ffn_strict", SCRIPTS / "v41_layer2_ffn_capture.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load strict layer-two fixture helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STRICT = _load_ffn()


def _obj(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _tensor(value: object, label: str, dtype: str, shape: list[int]) -> dict[str, Any]:
    record = _obj(value, label)
    actual = record.get("shape")
    if (
        not isinstance(actual, list)
        or not all(type(dim) is int and dim >= 0 for dim in actual)
        or type(record.get("numel")) is not int
    ):
        raise TypeError(f"{label} has invalid tensor metadata")
    return _STRICT._tensor(record, label, dtype=dtype, shape=shape)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _serialized(receipt: dict[str, object]) -> bytes:
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _source(receipt: dict[str, object]) -> dict[str, Any]:
    source = _obj(receipt.get("source"), "source")
    for key, value in PINNED.items():
        if source.get(key) != value:
            raise RuntimeError(f"layer-two HC has unexpected source {key}")
    for key, filename in (
        ("cpu_backend_sha256", "v41_cpu_kernels.py"),
        ("loader_sha256", "v41_source_loader.py"),
        ("runner_sha256", "v41-forward-reference.py"),
        ("forward_observers_sha256", "v41_forward_observers.py"),
    ):
        if source.get(key) != _sha((SCRIPTS / filename).read_bytes()):
            raise RuntimeError(f"layer-two HC has stale source {key}")
    return source


def layer2_hc_fixture(receipt: dict[str, object]) -> dict[str, object]:
    if (
        receipt.get("capture_status")
        != "completed synthetic source-forward capture; no parity claim"
    ):
        raise RuntimeError("layer-two HC requires completed capture")
    if _obj(receipt.get("runtime"), "runtime").get("storage_byteorder") != "little":
        raise RuntimeError("layer-two HC requires little-endian source storage")
    if _obj(receipt.get("coverage_status"), "coverage").get("pending") != []:
        raise RuntimeError("layer-two HC requires complete coverage")
    source = _source(receipt)
    model = _obj(receipt.get("model_args"), "model args")
    encoded = _obj(receipt.get("encoded_parameters"), "encoded parameters")
    steps = receipt.get("steps")
    if not isinstance(steps, list) or model.get("norm_eps") is None:
        raise TypeError("layer-two HC lacks trace or model config")
    parameters = {
        name: _tensor(encoded.get(name), name, dtype=dtype, shape=shape)
        for name, dtype, shape in (
            ("layers.2.hc_attn_fn", "torch.float32", [8, 256]),
            ("layers.2.hc_attn_base", "torch.float32", [8]),
            ("layers.2.hc_attn_scale", "torch.float32", [3]),
            ("layers.2.attn_norm.weight", "torch.bfloat16", [128]),
        )
    }
    cases = []
    for step, start, sequence in zip(steps, (0, 5, 6), (5, 1, 1), strict=True):
        item = _obj(step, "step")
        if item.get("start_pos") != start:
            raise RuntimeError("layer-two HC trace order changed")
        inter = _obj(item.get("intermediates"), "intermediates")
        layer_one = inter.get("layers.1")
        block = _obj(inter.get("layers.2.block_input"), "layer-two block input")
        if not isinstance(layer_one, list) or len(layer_one) != 2:
            raise TypeError("layer-one output must be residual plus pre")
        residual = _tensor(
            layer_one[0], "layer-one residual", "torch.bfloat16", [1, sequence, 2, 128]
        )
        incoming = _tensor(
            layer_one[1], "layer-one pre", "torch.float32", [1, sequence, 2]
        )
        block_residual = _tensor(
            block.get("residual"),
            "layer-two input residual",
            "torch.bfloat16",
            [1, sequence, 2, 128],
        )
        block_pre = _tensor(
            block.get("incoming_pre"),
            "layer-two input pre",
            "torch.float32",
            [1, sequence, 2],
        )
        if (
            residual["storage_sha256"] != block_residual["storage_sha256"]
            or incoming["storage_sha256"] != block_pre["storage_sha256"]
        ):
            raise RuntimeError("layer-one output does not feed layer-two block")
        calls = [
            c
            for c in item.get("hyper_connection_mixes", [])
            if isinstance(c, dict)
            and c.get("layer_id") == 2
            and c.get("sublayer") == "attention"
        ]
        if len(calls) != 1:
            raise RuntimeError("layer-two attention HC observation missing")
        outputs = _obj(calls[0].get("outputs"), "layer-two attention HC outputs")
        inputs = _obj(calls[0].get("inputs"), "layer-two attention HC inputs")
        coeff = {
            "pre": _tensor(
                outputs.get("pre"), "attention pre", "torch.float32", [1, sequence, 2]
            ),
            "post": _tensor(
                outputs.get("post"), "attention post", "torch.float32", [1, sequence, 2]
            ),
            "comb": _tensor(
                outputs.get("comb"),
                "attention comb",
                "torch.float32",
                [1, sequence, 2, 2],
            ),
        }
        if (
            inputs.get("hc_mult") != 2
            or inputs.get("eps") != model.get("hc_eps")
            or inputs.get("sinkhorn_iters") != model.get("hc_sinkhorn_iters")
        ):
            raise RuntimeError("layer-two attention HC configuration changed")
        for field, parameter in (
            ("hc_scale", "layers.2.hc_attn_scale"),
            ("hc_base", "layers.2.hc_attn_base"),
        ):
            captured = _tensor(
                inputs.get(field),
                field,
                parameters[parameter]["dtype"],
                parameters[parameter]["shape"],
            )
            if captured["storage_sha256"] != parameters[parameter]["storage_sha256"]:
                raise RuntimeError(
                    f"layer-two attention {field} does not match encoded parameter"
                )
        mixes = _tensor(
            inputs.get("mixes"),
            "layer-two attention HC mixes",
            "torch.float32",
            [1, sequence, 8],
        )
        after = _tensor(
            inter.get("layers.2.after_attention_residual"),
            "post-attention residual",
            "torch.bfloat16",
            [1, sequence, 2, 128],
        )
        cases.append(
            {
                "start_pos": start,
                "residual": residual,
                "incoming_pre": incoming,
                "attention_input": _tensor(
                    inter.get("layers.2.attention_input"),
                    "attention input",
                    "torch.bfloat16",
                    [1, sequence, 128],
                ),
                "attention_output": _tensor(
                    inter.get("layers.2.attn"),
                    "attention output",
                    "torch.bfloat16",
                    [1, sequence, 128],
                ),
                "after_attention_residual": after,
                "attention_pre": coeff["pre"],
                "attention_hc_mixes": mixes,
                "attention_coefficients": coeff,
            }
        )
    if model.get("hc_mult") != 2:
        raise RuntimeError("layer-two HC model copy count changed")
    return {
        "schema_version": 1,
        "scope": "source layer-one output into layer-two attention HC bridge; not native attention or full forward parity",
        "source": {
            **source,
            "hc_helper_sha256": _sha(Path(__file__).read_bytes()),
            "ffn_strict_helper_sha256": _sha(
                (SCRIPTS / "v41_layer2_ffn_capture.py").read_bytes()
            ),
            "complete_capture_sha256": _sha(_serialized(receipt)),
        },
        "block_config": {
            "copies": 2,
            "norm_eps": model["norm_eps"],
            "hc_eps": model.get("hc_eps"),
            "hc_sinkhorn_iters": model.get("hc_sinkhorn_iters"),
        },
        "block_parameters": parameters,
        "cases": cases,
    }
