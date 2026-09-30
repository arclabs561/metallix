#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Pinned SDPA full-22-layer Julia encoder plus decision-head source fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/julia-1/full-prefill-reference.json"
WIDTH, LAYERS, VOCAB = 384, 22, 8


def load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENCODER = load("julia_encoder_reference", "julia_encoder_reference.py")
HEAD = load("julia_head_reference", "julia_head_reference.py")


def near_one(ordinal: int) -> torch.Tensor:
    return 1.0 + ENCODER.values((WIDTH,), ordinal) / 10.0


def layer_weights(layer: int) -> dict[str, torch.Tensor]:
    weights = ENCODER.weights(layer * 6)
    weights["attn_norm"] = near_one(300 + layer * 2)
    weights["mlp_norm"] = near_one(301 + layer * 2)
    return weights


def source_encoder(attention_implementation: str = "sdpa") -> torch.nn.Module:
    if attention_implementation not in {"sdpa", "eager"}:
        raise ValueError("attention implementation must be sdpa or eager")
    import transformers.models.modernbert.modeling_modernbert as source_module
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
    from transformers.models.modernbert.modeling_modernbert import ModernBertModel

    path = Path(inspect.getsourcefile(source_module) or "")
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != ENCODER.SOURCE_SHA256
    ):
        raise RuntimeError("installed ModernBERT source does not match pin")
    config = ModernBertConfig(
        vocab_size=VOCAB,
        pad_token_id=0,
        hidden_size=WIDTH,
        num_hidden_layers=LAYERS,
        num_attention_heads=6,
        intermediate_size=1152,
        attention_bias=False,
        mlp_bias=False,
        norm_bias=False,
        norm_eps=1e-5,
        hidden_activation="gelu",
        embedding_dropout=0.0,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        global_attn_every_n_layers=3,
        local_attention=128,
        max_position_embeddings=8192,
        rope_parameters={
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 160000.0, "rope_type": "default"},
        },
    )
    config._attn_implementation = attention_implementation
    model = ModernBertModel(config).eval()
    with torch.no_grad():
        model.embeddings.tok_embeddings.weight.copy_(
            ENCODER.values((VOCAB, WIDTH), 200)
        )
        model.embeddings.norm.weight.copy_(near_one(201))
        model.final_norm.weight.copy_(near_one(202))
        for layer_id, layer in enumerate(model.layers):
            weights = layer_weights(layer_id)
            layer.attn.Wqkv.weight.copy_(weights["wqkv"])
            layer.attn.Wo.weight.copy_(weights["attn_wo"])
            layer.mlp.Wi.weight.copy_(weights["wi"])
            layer.mlp.Wo.weight.copy_(weights["mlp_wo"])
            if layer_id:
                layer.attn_norm.weight.copy_(weights["attn_norm"])
            layer.mlp_norm.weight.copy_(weights["mlp_norm"])
    return model


def source_case(case: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    encoder = source_encoder()
    head, fake_encoder = HEAD.build_source_oracle()
    input_ids = torch.tensor([case["input_ids"]], dtype=torch.int64)
    attention_mask = torch.tensor([case["attention_mask"]], dtype=torch.bool)
    marker_pos = torch.tensor([case["marker_pos"]], dtype=torch.int64)
    marker_mask = torch.tensor([case["marker_mask"]], dtype=torch.bool)
    qtype = torch.tensor([case["qtype"]], dtype=torch.int64)
    with torch.inference_mode():
        hidden = encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        fake_encoder.hidden = hidden
        scores = head(input_ids, attention_mask, marker_pos, marker_mask, qtype)
    return hidden, scores


def boundaries(case: dict[str, Any]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Capture actual default-SDPA boundaries without changing the reference fixture."""
    encoder = source_encoder()
    input_ids = torch.tensor([case["input_ids"]], dtype=torch.int64)
    attention_mask = torch.tensor([case["attention_mask"]], dtype=torch.bool)
    observed: dict[str, torch.Tensor] = {}
    hooks = []

    def capture(name: str):
        def hook(_: torch.nn.Module, __: tuple[object, ...], output: object) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"unexpected {name} output")
            observed[name] = value.detach().squeeze(0).clone()

        return hook

    hooks.append(encoder.embeddings.register_forward_hook(capture("embedding")))
    for layer_id, layer in enumerate(encoder.layers):
        hooks.append(layer.register_forward_hook(capture(f"layer_{layer_id}")))
    hooks.append(encoder.final_norm.register_forward_hook(capture("final_norm")))
    with (
        torch.inference_mode(),
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profiler,
    ):
        encoder(input_ids=input_ids, attention_mask=attention_mask)
    for hook in hooks:
        hook.remove()
    operators = sorted(
        event.key
        for event in profiler.key_averages()
        if "scaled_dot_product" in event.key or "flash_attention" in event.key
    )
    return observed, operators


def layer0_trace(
    case: dict[str, Any],
    attention_implementation: str = "sdpa",
    capture_rope: bool = False,
    embedding_override: torch.Tensor | None = None,
    qkv_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Capture layer zero, optionally replacing its embedding or `Wqkv` output."""

    encoder = source_encoder(attention_implementation)
    input_ids = torch.tensor([case["input_ids"]], dtype=torch.int64)
    attention_mask = torch.tensor([case["attention_mask"]], dtype=torch.bool)
    observed: dict[str, torch.Tensor] = {}

    def output(name: str):
        def hook(_: torch.nn.Module, __: tuple[object, ...], value: object) -> None:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"unexpected {name} output")
            observed[name] = value.detach().squeeze(0).clone()

        return hook

    def layer_output(name: str):
        def hook(_: torch.nn.Module, __: tuple[object, ...], value: object) -> None:
            if not isinstance(value, tuple) or not value:
                raise TypeError(f"unexpected {name} output envelope")
            hidden = value[0]
            if not isinstance(hidden, torch.Tensor):
                raise TypeError(f"unexpected {name} hidden output")
            observed[name] = hidden.detach().squeeze(0).clone()

        return hook

    def embedding_output(
        _: torch.nn.Module, __: tuple[object, ...], value: object
    ) -> torch.Tensor | None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("unexpected embedding output")
        if embedding_override is None:
            observed["embedding"] = value.detach().squeeze(0).clone()
            return None
        expected = (len(case["input_ids"]), WIDTH)
        if (
            embedding_override.dtype != torch.float32
            or tuple(embedding_override.shape) != expected
        ):
            raise ValueError("embedding override must be an F32 selected embedding")
        observed["embedding"] = embedding_override.detach().clone()
        return embedding_override.unsqueeze(0)

    def qkv_output(
        _: torch.nn.Module, __: tuple[object, ...], value: object
    ) -> torch.Tensor | None:
        if not isinstance(value, torch.Tensor):
            raise TypeError("unexpected Wqkv output")
        if qkv_override is None:
            observed["qkv"] = value.detach().squeeze(0).clone()
            return None
        expected = (len(case["input_ids"]), 3 * WIDTH)
        if qkv_override.dtype != torch.float32 or tuple(qkv_override.shape) != expected:
            raise ValueError("Wqkv override must be an F32 selected QKV tensor")
        observed["qkv"] = qkv_override.detach().clone()
        return qkv_override.unsqueeze(0)

    def input_hook(name: str):
        def hook(_: torch.nn.Module, value: tuple[object, ...]) -> None:
            if not value or not isinstance(value[0], torch.Tensor):
                raise TypeError(f"unexpected {name} input")
            observed[name] = value[0].detach().squeeze(0).clone()

        return hook

    layer = encoder.layers[0]
    hooks = [
        encoder.embeddings.register_forward_hook(embedding_output),
        encoder.embeddings.tok_embeddings.register_forward_hook(output("lookup")),
        layer.register_forward_hook(layer_output("layer_0")),
        layer.attn.Wqkv.register_forward_hook(qkv_output),
        layer.attn.Wo.register_forward_pre_hook(input_hook("attended")),
        layer.attn.Wo.register_forward_hook(output("wo")),
    ]
    rotary_calls = 0

    def observed_rotary(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal rotary_calls
        capture_layer_zero = rotary_calls == 0
        rotary_calls += 1
        if capture_layer_zero:
            observed["source_pre_rotary_query"] = query.detach().clone()
            observed["source_pre_rotary_key"] = key.detach().clone()
            observed["source_rotary_cos"] = cos.detach().clone()
            observed["source_rotary_sin"] = sin.detach().clone()
        rotated_query, rotated_key = original_rotary(
            query, key, cos, sin, *args, **kwargs
        )
        if capture_layer_zero:
            observed["source_rotary_query"] = rotated_query.detach().clone()
            observed["source_rotary_key"] = rotated_key.detach().clone()
        return rotated_query, rotated_key

    if capture_rope:
        import transformers.models.modernbert.modeling_modernbert as source_module

        original_rotary = source_module.apply_rotary_pos_emb
        source_module.apply_rotary_pos_emb = observed_rotary
    try:
        with torch.inference_mode():
            encoder(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        if capture_rope:
            source_module.apply_rotary_pos_emb = original_rotary
            if source_module.apply_rotary_pos_emb is not original_rotary:
                raise RuntimeError("source RoPE binding was not restored")
        for hook in hooks:
            hook.remove()
    if capture_rope and rotary_calls != LAYERS:
        raise RuntimeError(
            f"expected {LAYERS} source RoPE calls, observed {rotary_calls}"
        )
    observed["post_wo_residual"] = observed["embedding"] + observed["wo"]
    positions = observed["qkv"].shape[0]
    qkv = observed["qkv"].reshape(positions, 3, 6, 64)
    query, key, _ = qkv.unbind(dim=1)
    query, key = ENCODER.rope(query), ENCODER.rope(key)
    observed["rotated_query"] = query
    observed["rotated_key"] = key
    logits = torch.einsum("qhd,khd->hqk", query, key) / 64.0**0.5
    observed["logits"] = logits
    observed["probabilities"] = logits.softmax(dim=-1, dtype=torch.float32)
    return observed


def eager_boundaries(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Run the individually hash-gated eager layers to isolate the SDPA seam."""
    input_ids = torch.tensor(case["input_ids"], dtype=torch.int64)
    mask = torch.tensor(case["attention_mask"], dtype=torch.bool)
    x = ENCODER.values((VOCAB, WIDTH), 200)[input_ids]
    x = torch.nn.functional.layer_norm(x, (WIDTH,), near_one(201), None, 1e-5)
    observed = {"embedding": x.clone()}
    for layer_id in range(LAYERS):
        x = ENCODER.source_block(x, mask, layer_id, layer_weights(layer_id))
        observed[f"layer_{layer_id}"] = x.clone()
    observed["final_norm"] = torch.nn.functional.layer_norm(
        x, (WIDTH,), near_one(202), None, 1e-5
    )
    return observed


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def diagnostic(case: dict[str, Any]) -> dict[str, Any]:
    sdpa, operators = boundaries(case)
    eager = eager_boundaries(case)
    stages = ["embedding", *(f"layer_{index}" for index in range(LAYERS)), "final_norm"]
    return {
        "schema_version": 1,
        "case": case,
        "sources": {
            "modernbert": {
                "revision": ENCODER.REVISION,
                "sha256": ENCODER.SOURCE_SHA256,
                "attention_implementation": "sdpa",
            }
        },
        "sdpa_cpu_operators": operators,
        "sdpa_boundaries": {stage: sdpa[stage].tolist() for stage in stages},
        "eager_boundaries": {stage: eager[stage].tolist() for stage in stages},
        "eager_vs_sdpa_max_abs": {
            stage: max_abs(eager[stage], sdpa[stage]) for stage in stages
        },
    }


def report_native_comparison(
    source: dict[str, Any], native_path: Path, tolerance: float = 1e-5
) -> None:
    native = json.loads(native_path.read_text())
    first_sdpa = None
    first_eager = None
    for stage, expected_sdpa in source["sdpa_boundaries"].items():
        actual = native["boundaries"].get(stage)
        if actual is None:
            raise SystemExit(f"native diagnostic omitted {stage}")
        expected_values = torch.tensor(expected_sdpa, dtype=torch.float32)
        actual_values = torch.tensor(actual, dtype=torch.float32)
        if actual_values.shape != expected_values.shape:
            raise SystemExit(
                f"native {stage} shape {actual_values.shape} != {expected_values.shape}"
            )
        sdpa_difference = max_abs(actual_values, expected_values)
        eager_difference = max_abs(
            actual_values,
            torch.tensor(source["eager_boundaries"][stage], dtype=torch.float32),
        )
        print(
            f"{stage}: native-vs-SDPA max_abs={sdpa_difference:.8g} "
            f"native-vs-eager max_abs={eager_difference:.8g}"
        )
        if sdpa_difference > tolerance and first_sdpa is None:
            first_sdpa = stage
        if eager_difference > tolerance and first_eager is None:
            first_eager = stage
    print(f"first native-vs-SDPA crossing {tolerance:g}: {first_sdpa or 'none'}")
    print(f"first native-vs-eager crossing {tolerance:g}: {first_eager or 'none'}")


def report_native_outputs(native_path: Path) -> None:
    """Print the complete frozen-source error distribution without asserting it."""
    expected = {case["name"]: case["expected_hidden"] for case in records()["cases"]}
    native = json.loads(native_path.read_text())
    for case in native["cases"]:
        name = case["name"]
        expected_values = torch.tensor(expected[name], dtype=torch.float32).flatten()
        actual_values = torch.tensor(case["hidden"], dtype=torch.float32).flatten()
        if actual_values.shape != expected_values.shape:
            raise SystemExit(
                f"native {name} shape {actual_values.shape} != {expected_values.shape}"
            )
        differences = (actual_values - expected_values).abs()
        maximum, index = differences.max(dim=0)
        above = int((differences > 1e-5).sum().item())
        print(
            f"{name}: max_abs={maximum.item():.8g} index={index.item()} "
            f"above_1e-5={above}/{differences.numel()}"
        )


def cases() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "padded_base",
            "input_ids": [1, 2, 3, 4, 0, 0, 0, 0],
            "attention_mask": [True, True, True, True, False, False, False, False],
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
        },
        {
            "name": "padded_changed_rows",
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 0],
            "attention_mask": [True, True, True, True, False, False, False, False],
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
        },
        {
            "name": "unmasked_control",
            "input_ids": [1, 2, 3, 4, 5, 6, 7, 0],
            "attention_mask": [True] * 8,
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
        },
        {
            "name": "no_padding",
            "input_ids": [1, 2, 3, 4, 5, 6],
            "attention_mask": [True] * 6,
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
        },
        {
            "name": "masked_marker",
            "input_ids": [1, 2, 3, 4, 0, 0, 0, 0],
            "attention_mask": [True, True, True, True, False, False, False, False],
            "marker_pos": [1, 3, 2],
            "marker_mask": [True, True, False],
            "qtype": 2,
        },
    )


def records() -> dict[str, Any]:
    observed = []
    for case in cases():
        hidden, scores = source_case(case)
        observed.append(
            {
                **case,
                "expected_hidden": hidden.squeeze(0).tolist(),
                "expected_scores": scores.squeeze(0).tolist(),
            }
        )
    return {
        "schema_version": 1,
        "sources": {
            "julia_model": {
                "revision": "a85b127321d580d65176c89ced8273f305745d85",
                "sha256": HEAD.SOURCE_SHA256,
            },
            "modernbert": {
                "revision": ENCODER.REVISION,
                "sha256": ENCODER.SOURCE_SHA256,
                "attention_implementation": "sdpa",
            },
        },
        "operator_config": {
            "layers": 22,
            "positions_max": 8,
            "vocab_rows": VOCAB,
            "norm_eps": 1e-5,
            "norm_bias": False,
            "tolerance": 1e-5,
        },
        "weight_generation": "embedding rows ordinal=200; embedding/final norms 201/202 near one; layer i projections 6*i..6*i+3 and norms 300+2*i,301+2*i; Julia head uses head-reference ordinals",
        "cases": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        help="write source SDPA layer boundaries for the padded base case",
    )
    parser.add_argument(
        "--native-boundaries",
        type=Path,
        help="compare a test-private JuliaEncoder boundary capture against SDPA",
    )
    parser.add_argument(
        "--diagnostic-case",
        default="padded_base",
        help="named full fixture case to capture (default: padded_base)",
    )
    parser.add_argument(
        "--native-outputs",
        type=Path,
        help="report all frozen-source hidden errors from an opt-in native capture",
    )
    parser.add_argument(
        "--torch-config",
        action="store_true",
        help="print the source fixture Torch build configuration",
    )
    args = parser.parse_args()
    if args.torch_config:
        print(torch.__config__.show())
        return
    if args.native_outputs:
        report_native_outputs(args.native_outputs)
        return
    if args.diagnostic_output or args.native_boundaries:
        case = next(
            (case for case in cases() if case["name"] == args.diagnostic_case), None
        )
        if case is None:
            raise SystemExit(f"unknown diagnostic case: {args.diagnostic_case}")
        output = diagnostic(case)
        if args.diagnostic_output:
            args.diagnostic_output.parent.mkdir(parents=True, exist_ok=True)
            args.diagnostic_output.write_text(json.dumps(output, indent=2) + "\n")
        if args.native_boundaries:
            report_native_comparison(output, args.native_boundaries)
        return
    output = records()
    if args.write:
        FIXTURE.write_text(json.dumps(output, indent=2) + "\n")
    elif output != json.loads(FIXTURE.read_text()):
        raise SystemExit("full prefill fixture does not match pinned source")


if __name__ == "__main__":
    main()
