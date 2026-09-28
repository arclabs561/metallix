#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0"]
# ///
"""CPU-only, hash-gated synthetic numerical reference for Julia's decision head.

This executes the inspected ``JuliaDecisionModel`` class from a pinned local
source file with a fake encoder and deterministic synthetic F32 weights.  A
separate transparent implementation spells out the two pre-norm attention
blocks and scorer, so it does not call the source forward path twice.  It does
not load an encoder checkpoint or start a model service.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / ".agents/receipts/julia/model.py"
FIXTURE_PATH = ROOT / "fixtures/julia-1/head-reference.json"
SOURCE_SHA256 = "ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81"
WIDTH = 384
MAX_SOURCE_BYTES = 64 * 1024
RTOL = 1e-5
ATOL = 1e-5


class FakeEncoder(nn.Module):
    """Returns caller-supplied synthetic hidden states without loading a backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=WIDTH, reference_compile=False)
        self.hidden: torch.Tensor | None = None

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        del input_ids, attention_mask
        if self.hidden is None:
            raise RuntimeError("synthetic encoder hidden state was not set")
        return SimpleNamespace(last_hidden_state=self.hidden)


def read_pinned_source(path: Path = SOURCE_PATH) -> bytes:
    with path.open("rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("pinned source must be a regular file")
        size = os.fstat(source.fileno()).st_size
        if size > MAX_SOURCE_BYTES:
            raise ValueError(f"pinned source exceeds {MAX_SOURCE_BYTES} bytes")
        data = source.read(MAX_SOURCE_BYTES + 1)
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError(f"pinned source exceeds {MAX_SOURCE_BYTES} bytes")
    actual = hashlib.sha256(data).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError(f"refusing source SHA {actual}; expected {SOURCE_SHA256}")
    return data


def load_source_model() -> type[nn.Module]:
    """Extract only the pinned model class; never import its remote module."""
    source = read_pinned_source()
    parsed = ast.parse(source, filename=str(SOURCE_PATH))
    model_class = next(
        (
            node
            for node in parsed.body
            if isinstance(node, ast.ClassDef) and node.name == "JuliaDecisionModel"
        ),
        None,
    )
    if model_class is None or model_class.decorator_list:
        raise ValueError("pinned JuliaDecisionModel class is missing or decorated")
    isolated = ast.fix_missing_locations(
        ast.Module(body=[model_class], type_ignores=[])
    )
    namespace: dict[str, Any] = {"torch": torch, "nn": nn, "F": F}
    exec(compile(isolated, str(SOURCE_PATH), "exec"), namespace)  # noqa: S102
    extracted = namespace.get("JuliaDecisionModel")
    if not isinstance(extracted, type) or not issubclass(extracted, nn.Module):
        raise TypeError("AST extraction did not produce a module class")
    return extracted


def synthetic_hidden(*, perturb_padding: bool = False) -> torch.Tensor:
    values = torch.arange(6 * WIDTH, dtype=torch.float32).reshape(1, 6, WIDTH)
    hidden = ((values * 7).remainder(29) - 14) / 20
    if perturb_padding:
        perturbation = torch.arange(2 * WIDTH, dtype=torch.float32).reshape(1, 2, WIDTH)
        hidden[:, 4:] = hidden[:, 4:] + ((perturbation * 11).remainder(31) - 15) / 3
    return hidden


def fill_synthetic_parameters(model: nn.Module) -> None:
    """Use explicit F32 values, independent of PyTorch initialization RNG."""
    with torch.no_grad():
        for ordinal, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(
                parameter
            )
            parameter.copy_(((values + ordinal * 17).remainder(97) - 48) / 1000)


def build_source_oracle() -> tuple[nn.Module, FakeEncoder]:
    model_class = load_source_model()
    encoder = FakeEncoder()
    model = model_class(encoder, head_layers=2, n_act=2, dropout=0.1)
    fill_synthetic_parameters(model)
    model.eval()
    validate_source_configuration(model)
    return model, encoder


def validate_source_configuration(model: nn.Module) -> None:
    head = model.head
    if head is None or len(head.layers) != 2:
        raise ValueError("pinned source head must have exactly two layers")
    for layer in head.layers:
        if (
            not layer.norm_first
            or not layer.self_attn.batch_first
            or layer.self_attn.num_heads != 6
            or layer.linear1.out_features != 4 * WIDTH
            or layer.norm1.eps != 1e-5
            or layer.norm2.eps != 1e-5
            or layer.dropout.p != 0.1
            or layer.activation is not F.relu
        ):
            raise ValueError("pinned source decision-head configuration changed")
    scorer = model.scorer
    if not isinstance(scorer[0], nn.LayerNorm) or scorer[0].eps != 1e-5:
        raise ValueError("pinned source scorer LayerNorm configuration changed")
    if not isinstance(scorer[2], nn.GELU):
        raise TypeError("pinned source scorer activation changed")


def layer_norm(value: torch.Tensor, layer: nn.LayerNorm) -> torch.Tensor:
    return F.layer_norm(value, (WIDTH,), layer.weight, layer.bias, layer.eps)


def transparent_layer(
    value: torch.Tensor, layer: nn.TransformerEncoderLayer, padding: torch.Tensor
) -> torch.Tensor:
    """Explicit F32 pre-norm bidirectional TransformerEncoderLayer math."""
    normalized = layer_norm(value, layer.norm1)
    batch, tokens, _ = normalized.shape
    attention = layer.self_attn
    qkv = F.linear(normalized, attention.in_proj_weight, attention.in_proj_bias)
    query, key, val = qkv.chunk(3, dim=-1)
    heads = attention.num_heads
    head_width = WIDTH // heads

    def split(item: torch.Tensor) -> torch.Tensor:
        return item.reshape(batch, tokens, heads, head_width).transpose(1, 2)

    query, key, val = split(query), split(key), split(val)
    scores = query @ key.transpose(-2, -1) / math.sqrt(head_width)
    scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
    probabilities = scores.softmax(dim=-1)
    attended = probabilities @ val
    attended = attended.transpose(1, 2).reshape(batch, tokens, WIDTH)
    residual = value + F.linear(
        attended, attention.out_proj.weight, attention.out_proj.bias
    )
    normalized = layer_norm(residual, layer.norm2)
    feed_forward = F.linear(normalized, layer.linear1.weight, layer.linear1.bias)
    feed_forward = F.relu(feed_forward)
    feed_forward = F.linear(feed_forward, layer.linear2.weight, layer.linear2.bias)
    return residual + feed_forward


def transparent_scores(
    model: nn.Module,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    marker_pos: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
) -> torch.Tensor:
    value = hidden + model.type_emb(qtype)[:, None, :]
    padding = ~attention_mask.bool()
    for layer in model.head.layers:
        value = transparent_layer(value, layer, padding)
    positions = marker_pos[:, :, None].expand(-1, -1, WIDTH)
    markers = value.gather(1, positions)
    scorer = model.scorer
    scores = F.layer_norm(
        markers, (WIDTH,), scorer[0].weight, scorer[0].bias, scorer[0].eps
    )
    scores = F.linear(scores, scorer[1].weight, scorer[1].bias)
    scores = F.gelu(scores)
    scores = F.linear(scores, scorer[3].weight, scorer[3].bias).squeeze(-1).float()
    return scores.masked_fill(~marker_mask, -1e4)


def run_case(
    model: nn.Module, encoder: FakeEncoder, case: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    encoder.hidden = synthetic_hidden(
        perturb_padding=case.get("perturb_padding", False)
    )
    attention_mask = torch.tensor([case["attention_mask"]], dtype=torch.bool)
    marker_pos = torch.tensor([case["marker_pos"]], dtype=torch.int64)
    marker_mask = torch.tensor([case["marker_mask"]], dtype=torch.bool)
    qtype = torch.tensor([case["qtype"]], dtype=torch.int64)
    input_ids = torch.arange(6, dtype=torch.int64).unsqueeze(0)
    with torch.inference_mode():
        source = model(input_ids, attention_mask, marker_pos, marker_mask, qtype)
        reference = transparent_scores(
            model, encoder.hidden, attention_mask, marker_pos, marker_mask, qtype
        )
    return source, reference


def records() -> dict[str, Any]:
    model, encoder = build_source_oracle()
    cases = (
        {
            "name": "base",
            "attention_mask": [True, True, True, True, False, False],
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
        },
        {
            "name": "padding_perturbation",
            "attention_mask": [True, True, True, True, False, False],
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
            "perturb_padding": True,
        },
        {
            "name": "option_permutation",
            "attention_mask": [True, True, True, True, False, False],
            "marker_pos": [3, 1],
            "marker_mask": [True, True],
            "qtype": 2,
        },
        {
            "name": "unmasked_padding_control",
            "attention_mask": [True, True, True, True, True, True],
            "marker_pos": [1, 3],
            "marker_mask": [True, True],
            "qtype": 2,
            "perturb_padding": True,
        },
        {
            "name": "masked_marker",
            "attention_mask": [True, True, True, True, False, False],
            "marker_pos": [1, 3, 2],
            "marker_mask": [True, True, False],
            "qtype": 2,
        },
    )
    output = []
    for case in cases:
        source, reference = run_case(model, encoder, case)
        if not torch.allclose(source, reference, rtol=RTOL, atol=ATOL):
            raise AssertionError(f"transparent reference differs for {case['name']}")
        output.append({**case, "expected_scores": source.squeeze(0).tolist()})
    return {
        "schema_version": 1,
        "source": {
            "revision": "a85b127321d580d65176c89ced8273f305745d85",
            "path": "julia/model.py",
            "sha256": SOURCE_SHA256,
            "symbol": "JuliaDecisionModel",
        },
        "runtime": {"torch": "2.13.0", "device": "cpu", "dtype": "float32"},
        "operator_config": {
            "width": WIDTH,
            "head_layers": 2,
            "attention_heads": 6,
            "feed_forward_width": 1536,
            "dropout": 0.1,
            "norm_first": True,
            "activation": "relu",
            "scorer_activation": "gelu",
            "layer_norm_eps": 1e-5,
            "causal": False,
            "invalid_marker_score": -10000.0,
        },
        "synthetic_inputs": "arange(6*384), ((x*7)%29-14)/20; deterministic parameter fill ((x+ordinal*17)%97-48)/1000",
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "cases": output,
    }


def verify(fixture: dict[str, Any]) -> None:
    if torch.__version__ != fixture["runtime"]["torch"]:
        raise RuntimeError(f"requires exact torch {fixture['runtime']['torch']}")
    if fixture["source"]["sha256"] != SOURCE_SHA256:
        raise ValueError("fixture does not name the required pinned source")
    actual = records()
    for field in (
        "schema_version",
        "source",
        "runtime",
        "operator_config",
        "synthetic_inputs",
        "tolerances",
    ):
        if fixture[field] != actual[field]:
            raise AssertionError(f"frozen fixture {field} differs from source record")
    tolerance = fixture["tolerances"]
    if len(fixture["cases"]) != len(actual["cases"]):
        raise AssertionError("fixture case count differs from source record")
    for expected, observed in zip(fixture["cases"], actual["cases"], strict=True):
        expected_inputs = {
            field: value
            for field, value in expected.items()
            if field != "expected_scores"
        }
        observed_inputs = {
            field: value
            for field, value in observed.items()
            if field != "expected_scores"
        }
        if expected_inputs != observed_inputs:
            raise AssertionError("fixture case inputs differ from source record")
        source = torch.tensor(observed["expected_scores"], dtype=torch.float32)
        frozen = torch.tensor(expected["expected_scores"], dtype=torch.float32)
        if not torch.allclose(
            source, frozen, rtol=tolerance["rtol"], atol=tolerance["atol"]
        ):
            raise AssertionError(
                f"source output differs from frozen {expected['name']} vector"
            )
        if expected["name"] == "masked_marker" and float(source[-1]) != -10000.0:
            raise AssertionError("masked marker must receive the invalid score")
    actual_cases = {case["name"]: case["expected_scores"] for case in actual["cases"]}
    if not torch.allclose(
        torch.tensor(actual_cases["base"]),
        torch.tensor(actual_cases["padding_perturbation"]),
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    ):
        raise AssertionError("padding perturbation changed valid marker scores")
    if not torch.allclose(
        torch.tensor(actual_cases["base"]),
        torch.tensor(actual_cases["option_permutation"]).flip(0),
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    ):
        raise AssertionError("option permutation did not permute marker scores")
    if torch.allclose(
        torch.tensor(actual_cases["base"]),
        torch.tensor(actual_cases["unmasked_padding_control"]),
        rtol=tolerance["rtol"],
        atol=tolerance["atol"],
    ):
        raise AssertionError(
            "unmasked padding control did not change valid marker scores"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit", action="store_true", help="print capture payload for fixture review"
    )
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()
    if args.emit:
        print(json.dumps(records(), indent=2))
        return
    fixture = json.loads(args.fixture.read_text())
    verify(fixture)
    print(
        json.dumps(
            {
                "status": "passed",
                "cases": [case["name"] for case in fixture["cases"]],
                "scope": "synthetic CPU decision-head source-oracle versus transparent F32 reference; no encoder checkpoint",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
