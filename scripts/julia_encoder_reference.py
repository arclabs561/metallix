#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch==2.13.0", "transformers==5.0.0"]
# ///
"""Pinned Transformers-5.0 ModernBERT CPU reference for one Julia encoder block.

The calculation is intentionally spelled out rather than importing a moving
Transformers installation.  It hash-gates the exact implementation URL before
capturing a synthetic F32 fixture.  It never downloads a model checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures/julia-1/encoder-reference.json"
REVISION = "08810b1e278938278c50153ee1edfd7a20a759da"
URL = f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/src/transformers/models/modernbert/modeling_modernbert.py"
SOURCE_SHA256 = "83875f54a029339c62a8f5061801873d41e134b3e9abb8308b8e9b0f9f57b5dc"
WIDTH, HEADS, HEAD_DIM, FF, THETA = 384, 6, 64, 1152, 160000.0
MAX_SOURCE_BYTES = 1024 * 1024


def source_bytes() -> bytes:
    with urllib.request.urlopen(URL, timeout=20) as response:
        payload = response.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise RuntimeError("pinned ModernBERT implementation exceeds source bound")
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("pinned ModernBERT implementation hash changed")
    required = (
        b"class ModernBertEncoderLayer",
        b"self.attn_norm = nn.Identity()",
        b"self.mlp(self.mlp_norm(hidden_states))",
        b"def rotate_half",
    )
    if any(item not in payload for item in required):
        raise RuntimeError("pinned source no longer contains audited encoder contract")
    return payload


def values(shape: tuple[int, ...], ordinal: int) -> torch.Tensor:
    count = 1
    for item in shape:
        count *= item
    return (
        (
            torch.arange(count, dtype=torch.float32).reshape(shape) + ordinal * 17
        ).remainder(97)
        - 48
    ) / 1000


def weights(ordinal: int = 0) -> dict[str, torch.Tensor]:
    return {
        "wqkv": values((3 * WIDTH, WIDTH), ordinal),
        "attn_wo": values((WIDTH, WIDTH), ordinal + 1),
        "wi": values((2 * FF, WIDTH), ordinal + 2),
        "mlp_wo": values((WIDTH, FF), ordinal + 3),
        "attn_norm": values((WIDTH,), ordinal + 4),
        "mlp_norm": values((WIDTH,), ordinal + 5),
    }


def hidden(tokens: int, perturb: bool = False) -> torch.Tensor:
    x = (
        (
            torch.arange(tokens * WIDTH, dtype=torch.float32).reshape(tokens, WIDTH) * 7
        ).remainder(29)
        - 14
    ) / 20
    if perturb:
        x[-1:] += (
            (torch.arange(WIDTH, dtype=torch.float32).reshape(1, WIDTH) * 11).remainder(
                31
            )
            - 15
        ) / 3
    return x


def rope(item: torch.Tensor) -> torch.Tensor:
    # item [tokens, heads, 64], precisely the source split-half rotation.
    positions = torch.arange(item.shape[0], dtype=torch.float32)[:, None]
    inv = 1.0 / THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    freqs = positions * inv[None, :]
    cos = torch.cat((freqs, freqs), dim=-1).cos()[:, None, :]
    sin = torch.cat((freqs, freqs), dim=-1).sin()[:, None, :]
    return (
        item * cos
        + torch.cat((-item[..., HEAD_DIM // 2 :], item[..., : HEAD_DIM // 2]), dim=-1)
        * sin
    )


def transparent_block(
    x: torch.Tensor, mask: torch.Tensor, layer: int, w: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Independent spelling of the pinned layer arithmetic, checked against source below."""
    attn_x = x if layer == 0 else F.layer_norm(x, (WIDTH,), w["attn_norm"], None, 1e-5)
    qkv = F.linear(attn_x, w["wqkv"]).reshape(x.shape[0], 3, HEADS, HEAD_DIM)
    q, k, v = qkv.unbind(1)
    q, k = rope(q), rope(k)
    scores = torch.einsum("qhd,khd->hqk", q, k) / HEAD_DIM**0.5
    allowed = mask[None, :].expand(x.shape[0], -1).clone()
    if layer % 3:
        positions = torch.arange(x.shape[0])
        allowed &= (positions[:, None] - positions[None, :]).abs() <= 64
    # The source's additive-mask builder uses finfo.min, including for an
    # entirely masked local query.  Do not substitute -inf here.
    scores = scores.masked_fill(~allowed[None, :, :], torch.finfo(scores.dtype).min)
    attended = torch.einsum(
        "hqk,khd->qhd", scores.softmax(-1, dtype=torch.float32), v
    ).reshape_as(x)
    residual = x + F.linear(attended, w["attn_wo"])
    normed = F.layer_norm(residual, (WIDTH,), w["mlp_norm"], None, 1e-5)
    first, gate = F.linear(normed, w["wi"]).chunk(2, -1)
    return residual + F.linear(F.gelu(first, approximate="none") * gate, w["mlp_wo"])


def source_block(
    x: torch.Tensor, mask: torch.Tensor, layer_id: int, w: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Execute the installed, hash-gated Transformers 5.0 encoder layer itself."""
    import inspect

    import transformers.models.modernbert.modeling_modernbert as source_module
    from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertEncoderLayer,
    )

    source_path = Path(inspect.getsourcefile(source_module) or "")
    if (
        not source_path.is_file()
        or hashlib.sha256(source_path.read_bytes()).hexdigest() != SOURCE_SHA256
    ):
        raise RuntimeError(
            "installed ModernBERT implementation does not match pinned source"
        )
    config = ModernBertConfig(
        hidden_size=WIDTH,
        num_hidden_layers=22,
        num_attention_heads=HEADS,
        intermediate_size=FF,
        attention_bias=False,
        mlp_bias=False,
        norm_bias=False,
        norm_eps=1e-5,
        hidden_activation="gelu",
        attention_dropout=0.0,
        mlp_dropout=0.0,
        global_attn_every_n_layers=3,
        local_attention=128,
        max_position_embeddings=8192,
        rope_parameters={
            "full_attention": {"rope_theta": THETA, "rope_type": "default"},
            "sliding_attention": {"rope_theta": THETA, "rope_type": "default"},
        },
    )
    config._attn_implementation = "eager"
    module = ModernBertEncoderLayer(config, layer_id=layer_id).eval()
    with torch.no_grad():
        module.attn.Wqkv.weight.copy_(w["wqkv"])
        module.attn.Wo.weight.copy_(w["attn_wo"])
        module.mlp.Wi.weight.copy_(w["wi"])
        module.mlp.Wo.weight.copy_(w["mlp_wo"])
        if layer_id != 0:
            module.attn_norm.weight.copy_(w["attn_norm"])
        module.mlp_norm.weight.copy_(w["mlp_norm"])
    positions = torch.arange(x.shape[0], dtype=torch.float32)[:, None]
    inv = 1.0 / THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    freqs = positions * inv[None, :]
    cos = torch.cat((freqs, freqs), dim=-1).cos()[None, :, :]
    sin = torch.cat((freqs, freqs), dim=-1).sin()[None, :, :]
    allowed = mask[None, :].expand(x.shape[0], -1).clone()
    local = (
        torch.arange(x.shape[0])[:, None] - torch.arange(x.shape[0])[None, :]
    ).abs() <= 64
    sliding = allowed & local
    floor = torch.finfo(x.dtype).min
    full_mask = torch.where(allowed, 0.0, floor)[None, None, :, :]
    local_mask = torch.where(sliding, 0.0, floor)[None, None, :, :]
    return module(
        x[None, :, :],
        attention_mask=full_mask,
        sliding_window_mask=local_mask,
        position_embeddings=(cos, sin),
    )[0][0]


def digest(value: torch.Tensor) -> str:
    # The uv fixture environment intentionally needs only torch, not NumPy.
    return hashlib.sha256(
        b"".join(struct.pack("<f", item) for item in value.flatten().tolist())
    ).hexdigest()


def capture() -> dict:
    source_bytes()
    w = weights()
    cases = []
    for name, tokens, layer, mask, perturb, observed_positions in (
        ("global_padding", 8, 0, [True] * 6 + [False] * 2, False, 6),
        ("global_padding_perturbed", 8, 0, [True] * 6 + [False] * 2, True, 6),
        ("global_unmasked_control", 8, 0, [True] * 8, True, 6),
        ("local_window_crossing", 66, 1, [True] * 66, False, 66),
        ("global_window_control", 66, 3, [True] * 66, False, 66),
        ("local_distant_perturbation", 66, 1, [True] * 66, True, 1),
        ("global_distant_perturbation", 66, 3, [True] * 66, True, 1),
        # Query zero has no permitted local key; the source's F32-min additive
        # mask therefore produces a finite uniform softmax over all values.
        ("local_all_masked_query", 66, 1, [False] * 65 + [True], False, 1),
    ):
        # Distant perturbations intentionally touch only position 65, so the
        # local row-zero observation cannot read them while global can.
        state = hidden(tokens, perturb)
        transparent = transparent_block(state, torch.tensor(mask), layer, w)
        source = source_block(state, torch.tensor(mask), layer, w)
        if not torch.allclose(source, transparent, rtol=1e-5, atol=1e-5):
            raise RuntimeError(
                f"source layer and independent reference disagree for {name}"
            )
        observed = source[:observed_positions]
        cases.append(
            {
                "name": name,
                "positions": tokens,
                "layer": layer,
                "attention_mask": mask,
                "perturb_padding": perturb,
                "observed_positions": observed.shape[0],
                "output_sha256": digest(observed),
                "expected_output": observed.tolist(),
            }
        )
    return {
        "schema_version": 1,
        "source": {
            "revision": REVISION,
            "url": URL,
            "sha256": SOURCE_SHA256,
            "implementation": "ModernBertEncoderLayer/eager_attention_forward",
        },
        "operator_config": {
            "width": WIDTH,
            "attention_heads": HEADS,
            "head_dim": HEAD_DIM,
            "intermediate_width": FF,
            "rope_theta": THETA,
            "norm_eps": 1e-5,
            "attention_bias": False,
            "mlp_bias": False,
            "global_every": 3,
            "local_radius": 64,
            "activation": "gelu",
        },
        "weight_generation": "((flat_index + ordinal*17) % 97 - 48) / 1000; ordinals Wqkv=0, attention.Wo=1, mlp.Wi=2, mlp.Wo=3, attn_norm=4, mlp_norm=5",
        "input_generation": "((flat_index*7)%29-14)/20",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    captured = capture()
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(captured, indent=2) + "\n")
    else:
        if captured != json.loads(FIXTURE.read_text()):
            raise SystemExit("encoder fixture does not match pinned source capture")


if __name__ == "__main__":
    main()
