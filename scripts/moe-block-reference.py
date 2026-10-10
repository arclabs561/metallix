# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.13.0", "transformers==5.18.0"]
# ///
"""Capture Gemma 4 routed-expert block references for the shared MoE layer.

Runs transformers' own Gemma4TextRouter and Gemma4TextExperts on CPU, one
thread, with seeded weights at H 16, E 8, I 4, and writes the router logits
(captured at the projection), the chosen indices and weights, the expert
input and the routed output, in float32 and float64. It is an operator
reference for the layer, not model qualification.

Before generating cases, a probe records which index torch.topk keeps on an
exact tie; the tie cases are built so two experts' logits are bitwise equal
and straddle the top-k boundary.

Usage: uv run scripts/moe-block-reference.py OUTPUT.json
"""

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers.models.gemma4 import configuration_gemma4, modeling_gemma4

HIDDEN, EXPERTS, INTERMEDIATE, EPSILON = 16, 8, 4, 1e-6

# (name, top_k, tokens, seed); 3 x 30 = 90 selections, past the 64 at which
# the layer sorts rows by expert.
CASES = [
    ("k1_n1", 1, 1, 11),
    ("k1_n4", 1, 4, 12),
    ("k3_n1", 3, 1, 13),
    ("k3_n4", 3, 4, 14),
    ("k3_n30_sorted", 3, 30, 15),
    ("k8_n4_all_experts", 8, 4, 16),
]
# Router inputs whose first eight elements become the logits exactly (see
# tie_router): two equal values sit at ranks k and k + 1.
TIES = [
    ("tie_k1", 1, [3.0, 1.0, 3.0, 0.5, -1.0, 2.0, -2.0, 0.0]),
    ("tie_k3", 3, [5.0, 1.0, 4.0, 2.5, 0.0, 2.5, -1.0, -2.0]),
]


def tie_rule() -> str:
    """Which of two equal values torch.topk keeps when only one fits."""
    kept = torch.topk(torch.tensor([1.0, 3.0, 0.0, 3.0]), 1).indices.item()
    if kept == 1:
        return "lowest_index"
    if kept == 3:
        return "highest_index"
    raise RuntimeError(f"unexpected tie result {kept}")


def config(top_k: int) -> configuration_gemma4.Gemma4TextConfig:
    return configuration_gemma4.Gemma4TextConfig(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        top_k_experts=top_k,
        moe_intermediate_size=INTERMEDIATE,
        rms_norm_eps=EPSILON,
        hidden_activation="gelu_pytorch_tanh",
        enable_moe_block=True,
    )


def seeded(module: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            values = torch.randn(parameter.shape, generator=generator) * 0.5
            if name.endswith("scale"):  # scale and per_expert_scale
                values = 1.0 + 0.25 * values
            parameter.copy_(values)


def tie_router(router: modeling_gemma4.Gemma4TextRouter) -> None:
    """Make the logits equal the router input's first eight elements, up to
    one common factor: unit scale and unit-vector projection rows. Equal
    inputs then give bitwise-equal logits."""
    with torch.no_grad():
        router.scale.fill_(1.0)
        router.proj.weight.zero_()
        for expert in range(EXPERTS):
            router.proj.weight[expert, expert] = 1.0


def run(router, experts, router_input, expert_input, dtype):
    router, experts = router.to(dtype), experts.to(dtype)
    logits = []
    handle = router.proj.register_forward_hook(
        lambda _m, _i, out: logits.append(out.detach().clone())
    )
    with torch.inference_mode():
        _, weights, indices = router(router_input.to(dtype))
        output = experts(expert_input.to(dtype), indices, weights)
    handle.remove()
    return logits[0], indices, weights, output


def record(name, top_k, router, experts, router_input, expert_input):
    logits32, indices32, weights32, output32 = run(
        router, experts, router_input, expert_input, torch.float32
    )
    logits64, indices64, weights64, output64 = run(
        router, experts, router_input, expert_input, torch.float64
    )
    router.to(torch.float32)
    experts.to(torch.float32)
    return {
        "name": name,
        "top_k": top_k,
        "tokens": router_input.shape[0],
        "router_input": router_input.tolist(),
        "expert_input": expert_input.tolist(),
        "per_expert_scale": router.per_expert_scale.tolist(),
        "gate_up": experts.gate_up_proj.tolist(),
        "down": experts.down_proj.tolist(),
        "logits": logits32.tolist(),
        "indices": indices32.tolist(),
        "weights": weights32.tolist(),
        "output": output32.tolist(),
        "logits_float64": logits64.tolist(),
        "indices_float64": indices64.tolist(),
        "weights_float64": weights64.tolist(),
        "output_float64": output64.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rule = tie_rule()
    cases = []
    for name, top_k, tokens, seed in CASES:
        router = modeling_gemma4.Gemma4TextRouter(config(top_k)).eval()
        experts = modeling_gemma4.Gemma4TextExperts(config(top_k)).eval()
        seeded(router, seed)
        seeded(experts, seed + 1000)
        generator = torch.Generator().manual_seed(seed + 2000)
        router_input = torch.randn(tokens, HIDDEN, generator=generator)
        expert_input = torch.randn(tokens, HIDDEN, generator=generator)
        cases.append(record(name, top_k, router, experts, router_input, expert_input))
    for name, top_k, values in TIES:
        router = modeling_gemma4.Gemma4TextRouter(config(top_k)).eval()
        experts = modeling_gemma4.Gemma4TextExperts(config(top_k)).eval()
        seeded(router, 77)
        seeded(experts, 1077)
        tie_router(router)
        router_input = torch.tensor([values + [0.0] * (HIDDEN - EXPERTS)])
        expert_input = torch.randn(
            1, HIDDEN, generator=torch.Generator().manual_seed(78)
        )
        case = record(name, top_k, router, experts, router_input, expert_input)
        tied = [i for i, v in enumerate(values) if values.count(v) == 2]
        logits = case["logits"][0]
        if logits[tied[0]] != logits[tied[1]]:
            raise RuntimeError(f"{name}: logits not bitwise equal at {tied}")
        case["tied_experts"] = tied
        cases.append(case)
    result = {
        "source": "transformers 5.18.0 models/gemma4/modeling_gemma4.py "
        "Gemma4TextRouter and Gemma4TextExperts",
        "runtime": f"torch {torch.__version__} CPU, one thread; "
        f"transformers {transformers.__version__}",
        "script": "scripts/moe-block-reference.py",
        "hidden": HIDDEN,
        "experts": EXPERTS,
        "intermediate": INTERMEDIATE,
        "epsilon": EPSILON,
        "activation": "gelu_pytorch_tanh",
        "tie_rule": rule,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
