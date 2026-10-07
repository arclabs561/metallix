# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.13.0", "transformers==5.18.0"]
# ///
"""Capture actual Gemma4TextRouter rows from the pinned tiny MoE model.

Run with fixtures/gemma-4-tiny/moe-reference.json and an output JSON path.
These are independent CPU operator references, not full model qualification.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers.models.gemma4 import configuration_gemma4, modeling_gemma4

p = argparse.ArgumentParser()
p.add_argument("fixture", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()
fixture = json.loads(args.fixture.read_text())
torch.set_num_threads(1)
config = configuration_gemma4.Gemma4TextConfig(**fixture["config"])
config._attn_implementation = "eager"
model = modeling_gemma4.Gemma4ForCausalLM(config).to(torch.float32).eval()
state = {
    "model." + name: torch.tensor(t["values"], dtype=torch.float32).reshape(t["shape"])
    for name, t in fixture["tensors"].items()
    if not t.get("ignored", False)
}
state["lm_head.weight"] = state["model.embed_tokens.weight"]
model.load_state_dict(state, strict=True)
rows = []
handles = []
for index, layer in enumerate(model.model.layers):
    handles.append(
        layer.router.register_forward_pre_hook(
            lambda _m, inputs, layer=index: rows.append(
                (layer, inputs[0].detach().clone())
            )
        )
    )
with torch.inference_mode():
    model(
        input_ids=torch.tensor([fixture["cases"][0]["steps"][0]["input_ids"]]),
        use_cache=True,
    )
for handle in handles:
    handle.remove()
cases = []
with torch.inference_mode():
    for layer_index, inputs in rows:
        router = model.model.layers[layer_index].router
        for top_k in [1, 3, config.num_experts]:
            router.config.top_k_experts = top_k
            probabilities, weights, indices = router(inputs)
            cases.append(
                {
                    "layer": layer_index,
                    "top_k": top_k,
                    "input": inputs.tolist(),
                    "scale": router.scale.tolist(),
                    "projection": router.proj.weight.tolist(),
                    "expert_scale": router.per_expert_scale.tolist(),
                    "probabilities": probabilities.tolist(),
                    "indices": indices.tolist(),
                    "weights": weights.tolist(),
                }
            )
result = {
    "source": "https://github.com/huggingface/transformers/blob/a906d3c4b65095f2308b6a6a193e934d03b8eb5d/src/transformers/models/gemma4/modeling_gemma4.py",
    "runtime": "torch 2.13.0 CPU float32 eager one thread; transformers 5.18.0",
    "fixture": "gemma-4-tiny/moe-reference.json seed1001 first prompt",
    "hidden": config.hidden_size,
    "experts": config.num_experts,
    "intermediate": config.moe_intermediate_size,
    "epsilon": config.rms_norm_eps,
    "cases": cases,
}
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(f"captured {len(cases)} source router cases")
