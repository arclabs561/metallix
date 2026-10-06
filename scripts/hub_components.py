# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Component inventory of model families, and where metallix implements each.

For a family's config.json (plus, when available, its Transformers or Hub
modeling file) this lists the building blocks the architecture uses:
attention kind and pattern, positional encoding, MLP and MoE routing, norms,
heads, multimodal towers. Every field names its source: a config key or a
modeling-file line. Each component is then matched against the Rust sources
under crates/models by pattern, and the hit (or its absence) is reported as
evidence; whether a hit is a full implementation is a reviewer's call.

Pure functions over decoded configs and source text; `hub_trending.py
components` does the file and network I/O.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

# Config keys every decoder carries; anything else in a text config is listed
# as an unrecognized key so new mechanisms (hyper-connections, n-gram
# embeddings) surface even before a detector exists for them.
ORDINARY_KEYS = {
    "architectures", "model_type", "dtype", "torch_dtype", "transformers_version",
    "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "head_dim", "vocab_size", "max_position_embeddings",
    "initializer_range", "use_cache", "bos_token_id", "eos_token_id",
    "pad_token_id", "attention_bias", "attention_dropout", "mlp_bias",
    "hidden_act", "hidden_activation", "rms_norm_eps", "layer_norm_eps",
    "tie_word_embeddings", "rope_theta", "rope_scaling", "rope_parameters",
    "partial_rotary_factor", "layer_types", "sliding_window",
    "use_sliding_window", "max_window_layers", "auto_map", "quantization_config",
    "output_router_logits", "router_aux_loss_coef", "pretraining_tp",
    "text_config", "vision_config", "audio_config", "image_token_id",
    "video_token_id", "vision_start_token_id", "vision_end_token_id",
    "num_experts", "n_routed_experts", "num_local_experts", "num_experts_per_tok",
    "moe_intermediate_size", "shared_expert_intermediate_size", "n_shared_experts",
    "norm_topk_prob", "first_k_dense_replace", "decoder_sparse_step",
    "mlp_only_layers", "moe_layer_freq", "scoring_func", "topk_method",
    "routed_scaling_factor", "n_group", "topk_group", "ep_size",
    "kv_lora_rank", "q_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
    "v_head_dim", "qk_head_dim", "final_logit_softcapping",
    "attn_logit_softcapping", "num_nextn_predict_layers", "mtp_num_hidden_layers",
    "full_attention_interval", "linear_conv_kernel_dim", "linear_key_head_dim",
    "linear_num_key_heads", "linear_num_value_heads", "linear_value_head_dim",
    "attn_output_gate", "use_qk_norm", "qk_norm", "index_topk", "index_n_heads",
    "index_head_dim", "hc_mult", "hc_count", "ple_embed_dim",
    "hidden_size_per_layer_input", "add_swa_attention_sink_bias",
    "add_full_attention_sink_bias", "linear_attn_config", "output_gate_type",
    "indexer_budget", "enable_moe_block", "top_k_experts", "moe_num_experts",
    "num_mtp_layers", "no_rope_layers", "nope_layer_interval", "qk_layernorm",
}  # fmt: skip

# Python patterns for a modeling file. The first matching line is the
# provenance; a match confirms or adds a component the config implies.
CODE_SIGNALS = {
    "qk_norm": r"self\.(q_norm|q_layernorm|query_norm)\s*=",
    "attention_sinks": r"self\.sinks\s*=|\bs_aux\b|attention_sink",
    "logit_softcap": r"softcap",
    "zero_centered_norm": r"\(1(\.0)?\s*\+\s*self\.weight",
    "sandwich_norm": r"pre_feedforward_layernorm|post_feedforward_layernorm",
    "gated_delta": r"chunk_gated_delta_rule|GatedDeltaNet",
    "short_conv": r"short_conv|ShortConv",
    "qkv_bias": r"q_proj = nn\.Linear\(.*bias=True",
    "moe": r"class \w*(SparseMoe|Moe|MoE)\w*\(|self\.experts\s*=",
    "shared_expert": r"self\.shared_experts?\s*=",
    "mla": r"kv_a_proj_with_mqa|kv_lora_rank",
    "dsa_indexer": r"class \w*Indexer\w*\(|self\.indexer\s*=",
    "mtp": r"class \w*(MTP|Mtp|NextN)\w*\(",
    "mrope": r"mrope_section|apply_multimodal_rotary",
    "partial_rotary": r"partial_rotary_factor|rotary_dim",
    "per_layer_embeddings": r"per_layer_input",
    "hyper_connections": r"hc_mult|hyper_connection|sinkhorn",
    "vision_tower": r"class \w*Vision(Model|Transformer|Encoder)\w*\(",
    "audio_tower": r"class \w*Audio(Model|Encoder|Tower)\w*\(",
    "mm_projector": r"multi_modal_projector|class \w*(Projector|PatchMerger)\w*\(",
}

# Rust patterns searched in crates/models/*/src (comments and tests skipped).
# A hit is evidence of a mention, to be read before calling it implemented.
RUST_SIGNALS = {
    "gqa": r"num_key_value_heads|num_kv_heads|n_kv_heads",
    "mqa": r"num_key_value_heads|num_kv_heads|n_kv_heads",
    "mla": r"kv_lora_rank",
    "sliding_window": r"sliding_window",
    "gated_delta": r"gated_delta|GatedDelta",
    "short_conv": r"short_conv|ShortConv",
    "qk_norm": r"\bq_norm\b|qk_norm",
    "qkv_bias": r"qkv_bias|attention_bias|q_bias",
    "attention_sinks": r"attn_sink|attention_sink|\bsinks\b",
    "attn_output_gate": r"attn_output_gate|output_gate",
    "dsa_indexer": r"index_topk|Indexer",
    "rope_yarn": r"(?i)yarn",
    "rope_llama3": r"(?i)llama3",
    "mrope": r"(?i)mrope",
    "partial_rotary": r"partial_rotary|rotary_dim",
    "nope_layers": r"no_rope|nope_layer",
    "swiglu": r"(?i)silu|swiglu",
    "geglu": r"(?i)gelu",
    "moe": r"n_routed_experts|num_experts|experts_per_tok",
    "shared_expert": r"shared_expert",
    "rms_norm": r"rms_norm|RmsNorm",
    "zero_centered_norm": r"zero_centered|unit_offset",
    "sandwich_norm": r"pre_feedforward_layernorm|post_feedforward_layernorm",
    "logit_softcap": r"softcap",
    "mtp": r"\bmtp\b|nextn",
    "per_layer_embeddings": r"per_layer_input",
    "hyper_connections": r"hc_mult|hyper_conn",
    "vision_tower": r"vision_config|VisionTower|vision_tower|patch_embed",
    "audio_tower": r"audio_config|AudioEncoder|audio_tower",
    "mm_projector": r"(?i)projector|merger",
}


# Chat-template features the serving side has to parse or render.
TEMPLATE_SIGNALS = {
    "tool_call_tag": r"<tool_call>",
    "tool_call_function_tags": r"<function=",
    "tool_call_section_tokens": r"<\|tool_calls?_(section_)?begin\|>",
    "tool_call_brackets": r"\[TOOL_CALLS\]",
    "think_tags": r"<think>",
    "enable_thinking": r"enable_thinking",
    "reasoning_effort": r"reasoning_effort",
    "channels": r"<\|channel\|>",
    "tools_argument": r"\btools\b",
    "media_placeholders": r"image|video|audio",
}


def text_config(config: dict) -> tuple[dict, str]:
    """The decoder's config and the key prefix it lives under."""
    inner = config.get("text_config")
    if isinstance(inner, dict) and inner:
        return inner, "text_config."
    return config, ""


def first_of(config: dict, keys: Iterable[str]):
    for key in keys:
        if config.get(key) not in (None, 0, False, [], {}):
            return key, config[key]
    return None, None


def rope_entries(text: dict, prefix: str) -> list[tuple[str, dict]]:
    """(source, rope dict) pairs; rope_parameters may be keyed by layer type."""
    for key in ("rope_parameters", "rope_scaling"):
        value = text.get(key)
        if isinstance(value, dict) and value:
            if all(isinstance(v, dict) for v in value.values()):
                return [(f"{prefix}{key}.{k}", v) for k, v in value.items()]
            return [(f"{prefix}{key}", value)]
    return []


def config_components(config: dict) -> dict[str, dict]:
    """Components a config declares, as {component: {"value", "source"}}."""
    text, p = text_config(config)
    out: dict[str, dict] = {}

    def put(name, value, source):
        out[name] = {"value": value, "source": source}

    heads, kv = text.get("num_attention_heads"), text.get("num_key_value_heads")
    if text.get("kv_lora_rank"):
        put("mla", f"kv_lora_rank {text['kv_lora_rank']}", f"{p}kv_lora_rank")
    elif heads and kv and kv == 1:
        put("mqa", f"{heads}/1", f"{p}num_key_value_heads")
    elif heads and kv and kv < heads:
        put("gqa", f"{heads}/{kv}", f"{p}num_key_value_heads")
    elif heads:
        put("mha", f"{heads}/{kv or heads}", f"{p}num_attention_heads")

    if text.get("attention_bias"):
        put("qkv_bias", True, f"{p}attention_bias")

    layer_types = text.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        counts: dict[str, int] = {}
        for kind in layer_types:
            counts[str(kind)] = counts.get(str(kind), 0) + 1
        put("layer_pattern", counts, f"{p}layer_types")
    sliding = text.get("sliding_window")
    uses_sliding = sliding and text.get("use_sliding_window") is not False
    if uses_sliding or "sliding_attention" in (layer_types or []):
        put("sliding_window", sliding, f"{p}sliding_window")
    if text.get("linear_conv_kernel_dim") and text.get("linear_num_value_heads"):
        put("gated_delta", text["linear_num_value_heads"], f"{p}linear_num_value_heads")
    linear = text.get("linear_attn_config")
    if isinstance(linear, dict):
        put("linear_attention", sorted(linear), f"{p}linear_attn_config")
    if any(str(k) in ("conv", "short_conv") for k in layer_types or []):
        put("short_conv", True, f"{p}layer_types")
    key, value = first_of(text, ("use_qk_norm", "qk_norm", "qk_layernorm"))
    if key:
        put("qk_norm", value, p + key)
    if text.get("attn_output_gate") or text.get("output_gate_type"):
        key = "attn_output_gate" if text.get("attn_output_gate") else "output_gate_type"
        put("attn_output_gate", text[key], p + key)
    key, value = first_of(text, ("index_topk", "indexer_budget"))
    if key:
        put("dsa_indexer", value, p + key)
    key, value = first_of(
        text, ("add_swa_attention_sink_bias", "add_full_attention_sink_bias")
    )
    if key:
        put("attention_sinks", value, p + key)

    for source, rope in rope_entries(text, p):
        rope_type = rope.get("rope_type") or rope.get("type") or "default"
        if rope_type != "default":
            put(f"rope_{rope_type}", rope.get("factor", True), source)
        if rope.get("mrope_section"):
            put("mrope", rope["mrope_section"], f"{source}.mrope_section")
        if rope.get("partial_rotary_factor", 1) != 1:
            put("partial_rotary", rope["partial_rotary_factor"], source)
    if text.get("partial_rotary_factor", 1) not in (1, None):
        put(
            "partial_rotary", text["partial_rotary_factor"], f"{p}partial_rotary_factor"
        )
    key, value = first_of(text, ("no_rope_layers", "nope_layer_interval"))
    if key:
        put("nope_layers", value, p + key)

    act = text.get("hidden_act") or text.get("hidden_activation")
    act_key = p + ("hidden_act" if text.get("hidden_act") else "hidden_activation")
    if act in ("silu", "swish"):
        put("swiglu", act, act_key)
    elif act and str(act).startswith("gelu"):
        put("geglu", act, act_key)
    elif act:
        put(f"activation_{act}", act, act_key)

    key, experts = first_of(
        text,
        ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"),
    )
    if key and text.get("enable_moe_block") is not False:
        _, top_k = first_of(text, ("num_experts_per_tok", "top_k_experts", "moe_topk"))
        put("moe", f"{experts} experts, top-{top_k}", p + key)
        key, value = first_of(
            text, ("n_shared_experts", "shared_expert_intermediate_size")
        )
        if key:
            put("shared_expert", value, p + key)
        router = {
            k: text[k]
            for k in (
                "scoring_func",
                "topk_method",
                "norm_topk_prob",
                "routed_scaling_factor",
            )
            if k in text
        }
        if router:
            put("moe_router", router, p + ",".join(router))

    # Only rms_norm_eps names its norm; layer_norm_eps and layernorm_epsilon
    # appear on RMSNorm models too, so other norms are left to the code.
    if text.get("rms_norm_eps"):
        put("rms_norm", text["rms_norm_eps"], f"{p}rms_norm_eps")
    tie = config.get("tie_word_embeddings", text.get("tie_word_embeddings"))
    put(
        "tied_embeddings",
        True if tie is None else tie,
        "tie_word_embeddings"
        if tie is not None
        else "absent (Transformers default true)",
    )
    for key in ("final_logit_softcapping", "attn_logit_softcapping"):
        if text.get(key):
            put("logit_softcap", text[key], p + key)
    key, value = first_of(
        text, ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers")
    )
    if key:
        put("mtp", value, p + key)
    key, value = first_of(text, ("hc_mult", "hc_count"))
    if key:
        put("hyper_connections", value, p + key)
    key, value = first_of(text, ("hidden_size_per_layer_input", "ple_embed_dim"))
    if key:
        put("per_layer_embeddings", value, p + key)
    for tower, key in (
        ("vision_tower", "vision_config"),
        ("audio_tower", "audio_config"),
    ):
        sub = config.get(key)
        if isinstance(sub, dict):
            put(tower, sub.get("model_type") or True, key)
    return out


def unrecognized_keys(config: dict) -> list[str]:
    text, p = text_config(config)
    return sorted(p + k for k in text if k not in ORDINARY_KEYS)


def code_components(source: str, origin: str) -> dict[str, dict]:
    """Components a modeling file shows, each with its first matching line."""
    out = {}
    lines = source.splitlines()
    for name, pattern in CODE_SIGNALS.items():
        regex = re.compile(pattern)
        for number, line in enumerate(lines, 1):
            if regex.search(line) and not line.lstrip().startswith("#"):
                out[name] = {"value": True, "source": f"{origin}:{number}"}
                break
    return out


def template_components(template: str, origin: str) -> dict[str, dict]:
    """Chat-template features, each with its first matching line."""
    out = {}
    lines = template.splitlines()
    for name, pattern in TEMPLATE_SIGNALS.items():
        regex = re.compile(pattern)
        for number, line in enumerate(lines, 1):
            if regex.search(line):
                out[f"template_{name}"] = {
                    "value": True,
                    "source": f"{origin}:{number}",
                }
                break
    return out


def template_from_tokenizer_config(text: str) -> str | None:
    """chat_template from tokenizer_config.json: a string, or a list of
    {name, template} whose "default" entry (else the first) is used."""
    value = json.loads(text).get("chat_template")
    if isinstance(value, list) and value:
        named = {v.get("name"): v.get("template") for v in value if isinstance(v, dict)}
        value = named.get("default") or next(iter(named.values()), None)
    return value if isinstance(value, str) else None


def merge(config_part: dict, code_part: dict) -> dict[str, dict]:
    """Config fields win; code adds components the config does not name and
    marks those it confirms."""
    out = {k: dict(v) for k, v in config_part.items()}
    for name, hit in code_part.items():
        if name in out:
            out[name]["confirmed_by"] = hit["source"]
        else:
            out[name] = hit
    return out


def rust_sources(root: Path) -> dict[str, str]:
    """crates/models/*/src/**/*.rs, keyed by repo-relative path."""
    out = {}
    for path in sorted((root / "crates/models").glob("*/src/**/*.rs")):
        relative = str(path.relative_to(root))
        if "/tests/" not in relative and not relative.endswith("tests.rs"):
            out[relative] = path.read_text()
    return out


def rust_pattern(name: str) -> str | None:
    """The Rust pattern for a component; rope types match their own name."""
    if name in RUST_SIGNALS:
        return RUST_SIGNALS[name]
    if name.startswith("rope_"):
        kind = re.escape(name.removeprefix("rope_"))
        return rf"(?i)rope.*\b{kind}\b|\b{kind}\b.*rope"
    return None


def metallix_evidence(
    sources: dict[str, str], names: Iterable[str]
) -> dict[str, list[str]]:
    """First non-comment hit per crate for each named component that has a
    pattern, as path:line plus the start of the line."""
    out: dict[str, list[str]] = {}
    for name in sorted(set(names)):
        pattern = rust_pattern(name)
        if pattern is None:
            continue
        regex, hits, crates = re.compile(pattern), [], set()
        for path, text in sources.items():
            crate = path.split("/")[2] if path.startswith("crates/models/") else path
            if crate in crates:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                stripped = line.lstrip()
                # Comments and JSON fixtures inside unit tests are not code.
                fixture = re.search(r'"\w+"\s*:', line)
                if stripped.startswith("//") or fixture or not regex.search(line):
                    continue
                hits.append(f"{path}:{number} `{stripped[:60]}`")
                crates.add(crate)
                break
        out[name] = hits
    return out


def gaps(components: dict[str, dict], evidence: dict[str, list[str]]) -> list[str]:
    """Components the family uses that no metallix source mentions.

    Components without a Rust pattern (layer_pattern, moe_router, tied
    embeddings, unusual activations) are not judged here.
    """
    return sorted(
        name
        for name, component in components.items()
        if component["value"] not in (False, None)
        and name in evidence
        and not evidence[name]
    )


def signature(components: dict[str, dict]) -> frozenset[str]:
    """Component names with sizes dropped, for config-only variant matching.
    Chat-template features are left out: they are serving-side, not
    architecture."""
    skip = {"tied_embeddings", "moe_router"}
    names = {
        n
        for n, c in components.items()
        if n not in skip
        and not n.startswith("template_")
        and c["value"] not in (False, None)
    }
    pattern = components.get("layer_pattern", {}).get("value") or {}
    return frozenset(names | {f"layer:{k}" for k in pattern})


def nearest_family(sig: frozenset[str], known: dict[str, frozenset[str]]):
    """(family, only in this one, only in that one) with the smallest symmetric
    difference; empty differences mean a config-only variant."""
    best = None
    for name, other in sorted(known.items()):
        diff = (sorted(sig - other), sorted(other - sig))
        if best is None or len(diff[0]) + len(diff[1]) < len(best[1]) + len(best[2]):
            best = (name, *diff)
    return best


def modeling_dirs(model_type: str) -> list[str]:
    """Transformers model directories to try for a model_type."""
    out = [model_type]
    if model_type.endswith("_text"):
        out.append(model_type.removesuffix("_text"))
    return out


def remote_modeling_file(config: dict) -> str | None:
    """The Hub modeling file auto_map names (`modeling_x.Class`, or
    `owner/repo--modeling_x.Class` for code hosted in another repo, which is
    skipped)."""
    auto_map = config.get("auto_map") or {}
    for key in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        target = auto_map.get(key)
        if isinstance(target, list):
            target = target[0] if target else None
        if isinstance(target, str) and "--" not in target and "." in target:
            return target.rsplit(".", 1)[0] + ".py"
    return None


def share(part: float, whole: float) -> str:
    return f"{part:g} / {whole:g} ({100 * part / whole:.1f}%)" if whole else f"{part:g}"


def render(families: list[dict], evidence: dict[str, list[str]], total: float) -> str:
    """Markdown: gap table, component reuse table, per-family inventories.

    `families` entries carry family, weight, adapter, representative,
    modeling (origin or why missing), components, unrecognized, gaps and
    nearest (family, only here, only there) as built by the components phase.
    """
    lines = [
        "# Component inventory of trending families",
        "",
        (
            f"Weights are summed trendingScore of text repos, share of {total:g}. "
            "Missing means no non-comment line in crates/models/*/src matches the "
            "component's pattern; a hit is a mention to read, not proof of an "
            "implementation."
        ),
        "",
        "## Gap table",
        "",
        (
            "| Family | Weight | Adapter | Missing (grep) | Nearest adapter family: "
            "extra here / lacking here | Unrecognized keys |"
        ),
        "|---|---:|---|---|---|---:|",
    ]
    for f in families:
        near = f.get("nearest")
        near_text = (
            f"{near[0]}: +{', '.join(near[1]) or 'none'} / -{', '.join(near[2]) or 'none'}"
            if near
            else "n/a"
        )
        lines.append(
            f"| {f['family']} | {share(f['weight'], total)} | {f.get('adapter') or '-'} | "
            f"{', '.join(f.get('gaps') or []) or 'none'} | {near_text} | "
            f"{len(f.get('unrecognized') or [])} |"
        )
    variants = [
        f
        for f in families
        if f.get("nearest") and not f["nearest"][1] and not f["nearest"][2]
    ]
    lines += ["", "## Config-only variants", ""]
    if variants:
        lines += [
            f"- {f['family']} has the same component set as {f['nearest'][0]}; "
            "differences are sizes and flags (check unrecognized keys)."
            for f in variants
        ]
    else:
        lines.append(
            "None: every inventoried family differs from its nearest adapter family."
        )
    need: dict[str, list[dict]] = {}
    for f in families:
        for name in f.get("gaps") or []:
            need.setdefault(name, []).append(f)
    ranked = sorted(need.items(), key=lambda kv: -sum(f["weight"] for f in kv[1]))
    lines += [
        "",
        "## Component reuse (missing components by summed family weight)",
        "",
        "| Component | Families | Summed weight |",
        "|---|---|---:|",
    ]
    for name, members in ranked:
        weight = sum(f["weight"] for f in members)
        names = ", ".join(f["family"] for f in members)
        lines.append(f"| {name} | {names} | {share(weight, total)} |")
    lines += [
        "",
        "## Metallix evidence per component",
        "",
        "| Component | Hits |",
        "|---|---|",
    ]
    for name, hits in sorted(evidence.items()):
        cell = "; ".join(h.replace("|", "\\|") for h in hits) or "none"
        lines.append(f"| {name} | {cell} |")
    lines += ["", "## Inventories", ""]
    for f in families:
        lines += [
            f"### {f['family']}",
            "",
            (
                f"Representative: {f.get('representative') or 'none with a config'}; "
                f"modeling: {f.get('modeling') or 'n/a'}."
            ),
            "",
        ]
        for name, c in sorted((f.get("components") or {}).items()):
            confirmed = (
                f"; confirmed {c['confirmed_by']}" if "confirmed_by" in c else ""
            )
            lines.append(f"- {name}: {c['value']} ({c['source']}{confirmed})")
        if f.get("unrecognized"):
            lines.append(f"- unrecognized config keys: {', '.join(f['unrecognized'])}")
        lines.append("")
    return "\n".join(lines)
