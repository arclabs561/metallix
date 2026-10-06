# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Offline tests for the component inventory, on recorded configs and excerpts."""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hub_components

# config.json of Cloudflare/clef at 2f3de3d (qwen3_5), trimmed to the keys
# the inventory reads; layer_types shortened to one 4-layer period.
QWEN35: dict = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "vision_config": {"model_type": "qwen3_5_vision"},
    "text_config": {
        "model_type": "qwen3_5_text",
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        "linear_conv_kernel_dim": 4,
        "linear_num_value_heads": 48,
        "mamba_ssm_dtype": "float32",
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    },
}
# config.json of Infatoshi/GLM-5.3-UNCENSORED-EXL3-3.0bpw (glm_moe_dsa), trimmed.
GLM_MOE_DSA: dict = {
    "model_type": "glm_moe_dsa",
    "hidden_act": "silu",
    "index_topk": 2048,
    "kv_lora_rank": 512,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "norm_topk_prob": True,
    "num_attention_heads": 64,
    "num_experts_per_tok": 8,
    "num_key_value_heads": 64,
    "num_nextn_predict_layers": 1,
    "q_lora_rank": 2048,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
    "scoring_func": "sigmoid",
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
}
# google/gemma-4-31B-it text_config, trimmed; rope keyed by layer type.
GEMMA4: dict = {
    "model_type": "gemma4",
    "tie_word_embeddings": True,
    "vision_config": {"model_type": "gemma4_vision"},
    "text_config": {
        "model_type": "gemma4_text",
        "attention_k_eq_v": True,
        "enable_moe_block": False,
        "final_logit_softcapping": 30.0,
        "hidden_activation": "gelu_pytorch_tanh",
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "num_attention_heads": 32,
        "num_experts": None,
        "num_key_value_heads": 16,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
        "sliding_window": 1024,
    },
}
# PleIAs/baguettotron-600m (llama), trimmed; no tie_word_embeddings key in
# the second copy below to exercise the Transformers default.
LLAMA: dict = {
    "model_type": "llama",
    "hidden_act": "silu",
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000,
    "tie_word_embeddings": True,
}
LLAMA_8B: dict = {
    "model_type": "llama",
    "hidden_act": "silu",
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {"factor": 8.0, "rope_type": "llama3"},
}

# Lines from transformers v5.18.0 models/qwen3_5/modeling_qwen3_5.py
# (299, 1123) and models/qwen3/modeling_qwen3.py (237).
QWEN35_MODELING = """\
# chunk_gated_delta_rule in a comment does not count
@use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")
class Qwen3_5VisionModel(Qwen3_5PreTrainedModel):
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
"""
# Lines 46, 57 and 68 of Cloudflare/clef chat_template.jinja (shortened).
CLEF_TEMPLATE = """\
{%- if enable_thinking is undefined or enable_thinking is true %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '\\n\\nIf you choose to call a function ONLY reply in the following format: <tool_call>\\n<function=example_function_name>' }}
"""

# Shapes of real crates/models lines: a comment, a JSON test fixture, code.
RUST = {
    "crates/models/qwen35/src/forward.rs": (
        "// mrope is not applied for text-only prompts\n"
        "let out = gated_delta_net(weights, base)?;\n"
    ),
    "crates/models/gemma/src/lib.rs": (
        '"sliding_window": 3, "vocab_size": 8,\n'
        "if let Some(cap) = config.final_logit_softcapping() {\n"
        "Gemma4LayerKind::Sliding => Some(config.sliding_window()),\n"
    ),
}


class ConfigComponents(unittest.TestCase):
    def test_hybrid_vision_language_model(self):
        c = hub_components.config_components(QWEN35)
        self.assertEqual(c["gqa"]["value"], "24/4")
        self.assertEqual(c["gqa"]["source"], "text_config.num_key_value_heads")
        self.assertEqual(
            c["layer_pattern"]["value"], {"linear_attention": 3, "full_attention": 1}
        )
        self.assertEqual(c["gated_delta"]["value"], 48)
        self.assertEqual(c["mrope"]["value"], [11, 11, 10])
        self.assertEqual(c["partial_rotary"]["value"], 0.25)
        self.assertEqual(c["attn_output_gate"]["value"], True)
        self.assertEqual(c["vision_tower"]["value"], "qwen3_5_vision")
        self.assertEqual(c["tied_embeddings"]["value"], False)
        self.assertNotIn("sliding_window", c)
        self.assertNotIn("moe", c)

    def test_mla_moe_with_indexer_and_mtp(self):
        c = hub_components.config_components(GLM_MOE_DSA)
        self.assertEqual(c["mla"]["value"], "kv_lora_rank 512")
        self.assertNotIn("mha", c)
        self.assertEqual(c["moe"]["value"], "256 experts, top-8")
        self.assertEqual(c["shared_expert"]["source"], "n_shared_experts")
        self.assertEqual(c["dsa_indexer"]["value"], 2048)
        self.assertEqual(c["mtp"]["value"], 1)
        self.assertEqual(c["moe_router"]["value"]["scoring_func"], "sigmoid")

    def test_per_layer_rope_sliding_softcap_and_disabled_moe(self):
        c = hub_components.config_components(GEMMA4)
        self.assertEqual(c["sliding_window"]["value"], 1024)
        self.assertEqual(
            c["rope_proportional"]["source"],
            "text_config.rope_parameters.full_attention",
        )
        self.assertEqual(c["partial_rotary"]["value"], 0.25)
        self.assertEqual(c["geglu"]["value"], "gelu_pytorch_tanh")
        self.assertEqual(c["logit_softcap"]["value"], 30.0)
        self.assertNotIn("moe", c)
        self.assertIn(
            "text_config.attention_k_eq_v", hub_components.unrecognized_keys(GEMMA4)
        )

    def test_rope_scaling_and_tied_default(self):
        c = hub_components.config_components(LLAMA_8B)
        self.assertEqual(c["rope_llama3"]["value"], 8.0)
        self.assertEqual(c["tied_embeddings"]["value"], True)
        self.assertIn("default", c["tied_embeddings"]["source"])
        self.assertEqual(hub_components.unrecognized_keys(LLAMA), [])


class CodeAndTemplate(unittest.TestCase):
    def test_code_signals_skip_comments_and_merge_with_config(self):
        code = hub_components.code_components(QWEN35_MODELING, "modeling_qwen3_5.py")
        self.assertEqual(code["gated_delta"]["source"], "modeling_qwen3_5.py:2")
        self.assertEqual(code["vision_tower"]["source"], "modeling_qwen3_5.py:3")
        merged = hub_components.merge(hub_components.config_components(QWEN35), code)
        self.assertEqual(merged["gated_delta"]["confirmed_by"], "modeling_qwen3_5.py:2")
        self.assertEqual(merged["qk_norm"]["source"], "modeling_qwen3_5.py:4")

    def test_template_features(self):
        t = hub_components.template_components(
            CLEF_TEMPLATE, "clef chat_template.jinja"
        )
        self.assertEqual(
            t["template_enable_thinking"]["source"], "clef chat_template.jinja:1"
        )
        self.assertEqual(t["template_tools_argument"]["source"][-2:], ":2")
        self.assertIn("template_tool_call_tag", t)
        self.assertIn("template_tool_call_function_tags", t)
        self.assertNotIn("template_think_tags", t)

    def test_tokenizer_config_template_forms(self):
        read = hub_components.template_from_tokenizer_config
        self.assertEqual(read('{"chat_template": "{{ x }}"}'), "{{ x }}")
        listed = '{"chat_template": [{"name": "tool_use", "template": "a"}, {"name": "default", "template": "b"}]}'
        self.assertEqual(read(listed), "b")
        self.assertIsNone(read('{"model_max_length": 8}'))

    def test_modeling_lookup_names(self):
        self.assertEqual(
            hub_components.modeling_dirs("qwen3_5_text"), ["qwen3_5_text", "qwen3_5"]
        )
        remote = {
            "auto_map": {"AutoModelForCausalLM": "modeling_xing4_0.Xing4_0ForCausalLM"}
        }
        self.assertEqual(
            hub_components.remote_modeling_file(remote), "modeling_xing4_0.py"
        )
        elsewhere = {"auto_map": {"AutoModel": "owner/repo--modeling_x.Model"}}
        self.assertIsNone(hub_components.remote_modeling_file(elsewhere))


class Evidence(unittest.TestCase):
    def test_hits_skip_comments_and_json_fixtures(self):
        ev = hub_components.metallix_evidence(
            RUST,
            ["mrope", "gated_delta", "sliding_window", "logit_softcap", "vision_tower"],
        )
        self.assertEqual(ev["mrope"], [])
        self.assertEqual(ev["vision_tower"], [])
        self.assertEqual(len(ev["sliding_window"]), 1)
        self.assertTrue(
            ev["sliding_window"][0].startswith("crates/models/gemma/src/lib.rs:3 ")
        )
        self.assertTrue(
            ev["gated_delta"][0].startswith("crates/models/qwen35/src/forward.rs:2")
        )

    def test_rope_types_match_only_near_rope(self):
        sources = {"crates/models/a/src/lib.rs": "let y = linear(&x, w)?;\n"}
        self.assertEqual(
            hub_components.metallix_evidence(sources, ["rope_linear"]),
            {"rope_linear": []},
        )
        sources = {
            "crates/models/a/src/lib.rs": "RopeKind::Linear { factor } => scale,\n"
        }
        self.assertEqual(
            len(
                hub_components.metallix_evidence(sources, ["rope_linear"])[
                    "rope_linear"
                ]
            ),
            1,
        )

    def test_gaps_are_used_components_without_hits(self):
        components = hub_components.config_components(QWEN35)
        ev = hub_components.metallix_evidence(RUST, components)
        gaps = hub_components.gaps(components, ev)
        self.assertIn("mrope", gaps)
        self.assertIn("vision_tower", gaps)
        self.assertNotIn("gated_delta", gaps)
        self.assertNotIn("layer_pattern", gaps)  # no Rust pattern, not judged


class Variants(unittest.TestCase):
    def test_sizes_and_template_do_not_change_signature(self):
        small = hub_components.config_components(LLAMA)
        large = hub_components.config_components(LLAMA | {"num_attention_heads": 32})
        large["template_think_tags"] = {"value": True, "source": "x"}
        self.assertEqual(
            hub_components.signature(small), hub_components.signature(large)
        )

    def test_nearest_family_names_the_difference(self):
        known = {
            "llama": hub_components.signature(hub_components.config_components(LLAMA)),
            "glmmoedsa": hub_components.signature(
                hub_components.config_components(GLM_MOE_DSA)
            ),
        }
        sig = hub_components.signature(hub_components.config_components(LLAMA_8B))
        name, only_here, only_there = hub_components.nearest_family(sig, known)
        self.assertEqual((name, only_here, only_there), ("llama", ["rope_llama3"], []))

    def test_render_marks_config_only_variants_and_denominators(self):
        families = [
            {
                "family": "qwen2",
                "weight": 308,
                "adapter": None,
                "components": {},
                "gaps": ["vision_tower"],
                "nearest": ("llama", [], []),
                "unrecognized": [],
            },
            {
                "family": "glm5next",
                "weight": 584.1,
                "adapter": None,
                "components": {},
                "gaps": ["mla", "vision_tower"],
                "nearest": ("deepseekv41", ["mla"], []),
                "unrecognized": ["text_config.mhc"],
            },
        ]
        text = hub_components.render(families, {"mla": []}, 19354.7)
        self.assertIn("- qwen2 has the same component set as llama", text)
        self.assertIn(
            "| vision_tower | qwen2, glm5next | 892.1 / 19354.7 (4.6%) |", text
        )
        self.assertIn("| mla | glm5next | 584.1 / 19354.7 (3.0%) |", text)
        self.assertIn(
            "| glm5next | 584.1 / 19354.7 (3.0%) | - | mla, vision_tower |", text
        )


if __name__ == "__main__":
    unittest.main()
