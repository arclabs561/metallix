use std::path::PathBuf;

use serde_json::Value;

use super::{Adapter, Verdict, write_reports};

/// Qwen/Qwen3-0.6B@c1899de config.json.
const QWEN3_06B: &str = r#"{
  "architectures": ["Qwen3ForCausalLM"],
  "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 151643,
  "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "hidden_size": 1024,
  "initializer_range": 0.02, "intermediate_size": 3072, "max_position_embeddings": 40960,
  "max_window_layers": 28, "model_type": "qwen3", "num_attention_heads": 16,
  "num_hidden_layers": 28, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
  "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
  "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.51.0",
  "use_cache": true, "use_sliding_window": false, "vocab_size": 151936
}"#;

/// openbmb/MiniCPM5-2B@f974000 config.json.
const MINICPM5_2B: &str = r#"{
  "_name_or_path": "openbmb/MiniCPM5-2B", "architectures": ["LlamaForCausalLM"],
  "bos_token_id": 0, "eos_token_id": [1, 130073], "pad_token_id": 1, "hidden_act": "silu",
  "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 6144,
  "max_position_embeddings": 131072, "model_type": "llama", "num_attention_heads": 16,
  "num_hidden_layers": 42, "num_key_value_heads": 2, "head_dim": 128, "rms_norm_eps": 1e-06,
  "rope_theta": 5000000, "rope_scaling": null, "tie_word_embeddings": false,
  "torch_dtype": "bfloat16", "transformers_version": "5.6.2", "use_cache": true,
  "vocab_size": 130560
}"#;

/// A pplx-embed layout as the qwen crate's bidirectional tests write it.
const PPLX_QWEN3: &str = r#"{
  "architectures": ["PPLXQwen3Model"], "model_type": "bidirectional_pplx_qwen3",
  "use_bidirectional_attention": true, "num_hidden_layers": 28, "hidden_size": 1024,
  "intermediate_size": 3072, "vocab_size": 151671, "num_attention_heads": 16,
  "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-06, "rope_theta": 1000000, "hidden_act": "silu",
  "tie_word_embeddings": true, "attention_bias": false, "sliding_window": null,
  "use_sliding_window": false
}"#;

/// SupersonicLabs/Julia-1 top-level config.json (scripts/inspect-julia.py).
const JULIA_1: &str = r#"{
  "format_version": 1, "architecture": "JuliaDecisionModel",
  "julia_config_file": "julia_config.json", "encoder_config_file": "encoder/config.json",
  "weights_file": "model.safetensors", "tokenizer_directory": "tokenizer"
}"#;

/// Qwen/Qwen3.5-0.8B@2fc0636 config.json, layer list and vision tower kept.
const QWEN35_08B: &str = r#"{
  "architectures": ["Qwen3_5ForConditionalGeneration"], "image_token_id": 248056,
  "model_type": "qwen3_5",
  "text_config": {
    "attention_bias": false, "attention_dropout": 0.0, "attn_output_gate": true,
    "dtype": "bfloat16", "eos_token_id": 248044, "full_attention_interval": 4,
    "head_dim": 256, "hidden_act": "silu", "hidden_size": 1024, "initializer_range": 0.02,
    "intermediate_size": 3584,
    "layer_types": [
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention",
      "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ],
    "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
    "linear_num_value_heads": 16, "linear_value_head_dim": 128,
    "max_position_embeddings": 262144, "mlp_only_layers": [], "model_type": "qwen3_5_text",
    "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 8, "num_hidden_layers": 24, "num_key_value_heads": 2,
    "rms_norm_eps": 1e-06, "tie_word_embeddings": true, "use_cache": true,
    "vocab_size": 248320, "mamba_ssm_dtype": "float32",
    "rope_parameters": {
      "mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "default",
      "rope_theta": 10000000, "partial_rotary_factor": 0.25
    }
  },
  "tie_word_embeddings": true, "transformers_version": "4.57.0.dev0", "video_token_id": 248057,
  "vision_config": {
    "deepstack_visual_indexes": [], "depth": 12, "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 768, "in_channels": 3, "initializer_range": 0.02,
    "intermediate_size": 3072, "model_type": "qwen3_5", "num_heads": 12,
    "num_position_embeddings": 2304, "out_hidden_size": 1024, "patch_size": 16,
    "spatial_merge_size": 2, "temporal_patch_size": 2
  },
  "vision_end_token_id": 248054, "vision_start_token_id": 248053
}"#;

/// google/gemma-4-12B-it@707f0a3 text configuration, cut from 48 to 12
/// layers as in the gemma crate's tests.
const GEMMA4_12B: &str = r#"{
  "architectures": ["Gemma4UnifiedForConditionalGeneration"],
  "model_type": "gemma4_unified",
  "text_config": {
    "attention_bias": false, "attention_k_eq_v": true, "enable_moe_block": false,
    "final_logit_softcapping": 30.0, "global_head_dim": 512, "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 3840,
    "hidden_size_per_layer_input": 0, "intermediate_size": 15360,
    "layer_types": ["sliding_attention","sliding_attention","sliding_attention",
      "sliding_attention","sliding_attention","full_attention",
      "sliding_attention","sliding_attention","sliding_attention",
      "sliding_attention","sliding_attention","full_attention"],
    "max_position_embeddings": 262144, "model_type": "gemma4_unified_text",
    "num_attention_heads": 16, "num_global_key_value_heads": 1,
    "num_hidden_layers": 12, "num_key_value_heads": 8, "num_kv_shared_layers": 0,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
      "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
    },
    "sliding_window": 1024, "tie_word_embeddings": true,
    "use_bidirectional_attention": "vision", "vocab_size": 262144
  }
}"#;

/// The deepseek crate's V4.1 configuration projection.
const DEEPSEEK_V41: &str = r#"{
  "model_type":"deepseek_v41",
  "text_config":{"model_type":"deepseek_v41_text","num_hidden_layers":40,"n_routed_experts":384,"num_experts_per_tok":6,"engram_max_ngram_size":4},
  "quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}
}"#;

/// The shape of google/gemma-3-4b-it's config.json, vision tower omitted.
const GEMMA3_4B: &str = r#"{
  "architectures": ["Gemma3ForConditionalGeneration"], "boi_token_index": 255999,
  "eoi_token_index": 256000, "eos_token_id": [1, 106], "image_token_index": 262144,
  "initializer_range": 0.02, "mm_tokens_per_image": 256, "model_type": "gemma3",
  "text_config": {
    "hidden_size": 2560, "intermediate_size": 10240, "model_type": "gemma3_text",
    "num_hidden_layers": 34, "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
    "sliding_window": 1024
  },
  "torch_dtype": "bfloat16", "transformers_version": "4.50.0.dev0"
}"#;

/// The shape of a diffusers pipeline index (runwayml/stable-diffusion-v1-5).
const DIFFUSERS_INDEX: &str = r#"{
  "_class_name": "StableDiffusionPipeline", "_diffusers_version": "0.6.0",
  "feature_extractor": ["transformers", "CLIPImageProcessor"],
  "safety_checker": ["stable_diffusion", "StableDiffusionSafetyChecker"],
  "scheduler": ["diffusers", "PNDMScheduler"], "text_encoder": ["transformers", "CLIPTextModel"],
  "tokenizer": ["transformers", "CLIPTokenizer"], "unet": ["diffusers", "UNet2DConditionModel"],
  "vae": ["diffusers", "AutoencoderKL"]
}"#;

fn verdict(json: &str) -> Verdict {
    let document: Value = serde_json::from_str(json).expect("fixture is JSON");
    Verdict::of(json, &document)
}

/// Asserts the adapter and the `supported` value the command contract states.
fn assert_classified(json: &str, adapter: Option<Adapter>, supported: bool) -> Verdict {
    let verdict = verdict(json);
    assert_eq!(verdict.adapter(), adapter, "{verdict:?}");
    assert_eq!(verdict.supported(), supported, "{verdict:?}");
    verdict
}

#[test]
fn served_adapters_accept_their_published_configs() {
    assert_classified(QWEN3_06B, Some(Adapter::Qwen3), true);
    assert_classified(MINICPM5_2B, Some(Adapter::Llama), true);
    assert_classified(PPLX_QWEN3, Some(Adapter::PplxQwen3), true);
    assert_classified(JULIA_1, Some(Adapter::Julia), true);
}

#[test]
fn unserved_adapters_name_themselves_but_are_not_supported() {
    let qwen35 = assert_classified(QWEN35_08B, Some(Adapter::Qwen35), false);
    assert!(qwen35.reason().contains("no mx serve kind loads it"));
    let gemma4 = assert_classified(GEMMA4_12B, Some(Adapter::Gemma4), false);
    assert!(gemma4.reason().contains("vision encoder not loaded"));
    assert_classified(DEEPSEEK_V41, Some(Adapter::DeepseekV41), false);
}

#[test]
fn a_variant_the_owning_loader_refuses_keeps_its_reason() {
    let sliding = QWEN3_06B.replace(
        r#""use_sliding_window": false"#,
        r#""use_sliding_window": true"#,
    );
    let verdict = assert_classified(&sliding, Some(Adapter::Qwen3), false);
    assert!(
        verdict.reason().contains("sliding_window"),
        "{}",
        verdict.reason()
    );

    let format_2 = JULIA_1.replace(r#""format_version": 1"#, r#""format_version": 2"#);
    assert_classified(&format_2, Some(Adapter::Julia), false);
}

#[test]
fn unknown_architectures_are_unsupported_without_an_adapter() {
    let mistral = QWEN3_06B.replace(r#""model_type": "qwen3""#, r#""model_type": "mistral""#);
    let verdict = assert_classified(&mistral, None, false);
    assert!(verdict.reason().contains("\"mistral\""));

    let moe = QWEN35_08B
        .replace(
            r#""model_type": "qwen3_5","#,
            r#""model_type": "qwen3_5_moe","#,
        )
        .replace(
            r#""model_type": "qwen3_5_text""#,
            r#""model_type": "qwen3_5_moe_text""#,
        );
    // The qwen35 adapter reads qwen3_5_moe too; a dense config relabelled as
    // MoE lacks the expert fields its loader requires.
    let moe = assert_classified(&moe, Some(Adapter::Qwen35), false);
    assert!(moe.reason().contains("num_experts"), "{moe:?}");

    let gemma3 = assert_classified(GEMMA3_4B, None, false);
    assert!(gemma3.reason().contains("\"gemma3\""));
}

#[test]
fn a_diffusers_index_names_its_pipeline_class() {
    let verdict = assert_classified(DIFFUSERS_INDEX, None, false);
    assert!(verdict.reason().contains("StableDiffusionPipeline"));
}

#[test]
fn json_lines_follow_the_contract_and_io_errors_exit_2() {
    let dir = std::env::temp_dir().join(format!("mx-supports-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let qwen = dir.join("qwen3.json");
    let gemma = dir.join("gemma4.json");
    let broken = dir.join("broken.json");
    std::fs::write(&qwen, QWEN3_06B).expect("write fixture");
    std::fs::write(&gemma, GEMMA4_12B).expect("write fixture");
    std::fs::write(&broken, "{not json").expect("write fixture");
    let missing = dir.join("missing.json");

    let mut out = Vec::new();
    let status = write_reports(&[qwen.clone(), gemma.clone()], true, &mut out).expect("write");
    assert_eq!(status, 0);
    let lines: Vec<Value> = String::from_utf8(out)
        .expect("utf-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("one JSON object per line"))
        .collect();
    assert_eq!(lines.len(), 2);
    let mut keys: Vec<&str> = lines[0]
        .as_object()
        .expect("object")
        .keys()
        .map(String::as_str)
        .collect();
    let mut expected = [
        "path",
        "supported",
        "adapter",
        "model_type",
        "architectures",
        "reason",
    ];
    expected.sort_unstable();
    keys.sort_unstable();
    assert_eq!(keys, expected);
    assert_eq!(lines[0]["path"], qwen.display().to_string());
    assert_eq!(lines[0]["supported"], true);
    assert_eq!(lines[0]["adapter"], "qwen3");
    assert_eq!(lines[0]["model_type"], "qwen3");
    assert_eq!(
        lines[0]["architectures"],
        serde_json::json!(["Qwen3ForCausalLM"])
    );
    assert_eq!(lines[1]["supported"], false);
    assert_eq!(lines[1]["adapter"], "gemma4");

    let paths: Vec<PathBuf> = vec![broken, qwen, missing];
    let mut out = Vec::new();
    assert_eq!(write_reports(&paths, true, &mut out).expect("write"), 2);
    let lines: Vec<Value> = String::from_utf8(out)
        .expect("utf-8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("one JSON object per line"))
        .collect();
    assert_eq!(
        lines.len(),
        3,
        "a line for every path, failed ones included"
    );
    assert_eq!(lines[0]["supported"], false);
    assert_eq!(lines[0]["adapter"], Value::Null);
    assert_eq!(lines[1]["supported"], true);
    assert_eq!(lines[2]["supported"], false);

    let mut out = Vec::new();
    write_reports(&paths[1..2], false, &mut out).expect("write");
    let human = String::from_utf8(out).expect("utf-8");
    assert_eq!(human.lines().count(), 1);
    assert!(human.contains("supported (qwen3)"), "{human}");
    std::fs::remove_dir_all(&dir).expect("remove temp dir");
}

/// Adds a config field after the opening brace.
fn with_field(json: &str, field: &str) -> String {
    json.replacen('{', &format!("{{ {field},"), 1)
}

#[test]
fn quantized_configs_are_refused_while_the_gate_ignores_quantization() {
    // mlx-community/Qwen3-0.6B-4bit writes both fields.
    let mlx = with_field(
        &with_field(
            QWEN3_06B,
            r#""quantization": {"group_size": 64, "bits": 4}"#,
        ),
        r#""quantization_config": {"group_size": 64, "bits": 4}"#,
    );
    let verdict = assert_classified(&mlx, Some(Adapter::Qwen3), false);
    let reason = verdict.reason();
    assert!(
        reason.contains("4-bit") && reason.contains("group size 64"),
        "{reason}"
    );

    // Qwen/Qwen3-4B-FP8's quantization_config.
    let fp8 = with_field(
        QWEN3_06B,
        r#""quantization_config": {"activation_scheme": "dynamic", "modules_to_not_convert": ["lm_head"], "quant_method": "fp8", "weight_block_size": [128, 128]}"#,
    );
    let verdict = assert_classified(&fp8, Some(Adapter::Qwen3), false);
    assert!(
        verdict.reason().contains(r#""fp8""#),
        "{}",
        verdict.reason()
    );

    let llama = with_field(
        MINICPM5_2B,
        r#""quantization": {"group_size": 64, "bits": 4, "mode": "affine"}"#,
    );
    assert_classified(&llama, Some(Adapter::Llama), false);

    let unquantized = with_field(QWEN3_06B, r#""quantization_config": null"#);
    assert_classified(&unquantized, Some(Adapter::Qwen3), true);
}

#[test]
fn a_gate_that_reads_quantization_keeps_its_own_verdict() {
    assert_eq!(
        verdict(DEEPSEEK_V41),
        Verdict::Accepted(Adapter::DeepseekV41)
    );
}
