//! Checkpoints in mlx-lm's layout with MLX affine quantization: the loader's
//! contract on a synthetic checkpoint, and opt-in bit-for-bit parity of every
//! quantized module and `MoE` block with mlx-lm.

use std::{collections::HashMap, fmt::Write, fs, path::PathBuf};

use mlx_rs::{Array, Dtype, StreamOrDevice, ops};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{
    AffineWeight, MLX_TEXT_PREFIX, Numerics, Qwen35Error, Qwen35Precision, Qwen35Weights,
    check_dequantized_bytes, expected_shapes, mlp,
};
use crate::Qwen35Config;

/// A two-layer dense `qwen3_5` small enough to build in memory.
pub(super) fn tiny_config(quantization: &Value) -> Value {
    json!({
        "model_type": "qwen3_5",
        "tie_word_embeddings": true,
        "quantization": quantization,
        "text_config": {
            "model_type": "qwen3_5_text",
            "attn_output_gate": true,
            "head_dim": 32,
            "hidden_act": "silu",
            "hidden_size": 64,
            "intermediate_size": 128,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 32,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 32,
            "max_position_embeddings": 4096,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25},
            "vocab_size": 128
        }
    })
}

/// Every tensor of the tiny checkpoint, named as mlx-lm stores them: matrices
/// quantized (4-bit, group 32; the `in_proj_b` override 8-bit), the rest BF16.
pub(super) fn tiny_tensors(config: &Value) -> HashMap<String, Array> {
    let parsed = Qwen35Config::parse(&config.to_string()).expect("tiny config");
    let quantization = parsed.quantization().expect("quantized");
    let gpu = StreamOrDevice::gpu();
    let mut tensors = HashMap::new();
    let mut shapes: Vec<_> = expected_shapes(&parsed, Numerics::MlxLm)
        .expect("shapes")
        .into_iter()
        .collect();
    shapes.sort();
    for (seed, (name, shape)) in (1_u64..).zip(shapes) {
        let count = shape
            .iter()
            .map(|&d| usize::try_from(d).expect("dim"))
            .product();
        let values = Array::from_slice(&super::tests::values(seed, count), &shape)
            .as_dtype_device(Dtype::Bfloat16, &gpu)
            .expect("bf16");
        let stem = name.strip_suffix(".weight");
        let module = stem.and_then(|stem| quantization.module(stem));
        match (stem, module) {
            (Some(stem), Some(affine)) if shape.len() >= 2 && !name.contains("conv1d") => {
                let (codes, scales, biases) = ops::quantize_device(
                    &values,
                    i32::from(affine.group_size()),
                    i32::from(affine.bits()),
                    &gpu,
                )
                .expect("quantize");
                tensors.insert(format!("{MLX_TEXT_PREFIX}{stem}.weight"), codes);
                tensors.insert(format!("{MLX_TEXT_PREFIX}{stem}.scales"), scales);
                tensors.insert(format!("{MLX_TEXT_PREFIX}{stem}.biases"), biases);
            }
            _ => {
                tensors.insert(format!("{MLX_TEXT_PREFIX}{name}"), values);
            }
        }
    }
    tensors
}

fn quantization() -> Value {
    json!({
        "group_size": 32, "bits": 4, "mode": "affine",
        "language_model.model.layers.0.linear_attn.in_proj_b": {"group_size": 32, "bits": 8}
    })
}

/// Writes a checkpoint directory under the system temporary directory.
pub(super) fn write(name: &str, config: &Value, tensors: &HashMap<String, Array>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qwen35-mlx-affine-{name}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("temp dir");
    fs::write(dir.join("config.json"), config.to_string()).expect("config");
    let model = dir.join("model.safetensors");
    let _ = fs::remove_file(&model);
    Array::save_safetensors(tensors.iter(), None, &model).expect("save");
    dir
}

fn load(
    name: &str,
    config: &Value,
    tensors: &HashMap<String, Array>,
) -> Result<Qwen35Weights, Qwen35Error> {
    load_at(name, config, tensors, Qwen35Precision::Checkpoint)
}

fn load_at(
    name: &str,
    config: &Value,
    tensors: &HashMap<String, Array>,
    precision: Qwen35Precision,
) -> Result<Qwen35Weights, Qwen35Error> {
    let dir = write(name, config, tensors);
    let loaded = Qwen35Weights::load(&dir, precision);
    fs::remove_dir_all(&dir).expect("cleanup");
    loaded
}

#[test]
fn loads_an_mlx_affine_checkpoint_and_refuses_inconsistent_ones() {
    let config = tiny_config(&quantization());
    let tensors = tiny_tensors(&config);

    let weights = load("ok", &config, &tensors).expect("tiny checkpoint loads");
    assert_eq!(weights.numerics, Numerics::MlxLm);
    assert_eq!(weights.compute, Dtype::Bfloat16);
    assert_eq!(
        weights.quantized["layers.0.linear_attn.in_proj_b.weight"].bits,
        8
    );
    assert_eq!(
        weights.quantized["layers.1.self_attn.q_proj.weight"].bits,
        4
    );
    assert!(weights.quantized.contains_key("embed_tokens.weight"));
    // Norms are stored already shifted, so they load unchanged.
    let norm = "layers.0.input_layernorm.weight";
    let stored = &tensors[&format!("{MLX_TEXT_PREFIX}{norm}")];
    assert!(
        weights.tensors[norm]
            .array_eq(stored, None)
            .expect("compare")
            .item::<bool>()
    );
    let logits = weights
        .executor()
        .prefill_last_logits(&[1, 2, 3])
        .expect("prefill");
    assert_eq!(logits.len(), 128);
    assert!(logits.iter().all(|logit| logit.is_finite()));

    // The override's 8-bit codes read at the default 4 bits.
    let no_override = tiny_config(&json!({"group_size": 32, "bits": 4}));
    assert!(matches!(
        load("override", &no_override, &tensors),
        Err(Qwen35Error::TensorShape { name, .. }) if name == "layers.0.linear_attn.in_proj_b.weight"
    ));

    let mut missing = tensors.clone();
    missing.remove(&format!("{MLX_TEXT_PREFIX}layers.1.mlp.up_proj.biases"));
    assert!(matches!(
        load("missing", &config, &missing),
        Err(Qwen35Error::MissingTensor(name)) if name == "layers.1.mlp.up_proj.biases"
    ));

    let mut unquantized = config.clone();
    unquantized
        .as_object_mut()
        .expect("object")
        .remove("quantization");
    assert!(matches!(
        load("undeclared", &unquantized, &tensors),
        Err(Qwen35Error::UndeclaredQuantization(_))
    ));

    let mut mixed = tensors.clone();
    let norm_value = mixed[&format!("{MLX_TEXT_PREFIX}norm.weight")].clone();
    mixed.insert("model.language_model.norm.weight".to_owned(), norm_value);
    assert!(matches!(
        load("mixed", &config, &mixed),
        Err(Qwen35Error::MixedLayouts)
    ));

    let mut integer_scales = tensors.clone();
    let scales = format!("{MLX_TEXT_PREFIX}layers.1.mlp.up_proj.scales");
    let as_int = integer_scales[&scales]
        .as_dtype_device(Dtype::Int32, StreamOrDevice::gpu())
        .expect("int32");
    integer_scales.insert(scales, as_int);
    assert!(matches!(
        load("int-scales", &config, &integer_scales),
        Err(Qwen35Error::QuantizationParameterDtype { name, .. }) if name == "layers.1.mlp.up_proj.scales"
    ));

    // f32 dequantizes every module at load.
    let f32 = load_at("f32", &config, &tensors, Qwen35Precision::Float32).expect("f32 load");
    assert!(f32.quantized.is_empty());
    assert_eq!(
        f32.tensors["layers.1.self_attn.q_proj.weight"].dtype(),
        Dtype::Float32
    );
    let logits = f32
        .executor()
        .prefill_last_logits(&[1, 2, 3])
        .expect("f32 prefill");
    assert!(logits.iter().all(|logit| logit.is_finite()));
}

#[test]
fn refuses_an_f32_load_above_the_dequantized_limit() {
    let weight = |groups: i32| AffineWeight {
        codes: Array::from_slice(&[0_u32], &[1]),
        // Never evaluated, so never allocated.
        scales: ops::zeros_dtype_device(&[groups, 1 << 16], Dtype::Bfloat16, StreamOrDevice::gpu())
            .expect("zeros"),
        biases: Array::from_slice(&[0_u32], &[1]),
        bits: 4,
        group_size: 64,
    };
    // 2^10 x 2^16 groups x 64 values x 4 bytes = 16 GiB, the limit itself.
    let at_limit = HashMap::from([("a".to_owned(), weight(1 << 10))]);
    assert!(check_dequantized_bytes(&at_limit).is_ok());
    let above = HashMap::from([("a".to_owned(), weight((1 << 10) + 1))]);
    assert!(matches!(
        check_dequantized_bytes(&above),
        Err(Qwen35Error::DequantizedTooLarge { .. })
    ));
}

/// Element `i` is `(((i * 2654435761) mod 2^32) >> 24) - 128) / 64`, exact
/// in BF16; the reference script builds the same values.
fn synthetic(shape: &[i32]) -> Array {
    let count: u64 = shape
        .iter()
        .map(|&d| u64::try_from(d).expect("dim"))
        .product();
    let values: Vec<f32> = (0..count)
        .map(|i| {
            let code = (i.wrapping_mul(2_654_435_761) % (1 << 32)) >> 24;
            #[allow(clippy::cast_precision_loss, reason = "code < 256")]
            let code = code as f32;
            (code - 128.0) / 64.0
        })
        .collect();
    Array::from_slice(&values, shape)
        .as_dtype_device(Dtype::Bfloat16, StreamOrDevice::gpu())
        .expect("bf16")
}

/// SHA-256 of an array's raw BF16 bytes, little-endian.
fn digest(array: &Array) -> String {
    let gpu = StreamOrDevice::gpu();
    let bits = array
        .as_dtype_device(Dtype::Bfloat16, &gpu)
        .and_then(|bf16| bf16.view_dtype_device(Dtype::Uint16, &gpu))
        .expect("bf16 bits");
    bits.eval().expect("eval");
    let mut hash = Sha256::new();
    for value in bits.as_slice::<u16>() {
        hash.update(value.to_le_bytes());
    }
    hash.finalize().iter().fold(String::new(), |mut hex, byte| {
        write!(hex, "{byte:02x}").expect("write to String");
        hex
    })
}

fn dim(text: &str) -> i32 {
    text.parse().expect("row count")
}

/// One component's output for the reference key `path:detail`.
fn component(weights: &Qwen35Weights, key: &str) -> Array {
    let (path, detail) = key.split_once(':').expect("component key");
    let stem = match path.strip_prefix(MLX_TEXT_PREFIX) {
        Some(stem) => stem.to_owned(),
        None if path == "language_model.lm_head" => "lm_head".to_owned(),
        None => panic!("unexpected component {path}"),
    };
    let hidden = i32::try_from(weights.config.hidden_size).expect("hidden");
    if detail == "embed" {
        return weights.embed(&[0, 1, 2, 151_643, 248_319]).expect("embed");
    }
    if let Some(rows) = detail.strip_prefix("as_linear:") {
        return weights
            .project(
                &synthetic(&[1, dim(rows), hidden]),
                &format!("{stem}.weight"),
            )
            .expect("tied head");
    }
    if let Some(base) = stem.strip_suffix(".mlp") {
        return mlp(weights, base, &synthetic(&[1, dim(detail), hidden])).expect("moe block");
    }
    let weight = &weights.quantized[&format!("{stem}.weight")];
    let columns = weight.scales.shape()[1] * weight.group_size;
    weights
        .project(
            &synthetic(&[1, dim(detail), columns]),
            &format!("{stem}.weight"),
        )
        .expect("projection")
}

/// Cooperative materialization-boundary bound for the four-layer qualification.
/// MLX active plus cache and historical peak exclude other process/driver memory.
fn check_reference_memory(phase: &str, report: bool) {
    let active = mlx_rs::memory::active_memory().expect("active memory");
    let cached = mlx_rs::memory::cache_memory().expect("cached memory");
    let peak = mlx_rs::memory::peak_memory().expect("peak memory");
    let cap = 8_usize << 30;
    if report {
        eprintln!("MLX {phase}: active={active} cached={cached} peak={peak} cap={cap}");
    }
    assert!(
        peak <= cap && active.saturating_add(cached) <= cap,
        "MLX resource cap exceeded after {phase}: active={active} cached={cached} peak={peak} cap={cap}"
    );
}

/// Every quantized module, the embedding, the output head and every `MoE`
/// block reproduce mlx-lm's BF16 output bit for bit on the reference
/// script's synthetic inputs.
///
/// Set `METALLIX_QWEN35_AFFINE_MODEL` to the checkpoint (for the 35B
/// mixture-of-experts model, its 4-layer truncation from the script's
/// `truncate`) and `METALLIX_QWEN35_AFFINE_REFERENCE` to the script's
/// `reference` JSON for it.

#[test]
fn components_match_mlx_lm_bit_for_bit() {
    let (Some(model), Some(reference)) = (
        std::env::var_os("METALLIX_QWEN35_AFFINE_MODEL"),
        std::env::var_os("METALLIX_QWEN35_AFFINE_REFERENCE"),
    ) else {
        eprintln!("skipping: METALLIX_QWEN35_AFFINE_MODEL or _REFERENCE is not set");
        return;
    };
    let reference: Value =
        serde_json::from_str(&fs::read_to_string(reference).expect("reference")).expect("JSON");
    mlx_rs::memory::set_cache_limit(2 << 30).expect("allocator cache limit");
    mlx_rs::memory::reset_peak_memory().expect("reset peak");
    let weights =
        Qwen35Weights::load(PathBuf::from(model), Qwen35Precision::Checkpoint).expect("load");
    check_reference_memory("load", true);
    let components = reference["components"].as_object().expect("components");
    assert!(!components.is_empty());
    let mut mismatched = Vec::new();
    for (key, expected) in components {
        let actual = digest(&component(&weights, key));
        check_reference_memory("component", false);
        if actual != expected.as_str().expect("digest") {
            mismatched.push(key.clone());
        }
    }
    eprintln!(
        "components: {} checked, {} mismatched",
        components.len(),
        mismatched.len()
    );
    check_reference_memory("components complete", true);
    assert!(mismatched.is_empty(), "mismatched: {mismatched:?}");
}

/// Compaction must preserve every selected affine dot product, including when
/// the number of stored experts straddles MLX's sorted-gather dispatch boundary.
#[test]
fn logical_affine_gather_preserves_selected_dot_products() {
    let gpu = StreamOrDevice::gpu();
    let weights = Array::from_slice(&super::tests::values(91, 32 * 64 * 64), &[32, 64, 64])
        .as_dtype_device(Dtype::Bfloat16, &gpu)
        .expect("bf16 weights");
    let (codes, scales, biases) = ops::quantize_device(&weights, 32, 4, &gpu).expect("quantize");
    let full = AffineWeight {
        codes,
        scales,
        biases,
        bits: 4,
        group_size: 32,
    };
    let selected = Array::from_slice(&(0..16_i32).collect::<Vec<_>>(), &[16]);
    let compact = AffineWeight {
        codes: full
            .codes
            .take_axis_device(&selected, 0, &gpu)
            .expect("selected codes"),
        scales: full
            .scales
            .take_axis_device(&selected, 0, &gpu)
            .expect("selected scales"),
        biases: full
            .biases
            .take_axis_device(&selected, 0, &gpu)
            .expect("selected biases"),
        bits: 4,
        group_size: 32,
    };
    let input = Array::from_slice(&super::tests::values(73, 64 * 64), &[64, 1, 64])
        .as_dtype_device(Dtype::Bfloat16, &gpu)
        .expect("bf16 input");
    let indices = Array::from_slice(&(0..64_i32).map(|i| i / 4).collect::<Vec<_>>(), &[64]);
    let run = |weight, sorted| {
        let out = super::Matrix::Affine(weight)
            .gather(&input, &indices, sorted)
            .expect("gather");
        let bits = out.view_dtype_device(Dtype::Uint16, &gpu).expect("bits");
        bits.eval().expect("eval");
        bits.as_slice::<u16>().to_vec()
    };
    let unsorted_full = run(&full, false);
    let unsorted_compact = run(&compact, false);
    assert_eq!(
        unsorted_full, unsorted_compact,
        "common unsorted kernel control"
    );
    let sorted_full = run(&full, true);
    let sorted_compact = run(&compact, true);
    let different = sorted_full
        .iter()
        .zip(&sorted_compact)
        .filter(|(a, b)| a != b)
        .count();
    let max_abs = sorted_full
        .iter()
        .zip(&sorted_compact)
        .map(|(&a, &b)| {
            (f32::from_bits(u32::from(a) << 16) - f32::from_bits(u32::from(b) << 16)).abs()
        })
        .fold(0.0_f32, f32::max);
    eprintln!(
        "sorted compaction mismatches: {different}/{}; max_abs={max_abs}; full sorted vs unsorted: {}",
        sorted_full.len(),
        sorted_full
            .iter()
            .zip(&unsorted_full)
            .filter(|(a, b)| a != b)
            .count()
    );
    // Reconstruct through the same helper as the actual pager. Keep the raw
    // compact comparison diagnostic: upstream may change its dispatch later.
    let ids: Vec<u32> = (0..16).collect();
    let restore = |array: &Array| {
        let parts: Vec<Array> = ids
            .iter()
            .map(|&id| {
                array
                    .take_axis_device(&Array::from_slice(&[id], &[1]), 0, &gpu)
                    .expect("expert")
            })
            .collect();
        super::paged::logical_expert_stack(&ids, &parts.iter().collect::<Vec<_>>(), 32)
            .expect("logical stack")
    };
    let restored = AffineWeight {
        codes: restore(&full.codes),
        scales: restore(&full.scales),
        biases: restore(&full.biases),
        bits: full.bits,
        group_size: full.group_size,
    };
    assert_eq!(
        run(&restored, true),
        sorted_full,
        "logical stack preserves sorted arithmetic"
    );
    assert_eq!(
        run(&restored, false),
        unsorted_full,
        "logical stack preserves unsorted arithmetic"
    );
}
