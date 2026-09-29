//! Same-trace layer-zero block output consumed by the native layer-one Engram.

use std::{collections::BTreeMap, fmt::Write as _, num::NonZeroUsize};

use deepseek::{
    RotaryFrequency, StartupLayout,
    attention::layer::{
        Fp8Projection, LayerAttentionDiagnostic, LayerAttentionError, LayerAttentionLayout,
        LayerAttentionState, LayerAttentionWeights,
    },
    ffn::FfnSublayerReference,
    hc::{HcCoefficients, projection::project_hc_coefficients},
    moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEReference},
    reduced::{AttentionInput, BlockTailReference, StartupSession, StartupStepOutput},
    startup_bf16_reference,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[path = "support/hc_coefficient_bounds.rs"]
pub(crate) mod hc_coefficient_bounds;
#[path = "support/hc_projection_bounds.rs"]
pub(crate) mod hc_projection_bounds;
#[path = "support/layer1_engram_capture.rs"]
#[allow(
    dead_code,
    reason = "the bridge test needs the supplied-stream entry helper from the shared Engram fixture oracle"
)]
pub(crate) mod layer1_engram_capture;
#[path = "support/runtime_engram.rs"]
pub(crate) mod runtime_engram;

const FIXTURE_SHA256: &str = "2a0e294e62565be699c710fdfbf4f52bb63a0e9f3bea8e7eceae8ea11164b862";

fn field<'a>(value: &'a Value, key: &str) -> &'a Value {
    value.get(key).unwrap_or_else(|| panic!("missing {key}"))
}

fn bytes(value: &Value) -> Vec<u8> {
    let storage = field(value, "storage_hex").as_str().expect("storage hex");
    let bytes = storage
        .as_bytes()
        .chunks_exact(2)
        .map(|word| u8::from_str_radix(std::str::from_utf8(word).expect("UTF-8"), 16).expect("hex"))
        .collect::<Vec<_>>();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        field(value, "storage_sha256")
            .as_str()
            .expect("storage hash")
    );
    bytes
}

fn bf16(value: &Value) -> Vec<u16> {
    assert_eq!(field(value, "dtype").as_str(), Some("torch.bfloat16"));
    let shape = field(value, "shape").as_array().expect("tensor shape");
    let count = shape
        .iter()
        .map(|width| usize::try_from(width.as_u64().expect("dimension")).expect("usize"))
        .product::<usize>();
    assert_eq!(field(value, "numel").as_u64(), Some(count as u64));
    let storage = field(value, "storage_hex").as_str().expect("storage hex");
    assert_eq!(storage.len(), count * 4, "BF16 source storage length");
    let bytes = storage
        .as_bytes()
        .chunks_exact(2)
        .map(|word| u8::from_str_radix(std::str::from_utf8(word).expect("UTF-8"), 16).expect("hex"))
        .collect::<Vec<_>>();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        field(value, "storage_sha256")
            .as_str()
            .expect("storage hash")
    );
    bytes
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("BF16")))
        .collect()
}

fn fp32(value: &Value) -> Vec<f32> {
    assert_eq!(field(value, "dtype").as_str(), Some("torch.float32"));
    let shape = field(value, "shape").as_array().expect("tensor shape");
    let count = shape
        .iter()
        .map(|width| usize::try_from(width.as_u64().expect("dimension")).expect("usize"))
        .product::<usize>();
    assert_eq!(field(value, "numel").as_u64(), Some(count as u64));
    let storage = field(value, "storage_hex").as_str().expect("storage hex");
    assert_eq!(storage.len(), count * 8, "FP32 source storage length");
    let bytes = storage
        .as_bytes()
        .chunks_exact(2)
        .map(|word| u8::from_str_radix(std::str::from_utf8(word).expect("UTF-8"), 16).expect("hex"))
        .collect::<Vec<_>>();
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        field(value, "storage_sha256")
            .as_str()
            .expect("storage hash")
    );
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("FP32")))
        .collect()
}

fn fp8(value: &Value) -> Vec<u8> {
    let dtype = field(value, "dtype").as_str().expect("FP8 dtype");
    assert!(
        matches!(dtype, "torch.float8_e4m3fn" | "torch.float8_e8m0fnu"),
        "unexpected FP8 dtype {dtype}"
    );
    bytes(value)
}

fn i32s(value: &Value) -> Vec<i32> {
    assert_eq!(field(value, "dtype").as_str(), Some("torch.int32"));
    bytes(value)
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().expect("i32")))
        .collect()
}

fn u64s(value: &Value) -> Vec<u64> {
    assert_eq!(field(value, "dtype").as_str(), Some("torch.int64"));
    bytes(value)
        .chunks_exact(8)
        .map(|word| u64::from_le_bytes(word.try_into().expect("u64")))
        .collect()
}

fn frequencies(value: &Value) -> Vec<RotaryFrequency> {
    assert_eq!(field(value, "dtype").as_str(), Some("torch.complex64"));
    bytes(value)
        .chunks_exact(8)
        .map(|word| {
            RotaryFrequency::new(
                f32::from_le_bytes(word[..4].try_into().expect("frequency real")),
                f32::from_le_bytes(word[4..].try_into().expect("frequency imaginary")),
            )
            .expect("finite source rotary frequency")
        })
        .collect()
}

#[derive(Debug)]
struct AttentionWeights {
    wq_a_codes: Vec<u8>,
    wq_a_scales: Vec<u8>,
    q_norm: Vec<u16>,
    wq_b_codes: Vec<u8>,
    wq_b_scales: Vec<u8>,
    wkv_codes: Vec<u8>,
    wkv_scales: Vec<u8>,
    kv_norm: Vec<u16>,
    attn_sink: Vec<f32>,
    wo_a: Vec<u16>,
    wo_b_codes: Vec<u8>,
    wo_b_scales: Vec<u8>,
}

impl AttentionWeights {
    fn borrowed(&self) -> LayerAttentionWeights<'_> {
        LayerAttentionWeights {
            wq_a: Fp8Projection {
                codes: &self.wq_a_codes,
                scales: &self.wq_a_scales,
            },
            q_norm: &self.q_norm,
            wq_b: Fp8Projection {
                codes: &self.wq_b_codes,
                scales: &self.wq_b_scales,
            },
            wkv: Fp8Projection {
                codes: &self.wkv_codes,
                scales: &self.wkv_scales,
            },
            kv_norm: &self.kv_norm,
            attn_sink: &self.attn_sink,
            wo_a: &self.wo_a,
            wo_b: Fp8Projection {
                codes: &self.wo_b_codes,
                scales: &self.wo_b_scales,
            },
        }
    }
}

fn attention_weights(root: &Value) -> AttentionWeights {
    let parameters = field(root, "parameters");
    let tensor = |name| field(parameters, name);
    AttentionWeights {
        wq_a_codes: fp8(tensor("layers.0.attn.wq_a.weight")),
        wq_a_scales: fp8(tensor("layers.0.attn.wq_a.scale")),
        q_norm: bf16(tensor("layers.0.attn.q_norm.weight")),
        wq_b_codes: fp8(tensor("layers.0.attn.wq_b.weight")),
        wq_b_scales: fp8(tensor("layers.0.attn.wq_b.scale")),
        wkv_codes: fp8(tensor("layers.0.attn.wkv.weight")),
        wkv_scales: fp8(tensor("layers.0.attn.wkv.scale")),
        kv_norm: bf16(tensor("layers.0.attn.kv_norm.weight")),
        attn_sink: fp32(tensor("layers.0.attn.attn_sink")),
        wo_a: bf16(tensor("layers.0.attn.wo_a.weight")),
        wo_b_codes: fp8(tensor("layers.0.attn.wo_b.weight")),
        wo_b_scales: fp8(tensor("layers.0.attn.wo_b.scale")),
    }
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("nonzero source dimension")
}

fn usize_field(value: &Value, key: &str) -> usize {
    usize::try_from(field(value, key).as_u64().expect("source dimension")).expect("usize")
}

fn window_only_layout(root: &Value) -> LayerAttentionLayout {
    let model = field(root, "model");
    LayerAttentionLayout::new_window_only(
        nonzero(1),
        nonzero(usize_field(model, "dim")),
        nonzero(usize_field(model, "n_heads")),
        nonzero(usize_field(model, "head_dim")),
        nonzero(usize_field(model, "rope_head_dim") / 2),
        nonzero(usize_field(model, "q_lora_rank")),
        nonzero(usize_field(model, "window_size")),
        nonzero(usize_field(model, "o_groups")),
        nonzero(usize_field(model, "o_lora_rank")),
        1.0e-20,
        0.125,
    )
    .expect("layer-zero window-only layout")
}

fn assert_attention_diagnostic(case: &Value, diagnostic: &LayerAttentionDiagnostic) {
    let attention = field(case, "attention");
    let stages = field(attention, "stages");
    let window = field(attention, "window");
    let sparse = field(attention, "sparse");
    assert_eq!(diagnostic.wq_a, bf16(field(stages, "wq_a")), "WQ-A");
    assert_eq!(diagnostic.qr, bf16(field(stages, "q_norm")), "Q norm");
    assert_eq!(
        diagnostic.wq_b_pre_rope,
        bf16(field(stages, "wq_b")),
        "WQ-B"
    );
    assert_eq!(
        diagnostic.q_after_rope,
        bf16(field(sparse, "q_after_rope")),
        "Q RoPE"
    );
    assert_eq!(
        diagnostic.prepared_window,
        bf16(field(window, "prepared")),
        "prepared window"
    );
    assert_eq!(
        diagnostic.window_read,
        bf16(field(window, "read")),
        "window read"
    );
    assert_eq!(
        diagnostic.window_indices,
        i32s(field(window, "indices")),
        "window indices"
    );
    assert_eq!(
        diagnostic.ring_after,
        bf16(field(window, "ring_after")),
        "window ring"
    );
    assert_eq!(
        diagnostic.sparse_output,
        bf16(field(sparse, "output_pre_inverse_rope")),
        "sparse output"
    );
    assert_eq!(
        diagnostic.final_output,
        bf16(field(case, "attention_output")),
        "attention output"
    );
}

fn native_startup(root: &Value, case: &Value) -> (Vec<u16>, Vec<f32>) {
    let startup = field(case, "startup");
    let ids = u64s(field(startup, "input_ids"));
    let table = bf16(field(field(root, "parameters"), "embed.weight"));
    let output = startup_bf16_reference(
        &ids,
        &table,
        StartupLayout::new(8, 128, 2).expect("startup layout"),
    )
    .expect("source token embedding startup");
    let embedding = ids
        .iter()
        .flat_map(|&id| {
            let row = usize::try_from(id).expect("validated source token");
            table[row * 128..(row + 1) * 128].iter().copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        embedding,
        bf16(field(startup, "embedding")),
        "native embedding"
    );
    let block_input = field(case, "block_input");
    assert_eq!(
        output.residual_bf16(),
        bf16(field(block_input, "residual")),
        "native two-copy startup residual"
    );
    assert_eq!(
        output.identity_pre(),
        fp32(field(block_input, "incoming_pre")),
        "native identity startup pre-mix"
    );
    (
        output.residual_bf16().to_vec(),
        output.identity_pre().to_vec(),
    )
}

fn native_attention_input(root: &Value, case: &Value) -> Vec<u16> {
    let positions = usize::try_from(
        field(field(case, "attention_input"), "shape")
            .as_array()
            .expect("attention input shape")[1]
            .as_u64()
            .expect("positions"),
    )
    .expect("usize positions");
    let (residual, incoming_pre) = native_startup(root, case);
    let norm_weight = bf16(field(
        field(root, "parameters"),
        "layers.0.attn_norm.weight",
    ));
    let preparation = AttentionInput::new(&norm_weight, 2, 1.0e-20).unwrap();
    let mut attention_input = Vec::with_capacity(positions * 128);
    for position in 0..positions {
        let output = preparation
            .forward(
                &residual[position * 256..(position + 1) * 256],
                &incoming_pre[position * 2..(position + 1) * 2],
            )
            .expect("runtime attention preparation");
        attention_input.extend_from_slice(output.normalized_bf16());
    }
    assert_eq!(
        attention_input,
        bf16(field(case, "attention_input")),
        "native layer-zero HC/RMSNorm attention input"
    );
    attention_input
}

fn forward_window_case(
    state: &mut LayerAttentionState,
    case: &Value,
    attention_input: &[u16],
    all_frequencies: &[RotaryFrequency],
    weights: LayerAttentionWeights<'_>,
) -> Result<LayerAttentionDiagnostic, LayerAttentionError> {
    let start = usize_field(case, "start_pos");
    let positions = attention_input.len() / 128;
    let from = start.checked_mul(16).expect("frequency offset");
    let to = from + positions * 16;
    state.forward_window_only(attention_input, start, &all_frequencies[from..to], weights)
}

fn native_hc_coefficients(root: &Value, sublayer: &str, residual: &[u16]) -> HcCoefficients {
    let model = field(root, "model");
    let parameters = field(root, "parameters");
    let prefix = format!("layers.0.hc_{sublayer}");
    let projection = fp32(field(parameters, &format!("{prefix}_fn")));
    let scale: [f32; 3] = fp32(field(parameters, &format!("{prefix}_scale")))
        .try_into()
        .expect("three HC scales");
    let base = fp32(field(parameters, &format!("{prefix}_base")));
    project_hc_coefficients(
        residual,
        &projection,
        &scale,
        &base,
        2,
        serde_json::from_value(field(model, "norm_eps").clone()).expect("HC norm epsilon"),
        usize_field(model, "hc_sinkhorn_iters"),
        serde_json::from_value(field(model, "hc_eps").clone()).expect("HC epsilon"),
    )
    .expect("native layer-zero HC projection")
}

fn assert_native_next_pre_envelope(
    root: &Value,
    case: &Value,
    position: usize,
    residual: &[u16],
    native: &HcCoefficients,
) {
    let model = field(root, "model");
    let parameters = field(root, "parameters");
    let projection = fp32(field(parameters, "layers.0.hc_ffn_fn"));
    let scale: [f32; 3] = fp32(field(parameters, "layers.0.hc_ffn_scale"))
        .try_into()
        .expect("three HC scales");
    let base = fp32(field(parameters, "layers.0.hc_ffn_base"));
    let norm_eps =
        serde_json::from_value(field(model, "norm_eps").clone()).expect("HC norm epsilon");
    let projection_bounds =
        hc_projection_bounds::normalized_projection_envelopes(residual, &projection, norm_eps)
            .expect("source-grounded HC projection bounds");
    let mix_bounds = projection_bounds
        .iter()
        .map(|bound| [bound.lo, bound.hi])
        .collect::<Vec<_>>();
    let coefficient_bounds = hc_coefficient_bounds::coefficient_envelopes(
        &mix_bounds,
        &scale,
        &base,
        usize_field(model, "hc_sinkhorn_iters"),
        serde_json::from_value(field(model, "hc_eps").clone()).expect("HC epsilon"),
    )
    .expect("source-grounded HC coefficient bounds");
    let source = fp32(field(case, "block_next_pre"));
    for ((bound, &source), &actual) in coefficient_bounds
        .pre
        .iter()
        .zip(&source[position * 2..(position + 1) * 2])
        .zip(native.pre())
    {
        assert!(
            source.is_finite() && bound[0] <= f64::from(source) && f64::from(source) <= bound[1],
            "source next HC pre within analytic envelope"
        );
        assert!(
            actual.is_finite() && bound[0] <= f64::from(actual) && f64::from(actual) <= bound[1],
            "native next HC pre within analytic envelope"
        );
    }
}

fn with_native_layer_zero_moe<R>(root: &Value, body: impl FnOnce(MoEReference<'_>) -> R) -> R {
    let parameters = field(root, "parameters");
    let encoded = parameters
        .as_object()
        .expect("layer-zero parameter object")
        .iter()
        .map(|(name, value)| (name.clone(), bytes(value)))
        .collect::<BTreeMap<_, _>>();
    let expert_bytes = |prefix: &str| -> [&[u8]; 6] {
        [
            "w1.weight",
            "w1.scale",
            "w2.weight",
            "w2.scale",
            "w3.weight",
            "w3.scale",
        ]
        .map(|suffix| encoded[&format!("layers.0.ffn.{prefix}.{suffix}")].as_slice())
    };
    let routed = (0..4)
        .map(|index| {
            let [w1, s1, w2, s2, w3, s3] = expert_bytes(&format!("experts.{index}"));
            Fp4ExpertWeights::new(128, 128, w1, s1, w2, s2, w3, s3)
                .expect("layer-zero routed expert")
        })
        .collect::<Vec<_>>();
    let [w1, s1, w2, s2, w3, s3] = expert_bytes("shared_experts");
    let shared =
        Fp8ExpertWeights::new(128, 128, w1, s1, w2, s2, w3, s3).expect("layer-zero shared expert");
    let model = field(root, "model");
    let config = MoEConfig::new(
        usize::try_from(field(model, "dim").as_u64().expect("dimension")).expect("usize"),
        usize::try_from(
            field(model, "moe_inter_dim")
                .as_u64()
                .expect("intermediate"),
        )
        .expect("usize"),
        serde_json::from_value(field(model, "swiglu_limit").clone()).expect("SwiGLU limit"),
        usize::try_from(
            field(model, "n_activated_experts")
                .as_u64()
                .expect("active experts"),
        )
        .expect("usize"),
        serde_json::from_value(field(model, "gate_temp").clone()).expect("gate temperature"),
        field(model, "norm_topk_prob")
            .as_bool()
            .expect("normalized top-k"),
        serde_json::from_value(field(model, "route_scale").clone()).expect("route scale"),
    )
    .expect("layer-zero MoE config");
    let gate = bf16(field(parameters, "layers.0.ffn.gate.weight"));
    let bias = fp32(field(parameters, "layers.0.ffn.gate.bias"));
    body(MoEReference::new(config, &gate, &bias, &routed, shared).expect("layer-zero MoE"))
}

fn with_runtime_startup<R>(root: &Value, body: impl FnOnce(StartupSession<'_>) -> R) -> R {
    let parameters = field(root, "parameters");
    let model = field(root, "model");
    let table = bf16(field(parameters, "embed.weight"));
    let norm = bf16(field(parameters, "layers.0.attn_norm.weight"));
    let ffn_norm = bf16(field(parameters, "layers.0.ffn_norm.weight"));
    let attn_projection = fp32(field(parameters, "layers.0.hc_attn_fn"));
    let attn_scale: [f32; 3] = fp32(field(parameters, "layers.0.hc_attn_scale"))
        .try_into()
        .unwrap();
    let attn_base = fp32(field(parameters, "layers.0.hc_attn_base"));
    let ffn_projection = fp32(field(parameters, "layers.0.hc_ffn_fn"));
    let ffn_scale: [f32; 3] = fp32(field(parameters, "layers.0.hc_ffn_scale"))
        .try_into()
        .unwrap();
    let ffn_base = fp32(field(parameters, "layers.0.hc_ffn_base"));
    let epsilon = serde_json::from_value(field(model, "norm_eps").clone()).unwrap();
    let iterations = usize_field(model, "hc_sinkhorn_iters");
    let hc_epsilon = serde_json::from_value(field(model, "hc_eps").clone()).unwrap();
    let attention = attention_weights(root);
    with_native_layer_zero_moe(root, |moe| {
        let ffn = FfnSublayerReference::new(
            moe,
            &ffn_norm,
            &ffn_projection,
            &ffn_scale,
            &ffn_base,
            2,
            epsilon,
            iterations,
            hc_epsilon,
        )
        .unwrap();
        let tail = BlockTailReference::new(
            ffn,
            &attn_projection,
            &attn_scale,
            &attn_base,
            2,
            epsilon,
            iterations,
            hc_epsilon,
        )
        .unwrap();
        body(
            StartupSession::new(
                &table,
                &norm,
                epsilon,
                window_only_layout(root),
                attention.borrowed(),
                tail,
            )
            .unwrap(),
        )
    })
}

fn assert_runtime_startup(root: &Value, case: &Value, output: &StartupStepOutput) {
    let (initial, pre) = native_startup(root, case);
    assert_eq!(output.startup().residual_bf16(), initial);
    assert_eq!(output.startup().identity_pre(), pre);
    assert_eq!(
        output.attention_input(),
        bf16(field(case, "attention_input"))
    );
    assert_attention_diagnostic(case, output.attention());
    assert_eq!(
        output.attention().final_output,
        bf16(field(case, "attention_output"))
    );
    let after = bf16(field(case, "after_attention_residual"));
    let collapsed = bf16(field(case, "ffn_collapsed"));
    let normalized = bf16(field(case, "ffn_input"));
    let moe = bf16(field(case, "ffn_output"));
    let terminal = bf16(field(case, "block_output"));
    for (position, tail) in output.tails().iter().enumerate() {
        assert_eq!(
            tail.after_attention_bf16(),
            &after[position * 256..(position + 1) * 256]
        );
        assert_eq!(
            tail.ffn().collapsed_bf16(),
            &collapsed[position * 128..(position + 1) * 128]
        );
        assert_eq!(
            tail.ffn().normalized_bf16(),
            &normalized[position * 128..(position + 1) * 128]
        );
        assert_eq!(
            tail.ffn().moe().output_bf16(),
            &moe[position * 128..(position + 1) * 128]
        );
        assert_eq!(
            tail.ffn().output_bf16(),
            &terminal[position * 256..(position + 1) * 256]
        );
        assert_native_next_pre_envelope(
            root,
            case,
            position,
            tail.after_attention_bf16(),
            tail.ffn().coefficients(),
        );
    }
    assert_eq!(output.residual(), terminal);
    assert_eq!(
        output.next_pre().len(),
        fp32(field(case, "block_next_pre")).len()
    );
    assert!(output.next_pre().iter().all(|value| value.is_finite()));
}

/// Runs the source-pinned layer-zero bridge from the supplied projection.
///
/// This is deliberately test-private: it is a reduced-oracle seam, not a
/// decoder API.  Keeping the parsed projection as its input prevents a
/// composition test from silently falling back to the older stand-alone
/// capture.
pub(crate) fn native_layer_zero_entries_from_projection(
    root: &Value,
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    assert_eq!(field(root, "schema_version").as_u64(), Some(1));
    assert_eq!(
        field(field(root, "contract"), "layer_zero_producer").as_str(),
        Some("source-pinned native startup, attention, FFN, and HC composition")
    );
    let cases = field(root, "cases").as_array().expect("bridge cases");
    let all_frequencies = frequencies(field(field(&cases[0], "attention"), "frequencies"));
    assert_eq!(all_frequencies.len(), 8 * 16, "source frequency table");
    with_runtime_startup(root, |mut session| {
        cases
            .iter()
            .map(|case| {
                let start = usize_field(case, "start_pos");
                let ids = u64s(field(field(case, "startup"), "input_ids"));
                let output = session
                    .step(
                        start,
                        &ids,
                        &all_frequencies[start * 16..(start + ids.len()) * 16],
                    )
                    .expect("runtime first block");
                assert_eq!(
                    field(field(case, "block_output"), "storage_sha256"),
                    field(field(case, "layer_one_engram_stream"), "storage_sha256")
                );
                assert_runtime_startup(root, case, &output);
                (
                    start,
                    output.residual().to_vec(),
                    output.next_pre().to_vec(),
                )
            })
            .collect()
    })
}

pub(crate) fn alternate_startup_projection() -> Value {
    let raw = include_str!("../../../../fixtures/deepseek-v41/partition-startup-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "440503a08fb157cc9c215ce2f3fa8a8ccb31e8cdf608e0558f07e1f57f6206e5"
    );
    let root: Value = serde_json::from_str(raw).expect("alternate startup projection");
    assert_eq!(
        field(&root, "source_receipt_sha256").as_str(),
        Some("9613150fea8010a7435dab0443a1f9e0d73fd8d0f32455b8d67dd572617f3906")
    );
    assert_eq!(
        field(field(&root, "capture_identity"), "schedule"),
        &serde_json::json!([4, 1, 1, 1])
    );
    root
}

#[test]
fn alternate_partition_native_startup_reaches_layer_one_stream() {
    let root = alternate_startup_projection();
    let entries = native_layer_zero_entries_from_projection(&root);
    assert_eq!(
        entries
            .iter()
            .map(|(start, _, _)| *start)
            .collect::<Vec<_>>(),
        [0, 4, 5, 6]
    );
    for ((start, residual, pre), case) in entries
        .iter()
        .zip(field(&root, "cases").as_array().unwrap())
    {
        assert_eq!(*start, usize_field(case, "start_pos"));
        assert_eq!(residual, &bf16(field(case, "layer_one_engram_stream")));
        assert_eq!(pre.len(), residual.len() / 128);
    }
}

pub(crate) fn alternate_layer_one_inputs() -> Vec<(usize, Vec<u16>)> {
    let root = alternate_startup_projection();
    let upstream = native_layer_zero_entries_from_projection(&root);
    let mut engram =
        layer1_engram_capture::NativeLayerOneEngramSession::from_alternate_startup(&root);
    let norm = bf16(field(
        field(&root, "parameters"),
        "layers.1.attn_norm.weight",
    ));
    let mut inputs = Vec::new();
    for ((start, residual, pre), case) in upstream.iter().zip(root["cases"].as_array().unwrap()) {
        let (entry_start, entry) = engram.step(Some(&(*start, residual.clone())));
        assert_eq!(entry_start, *start);
        assert_eq!(entry, bf16(&case["downstream"]["layer_one_engram_output"]));
        let mut attention_input = Vec::new();
        for (residual, pre) in entry.chunks_exact(256).zip(pre.chunks_exact(2)) {
            let prepared = AttentionInput::new(&norm, 2, 1.0e-20)
                .unwrap()
                .forward(residual, pre)
                .unwrap();
            attention_input.extend_from_slice(prepared.normalized_bf16());
        }
        assert_eq!(
            attention_input,
            bf16(&case["downstream"]["layer_one_attention_input"])
        );
        inputs.push((*start, attention_input));
    }
    inputs
}

#[test]
fn alternate_partition_native_startup_and_engram_reach_layer_one_attention_input() {
    assert_eq!(alternate_layer_one_inputs().len(), 4);
}

#[test]
fn alternate_partition_engram_rejects_bad_stream_then_continues_same_history() {
    let root = alternate_startup_projection();
    let upstream = native_layer_zero_entries_from_projection(&root);
    let mut engram =
        layer1_engram_capture::NativeLayerOneEngramSession::from_alternate_startup(&root);
    let mut control =
        layer1_engram_capture::NativeLayerOneEngramSession::from_alternate_startup(&root);
    for (index, (start, residual, _)) in upstream.iter().enumerate() {
        if index == 2 {
            let mut invalid = residual.clone();
            invalid[0] ^= 1;
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engram.step(Some(&(*start, invalid)));
                }))
                .is_err()
            );
        }
        let input = (*start, residual.clone());
        assert_eq!(engram.step(Some(&input)), control.step(Some(&input)));
    }
}

#[test]
fn alternate_partition_startup_rejects_changed_embedding() {
    let mut root = alternate_startup_projection();
    let tensor = &mut root["parameters"]["embed.weight"];
    let mut storage = bytes(tensor);
    storage[..256].fill(0);
    let mut hex = String::with_capacity(storage.len() * 2);
    for byte in &storage {
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    tensor["storage_hex"] = Value::String(hex);
    tensor["storage_sha256"] = Value::String(format!("{:x}", Sha256::digest(&storage)));
    let case = &root["cases"][0];
    assert!(
        std::panic::catch_unwind(|| native_startup(&root, case)).is_err(),
        "changed numerical weights must fail startup even with a valid storage digest"
    );
}

#[test]
fn source_layer_zero_output_feeds_native_layer_one_engram_entries() {
    let raw = include_str!("../../../../fixtures/deepseek-v41/layer0-to-layer1-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let root: Value = serde_json::from_str(raw).expect("layer-zero bridge fixture JSON");
    let cases = field(&root, "cases").as_array().expect("bridge cases");
    let native_layer_zero = native_layer_zero_entries_from_projection(&root);
    let streams = native_layer_zero
        .iter()
        .map(|(start, output, _)| (*start, output.clone()))
        .collect::<Vec<_>>();
    assert_eq!(
        streams.iter().map(|(start, _)| *start).collect::<Vec<_>>(),
        [0, 5, 6]
    );

    let entries =
        layer1_engram_capture::native_layer_one_block_entries_from_streams(Some(&streams));
    assert_eq!(entries.len(), streams.len());
    assert_eq!(
        entries.iter().map(|(start, _)| *start).collect::<Vec<_>>(),
        [0, 5, 6]
    );
    let layer_one_norm = bf16(field(
        field(&root, "parameters"),
        "layers.1.attn_norm.weight",
    ));
    for ((case, (start, entry)), (_, _, native_pre)) in
        cases.iter().zip(&entries).zip(&native_layer_zero)
    {
        assert_eq!(*start, usize_field(case, "start_pos"));
        assert_eq!(
            entry,
            &bf16(field(field(case, "downstream"), "layer_one_engram_output")),
            "native layer-one Engram output"
        );
        let mut attention_input = Vec::new();
        for (residual, pre) in entry.chunks_exact(256).zip(native_pre.chunks_exact(2)) {
            let prepared = AttentionInput::new(&layer_one_norm, 2, 1.0e-20)
                .unwrap()
                .forward(residual, pre)
                .unwrap();
            attention_input.extend_from_slice(prepared.normalized_bf16());
        }
        assert_eq!(
            attention_input,
            bf16(field(
                field(case, "downstream"),
                "layer_one_attention_input"
            )),
            "native layer-zero pre-mix feeds layer-one attention input"
        );
    }
}

#[test]
fn corrupting_layer_zero_hc_projection_changes_native_coefficients() {
    let root: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/layer0-to-layer1-reference.json"
    ))
    .expect("layer-zero bridge fixture JSON");
    let case = &field(&root, "cases").as_array().expect("bridge cases")[0];
    let (residual, _) = native_startup(&root, case);
    let baseline = native_hc_coefficients(&root, "attn", &residual[..256]);
    let model = field(&root, "model");
    let parameters = field(&root, "parameters");
    let projection = fp32(field(parameters, "layers.0.hc_attn_fn"));
    let scale: [f32; 3] = fp32(field(parameters, "layers.0.hc_attn_scale"))
        .try_into()
        .expect("three HC scales");
    let changed = projection[..512]
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0.0)
        .find_map(|(index, _)| {
            let mut changed_projection = projection.clone();
            changed_projection[index] *= 2.0;
            let changed = project_hc_coefficients(
                &residual[..256],
                &changed_projection,
                &scale,
                &fp32(field(parameters, "layers.0.hc_attn_base")),
                2,
                serde_json::from_value(field(model, "norm_eps").clone()).expect("HC norm epsilon"),
                usize_field(model, "hc_sinkhorn_iters"),
                serde_json::from_value(field(model, "hc_eps").clone()).expect("HC epsilon"),
            )
            .expect("changed layer-zero HC projection");
            (changed.pre() != baseline.pre()).then_some(changed)
        });
    assert!(
        changed.is_some(),
        "a nonzero attention-pre projection coefficient must affect native HC pre"
    );
}

#[test]
fn window_only_attention_rejects_out_of_order_decode_then_resets_and_retries() {
    let root: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/layer0-to-layer1-reference.json"
    ))
    .expect("layer-zero bridge fixture JSON");
    let cases = field(&root, "cases").as_array().expect("cases");
    let frequencies = frequencies(field(field(&cases[0], "attention"), "frequencies"));
    let weights = attention_weights(&root);
    let mut state = LayerAttentionState::new(window_only_layout(&root));

    assert!(matches!(
        forward_window_case(
            &mut state,
            &cases[1],
            &native_attention_input(&root, &cases[1]),
            &frequencies,
            weights.borrowed(),
        ),
        Err(LayerAttentionError::DiscontinuousPosition {
            expected: None,
            actual: 5
        })
    ));
    let prefill = forward_window_case(
        &mut state,
        &cases[0],
        &native_attention_input(&root, &cases[0]),
        &frequencies,
        weights.borrowed(),
    )
    .expect("rejected decode leaves prefill retryable");
    assert_attention_diagnostic(&cases[0], &prefill);
    let decode = forward_window_case(
        &mut state,
        &cases[1],
        &native_attention_input(&root, &cases[1]),
        &frequencies,
        weights.borrowed(),
    )
    .expect("prefill then decode");
    assert_attention_diagnostic(&cases[1], &decode);
    let reset = forward_window_case(
        &mut state,
        &cases[0],
        &native_attention_input(&root, &cases[0]),
        &frequencies,
        weights.borrowed(),
    )
    .expect("prefill resets the local window");
    assert_attention_diagnostic(&cases[0], &reset);
    let retry = forward_window_case(
        &mut state,
        &cases[1],
        &native_attention_input(&root, &cases[1]),
        &frequencies,
        weights.borrowed(),
    )
    .expect("reset prefill admits decode retry");
    assert_eq!(
        retry.final_output, decode.final_output,
        "reset retry must replay source output"
    );
}

#[test]
fn layer_zero_startup_rejects_unknown_tokens_and_changed_embedding_rows() {
    let root: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/layer0-to-layer1-reference.json"
    ))
    .expect("layer-zero bridge fixture JSON");
    let case = &field(&root, "cases").as_array().expect("cases")[0];
    let startup = field(case, "startup");
    let ids = u64s(field(startup, "input_ids"));
    let mut table = bf16(field(field(&root, "parameters"), "embed.weight"));
    let layout = StartupLayout::new(8, 128, 2).expect("startup layout");
    assert!(startup_bf16_reference(&[8], &table, layout).is_err());
    let expected = startup_bf16_reference(&ids, &table, layout)
        .expect("source startup")
        .residual_bf16()
        .to_vec();
    let row = usize::try_from(ids[0]).expect("source token row");
    table[row * 128] ^= 1;
    assert_ne!(
        startup_bf16_reference(&ids, &table, layout)
            .expect("changed in-range startup row")
            .residual_bf16(),
        expected,
        "a changed source embedding row must not pass the startup oracle"
    );
}

#[test]
fn runtime_startup_reset_replays_both_source_schedules() {
    let raw = include_str!("../../../../fixtures/deepseek-v41/layer0-to-layer1-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let canonical: Value = serde_json::from_str(raw).unwrap();
    for root in [canonical, alternate_startup_projection()] {
        let cases = root["cases"].as_array().unwrap();
        let all = frequencies(&cases[0]["attention"]["frequencies"]);
        with_runtime_startup(&root, |mut runtime| {
            assert!(matches!(
                runtime.step(1, &[0], &[]),
                Err(deepseek::reduced::StartupSessionError::UnexpectedStart { .. })
            ));
            assert!(!runtime.is_poisoned());
            for _ in 0..2 {
                for case in cases {
                    let start = usize_field(case, "start_pos");
                    let ids = u64s(&case["startup"]["input_ids"]);
                    let output = runtime
                        .step(start, &ids, &all[start * 16..(start + ids.len()) * 16])
                        .unwrap();
                    assert_runtime_startup(&root, case, &output);
                    assert_eq!(runtime.next_start(), start + ids.len());
                }
                runtime.reset().unwrap();
                assert_eq!(runtime.next_start(), 0);
                assert!(!runtime.is_poisoned());
            }
        });
    }
}

#[test]
fn runtime_startup_late_tail_failure_invalidates_attention_state() {
    let mut root = alternate_startup_projection();
    let tensor = &mut root["parameters"]["layers.0.ffn_norm.weight"];
    let mut storage = bytes(tensor);
    storage[..2].copy_from_slice(&0x7fc0_u16.to_le_bytes());
    let mut hex = String::with_capacity(storage.len() * 2);
    for byte in &storage {
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    tensor["storage_hex"] = Value::String(hex);
    tensor["storage_sha256"] = Value::String(format!("{:x}", Sha256::digest(&storage)));
    let case = &root["cases"][0];
    let ids = u64s(&case["startup"]["input_ids"]);
    let all = frequencies(&case["attention"]["frequencies"]);
    with_runtime_startup(&root, |mut runtime| {
        for _ in 0..2 {
            assert!(matches!(
                runtime.step(0, &ids, &all[..ids.len() * 16]),
                Err(deepseek::reduced::StartupSessionError::Tail(_))
            ));
            assert!(runtime.is_poisoned());
            assert_eq!(runtime.next_start(), 0);
            assert!(matches!(
                runtime.step(0, &ids, &all[..ids.len() * 16]),
                Err(deepseek::reduced::StartupSessionError::Poisoned)
            ));
            runtime.reset().unwrap();
            assert!(!runtime.is_poisoned());
        }
    });
}
