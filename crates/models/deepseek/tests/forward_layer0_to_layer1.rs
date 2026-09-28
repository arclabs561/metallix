//! Same-trace layer-zero block output consumed by the native layer-one Engram.

use std::{collections::BTreeMap, num::NonZeroUsize};

use deepseek::{
    RotaryFrequency, StartupLayout,
    attention::layer::{
        Fp8Projection, LayerAttentionDiagnostic, LayerAttentionError, LayerAttentionLayout,
        LayerAttentionState, LayerAttentionWeights,
    },
    hc::mixing::{hc_post_bf16_reference, hc_pre_bf16_reference},
    moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEReference},
    rms_norm_bf16_reference, startup_bf16_reference,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[path = "support/layer1_engram_capture.rs"]
#[allow(
    dead_code,
    reason = "the bridge test needs the supplied-stream entry helper from the shared Engram fixture oracle"
)]
mod layer1_engram_capture;

const FIXTURE_SHA256: &str = "75b65ac8b23ea8fbe18aa8741098d511d1a7e2f3203252a246fec6ecc09f285a";

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
    let mut attention_input = Vec::with_capacity(positions * 128);
    for position in 0..positions {
        let mut collapsed = vec![0; 128];
        hc_pre_bf16_reference(
            &residual[position * 256..(position + 1) * 256],
            &incoming_pre[position * 2..(position + 1) * 2],
            128,
            &mut collapsed,
        )
        .expect("native layer-zero incoming HC pre-mix");
        let mut normalized = vec![0; 128];
        rms_norm_bf16_reference(&collapsed, &norm_weight, 1.0e-20, &mut normalized)
            .expect("native layer-zero attention RMSNorm");
        attention_input.extend(normalized);
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

fn native_layer_zero_attention_outputs(root: &Value) -> Vec<(usize, Vec<u16>)> {
    let all_frequencies = frequencies(field(
        field(
            &field(root, "cases").as_array().expect("cases")[0],
            "attention",
        ),
        "frequencies",
    ));
    assert_eq!(all_frequencies.len(), 8 * 16, "source frequency table");
    let weights = attention_weights(root);
    let mut state = LayerAttentionState::new(window_only_layout(root));
    field(root, "cases")
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| {
            let start = usize_field(case, "start_pos");
            let attention_input = native_attention_input(root, case);
            let diagnostic = forward_window_case(
                &mut state,
                case,
                &attention_input,
                &all_frequencies,
                weights.borrowed(),
            )
            .expect("native layer-zero window-only attention");
            assert_attention_diagnostic(case, &diagnostic);
            (start, diagnostic.final_output)
        })
        .collect()
}

fn coefficients(case: &Value, sublayer: &str) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let values = field(field(field(case, "hc"), sublayer), "coefficients");
    (
        fp32(field(values, "pre")),
        fp32(field(values, "post")),
        fp32(field(values, "comb")),
    )
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

fn native_layer_zero_output(
    root: &Value,
    case: &Value,
    attention: &[u16],
    moe: &MoEReference<'_>,
) -> (Vec<u16>, Vec<f32>) {
    let positions = usize::try_from(
        field(field(case, "block_output"), "shape")
            .as_array()
            .expect("shape")[1]
            .as_u64()
            .expect("positions"),
    )
    .expect("usize positions");
    let residual = bf16(field(field(case, "block_input"), "residual"));
    let expected_attention = bf16(field(case, "attention_output"));
    assert_eq!(
        attention, expected_attention,
        "native layer-zero attention output"
    );
    let expected_after_attention = bf16(field(case, "after_attention_residual"));
    let expected_ffn_collapse = bf16(field(case, "ffn_collapsed"));
    let expected_ffn = bf16(field(case, "ffn_output"));
    let ffn_norm = bf16(field(field(root, "parameters"), "layers.0.ffn_norm.weight"));
    let expected_output = bf16(field(case, "block_output"));
    let expected_next_pre = fp32(field(case, "block_next_pre"));
    let (attention_pre, attention_post, attention_comb) = coefficients(case, "attention");
    let (ffn_pre, ffn_post, ffn_comb) = coefficients(case, "ffn");
    let mut output = Vec::with_capacity(expected_output.len());
    for position in 0..positions {
        let residual = &residual[position * 256..(position + 1) * 256];
        let mut after_attention = vec![0; 256];
        hc_post_bf16_reference(
            &attention[position * 128..(position + 1) * 128],
            residual,
            &attention_post[position * 2..(position + 1) * 2],
            &attention_comb[position * 4..(position + 1) * 4],
            &mut after_attention,
        )
        .expect("native layer-zero attention HC post mix");
        assert_eq!(
            after_attention,
            expected_after_attention[position * 256..(position + 1) * 256],
            "native layer-zero attention residual at position {position}"
        );
        let mut collapsed = vec![0; 128];
        hc_pre_bf16_reference(
            &after_attention,
            &attention_pre[position * 2..(position + 1) * 2],
            128,
            &mut collapsed,
        )
        .expect("native layer-zero attention HC pre mix");
        assert_eq!(
            collapsed,
            expected_ffn_collapse[position * 128..(position + 1) * 128],
            "native layer-zero FFN collapse at position {position}"
        );
        let mut normalized = vec![0; 128];
        rms_norm_bf16_reference(&collapsed, &ffn_norm, 1e-20, &mut normalized)
            .expect("native layer-zero FFN normalization");
        assert_eq!(
            normalized,
            bf16(field(case, "ffn_input"))[position * 128..(position + 1) * 128],
            "native layer-zero FFN input at position {position}"
        );
        let moe_output = moe
            .forward_token(&normalized)
            .expect("native layer-zero MoE");
        assert_eq!(
            moe_output.output_bf16(),
            &expected_ffn[position * 128..(position + 1) * 128],
            "native layer-zero MoE output at position {position}"
        );
        let mut terminal = vec![0; 256];
        hc_post_bf16_reference(
            moe_output.output_bf16(),
            &after_attention,
            &ffn_post[position * 2..(position + 1) * 2],
            &ffn_comb[position * 4..(position + 1) * 4],
            &mut terminal,
        )
        .expect("native layer-zero FFN HC post mix");
        assert_eq!(
            terminal,
            expected_output[position * 256..(position + 1) * 256],
            "native layer-zero terminal residual at position {position}"
        );
        output.extend(terminal);
    }
    assert_eq!(
        ffn_pre, expected_next_pre,
        "native layer-zero next HC pre-mix"
    );
    (output, ffn_pre)
}

#[test]
fn source_layer_zero_output_feeds_native_layer_one_engram_entries() {
    let raw = include_str!("../../../../fixtures/deepseek-v41/layer0-to-layer1-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let root: Value = serde_json::from_str(raw).expect("layer-zero bridge fixture JSON");
    assert_eq!(field(&root, "schema_version").as_u64(), Some(1));
    assert_eq!(
        field(field(&root, "contract"), "layer_zero_producer").as_str(),
        Some("source-pinned native attention and FFN with captured upstream HC inputs")
    );
    let cases = field(&root, "cases").as_array().expect("bridge cases");
    let attention_outputs = native_layer_zero_attention_outputs(&root);
    let streams = with_native_layer_zero_moe(&root, |moe| {
        cases
            .iter()
            .map(|case| {
                let start = usize::try_from(field(case, "start_pos").as_u64().expect("start"))
                    .expect("usize start");
                let output = field(case, "block_output");
                let stream = field(case, "layer_one_engram_stream");
                assert_eq!(
                    field(output, "storage_sha256"),
                    field(stream, "storage_sha256"),
                    "layer-zero output must be the exact layer-one stream"
                );
                let attention = attention_outputs
                    .iter()
                    .find_map(|(attention_start, output)| {
                        (*attention_start == start).then_some(output.as_slice())
                    })
                    .expect("native attention case");
                let (native_output, _next_pre) =
                    native_layer_zero_output(&root, case, attention, &moe);
                assert_eq!(native_output, bf16(output));
                (start, native_output)
            })
            .collect::<Vec<_>>()
    });
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
}

#[test]
fn corrupting_layer_zero_hc_post_coefficient_fails_exact_terminal_oracle() {
    let root: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/layer0-to-layer1-reference.json"
    ))
    .expect("layer-zero bridge fixture JSON");
    let case = &field(&root, "cases").as_array().expect("bridge cases")[0];
    let residual = bf16(field(field(case, "block_input"), "residual"));
    let attention = bf16(field(case, "attention_output"));
    let (_, mut post, comb) = coefficients(case, "attention");
    post[0] += 1.0;
    let mut corrupted = vec![0; 256];
    hc_post_bf16_reference(
        &attention[..128],
        &residual[..256],
        &post[..2],
        &comb[..4],
        &mut corrupted,
    )
    .expect("corrupted layer-zero HC control");
    assert_ne!(
        corrupted,
        bf16(field(case, "after_attention_residual"))[..256],
        "a changed source HC coefficient must fail the exact source residual"
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
