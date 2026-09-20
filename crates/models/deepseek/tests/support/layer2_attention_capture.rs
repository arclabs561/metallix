//! Test-only native layer-two attention against its source capture.
//!
//! Layer one owns the ratio-two compressed publication.  This helper checks
//! that publication's captured boundary and uses the production layer-attention
//! adapter for layer two; it does not recreate layer-one candidate scoring.

#![allow(
    dead_code,
    reason = "the standalone attention gate and layer-two HC/FFN join consume distinct helper entry points"
)]

#[allow(
    dead_code,
    reason = "the layer-one publication oracle has controls used only by its standalone binary"
)]
#[path = "layer1_owner_capture.rs"]
mod layer1_owner_capture;

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    attention::layer::{
        CompressedAttentionPublication, Fp8Projection, LayerAttentionDiagnostic,
        LayerAttentionError, LayerAttentionLayout, LayerAttentionState, LayerAttentionWeights,
    },
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE_SHA256: &str = "d5c2225419cf3eeb36a2619b1c6829716db390d9b216a4e26275e7bb86a1be54";
const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";

pub(super) struct NativeOutput {
    pub(super) start_pos: usize,
    pub(super) output: Vec<u16>,
}

fn fixture() -> Value {
    let raw = include_str!("../../../../../fixtures/deepseek-v41/layer2-attention-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let fixture: Value = serde_json::from_str(raw).expect("layer-two attention fixture JSON");
    assert_eq!(field(&fixture, "schema_version").as_u64(), Some(1));
    assert_eq!(
        field(field(&fixture, "source"), "revision").as_str(),
        Some(REVISION)
    );
    assert_eq!(
        field(field(&fixture, "source"), "storage_byteorder").as_str(),
        Some("little")
    );
    assert!(field(&fixture, "scope").as_str().is_some());
    fixture
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value
        .get(name)
        .unwrap_or_else(|| panic!("missing fixture field {name}"))
}

fn usize_field(value: &Value, name: &str) -> usize {
    usize::try_from(field(value, name).as_u64().expect("fixture unsigned value"))
        .expect("fixture value fits usize")
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("source dimensions are nonzero")
}

fn bytes(tensor: &Value, width: usize) -> Vec<u8> {
    let shape = field(tensor, "shape")
        .as_array()
        .expect("fixture tensor shape");
    let elements = shape
        .iter()
        .map(|value| usize::try_from(value.as_u64().expect("shape dimension")).expect("usize"))
        .try_fold(1_usize, usize::checked_mul)
        .expect("fixture tensor size");
    assert_eq!(usize_field(tensor, "numel"), elements);
    let hex = field(tensor, "storage_hex")
        .as_str()
        .expect("fixture storage hex");
    assert!(hex.len().is_multiple_of(2));
    let bytes: Vec<_> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).expect("hex"), 16).expect("byte"))
        .collect();
    assert_eq!(bytes.len(), elements * width);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        field(tensor, "storage_sha256")
            .as_str()
            .expect("storage hash")
    );
    bytes
}

fn bf16(tensor: &Value) -> Vec<u16> {
    assert_eq!(field(tensor, "dtype").as_str(), Some("torch.bfloat16"));
    bytes(tensor, 2)
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
        .collect()
}

fn fp8(tensor: &Value) -> Vec<u8> {
    assert!(matches!(
        field(tensor, "dtype").as_str(),
        Some("torch.float8_e4m3fn" | "torch.float8_e8m0fnu")
    ));
    bytes(tensor, 1)
}

fn fp32(tensor: &Value) -> Vec<f32> {
    assert_eq!(field(tensor, "dtype").as_str(), Some("torch.float32"));
    bytes(tensor, 4)
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("FP32 word")))
        .collect()
}

fn i32s(tensor: &Value) -> Vec<i32> {
    assert_eq!(field(tensor, "dtype").as_str(), Some("torch.int32"));
    bytes(tensor, 4)
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().expect("i32 word")))
        .collect()
}

fn frequencies(root: &Value) -> Vec<RotaryFrequency> {
    let frequencies = field(root, "frequencies");
    assert_eq!(field(frequencies, "shape"), &serde_json::json!([8, 16]));
    field(frequencies, "fp32_pairs")
        .as_array()
        .expect("frequency pairs")
        .iter()
        .map(|pair| {
            let pair = pair.as_array().expect("complex pair");
            RotaryFrequency::new(
                f32::from_bits(u32::try_from(pair[0].as_u64().expect("real")).expect("u32")),
                f32::from_bits(u32::try_from(pair[1].as_u64().expect("imag")).expect("u32")),
            )
            .expect("finite source frequency")
        })
        .collect()
}

struct Weights {
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

impl Weights {
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

fn weights(root: &Value) -> Weights {
    let parameters = field(root, "encoded_parameters");
    let tensor = |name| field(parameters, name);
    Weights {
        wq_a_codes: fp8(tensor("layers.2.attn.wq_a.weight")),
        wq_a_scales: fp8(tensor("layers.2.attn.wq_a.scale")),
        q_norm: bf16(tensor("layers.2.attn.q_norm.weight")),
        wq_b_codes: fp8(tensor("layers.2.attn.wq_b.weight")),
        wq_b_scales: fp8(tensor("layers.2.attn.wq_b.scale")),
        wkv_codes: fp8(tensor("layers.2.attn.wkv.weight")),
        wkv_scales: fp8(tensor("layers.2.attn.wkv.scale")),
        kv_norm: bf16(tensor("layers.2.attn.kv_norm.weight")),
        attn_sink: fp32(tensor("layers.2.attn.attn_sink")),
        wo_a: bf16(tensor("layers.2.attn.wo_a.weight")),
        wo_b_codes: fp8(tensor("layers.2.attn.wo_b.weight")),
        wo_b_scales: fp8(tensor("layers.2.attn.wo_b.scale")),
    }
}

fn layout(root: &Value) -> LayerAttentionLayout {
    let model = field(root, "model");
    let ratios = field(model, "compress_ratios").as_array().expect("ratios");
    assert_eq!(ratios[1].as_u64(), Some(2));
    assert_eq!(field(model, "norm_eps").as_f64(), Some(1e-20));
    LayerAttentionLayout::new(
        nonzero(1),
        nonzero(usize_field(model, "dim")),
        nonzero(usize_field(model, "n_heads")),
        nonzero(usize_field(model, "head_dim")),
        nonzero(usize_field(model, "rope_head_dim") / 2),
        nonzero(usize_field(model, "q_lora_rank")),
        nonzero(usize_field(model, "window_size")),
        nonzero(usize_field(model, "o_groups")),
        nonzero(usize_field(model, "o_lora_rank")),
        1,
        nonzero(2),
        1.0e-20,
        0.125,
    )
    .expect("layer-two layout")
}

fn assert_exact(start: usize, stage: &str, native: &[u16], source: &[u16]) {
    assert_eq!(native, source, "start {start} {stage}");
}

fn assert_diagnostic(case: &Value, diagnostic: &LayerAttentionDiagnostic) {
    let start = usize_field(case, "start_pos");
    assert_exact(
        start,
        "WQ-A",
        &diagnostic.wq_a,
        &bf16(field(case, "wq_a_output")),
    );
    assert_exact(
        start,
        "Q norm",
        &diagnostic.qr,
        &bf16(field(case, "q_norm_output")),
    );
    assert_exact(
        start,
        "WQ-B",
        &diagnostic.wq_b_pre_rope,
        &bf16(field(case, "wq_b_pre_rope")),
    );
    assert_exact(
        start,
        "Q RoPE",
        &diagnostic.q_after_rope,
        &bf16(field(case, "q_after_rope")),
    );
    assert_exact(
        start,
        "window preparation",
        &diagnostic.prepared_window,
        &bf16(field(case, "prepared_window_kv")),
    );
    assert_exact(
        start,
        "window read",
        &diagnostic.window_read,
        &bf16(field(case, "window_kv")),
    );
    assert_eq!(
        diagnostic.window_indices,
        i32s(field(case, "window_indices")),
        "start {start} window IDs"
    );
    assert_exact(
        start,
        "window ring",
        &diagnostic.ring_after,
        &bf16(field(case, "window_ring_after")),
    );
    assert_exact(
        start,
        "sparse output",
        &diagnostic.sparse_output,
        &bf16(field(case, "sparse_output_pre_inverse_rope")),
    );
    assert_exact(
        start,
        "attention output",
        &diagnostic.final_output,
        &bf16(field(case, "output")),
    );
}

/// Runs layer-two attention with supplied native inputs and native layer-one publications.
///
/// The caller owns the earlier HC/FFN derivation.  Each supplied BF16 input is
/// checked only for captured call identity and geometry here; the source stage
/// assertions below then reject a numerically divergent attention continuation.
pub(super) fn native_outputs_from_inputs(inputs: &[(usize, Vec<u16>)]) -> Vec<(usize, Vec<u16>)> {
    let root = fixture();
    let frequencies = frequencies(&root);
    let weights = weights(&root);
    let owner = layer1_owner_capture::native_publications();
    let mut state = LayerAttentionState::new(layout(&root));
    let cases = field(&root, "cases").as_array().expect("source cases");
    assert_eq!(inputs.len(), cases.len(), "source call count");
    cases
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(call_id, (case, (supplied_start, input)))| {
            let start = usize_field(case, "start_pos");
            assert_eq!(*supplied_start, start, "captured call start {call_id}");
            assert_eq!(
                input.len(),
                bf16(field(case, "input")).len(),
                "start {start} input geometry"
            );
            let source_kv = bf16(field(case, "layer_one_published_kv"));
            let source_ids = i32s(field(case, "layer_one_published_indices"));
            assert_eq!(
                source_kv,
                bf16(field(case, "compressed_kv")),
                "start {start} source layer-one KV boundary"
            );
            assert_eq!(
                source_ids,
                i32s(field(case, "compressed_indices")),
                "start {start} source layer-one IDs boundary"
            );
            assert_eq!(owner[call_id].start_pos, start);
            assert_eq!(
                owner[call_id].kv_prefix, source_kv,
                "start {start} native layer-one KV publication"
            );
            assert_eq!(
                owner[call_id].selected_indices, source_ids,
                "start {start} native layer-one selected IDs"
            );
            let positions = input.len() / 128;
            let diagnostic = state
                .forward(
                    input,
                    start,
                    &frequencies[start * 16..(start + positions) * 16],
                    weights.borrowed(),
                    CompressedAttentionPublication {
                        source_layer: 1,
                        epoch: 0,
                        call_id: u64::try_from(call_id).expect("three calls"),
                        numerical_bf16: &owner[call_id].kv_prefix,
                        indices: &owner[call_id].selected_indices,
                    },
                )
                .expect("native layer-two attention");
            assert_diagnostic(case, &diagnostic);
            (start, diagnostic.final_output)
        })
        .collect()
}

/// Runs the captured layer-two inputs through the native attention chain.
pub(super) fn native_outputs() -> Vec<NativeOutput> {
    let root = fixture();
    let inputs = field(&root, "cases")
        .as_array()
        .expect("source cases")
        .iter()
        .map(|case| (usize_field(case, "start_pos"), bf16(field(case, "input"))))
        .collect::<Vec<_>>();
    native_outputs_from_inputs(&inputs)
        .into_iter()
        .map(|(start_pos, output)| NativeOutput { start_pos, output })
        .collect()
}

/// A publication from any layer other than the native layer-one owner is rejected.
pub(super) fn wrong_owner_publication_is_rejected() -> bool {
    let root = fixture();
    let case = &field(&root, "cases").as_array().expect("source cases")[0];
    let input = bf16(field(case, "input"));
    let frequencies = frequencies(&root);
    let weights = weights(&root);
    let start = usize_field(case, "start_pos");
    let positions = input.len() / 128;
    let result = LayerAttentionState::new(layout(&root)).forward(
        &input,
        start,
        &frequencies[start * 16..(start + positions) * 16],
        weights.borrowed(),
        CompressedAttentionPublication {
            source_layer: 3,
            epoch: 0,
            call_id: 0,
            numerical_bf16: &bf16(field(case, "compressed_kv")),
            indices: &i32s(field(case, "compressed_indices")),
        },
    );
    matches!(
        result,
        Err(LayerAttentionError::WrongSourceLayer {
            actual: 3,
            expected: 1
        })
    )
}

/// A causal, in-range replacement of one selected compressed key changes output.
pub(super) fn legal_wrong_index_changes_output() -> bool {
    let root = fixture();
    let case = &field(&root, "cases").as_array().expect("source cases")[0];
    let input = bf16(field(case, "input"));
    let frequencies = frequencies(&root);
    let weights = weights(&root);
    let owner = layer1_owner_capture::native_publications();
    let start = usize_field(case, "start_pos");
    let positions = input.len() / 128;
    let mut indices = owner[0].selected_indices.clone();
    assert_eq!(
        indices[4], 5,
        "captured prefill selects first compressed key"
    );
    indices[4] = 6;
    let changed = LayerAttentionState::new(layout(&root))
        .forward(
            &input,
            start,
            &frequencies[start * 16..(start + positions) * 16],
            weights.borrowed(),
            CompressedAttentionPublication {
                source_layer: 1,
                epoch: 0,
                call_id: 0,
                numerical_bf16: &owner[0].kv_prefix,
                indices: &indices,
            },
        )
        .expect("legal wrong layer-one publication index");
    changed.final_output != bf16(field(case, "output"))
}
