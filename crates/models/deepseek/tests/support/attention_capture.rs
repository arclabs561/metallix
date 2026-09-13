//! Shared source-captured layer-four attention oracle for integration tests.
//!
//! The compressed KV publication is a layer-three source view.  This helper
//! consumes its supplied layer-four IDs; it does not recreate the upstream
//! indexer, compressor, or reindexing.

#![allow(
    dead_code,
    reason = "each integration-test binary uses a deliberately narrow subset of this shared oracle"
)]

use std::{collections::BTreeMap, num::NonZeroUsize};

use deepseek::{
    RotaryFrequency,
    attention::layer::{
        CompressedAttentionPublication, Fp8Projection, LayerAttentionDiagnostic,
        LayerAttentionError, LayerAttentionLayout, LayerAttentionState, LayerAttentionWeights,
    },
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

pub(super) const SOURCE_LAYER: u16 = 3;
const FREQUENCY_PAIRS_PER_POSITION: usize = 16;

#[derive(Deserialize)]
pub(super) struct Fixture {
    schema_version: u32,
    source: BTreeMap<String, String>,
    pub(super) model: Model,
    frequencies: Frequencies,
    pub(super) encoded_parameters: BTreeMap<String, Tensor>,
    pub(super) cases: Vec<Case>,
    comparison_policy: Policy,
}

#[derive(Deserialize)]
pub(super) struct Model {
    pub(super) dim: usize,
    pub(super) head_dim: usize,
    pub(super) n_heads: usize,
    pub(super) q_lora_rank: usize,
    pub(super) rope_head_dim: usize,
    pub(super) window_size: usize,
    pub(super) o_groups: usize,
    pub(super) o_lora_rank: usize,
    pub(super) compress_ratios: Vec<usize>,
    candidate_source_layer: usize,
    pub(super) norm_eps: f32,
}

#[derive(Deserialize)]
struct Frequencies {
    shape: Vec<usize>,
    complex_dtype: String,
    fp32_pairs: Vec<[u32; 2]>,
}

#[derive(Deserialize)]
struct Policy {
    fixed_before_candidate_execution: bool,
    output_bf16: String,
    compressed_kv_and_indices: String,
}

#[derive(Deserialize)]
pub(super) struct Case {
    pub(super) start_pos: usize,
    pub(super) input: Tensor,
    pub(super) wq_a_output: Tensor,
    pub(super) q_norm_output: Tensor,
    pub(super) wq_b_pre_rope: Tensor,
    pub(super) q_after_rope: Tensor,
    pub(super) prepared_window_kv: Tensor,
    pub(super) window_kv: Tensor,
    pub(super) window_indices: Tensor,
    pub(super) window_ring_after: Tensor,
    pub(super) compressed_kv: Tensor,
    pub(super) compressed_indices: Tensor,
    pub(super) sparse_output_pre_inverse_rope: Tensor,
    pub(super) wo_b_input: Tensor,
    pub(super) output: Tensor,
}

#[derive(Deserialize)]
pub(super) struct Tensor {
    pub(super) dtype: String,
    pub(super) shape: Vec<usize>,
    storage_hex: String,
    storage_sha256: String,
}

impl Tensor {
    fn bytes(&self) -> Vec<u8> {
        assert_eq!(self.storage_hex.len() % 2, 0, "hex storage alignment");
        assert_eq!(
            self.storage_sha256.len(),
            64,
            "tensor storage SHA-256 width"
        );
        assert!(
            self.storage_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        );
        let bytes: Vec<_> = self
            .storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), self.storage_sha256);
        bytes
    }

    pub(super) fn bf16(&self) -> Vec<u16> {
        assert_eq!(self.dtype, "torch.bfloat16");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 2);
        bytes
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    fn fp32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "torch.float32");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 4);
        bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    fn i32(&self) -> Vec<i32> {
        assert_eq!(self.dtype, "torch.int32");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 4);
        bytes
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }
}

pub(super) fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-attention-reference.json"
    ))
    .expect("valid source attention fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.cases.len(), 3);
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| case.start_pos)
            .collect::<Vec<_>>(),
        [0, 5, 6]
    );
    assert_source_provenance(&fixture.source);
    assert_fixture_contract(&fixture);
    fixture
}

fn assert_source_provenance(source: &BTreeMap<String, String>) {
    for field in [
        "model_sha256",
        "engram_sha256",
        "kernel_source_sha256",
        "cpu_backend_sha256",
        "loader_sha256",
        "runner_sha256",
        "attention_helper_sha256",
        "complete_capture_sha256",
        "manifest_canonical_sha256",
    ] {
        let value = source.get(field).expect("required source provenance field");
        assert_eq!(value.len(), 64, "{field} SHA-256 width");
        assert!(
            value.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "{field} hex"
        );
    }
    assert_eq!(
        source.get("revision").map(String::as_str),
        Some("dba1be0a40aa45a94ad051997016db3960a90277")
    );
    assert_eq!(
        source.get("storage_byteorder").map(String::as_str),
        Some("little")
    );
}

fn assert_fixture_contract(fixture: &Fixture) {
    let model = &fixture.model;
    assert_eq!((model.dim, model.head_dim, model.n_heads), (128, 64, 2));
    assert_eq!((model.q_lora_rank, model.rope_head_dim), (32, 32));
    assert_eq!(
        (model.window_size, model.o_groups, model.o_lora_rank),
        (6, 2, 32)
    );
    assert_eq!(model.candidate_source_layer, usize::from(SOURCE_LAYER));
    assert_eq!(model.compress_ratios, [0, 2, 2, 1, 1]);
    assert_eq!(model.norm_eps.to_bits(), 1.0e-20_f32.to_bits());
    assert_eq!(fixture.frequencies.shape, [8, 16]);
    assert_eq!(fixture.frequencies.complex_dtype, "torch.complex64");
    assert_eq!(fixture.frequencies.fp32_pairs.len(), 128);
    assert_eq!(
        fixture.comparison_policy.output_bf16,
        "exact Attention.forward output after source inverse RoPE and output projections"
    );
    assert!(fixture.comparison_policy.fixed_before_candidate_execution);
    assert_eq!(
        fixture.comparison_policy.compressed_kv_and_indices,
        "source _compress_kv read from shared layer-three publication; indices retain source offset domain"
    );
    for name in [
        "layers.4.attn.wq_a.weight",
        "layers.4.attn.wq_a.scale",
        "layers.4.attn.q_norm.weight",
        "layers.4.attn.wq_b.weight",
        "layers.4.attn.wq_b.scale",
        "layers.4.attn.wkv.weight",
        "layers.4.attn.wkv.scale",
        "layers.4.attn.kv_norm.weight",
        "layers.4.attn.attn_sink",
        "layers.4.attn.wo_a.weight",
        "layers.4.attn.wo_b.weight",
        "layers.4.attn.wo_b.scale",
        "layers.4.attn.indexer.wq_b.weight",
        "layers.4.attn.indexer.wq_b.scale",
        "layers.4.attn.indexer.weights_proj.weight",
    ] {
        assert!(
            fixture.encoded_parameters.contains_key(name),
            "captured {name}"
        );
    }
}

pub(super) fn frequencies(fixture: &Fixture) -> Vec<RotaryFrequency> {
    fixture
        .frequencies
        .fp32_pairs
        .iter()
        .map(|&[real, imaginary]| {
            RotaryFrequency::new(f32::from_bits(real), f32::from_bits(imaginary)).unwrap()
        })
        .collect()
}

pub(super) fn call_frequencies<'a>(
    all: &'a [RotaryFrequency],
    case: &Case,
) -> &'a [RotaryFrequency] {
    let [batches, positions, hidden] = case.input.shape.as_slice() else {
        panic!("source attention input must have [batch, position, hidden] shape");
    };
    assert_eq!(*batches, 1, "captured attention batch count");
    assert_eq!(*hidden, 128, "captured attention hidden width");
    let start = case
        .start_pos
        .checked_mul(FREQUENCY_PAIRS_PER_POSITION)
        .expect("bounded captured frequency offset");
    let count = positions
        .checked_mul(FREQUENCY_PAIRS_PER_POSITION)
        .expect("bounded captured frequency count");
    all.get(
        start
            ..start
                .checked_add(count)
                .expect("bounded captured frequency end"),
    )
    .expect("fixture has the call-local RoPE frequency span")
}

fn source_input_elements(case: &Case) -> usize {
    let [batches, sequence, hidden] = case.input.shape.as_slice() else {
        panic!("source attention input must have [batch, position, hidden] shape");
    };
    assert_eq!(*batches, 1, "captured attention batch count");
    assert_eq!(*hidden, 128, "captured attention hidden width");
    batches
        .checked_mul(*sequence)
        .and_then(|rows| rows.checked_mul(*hidden))
        .expect("bounded captured attention input elements")
}

pub(super) fn layout(model: &Model) -> LayerAttentionLayout {
    LayerAttentionLayout::new(
        nonzero(1),
        nonzero(model.dim),
        nonzero(model.n_heads),
        nonzero(model.head_dim),
        nonzero(model.rope_head_dim / 2),
        nonzero(model.q_lora_rank),
        nonzero(model.window_size),
        nonzero(model.o_groups),
        nonzero(model.o_lora_rank),
        SOURCE_LAYER,
        nonzero(model.compress_ratios[usize::from(SOURCE_LAYER)]),
        model.norm_eps,
        0.125,
    )
    .expect("source-shaped layer attention layout")
}

pub(super) fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("fixture dimensions are nonzero")
}

pub(super) struct EncodedWeights {
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

impl EncodedWeights {
    pub(super) fn borrowed(&self) -> LayerAttentionWeights<'_> {
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

pub(super) fn weights(parameters: &BTreeMap<String, Tensor>) -> EncodedWeights {
    EncodedWeights {
        wq_a_codes: fp8_codes(parameters, "layers.4.attn.wq_a.weight"),
        wq_a_scales: fp8_scales(parameters, "layers.4.attn.wq_a.scale"),
        q_norm: bf16(parameters, "layers.4.attn.q_norm.weight"),
        wq_b_codes: fp8_codes(parameters, "layers.4.attn.wq_b.weight"),
        wq_b_scales: fp8_scales(parameters, "layers.4.attn.wq_b.scale"),
        wkv_codes: fp8_codes(parameters, "layers.4.attn.wkv.weight"),
        wkv_scales: fp8_scales(parameters, "layers.4.attn.wkv.scale"),
        kv_norm: bf16(parameters, "layers.4.attn.kv_norm.weight"),
        attn_sink: fp32(parameters, "layers.4.attn.attn_sink"),
        wo_a: bf16(parameters, "layers.4.attn.wo_a.weight"),
        wo_b_codes: fp8_codes(parameters, "layers.4.attn.wo_b.weight"),
        wo_b_scales: fp8_scales(parameters, "layers.4.attn.wo_b.scale"),
    }
}

fn bf16(parameters: &BTreeMap<String, Tensor>, name: &str) -> Vec<u16> {
    parameters[name].bf16()
}
fn fp32(parameters: &BTreeMap<String, Tensor>, name: &str) -> Vec<f32> {
    parameters[name].fp32()
}
fn fp8_codes(parameters: &BTreeMap<String, Tensor>, name: &str) -> Vec<u8> {
    assert_eq!(
        parameters[name].dtype, "torch.float8_e4m3fn",
        "{name} dtype"
    );
    parameters[name].bytes()
}
fn fp8_scales(parameters: &BTreeMap<String, Tensor>, name: &str) -> Vec<u8> {
    assert_eq!(
        parameters[name].dtype, "torch.float8_e8m0fnu",
        "{name} dtype"
    );
    parameters[name].bytes()
}

pub(super) fn forward_case(
    state: &mut LayerAttentionState,
    case: &Case,
    epoch: u64,
    call_id: u64,
    source_layer: u16,
    frequencies: &[RotaryFrequency],
    weights: LayerAttentionWeights<'_>,
) -> Result<LayerAttentionDiagnostic, LayerAttentionError> {
    let input = case.input.bf16();
    let values = case.compressed_kv.bf16();
    let indices = case.compressed_indices.i32();
    forward_with_publication(
        state,
        &input,
        case.start_pos,
        epoch,
        call_id,
        source_layer,
        &values,
        &indices,
        call_frequencies(frequencies, case),
        weights,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "negative controls keep each publication boundary explicit"
)]
pub(super) fn forward_with_publication(
    state: &mut LayerAttentionState,
    input: &[u16],
    start_position: usize,
    epoch: u64,
    call_id: u64,
    source_layer: u16,
    numerical_bf16: &[u16],
    indices: &[i32],
    frequencies: &[RotaryFrequency],
    weights: LayerAttentionWeights<'_>,
) -> Result<LayerAttentionDiagnostic, LayerAttentionError> {
    state.forward(
        input,
        start_position,
        frequencies,
        weights,
        CompressedAttentionPublication {
            source_layer,
            epoch,
            call_id,
            numerical_bf16,
            indices,
        },
    )
}

pub(super) fn assert_diagnostic(case: &Case, diagnostic: &LayerAttentionDiagnostic) {
    assert_bf16_exact(case, "WQ-A", &diagnostic.wq_a, &case.wq_a_output.bf16());
    assert_bf16_exact(case, "Q norm", &diagnostic.qr, &case.q_norm_output.bf16());
    assert_bf16_exact(
        case,
        "WQ-B before RoPE",
        &diagnostic.wq_b_pre_rope,
        &case.wq_b_pre_rope.bf16(),
    );
    assert_bf16_exact(
        case,
        "sparse query after RoPE",
        &diagnostic.q_after_rope,
        &case.q_after_rope.bf16(),
    );
    assert_bf16_exact(
        case,
        "newly prepared window KV",
        &diagnostic.prepared_window,
        &case.prepared_window_kv.bf16(),
    );
    assert_bf16_exact(
        case,
        "window KV read",
        &diagnostic.window_read,
        &case.window_kv.bf16(),
    );
    assert_i32_exact(
        case,
        "window top-k IDs before compressed concatenation",
        &diagnostic.window_indices,
        &case.window_indices.i32(),
    );
    assert_bf16_exact(
        case,
        "window ring after write",
        &diagnostic.ring_after,
        &case.window_ring_after.bf16(),
    );
    assert_bf16_exact(
        case,
        "sparse output before inverse RoPE",
        &diagnostic.sparse_output,
        &case.sparse_output_pre_inverse_rope.bf16(),
    );
    assert_bf16_exact(
        case,
        "source attention output",
        &diagnostic.final_output,
        &case.output.bf16(),
    );
}

fn assert_bf16_exact(case: &Case, stage: &str, native: &[u16], source: &[u16]) {
    assert_eq!(
        native.len(),
        source.len(),
        "{stage} start {} BF16 length",
        case.start_pos
    );
    if let Some((index, (&native, &source))) = native
        .iter()
        .zip(source)
        .enumerate()
        .find(|&(_, (&native, &source))| native != source)
    {
        panic!(
            "{stage} start {} first BF16 mismatch at {index}: native {native:#06x}, source {source:#06x}",
            case.start_pos
        );
    }
}

fn assert_i32_exact(case: &Case, stage: &str, native: &[i32], source: &[i32]) {
    assert_eq!(
        native.len(),
        source.len(),
        "{stage} start {} ID length",
        case.start_pos
    );
    if let Some((index, (&native, &source))) = native
        .iter()
        .zip(source)
        .enumerate()
        .find(|&(_, (&native, &source))| native != source)
    {
        panic!(
            "{stage} start {} first ID mismatch at {index}: native {native}, source {source}",
            case.start_pos
        );
    }
}

pub(super) fn prefetched_state(
    fixture: &Fixture,
    frequencies: &[RotaryFrequency],
    weights: LayerAttentionWeights<'_>,
) -> LayerAttentionState {
    let mut state = LayerAttentionState::new(layout(&fixture.model));
    forward_case(
        &mut state,
        &fixture.cases[0],
        0,
        0,
        SOURCE_LAYER,
        frequencies,
        weights,
    )
    .expect("source prefill");
    state
}

/// Runs the three captured attention calls as one native cache-continuous sequence.
/// Supplied call starts and BF16 inputs are checked against the pinned source fixture.
pub(super) fn native_outputs_from_inputs(
    inputs: &[(usize, Vec<u16>)],
    expected_capture_sha256: &str,
) -> Vec<Vec<u16>> {
    let fixture = fixture();
    assert_eq!(
        fixture
            .source
            .get("complete_capture_sha256")
            .map(String::as_str),
        Some(expected_capture_sha256),
        "attention complete capture identity"
    );
    assert_eq!(
        inputs.len(),
        fixture.cases.len(),
        "complete captured attention call count"
    );
    for (call_id, (start, input)) in inputs.iter().enumerate() {
        let case = &fixture.cases[call_id];
        assert_eq!(
            *start, case.start_pos,
            "captured attention start at call {call_id}"
        );
        assert_eq!(
            input.len(),
            source_input_elements(case),
            "captured attention B/S input shape at call {call_id}"
        );
        assert_bf16_exact(case, "supplied attention input", input, &case.input.bf16());
    }
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let mut state = LayerAttentionState::new(layout(&fixture.model));
    inputs
        .iter()
        .enumerate()
        .map(|(call_id, (start, input))| {
            let case = &fixture.cases[call_id];
            let compressed = case.compressed_kv.bf16();
            let indices = case.compressed_indices.i32();
            let diagnostic = forward_with_publication(
                &mut state,
                input,
                *start,
                0,
                u64::try_from(call_id).expect("three captured calls"),
                SOURCE_LAYER,
                &compressed,
                &indices,
                call_frequencies(&frequencies, case),
                weights.borrowed(),
            )
            .expect("exact source-shaped attention execution");
            assert_diagnostic(case, &diagnostic);
            diagnostic.final_output
        })
        .collect()
}
