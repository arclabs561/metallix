//! Native layer-four attention against source-captured attention boundaries.
//!
//! The compressed KV publication is deliberately supplied by the fixture as a
//! layer-three source view. This test validates consumption of source-provided
//! layer-four IDs; it does not claim to recreate the upstream indexer,
//! compressor, or reindexing.

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

const SOURCE_LAYER: u16 = 3;
// The source fixture stores one complex RoPE pair for each of the 16 rotary
// pairs at every captured absolute position. Calls borrow their local span.
const FREQUENCY_PAIRS_PER_POSITION: usize = 16;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: BTreeMap<String, String>,
    model: Model,
    frequencies: Frequencies,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
    comparison_policy: Policy,
}

#[derive(Deserialize)]
struct Model {
    dim: usize,
    head_dim: usize,
    n_heads: usize,
    q_lora_rank: usize,
    rope_head_dim: usize,
    window_size: usize,
    o_groups: usize,
    o_lora_rank: usize,
    compress_ratios: Vec<usize>,
    candidate_source_layer: usize,
    norm_eps: f32,
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
struct Case {
    start_pos: usize,
    input: Tensor,
    wq_a_output: Tensor,
    q_norm_output: Tensor,
    wq_b_pre_rope: Tensor,
    q_after_rope: Tensor,
    prepared_window_kv: Tensor,
    window_kv: Tensor,
    window_indices: Tensor,
    window_ring_after: Tensor,
    compressed_kv: Tensor,
    compressed_indices: Tensor,
    sparse_output_pre_inverse_rope: Tensor,
    wo_b_input: Tensor,
    output: Tensor,
}

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
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
                .all(|byte| byte.is_ascii_hexdigit()),
            "tensor storage SHA-256 hex"
        );
        let bytes: Vec<_> = self
            .storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            self.storage_sha256,
            "tensor storage SHA-256"
        );
        bytes
    }

    fn bf16(&self) -> Vec<u16> {
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

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-attention-reference.json"
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

fn frequencies(fixture: &Fixture) -> Vec<RotaryFrequency> {
    fixture
        .frequencies
        .fp32_pairs
        .iter()
        .map(|&[real, imaginary]| {
            RotaryFrequency::new(f32::from_bits(real), f32::from_bits(imaginary)).unwrap()
        })
        .collect()
}

fn call_frequencies<'a>(all: &'a [RotaryFrequency], case: &Case) -> &'a [RotaryFrequency] {
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
    let end = start
        .checked_add(count)
        .expect("bounded captured frequency end");
    all.get(start..end)
        .expect("fixture has the call-local RoPE frequency span")
}

fn layout(model: &Model) -> LayerAttentionLayout {
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

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("fixture dimensions are nonzero")
}

struct EncodedWeights {
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

fn weights(parameters: &BTreeMap<String, Tensor>) -> EncodedWeights {
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

fn forward_case(
    state: &mut LayerAttentionState,
    case: &Case,
    epoch: u64,
    call_id: u64,
    source_layer: u16,
    frequencies: &[RotaryFrequency],
    weights: LayerAttentionWeights<'_>,
) -> Result<LayerAttentionDiagnostic, LayerAttentionError> {
    let input = case.input.bf16();
    let numerical_bf16 = case.compressed_kv.bf16();
    let indices = case.compressed_indices.i32();
    let call_frequencies = call_frequencies(frequencies, case);
    forward_with_publication(
        state,
        &input,
        case.start_pos,
        epoch,
        call_id,
        source_layer,
        &numerical_bf16,
        &indices,
        call_frequencies,
        weights,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the negative controls need each publication boundary explicit"
)]
fn forward_with_publication(
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

fn assert_diagnostic(case: &Case, diagnostic: &LayerAttentionDiagnostic) {
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

#[test]
fn native_attention_matches_captured_prefill_and_decode_boundaries() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let mut state = LayerAttentionState::new(layout(&fixture.model));
    for (call_id, case) in fixture.cases.iter().enumerate() {
        let diagnostic = forward_case(
            &mut state,
            case,
            0,
            u64::try_from(call_id).expect("bounded calls"),
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed(),
        )
        .expect("source-shaped attention execution");
        assert_diagnostic(case, &diagnostic);
        assert_eq!(
            case.window_indices.dtype, "torch.int32",
            "source window index storage"
        );
        assert_eq!(
            case.wo_b_input.dtype, "torch.bfloat16",
            "source WO-B input storage"
        );
    }
}

#[test]
fn stale_publications_wrong_source_and_discontinuous_position_are_atomic() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let mut state = LayerAttentionState::new(layout(&fixture.model));
    forward_case(
        &mut state,
        &fixture.cases[0],
        0,
        0,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("first source prefill");
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[1],
            0,
            1,
            SOURCE_LAYER + 1,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::WrongSourceLayer { .. })
    ));
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[1],
            1,
            1,
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::WrongEpoch { .. })
    ));
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[2],
            0,
            2,
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::DiscontinuousPosition { .. })
    ));
    let diagnostic = forward_case(
        &mut state,
        &fixture.cases[1],
        0,
        1,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("failed calls do not advance state");
    assert_diagnostic(&fixture.cases[1], &diagnostic);
    state.reset().expect("bounded epoch increment");
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[0],
            0,
            0,
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::WrongEpoch { .. })
    ));
    let diagnostic = forward_case(
        &mut state,
        &fixture.cases[0],
        1,
        0,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("stale publication failure does not consume reset prefill");
    assert_diagnostic(&fixture.cases[0], &diagnostic);
}

#[test]
fn wrong_call_and_missing_weight_fail_without_consuming_the_prefill() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let mut state = LayerAttentionState::new(layout(&fixture.model));
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[0],
            0,
            1,
            SOURCE_LAYER,
            &frequencies,
            weights.borrowed()
        ),
        Err(LayerAttentionError::WrongCallId { .. })
    ));
    let mut missing = weights.borrowed();
    missing.wq_a.codes = &[];
    assert!(matches!(
        forward_case(
            &mut state,
            &fixture.cases[0],
            0,
            0,
            SOURCE_LAYER,
            &frequencies,
            missing
        ),
        Err(LayerAttentionError::Fp8Linear(_))
    ));
    let diagnostic = forward_case(
        &mut state,
        &fixture.cases[0],
        0,
        0,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("failed prefill remains atomic");
    assert_diagnostic(&fixture.cases[0], &diagnostic);
}

#[test]
fn compressed_prefix_length_is_exact_and_failures_are_atomic() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let case = &fixture.cases[1];
    let input = case.input.bf16();
    let compressed = case.compressed_kv.bf16();
    let legal_indices = vec![i32::try_from(fixture.model.window_size).expect("small window")];
    let short = &compressed[..5 * fixture.model.head_dim];
    let mut after_prefill = prefetched_state(&fixture, &frequencies, weights.borrowed());
    assert!(matches!(
        forward_with_publication(
            &mut after_prefill,
            &input,
            case.start_pos,
            0,
            1,
            SOURCE_LAYER,
            short,
            &legal_indices,
            call_frequencies(&frequencies, case),
            weights.borrowed(),
        ),
        Err(LayerAttentionError::CompressedKeyCount {
            actual: 5,
            expected: 6,
        })
    ));
    let diagnostic = forward_case(
        &mut after_prefill,
        case,
        0,
        1,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("short prefix failure does not consume decode");
    assert_diagnostic(case, &diagnostic);

    let mut too_long = compressed;
    too_long.extend(std::iter::repeat_n(0, fixture.model.head_dim));
    let mut after_prefill = prefetched_state(&fixture, &frequencies, weights.borrowed());
    assert!(matches!(
        forward_with_publication(
            &mut after_prefill,
            &input,
            case.start_pos,
            0,
            1,
            SOURCE_LAYER,
            &too_long,
            &legal_indices,
            call_frequencies(&frequencies, case),
            weights.borrowed(),
        ),
        Err(LayerAttentionError::CompressedKeyCount {
            actual: 7,
            expected: 6,
        })
    ));
    let diagnostic = forward_case(
        &mut after_prefill,
        case,
        0,
        1,
        SOURCE_LAYER,
        &frequencies,
        weights.borrowed(),
    )
    .expect("long prefix failure does not consume decode");
    assert_diagnostic(case, &diagnostic);
}

fn prefetched_state(
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

#[test]
fn zero_compressed_prefix_is_a_window_only_attention_call() {
    let fixture = fixture();
    let frequencies = frequencies(&fixture);
    let weights = weights(&fixture.encoded_parameters);
    let layout = LayerAttentionLayout::new(
        nonzero(1),
        nonzero(fixture.model.dim),
        nonzero(fixture.model.n_heads),
        nonzero(fixture.model.head_dim),
        nonzero(fixture.model.rope_head_dim / 2),
        nonzero(fixture.model.q_lora_rank),
        nonzero(fixture.model.window_size),
        nonzero(fixture.model.o_groups),
        nonzero(fixture.model.o_lora_rank),
        SOURCE_LAYER,
        nonzero(8),
        fixture.model.norm_eps,
        0.125,
    )
    .expect("window-only ratio layout");
    let mut state = LayerAttentionState::new(layout);
    let case = &fixture.cases[0];
    let call_frequencies = call_frequencies(&frequencies, case);
    let empty_slots = vec![-1; case.input.shape[1]];
    assert!(matches!(
        forward_with_publication(
            &mut state,
            &case.input.bf16(),
            case.start_pos,
            0,
            0,
            SOURCE_LAYER,
            &[],
            &empty_slots,
            call_frequencies,
            weights.borrowed(),
        ),
        Err(LayerAttentionError::CompressedSlotsWithoutKeys { slots: 1 })
    ));
    forward_with_publication(
        &mut state,
        &case.input.bf16(),
        case.start_pos,
        0,
        0,
        SOURCE_LAYER,
        &[],
        &[],
        call_frequencies,
        weights.borrowed(),
    )
    .expect("all-negative empty publication failure does not consume window-only prefill");
}
