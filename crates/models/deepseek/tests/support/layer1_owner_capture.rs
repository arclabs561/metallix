//! Test-only native replay of the captured layer-one ratio-two owner publication.

use std::{collections::BTreeMap, num::NonZeroUsize};

use deepseek::{
    RotaryFrequency,
    attention::layer::Fp8Projection,
    compressor::{CompressorInput, CompressorState},
    indexer::{
        bf16::index_scores_bf16_reference,
        cache::{IndexKeyPublicationId, IndexKeyState},
        compressed_kv::{CompressedKvLayout, prepare_compressed_kv},
        key::{IndexKeyLayout, IndexKeyWeights, prepare_index_keys},
        query::{
            CandidateQueryLayout, CandidateQueryWeights, IndexQueryDiagnostic, IndexQueryLayout,
            IndexQueryWeights, prepare_candidate_query, prepare_index_query,
        },
    },
    precision::fp32_linear_reference,
    select_indices,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const CAPTURE_SHA256: &str = "16c949df47afd5cffc5f8ce95612f2d27003dcb23a80c764f7a8d02bef79be16";
const FIXTURE_SHA256: &str = "966122b3fd74fe3f5965d17aad1f863bb67164c69fd66bc52cc7df836db9407a";
const FP32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

#[derive(Clone, Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    model: Model,
    frequency_table: Tensor,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
    #[serde(default)]
    query_parameters: BTreeMap<String, Tensor>,
}

#[derive(Clone, Deserialize)]
struct Source {
    revision: String,
    complete_capture_sha256: Option<String>,
}

#[derive(Clone, Deserialize)]
struct Model {
    owner_layer: usize,
    ratio: usize,
    norm_eps: f32,
    index_topk: usize,
}

#[derive(Clone, Deserialize)]
struct Case {
    start_pos: usize,
    sequence: usize,
    compressed_prefix: usize,
    group_frequency_positions: Vec<usize>,
    input: Tensor,
    index_input: Tensor,
    index_qr: Tensor,
    wq_a_output: Option<Tensor>,
    index_operations: IndexOperations,
    offset: usize,
    selected_indices: Tensor,
    wkv_projection: Tensor,
    wgate_projection: Tensor,
    latent: Option<Tensor>,
    index_key_prefix: Tensor,
    index_score_key_prefix: Tensor,
    compressed_kv_prefix: Tensor,
}

#[derive(Clone, Deserialize)]
struct IndexOperations {
    q_after_rope_fp4: Tensor,
    weights_proj_output: Tensor,
    scaled_weights: Tensor,
    scores_einsum: Tensor,
    scores_after_relu: Tensor,
    scores_weighted_per_head: Tensor,
    scores_after_head_sum: Tensor,
    scores_after_causal_mask: Option<Tensor>,
}

#[derive(Clone, Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    numel: usize,
    storage_hex: String,
    storage_sha256: String,
}

/// One native owner publication and its source-qualified selection after a call.
#[derive(Clone)]
pub(super) struct NativeCase {
    pub start_pos: usize,
    pub latent: Option<Vec<u16>>,
    pub key_prefix: Vec<u16>,
    pub kv_prefix: Vec<u16>,
    pub source_score_key_prefix: Vec<u16>,
    pub selected_indices: Vec<i32>,
}

fn fixture() -> Fixture {
    let raw =
        include_str!("../../../../../fixtures/deepseek-v41/layer1-ratio2-owner-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    fixture_from_raw(raw, CAPTURE_SHA256)
}

fn fixture_from_raw(raw: &str, expected_capture: &str) -> Fixture {
    let fixture: Fixture = serde_json::from_str(raw).expect("layer-one owner fixture JSON");
    validate_fixture(&fixture, expected_capture);
    fixture
}

fn validate_fixture(fixture: &Fixture, expected_capture: &str) {
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source.revision, REVISION);
    assert_eq!(
        fixture.source.complete_capture_sha256.as_deref(),
        Some(expected_capture)
    );
    assert_eq!(fixture.model.owner_layer, 1);
    assert_eq!(fixture.model.ratio, 2);
    assert_eq!(fixture.model.index_topk, 1);
    assert_eq!(fixture.cases.len(), 3);
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("captured geometry is nonzero")
}

fn bytes(tensor: &Tensor, bytes_per_element: usize) -> Vec<u8> {
    let expected = tensor.shape.iter().copied().product::<usize>();
    assert_eq!(tensor.numel, expected);
    let bytes: Vec<_> = tensor
        .storage_hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("hex"), 16).expect("hex byte")
        })
        .collect();
    assert_eq!(bytes.len(), expected * bytes_per_element);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        tensor.storage_sha256
    );
    bytes
}

fn bf16(tensor: &Tensor) -> Vec<u16> {
    assert_eq!(tensor.dtype, "torch.bfloat16");
    bytes(tensor, 2)
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
        .collect()
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn fp32(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype, "torch.float32");
    bytes(tensor, 4)
        .chunks_exact(4)
        .map(|word| f32::from_bits(u32::from_le_bytes(word.try_into().expect("FP32 word"))))
        .collect()
}

fn fp8(tensor: &Tensor) -> Vec<u8> {
    assert!(matches!(
        tensor.dtype.as_str(),
        "torch.float8_e4m3fn" | "torch.float8_e8m0fnu"
    ));
    bytes(tensor, 1)
}

fn i32s(tensor: &Tensor) -> Vec<i32> {
    assert_eq!(tensor.dtype, "torch.int32");
    bytes(tensor, 4)
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().expect("i32 word")))
        .collect()
}

/// Return the bounded layer-three score operand consumed by the partial
/// layer-one call. It must come from the same complete source capture as the
/// owner trace: a separately captured candidate fixture can share a model
/// revision while still have different observer provenance and inputs.
fn prior_layer_three_prefix(fixture: &Fixture) -> Vec<u16> {
    let partial = fixture
        .cases
        .iter()
        .find(|case| case.start_pos == 6)
        .expect("captured partial layer-one decode");
    let owner = bf16(&partial.index_key_prefix);
    let source = bf16(&partial.index_score_key_prefix);
    assert_eq!(source.len(), 3 * 64, "partial source score-key geometry");
    assert_ne!(
        source, owner,
        "partial source score key must remain distinct from owner publication"
    );
    source
}

fn frequencies(tensor: &Tensor) -> Vec<RotaryFrequency> {
    assert_eq!(tensor.dtype, "torch.complex64");
    assert_eq!(tensor.shape, [8, 16]);
    bytes(tensor, 8)
        .chunks_exact(8)
        .map(|pair| {
            RotaryFrequency::new(
                f32::from_bits(u32::from_le_bytes(pair[..4].try_into().expect("real"))),
                f32::from_bits(u32::from_le_bytes(pair[4..].try_into().expect("imag"))),
            )
            .expect("finite source frequency")
        })
        .collect()
}

fn parameter<'a>(fixture: &'a Fixture, name: &str) -> &'a Tensor {
    fixture
        .encoded_parameters
        .get(name)
        .unwrap_or_else(|| panic!("missing layer-one owner parameter {name}"))
}

/// Bounds two valid FP32 dot reductions against their shared exact dot.
///
/// Each stream is bounded by a multiply/add reduction with `2n` roundings;
/// their difference is bounded by `gamma(4n) * Σ|xᵢwᵢ|`.  This comes only
/// from operands, term count, and IEEE FP32 unit roundoff, never an observed
/// source/native delta.
fn assert_fp32_projection_envelope(
    name: &str,
    activations: &[f32],
    weights: &[f32],
    rows: usize,
    outputs: usize,
    native: &[f32],
    source: &[f32],
) -> Result<(), String> {
    if rows == 0 || outputs != 64 {
        return Err("layer-one projection envelope has unexpected row or output geometry".into());
    }
    let terms = activations.len() / rows;
    if terms != 128 {
        return Err("layer-one projection envelope has unexpected reduction geometry".into());
    }
    assert_eq!(activations.len(), rows * terms);
    assert_eq!(weights.len(), outputs * terms);
    assert_eq!(native.len(), rows * outputs);
    assert_eq!(source.len(), rows * outputs);
    let operations = u32::try_from(terms.checked_mul(4).expect("bounded dot operation count"))
        .map_err(|_| "layer-one projection envelope operation count exceeds u32".to_owned())?;
    let roundoff = f64::from(operations) * FP32_UNIT_ROUNDOFF;
    if !roundoff.is_finite() || roundoff >= 1.0 {
        return Err("layer-one projection envelope has invalid gamma denominator".into());
    }
    let gamma = roundoff / (1.0 - roundoff);
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err("layer-one projection envelope has nonfinite gamma".into());
    }
    for (field, values) in [
        ("activation", activations),
        ("weight", weights),
        ("native", native),
        ("source", source),
    ] {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(format!(
                "{name} projection has nonfinite {field} at {index}"
            ));
        }
    }
    let mut maximum_error = 0.0_f64;
    let mut maximum_index = 0_usize;
    for row in 0..rows {
        for output in 0..outputs {
            let index = row * outputs + output;
            let magnitude = activations[row * terms..(row + 1) * terms]
                .iter()
                .zip(&weights[output * terms..(output + 1) * terms])
                .map(|(&left, &right)| f64::from(left).abs() * f64::from(right).abs())
                .sum::<f64>();
            let bound = gamma * magnitude;
            if !magnitude.is_finite() || !bound.is_finite() {
                return Err(format!(
                    "{name} projection has nonfinite magnitude or bound at {index}"
                ));
            }
            let error = (f64::from(native[index]) - f64::from(source[index])).abs();
            if error > maximum_error {
                maximum_error = error;
                maximum_index = index;
            }
            if error > bound {
                return Err(format!(
                    "{name} projection envelope rejected element {index}: error {error:e}, bound {bound:e}, maximum error {maximum_error:e} at {maximum_index}"
                ));
            }
        }
    }
    Ok(())
}

fn source_projections(
    case: &Case,
    input: &[u16],
    wkv: &[f32],
    wgate: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(case.input.shape, [1, case.sequence, 128]);
    assert_eq!(
        input,
        bf16(&case.input),
        "start {} owner input BF16 boundary",
        case.start_pos
    );
    let input_f32: Vec<_> = input.iter().copied().map(bf16_to_f32).collect();
    let mut projected = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input_f32, wkv, case.sequence, 128, 64, &mut projected)
        .expect("bounded native WKV projection");
    assert_fp32_projection_envelope(
        "WKV",
        &input_f32,
        wkv,
        case.sequence,
        64,
        &projected,
        &fp32(&case.wkv_projection),
    )
    .unwrap_or_else(|error| panic!("start {} {error}", case.start_pos));
    let mut gate = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input_f32, wgate, case.sequence, 128, 64, &mut gate)
        .expect("bounded native gate projection");
    assert_fp32_projection_envelope(
        "gate",
        &input_f32,
        wgate,
        case.sequence,
        64,
        &gate,
        &fp32(&case.wgate_projection),
    )
    .unwrap_or_else(|error| panic!("start {} {error}", case.start_pos));
    (projected, gate)
}

fn call_frequencies(all: &[RotaryFrequency], case: &Case) -> Vec<RotaryFrequency> {
    all[case.start_pos * 16..(case.start_pos + case.sequence) * 16].to_vec()
}

fn score_positions(
    query: &[u16],
    keys: &[u16],
    weights: &[u16],
    positions: usize,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    assert_eq!(query.len(), positions * 2 * 64);
    assert_eq!(weights.len(), positions * 2);
    let mut dots = Vec::new();
    let mut relu = Vec::new();
    let mut weighted = Vec::new();
    let mut scores = Vec::new();
    for position in 0..positions {
        let diagnostic = index_scores_bf16_reference(
            &query[position * 128..(position + 1) * 128],
            keys,
            &weights[position * 2..(position + 1) * 2],
            nonzero(64),
        )
        .expect("bounded ratio-two BF16 index score");
        dots.extend(diagnostic.dot_products);
        relu.extend(diagnostic.rectified);
        weighted.extend(diagnostic.weighted);
        scores.extend(diagnostic.scores);
    }
    (dots, relu, weighted, scores)
}

fn causal_scores(scores: &[u16], case: &Case, keys: usize, ratio: usize) -> Vec<u16> {
    let mut causal = scores.to_vec();
    for position in 0..case.sequence {
        let reachable = (case.start_pos + position + 1) / ratio;
        for key in reachable..keys {
            causal[position * keys + key] = 0xff80;
        }
    }
    causal
}

fn selected_indices(
    causal: &[u16],
    case: &Case,
    keys: usize,
    ratio: usize,
    topk: usize,
) -> Vec<i32> {
    (0..case.sequence)
        .flat_map(|position| {
            select_indices(
                &causal[position * keys..(position + 1) * keys]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
                    .collect::<Vec<_>>(),
                (case.start_pos + position + 1) / ratio,
                topk,
                case.offset,
            )
            .expect("source ratio-two selection cutoff")
        })
        .collect()
}

fn assert_native_score(
    fixture: &Fixture,
    case: &Case,
    all_frequencies: &[RotaryFrequency],
    keys: &[u16],
) -> Vec<i32> {
    let layout = IndexQueryLayout::new(
        nonzero(1),
        nonzero(128),
        nonzero(32),
        nonzero(2),
        nonzero(64),
        nonzero(16),
    )
    .expect("ratio-two query layout");
    let query = prepare_index_query(
        &bf16(&case.index_qr),
        &bf16(&case.index_input),
        &call_frequencies(all_frequencies, case),
        IndexQueryWeights {
            wq_b_codes: &fp8(parameter(fixture, "layers.1.attn.indexer.wq_b.weight")),
            wq_b_scales: &fp8(parameter(fixture, "layers.1.attn.indexer.wq_b.scale")),
            weights_proj: &bf16(parameter(
                fixture,
                "layers.1.attn.indexer.weights_proj.weight",
            )),
        },
        layout,
    )
    .expect("native ratio-two index query");
    assert_score_from_query(fixture, case, keys, &query)
}

fn assert_score_from_query(
    fixture: &Fixture,
    case: &Case,
    keys: &[u16],
    query: &IndexQueryDiagnostic,
) -> Vec<i32> {
    let operations = &case.index_operations;
    assert_eq!(query.query_post_fp4, bf16(&operations.q_after_rope_fp4));
    assert_eq!(
        query.projected_head_weights,
        bf16(&operations.weights_proj_output)
    );
    assert_eq!(query.scaled_head_weights, bf16(&operations.scaled_weights));
    let key_count = keys.len() / 64;
    let (dots, relu, weighted, scores) = score_positions(
        &query.query_post_fp4,
        keys,
        &query.scaled_head_weights,
        case.sequence,
    );
    assert_eq!(dots, bf16(&operations.scores_einsum));
    assert_eq!(relu, bf16(&operations.scores_after_relu));
    assert_eq!(weighted, bf16(&operations.scores_weighted_per_head));
    assert_eq!(scores, bf16(&operations.scores_after_head_sum));
    let causal = causal_scores(&scores, case, key_count, fixture.model.ratio);
    if let Some(expected) = &operations.scores_after_causal_mask {
        assert_eq!(causal, bf16(expected));
    } else {
        assert_eq!(
            causal, scores,
            "start {} has no future score",
            case.start_pos
        );
    }
    let indices = selected_indices(
        &causal,
        case,
        key_count,
        fixture.model.ratio,
        fixture.model.index_topk,
    );
    assert_eq!(indices, i32s(&case.selected_indices));
    indices
}

fn alternate_native_selection(
    fixture: &Fixture,
    case: &Case,
    frequencies: &[RotaryFrequency],
    input: &[u16],
    keys: &[u16],
) -> Vec<i32> {
    let parameters = &fixture.query_parameters;
    let codes = fp8(&parameters["layers.1.attn.wq_a.weight"]);
    let scales = fp8(&parameters["layers.1.attn.wq_a.scale"]);
    let norm = bf16(&parameters["layers.1.attn.q_norm.weight"]);
    let index_codes = fp8(parameter(fixture, "layers.1.attn.indexer.wq_b.weight"));
    let index_scales = fp8(parameter(fixture, "layers.1.attn.indexer.wq_b.scale"));
    let weights = bf16(parameter(
        fixture,
        "layers.1.attn.indexer.weights_proj.weight",
    ));
    assert_eq!(input, bf16(&case.index_input));
    let query = prepare_candidate_query(
        input,
        &call_frequencies(frequencies, case),
        CandidateQueryWeights {
            wq_a: Fp8Projection {
                codes: &codes,
                scales: &scales,
            },
            q_norm: &norm,
            index: IndexQueryWeights {
                wq_b_codes: &index_codes,
                wq_b_scales: &index_scales,
                weights_proj: &weights,
            },
        },
        CandidateQueryLayout::new(
            IndexQueryLayout::new(
                nonzero(1),
                nonzero(128),
                nonzero(32),
                nonzero(2),
                nonzero(64),
                nonzero(16),
            )
            .unwrap(),
            fixture.model.norm_eps,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        query.wq_a,
        bf16(case.wq_a_output.as_ref().expect("alternate WQ-A oracle"))
    );
    assert_eq!(query.qr, bf16(&case.index_qr));
    assert_score_from_query(fixture, case, keys, &query.index)
}

struct LayerThreeSharedScoreState {
    layer_three_keys: IndexKeyState,
}

impl LayerThreeSharedScoreState {
    fn new() -> Self {
        Self {
            // The fixture exports exactly the three source-layer-three keys
            // consumed by the partial layer-one call. Do not retain a wider
            // candidate trace whose provenance belongs to another capture.
            layer_three_keys: IndexKeyState::new(nonzero(1), nonzero(64), nonzero(3), 3)
                .expect("bounded layer-three shared score state"),
        }
    }

    fn reset(&mut self) {
        self.layer_three_keys
            .reset()
            .expect("layer-three shared score state reset");
    }

    fn publish_layer_three(&mut self, prefix: &[u16]) {
        self.reset();
        let epoch = self.layer_three_keys.epoch();
        let call_id = self.layer_three_keys.next_call_id();
        let start = self.layer_three_keys.valid_positions();
        let publication = IndexKeyPublicationId::new(3, epoch, call_id);
        self.layer_three_keys
            .append_prepared(publication, start, prefix)
            .expect("source-grounded layer-three shared-key publication");
    }

    fn score_keys_for(&self, case: &Case, owned_keys: &[u16], source_keys: &[u16]) -> Vec<u16> {
        if case.latent.is_none() {
            assert_ne!(
                owned_keys, source_keys,
                "partial owner prefix stays distinct"
            );
            let previous = self.layer_three_keys.prefix(0).expect("L3 shared keys");
            assert_eq!(previous, source_keys, "previous L3 score prefix");
            previous.to_vec()
        } else {
            assert_eq!(owned_keys, source_keys, "owned score prefix");
            owned_keys.to_vec()
        }
    }
}

/// A partial decode may consume score keys published by an earlier layer call,
/// but that publication belongs to one request and must not survive reset.
pub(super) fn request_local_score_state_rejects_cross_request_reuse() -> bool {
    let mut state = LayerThreeSharedScoreState::new();
    let fixture = fixture();
    let published = prior_layer_three_prefix(&fixture);
    state.publish_layer_three(&published);
    assert_eq!(
        state.layer_three_keys.prefix(0).expect("published prefix"),
        published.as_slice()
    );
    state.reset();
    state
        .layer_three_keys
        .prefix(0)
        .expect("reset prefix")
        .is_empty()
}

/// The bounded layer-three publication is an operand from this source trace,
/// not an interchangeable same-revision candidate capture.
pub(super) fn partial_score_prefix_provenance_gate_rejects_mismatch() -> bool {
    let fixture = fixture();
    let partial = fixture
        .cases
        .iter()
        .find(|case| case.start_pos == 6)
        .expect("captured partial layer-one decode");
    let owner = bf16(&partial.index_key_prefix);
    let expected = prior_layer_three_prefix(&fixture);
    let mut substituted = expected.clone();
    substituted[0] ^= 1;
    let mut state = LayerThreeSharedScoreState::new();
    state.publish_layer_three(&expected);
    std::panic::catch_unwind(|| state.score_keys_for(partial, &owner, &substituted)).is_err()
}

/// Score the captured partial layer-one call with a complete, natively produced
/// layer-three prefix. The source consumes only its leading three keys; keeping
/// that projection explicit prevents a six-key producer cache from being
/// silently substituted for the smaller score domain.
#[allow(
    dead_code,
    reason = "used by the separate layer-three bridge integration test"
)]
pub(super) fn partial_score_with_native_layer_three_prefix(
    owner_fixture_json: &str,
    expected_capture: &str,
    producer_prefix: &[u16],
) -> Vec<i32> {
    let fixture = fixture_from_raw(owner_fixture_json, expected_capture);
    let partial = fixture
        .cases
        .iter()
        .find(|case| case.start_pos == 6)
        .expect("captured partial layer-one decode");
    assert_eq!(
        producer_prefix.len(),
        6 * 64,
        "complete layer-three prefix geometry"
    );
    let source_score_keys = bf16(&partial.index_score_key_prefix);
    let consumed = &producer_prefix[..source_score_keys.len()];
    let owner_keys = bf16(&partial.index_key_prefix);
    let mut state = LayerThreeSharedScoreState::new();
    state.publish_layer_three(consumed);
    let score_keys = state.score_keys_for(partial, &owner_keys, &source_score_keys);
    assert_native_score(
        &fixture,
        partial,
        &frequencies(&fixture.frequency_table),
        &score_keys,
    )
}

/// Replays source-owned KV/key publication through direct index selection.
///
/// The final partial call deliberately exposes its captured score-key operand
/// separately: source global score state was subsequently owned by layer three,
/// whereas this owner retains its own key and KV prefixes.
pub(super) fn native_publications() -> Vec<NativeCase> {
    let fixture = fixture();
    native_publications_with_previous_layer_three_prefix_for_calls(None, fixture.cases.len())
}

/// Replays an ordered nonempty prefix of the source owner calls. This permits
/// the preceding calls to publish their state without reading the following
/// partial decode's operands.
pub(super) fn native_publications_with_previous_layer_three_prefix_for_calls(
    previous_layer_three_prefix: Option<&[u16]>,
    call_count: usize,
) -> Vec<NativeCase> {
    let fixture = fixture();
    assert!(
        (1..=fixture.cases.len()).contains(&call_count),
        "native layer-one owner call prefix"
    );
    let mut session = NativeLayerOneOwnerSession::new(previous_layer_three_prefix);
    (0..call_count).map(|_| session.step()).collect()
}

/// Test-private request session for the layer-one ratio-two owner.  It keeps
/// the compressor, owner key/KV prefixes, and the source-qualified prior-L3
/// score operand live across source calls.
pub(super) struct NativeLayerOneOwnerSession {
    fixture: Fixture,
    all_frequencies: Vec<RotaryFrequency>,
    wkv: Vec<f32>,
    wgate: Vec<f32>,
    wk: Vec<u16>,
    key_norm: Vec<u16>,
    compressor: CompressorState,
    key_layout: IndexKeyLayout,
    kv_layout: CompressedKvLayout,
    keys: Vec<u16>,
    kv: Vec<u16>,
    score_state: LayerThreeSharedScoreState,
    previous_layer_three_prefix: Option<Vec<u16>>,
    require_previous_layer_three_prefix: bool,
    next_case: usize,
    next_start: usize,
}

impl NativeLayerOneOwnerSession {
    pub(super) fn new(previous_layer_three_prefix: Option<&[u16]>) -> Self {
        Self::from_fixture(fixture(), previous_layer_three_prefix, false)
    }

    /// Decodes the layer-one owner operands from the unified reduced bundle.
    /// The request must later supply the preceding live L3 key publication
    /// before processing its start-six partial decode.
    #[allow(
        dead_code,
        reason = "used by the composed forward test, not the standalone owner test binary"
    )]
    pub(super) fn from_bundle(bundle: &Value) -> Self {
        let pinned: Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
        ))
        .expect("pinned reduced bundle metadata");
        assert_eq!(bundle["schema_version"].as_u64(), Some(1));
        assert_eq!(bundle["source"], pinned["source"], "bundle source metadata");
        let projection = bundle["projections"]["layer1_owner"].clone();
        assert_eq!(
            projection["source"], pinned["projections"]["layer1_owner"]["source"],
            "layer-one owner source metadata"
        );
        let capture = bundle["source"]["complete_capture_sha256"]
            .as_str()
            .expect("bundle capture identity");
        let fixture: Fixture =
            serde_json::from_value(projection).expect("layer-one owner bundle JSON");
        validate_fixture(&fixture, capture);
        Self::from_fixture(fixture, None, true)
    }

    #[allow(
        dead_code,
        reason = "alternate owner is exercised by the composed forward test"
    )]
    pub(super) fn from_alternate(projection: &Value, startup: &Value) -> Self {
        for name in ["source", "source_receipt_sha256", "capture_identity"] {
            assert_eq!(projection[name], startup[name], "alternate L1 provenance");
        }
        let fixture: Fixture =
            serde_json::from_value(projection.clone()).expect("alternate L1 owner");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.source.revision, REVISION);
        assert!(fixture.source.complete_capture_sha256.is_none());
        assert_eq!(
            (
                fixture.model.owner_layer,
                fixture.model.ratio,
                fixture.model.index_topk
            ),
            (1, 2, 1)
        );
        assert_eq!(fixture.query_parameters.len(), 3);
        assert_eq!(fixture.cases.len(), 4);
        for (case, (start, count, partial)) in
            fixture
                .cases
                .iter()
                .zip([(0, 4, false), (4, 1, true), (5, 1, false), (6, 1, true)])
        {
            assert_eq!(
                (case.start_pos, case.sequence, case.latent.is_none()),
                (start, count, partial)
            );
        }
        Self::from_fixture(fixture, None, true)
    }

    fn from_fixture(
        fixture: Fixture,
        previous_layer_three_prefix: Option<&[u16]>,
        require_previous_layer_three_prefix: bool,
    ) -> Self {
        if let Some(prefix) = previous_layer_three_prefix {
            assert_eq!(prefix.len(), 6 * 64, "complete layer-three score prefix");
        }
        let compressor_norm = bf16(parameter(&fixture, "layers.1.attn.compressor.norm.weight"));
        let norm_eps = fixture.model.norm_eps;
        Self {
            all_frequencies: frequencies(&fixture.frequency_table),
            wkv: fp32(parameter(&fixture, "layers.1.attn.compressor.wkv.weight")),
            wgate: fp32(parameter(&fixture, "layers.1.attn.compressor.wgate.weight")),
            wk: bf16(parameter(&fixture, "layers.1.attn.indexer.wk.weight")),
            key_norm: bf16(parameter(&fixture, "layers.1.attn.indexer.k_norm.weight")),
            compressor: CompressorState::new(1, 64, 2, &compressor_norm, norm_eps)
                .expect("source ratio-two compressor"),
            key_layout: IndexKeyLayout::new(
                nonzero(1),
                nonzero(64),
                nonzero(64),
                nonzero(16),
                norm_eps,
            )
            .expect("source index-key layout"),
            kv_layout: CompressedKvLayout::new(nonzero(1), nonzero(64), nonzero(16))
                .expect("source compressed-KV layout"),
            fixture,
            keys: Vec::new(),
            kv: Vec::new(),
            score_state: LayerThreeSharedScoreState::new(),
            previous_layer_three_prefix: previous_layer_three_prefix.map(ToOwned::to_owned),
            require_previous_layer_three_prefix,
            next_case: 0,
            next_start: 0,
        }
    }

    /// Advances exactly one captured owner call.  Prefix and compressor
    /// mutation is supplied by their native bounded operations; this fixture
    /// composition does not claim a cross-owner rollback transaction.
    /// Compatibility step for the standalone owner fixture. Composed callers
    /// use [`Self::step_with_input`] to carry their derived L1 attention input.
    pub(super) fn step(&mut self) -> NativeCase {
        let input = bf16(&self.fixture.cases[self.next_case].input);
        self.step_with_input(&input)
    }

    pub(super) fn step_with_input(&mut self, input: &[u16]) -> NativeCase {
        let case = self
            .fixture
            .cases
            .get(self.next_case)
            .expect("native layer-one owner calls exhausted");
        assert_eq!(
            case.start_pos, self.next_start,
            "native layer-one owner call order"
        );
        if case.start_pos == 0 {
            self.score_state.reset();
        }
        if case.latent.is_none()
            && self.require_previous_layer_three_prefix
            && self.previous_layer_three_prefix.is_none()
        {
            panic!("bundle layer-one owner requires a live previous-layer-three prefix");
        }
        let (projected, gate) = source_projections(case, input, &self.wkv, &self.wgate);
        let latent = self
            .compressor
            .forward(
                CompressorInput::Gated {
                    kv: &projected,
                    scores: &gate,
                },
                case.sequence,
                case.start_pos,
            )
            .expect("source ratio-two stream call");
        assert_latent_matches_source(case, latent.as_deref());
        if let Some(latent) = &latent {
            let (prepared_keys, prepared_kv) = self.prepare_latent_publication(case, latent);
            self.keys.extend_from_slice(&prepared_keys);
            self.kv.extend_from_slice(&prepared_kv);
        }
        assert_eq!(
            self.keys,
            bf16(&case.index_key_prefix),
            "start {} owned key prefix",
            case.start_pos
        );
        assert_eq!(
            self.kv,
            bf16(&case.compressed_kv_prefix),
            "start {} owned KV prefix",
            case.start_pos
        );
        assert_eq!(self.keys.len(), case.compressed_prefix * 64);
        let source_score_keys = bf16(&case.index_score_key_prefix);
        if case.latent.is_none() {
            publish_previous_layer_three_prefix(
                &mut self.score_state,
                &self.fixture,
                &source_score_keys,
                case.start_pos,
                self.previous_layer_three_prefix.as_deref(),
                self.require_previous_layer_three_prefix,
            );
        }
        let score_keys = self
            .score_state
            .score_keys_for(case, &self.keys, &source_score_keys);
        let selected_indices = if self.fixture.query_parameters.is_empty() {
            assert_native_score(&self.fixture, case, &self.all_frequencies, &score_keys)
        } else {
            alternate_native_selection(
                &self.fixture,
                case,
                &self.all_frequencies,
                input,
                &score_keys,
            )
        };
        if case.latent.is_none() {
            self.previous_layer_three_prefix = None;
        }
        self.next_start += case.sequence;
        self.next_case += 1;
        NativeCase {
            start_pos: case.start_pos,
            latent,
            key_prefix: self.keys.clone(),
            kv_prefix: self.kv.clone(),
            source_score_key_prefix: source_score_keys,
            selected_indices,
        }
    }

    fn prepare_latent_publication(&self, case: &Case, latent: &[u16]) -> (Vec<u16>, Vec<u16>) {
        assert_eq!(case.group_frequency_positions.len(), latent.len() / 64);
        let mut selected = Vec::new();
        for &position in &case.group_frequency_positions {
            selected.extend_from_slice(&self.all_frequencies[position * 16..(position + 1) * 16]);
        }
        let prepared_keys = prepare_index_keys(
            latent,
            &selected,
            IndexKeyWeights::new(&self.wk, &self.key_norm),
            self.key_layout,
        )
        .expect("native source-shaped index keys");
        let prepared_kv = prepare_compressed_kv(latent, &selected, self.kv_layout)
            .expect("native source-shaped compressed KV");
        (prepared_keys.post_fp4, prepared_kv.post_fp4)
    }

    /// Supplies the complete producer prefix after the preceding L3 call has
    /// committed and before L1's next partial decode reads it.
    pub(super) fn supply_previous_layer_three_prefix(&mut self, prefix: &[u16]) {
        let case = &self.fixture.cases[self.next_case];
        assert!(
            case.latent.is_none(),
            "prior L3 prefix is only used by partial groups"
        );
        assert_eq!(
            prefix.len(),
            case.start_pos * 64,
            "complete preceding L3 prefix"
        );
        self.previous_layer_three_prefix = Some(prefix.to_vec());
    }

    /// Reconstructs a fresh fixture request. It intentionally clears the
    /// previous request's supplied L3 publication; this is not an epoch-level
    /// reset contract for the production owners.
    pub(super) fn restart_request(&mut self) {
        *self = Self::from_fixture(
            self.fixture.clone(),
            None,
            self.require_previous_layer_three_prefix,
        );
    }
}

/// Resetting a request session removes old owner prefixes before the next
/// start-zero publication is admitted.
pub(super) fn request_session_reset_clears_owner_prefixes() -> bool {
    let mut session = NativeLayerOneOwnerSession::new(None);
    // The owner fixture records only the three source-visible L3 keys.  The
    // unused suffix supplies the required complete-prefix geometry for this
    // restart control; it is deliberately not presented as producer evidence.
    let mut source_prefix_with_unused_test_suffix = prior_layer_three_prefix(&fixture());
    source_prefix_with_unused_test_suffix.resize(6 * 64, 0);
    session.step();
    session.step();
    session.supply_previous_layer_three_prefix(&source_prefix_with_unused_test_suffix);
    let partial = session.step();
    session.restart_request();
    let cleared = session.previous_layer_three_prefix.is_none()
        && session.keys.is_empty()
        && session.kv.is_empty()
        && session.next_case == 0
        && session
            .score_state
            .layer_three_keys
            .prefix(0)
            .expect("fresh score prefix")
            .is_empty();
    let reset_first = session.step();
    partial.source_score_key_prefix == source_prefix_with_unused_test_suffix[..3 * 64]
        && cleared
        && reset_first.start_pos == 0
        && reset_first.key_prefix == bf16(&fixture().cases[0].index_key_prefix)
        && reset_first.kv_prefix == bf16(&fixture().cases[0].compressed_kv_prefix)
        && reset_first.selected_indices == i32s(&fixture().cases[0].selected_indices)
}

fn assert_latent_matches_source(case: &Case, latent: Option<&[u16]>) {
    match (latent, &case.latent) {
        (Some(actual), Some(expected)) => assert_eq!(
            actual,
            &bf16(expected),
            "start {} compressor latent",
            case.start_pos
        ),
        (None, None) => {}
        _ => panic!("start {} latent publication disagrees", case.start_pos),
    }
}

/// Publishes the source-visible prior layer-three score keys for the following
/// partial call. Native producers publish six keys after start five; the L1
/// scorer consumes its source-qualified leading three-key window.
fn publish_previous_layer_three_prefix(
    state: &mut LayerThreeSharedScoreState,
    fixture: &Fixture,
    source_score_keys: &[u16],
    producer_end: usize,
    previous_layer_three_prefix: Option<&[u16]>,
    require_previous_layer_three_prefix: bool,
) {
    let prefix = if let Some(prefix) = previous_layer_three_prefix {
        assert_eq!(
            prefix.len(),
            producer_end * 64,
            "complete preceding L3 score prefix"
        );
        prefix[..source_score_keys.len()].to_vec()
    } else if require_previous_layer_three_prefix {
        panic!("bundle layer-one owner requires a live previous-layer-three prefix");
    } else {
        let captured = prior_layer_three_prefix(fixture);
        assert_eq!(captured.len(), source_score_keys.len());
        captured
    };
    state.publish_layer_three(&prefix);
}

/// Replacing the source's partial-decode score keys with the native layer-one
/// owner prefix must trip an observed score-stage gate.
pub(super) fn partial_owner_keys_fail_score_gate() -> bool {
    let outputs = native_publications();
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.start_pos == 6)
        .expect("captured partial decode");
    let frequencies = frequencies(&fixture.frequency_table);
    let partial = outputs
        .iter()
        .find(|output| output.start_pos == 6)
        .expect("native partial owner publication");
    assert_ne!(partial.key_prefix, partial.source_score_key_prefix);
    std::panic::catch_unwind(|| {
        let _ = assert_native_score(&fixture, case, &frequencies, &partial.key_prefix);
    })
    .is_err()
}

/// The first gate projection is numerically load-bearing to ratio-two pooling.
pub(super) fn first_gate_mutation_changes_latent() -> bool {
    let fixture = fixture();
    let case = &fixture.cases[0];
    let input: Vec<_> = bf16(&case.input).into_iter().map(bf16_to_f32).collect();
    let wkv = fp32(parameter(&fixture, "layers.1.attn.compressor.wkv.weight"));
    let mut wgate = fp32(parameter(&fixture, "layers.1.attn.compressor.wgate.weight"));
    wgate[0] = -wgate[0];
    let mut projected = vec![0.0; case.sequence * 64];
    let mut gate = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input, &wkv, case.sequence, 128, 64, &mut projected)
        .expect("native WKV projection");
    fp32_linear_reference(&input, &wgate, case.sequence, 128, 64, &mut gate)
        .expect("mutated native gate projection");
    let norm = bf16(parameter(&fixture, "layers.1.attn.compressor.norm.weight"));
    let mut compressor = CompressorState::new(1, 64, 2, &norm, fixture.model.norm_eps)
        .expect("source ratio-two compressor");
    let mutated = compressor
        .forward(
            CompressorInput::Gated {
                kv: &projected,
                scores: &gate,
            },
            5,
            0,
        )
        .expect("mutated compressor call")
        .expect("prefill completes two groups");
    mutated != bf16(case.latent.as_ref().expect("prefill latent"))
}

/// A finite source-projection mutation outside the declared FP32 envelope fails.
pub(super) fn projection_envelope_rejects_corruption() -> bool {
    let fixture = fixture();
    let case = &fixture.cases[0];
    let input: Vec<_> = bf16(&case.input).into_iter().map(bf16_to_f32).collect();
    let weights = fp32(parameter(&fixture, "layers.1.attn.compressor.wkv.weight"));
    let mut native = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input, &weights, case.sequence, 128, 64, &mut native)
        .expect("native WKV projection");
    let mut corrupted = native.clone();
    corrupted[0] = native[0] + native[0].abs().max(1.0) * 0.01;
    assert_fp32_projection_envelope(
        "corrupted WKV",
        &input,
        &weights,
        case.sequence,
        64,
        &native,
        &corrupted,
    )
    .is_err()
}

/// Nonfinite values are rejected before a floating-point comparison can mask them.
pub(super) fn projection_envelope_rejects_nonfinite() -> bool {
    let fixture = fixture();
    let case = &fixture.cases[0];
    let input: Vec<_> = bf16(&case.input).into_iter().map(bf16_to_f32).collect();
    let weights = fp32(parameter(&fixture, "layers.1.attn.compressor.wkv.weight"));
    let mut native = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input, &weights, case.sequence, 128, 64, &mut native)
        .expect("native WKV projection");
    let mut source = native.clone();
    source[0] = f32::NAN;
    let source_rejected = assert_fp32_projection_envelope(
        "nonfinite WKV",
        &input,
        &weights,
        case.sequence,
        64,
        &native,
        &source,
    )
    .is_err();
    source[0] = native[0];
    native[0] = f32::INFINITY;
    let native_rejected = assert_fp32_projection_envelope(
        "nonfinite WKV",
        &input,
        &weights,
        case.sequence,
        64,
        &native,
        &source,
    )
    .is_err();
    source_rejected && native_rejected
}
