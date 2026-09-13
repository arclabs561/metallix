//! Typed layer-three candidate-producer capture oracle.
//!
//! The fixture provides residual input and already-published index keys.
//! It qualifies those arithmetic and masking boundaries, not cache ownership or
//! a complete candidate producer.

#![allow(
    dead_code,
    reason = "the attention and standalone candidate integration binaries use different narrow slices of this oracle"
)]

use std::{collections::BTreeMap, num::NonZeroUsize};

use deepseek::{
    RotaryFrequency,
    attention::layer::Fp8Projection,
    csa2::{CandidateError, candidate_mask},
    indexer::{
        bf16::index_scores_bf16_reference,
        cache::IndexKeyPublicationId,
        query::{
            CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights,
            prepare_candidate_query, prepare_index_query,
        },
        selection::{CandidateSelection, SelectionCall, SelectionGeometry, produce_candidates},
    },
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const CAPTURE_SHA256: &str = "7c5cc8541da338fa3426d63e32b9a66e9132e07ab68ee26d86fbf9e29f62f48d";
const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";

#[derive(Deserialize)]
struct Fixture {
    schema_version: u8,
    source: Source,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    frequencies: Tensor,
    cases: Vec<Case>,
    scope: String,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    complete_capture_sha256: String,
    storage_byteorder: String,
}

#[derive(Deserialize)]
struct Model {
    batches: usize,
    candidate_block_size: usize,
    candidate_topk_blocks: usize,
    expected_offsets: Vec<usize>,
    expected_start_positions: Vec<usize>,
    index_head_dimension: usize,
    index_heads: usize,
    input_dimension: usize,
    query_rank: usize,
    rope_pairs: usize,
    norm_epsilon: f32,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    attention_input: Tensor,
    wq_a_output: Tensor,
    q_norm_output: Tensor,
    candidate_mask: Tensor,
    inputs: Inputs,
    operations: Operations,
    output_indices: Tensor,
}

#[derive(Deserialize)]
struct Inputs {
    offset: usize,
    qr: Tensor,
    shared_index_k_prefix: Tensor,
    start_pos: usize,
    x: Tensor,
}

#[derive(Deserialize)]
struct Operations {
    k_after_rope_fp4: Tensor,
    q_after_rope_fp4: Tensor,
    weights_proj_output: Tensor,
    scaled_weights: Tensor,
    scores_einsum: Tensor,
    scores_after_relu: Tensor,
    scores_weighted_per_head: Tensor,
    scores_after_head_sum: Tensor,
    scores_after_causal_mask: Option<Tensor>,
}

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    finite: bool,
    numel: usize,
    shape: Vec<usize>,
    storage_hex: String,
    storage_sha256: String,
}

impl Tensor {
    fn bytes(&self) -> Vec<u8> {
        let elements = self
            .shape
            .iter()
            .copied()
            .try_fold(1_usize, usize::checked_mul)
            .expect("source tensor shape product fits usize");
        assert_eq!(self.numel, elements, "source tensor numel");
        assert!(
            self.storage_hex.len().is_multiple_of(2),
            "source hex alignment"
        );
        let bytes: Vec<_> = self
            .storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("fixture UTF-8"), 16)
                    .expect("fixture hex")
            })
            .collect();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(self.storage_sha256, digest, "source tensor storage hash");
        bytes
    }

    fn bf16(&self) -> Vec<u16> {
        assert_eq!(self.dtype, "torch.bfloat16");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.numel * 2, "BF16 source byte length");
        bytes
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
            .collect()
    }

    fn fp8(&self) -> Vec<u8> {
        assert!(matches!(
            self.dtype.as_str(),
            "torch.float8_e4m3fn" | "torch.float8_e8m0fnu"
        ));
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.numel, "FP8 source byte length");
        bytes
    }

    fn bools(&self) -> Vec<bool> {
        assert_eq!(self.dtype, "torch.bool");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.numel, "bool source byte length");
        bytes
            .into_iter()
            .map(|value| match value {
                0 => false,
                1 => true,
                _ => panic!("source bool storage must contain only 0 or 1"),
            })
            .collect()
    }
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-candidate-reference.json"
    ))
    .expect("valid candidate fixture JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source.revision, REVISION);
    assert_eq!(fixture.source.complete_capture_sha256, CAPTURE_SHA256);
    assert_eq!(fixture.source.storage_byteorder, "little");
    assert_eq!(fixture.model.batches, 1);
    assert_eq!(fixture.model.expected_start_positions, [0, 5, 6]);
    assert_eq!(fixture.model.expected_offsets, [5, 6, 6]);
    assert_eq!(fixture.model.candidate_block_size, 1);
    assert_eq!(fixture.model.candidate_topk_blocks, 2);
    assert_eq!(fixture.frequencies.dtype, "torch.complex64");
    assert_eq!(fixture.frequencies.shape, [8, 16]);
    assert_eq!(fixture.frequencies.numel, 128);
    assert!(fixture.frequencies.finite, "source frequencies finite");
    assert!(fixture.scope.contains("not native arithmetic"));
    assert_eq!(fixture.cases.len(), 3);
    for (case, (&start, &offset)) in fixture.cases.iter().zip(
        fixture
            .model
            .expected_start_positions
            .iter()
            .zip(&fixture.model.expected_offsets),
    ) {
        assert_eq!(case.start_pos, start, "case start");
        assert_eq!(case.inputs.start_pos, start, "index input start");
        assert_eq!(case.inputs.offset, offset, "index offset");
        assert_eq!(
            case.attention_input.bf16(),
            case.inputs.x.bf16(),
            "source X continuity"
        );
        assert_eq!(
            case.q_norm_output.bf16(),
            case.inputs.qr.bf16(),
            "source QR continuity"
        );
        let prefix = case.inputs.shared_index_k_prefix.bf16();
        let appended = case.operations.k_after_rope_fp4.bf16();
        assert!(
            prefix.ends_with(&appended),
            "source FP4 key append continuity"
        );
        assert!(case.attention_input.finite, "source attention input finite");
        assert!(case.q_norm_output.finite, "source QR finite");
        assert!(case.candidate_mask.finite, "source candidate bits finite");
        assert_eq!(case.candidate_mask.dtype, "torch.bool");
        assert_eq!(case.output_indices.dtype, "torch.int32");
    }
    fixture
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("captured dimension is nonzero")
}

fn f32_from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn call_frequencies(fixture: &Fixture, start: usize, positions: usize) -> Vec<RotaryFrequency> {
    let pairs = fixture.model.rope_pairs;
    let end = start.checked_add(positions).expect("bounded frequency end");
    assert!(end <= fixture.frequencies.shape[0], "frequency call span");
    let bytes = fixture.frequencies.bytes();
    assert_eq!(
        bytes.len(),
        fixture.frequencies.numel * 8,
        "complex64 bytes"
    );
    bytes[start * pairs * 8..end * pairs * 8]
        .chunks_exact(8)
        .map(|pair| {
            RotaryFrequency::new(
                f32::from_le_bytes(pair[..4].try_into().expect("complex real")),
                f32::from_le_bytes(pair[4..].try_into().expect("complex imaginary")),
            )
            .expect("finite source frequency")
        })
        .collect()
}

fn index_layout(model: &Model) -> IndexQueryLayout {
    IndexQueryLayout::new(
        nonzero(model.batches),
        nonzero(model.input_dimension),
        nonzero(model.query_rank),
        nonzero(model.index_heads),
        nonzero(model.index_head_dimension),
        nonzero(model.rope_pairs),
    )
    .expect("captured index-query layout")
}

fn source_case(fixture: &Fixture, start: usize) -> &Case {
    fixture
        .cases
        .iter()
        .find(|case| case.start_pos == start)
        .unwrap_or_else(|| panic!("no captured candidate case starts at {start}"))
}

/// Maps a captured call to the test owner's epoch-zero publication sequence.
pub(super) fn source_call(start: usize) -> SelectionCall {
    let fixture = fixture();
    let (call_id, case) = fixture
        .cases
        .iter()
        .enumerate()
        .find(|(_, case)| case.start_pos == start)
        .unwrap_or_else(|| panic!("no captured candidate call starts at {start}"));
    let geometry = SelectionGeometry::new(
        start,
        nonzero(case.inputs.x.shape[1]),
        nonzero(case.inputs.shared_index_k_prefix.shape[1]),
        nonzero(1),
        case.inputs.offset,
    )
    .expect("captured selection geometry");
    SelectionCall::new(
        IndexKeyPublicationId::new(3, 0, u64::try_from(call_id).expect("three captured calls")),
        0,
        geometry,
    )
}

/// Returns the source-captured, already FP4-reconstructed index key prefix.
///
/// The public test seam lets a downstream attention integration prove it is
/// using the same native-produced prefix before asking [`generated_candidates`]
/// to consume it.
pub(super) fn captured_keys(start: usize) -> Vec<u16> {
    let fixture = fixture();
    source_case(&fixture, start)
        .inputs
        .shared_index_k_prefix
        .bf16()
}

/// Generates one source-captured candidate mask from supplied native index keys.
///
/// X remains a fixture-fed source boundary. `native_keys` must exactly
/// equal the independently captured layer-three FP4 key prefix at `start`.
#[expect(
    clippy::too_many_lines,
    reason = "keep source-stage parity assertions in execution order"
)]
#[allow(
    clippy::similar_names,
    reason = "retain source wq_a and wq_b projection names"
)]
pub(super) fn generated_candidates(
    start: usize,
    native_keys: &[u16],
    call: SelectionCall,
) -> CandidateSelection {
    let fixture = fixture();
    let case = source_case(&fixture, start);
    assert_eq!(
        call,
        source_call(start),
        "source selection call at start {start}"
    );
    let expected_keys = case.inputs.shared_index_k_prefix.bf16();
    assert_eq!(
        native_keys, expected_keys,
        "native key prefix at start {start}"
    );
    let parameters = &fixture.encoded_parameters;
    let wq_b_codes = parameters["layers.3.attn.indexer.wq_b.weight"].fp8();
    let wq_b_scales = parameters["layers.3.attn.indexer.wq_b.scale"].fp8();
    let weights_proj = parameters["layers.3.attn.indexer.weights_proj.weight"].bf16();
    let wq_a_codes = parameters["layers.3.attn.wq_a.weight"].fp8();
    let wq_a_scales = parameters["layers.3.attn.wq_a.scale"].fp8();
    let q_norm = parameters["layers.3.attn.q_norm.weight"].bf16();
    let positions = case.inputs.x.shape[1];
    let prepared = prepare_candidate_query(
        &case.inputs.x.bf16(),
        &call_frequencies(&fixture, start, positions),
        CandidateQueryWeights {
            wq_a: Fp8Projection {
                codes: &wq_a_codes,
                scales: &wq_a_scales,
            },
            q_norm: &q_norm,
            index: IndexQueryWeights {
                wq_b_codes: &wq_b_codes,
                wq_b_scales: &wq_b_scales,
                weights_proj: &weights_proj,
            },
        },
        CandidateQueryLayout::new(index_layout(&fixture.model), fixture.model.norm_epsilon)
            .expect("bounded candidate QR layout"),
    )
    .expect("bounded candidate index query");
    assert_eq!(prepared.wq_a, case.wq_a_output.bf16(), "start {start} wq_a");
    assert_eq!(prepared.qr, case.q_norm_output.bf16(), "start {start} QR");
    let query = prepared.index;
    assert_eq!(
        query.query_post_fp4,
        case.operations.q_after_rope_fp4.bf16(),
        "start {start} index Q"
    );
    assert_eq!(
        query.projected_head_weights,
        case.operations.weights_proj_output.bf16(),
        "start {start} head weights"
    );
    assert_eq!(
        query.scaled_head_weights,
        case.operations.scaled_weights.bf16(),
        "start {start} scaled weights"
    );

    let keys = native_keys;
    let key_count = case.inputs.shared_index_k_prefix.shape[1];
    let heads = fixture.model.index_heads;
    let dimension = fixture.model.index_head_dimension;
    let mut dots = Vec::new();
    let mut rectified = Vec::new();
    let mut weighted = Vec::new();
    let mut scores = Vec::new();
    for position in 0..positions {
        let query_start = position * heads * dimension;
        let weight_start = position * heads;
        let diagnostic = index_scores_bf16_reference(
            &query.query_post_fp4[query_start..query_start + heads * dimension],
            keys,
            &query.scaled_head_weights[weight_start..weight_start + heads],
            nonzero(dimension),
        )
        .expect("bounded candidate BF16 scorer");
        dots.extend(diagnostic.dot_products);
        rectified.extend(diagnostic.rectified);
        weighted.extend(diagnostic.weighted);
        scores.extend(diagnostic.scores);
    }
    assert_eq!(
        dots,
        case.operations.scores_einsum.bf16(),
        "start {start} dot scores"
    );
    assert_eq!(
        rectified,
        case.operations.scores_after_relu.bf16(),
        "start {start} relu scores"
    );
    assert_eq!(
        weighted,
        case.operations.scores_weighted_per_head.bf16(),
        "start {start} weighted scores"
    );
    assert_eq!(
        scores,
        case.operations.scores_after_head_sum.bf16(),
        "start {start} head sum scores"
    );
    assert_eq!(scores.len(), positions * key_count, "score geometry");
    let candidates = produce_candidates(
        &scores,
        call,
        fixture.model.candidate_topk_blocks,
        nonzero(fixture.model.candidate_block_size),
    )
    .expect("captured candidate rows");
    assert_eq!(
        candidates.call(),
        call,
        "start {start} candidate call identity"
    );
    let expected_causal = case
        .operations
        .scores_after_causal_mask
        .as_ref()
        .map_or_else(
            || case.operations.scores_after_head_sum.bf16(),
            Tensor::bf16,
        );
    assert_eq!(
        candidates.causal_scores(),
        expected_causal,
        "start {start} causal scores"
    );
    assert_eq!(
        candidates.mask(),
        case.candidate_mask.bools(),
        "start {start} candidate mask"
    );
    candidates
}

pub(super) fn rejects_unmasked_future_candidate() {
    let fixture = fixture();
    let case = source_case(&fixture, 0);
    let scores = case.operations.scores_after_head_sum.bf16();
    assert!(matches!(
        candidate_mask(
            &scores[..case.inputs.shared_index_k_prefix.shape[1]]
                .iter()
                .map(|&bits| f32_from_bf16(bits))
                .collect::<Vec<_>>(),
            1,
            fixture.model.candidate_topk_blocks,
            nonzero(fixture.model.candidate_block_size),
        ),
        Err(CandidateError::UnmaskedFuturePosition { position: 1 })
    ));
}

pub(super) fn query_weight_perturbation_is_observable() {
    let fixture = fixture();
    let case = source_case(&fixture, 0);
    let parameters = &fixture.encoded_parameters;
    let wq_b_codes = parameters["layers.3.attn.indexer.wq_b.weight"].fp8();
    let wq_b_scales = parameters["layers.3.attn.indexer.wq_b.scale"].fp8();
    let mut weights_proj = parameters["layers.3.attn.indexer.weights_proj.weight"].bf16();
    let expected = prepare_index_query(
        &case.inputs.qr.bf16(),
        &case.inputs.x.bf16(),
        &call_frequencies(&fixture, 0, case.inputs.qr.shape[1]),
        IndexQueryWeights {
            wq_b_codes: &wq_b_codes,
            wq_b_scales: &wq_b_scales,
            weights_proj: &weights_proj,
        },
        index_layout(&fixture.model),
    )
    .expect("source query");
    assert_eq!(
        expected.projected_head_weights,
        case.operations.weights_proj_output.bf16(),
        "unmodified source weights"
    );
    weights_proj[0] = 0;
    let perturbed = prepare_index_query(
        &case.inputs.qr.bf16(),
        &case.inputs.x.bf16(),
        &call_frequencies(&fixture, 0, case.inputs.qr.shape[1]),
        IndexQueryWeights {
            wq_b_codes: &wq_b_codes,
            wq_b_scales: &wq_b_scales,
            weights_proj: &weights_proj,
        },
        index_layout(&fixture.model),
    )
    .expect("perturbed query remains shaped");
    assert_ne!(
        perturbed.projected_head_weights, expected.projected_head_weights,
        "changing a captured query weight is observable before candidate masking"
    );
}
