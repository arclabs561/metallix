//! Native ratio-one owner over the alternate 4/1/1/1 source partition.
//!
//! The source fixture supplies inputs and numerical oracles.
//! `RatioOneCompressedOwner` publishes native keys/KV after native selection;
//! `LayerAttentionState` then consumes the committed KV and computed IDs.
//! Owner and attention have separate commit boundaries.

#[path = "support/attention_capture.rs"]
pub(crate) mod attention_capture;

use std::{collections::BTreeMap, num::NonZeroUsize};

use deepseek::{
    RotaryFrequency,
    attention::layer::{Fp8Projection, LayerAttentionState},
    indexer::{
        cache::IndexKeyPublicationId,
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{
            PendingRatioOneCompressedOwner, RatioOneCompressedOwner, RatioOneOwnerCall,
            RatioOneOwnerWeights,
        },
        query::{
            CandidateQueryLayout, CandidateQueryWeights, IndexKeyView, IndexQueryLayout,
            IndexQueryWeights, ScoredQueryDiagnostic, prepare_scored_query,
        },
        selection::{
            CandidateSelection, SelectionCall, SelectionGeometry, produce_candidates,
            select_from_candidates,
        },
    },
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const SOURCE_RECEIPT_SHA256: &str =
    "9613150fea8010a7435dab0443a1f9e0d73fd8d0f32455b8d67dd572617f3906";
const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const MODEL_SHA256: &str = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65";
const SCHEDULE: &[(usize, usize)] = &[(0, 4), (4, 1), (5, 1), (6, 1)];

#[derive(Deserialize)]
struct Fixture {
    schema_version: u8,
    source_receipt_sha256: String,
    source: Value,
    capture_identity: Value,
    model: Value,
    weights: Weights,
    frequencies: Tensor,
    selection_model: SelectionModel,
    selection_weights: SelectionWeights,
    attention_model: attention_capture::Model,
    attention_weights: BTreeMap<String, attention_capture::Tensor>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Weights {
    wkv: Tensor,
    compressor_norm: Tensor,
    wk: Tensor,
    key_norm: Tensor,
}

#[derive(Deserialize)]
struct SelectionModel {
    query_rank: usize,
    index_heads: usize,
    candidate_block_size: usize,
    candidate_topk_blocks: usize,
    index_topk: usize,
}

#[derive(Deserialize)]
struct SelectionWeights {
    wq_a_codes: Tensor,
    wq_a_scales: Tensor,
    q_norm: Tensor,
    wq_b_codes: Tensor,
    wq_b_scales: Tensor,
    weights_proj: Tensor,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    token_count: usize,
    input: Tensor,
    projected: Tensor,
    latent: Tensor,
    index_key_prefix: Tensor,
    compressed_kv_prefix: Tensor,
    next_layer1_score_prefix: Option<Tensor>,
    selection: Selection,
    attention: attention_capture::Case,
}

#[derive(Deserialize)]
struct Selection {
    offset: usize,
    wq_a: Tensor,
    qr: Tensor,
    q_after_rope_fp4: Tensor,
    weights_proj_output: Tensor,
    scaled_weights: Tensor,
    dot_products: Tensor,
    rectified: Tensor,
    weighted: Tensor,
    scores: Tensor,
    causal_scores: Option<Tensor>,
    candidate_mask: Tensor,
    indices: Tensor,
}

/// A committed producer boundary retained for the alternate L4 consumer.
/// The consumer scores this key prefix with its own query and selects its own
/// IDs from the retained producer candidates.
pub(crate) struct AlternateLayerThreePublication {
    pub(crate) start_pos: usize,
    pub(crate) publication: IndexKeyPublicationId,
    pub(crate) input: Vec<u16>,
    pub(crate) key_prefix: Vec<u16>,
    pub(crate) kv_prefix: Vec<u16>,
    pub(crate) producer_candidates: CandidateSelection,
    pub(crate) attention_output: Vec<u16>,
}

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    finite: bool,
    shape: Vec<usize>,
    numel: usize,
    storage_hex: String,
    storage_sha256: String,
}

impl Tensor {
    fn raw_bytes(&self, width: usize) -> Vec<u8> {
        let elements = self.shape.iter().copied().product::<usize>();
        assert_eq!(self.numel, elements, "source tensor element count");
        assert_eq!(self.storage_hex.len(), elements * width * 2, "source bytes");
        let bytes: Vec<_> = self
            .storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("fixture UTF-8"), 16)
                    .expect("fixture hex")
            })
            .collect();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), self.storage_sha256);
        bytes
    }

    fn bytes(&self, width: usize) -> Vec<u8> {
        assert!(self.finite, "source tensor must be finite");
        self.raw_bytes(width)
    }

    fn bf16(&self) -> Vec<u16> {
        assert_eq!(self.dtype, "torch.bfloat16");
        self.bytes(2)
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
            .collect()
    }

    fn fp8(&self) -> Vec<u8> {
        assert!(matches!(
            self.dtype.as_str(),
            "torch.float8_e4m3fn" | "torch.float8_e8m0fnu"
        ));
        self.bytes(1)
    }

    fn bools(&self) -> Vec<bool> {
        assert_eq!(self.dtype, "torch.bool");
        self.bytes(1)
            .into_iter()
            .map(|value| match value {
                0 => false,
                1 => true,
                _ => panic!("source bool storage must contain only 0 or 1"),
            })
            .collect()
    }

    fn i32(&self) -> Vec<i32> {
        assert_eq!(self.dtype, "torch.int32");
        self.bytes(4)
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().expect("i32 word")))
            .collect()
    }

    fn causal_bf16(&self) -> Vec<u16> {
        assert!(!self.finite, "causal scores record masking infinities");
        assert_eq!(self.dtype, "torch.bfloat16");
        let values: Vec<_> = self
            .raw_bytes(2)
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
            .collect();
        assert!(
            values.iter().all(|&bits| {
                let value = f32::from_bits(u32::from(bits) << 16);
                value.is_finite() || bits == 0xff80
            }),
            "causal source scores permit only negative infinity"
        );
        assert!(
            values.contains(&0xff80),
            "causal source scores include masking"
        );
        values
    }

    fn frequencies(&self) -> Vec<RotaryFrequency> {
        assert_eq!(self.dtype, "torch.complex64");
        self.bytes(8)
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
}

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("captured nonzero dimension")
}

fn object<'a>(value: &'a Value, label: &str) -> &'a serde_json::Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{label} must be an object"))
}

fn usize_field(value: &Value, field: &str) -> usize {
    usize::try_from(
        object(value, "source model")[field]
            .as_u64()
            .unwrap_or_else(|| panic!("source model {field}")),
    )
    .expect("source model usize")
}

fn fixture() -> Fixture {
    let raw = include_str!("../../../../fixtures/deepseek-v41/partition-owner-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "3e9c27c53e1bb2240912bdb3ee68747b17287d12e9d40fb6cd9d02088f277f68"
    );
    let fixture: Fixture = serde_json::from_str(raw).expect("partition owner fixture JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source_receipt_sha256, SOURCE_RECEIPT_SHA256);
    let source = object(&fixture.source, "source");
    assert_eq!(source["revision"].as_str(), Some(REVISION));
    assert_eq!(source["model_sha256"].as_str(), Some(MODEL_SHA256));
    assert_eq!(
        object(&fixture.capture_identity, "capture identity")["schedule"].as_array(),
        Some(&vec![
            Value::from(4),
            Value::from(1),
            Value::from(1),
            Value::from(1)
        ]),
        "source capture identifies the alternate partition"
    );
    assert_eq!(usize_field(&fixture.model, "batches"), 1);
    assert_eq!(usize_field(&fixture.model, "input_dimension"), 128);
    assert_eq!(usize_field(&fixture.model, "latent_dimension"), 64);
    assert_eq!(usize_field(&fixture.model, "key_dimension"), 64);
    assert_eq!(usize_field(&fixture.model, "rope_pairs"), 16);
    assert_eq!(usize_field(&fixture.model, "cache_capacity"), 8);
    assert_eq!(usize_field(&fixture.model, "owner_layer"), 3);
    let epsilon = object(&fixture.model, "source model")["norm_epsilon"]
        .as_f64()
        .expect("source epsilon");
    assert_eq!(epsilon.to_bits(), 1.0e-20_f64.to_bits());
    assert_eq!(fixture.weights.wkv.shape, [64, 128]);
    assert_eq!(fixture.weights.compressor_norm.shape, [64]);
    assert_eq!(fixture.weights.wk.shape, [64, 64]);
    assert_eq!(fixture.weights.key_norm.shape, [64]);
    assert_eq!(fixture.frequencies.shape, [8, 16]);
    assert_eq!(fixture.selection_model.query_rank, 32);
    assert_eq!(fixture.selection_model.index_heads, 2);
    assert_eq!(fixture.selection_model.candidate_block_size, 1);
    assert_eq!(fixture.selection_model.candidate_topk_blocks, 2);
    assert_eq!(fixture.selection_model.index_topk, 1);
    assert_eq!(fixture.selection_weights.wq_a_codes.shape, [32, 128]);
    assert_eq!(fixture.selection_weights.wq_a_scales.shape, [1, 4]);
    assert_eq!(fixture.selection_weights.q_norm.shape, [32]);
    assert_eq!(fixture.selection_weights.wq_b_codes.shape, [128, 32]);
    assert_eq!(fixture.selection_weights.wq_b_scales.shape, [4, 1]);
    assert_eq!(fixture.selection_weights.weights_proj.shape, [2, 128]);
    assert_attention_weights(&fixture);
    assert_eq!(fixture.cases.len(), SCHEDULE.len());
    for (index, (case, &(start, count))) in fixture.cases.iter().zip(SCHEDULE).enumerate() {
        assert_eq!((case.start_pos, case.token_count), (start, count));
        assert_eq!(case.input.shape, [1, count, 128]);
        assert_eq!(case.projected.shape, [1, count, 64]);
        assert_eq!(case.latent.shape, [1, count, 64]);
        assert_eq!(case.index_key_prefix.shape, [1, start + count, 64]);
        assert_eq!(case.compressed_kv_prefix.shape, [1, start + count, 64]);
        assert_eq!(case.selection.offset, [4, 6, 6, 6][index]);
        assert_eq!(case.selection.wq_a.shape, [1, count, 32]);
        assert_eq!(case.selection.qr.shape, [1, count, 32]);
        assert_eq!(case.selection.q_after_rope_fp4.shape, [1, count, 2, 64]);
        assert_eq!(case.selection.weights_proj_output.shape, [1, count, 2]);
        assert_eq!(case.selection.scaled_weights.shape, [1, count, 2]);
        assert_eq!(
            case.selection.dot_products.shape,
            [1, count, 2, start + count]
        );
        assert_eq!(case.selection.rectified.shape, [1, count, 2, start + count]);
        assert_eq!(case.selection.weighted.shape, [1, count, 2, start + count]);
        assert_eq!(case.selection.scores.shape, [1, count, start + count]);
        assert_eq!(
            case.selection.candidate_mask.shape,
            [1, count, start + count]
        );
        assert_eq!(case.selection.indices.shape, [1, count, 1]);
        assert_eq!(case.selection.causal_scores.is_some(), start == 0);
        assert_eq!(case.attention.start_pos, start, "attention source start");
        assert_eq!(
            case.next_layer1_score_prefix.is_some(),
            matches!(start, 0 | 5),
            "both partial handoffs are mandatory"
        );
        if let Some(next) = &case.next_layer1_score_prefix {
            let end = start.checked_add(count).expect("source token endpoint");
            assert_eq!(next.shape, [1, end / 2, 64]);
        }
    }
    fixture
}

fn assert_attention_weights(fixture: &Fixture) {
    for suffix in [
        "wq_a.weight",
        "wq_a.scale",
        "q_norm.weight",
        "wq_b.weight",
        "wq_b.scale",
        "wkv.weight",
        "wkv.scale",
        "kv_norm.weight",
        "attn_sink",
        "wo_a.weight",
        "wo_b.weight",
        "wo_b.scale",
    ] {
        assert!(
            fixture
                .attention_weights
                .contains_key(&format!("layers.3.attn.{suffix}")),
            "captured layer-three attention {suffix}"
        );
    }
}

fn owner(fixture: &Fixture) -> RatioOneCompressedOwner {
    let layout =
        IndexKeyLayout::new(nz(1), nz(64), nz(64), nz(16), 1.0e-20).expect("captured owner layout");
    RatioOneCompressedOwner::new(
        layout,
        nz(128),
        nz(8),
        3,
        &fixture.weights.compressor_norm.bf16(),
        1.0e-20,
    )
    .expect("captured ratio-one owner")
}

fn assert_query_stages(prepared: &ScoredQueryDiagnostic, selection: &Selection) {
    assert_eq!(prepared.query.wq_a, selection.wq_a.bf16(), "native wq_a");
    assert_eq!(prepared.query.qr, selection.qr.bf16(), "native QR");
    assert_eq!(
        prepared.query.index.query_post_fp4,
        selection.q_after_rope_fp4.bf16(),
        "native index Q"
    );
    assert_eq!(
        prepared.query.index.projected_head_weights,
        selection.weights_proj_output.bf16(),
        "native head weights"
    );
    assert_eq!(
        prepared.query.index.scaled_head_weights,
        selection.scaled_weights.bf16(),
        "native scaled weights"
    );
    assert_eq!(
        prepared.dot_products,
        selection.dot_products.bf16(),
        "native dot products"
    );
    assert_eq!(
        prepared.rectified,
        selection.rectified.bf16(),
        "native rectified scores"
    );
    assert_eq!(
        prepared.weighted,
        selection.weighted.bf16(),
        "native weighted scores"
    );
    assert_eq!(
        prepared.scores,
        selection.scores.bf16(),
        "native reduced scores"
    );
}

fn assert_pending_selection(
    pending: &PendingRatioOneCompressedOwner<'_>,
    fixture: &Fixture,
    index: usize,
    input: &[u16],
) -> (Vec<i32>, CandidateSelection) {
    let case = &fixture.cases[index];
    let selection = &case.selection;
    let keys = pending.key_prefix(0).expect("staged native key prefix");
    let positions = case.token_count;
    let frequencies = fixture.frequencies.frequencies();
    let query_codes = fixture.selection_weights.wq_a_codes.fp8();
    let query_scales = fixture.selection_weights.wq_a_scales.fp8();
    let q_norm = fixture.selection_weights.q_norm.bf16();
    let index_codes = fixture.selection_weights.wq_b_codes.fp8();
    let index_scales = fixture.selection_weights.wq_b_scales.fp8();
    let weights_proj = fixture.selection_weights.weights_proj.bf16();
    let layout = CandidateQueryLayout::new(
        IndexQueryLayout::new(nz(1), nz(128), nz(32), nz(2), nz(64), nz(16))
            .expect("captured selection layout"),
        1.0e-20,
    )
    .expect("captured candidate query layout");
    let geometry = SelectionGeometry::new(
        case.start_pos,
        nz(positions),
        nz(keys.len() / 64),
        nz(1),
        selection.offset,
    )
    .expect("source selection geometry");
    let call = SelectionCall::new(pending.publication(), 0, geometry);
    let prepared = prepare_scored_query(
        input,
        &frequencies[case.start_pos * 16..(case.start_pos + positions) * 16],
        CandidateQueryWeights {
            wq_a: Fp8Projection {
                codes: &query_codes,
                scales: &query_scales,
            },
            q_norm: &q_norm,
            index: IndexQueryWeights {
                wq_b_codes: &index_codes,
                wq_b_scales: &index_scales,
                weights_proj: &weights_proj,
            },
        },
        layout,
        IndexKeyView::new(keys, nz(64)).expect("live pending key view"),
    )
    .expect("live pending candidate query");
    assert_query_stages(&prepared, selection);
    let candidates = produce_candidates(
        &prepared.scores,
        call,
        fixture.selection_model.candidate_topk_blocks,
        nz(fixture.selection_model.candidate_block_size),
    )
    .expect("live native candidates");
    let expected_causal = selection
        .causal_scores
        .as_ref()
        .map_or_else(|| selection.scores.bf16(), Tensor::causal_bf16);
    assert_eq!(
        candidates.causal_scores(),
        expected_causal,
        "native causal scores"
    );
    assert_eq!(
        candidates.mask(),
        selection.candidate_mask.bools(),
        "native candidate mask"
    );
    let selected = select_from_candidates(
        &prepared.scores,
        call,
        &candidates,
        fixture.selection_model.index_topk,
    )
    .expect("live native selection");
    assert_eq!(
        selected.indices,
        selection.indices.i32(),
        "native selected IDs"
    );
    let wrong_call = SelectionCall::new(
        IndexKeyPublicationId::new(
            3,
            pending.publication().epoch(),
            pending.publication().call_id() + 1,
        ),
        0,
        geometry,
    );
    assert!(
        select_from_candidates(
            &prepared.scores,
            wrong_call,
            &candidates,
            fixture.selection_model.index_topk
        )
        .is_err(),
        "candidate selection rejects a changed publication identity"
    );
    (selected.indices, candidates)
}

fn commit_case(
    owner: &mut RatioOneCompressedOwner,
    fixture: &Fixture,
    index: usize,
) -> (Vec<i32>, CandidateSelection) {
    let input = fixture.cases[index].input.bf16();
    commit_case_with_input(owner, fixture, index, &input)
}

fn commit_case_with_input(
    owner: &mut RatioOneCompressedOwner,
    fixture: &Fixture,
    index: usize,
    input: &[u16],
) -> (Vec<i32>, CandidateSelection) {
    let case = &fixture.cases[index];
    let frequencies = fixture.frequencies.frequencies();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    assert_eq!(input, case.input.bf16(), "native L3 supplied owner input");
    let start = case.start_pos;
    let end = start + case.token_count;
    let pending = owner
        .prepare(RatioOneOwnerCall::new(
            IndexKeyPublicationId::new(3, owner.epoch(), owner.next_call_id()),
            start,
            nz(case.token_count),
            input,
            &frequencies[start * 16..end * 16],
            RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
        ))
        .expect("source-shaped owner preparation");
    let (selected_indices, producer_candidates) =
        assert_pending_selection(&pending, fixture, index, input);
    let diagnostic = pending.commit().expect("source-shaped owner commit");
    assert_eq!(
        diagnostic.owner.projected,
        case.projected.bf16(),
        "native WKV"
    );
    assert_eq!(
        diagnostic.owner.latent,
        case.latent.bf16(),
        "native compressor"
    );
    assert_eq!(
        owner.key_prefix(0).expect("native index prefix"),
        case.index_key_prefix.bf16(),
        "native index keys"
    );
    assert_eq!(
        owner.kv_prefix(0).expect("native compressed-KV prefix"),
        case.compressed_kv_prefix.bf16(),
        "native compressed KV"
    );
    if let Some(next_layer_one) = &case.next_layer1_score_prefix {
        let prefix = owner.key_prefix(0).expect("published key prefix");
        assert_eq!(
            &prefix[..next_layer_one.numel],
            next_layer_one.bf16(),
            "native committed L3 keys feed the next L1 partial score prefix"
        );
    }
    (selected_indices, producer_candidates)
}

fn state(owner: &RatioOneCompressedOwner) -> (u64, u64, usize, usize, Vec<u16>, Vec<u16>) {
    (
        owner.epoch(),
        owner.next_call_id(),
        owner.next_position(),
        owner.valid_positions(),
        owner.key_prefix(0).expect("key prefix").to_vec(),
        owner.kv_prefix(0).expect("KV prefix").to_vec(),
    )
}

fn native_owner_attention_step(
    owner: &mut RatioOneCompressedOwner,
    attention: &mut LayerAttentionState,
    fixture: &Fixture,
    index: usize,
    supplied_input: &[u16],
) -> AlternateLayerThreePublication {
    let case = &fixture.cases[index];
    let attention_case = &case.attention;
    assert_eq!(
        supplied_input,
        case.input.bf16(),
        "native L3 supplied attention input"
    );
    let (selected_indices, producer_candidates) =
        commit_case_with_input(owner, fixture, index, supplied_input);
    assert_eq!(
        attention_case.input.bf16(),
        supplied_input,
        "owner input crosses the layer-three attention boundary"
    );
    assert_eq!(
        selected_indices,
        attention_case.compressed_indices.i32(),
        "live owner selection IDs drive layer-three attention"
    );
    let start = case.start_pos;
    let end = start + case.token_count;
    let call_id = owner
        .next_call_id()
        .checked_sub(1)
        .expect("completed owner call ordinal");
    let frequencies = fixture.frequencies.frequencies();
    let weights = attention_capture::weights_for_layer(&fixture.attention_weights, 3);
    let diagnostic = attention_capture::forward_with_publication(
        attention,
        supplied_input,
        start,
        owner.epoch(),
        call_id,
        3,
        owner.kv_prefix(0).expect("committed live owner KV prefix"),
        &selected_indices,
        &frequencies[start * 16..end * 16],
        weights.borrowed(),
    )
    .expect("live owner publication drives layer-three attention");
    attention_capture::assert_diagnostic(attention_case, &diagnostic);
    AlternateLayerThreePublication {
        start_pos: start,
        publication: IndexKeyPublicationId::new(3, owner.epoch(), call_id),
        input: supplied_input.to_vec(),
        key_prefix: owner
            .key_prefix(0)
            .expect("committed live owner key prefix")
            .to_vec(),
        kv_prefix: owner
            .kv_prefix(0)
            .expect("committed live owner KV prefix")
            .to_vec(),
        producer_candidates,
        attention_output: diagnostic.final_output,
    }
}

/// Persistent alternate L3 owner and attention state for a caller-derived HC
/// attention input. The fixture remains an exact boundary oracle.
pub(crate) struct NativeAlternateLayerThreeSession {
    fixture: Fixture,
    owner: RatioOneCompressedOwner,
    attention: LayerAttentionState,
    next_case: usize,
}

impl NativeAlternateLayerThreeSession {
    pub(crate) fn new() -> Self {
        let fixture = fixture();
        let owner = owner(&fixture);
        let attention =
            LayerAttentionState::new(attention_capture::layout(&fixture.attention_model));
        Self {
            fixture,
            owner,
            attention,
            next_case: 0,
        }
    }

    pub(crate) fn step(&mut self, supplied: &(usize, Vec<u16>)) -> AlternateLayerThreePublication {
        let case = &self.fixture.cases[self.next_case];
        assert_eq!(
            supplied.0, case.start_pos,
            "native L3 supplied attention start"
        );
        assert_eq!(
            supplied.1,
            case.input.bf16(),
            "native L3 supplied attention input"
        );
        let publication = native_owner_attention_step(
            &mut self.owner,
            &mut self.attention,
            &self.fixture,
            self.next_case,
            &supplied.1,
        );
        self.next_case += 1;
        publication
    }
}

fn run_native_owner_attention_partition(
    owner: &mut RatioOneCompressedOwner,
    attention: &mut LayerAttentionState,
    fixture: &Fixture,
) -> Vec<AlternateLayerThreePublication> {
    let mut outputs = Vec::with_capacity(fixture.cases.len());
    for index in 0..fixture.cases.len() {
        let input = fixture.cases[index].input.bf16();
        outputs.push(native_owner_attention_step(
            owner, attention, fixture, index, &input,
        ));
    }
    outputs
}

pub(crate) fn alternate_partition_owner_attention_outputs() -> Vec<(usize, Vec<u16>)> {
    alternate_partition_layer_three_publications()
        .into_iter()
        .map(|publication| (publication.start_pos, publication.attention_output))
        .collect()
}

pub(crate) fn alternate_partition_layer_three_publications() -> Vec<AlternateLayerThreePublication>
{
    let fixture = fixture();
    let mut owner = owner(&fixture);
    let mut attention =
        LayerAttentionState::new(attention_capture::layout(&fixture.attention_model));
    run_native_owner_attention_partition(&mut owner, &mut attention, &fixture)
}

#[test]
fn alternate_partition_owner_matches_source_prefixes_and_partial_bridges() {
    let fixture = fixture();
    let publications = alternate_partition_layer_three_publications();
    assert_eq!(publications.len(), fixture.cases.len());
    // Verify retained history after all calls: earlier prefixes and masks must
    // not become aliases of the final owner's state.
    for (index, (publication, case)) in publications.iter().zip(&fixture.cases).enumerate() {
        assert_eq!(publication.start_pos, case.start_pos);
        assert_eq!(
            publication.publication,
            IndexKeyPublicationId::new(3, 0, u64::try_from(index).unwrap())
        );
        assert_eq!(publication.input, case.input.bf16());
        assert_eq!(publication.key_prefix, case.index_key_prefix.bf16());
        assert_eq!(publication.kv_prefix, case.compressed_kv_prefix.bf16());
        let geometry = SelectionGeometry::new(
            case.start_pos,
            nz(case.token_count),
            nz(case.start_pos + case.token_count),
            nz(1),
            case.selection.offset,
        )
        .unwrap();
        assert_eq!(
            publication.producer_candidates.call(),
            SelectionCall::new(publication.publication, 0, geometry)
        );
        assert_eq!(
            publication.producer_candidates.mask(),
            case.selection.candidate_mask.bools()
        );
    }
}

#[test]
fn alternate_partition_owner_publications_drive_native_layer_three_attention() {
    let outputs = alternate_partition_owner_attention_outputs();
    assert_eq!(outputs.len(), SCHEDULE.len());
}

#[test]
fn live_layer_three_session_uses_supplied_input_before_owner_mutation() {
    let mut session = NativeAlternateLayerThreeSession::new();
    let before = state(&session.owner);
    let mut changed = session.fixture.cases[0].input.bf16();
    changed[0] ^= 1;
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            session.step(&(0, changed));
        }))
        .is_err()
    );
    assert_eq!(
        state(&session.owner),
        before,
        "rejected input leaves owner unchanged"
    );
    assert_eq!(
        session.next_case, 0,
        "rejected input leaves session ordering unchanged"
    );

    for index in 0..session.fixture.cases.len() {
        let (start, input, expected_attention) = {
            let case = &session.fixture.cases[index];
            (
                case.start_pos,
                case.input.bf16(),
                case.attention.output.bf16(),
            )
        };
        let publication = session.step(&(start, input.clone()));
        assert_eq!(publication.start_pos, start);
        assert_eq!(publication.input, input);
        assert_eq!(publication.attention_output, expected_attention);
    }
}

#[test]
fn owner_and_layer_three_attention_reset_then_replay_together() {
    let fixture = fixture();
    let mut owner = owner(&fixture);
    let mut attention =
        LayerAttentionState::new(attention_capture::layout(&fixture.attention_model));
    let _ = run_native_owner_attention_partition(&mut owner, &mut attention, &fixture);
    owner.reset().expect("reset live owner publication state");
    attention
        .reset()
        .expect("reset layer-three attention state");
    let _ = run_native_owner_attention_partition(&mut owner, &mut attention, &fixture);
}

#[test]
fn cancelled_pending_decode_keeps_owner_invisible_and_retryable() {
    let fixture = fixture();
    let mut owner = owner(&fixture);
    commit_case(&mut owner, &fixture, 0);
    let before = state(&owner);
    let case = &fixture.cases[1];
    let frequencies = fixture.frequencies.frequencies();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    let input = case.input.bf16();
    let end = case.start_pos + case.token_count;
    let pending = owner
        .prepare(RatioOneOwnerCall::new(
            IndexKeyPublicationId::new(3, owner.epoch(), owner.next_call_id()),
            case.start_pos,
            nz(case.token_count),
            &input,
            &frequencies[case.start_pos * 16..end * 16],
            RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
        ))
        .expect("staged source decode");
    assert_eq!(
        pending.diagnostic().owner.projected,
        case.projected.bf16(),
        "staged native WKV"
    );
    assert_eq!(
        pending.diagnostic().owner.latent,
        case.latent.bf16(),
        "staged native compressor"
    );
    assert_eq!(
        pending.key_prefix(0).expect("staged native keys"),
        case.index_key_prefix.bf16(),
        "staged native key prefix"
    );
    assert_eq!(
        pending.kv_prefix(0).expect("staged native KV"),
        case.compressed_kv_prefix.bf16(),
        "staged native compressed-KV prefix"
    );
    assert_pending_selection(&pending, &fixture, 1, &input);
    drop(pending);
    assert_eq!(state(&owner), before, "dropped pending decode is invisible");
    commit_case(&mut owner, &fixture, 1);
    commit_case(&mut owner, &fixture, 2);
    commit_case(&mut owner, &fixture, 3);
}

#[test]
fn rejected_calls_leave_the_owner_retryable() {
    let fixture = fixture();
    let mut owner = owner(&fixture);
    let before = state(&owner);
    let case = &fixture.cases[1];
    let frequencies = fixture.frequencies.frequencies();
    let input = case.input.bf16();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    assert!(
        owner
            .prepare(RatioOneOwnerCall::new(
                IndexKeyPublicationId::new(3, 0, 0),
                case.start_pos,
                nz(case.token_count),
                &input,
                &frequencies[case.start_pos * 16..(case.start_pos + case.token_count) * 16],
                RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
            ))
            .is_err()
    );
    assert_eq!(state(&owner), before, "out-of-order call is invisible");

    commit_case(&mut owner, &fixture, 0);
    commit_case(&mut owner, &fixture, 1);
    commit_case(&mut owner, &fixture, 2);
    let before_late = state(&owner);
    let late = &fixture.cases[3];
    let mut malformed = late.input.bf16();
    malformed.pop();
    let frequencies = fixture.frequencies.frequencies();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    assert!(
        owner
            .prepare(RatioOneOwnerCall::new(
                IndexKeyPublicationId::new(3, owner.epoch(), owner.next_call_id()),
                late.start_pos,
                nz(late.token_count),
                &malformed,
                &frequencies[late.start_pos * 16..(late.start_pos + late.token_count) * 16],
                RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
            ))
            .is_err()
    );
    assert_eq!(
        state(&owner),
        before_late,
        "malformed late call is invisible"
    );
    commit_case(&mut owner, &fixture, 3);
}

#[test]
fn reset_requires_fresh_publications_and_replays_the_source_partition() {
    let fixture = fixture();
    let mut owner = owner(&fixture);
    for index in 0..fixture.cases.len() {
        commit_case(&mut owner, &fixture, index);
    }
    let previous_epoch = owner.epoch();
    owner.reset().expect("reset coupled owner state");
    assert_eq!(owner.epoch(), previous_epoch + 1);
    assert_eq!(owner.next_call_id(), 0);
    assert_eq!(owner.next_position(), 0);
    assert_eq!(owner.valid_positions(), 0);
    assert_eq!(state(&owner).4, Vec::<u16>::new(), "reset key prefix");
    assert_eq!(state(&owner).5, Vec::<u16>::new(), "reset KV prefix");

    let first = &fixture.cases[0];
    let input = first.input.bf16();
    let frequencies = fixture.frequencies.frequencies();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    let after_reset = state(&owner);
    assert!(
        owner
            .prepare(RatioOneOwnerCall::new(
                IndexKeyPublicationId::new(3, 0, 0),
                0,
                nz(first.token_count),
                &input,
                &frequencies[..first.token_count * 16],
                RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
            ))
            .is_err()
    );
    assert_eq!(state(&owner), after_reset, "stale publication is invisible");
    for index in 0..fixture.cases.len() {
        commit_case(&mut owner, &fixture, index);
    }
}
