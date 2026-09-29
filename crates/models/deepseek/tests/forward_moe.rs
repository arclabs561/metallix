//! Native complete `MoE` sublayer against encoded synthetic source-forward data.
//! The connected layer-three/four suffix joins native owner/producer attention
//! through HC/FFN and final logits; earlier block inputs remain captured.
//! The isolated layer-four test retains its captured entry as a diagnostic.
//! It is not complete native model execution.

use std::{collections::BTreeMap, num::NonZeroUsize};

use deepseek::moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEReference};
use deepseek::reduced::{
    AttentionInput, BlockTailReference, FinalHead, LayerFourCall, LayerFourConfig,
    LayerFourSession, LayerThreePublication,
};
use deepseek::{
    RotaryFrequency,
    attention::layer::Fp8Projection,
    ffn::FfnSublayerReference,
    hc::{
        HcCoefficients,
        mixing::{hc_post_bf16_reference, hc_pre_bf16_reference},
        projection::project_hc_coefficients,
        split_hc_coefficients,
    },
    indexer::{
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights},
        selection::{
            SelectionAdapterError, SelectionCall, SelectionGeometry, select_from_candidates,
        },
    },
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use partition_owner::attention_capture;
#[path = "support/candidate_capture.rs"]
mod candidate_capture;
#[path = "support/engram_capture.rs"]
mod engram_capture;
#[path = "support/hc_chain_bounds.rs"]
mod hc_chain_bounds;
#[path = "support/layer1_attention_capture.rs"]
mod layer1_attention_capture;
#[path = "support/layer1_join.rs"]
mod layer1_join;
#[allow(
    dead_code,
    reason = "standalone owner controls have a separate test binary"
)]
#[path = "support/layer1_owner_capture.rs"]
mod layer1_owner_capture;
#[path = "support/layer2_attention_capture.rs"]
mod layer2_attention_capture;
#[path = "support/layer2_ffn.rs"]
mod layer2_ffn;
#[path = "support/layer2_join.rs"]
mod layer2_join;
#[path = "forward_layer0_to_layer1.rs"]
mod layer_zero;
#[path = "support/owner_attention_capture.rs"]
mod owner_attention_capture;
#[allow(
    dead_code,
    reason = "the alternate owner-attention runner shares its test-only source qualification"
)]
#[path = "forward_partition_owner.rs"]
mod partition_owner;
#[path = "support/request_capture.rs"]
mod request_capture;
#[path = "support/rounding_interval.rs"]
mod rounding_interval;

use layer_zero::{hc_coefficient_bounds, hc_projection_bounds, layer1_engram_capture};
use layer2_ffn::native_layer_two_entries;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
    comparison_policy: Policy,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
}

#[derive(Deserialize)]
struct BlockConfig {
    copies: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    norm_eps: f32,
}

#[derive(Debug, Deserialize, PartialEq)]
struct Source {
    revision: String,
    model_sha256: String,
    cpu_backend_sha256: String,
    #[serde(default)]
    complete_capture_sha256: Option<String>,
    #[serde(default)]
    storage_byteorder: Option<String>,
    kernel_source_sha256: String,
    loader_sha256: String,
    engram_sha256: String,
    #[serde(default)]
    manifest_canonical_sha256: Option<String>,
    runner_sha256: String,
}

#[derive(Deserialize)]
struct AlternatePartitionFixture {
    schema_version: u32,
    source_receipt_sha256: String,
    source: Source,
    capture_identity: Value,
    post_attention: AlternatePostAttention,
}

#[derive(Deserialize)]
struct AlternatePostAttention {
    source_receipt_sha256: String,
    source: Source,
    capture_identity: Value,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
    comparison_policy: Policy,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
}

#[derive(Deserialize)]
struct AlternatePartitionL4Fixture {
    schema_version: u32,
    source_receipt_sha256: String,
    source: Source,
    capture_identity: Value,
    post_layer_three: AlternatePostLayerThree,
}

#[derive(Deserialize)]
struct AlternatePostLayerThree {
    source_receipt_sha256: String,
    source: Source,
    capture_identity: Value,
    attention: AlternateAttentionFixture,
    selection: AlternateL4SelectionFixture,
    tail: AlternateL4Tail,
    head: AlternateHead,
}

#[derive(Deserialize)]
struct AlternateAttentionFixture {
    model: attention_capture::Model,
    frequencies: Tensor,
    encoded_parameters: BTreeMap<String, attention_capture::Tensor>,
    cases: Vec<attention_capture::Case>,
}

#[derive(Deserialize)]
struct AlternateL4SelectionFixture {
    model: AlternateL4SelectionModel,
    weights: AlternateL4SelectionWeights,
    cases: Vec<AlternateL4SelectionCase>,
}

#[derive(Deserialize)]
struct AlternateL4SelectionModel {
    query_rank: usize,
    index_heads: usize,
    index_topk: usize,
}

#[derive(Deserialize)]
struct AlternateL4SelectionWeights {
    wq_a_codes: Tensor,
    wq_a_scales: Tensor,
    q_norm: Tensor,
    wq_b_codes: Tensor,
    wq_b_scales: Tensor,
    weights_proj: Tensor,
}

#[derive(Deserialize)]
struct AlternateL4SelectionCase {
    start_pos: usize,
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
    scores_after_candidate_mask: Tensor,
    indices: Tensor,
}

#[derive(Deserialize)]
struct AlternateL4Tail {
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
    comparison_policy: Policy,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
}

#[derive(Deserialize)]
struct AlternateHead {
    norm_weight: Tensor,
    head_weight: Tensor,
    norm_epsilon: f32,
    cases: Vec<AlternateHeadCase>,
}

#[derive(Deserialize)]
struct AlternateHeadCase {
    start_pos: usize,
    norm_input: Tensor,
    norm: Tensor,
    logits: Tensor,
}

struct AlternateL4RunFixture {
    attention: AlternateAttentionFixture,
    selection: AlternateL4SelectionFixture,
    tail: Fixture,
    head: AlternateHead,
}

struct AlternateL4Query {
    layout: CandidateQueryLayout,
    query_codes: Vec<u8>,
    query_scales: Vec<u8>,
    query_norm: Vec<u16>,
    index_codes: Vec<u8>,
    index_scales: Vec<u8>,
    weights_projection: Vec<u16>,
    head_dimension: usize,
    index_topk: usize,
}

#[derive(Deserialize)]
struct Model {
    dim: usize,
    moe_inter_dim: usize,
    n_routed_experts: usize,
    n_activated_experts: usize,
    n_shared_experts: usize,
    score_func: String,
    gate_temp: f32,
    norm_topk_prob: bool,
    route_scale: f32,
    swiglu_limit: f32,
    expert_dtype: String,
}

#[derive(Deserialize)]
struct Policy {
    output_bf16: String,
    route_weight_abs_error_max: f32,
    fixed_before_candidate_execution: bool,
    block_next_pre_abs_error_max: f32,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    input: Tensor,
    gate_weights: Tensor,
    gate_indices: Tensor,
    output: Tensor,
    block_input: Tensor,
    block_incoming_pre: Tensor,
    attention_input: Tensor,
    attention_output: Tensor,
    after_attention_residual: Tensor,
    attention_hc_mixes: Tensor,
    attention_coefficients: Coefficients,
    ffn_collapsed: Tensor,
    ffn_hc_mixes: Tensor,
    ffn_coefficients: Coefficients,
    block_output: Tensor,
    block_next_pre: Tensor,
    next_block_entry: Option<BlockEntry>,
}

#[derive(Deserialize)]
struct BlockEntry {
    residual: Tensor,
    incoming_pre: Tensor,
}

#[derive(Deserialize)]
struct Coefficients {
    pre: Tensor,
    post: Tensor,
    comb: Tensor,
}

#[derive(Deserialize)]
struct Tensor {
    shape: Vec<usize>,
    dtype: String,
    storage_hex: String,
}

#[derive(Deserialize)]
struct HeadFixture {
    source: HeadSource,
    weight_shape: [usize; 2],
    weight_fp32_bits: Vec<u32>,
    norm_weight_bf16: Vec<u16>,
    norm_epsilon_bits: u32,
    cases: Vec<HeadCase>,
}

#[derive(Deserialize)]
struct HeadSource {
    revision: String,
    model_sha256: String,
    complete_capture_sha256: String,
}

#[derive(Deserialize)]
struct HeadCase {
    start_pos: usize,
    input_shape: [usize; 3],
    input_bf16: Vec<u16>,
    logits_shape: [usize; 2],
    logits_fp32_bits: Vec<u32>,
    final_block_shape: [usize; 4],
    final_block_bf16: Vec<u16>,
    final_pre_shape: [usize; 3],
    final_pre_fp32_bits: Vec<u32>,
    collapsed_bf16: Vec<u16>,
}

#[derive(Deserialize)]
struct LayerTwoFixture {
    schema_version: u32,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
    cases: Vec<LayerTwoCase>,
}

#[derive(Deserialize)]
struct LayerTwoCase {
    start_pos: usize,
    after_attention_residual: Tensor,
    attention_pre: Tensor,
    ffn_collapsed: Tensor,
    ffn_hc_mixes: Tensor,
    ffn_coefficients: Coefficients,
    moe_input: Tensor,
    moe_output: Tensor,
    output: Tensor,
    next_pre: Tensor,
    engram_stream: Tensor,
    layer_three_incoming_pre: Tensor,
}

impl Tensor {
    fn bytes(&self) -> Vec<u8> {
        assert_eq!(self.storage_hex.len() % 2, 0);
        self.storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
            .collect()
    }

    fn bf16(&self) -> Vec<u16> {
        assert_eq!(self.dtype, "torch.bfloat16");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 2);
        bytes
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn fp32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "torch.float32");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 4);
        bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn fp8(&self) -> Vec<u8> {
        assert!(matches!(
            self.dtype.as_str(),
            "torch.float8_e4m3fn" | "torch.float8_e8m0fnu"
        ));
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>());
        bytes
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

    fn bools(&self) -> Vec<bool> {
        assert_eq!(self.dtype, "torch.bool");
        self.bytes()
            .into_iter()
            .map(|value| match value {
                0 => false,
                1 => true,
                _ => panic!("alternate source bool storage"),
            })
            .collect()
    }

    fn frequencies(&self) -> Vec<RotaryFrequency> {
        assert_eq!(self.dtype, "torch.complex64");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 8);
        bytes
            .chunks_exact(8)
            .map(|pair| {
                RotaryFrequency::new(
                    f32::from_le_bytes(pair[..4].try_into().unwrap()),
                    f32::from_le_bytes(pair[4..].try_into().unwrap()),
                )
                .expect("finite alternate RoPE frequency")
            })
            .collect()
    }

    fn indices(&self) -> Vec<usize> {
        assert_eq!(self.dtype, "torch.int64");
        let bytes = self.bytes();
        assert_eq!(bytes.len(), self.shape.iter().product::<usize>() * 8);
        bytes
            .chunks_exact(8)
            .map(|b| usize::try_from(i64::from_le_bytes(b.try_into().unwrap())).unwrap())
            .collect()
    }
}

fn fixture() -> Fixture {
    let f = fixture_from(include_str!(
        "../../../../fixtures/deepseek-v41/forward-moe-reference.json"
    ));
    assert_eq!(f.schema_version, 1);
    assert_source_provenance(&f);
    assert_encoded_parameter_schema(&f);
    assert_model_and_case_contract(&f);
    f
}

fn alternate_partition_tail_fixture() -> Fixture {
    const SOURCE_RECEIPT_SHA256: &str =
        "9613150fea8010a7435dab0443a1f9e0d73fd8d0f32455b8d67dd572617f3906";
    const FIXTURE_SHA256: &str = "3e9c27c53e1bb2240912bdb3ee68747b17287d12e9d40fb6cd9d02088f277f68";
    let raw = include_str!("../../../../fixtures/deepseek-v41/partition-owner-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let alternate: AlternatePartitionFixture =
        serde_json::from_str(raw).expect("typed alternate partition fixture");
    assert_eq!(alternate.schema_version, 1);
    assert_eq!(alternate.source_receipt_sha256, SOURCE_RECEIPT_SHA256);
    assert_eq!(
        alternate.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        alternate.source.model_sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        alternate.post_attention.source_receipt_sha256, alternate.source_receipt_sha256,
        "post-attention source receipt"
    );
    assert_eq!(
        alternate.post_attention.source, alternate.source,
        "post-attention source provenance"
    );
    assert_eq!(
        alternate.post_attention.capture_identity, alternate.capture_identity,
        "post-attention capture identity"
    );
    assert_eq!(
        alternate.capture_identity["schedule"],
        serde_json::json!([4, 1, 1, 1]),
        "alternate capture schedule"
    );
    let post = alternate.post_attention;
    let fixture = Fixture {
        schema_version: alternate.schema_version,
        source: post.source,
        model: post.model,
        encoded_parameters: post.encoded_parameters,
        cases: post.cases,
        comparison_policy: post.comparison_policy,
        block_parameters: post.block_parameters,
        block_config: post.block_config,
    };
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| case.start_pos)
            .collect::<Vec<_>>(),
        [0, 4, 5, 6],
        "alternate layer-three calls"
    );
    assert!(fixture.source.complete_capture_sha256.is_none());
    assert!(fixture.source.manifest_canonical_sha256.is_none());
    assert!(fixture.source.storage_byteorder.is_none());
    assert_encoded_parameter_schema_for(&fixture, 3);
    validate_block_tail_fixture(&fixture);
    fixture
}

impl AlternateL4Query {
    fn new(fixture: &AlternateL4RunFixture) -> Self {
        let model = &fixture.selection.model;
        let attention = &fixture.attention.model;
        let weights = &fixture.selection.weights;
        let index_layout = IndexQueryLayout::new(
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(attention.dim).unwrap(),
            NonZeroUsize::new(model.query_rank).unwrap(),
            NonZeroUsize::new(model.index_heads).unwrap(),
            NonZeroUsize::new(attention.head_dim).unwrap(),
            NonZeroUsize::new(attention.rope_head_dim / 2).unwrap(),
        )
        .unwrap();
        Self {
            layout: CandidateQueryLayout::new(index_layout, attention.norm_eps).unwrap(),
            query_codes: weights.wq_a_codes.fp8(),
            query_scales: weights.wq_a_scales.fp8(),
            query_norm: weights.q_norm.bf16(),
            index_codes: weights.wq_b_codes.fp8(),
            index_scales: weights.wq_b_scales.fp8(),
            weights_projection: weights.weights_proj.bf16(),
            head_dimension: attention.head_dim,
            index_topk: model.index_topk,
        }
    }
}

fn alternate_l4_input(
    l3: &BlockTailOutput,
    tail_case: &Case,
    attention_case: &attention_capture::Case,
    norm_weight: &[u16],
    epsilon: f32,
) -> Vec<u16> {
    assert_eq!(
        l3.residual,
        tail_case.block_input.bf16(),
        "native alternate L3 residual enters L4"
    );
    let source_pre = tail_case.block_incoming_pre.fp32();
    let source_residual = tail_case.block_input.bf16();
    for (position, envelope) in l3.terminal_envelopes.as_ref().unwrap().iter().enumerate() {
        assert!(
            envelope.accepts(
                &l3.residual[position * 256..(position + 1) * 256],
                &l3.next_pre[position * 2..(position + 1) * 2],
                &source_residual[position * 256..(position + 1) * 256],
                &source_pre[position * 2..(position + 1) * 2],
            ),
            "native L3 entry retains its established coefficient bounds"
        );
    }
    let input: Vec<_> = (0..attention_case.input.shape[1])
        .flat_map(|position| {
            derive_attention_input(
                &l3.residual[position * 256..(position + 1) * 256],
                &l3.next_pre[position * 2..(position + 1) * 2],
                norm_weight,
                epsilon,
            )
        })
        .collect();
    assert_eq!(
        input,
        attention_case.input.bf16(),
        "native alternate L3 tail feeds L4 attention input"
    );
    assert_eq!(
        input,
        tail_case.attention_input.bf16(),
        "alternate L4 tail input"
    );
    input
}

fn assert_alternate_l4_query(
    prepared: &deepseek::indexer::query::ScoredQueryDiagnostic,
    selection_case: &AlternateL4SelectionCase,
) {
    assert_eq!(
        prepared.query.wq_a,
        selection_case.wq_a.bf16(),
        "alternate L4 WQ-A"
    );
    assert_eq!(
        prepared.query.qr,
        selection_case.qr.bf16(),
        "alternate L4 QR"
    );
    assert_eq!(
        prepared.query.index.query_post_fp4,
        selection_case.q_after_rope_fp4.bf16(),
        "alternate L4 index query"
    );
    assert_eq!(
        prepared.query.index.projected_head_weights,
        selection_case.weights_proj_output.bf16(),
        "alternate L4 index weights projection"
    );
    assert_eq!(
        prepared.query.index.scaled_head_weights,
        selection_case.scaled_weights.bf16(),
        "alternate L4 scaled index weights"
    );
    assert_eq!(
        prepared.dot_products,
        selection_case.dot_products.bf16(),
        "alternate L4 dots"
    );
    assert_eq!(
        prepared.rectified,
        selection_case.rectified.bf16(),
        "alternate L4 rectified"
    );
    assert_eq!(
        prepared.weighted,
        selection_case.weighted.bf16(),
        "alternate L4 weighted"
    );
    assert_eq!(
        prepared.scores,
        selection_case.scores.bf16(),
        "alternate L4 scores"
    );
}

#[allow(
    clippy::too_many_lines,
    reason = "keep the source stage comparisons beside live L4 execution"
)]
fn alternate_l4_attention_outputs(
    fixture: &AlternateL4RunFixture,
    publications: &[partition_owner::AlternateLayerThreePublication],
    l3_tail: &[BlockTailOutput],
) -> Vec<Vec<u16>> {
    assert_eq!(publications.len(), fixture.attention.cases.len());
    assert_eq!(l3_tail.len(), fixture.attention.cases.len());
    assert_eq!(fixture.selection.cases.len(), fixture.attention.cases.len());
    assert_eq!(fixture.tail.cases.len(), fixture.attention.cases.len());
    let query = AlternateL4Query::new(fixture);
    let weights = attention_capture::weights_for_layer(&fixture.attention.encoded_parameters, 4);
    let frequencies = fixture.attention.frequencies.frequencies();
    let parameters = block_tail_parameters(&fixture.tail);
    let mut attention_state = LayerFourSession::new(
        LayerFourConfig::new(
            query.layout,
            attention_capture::layout(&fixture.attention.model),
            NonZeroUsize::new(query.index_topk).unwrap(),
        )
        .expect("alternate L4 runtime layout"),
    );
    fixture
        .attention
        .cases
        .iter()
        .zip(&fixture.selection.cases)
        .zip(&fixture.tail.cases)
        .zip(publications)
        .zip(l3_tail)
        .map(
            |((((attention_case, selection_case), tail_case), publication), l3)| {
                assert_eq!(attention_case.start_pos, selection_case.start_pos);
                assert_eq!(attention_case.start_pos, tail_case.start_pos);
                assert_eq!(attention_case.start_pos, publication.start_pos);
                let input = alternate_l4_input(
                    l3,
                    tail_case,
                    attention_case,
                    &parameters.attn_norm,
                    fixture.tail.block_config.norm_eps,
                );
                let geometry = SelectionGeometry::new(
                    attention_case.start_pos,
                    NonZeroUsize::new(attention_case.input.shape[1]).unwrap(),
                    NonZeroUsize::new(publication.key_prefix.len() / query.head_dimension).unwrap(),
                    NonZeroUsize::new(fixture.attention.model.compress_ratios[4]).unwrap(),
                    selection_case.offset,
                )
                .unwrap();
                let call = SelectionCall::new(publication.publication, 0, geometry);
                assert_eq!(
                    publication.producer_candidates.call(),
                    call,
                    "L4 consumes the committed L3 candidate set"
                );
                assert_eq!(
                    publication.producer_candidates.mask(),
                    selection_case.candidate_mask.bools(),
                    "L4 producer mask"
                );
                let call_frequencies =
                    attention_capture::call_frequencies(&frequencies, attention_case);
                let query_weights = CandidateQueryWeights {
                    wq_a: Fp8Projection {
                        codes: &query.query_codes,
                        scales: &query.query_scales,
                    },
                    q_norm: &query.query_norm,
                    index: IndexQueryWeights {
                        wq_b_codes: &query.index_codes,
                        wq_b_scales: &query.index_scales,
                        weights_proj: &query.weights_projection,
                    },
                };
                let publication_input = LayerThreePublication::new(
                    publication.publication,
                    &publication.key_prefix,
                    &publication.kv_prefix,
                    &publication.producer_candidates,
                );
                let output = attention_state
                    .step(LayerFourCall::new(
                        &input,
                        call_frequencies,
                        query_weights,
                        weights.borrowed(),
                        publication_input,
                    ))
                    .expect("alternate committed L3 publication drives L4 runtime");
                assert_alternate_l4_query(output.scored(), selection_case);
                let stale = publications
                    .iter()
                    .find(|other| other.publication != publication.publication);
                if let Some(stale) = stale {
                    assert!(matches!(
                        select_from_candidates(
                            &output.scored().scores,
                            call,
                            &stale.producer_candidates,
                            query.index_topk,
                        ),
                        Err(SelectionAdapterError::CandidateCallMismatch)
                    ));
                }
                if let Some(expected_causal) = &selection_case.causal_scores {
                    assert_eq!(output.selection().causal_scores, expected_causal.bf16());
                }
                assert_eq!(
                    output.selection().masked_scores,
                    selection_case.scores_after_candidate_mask.bf16(),
                    "alternate L4 candidate-masked scores"
                );
                assert_eq!(
                    output.selection().indices,
                    selection_case.indices.i32(),
                    "alternate L4 IDs"
                );
                attention_capture::assert_diagnostic(attention_case, output.attention());
                output.attention().final_output.clone()
            },
        )
        .collect()
}

fn alternate_partition_l4_fixture() -> AlternateL4RunFixture {
    const SOURCE_RECEIPT_SHA256: &str =
        "9613150fea8010a7435dab0443a1f9e0d73fd8d0f32455b8d67dd572617f3906";
    const FIXTURE_SHA256: &str = "3e9c27c53e1bb2240912bdb3ee68747b17287d12e9d40fb6cd9d02088f277f68";
    let raw = include_str!("../../../../fixtures/deepseek-v41/partition-owner-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let alternate: AlternatePartitionL4Fixture =
        serde_json::from_str(raw).expect("typed alternate L4 fixture");
    assert_eq!(alternate.schema_version, 1);
    assert_eq!(alternate.source_receipt_sha256, SOURCE_RECEIPT_SHA256);
    assert_eq!(
        alternate.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        alternate.source.model_sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        alternate.post_layer_three.source_receipt_sha256, alternate.source_receipt_sha256,
        "alternate L4 source receipt"
    );
    assert_eq!(
        alternate.post_layer_three.source, alternate.source,
        "alternate L4 source provenance"
    );
    assert_eq!(
        alternate.post_layer_three.capture_identity, alternate.capture_identity,
        "alternate L4 capture identity"
    );
    let post = alternate.post_layer_three;
    let tail = Fixture {
        schema_version: alternate.schema_version,
        source: alternate.source,
        model: post.tail.model,
        encoded_parameters: post.tail.encoded_parameters,
        cases: post.tail.cases,
        comparison_policy: post.tail.comparison_policy,
        block_parameters: post.tail.block_parameters,
        block_config: post.tail.block_config,
    };
    assert_eq!(
        tail.cases
            .iter()
            .map(|case| case.start_pos)
            .collect::<Vec<_>>(),
        [0, 4, 5, 6],
        "alternate L4 tail calls"
    );
    assert_encoded_parameter_schema(&tail);
    validate_block_tail_fixture(&tail);
    AlternateL4RunFixture {
        attention: post.attention,
        selection: post.selection,
        tail,
        head: post.head,
    }
}

fn assert_alternate_head(head: &AlternateHead, l4_tail: &[BlockTailOutput]) {
    assert_eq!(head.norm_weight.shape, [128]);
    assert_eq!(head.head_weight.shape[1], 128);
    let norm_weight = head.norm_weight.bf16();
    let head_weight = head.head_weight.fp32();
    let vocabulary = head.head_weight.shape[0];
    let executor = FinalHead::new(&norm_weight, &head_weight, vocabulary, 2, head.norm_epsilon)
        .expect("bounded alternate final head");
    assert_eq!(head.cases.len(), l4_tail.len());
    for (case, block) in head.cases.iter().zip(l4_tail) {
        let positions = case.norm_input.shape[1];
        assert_eq!(case.norm_input.shape, [1, positions, 128]);
        assert_eq!(case.norm.shape, case.norm_input.shape);
        assert_eq!(case.logits.shape, [1, vocabulary]);
        assert_eq!(block.residual.len(), positions * 256);
        assert_eq!(block.next_pre.len(), positions * 2);
        let expected_collapsed = case.norm_input.bf16();
        let expected_normalized = case.norm.bf16();
        let mut last_logits = Vec::new();
        let mut last_bounds = Vec::new();
        for position in 0..positions {
            let output = executor
                .forward(
                    &block.residual[position * 256..(position + 1) * 256],
                    &block.next_pre[position * 2..(position + 1) * 2],
                )
                .expect("native alternate final head");
            let collapsed = output.collapsed_bf16();
            let normalized = output.normalized_bf16();
            let envelope = block.terminal_envelopes.as_ref().unwrap()[position]
                .final_norm_envelope(&norm_weight, head.norm_epsilon);
            let source_collapsed = &expected_collapsed[position * 128..(position + 1) * 128];
            let source_normalized = &expected_normalized[position * 128..(position + 1) * 128];
            assert!(
                envelope.accepts(collapsed, source_collapsed, normalized, source_normalized),
                "alternate final HC/norm remains inside the established source bounds"
            );
            assert!(
                !envelope.accepts(collapsed, source_collapsed, &[0; 128], source_normalized),
                "discarded final normalization must be rejected"
            );
            last_bounds = envelope.head_bounds(source_normalized, &head_weight);
            last_logits = output.logits().to_vec();
        }
        let source_logits: Vec<_> = case.logits.fp32().into_iter().map(f32::to_bits).collect();
        assert!(
            agrees_with_head_oracle(&last_logits, &source_logits, &last_bounds,),
            "alternate final logits at start {}",
            case.start_pos
        );
    }
}

fn fixture_from(source: &str) -> Fixture {
    serde_json::from_str(source).expect("valid source MoE fixture")
}

fn layer_three_fixture() -> Fixture {
    let f = fixture_from(include_str!(
        "../../../../fixtures/deepseek-v41/forward-layer3-moe-reference.json"
    ));
    assert_eq!(f.schema_version, 1);
    assert_eq!(
        f.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        f.source.model_sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        f.source.complete_capture_sha256.as_deref(),
        Some("7a6290921f79573e976aba42ec296f038adda2efc0d3d89b8583c6f58c79cb92")
    );
    assert_eq!(
        f.source.cpu_backend_sha256,
        "b1f1f3cfdb93b674a5f96a114cf45bf5be9ad3a555ae95ac24add567f9f5232e"
    );
    assert_eq!(
        f.source.kernel_source_sha256,
        "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    );
    assert_eq!(
        f.source.loader_sha256,
        "359c4c961bdc8e200e2ccd13e7499974220a8d6942b5f6627316ab54210bef03"
    );
    assert_eq!(
        f.source.engram_sha256,
        "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
    );
    assert_eq!(
        f.source.manifest_canonical_sha256.as_deref(),
        Some("fd69a8fce4d5048f87db705603e05e3077c4f9bda402ec08be848aaa5cbdb92e")
    );
    assert_eq!(
        f.source.runner_sha256,
        "48f10d6a0ba0888580132a08e9821bf0f707ec5a0cc609a777c0a37c59666684"
    );
    assert_eq!(f.source.storage_byteorder.as_deref(), Some("little"));
    assert_model_and_case_contract(&f);
    f
}

fn layer_three_fixture_from_bundle(bundle: &Value) -> Fixture {
    moe_fixture_from_bundle(bundle, 3)
}

fn moe_fixture_from_bundle(bundle: &Value, layer: usize) -> Fixture {
    assert_eq!(bundle["schema_version"].as_u64(), Some(1));
    let name = format!("layer{layer}_moe");
    let projection = &bundle["projections"][&name];
    let pinned: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("pinned bundle metadata");
    assert_eq!(bundle["source"], pinned["source"], "bundle source metadata");
    assert_eq!(
        projection["source"], pinned["projections"][&name]["source"],
        "{name} source metadata"
    );
    let fixture: Fixture = serde_json::from_value(projection.clone()).expect("typed bundle MoE");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.source.revision,
        bundle["source"]["revision"]
            .as_str()
            .expect("bundle revision")
    );
    assert_eq!(
        fixture.source.model_sha256,
        bundle["source"]["model_sha256"]
            .as_str()
            .expect("bundle model")
    );
    assert_eq!(
        fixture.source.complete_capture_sha256.as_deref(),
        Some(
            bundle["source"]["complete_capture_sha256"]
                .as_str()
                .expect("bundle capture")
        )
    );
    assert_eq!(
        fixture.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture.source.model_sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(fixture.cases.len(), 3);
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| case.start_pos)
            .collect::<Vec<_>>(),
        [0, 5, 6]
    );
    assert_encoded_parameter_schema_for(&fixture, layer);
    assert_model_and_case_contract(&fixture);
    validate_block_tail_fixture(&fixture);
    fixture
}

fn head_fixture() -> HeadFixture {
    serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-head-reference.json"
    ))
    .expect("source-forward head fixture")
}

fn assert_source_provenance(f: &Fixture) {
    assert_eq!(
        f.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        f.source.model_sha256,
        "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
    );
    assert_eq!(
        f.source.cpu_backend_sha256,
        "b1f1f3cfdb93b674a5f96a114cf45bf5be9ad3a555ae95ac24add567f9f5232e"
    );
    assert_eq!(
        f.source.complete_capture_sha256.as_deref(),
        Some("e27dde6ead409c74f7bb2c9e08d4cd5a2b0cfc3c9505c7d6b8908b1cd78b1cc6")
    );
    assert_eq!(
        f.source.runner_sha256,
        "bc1a1cca7c3570831152cd98b829b41905c50b9d0485c1354268a647c3e5dff8"
    );
    assert_eq!(f.source.storage_byteorder.as_deref(), Some("little"));
    assert_eq!(
        f.source.kernel_source_sha256,
        "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
    );
    assert_eq!(
        f.source.loader_sha256,
        "359c4c961bdc8e200e2ccd13e7499974220a8d6942b5f6627316ab54210bef03"
    );
    assert_eq!(
        f.source.engram_sha256,
        "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897"
    );
    assert_eq!(
        f.source.manifest_canonical_sha256.as_deref(),
        Some("fd69a8fce4d5048f87db705603e05e3077c4f9bda402ec08be848aaa5cbdb92e")
    );
}

fn assert_encoded_parameter_schema(f: &Fixture) {
    assert_encoded_parameter_schema_for(f, 4);
}

fn assert_encoded_parameter_schema_for(f: &Fixture, layer: usize) {
    assert_eq!(f.encoded_parameters.len(), 32);
    for expert in (0..4)
        .map(|id| format!("experts.{id}"))
        .chain(["shared_experts".to_owned()])
    {
        let shared = expert == "shared_experts";
        for projection in ["w1", "w2", "w3"] {
            let prefix = format!("layers.{layer}.ffn.{expert}.{projection}");
            let weight = &f.encoded_parameters[&format!("{prefix}.weight")];
            let scale = &f.encoded_parameters[&format!("{prefix}.scale")];
            assert_eq!(
                weight.dtype,
                if shared {
                    "torch.float8_e4m3fn"
                } else {
                    "torch.float4_e2m1fn_x2"
                }
            );
            assert_eq!(weight.shape, [128, if shared { 128 } else { 64 }]);
            assert_eq!(scale.dtype, "torch.float8_e8m0fnu");
            assert_eq!(scale.shape, [if shared { 4 } else { 128 }, 4]);
        }
    }
}

fn assert_model_and_case_contract(f: &Fixture) {
    assert_eq!((f.model.dim, f.model.moe_inter_dim), (128, 128));
    assert_eq!(
        (
            f.model.n_routed_experts,
            f.model.n_activated_experts,
            f.model.n_shared_experts
        ),
        (4, 2, 1)
    );
    assert_eq!(f.model.expert_dtype, "fp4");
    assert_eq!(f.model.score_func, "sqrtsoftplus");
    assert_eq!(f.comparison_policy.output_bf16, "exact storage bits");
    assert_eq!(
        f.comparison_policy.route_weight_abs_error_max.to_bits(),
        2.0_f32.powi(-20).to_bits()
    );
    assert!(f.comparison_policy.fixed_before_candidate_execution);
    assert_eq!(f.cases.len(), 3);
    for (case, (start, positions)) in f.cases.iter().zip([(0, 5), (5, 1), (6, 1)]) {
        assert_eq!(case.start_pos, start);
        assert_eq!(case.input.shape, [1, positions, 128]);
        assert_eq!(case.output.shape, [1, positions, 128]);
        assert_eq!(case.gate_weights.shape, [positions, 2]);
        assert_eq!(case.gate_indices.shape, [positions, 2]);
        assert_eq!(case.block_input.shape, [1, positions, 2, 128]);
        assert_eq!(case.after_attention_residual.shape, case.block_input.shape);
        assert_eq!(case.attention_input.shape, [1, positions, 128]);
        assert_eq!(case.attention_output.shape, case.attention_input.shape);
        assert_eq!(case.ffn_collapsed.shape, case.attention_input.shape);
        for coefficients in [&case.attention_coefficients, &case.ffn_coefficients] {
            assert_eq!(coefficients.pre.shape, [1, positions, 2]);
            assert_eq!(coefficients.post.shape, [1, positions, 2]);
            assert_eq!(coefficients.comb.shape, [1, positions, 2, 2]);
        }
        assert_eq!(case.attention_hc_mixes.shape, [1, positions, 8]);
        assert_eq!(case.ffn_hc_mixes.shape, [1, positions, 8]);
    }
}

fn coefficient_bit_differences(
    actual: &HcCoefficients,
    expected_pre: &[f32],
    expected_post: &[f32],
    expected_comb: &[f32],
    context: &str,
) -> Vec<String> {
    assert_eq!(actual.copies(), 2, "{context} copies");
    let mut differences = Vec::new();
    for (field, actual, expected) in [
        ("pre", actual.pre(), expected_pre),
        ("post", actual.post(), expected_post),
        ("comb", actual.comb(), expected_comb),
    ] {
        assert_eq!(actual.len(), expected.len(), "{context} {field} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            if actual.to_bits() != expected.to_bits() {
                differences.push(format!(
                    "{context} {field}[{index}]: {actual:?} ({:#010x}) != {expected:?} ({:#010x})",
                    actual.to_bits(),
                    expected.to_bits(),
                ));
            }
        }
    }
    differences
}

fn with_model<R>(f: &Fixture, omit_shared: bool, body: impl FnOnce(MoEReference<'_>) -> R) -> R {
    with_model_for(f, 4, omit_shared, body)
}

fn with_model_for<R>(
    f: &Fixture,
    layer: usize,
    omit_shared: bool,
    body: impl FnOnce(MoEReference<'_>) -> R,
) -> R {
    with_model_parameters(&f.model, &f.encoded_parameters, layer, omit_shared, body)
}

fn with_model_parameters<R>(
    c: &Model,
    parameters: &BTreeMap<String, Tensor>,
    layer: usize,
    omit_shared: bool,
    body: impl FnOnce(MoEReference<'_>) -> R,
) -> R {
    let mut encoded: BTreeMap<String, Vec<u8>> = parameters
        .iter()
        .map(|(k, v)| (k.clone(), v.bytes()))
        .collect();
    if omit_shared {
        encoded
            .get_mut(&format!("layers.{layer}.ffn.shared_experts.w2.weight"))
            .unwrap()
            .fill(0);
    }
    let expert_bytes = |prefix: &str| -> [&[u8]; 6] {
        [
            "w1.weight",
            "w1.scale",
            "w2.weight",
            "w2.scale",
            "w3.weight",
            "w3.scale",
        ]
        .map(|suffix| encoded[&format!("layers.{layer}.ffn.{prefix}.{suffix}")].as_slice())
    };
    let routed: Vec<_> = (0..4)
        .map(|id| {
            let [w1, s1, w2, s2, w3, s3] = expert_bytes(&format!("experts.{id}"));
            Fp4ExpertWeights::new(128, 128, w1, s1, w2, s2, w3, s3).unwrap()
        })
        .collect();
    let [w1, s1, w2, s2, w3, s3] = expert_bytes("shared_experts");
    let shared = Fp8ExpertWeights::new(128, 128, w1, s1, w2, s2, w3, s3).unwrap();
    let gate = parameters[&format!("layers.{layer}.ffn.gate.weight")].bf16();
    let bias = parameters[&format!("layers.{layer}.ffn.gate.bias")].fp32();
    let config = MoEConfig::new(
        c.dim,
        c.moe_inter_dim,
        c.swiglu_limit,
        c.n_activated_experts,
        c.gate_temp,
        c.norm_topk_prob,
        c.route_scale,
    )
    .unwrap();
    let model = MoEReference::new(config, &gate, &bias, &routed, shared).unwrap();
    body(model)
}

fn run(f: &Fixture, omit_shared: bool) -> Vec<Vec<u16>> {
    with_model(f, omit_shared, |model| {
        let mut outputs = Vec::new();
        for case in &f.cases {
            let input = case.input.bf16();
            let expected_ids = case.gate_indices.indices();
            let expected_weights = case.gate_weights.fp32();
            let mut output = Vec::new();
            for (position, row) in input.chunks_exact(128).enumerate() {
                let result = model.forward_token(row).unwrap();
                assert_eq!(result.routes().len(), 2);
                let ids = &expected_ids[position * 2..(position + 1) * 2];
                let weights = &expected_weights[position * 2..(position + 1) * 2];
                let mut sorted_ids = ids.to_vec();
                sorted_ids.sort_unstable();
                let native_ids: Vec<_> = result.routes().iter().map(|r| r.expert_index()).collect();
                assert_eq!(
                    native_ids, sorted_ids,
                    "complete selected expert set in ascending execution order"
                );
                for route in result.routes() {
                    let index = ids
                        .iter()
                        .position(|&id| id == route.expert_index())
                        .expect("native selected expert is selected by source");
                    assert!(
                        (route.weight() - weights[index]).abs()
                            <= f.comparison_policy.route_weight_abs_error_max,
                        "start {} position {position} expert {} route weight {} != {}",
                        case.start_pos,
                        route.expert_index(),
                        route.weight(),
                        weights[index]
                    );
                }
                output.extend_from_slice(result.output_bf16());
            }
            outputs.push(output);
        }
        outputs
    })
}

#[test]
fn native_moe_matches_all_source_prefill_and_decode_outputs() {
    let f = fixture();
    for (case, actual) in f.cases.iter().zip(run(&f, false)) {
        assert_eq!(
            actual,
            case.output.bf16(),
            "source MoE at start {}",
            case.start_pos
        );
    }
}

#[test]
fn source_oracle_rejects_omitting_shared_expert() {
    let f = fixture();
    for (case, actual) in f.cases.iter().zip(run(&f, true)) {
        assert_ne!(
            actual,
            case.output.bf16(),
            "shared contribution must matter at start {}",
            case.start_pos
        );
    }
}

#[test]
fn captured_attention_hc_post_reproduces_source_residual_exactly() {
    let f = fixture();
    for case in &f.cases {
        let positions = case.input.shape[1];
        let residual = case.block_input.bf16();
        let sublayer = case.attention_output.bf16();
        let expected = case.after_attention_residual.bf16();
        let post = case.attention_coefficients.post.fp32();
        let comb = case.attention_coefficients.comb.fp32();
        for position in 0..positions {
            let mut actual = vec![0; 256];
            hc_post_bf16_reference(
                &sublayer[position * 128..(position + 1) * 128],
                &residual[position * 256..(position + 1) * 256],
                &post[position * 2..(position + 1) * 2],
                &comb[position * 4..(position + 1) * 4],
                &mut actual,
            )
            .unwrap();
            assert_eq!(
                actual,
                expected[position * 256..(position + 1) * 256],
                "captured attention post start {} position {position}",
                case.start_pos
            );
        }
    }
}

#[test]
fn captured_attention_hc_pre_reproduces_source_ffn_collapse_exactly() {
    let f = fixture();
    for case in &f.cases {
        let positions = case.input.shape[1];
        let residual = case.after_attention_residual.bf16();
        let expected = case.ffn_collapsed.bf16();
        let pre = case.attention_coefficients.pre.fp32();
        for position in 0..positions {
            let mut actual = vec![0; 128];
            hc_pre_bf16_reference(
                &residual[position * 256..(position + 1) * 256],
                &pre[position * 2..(position + 1) * 2],
                128,
                &mut actual,
            )
            .unwrap();
            assert_eq!(
                actual,
                expected[position * 128..(position + 1) * 128],
                "captured attention pre start {} position {position}",
                case.start_pos
            );
        }
    }
}

#[test]
fn attention_hc_coefficients_isolate_split_before_projection() {
    let f = fixture();
    let c = &f.block_config;
    let scale: [f32; 3] = f.block_parameters["layers.4.hc_attn_scale"]
        .fp32()
        .try_into()
        .unwrap();
    let base = f.block_parameters["layers.4.hc_attn_base"].fp32();
    let projection = f.block_parameters["layers.4.hc_attn_fn"].fp32();
    let parameters = AttentionHcParameters {
        scale: &scale,
        base: &base,
        projection: &projection,
        config: c,
    };
    let mut differences = Vec::new();
    for case in &f.cases {
        for position in 0..case.input.shape[1] {
            check_attention_hc_position(case, position, &parameters, &mut differences);
        }
    }
    if !differences.is_empty() {
        eprintln!(
            "attention HC exact-bit diagnostic observed {} differences:\n{}",
            differences.len(),
            differences.join("\n")
        );
    }
}

struct AttentionHcParameters<'a> {
    scale: &'a [f32; 3],
    base: &'a [f32],
    projection: &'a [f32],
    config: &'a BlockConfig,
}

fn check_attention_hc_position(
    case: &Case,
    position: usize,
    parameters: &AttentionHcParameters<'_>,
    differences: &mut Vec<String>,
) {
    let residual = case.block_input.bf16();
    let mixes = case.attention_hc_mixes.fp32();
    let expected_pre = case.attention_coefficients.pre.fp32();
    let expected_post = case.attention_coefficients.post.fp32();
    let expected_comb = case.attention_coefficients.comb.fp32();
    let context = format!(
        "attention HC coefficients start {} position {position}",
        case.start_pos
    );
    let split = split_hc_coefficients(
        &mixes[position * 8..(position + 1) * 8],
        parameters.scale,
        parameters.base,
        parameters.config.copies,
        parameters.config.hc_sinkhorn_iters,
        parameters.config.hc_eps,
    )
    .unwrap();
    differences.extend(coefficient_bit_differences(
        &split,
        &expected_pre[position * 2..(position + 1) * 2],
        &expected_post[position * 2..(position + 1) * 2],
        &expected_comb[position * 4..(position + 1) * 4],
        &format!("{context} source mixes split"),
    ));
    let projected = project_hc_coefficients(
        &residual[position * 256..(position + 1) * 256],
        parameters.projection,
        parameters.scale,
        parameters.base,
        parameters.config.copies,
        parameters.config.norm_eps,
        parameters.config.hc_sinkhorn_iters,
        parameters.config.hc_eps,
    )
    .unwrap();
    differences.extend(coefficient_bit_differences(
        &projected,
        &expected_pre[position * 2..(position + 1) * 2],
        &expected_post[position * 2..(position + 1) * 2],
        &expected_comb[position * 4..(position + 1) * 4],
        &format!("{context} scalar projection"),
    ));
    if case.start_pos == 6 && position == 0 {
        diagnose_decode_hc_position(case, position, parameters, &split, &projected, differences);
    }
}

fn diagnose_decode_hc_position(
    case: &Case,
    position: usize,
    parameters: &AttentionHcParameters<'_>,
    split: &HcCoefficients,
    projected: &HcCoefficients,
    differences: &mut Vec<String>,
) {
    let residual = case.block_input.bf16();
    let attention = case.attention_output.bf16();
    let expected_residual = case.after_attention_residual.bf16();
    let expected_collapse = case.ffn_collapsed.bf16();
    let mixes = case.attention_hc_mixes.fp32();
    let expected_pre = case.attention_coefficients.pre.fp32();
    let expected_post = case.attention_coefficients.post.fp32();
    let expected_comb = case.attention_coefficients.comb.fp32();
    let residual = &residual[position * 256..(position + 1) * 256];
    let attention = &attention[position * 128..(position + 1) * 128];
    let expected_residual = &expected_residual[position * 256..(position + 1) * 256];
    let expected_collapse = &expected_collapse[position * 128..(position + 1) * 128];
    let captured_pre = &expected_pre[position * 2..(position + 1) * 2];
    let captured_post = &expected_post[position * 2..(position + 1) * 2];
    let captured_comb = &expected_comb[position * 4..(position + 1) * 4];
    let mut native_residual = vec![0; 256];
    hc_post_bf16_reference(
        attention,
        residual,
        projected.post(),
        projected.comb(),
        &mut native_residual,
    )
    .unwrap();
    append_bf16_differences(
        differences,
        "decode start 6 scalar projection attention residual",
        &native_residual,
        expected_residual,
    );

    let reciprocal_post: Vec<f32> = (0..parameters.config.copies)
        .map(|copy| {
            let affine = mixes[position * 8 + parameters.config.copies + copy]
                * parameters.scale[1]
                + parameters.base[parameters.config.copies + copy];
            2.0 / (1.0 + (-affine).exp())
        })
        .collect();
    eprintln!(
        "decode start 6 sigmoid-form probe: scalar post={:?}; reciprocal post={reciprocal_post:?}",
        projected.post(),
    );
    for (label, comb) in [
        ("reciprocal raw post plus captured comb", captured_comb),
        ("reciprocal raw post plus projected comb", projected.comb()),
    ] {
        report_residual_mismatches(
            label,
            attention,
            residual,
            expected_residual,
            &reciprocal_post,
            comb,
        );
    }
    let state = DecodeHcState {
        attention,
        residual,
        expected_residual,
        expected_collapse,
        captured_pre,
        captured_post,
        captured_comb,
        native_residual: &native_residual,
    };
    diagnose_decode_coefficients(&state, split, projected, differences);
}

struct DecodeHcState<'a> {
    attention: &'a [u16],
    residual: &'a [u16],
    expected_residual: &'a [u16],
    expected_collapse: &'a [u16],
    captured_pre: &'a [f32],
    captured_post: &'a [f32],
    captured_comb: &'a [f32],
    native_residual: &'a [u16],
}

fn diagnose_decode_coefficients(
    state: &DecodeHcState<'_>,
    split: &HcCoefficients,
    projected: &HcCoefficients,
    differences: &mut Vec<String>,
) {
    for (label, coefficients) in [
        ("source-mix scalar split", split),
        ("scalar projection", projected),
    ] {
        report_comb_differences(label, coefficients.comb(), state.captured_comb);
        report_residual_mismatches(
            &format!("{label} comb plus captured post"),
            state.attention,
            state.residual,
            state.expected_residual,
            state.captured_post,
            coefficients.comb(),
        );
        append_collapse_differences(
            differences,
            label,
            state.expected_residual,
            coefficients.pre(),
            state.expected_collapse,
            state.captured_pre,
        );
    }
    report_residual_mismatches(
        "source-mix scalar split post plus captured comb",
        state.attention,
        state.residual,
        state.expected_residual,
        split.post(),
        state.captured_comb,
    );
    for (label, pre) in [
        ("captured pre", state.captured_pre),
        ("scalar projection pre", projected.pre()),
    ] {
        append_collapse_differences(
            differences,
            &format!("native attention residual with {label}"),
            state.native_residual,
            pre,
            state.expected_collapse,
            state.captured_pre,
        );
    }
}

fn append_bf16_differences(
    differences: &mut Vec<String>,
    label: &str,
    actual: &[u16],
    expected: &[u16],
) {
    for (feature, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if actual != expected {
            differences.push(format!("{label}[{feature}]: {actual} != {expected}"));
        }
    }
}

fn report_residual_mismatches(
    label: &str,
    attention: &[u16],
    residual: &[u16],
    expected: &[u16],
    post: &[f32],
    comb: &[f32],
) {
    let mut candidate = vec![0; 256];
    hc_post_bf16_reference(attention, residual, post, comb, &mut candidate).unwrap();
    let mismatches: Vec<_> = candidate
        .iter()
        .zip(expected)
        .enumerate()
        .filter_map(|(feature, (&actual, &expected))| {
            (actual != expected).then_some(format!("{feature}: {actual} != {expected}"))
        })
        .collect();
    eprintln!(
        "decode start 6 {label} residual mismatches: {}{}",
        mismatches.len(),
        if mismatches.is_empty() {
            String::new()
        } else {
            format!(" ({})", mismatches.join(", "))
        }
    );
}

fn report_comb_differences(label: &str, actual: &[f32], expected: &[f32]) {
    let differences: Vec<_> = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter_map(|(index, (&actual, &expected))| {
            (actual.to_bits() != expected.to_bits()).then_some(format!(
                "comb[{index}]={actual:?} ({:#010x}) != {expected:?} ({:#010x})",
                actual.to_bits(),
                expected.to_bits(),
            ))
        })
        .collect();
    eprintln!(
        "decode start 6 {label} comb differences: {}{}",
        differences.len(),
        if differences.is_empty() {
            String::new()
        } else {
            format!(" ({})", differences.join(", "))
        }
    );
}

fn append_collapse_differences(
    differences: &mut Vec<String>,
    label: &str,
    residual: &[u16],
    pre: &[f32],
    expected: &[u16],
    source_pre: &[f32],
) {
    let mut collapse = vec![0; 128];
    hc_pre_bf16_reference(residual, pre, 128, &mut collapse).unwrap();
    for (feature, (&actual, &expected)) in collapse.iter().zip(expected).enumerate() {
        if actual != expected {
            differences.push(format!(
                "decode start 6 {label} collapse[{feature}]: {actual} != {expected}; pre={pre:?} source_pre={source_pre:?}"
            ));
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum BlockControl {
    SourceAttention,
    NativeAttention,
    NativeAttentionZeroed,
    WrongFfnPre,
    ZeroAttention,
}

struct BlockTailParameters {
    attn_projection: Vec<f32>,
    attn_scale: [f32; 3],
    attn_base: Vec<f32>,
    ffn_projection: Vec<f32>,
    ffn_scale: [f32; 3],
    ffn_base: Vec<f32>,
    attn_norm: Vec<u16>,
    ffn_norm: Vec<u16>,
}

struct BlockTailOutput {
    residual: Vec<u16>,
    next_pre: Vec<f32>,
    terminal_envelopes: Option<Vec<hc_chain_bounds::TerminalEnvelope>>,
}

struct BlockTailPosition {
    residual: Vec<u16>,
    next_pre: Vec<f32>,
    terminal_envelope: Option<hc_chain_bounds::TerminalEnvelope>,
}

fn block_tail(f: &Fixture, control: BlockControl, verify_contract: bool) -> Vec<BlockTailOutput> {
    block_tail_from_entries(f, control, verify_contract, None)
}

fn block_tail_from_entries(
    f: &Fixture,
    control: BlockControl,
    verify_contract: bool,
    entries: Option<&[BlockTailOutput]>,
) -> Vec<BlockTailOutput> {
    block_tail_from_entries_with_bundle(f, control, verify_contract, entries, None)
}

fn block_tail_from_supplied_attention(
    f: &Fixture,
    supplied_attention: &[Vec<u16>],
    entries: &[BlockTailOutput],
) -> Vec<BlockTailOutput> {
    assert_eq!(supplied_attention.len(), f.cases.len());
    assert_eq!(entries.len(), f.cases.len());
    validate_block_tail_fixture(f);
    let parameters = block_tail_parameters(f);
    let config = &f.block_config;
    with_model(f, false, |model| {
        let ffn = FfnSublayerReference::new(
            model,
            &parameters.ffn_norm,
            &parameters.ffn_projection,
            &parameters.ffn_scale,
            &parameters.ffn_base,
            config.copies,
            config.norm_eps,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )
        .expect("alternate L4 FFN contract");
        let context = BlockTailContext {
            fixture: f,
            parameters: &parameters,
            ffn: &ffn,
            control: BlockControl::NativeAttention,
            verify_contract: true,
        };
        f.cases
            .iter()
            .zip(supplied_attention)
            .zip(entries)
            .map(|((case, attention), entry)| {
                run_block_tail_case(&context, case, Some(attention), Some(entry))
            })
            .collect()
    })
}

#[derive(Clone, Copy)]
struct LayerFourBundleInputs<'a> {
    bundle: &'a Value,
    publications: &'a [owner_attention_capture::NativeLayerThreePublication],
}

fn block_tail_from_entries_with_bundle(
    f: &Fixture,
    control: BlockControl,
    verify_contract: bool,
    entries: Option<&[BlockTailOutput]>,
    bundle: Option<LayerFourBundleInputs<'_>>,
) -> Vec<BlockTailOutput> {
    if let Some(entries) = entries {
        assert_eq!(entries.len(), f.cases.len());
    }
    validate_block_tail_fixture(f);
    let parameters = block_tail_parameters(f);
    let config = &f.block_config;
    let native_attention = matches!(
        control,
        BlockControl::NativeAttention | BlockControl::NativeAttentionZeroed
    )
    .then(|| native_block_attention_outputs(f, &parameters, entries, bundle));
    with_model(f, false, |model| {
        let ffn = FfnSublayerReference::new(
            model,
            &parameters.ffn_norm,
            &parameters.ffn_projection,
            &parameters.ffn_scale,
            &parameters.ffn_base,
            config.copies,
            config.norm_eps,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )
        .unwrap();
        let context = BlockTailContext {
            fixture: f,
            parameters: &parameters,
            ffn: &ffn,
            control,
            verify_contract,
        };
        f.cases
            .iter()
            .enumerate()
            .map(|(index, case)| {
                run_block_tail_case(
                    &context,
                    case,
                    native_attention
                        .as_ref()
                        .map(|outputs| outputs[index].as_slice()),
                    entries.map(|entries| &entries[index]),
                )
            })
            .collect()
    })
}

fn validate_block_tail_fixture(f: &Fixture) {
    let c = &f.block_config;
    assert_eq!(c.copies, 2);
    assert_eq!(c.hc_sinkhorn_iters, 20);
    assert_eq!(c.norm_eps.to_bits(), 1e-20_f32.to_bits());
    assert_eq!(c.hc_eps.to_bits(), 1e-6_f32.to_bits());
    assert_eq!(f.block_parameters.len(), 8);
    assert_eq!(
        f.comparison_policy.block_next_pre_abs_error_max.to_bits(),
        2.0_f32.powi(-20).to_bits()
    );
}

fn block_tail_parameters(f: &Fixture) -> BlockTailParameters {
    block_tail_parameters_for(f, 4)
}

fn block_tail_parameters_for(f: &Fixture, layer: usize) -> BlockTailParameters {
    let fp32 = |name: &str| f.block_parameters[&format!("layers.{layer}.{name}")].fp32();
    BlockTailParameters {
        attn_projection: fp32("hc_attn_fn"),
        attn_scale: fp32("hc_attn_scale").try_into().unwrap(),
        attn_base: fp32("hc_attn_base"),
        ffn_projection: fp32("hc_ffn_fn"),
        ffn_scale: fp32("hc_ffn_scale").try_into().unwrap(),
        ffn_base: fp32("hc_ffn_base"),
        attn_norm: f.block_parameters[&format!("layers.{layer}.attn_norm.weight")].bf16(),
        ffn_norm: f.block_parameters[&format!("layers.{layer}.ffn_norm.weight")].bf16(),
    }
}

struct BlockTailContext<'a> {
    fixture: &'a Fixture,
    parameters: &'a BlockTailParameters,
    ffn: &'a FfnSublayerReference<'a>,
    control: BlockControl,
    verify_contract: bool,
}

struct BlockCaseData<'a> {
    block_input: &'a [u16],
    incoming: &'a [f32],
    attention: &'a [u16],
    expected_attention_input: &'a [u16],
}

fn run_block_tail_case(
    context: &BlockTailContext<'_>,
    case: &Case,
    attention_override: Option<&[u16]>,
    entry: Option<&BlockTailOutput>,
) -> BlockTailOutput {
    let positions = case.input.shape[1];
    assert_block_case_shapes(case, positions);
    let (block_input, incoming) = block_entry(case, entry);
    let attention =
        attention_override.map_or_else(|| case.attention_output.bf16(), <[u16]>::to_vec);
    assert_eq!(attention.len(), positions * 128);
    let expected_attention_input = case.attention_input.bf16();
    let data = BlockCaseData {
        block_input: &block_input,
        incoming: &incoming,
        attention: &attention,
        expected_attention_input: &expected_attention_input,
    };
    let mut residual = Vec::with_capacity(positions * 256);
    let mut next_pre = Vec::with_capacity(positions * 2);
    let mut terminal_envelopes = Vec::with_capacity(positions);
    for position in 0..positions {
        let result = run_block_tail_position(context, case, position, &data);
        residual.extend(result.residual);
        next_pre.extend(result.next_pre);
        if context.verify_contract {
            terminal_envelopes.push(
                result
                    .terminal_envelope
                    .expect("verified block position has a terminal envelope"),
            );
        }
    }
    BlockTailOutput {
        residual,
        next_pre,
        terminal_envelopes: context.verify_contract.then_some(terminal_envelopes),
    }
}

fn assert_block_case_shapes(case: &Case, positions: usize) {
    assert_eq!(case.block_input.shape, [1, positions, 2, 128]);
    assert_eq!(case.block_output.shape, case.block_input.shape);
    assert_eq!(case.block_incoming_pre.shape, [1, positions, 2]);
    assert_eq!(case.block_next_pre.shape, [1, positions, 2]);
    assert_eq!(case.attention_input.shape, [1, positions, 128]);
    assert_eq!(case.attention_output.shape, case.attention_input.shape);
    assert_eq!(case.ffn_collapsed.shape, case.attention_input.shape);
}

fn run_block_tail_position(
    context: &BlockTailContext<'_>,
    case: &Case,
    position: usize,
    data: &BlockCaseData<'_>,
) -> BlockTailPosition {
    let parameters = context.parameters;
    let config = &context.fixture.block_config;
    let residual = &data.block_input[position * 256..(position + 1) * 256];
    let row = position * 128..(position + 1) * 128;
    assert_attention_input(context, case, position, residual, data, &row);
    let attention_row = if matches!(
        context.control,
        BlockControl::ZeroAttention | BlockControl::NativeAttentionZeroed
    ) {
        &[0; 128][..]
    } else {
        &data.attention[row.clone()]
    };
    let executor = BlockTailReference::new(
        *context.ffn,
        &parameters.attn_projection,
        &parameters.attn_scale,
        &parameters.attn_base,
        config.copies,
        config.norm_eps,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .expect("bounded native block tail");
    let diagnostic = executor
        .forward_token(residual, attention_row)
        .expect("native block tail");
    let attn_coefficients = diagnostic.attention_coefficients();
    let after_attention = diagnostic.after_attention_bf16();
    // Fault injection remains test-owned: ordinary runtime execution always
    // supplies the attention pre-mix to FFN, never FFN's newly derived pre-mix.
    let wrong_result;
    let result = if context.control == BlockControl::WrongFfnPre {
        let own_coefficients = project_hc_coefficients(
            after_attention,
            &parameters.ffn_projection,
            &parameters.ffn_scale,
            &parameters.ffn_base,
            config.copies,
            config.norm_eps,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )
        .unwrap();
        wrong_result = context
            .ffn
            .forward_token(after_attention, own_coefficients.pre())
            .unwrap();
        &wrong_result
    } else {
        diagnostic.ffn()
    };
    let terminal_envelope = if context.verify_contract {
        Some(hc_chain_bounds::check_position(
            context.fixture,
            case,
            position,
            attn_coefficients,
            after_attention,
            result,
        ))
    } else {
        None
    };
    BlockTailPosition {
        residual: result.output_bf16().to_vec(),
        next_pre: result.coefficients().pre().to_vec(),
        terminal_envelope,
    }
}

fn assert_attention_input(
    context: &BlockTailContext<'_>,
    case: &Case,
    position: usize,
    residual: &[u16],
    data: &BlockCaseData<'_>,
    row: &std::ops::Range<usize>,
) {
    let normalized = derive_attention_input(
        residual,
        &data.incoming[position * 2..(position + 1) * 2],
        &context.parameters.attn_norm,
        context.fixture.block_config.norm_eps,
    );
    assert_eq!(
        normalized,
        data.expected_attention_input[row.clone()],
        "attention input start {} position {position}",
        case.start_pos
    );
}

fn derive_attention_input(
    residual: &[u16],
    incoming_pre: &[f32],
    norm_weight: &[u16],
    norm_eps: f32,
) -> Vec<u16> {
    AttentionInput::new(norm_weight, incoming_pre.len(), norm_eps)
        .expect("runtime attention-input operands")
        .forward(residual, incoming_pre)
        .expect("runtime attention-input preparation")
        .normalized_bf16()
        .to_vec()
}

fn block_entry(case: &Case, native: Option<&BlockTailOutput>) -> (Vec<u16>, Vec<f32>) {
    if let Some(native) = native {
        // The HC envelope treats this residual as an exact point. Verify its
        // identity before using the envelope, but retain native operands.
        assert_eq!(
            native.residual,
            case.block_input.bf16(),
            "native entry residual"
        );
        assert_eq!(native.next_pre.len(), case.input.shape[1] * 2);
        (native.residual.clone(), native.next_pre.clone())
    } else {
        (case.block_input.bf16(), case.block_incoming_pre.fp32())
    }
}

fn native_block_attention_outputs(
    f: &Fixture,
    parameters: &BlockTailParameters,
    entries: Option<&[BlockTailOutput]>,
    bundle: Option<LayerFourBundleInputs<'_>>,
) -> Vec<Vec<u16>> {
    let inputs = f
        .cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let positions = case.input.shape[1];
            assert_block_case_shapes(case, positions);
            let (residual, incoming) = block_entry(case, entries.map(|entries| &entries[index]));
            let input: Vec<u16> = (0..positions)
                .flat_map(|position| {
                    derive_attention_input(
                        &residual[position * 256..(position + 1) * 256],
                        &incoming[position * 2..(position + 1) * 2],
                        &parameters.attn_norm,
                        f.block_config.norm_eps,
                    )
                })
                .collect();
            assert_eq!(
                input,
                case.attention_input.bf16(),
                "native layer-four attention input"
            );
            (case.start_pos, input)
        })
        .collect::<Vec<_>>();
    let outputs = if let Some(bundle) = bundle {
        assert_eq!(
            f.source.complete_capture_sha256.as_deref(),
            bundle.bundle["source"]["complete_capture_sha256"].as_str()
        );
        owner_attention_capture::native_outputs_from_bundle_publications(
            &inputs,
            bundle.bundle,
            bundle.publications,
        )
    } else {
        owner_attention_capture::native_outputs_from_ownered_inputs(
            &inputs,
            f.source
                .complete_capture_sha256
                .as_deref()
                .expect("complete source capture for owner replay"),
        )
    };
    assert_eq!(outputs.len(), f.cases.len());
    for (output, case) in outputs.iter().zip(&f.cases) {
        // The HC envelope uses this source tensor as an exact point. Its
        // identity with the actual native output is a prerequisite, not a
        // tolerance or substitution of source values into native execution.
        assert_eq!(
            output,
            &case.attention_output.bf16(),
            "native attention must equal HC contract point at start {}",
            case.start_pos
        );
    }
    outputs
}

#[test]
fn native_block_tail_matches_source_numerical_contract() {
    let f = fixture();
    // Numerical and exact discrete assertions run at each joined boundary.
    let output = block_tail(&f, BlockControl::SourceAttention, true);
    assert_eq!(output.len(), f.cases.len());
}

#[test]
fn native_attention_hc_ffn_chain_matches_source_numerical_contract() {
    let f = fixture();
    let output = block_tail(&f, BlockControl::NativeAttention, true);
    assert_eq!(output.len(), f.cases.len());
}

#[test]
fn native_layer_three_owner_attention_hc_ffn_reaches_layer_four_entry() {
    let f = layer_three_fixture();
    assert_eq!(native_layer_three_block_tail(&f).len(), f.cases.len());
}

#[test]
fn native_layer_one_engram_reaches_layer_two_suffix() {
    let engram_entries = layer1_engram_capture::native_layer_one_block_entries();
    let layer_one = layer1_join::native_layer_one_entries_from_engram_entries(&engram_entries);
    layer2_join::native_layer_two_from_entries(&layer_one);
}

#[test]
fn projection_fed_prefill_reaches_final_reduced_logits() {
    let bundle: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("reduced runner bundle JSON");
    assert_eq!(bundle["schema_version"].as_u64(), Some(1));
    let trace = &bundle["trace"];
    assert_eq!(trace["input_ids"].as_array().map(Vec::len), Some(1));
    assert_eq!(trace["starts"].as_array().map(Vec::len), Some(3));
    let layer_zero_projection = &bundle["projections"]["layer0_to_layer1"];
    let layer_zero = layer_zero::native_layer_zero_entries_from_projection(layer_zero_projection);
    assert_eq!(
        layer_zero
            .iter()
            .map(|(start, _, _)| *start)
            .collect::<Vec<_>>(),
        [0, 5, 6],
        "the projection-fed prefill and decode partitions retain source order"
    );

    // The native layer-zero residual and pre-mix enter Engram1 and layer one.
    // Later weights and attention arithmetic remain fixture-backed, but the
    // layer-three score prefix below is produced from this composed input.
    let streams = layer_zero
        .iter()
        .map(|(start, residual, _)| (*start, residual.clone()))
        .collect::<Vec<_>>();
    let incoming_pre = layer_zero
        .iter()
        .map(|(start, _, pre)| (*start, pre.clone()))
        .collect::<Vec<_>>();
    let mut request = ReducedLiveRequest::new();
    request.step(&streams[0], &incoming_pre[0], None);
    request.step(&streams[1], &incoming_pre[1], None);
    let previous_layer_three_prefix = request.prior_l3_prefix().to_vec();

    // The live L2 and Engram3 prefix produces the complete prior L3
    // publication after start five and before the partial start-six L1 call.
    let mut corrupted_prefix = previous_layer_three_prefix.clone();
    corrupted_prefix[0] ^= 1;
    assert!(
        std::panic::catch_unwind(|| {
            let mut session = ReducedLiveRequest::new();
            session.step(&streams[0], &incoming_pre[0], None);
            session.step(&streams[1], &incoming_pre[1], None);
            session.step(&streams[2], &incoming_pre[2], Some(&corrupted_prefix));
        })
        .is_err(),
        "a changed native previous-call publication must fail the layer-one score oracle"
    );
    request.step(
        &streams[2],
        &incoming_pre[2],
        Some(&previous_layer_three_prefix),
    );
    let _ = request.finish();
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReducedRequestLifecycle {
    Healthy,
    Poisoned,
    Completed,
    Finalized,
}

/// Test-only request cursor. A panicking numerical/source assertion leaves it
/// poisoned; this deliberately does not claim rollback of its component state.
struct ReducedLiveRequest {
    l1: layer1_join::NativeLayerOneSession,
    l2: layer2_join::NativeLayerTwoSession,
    engram3: engram_capture::NativeLayerThreeEngramSession,
    l3_fixture: Fixture,
    bundle: Value,
    l3: Option<owner_attention_capture::NativeLayerThreePublisher>,
    l2_entries: Vec<(usize, Vec<u16>, Vec<f32>)>,
    engram3_entries: Vec<(usize, Vec<u16>)>,
    lifecycle: ReducedRequestLifecycle,
    cursor: usize,
}

impl ReducedLiveRequest {
    fn new() -> Self {
        let bundle: Value = serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
        ))
        .expect("unified reduced bundle");
        Self {
            l1: layer1_join::NativeLayerOneSession::from_bundle(&bundle),
            l2: layer2_join::NativeLayerTwoSession::from_bundle(&bundle),
            engram3: engram_capture::NativeLayerThreeEngramSession::from_bundle(&bundle),
            l3_fixture: layer_three_fixture_from_bundle(&bundle),
            bundle,
            l3: None,
            l2_entries: Vec::new(),
            engram3_entries: Vec::new(),
            lifecycle: ReducedRequestLifecycle::Healthy,
            cursor: 0,
        }
    }

    fn step(
        &mut self,
        stream: &(usize, Vec<u16>),
        pre: &(usize, Vec<f32>),
        prior_l3: Option<&[u16]>,
    ) {
        assert_eq!(
            self.lifecycle,
            ReducedRequestLifecycle::Healthy,
            "poisoned reduced request rejects step"
        );
        assert_eq!(stream.0, [0, 5, 6][self.cursor], "reduced request cursor");
        self.lifecycle = ReducedRequestLifecycle::Poisoned;
        if self.cursor == 2 {
            let publication = self
                .l3
                .as_ref()
                .expect("preceding live L3 publisher")
                .publications()
                .last()
                .expect("preceding committed L3 publication")
                .publication;
            self.l1.supply_previous_layer_three_prefix(
                publication,
                prior_l3.expect("start six needs prior L3 prefix"),
            );
        }
        let l1 = self.l1.step(stream, pre);
        let l2 = self.l2.step(&l1, Some(self.l1.last_publication()));
        let engram = self.engram3.step(Some(&(l2.0, l2.1.clone())));
        self.l2_entries.push(l2);
        self.engram3_entries.push(engram);
        self.cursor += 1;
        if self.cursor == 2 {
            let (publisher, _) = native_previous_layer_three_publisher(
                &self.l2_entries,
                &self.engram3_entries,
                &self.l3_fixture,
                &self.bundle,
            );
            self.l3 = Some(publisher);
        }
        if self.cursor == 3 {
            let pre = self
                .l2_entries
                .iter()
                .map(|(start, _, pre)| (*start, pre.clone()))
                .collect::<Vec<_>>();
            let inputs = native_layer_three_attention_inputs_from_entries(
                &self.l3_fixture,
                &self.engram3_entries,
                &pre,
            );
            self.l3
                .as_mut()
                .expect("L3 publisher before start six")
                .step(&inputs[2]);
        }
        self.lifecycle = if self.cursor == 3 {
            ReducedRequestLifecycle::Completed
        } else {
            ReducedRequestLifecycle::Healthy
        };
    }

    fn prior_l3_prefix(&self) -> &[u16] {
        self.l3
            .as_ref()
            .expect("L3 prefix after start five")
            .previous_call_key_prefix()
    }

    fn restart(&mut self) {
        *self = Self::new();
    }

    fn state_marker(&self) -> (ReducedRequestLifecycle, usize, usize, usize) {
        (
            self.lifecycle,
            self.cursor,
            self.l2_entries.len(),
            self.engram3_entries.len(),
        )
    }

    fn finish(&mut self) -> Vec<BlockTailOutput> {
        assert_eq!(
            self.lifecycle,
            ReducedRequestLifecycle::Completed,
            "only complete traversal may finalize"
        );
        self.lifecycle = ReducedRequestLifecycle::Poisoned;
        let pre = self
            .l2_entries
            .iter()
            .map(|(start, _, pre)| (*start, pre.clone()))
            .collect::<Vec<_>>();
        let third = native_layer_three_block_tail_from_entries_with_attention(
            &self.l3_fixture,
            Some(&self.engram3_entries),
            Some(&pre),
            Some(self.l3.as_ref().expect("complete L3 publisher").outputs()),
        );
        let fourth = moe_fixture_from_bundle(&self.bundle, 4);
        let output = block_tail_from_entries_with_bundle(
            &fourth,
            BlockControl::NativeAttention,
            true,
            Some(&third),
            Some(LayerFourBundleInputs {
                bundle: &self.bundle,
                publications: self
                    .l3
                    .as_ref()
                    .expect("completed L3 publisher")
                    .publications(),
            }),
        );
        assert_eq!(
            self.bundle["projections"]["head"]["schema_version"].as_u64(),
            Some(1)
        );
        let head: HeadFixture = serde_json::from_value(self.bundle["projections"]["head"].clone())
            .expect("unified head fixture");
        assert_final_suffix_with_head(&fourth, &output, &head);
        self.lifecycle = ReducedRequestLifecycle::Finalized;
        output
    }
}

#[test]
fn reduced_l3_bundle_rejects_mixed_capture() {
    let mut bundle: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .unwrap();
    bundle["projections"]["layer3_moe"]["source"]["complete_capture_sha256"] =
        Value::String("0".repeat(64));
    assert!(std::panic::catch_unwind(|| layer_three_fixture_from_bundle(&bundle)).is_err());
}

fn completed_reduced_request() -> ReducedLiveRequest {
    let bundle: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .unwrap();
    let entries = layer_zero::native_layer_zero_entries_from_projection(
        &bundle["projections"]["layer0_to_layer1"],
    );
    let mut request = ReducedLiveRequest::new();
    for (start, residual, pre) in &entries {
        let prefix = (*start == 6).then(|| request.prior_l3_prefix().to_vec());
        request.step(
            &(*start, residual.clone()),
            &(*start, pre.clone()),
            prefix.as_deref(),
        );
    }
    request
}

fn unified_l1_session_fixture() -> (Value, (usize, Vec<u16>, Vec<f32>)) {
    let bundle: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("unified bundle");
    let stream: Tensor = serde_json::from_value(
        bundle["projections"]["layer1_engram"]["cases"][0]["stream"].clone(),
    )
    .unwrap();
    let pre: Tensor = serde_json::from_value(
        bundle["projections"]["layer1_tail"]["cases"][0]["incoming_pre"].clone(),
    )
    .unwrap();
    let input = (0, stream.bf16(), pre.fp32());
    (bundle, input)
}

#[test]
fn unified_l1_rejects_changed_source_metadata() {
    let (bundle, _) = unified_l1_session_fixture();
    for name in [
        "layer1_engram",
        "layer1_owner",
        "layer1_attention",
        "layer1_tail",
    ] {
        let mut changed = bundle.clone();
        changed["projections"][name]["source"]["loader_sha256"] = Value::String("0".repeat(64));
        assert!(
            std::panic::catch_unwind(|| layer1_join::NativeLayerOneSession::from_bundle(&changed))
                .is_err(),
            "changed source accepted: {name}"
        );
    }
}

#[test]
fn unified_l1_consumes_supplied_weights() {
    let (bundle, (start, stream, pre)) = unified_l1_session_fixture();
    let stream = (start, stream);
    let pre = (start, pre);
    for (projection, container, weight) in [
        (
            "layer1_engram",
            "encoded_parameters",
            "layers.1.engram.q_weight",
        ),
        (
            "layer1_owner",
            "encoded_parameters",
            "layers.1.attn.compressor.norm.weight",
        ),
        (
            "layer1_attention",
            "encoded_parameters",
            "layers.1.attn.q_norm.weight",
        ),
        (
            "layer1_tail",
            "block_parameters",
            "layers.1.ffn_norm.weight",
        ),
    ] {
        let mut changed = bundle.clone();
        let tensor = &mut changed["projections"][projection][container][weight];
        let bytes = vec![0_u8; tensor["storage_hex"].as_str().unwrap().len() / 2];
        tensor["storage_hex"] = Value::String("0".repeat(bytes.len() * 2));
        tensor["storage_sha256"] = Value::String(format!("{:x}", Sha256::digest(&bytes)));
        let mut session = layer1_join::NativeLayerOneSession::from_bundle(&changed);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.step(&stream, &pre)))
                .is_err(),
            "changed supplied weight bypassed: {projection}"
        );
    }
}

#[test]
#[should_panic(expected = "bundle layer-one owner requires a live previous-layer-three prefix")]
fn unified_l1_requires_live_prior_l3_prefix() {
    let (bundle, _) = unified_l1_session_fixture();
    let mut owner = layer1_owner_capture::NativeLayerOneOwnerSession::from_bundle(&bundle);
    for case in bundle["projections"]["layer1_owner"]["cases"]
        .as_array()
        .unwrap()
    {
        let input: Tensor = serde_json::from_value(case["input"].clone()).unwrap();
        owner.step_with_input(&input.bf16());
    }
}

fn unified_l2_session_fixture() -> (Value, (usize, Vec<u16>, Vec<f32>)) {
    let bundle: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("unified bundle");
    let case = &bundle["projections"]["layer2_hc"]["cases"][0];
    let residual: Tensor = serde_json::from_value(case["residual"].clone()).unwrap();
    let pre: Tensor = serde_json::from_value(case["incoming_pre"].clone()).unwrap();
    let input = (0, residual.bf16(), pre.fp32());
    (bundle, input)
}

#[test]
fn unified_l2_rejects_changed_source_metadata() {
    let (bundle, _) = unified_l2_session_fixture();
    for name in ["layer2_hc", "layer2_attention", "layer2_ffn"] {
        let mut changed = bundle.clone();
        changed["projections"][name]["source"]["loader_sha256"] = Value::String("0".repeat(64));
        assert!(
            std::panic::catch_unwind(|| layer2_join::NativeLayerTwoSession::from_bundle(&changed))
                .is_err(),
            "changed source accepted: {name}"
        );
    }
}

#[test]
fn unified_l2_consumes_supplied_weights() {
    let (bundle, input) = unified_l2_session_fixture();
    let mut producer = layer1_owner_capture::NativeLayerOneOwnerSession::new(None);
    let publication = producer.step();
    for (projection, container, weight) in [
        ("layer2_hc", "block_parameters", "layers.2.attn_norm.weight"),
        (
            "layer2_attention",
            "encoded_parameters",
            "layers.2.attn.q_norm.weight",
        ),
        ("layer2_ffn", "block_parameters", "layers.2.ffn_norm.weight"),
    ] {
        let mut changed = bundle.clone();
        let tensor = &mut changed["projections"][projection][container][weight];
        let bytes = vec![0_u8; tensor["storage_hex"].as_str().unwrap().len() / 2];
        tensor["storage_hex"] = Value::String("0".repeat(bytes.len() * 2));
        tensor["storage_sha256"] = Value::String(format!("{:x}", Sha256::digest(&bytes)));
        let mut session = layer2_join::NativeLayerTwoSession::from_bundle(&changed);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                || session.step(&input, Some(&publication))
            ))
            .is_err(),
            "changed supplied weight bypassed: {projection}"
        );
    }
}

#[test]
fn unified_l2_requires_live_l1_publication() {
    let (bundle, input) = unified_l2_session_fixture();
    let mut session = layer2_join::NativeLayerTwoSession::from_bundle(&bundle);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.step(&input, None)))
            .is_err(),
        "bundled L2 must not replay a legacy L1 owner"
    );
}

fn unified_l3_publisher_fixture() -> (Value, Vec<(usize, Vec<u16>)>) {
    let bundle: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("unified bundle");
    let inputs = bundle["projections"]["layer3_attention"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let tensor: Tensor = serde_json::from_value(case["input"].clone()).unwrap();
            (
                usize::try_from(case["start_pos"].as_u64().unwrap()).unwrap(),
                tensor.bf16(),
            )
        })
        .collect();
    (bundle, inputs)
}

#[test]
fn unified_l3_engram_rejects_changed_metadata_and_consumes_weights() {
    let (bundle, _) = unified_l3_publisher_fixture();
    let mut changed = bundle.clone();
    changed["projections"]["layer3_engram"]["source"]["forward_observers_sha256"] =
        Value::String("0".repeat(64));
    assert!(
        std::panic::catch_unwind(
            || engram_capture::NativeLayerThreeEngramSession::from_bundle(&changed)
        )
        .is_err()
    );
    let mut changed = bundle;
    let tensor = &mut changed["projections"]["layer3_engram"]["encoded_parameters"]["layers.3.engram.q_weight"];
    let bytes = vec![0_u8; tensor["storage_hex"].as_str().unwrap().len() / 2];
    tensor["storage_hex"] = Value::String("0".repeat(bytes.len() * 2));
    tensor["storage_sha256"] = Value::String(format!("{:x}", Sha256::digest(&bytes)));
    let mut session = engram_capture::NativeLayerThreeEngramSession::from_bundle(&changed);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.step(None))).is_err(),
        "changed supplied Engram3 weight bypassed"
    );
}

#[test]
fn unified_moe_rejects_changed_loader_provenance() {
    let (bundle, _) = unified_l3_publisher_fixture();
    for layer in [3, 4] {
        let mut changed = bundle.clone();
        changed["projections"][format!("layer{layer}_moe")]["source"]["loader_sha256"] =
            Value::String("0".repeat(64));
        assert!(
            std::panic::catch_unwind(|| moe_fixture_from_bundle(&changed, layer)).is_err(),
            "changed layer {layer} loader accepted"
        );
    }
}

#[test]
fn unified_l3_publisher_rejects_changed_source_metadata() {
    let (bundle, _) = unified_l3_publisher_fixture();
    for name in [
        "layer3_attention",
        "layer3_index_key",
        "layer3_compressor",
        "layer3_candidate",
    ] {
        let mut changed = bundle.clone();
        changed["projections"][name]["source"]["forward_observers_sha256"] =
            Value::String("0".repeat(64));
        assert!(
            std::panic::catch_unwind(|| {
                owner_attention_capture::NativeLayerThreePublisher::from_bundle(&changed)
            })
            .is_err(),
            "changed source accepted: {name}"
        );
    }
}

#[test]
fn unified_l3_publisher_consumes_supplied_weights() {
    let (bundle, inputs) = unified_l3_publisher_fixture();
    for (projection, container, weight) in [
        ("layer3_index_key", "weights", "norm"),
        ("layer3_compressor", "weights", "norm"),
        (
            "layer3_candidate",
            "encoded_parameters",
            "layers.3.attn.q_norm.weight",
        ),
        (
            "layer3_attention",
            "encoded_parameters",
            "layers.3.attn.q_norm.weight",
        ),
    ] {
        let mut changed = bundle.clone();
        let tensor = &mut changed["projections"][projection][container][weight];
        let bytes = vec![0_u8; tensor["storage_hex"].as_str().unwrap().len() / 2];
        tensor["storage_hex"] = Value::String("0".repeat(bytes.len() * 2));
        tensor["storage_sha256"] = Value::String(format!("{:x}", Sha256::digest(&bytes)));
        let mut publisher =
            owner_attention_capture::NativeLayerThreePublisher::from_bundle(&changed);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| publisher.step(&inputs[0])))
                .is_err(),
            "changed supplied weight bypassed: {projection}"
        );
        assert!(
            publisher.publications().is_empty(),
            "failed call published: {projection}"
        );
    }
}

#[test]
fn unified_l3_publisher_reset_preserves_operands_and_advances_epoch() {
    let (bundle, inputs) = unified_l3_publisher_fixture();
    let mut publisher = owner_attention_capture::NativeLayerThreePublisher::from_bundle(&bundle);
    for input in &inputs {
        publisher.step(input);
    }
    let original = publisher.publications().to_vec();
    let outputs = publisher.outputs().to_vec();
    publisher.reset_and_retry(&inputs[0]);
    for input in &inputs[1..] {
        publisher.step(input);
    }
    assert_eq!(publisher.outputs(), outputs);
    assert_eq!(publisher.publications().len(), original.len());
    for (new, old) in publisher.publications().iter().zip(original) {
        assert_eq!(new.publication.epoch(), old.publication.epoch() + 1);
        assert_eq!(new.publication.call_id(), old.publication.call_id());
        assert_eq!(
            new.publication.source_layer(),
            old.publication.source_layer()
        );
        assert_eq!(new.key_prefix, old.key_prefix);
        assert_eq!(new.kv_prefix, old.kv_prefix);
        assert_eq!(new.input, old.input);
    }
}

#[test]
fn reduced_l3_bundle_weight_is_consumed_by_final_suffix() {
    let mut request = completed_reduced_request();
    let weight = request
        .l3_fixture
        .block_parameters
        .get_mut("layers.3.ffn_norm.weight")
        .unwrap();
    weight.storage_hex = "0".repeat(weight.storage_hex.len());
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.finish())).is_err(),
        "changed unified L3 weights must not be replaced by legacy operands"
    );
    assert_eq!(request.lifecycle, ReducedRequestLifecycle::Poisoned);
}

#[test]
fn reduced_l4_bundle_rejects_mixed_owner_and_changed_weights() {
    // Candidate weights are consumed before this finish boundary. Their
    // corruption is checked by unified_l3_publisher_consumes_supplied_weights.
    for defect in [
        "owner_capture",
        "observer_identity",
        "l4_norm",
        "head_weight",
    ] {
        let mut request = completed_reduced_request();
        let projections = &mut request.bundle["projections"];
        match defect {
            "owner_capture" => {
                projections["layer3_compressor"]["source"]["complete_capture_sha256"] =
                    Value::String("0".repeat(64));
            }
            "observer_identity" => {
                projections["layer3_candidate"]["source"]["forward_observers_sha256"] =
                    Value::String("0".repeat(64));
            }
            "l4_norm" => {
                let weight = &mut projections["layer4_moe"]["block_parameters"]["layers.4.ffn_norm.weight"]
                    ["storage_hex"];
                *weight = Value::String("0".repeat(weight.as_str().unwrap().len()));
            }
            "head_weight" => {
                for weight in projections["head"]["weight_fp32_bits"]
                    .as_array_mut()
                    .unwrap()
                {
                    *weight = Value::from(0);
                }
            }
            _ => unreachable!(),
        }
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.finish())).is_err(),
            "unified L4/head defect escaped: {defect}"
        );
        assert_eq!(request.lifecycle, ReducedRequestLifecycle::Poisoned);
    }
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "keep the publication corruption matrix beside its producer-preservation assertions"
)]
fn layer_four_rejects_malformed_committed_layer_three_publications() {
    let mut request = completed_reduced_request();
    let publisher = request.l3.as_ref().unwrap();
    let original = publisher.publications().to_vec();
    assert_eq!(
        original
            .iter()
            .map(|record| record.start_pos)
            .collect::<Vec<_>>(),
        [0, 5, 6]
    );
    let attention_inputs = request.bundle["projections"]["layer4_attention"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let tensor: Tensor = serde_json::from_value(case["input"].clone()).unwrap();
            (
                usize::try_from(case["start_pos"].as_u64().unwrap()).unwrap(),
                tensor.bf16(),
            )
        })
        .collect::<Vec<_>>();
    let outputs = publisher.outputs().to_vec();
    let prefix = publisher.previous_call_key_prefix().to_vec();
    for defect in [
        "missing_call",
        "extra_call",
        "reordered",
        "wrong_start",
        "truncated_row",
        "changed_value",
        "wrong_layer",
        "wrong_epoch",
        "wrong_call",
        "truncated_keys",
        "changed_keys",
        "truncated_kv",
        "changed_kv",
    ] {
        let mut changed = original.clone();
        match defect {
            "missing_call" => {
                changed.pop();
            }
            "extra_call" => changed.push(original[2].clone()),
            "reordered" => changed.swap(1, 2),
            "wrong_start" => changed[2].start_pos = 5,
            "truncated_row" => {
                changed[2].input.pop();
            }
            "changed_value" => changed[2].input[0] ^= 1,
            "wrong_layer" | "wrong_epoch" | "wrong_call" => {
                let id = changed[2].publication;
                changed[2].publication = deepseek::indexer::cache::IndexKeyPublicationId::new(
                    if defect == "wrong_layer" {
                        4
                    } else {
                        id.source_layer()
                    },
                    id.epoch() + u64::from(defect == "wrong_epoch"),
                    id.call_id() + u64::from(defect == "wrong_call"),
                );
            }
            "truncated_keys" => {
                changed[2].key_prefix.pop();
            }
            "changed_keys" => changed[2].key_prefix[0] ^= 1,
            "truncated_kv" => {
                changed[2].kv_prefix.pop();
            }
            "changed_kv" => changed[2].kv_prefix[0] ^= 1,
            _ => unreachable!(),
        }
        assert!(
            std::panic::catch_unwind(|| {
                owner_attention_capture::native_outputs_from_bundle_publications(
                    &attention_inputs,
                    &request.bundle,
                    &changed,
                )
            })
            .is_err(),
            "invalid handoff accepted: {defect}"
        );
        assert_eq!(
            publisher.publications(),
            original,
            "producer publication history after {defect}"
        );
        assert_eq!(
            publisher.outputs(),
            outputs,
            "producer outputs after {defect}"
        );
        assert_eq!(
            publisher.previous_call_key_prefix(),
            prefix,
            "producer key prefix after {defect}"
        );
    }
    let _ = request.finish();
    assert_eq!(request.lifecycle, ReducedRequestLifecycle::Finalized);
}

fn assert_finalization_failure_requires_restart(request: &mut ReducedLiveRequest) {
    // A late suffix failure must invalidate a traversal that already completed.
    // Repairing the operand alone cannot authorize a second finalization attempt.
    let valid_residual = request.engram3_entries[2].1.clone();
    request.engram3_entries[2].1[0] ^= 1;
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.finish())).is_err());
    assert_eq!(request.lifecycle, ReducedRequestLifecycle::Poisoned);
    request.engram3_entries[2].1 = valid_residual;
    let poisoned_finish = request.state_marker();
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.finish())).is_err());
    assert_eq!(request.state_marker(), poisoned_finish);
}

#[test]
fn reduced_request_poison_blocks_retry_and_restart_is_fresh() {
    let bundle: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("reduced runner bundle JSON");
    let entries = layer_zero::native_layer_zero_entries_from_projection(
        &bundle["projections"]["layer0_to_layer1"],
    );
    let streams = entries
        .iter()
        .map(|(start, residual, _)| (*start, residual.clone()))
        .collect::<Vec<_>>();
    let pre = entries
        .iter()
        .map(|(start, _, pre)| (*start, pre.clone()))
        .collect::<Vec<_>>();
    let mut partial = ReducedLiveRequest::new();
    partial.step(&streams[0], &pre[0], None);
    partial.restart();
    partial.step(&streams[0], &pre[0], None);
    partial.step(&streams[1], &pre[1], None);
    let partial_prefix = partial.prior_l3_prefix().to_vec();
    partial.step(&streams[2], &pre[2], Some(&partial_prefix));
    let _ = partial.finish();

    let mut request = ReducedLiveRequest::new();
    request.step(&streams[0], &pre[0], None);
    request.step(&streams[1], &pre[1], None);
    let prefix = request.prior_l3_prefix().to_vec();
    let mut corrupt = prefix.clone();
    corrupt[0] ^= 1;
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.step(
            &streams[2],
            &pre[2],
            Some(&corrupt)
        )))
        .is_err()
    );
    let poisoned = request.state_marker();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.step(
            &streams[2],
            &pre[2],
            Some(&prefix)
        )))
        .is_err()
    );
    assert_eq!(
        request.state_marker(),
        poisoned,
        "poisoned retry must not mutate request"
    );
    request.restart();
    request.step(&streams[0], &pre[0], None);
    request.step(&streams[1], &pre[1], None);
    let fresh_prefix = request.prior_l3_prefix().to_vec();
    request.step(&streams[2], &pre[2], Some(&fresh_prefix));
    assert_eq!(request.lifecycle, ReducedRequestLifecycle::Completed);
    let restarted = request.finish();
    let finalized = request.state_marker();
    assert_eq!(request.lifecycle, ReducedRequestLifecycle::Finalized);
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.finish())).is_err());
    assert_eq!(
        request.state_marker(),
        finalized,
        "repeat finish must not mutate request"
    );
    let mut fresh = ReducedLiveRequest::new();
    fresh.step(&streams[0], &pre[0], None);
    fresh.step(&streams[1], &pre[1], None);
    let fresh_prefix = fresh.prior_l3_prefix().to_vec();
    fresh.step(&streams[2], &pre[2], Some(&fresh_prefix));
    assert_finalization_failure_requires_restart(&mut fresh);
    fresh.restart();
    fresh.step(&streams[0], &pre[0], None);
    fresh.step(&streams[1], &pre[1], None);
    let fresh_prefix = fresh.prior_l3_prefix().to_vec();
    fresh.step(&streams[2], &pre[2], Some(&fresh_prefix));
    let fresh_output = fresh.finish();
    assert_eq!(restarted.len(), fresh_output.len());
    for (restarted, fresh) in restarted.iter().zip(&fresh_output) {
        assert_eq!(
            restarted.residual, fresh.residual,
            "restart matches fresh final L4 output residual"
        );
        assert_eq!(
            restarted.next_pre, fresh.next_pre,
            "restart matches fresh terminal pre"
        );
    }
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| request.step(
            &streams[2],
            &pre[2],
            Some(&fresh_prefix)
        )))
        .is_err()
    );
}

/// Produces only the two preceding layer-three calls that publish the prefix
/// consumed by the following L1 partial decode. The source start-six operands
/// remain outside this bootstrap traversal.
fn bootstrap_layer_three_inputs(
    layer_two: &[(usize, Vec<u16>, Vec<f32>)],
    engram: &[(usize, Vec<u16>)],
    fixture: &Fixture,
) -> Vec<(usize, Vec<u16>)> {
    assert_eq!(layer_two.len(), 2, "two preceding native L2 calls");
    assert_eq!(
        layer_two
            .iter()
            .map(|(start, _, _)| *start)
            .collect::<Vec<_>>(),
        [0, 5],
        "L3 bootstrap sees the already-committed L1 prefix"
    );
    assert_eq!(engram.len(), layer_two.len(), "live L3 Engram prefix count");
    let pre = layer_two
        .iter()
        .map(|(start, _, pre)| (*start, pre.clone()))
        .collect::<Vec<_>>();
    native_layer_three_attention_inputs_from_entries(fixture, engram, &pre)
}

fn native_previous_layer_three_publisher(
    layer_two: &[(usize, Vec<u16>, Vec<f32>)],
    engram: &[(usize, Vec<u16>)],
    fixture: &Fixture,
    bundle: &Value,
) -> (owner_attention_capture::NativeLayerThreePublisher, Vec<u16>) {
    let inputs = bootstrap_layer_three_inputs(layer_two, engram, fixture);
    let mut publisher = owner_attention_capture::NativeLayerThreePublisher::from_bundle(bundle);
    publisher.step(&inputs[0]);
    publisher.step(&inputs[1]);
    let prefix = publisher.previous_call_key_prefix().to_vec();
    (publisher, prefix)
}

#[test]
#[should_panic(expected = "native Engram layer-one residual at block boundary")]
fn native_layer_one_engram_rejects_corrupted_block_entry() {
    let mut entries = layer1_engram_capture::native_layer_one_block_entries();
    entries[0].1[0] ^= 1;
    layer1_join::native_layer_one_entries_from_engram_entries(&entries);
}

#[test]
fn native_layer_three_engram_through_final_suffix_matches_source_logits() {
    let f = layer_three_fixture();
    let entries = engram_capture::native_layer_three_block_entries();
    let native = native_layer_three_block_tail_from_entries(&f, Some(&entries), None);
    let layer_four = fixture();
    let output = block_tail_from_entries(
        &layer_four,
        BlockControl::NativeAttention,
        true,
        Some(&native),
    );
    assert_final_suffix(&layer_four, &output);
}

#[test]
fn alternate_partition_owner_attention_reaches_layer_four_through_native_layer_three_tail() {
    let f = alternate_partition_tail_fixture();
    let attention = partition_owner::alternate_partition_owner_attention_outputs();
    assert_eq!(
        attention
            .iter()
            .map(|(start, _)| *start)
            .collect::<Vec<_>>(),
        [0, 4, 5, 6],
        "alternate native owner-attention calls"
    );
    let supplied: Vec<_> = attention.into_iter().map(|(_, output)| output).collect();
    let tail =
        native_layer_three_block_tail_from_entries_with_attention(&f, None, None, Some(&supplied));
    for (case, output) in f.cases.iter().zip(&tail) {
        let (residual, _) = source_layer_four_entry(case);
        assert_eq!(output.residual, residual, "alternate L3 terminal residual");
        assert_eq!(
            output.terminal_envelopes.as_ref().map(Vec::len),
            Some(case.input.shape[1]),
            "alternate L3 source layer-four entry envelopes"
        );
    }
}

#[test]
fn alternate_partition_native_l3_to_l4_tail_reaches_final_logits() {
    let l3_fixture = alternate_partition_tail_fixture();
    let publications = partition_owner::alternate_partition_layer_three_publications();
    let l3_attention: Vec<_> = publications
        .iter()
        .map(|publication| publication.attention_output.clone())
        .collect();
    let l3_tail = native_layer_three_block_tail_from_entries_with_attention(
        &l3_fixture,
        None,
        None,
        Some(&l3_attention),
    );
    let l4_fixture = alternate_partition_l4_fixture();
    let l4_attention = alternate_l4_attention_outputs(&l4_fixture, &publications, &l3_tail);
    let l4_tail = block_tail_from_supplied_attention(&l4_fixture.tail, &l4_attention, &l3_tail);
    assert_alternate_head(&l4_fixture.head, &l4_tail);
}

#[test]
fn alternate_partition_rejects_changed_layer_three_attention_hc_projection() {
    let mut f = alternate_partition_tail_fixture();
    let projection = f
        .block_parameters
        .get_mut("layers.3.hc_attn_fn")
        .expect("alternate layer-three attention HC projection");
    projection.storage_hex = "0".repeat(projection.storage_hex.len());
    let attention = partition_owner::alternate_partition_owner_attention_outputs();
    let supplied: Vec<_> = attention.into_iter().map(|(_, output)| output).collect();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            native_layer_three_block_tail_from_entries_with_attention(
                &f,
                None,
                None,
                Some(&supplied),
            )
        }))
        .is_err(),
        "changed captured layer-three attention HC projection must alter native arithmetic"
    );
}

#[test]
fn native_layer_two_ffn_engram_through_final_suffix_matches_source_logits() {
    let layer_two = native_layer_two_entries();
    let streams: Vec<_> = layer_two
        .iter()
        .map(|(start, residual, _)| (*start, residual.clone()))
        .collect();
    let engram_entries =
        engram_capture::native_layer_three_block_entries_from_streams(Some(&streams));
    let incoming_pre: Vec<_> = layer_two
        .iter()
        .map(|(start, _, pre)| (*start, pre.clone()))
        .collect();
    let layer_three = layer_three_fixture();
    let native = native_layer_three_block_tail_from_entries(
        &layer_three,
        Some(&engram_entries),
        Some(&incoming_pre),
    );
    let layer_four = fixture();
    let output = block_tail_from_entries(
        &layer_four,
        BlockControl::NativeAttention,
        true,
        Some(&native),
    );
    assert_final_suffix(&layer_four, &output);
}

#[test]
#[should_panic(expected = "native Engram stream boundary")]
fn native_layer_two_join_rejects_changed_ffn_residual() {
    let layer_two = native_layer_two_entries();
    let streams: Vec<_> = layer_two
        .into_iter()
        .map(|(start, mut residual, _)| {
            residual[0] ^= 1;
            (start, residual)
        })
        .collect();
    engram_capture::native_layer_three_block_entries_from_streams(Some(&streams));
}

#[test]
#[should_panic(expected = "native Engram HC attention input")]
fn native_layer_two_join_rejects_changed_hc_pre() {
    let layer_two = native_layer_two_entries();
    let streams: Vec<_> = layer_two
        .iter()
        .map(|(start, residual, _)| (*start, residual.clone()))
        .collect();
    let engram_entries =
        engram_capture::native_layer_three_block_entries_from_streams(Some(&streams));
    let incoming_pre: Vec<_> = layer_two
        .into_iter()
        .map(|(start, _, mut pre)| {
            pre[0] = 0.0;
            (start, pre)
        })
        .collect();
    native_layer_three_block_tail_from_entries(
        &layer_three_fixture(),
        Some(&engram_entries),
        Some(&incoming_pre),
    );
}

#[test]
#[should_panic(expected = "native Engram layer-three entry")]
fn joined_suffix_rejects_corrupted_engram_entry() {
    let mut entries = engram_capture::native_layer_three_block_entries();
    entries[0].1.fill(0);
    native_layer_three_block_tail_from_entries(&layer_three_fixture(), Some(&entries), None);
}

fn native_layer_three_block_tail(f: &Fixture) -> Vec<BlockTailOutput> {
    native_layer_three_block_tail_from_entries(f, None, None)
}

fn native_layer_three_attention_inputs_from_entries(
    f: &Fixture,
    entries: &[(usize, Vec<u16>)],
    incoming_pre: &[(usize, Vec<f32>)],
) -> Vec<(usize, Vec<u16>)> {
    assert!(
        (1..=f.cases.len()).contains(&entries.len()),
        "native layer-three entry prefix count"
    );
    assert_eq!(
        incoming_pre.len(),
        entries.len(),
        "native layer-two pre prefix count"
    );
    let parameters = block_tail_parameters_for(f, 3);
    f.cases
        .iter()
        .take(entries.len())
        .zip(entries)
        .zip(incoming_pre)
        .map(|((case, (start, residual)), (native_start, pre))| {
            assert_eq!(*start, case.start_pos);
            assert_eq!(*native_start, case.start_pos, "native layer-two pre start");
            assert_eq!(
                *residual,
                case.block_input.bf16(),
                "native Engram layer-three entry"
            );
            assert_eq!(
                pre.len(),
                case.block_incoming_pre.fp32().len(),
                "native layer-two pre width"
            );
            let input: Vec<_> = (0..case.input.shape[1])
                .flat_map(|position| {
                    derive_attention_input(
                        &residual[position * 256..(position + 1) * 256],
                        &pre[position * 2..(position + 1) * 2],
                        &parameters.attn_norm,
                        f.block_config.norm_eps,
                    )
                })
                .collect();
            assert_eq!(
                input,
                case.attention_input.bf16(),
                "native Engram HC attention input"
            );
            (case.start_pos, input)
        })
        .collect()
}

fn native_layer_three_block_tail_from_entries(
    f: &Fixture,
    entries: Option<&[(usize, Vec<u16>)]>,
    incoming_pre: Option<&[(usize, Vec<f32>)]>,
) -> Vec<BlockTailOutput> {
    native_layer_three_block_tail_from_entries_with_attention(f, entries, incoming_pre, None)
}

fn native_layer_three_block_tail_from_entries_with_attention(
    f: &Fixture,
    entries: Option<&[(usize, Vec<u16>)]>,
    incoming_pre: Option<&[(usize, Vec<f32>)]>,
    supplied_attention: Option<&[Vec<u16>]>,
) -> Vec<BlockTailOutput> {
    let parameters = block_tail_parameters_for(f, 3);
    let config = &f.block_config;
    let attention = if let Some(outputs) = supplied_attention {
        assert_eq!(outputs.len(), f.cases.len(), "live L3 attention call count");
        outputs.to_vec()
    } else if let Some(entries) = entries {
        assert_eq!(entries.len(), f.cases.len());
        if let Some(incoming_pre) = incoming_pre {
            assert_eq!(
                incoming_pre.len(),
                f.cases.len(),
                "native layer-two pre count"
            );
        }
        let inputs: Vec<_> = f
            .cases
            .iter()
            .zip(entries)
            .enumerate()
            .map(|(index, (case, (start, residual)))| {
                assert_eq!(*start, case.start_pos);
                assert_eq!(
                    *residual,
                    case.block_input.bf16(),
                    "native Engram layer-three entry"
                );
                let captured_incoming = case.block_incoming_pre.fp32();
                let incoming = if let Some(native_pre) = incoming_pre {
                    let (native_start, pre) = &native_pre[index];
                    assert_eq!(*native_start, case.start_pos, "native layer-two pre start");
                    assert_eq!(
                        pre.len(),
                        captured_incoming.len(),
                        "native layer-two pre width"
                    );
                    pre.as_slice()
                } else {
                    captured_incoming.as_slice()
                };
                let input: Vec<_> = (0..case.input.shape[1])
                    .flat_map(|position| {
                        derive_attention_input(
                            &residual[position * 256..(position + 1) * 256],
                            &incoming[position * 2..(position + 1) * 2],
                            &parameters.attn_norm,
                            config.norm_eps,
                        )
                    })
                    .collect();
                assert_eq!(
                    input,
                    case.attention_input.bf16(),
                    "native Engram HC attention input"
                );
                (*start, input)
            })
            .collect();
        owner_attention_capture::native_layer_three_outputs_from_supplied_inputs(&inputs)
    } else {
        owner_attention_capture::native_layer_three_outputs_from_ownered_inputs()
    };
    assert_eq!(attention.len(), f.cases.len());
    with_model_for(f, 3, false, |model| {
        let ffn = FfnSublayerReference::new(
            model,
            &parameters.ffn_norm,
            &parameters.ffn_projection,
            &parameters.ffn_scale,
            &parameters.ffn_base,
            config.copies,
            config.norm_eps,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )
        .expect("layer-three FFN contract");
        f.cases
            .iter()
            .zip(attention)
            .enumerate()
            .map(|(index, (case, attention_output))| {
                check_layer_three_case(
                    f,
                    &parameters,
                    &ffn,
                    case,
                    &attention_output,
                    entries.map(|entries| entries[index].1.as_slice()),
                )
            })
            .collect()
    })
}

fn source_layer_four_entry(case: &Case) -> (Vec<u16>, Vec<f32>) {
    let next = case
        .next_block_entry
        .as_ref()
        .expect("source layer-four entry");
    let residual = next.residual.bf16();
    let incoming = next.incoming_pre.fp32();
    assert_eq!(
        case.block_output.bf16(),
        residual,
        "source block continuity"
    );
    assert_eq!(
        case.block_next_pre.fp32(),
        incoming,
        "source coefficient continuity"
    );
    (residual, incoming)
}

fn check_layer_three_case(
    f: &Fixture,
    parameters: &BlockTailParameters,
    ffn: &FfnSublayerReference<'_>,
    case: &Case,
    attention_output: &[u16],
    native_entry: Option<&[u16]>,
) -> BlockTailOutput {
    let config = &f.block_config;
    let positions = case.input.shape[1];
    assert_eq!(
        attention_output,
        case.attention_output.bf16(),
        "native layer-three attention"
    );
    let captured_input = case.block_input.bf16();
    let block_input = native_entry.unwrap_or(&captured_input);
    assert_eq!(
        block_input, captured_input,
        "native Engram layer-three entry"
    );
    let expected_after_attention = case.after_attention_residual.bf16();
    let expected_terminal = case.block_output.bf16();
    let expected_pre = case.block_next_pre.fp32();
    let (next_residual, next_incoming) = source_layer_four_entry(case);
    let mut terminal = Vec::with_capacity(expected_terminal.len());
    let mut next_pre = Vec::with_capacity(expected_pre.len());
    let mut envelopes = Vec::with_capacity(positions);
    let executor = BlockTailReference::new(
        *ffn,
        &parameters.attn_projection,
        &parameters.attn_scale,
        &parameters.attn_base,
        config.copies,
        config.norm_eps,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .expect("bounded layer-three block tail");
    for position in 0..positions {
        let residual = &block_input[position * 256..(position + 1) * 256];
        let diagnostic = executor
            .forward_token(
                residual,
                &attention_output[position * 128..(position + 1) * 128],
            )
            .expect("layer-three native block tail");
        let attn_coefficients = diagnostic.attention_coefficients();
        let after_attention = diagnostic.after_attention_bf16();
        assert_eq!(
            after_attention,
            &expected_after_attention[position * 256..(position + 1) * 256],
            "layer-three native attention HC residual at start {} position {position}",
            case.start_pos
        );
        let result = diagnostic.ffn();
        let envelope = hc_chain_bounds::check_position_for(
            f,
            case,
            position,
            attn_coefficients,
            after_attention,
            result,
            3,
        );
        let source_residual = &next_residual[position * 256..(position + 1) * 256];
        let source_pre = &next_incoming[position * 2..(position + 1) * 2];
        assert!(
            envelope.accepts(
                result.output_bf16(),
                result.coefficients().pre(),
                source_residual,
                source_pre,
            ),
            "native layer-three state reaches layer-four entry at start {} position {position}",
            case.start_pos
        );
        if position == 0 {
            reject_zeroed_layer_three_attention(
                ffn,
                residual,
                attn_coefficients,
                &envelope,
                source_residual,
                source_pre,
            );
        }
        terminal.extend_from_slice(result.output_bf16());
        next_pre.extend_from_slice(result.coefficients().pre());
        envelopes.push(envelope);
    }
    assert_eq!(
        terminal, expected_terminal,
        "layer-three native terminal residual at start {}",
        case.start_pos
    );
    BlockTailOutput {
        residual: terminal,
        next_pre,
        terminal_envelopes: Some(envelopes),
    }
}

fn reject_zeroed_layer_three_attention(
    ffn: &FfnSublayerReference<'_>,
    residual: &[u16],
    attn_coefficients: &HcCoefficients,
    envelope: &hc_chain_bounds::TerminalEnvelope,
    source_residual: &[u16],
    source_pre: &[f32],
) {
    let mut zeroed_attention = vec![0; 256];
    hc_post_bf16_reference(
        &[0; 128],
        residual,
        attn_coefficients.post(),
        attn_coefficients.comb(),
        &mut zeroed_attention,
    )
    .expect("zeroed layer-three attention control");
    let wrong = ffn
        .forward_token(&zeroed_attention, attn_coefficients.pre())
        .expect("zeroed-attention FFN control");
    assert!(
        !envelope.accepts(
            wrong.output_bf16(),
            wrong.coefficients().pre(),
            source_residual,
            source_pre,
        ),
        "zeroed attention must fail the layer-four entry envelope"
    );
}

fn agrees_with_head_oracle(actual: &[f32], expected: &[u32], bounds: &[f64]) -> bool {
    actual
        .iter()
        .zip(expected)
        .zip(bounds)
        .all(|((&actual, &expected), &bound)| {
            actual.is_finite()
                && (f64::from(actual) - f64::from(f32::from_bits(expected))).abs() <= bound
        })
}

#[test]
fn native_layer_four_final_suffix_matches_source_logits_with_propagated_input_bounds() {
    let f = fixture();
    let native = block_tail(&f, BlockControl::NativeAttention, true);
    assert_final_suffix(&f, &native);
}

#[test]
fn native_layer_three_through_final_suffix_matches_source_logits() {
    let layer_three = layer_three_fixture();
    let entries = native_layer_three_block_tail(&layer_three);
    let layer_four = fixture();
    assert_eq!(layer_three.cases.len(), layer_four.cases.len());
    for (source, consumer) in layer_three.cases.iter().zip(&layer_four.cases) {
        assert_eq!(
            source.start_pos, consumer.start_pos,
            "cross-capture position"
        );
        let entry = source.next_block_entry.as_ref().unwrap();
        assert_eq!(
            entry.residual.bf16(),
            consumer.block_input.bf16(),
            "cross-capture residual"
        );
        assert_eq!(
            entry.incoming_pre.fp32(),
            consumer.block_incoming_pre.fp32(),
            "cross-capture coefficients"
        );
    }
    let native = block_tail_from_entries(
        &layer_four,
        BlockControl::NativeAttention,
        true,
        Some(&entries),
    );
    assert_final_suffix(&layer_four, &native);
}

#[test]
#[should_panic(expected = "native layer-four attention input")]
fn joined_suffix_rejects_corrupted_layer_three_coefficients() {
    let mut entries = native_layer_three_block_tail(&layer_three_fixture());
    entries[0].next_pre.fill(0.0);
    block_tail_from_entries(
        &fixture(),
        BlockControl::NativeAttention,
        true,
        Some(&entries),
    );
}

fn assert_final_suffix(f: &Fixture, native: &[BlockTailOutput]) {
    assert_final_suffix_with_head(f, native, &head_fixture());
}

fn assert_final_suffix_with_head(f: &Fixture, native: &[BlockTailOutput], head: &HeadFixture) {
    assert_eq!(head.source.revision, f.source.revision);
    assert_eq!(head.source.model_sha256, f.source.model_sha256);
    assert_eq!(
        head.source.complete_capture_sha256,
        f.source
            .complete_capture_sha256
            .as_deref()
            .expect("complete source capture for final suffix")
    );
    assert_eq!(head.cases.len(), f.cases.len());
    assert_eq!(head.weight_shape[1], 128);
    assert_eq!(head.weight_fp32_bits.len(), head.weight_shape[0] * 128);
    assert_eq!(head.norm_weight_bf16.len(), 128);

    let weights: Vec<f32> = head
        .weight_fp32_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    let executor = FinalHead::new(
        &head.norm_weight_bf16,
        &weights,
        head.weight_shape[0],
        2,
        f32::from_bits(head.norm_epsilon_bits),
    )
    .expect("bounded canonical final head");
    assert_eq!(native.len(), f.cases.len());
    for ((case, block), head_case) in f.cases.iter().zip(native).zip(&head.cases) {
        let positions = case.input.shape[1];
        assert_eq!(case.start_pos, head_case.start_pos);
        assert_eq!(head_case.input_shape, [1, positions, 128]);
        assert_eq!(head_case.final_block_shape, [1, positions, 2, 128]);
        assert_eq!(head_case.final_pre_shape, [1, positions, 2]);
        assert_eq!(head_case.logits_shape, [1, head.weight_shape[0]]);
        assert_eq!(head_case.collapsed_bf16.len(), positions * 128);
        assert_eq!(head_case.final_block_bf16, case.block_output.bf16());
        assert_eq!(
            head_case.final_pre_fp32_bits,
            case.block_next_pre
                .fp32()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
        assert_eq!(block.residual.len(), positions * 256);
        assert_eq!(block.next_pre.len(), positions * 2);

        let last = positions - 1;
        let residual = &block.residual[last * 256..(last + 1) * 256];
        let pre = &block.next_pre[last * 2..(last + 1) * 2];
        let output = executor
            .forward(residual, pre)
            .expect("native canonical final head");
        let native_collapsed = output.collapsed_bf16();
        let native_normalized = output.normalized_bf16();
        let source_collapsed = &head_case.collapsed_bf16[last * 128..(last + 1) * 128];
        let source_normalized = &head_case.input_bf16[last * 128..(last + 1) * 128];
        let terminal_envelope = block
            .terminal_envelopes
            .as_ref()
            .and_then(|envelopes| envelopes.get(last))
            .expect("verified native block has one terminal envelope per position");
        let final_envelope = terminal_envelope.final_norm_envelope(
            &head.norm_weight_bf16,
            f32::from_bits(head.norm_epsilon_bits),
        );
        assert!(
            final_envelope.accepts(
                native_collapsed,
                source_collapsed,
                native_normalized,
                source_normalized,
            ),
            "native final HC and RMSNorm rows must stay inside fixed source bounds at start {}",
            case.start_pos
        );
        let zero_normalized = vec![0; 128];
        assert!(
            !final_envelope.accepts(
                native_collapsed,
                source_collapsed,
                &zero_normalized,
                source_normalized,
            ),
            "zeroing final normalization must fail the fixed source bounds at start {}",
            case.start_pos
        );
        let bounds = final_envelope.head_bounds(source_normalized, &weights);
        assert!(
            agrees_with_head_oracle(output.logits(), &head_case.logits_fp32_bits, &bounds,),
            "native layer-four suffix logits at start {}",
            case.start_pos
        );
    }
}

#[test]
#[should_panic(expected = "MoE checkpoint must agree exactly")]
fn joined_contract_rejects_discarded_native_attention_output() {
    block_tail(&fixture(), BlockControl::NativeAttentionZeroed, true);
}

#[test]
fn block_tail_rejects_wrong_hc_handoff_and_unused_attention() {
    let f = fixture();
    for control in [BlockControl::WrongFfnPre, BlockControl::ZeroAttention] {
        let changed = block_tail(&f, control, false);
        assert!(
            f.cases
                .iter()
                .zip(changed)
                .any(|(case, actual)| actual.residual != case.block_output.bf16()),
            "counterexample must change the captured block output"
        );
    }
}

#[test]
#[should_panic(expected = "FFN collapse native")]
fn numerical_contract_rejects_wrong_hc_handoff() {
    block_tail(&fixture(), BlockControl::WrongFfnPre, true);
}

#[test]
#[should_panic(expected = "MoE checkpoint must agree exactly")]
fn numerical_contract_rejects_omitted_attention() {
    block_tail(&fixture(), BlockControl::ZeroAttention, true);
}

fn alternate_layer_one_projection() -> Value {
    let raw = include_str!("../../../../fixtures/deepseek-v41/partition-layer1-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "331f3abfc14c2918e200e9af3de181b1aa619949ace235eb553beb7bc12b6b7f"
    );
    serde_json::from_str(raw).unwrap()
}

#[test]
fn alternate_partition_native_upstream_reaches_layer_one_attention() {
    let projection = alternate_layer_one_projection();
    let startup = layer_zero::alternate_startup_projection();
    let inputs = layer_zero::alternate_layer_one_inputs();
    let mut owner =
        layer1_owner_capture::NativeLayerOneOwnerSession::from_alternate(&projection, &startup);
    // Isolated component qualification: L3 inputs remain source-fed. Retained
    // native snapshots enforce the preceding-call boundary, not a live full graph.
    let publications = partition_owner::alternate_partition_layer_three_publications();
    for _ in 0..2 {
        let mut attention =
            layer1_attention_capture::NativeLayerOneAttentionSession::from_alternate(&projection);
        for (index, input) in inputs.iter().enumerate() {
            if matches!(input.0, 4 | 6) {
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        owner.step_with_input(&input.1);
                    }))
                    .is_err(),
                    "each partial group requires a fresh preceding L3 publication"
                );
                let preceding = &publications[index - 1];
                assert_eq!(preceding.key_prefix.len(), input.0 * 64);
                owner.supply_previous_layer_three_prefix(&preceding.key_prefix);
            }
            let published = owner.step_with_input(&input.1);
            assert_eq!(published.start_pos, input.0);
            assert_eq!(published.latent.is_none(), matches!(input.0, 4 | 6));
            let output = attention.step(input, &published);
            assert_eq!(output.0, input.0);
        }
        owner.restart_request();
    }
}

#[test]
fn alternate_layer_one_query_rejects_changed_native_normalization() {
    let mut projection = alternate_layer_one_projection();
    let norm = &mut projection["query_parameters"]["layers.1.attn.q_norm.weight"];
    let zeros = vec![0; 64];
    norm["storage_hex"] = Value::String("00".repeat(64));
    norm["storage_sha256"] = Value::String(format!("{:x}", Sha256::digest(&zeros)));
    let startup = layer_zero::alternate_startup_projection();
    let inputs = layer_zero::alternate_layer_one_inputs();
    let mut owner =
        layer1_owner_capture::NativeLayerOneOwnerSession::from_alternate(&projection, &startup);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            owner.step_with_input(&inputs[0].1);
        }))
        .is_err(),
        "native query normalization must affect the numerical gate"
    );
}

#[test]
fn alternate_partition_native_layer_one_tail_reaches_layer_two_entry() {
    let projection = alternate_layer_one_projection();
    let startup = layer_zero::alternate_startup_projection();
    let upstream = layer_zero::native_layer_zero_entries_from_projection(&startup);
    let publications = partition_owner::alternate_partition_layer_three_publications();
    let mut layer_one = layer1_join::NativeLayerOneSession::from_alternate(&projection, &startup);
    for (index, (start, residual, pre)) in upstream.iter().enumerate() {
        if matches!(start, 4 | 6) {
            layer_one.supply_previous_layer_three_prefix(
                publications[index - 1].publication,
                &publications[index - 1].key_prefix,
            );
        }
        let (output_start, output, next_pre) =
            layer_one.step(&(*start, residual.clone()), &(*start, pre.clone()));
        assert_eq!(output_start, *start);
        assert_eq!(next_pre.len(), output.len() / 128);
        assert_eq!(layer_one.last_publication().start_pos, *start);
    }
}

#[test]
#[should_panic(expected = "native L1 HC residual boundary")]
fn alternate_layer_one_tail_rejects_a_detached_residual_oracle() {
    let mut projection = alternate_layer_one_projection();
    let tensor = &mut projection["tail"]["cases"][0]["residual"];
    let byte_count = usize::try_from(tensor["numel"].as_u64().unwrap()).unwrap() * 2;
    let zeros = vec![0_u8; byte_count];
    tensor["storage_hex"] = Value::String("00".repeat(byte_count));
    tensor["storage_sha256"] = Value::String(format!("{:x}", Sha256::digest(&zeros)));
    let startup = layer_zero::alternate_startup_projection();
    let upstream = layer_zero::native_layer_zero_entries_from_projection(&startup);
    let mut layer_one = layer1_join::NativeLayerOneSession::from_alternate(&projection, &startup);
    let (start, residual, pre) = &upstream[0];
    layer_one.step(&(*start, residual.clone()), &(*start, pre.clone()));
}

fn alternate_layer_two_projection() -> Value {
    let raw = include_str!("../../../../fixtures/deepseek-v41/partition-layer2-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "8001ee63770dc17301d93936c86ddd1d8e7c6737315a2882d79dc0a2b2f42355"
    );
    serde_json::from_str(raw).unwrap()
}

#[test]
fn alternate_partition_native_layer_one_and_two_reach_layer_three_stream() {
    let one_projection = alternate_layer_one_projection();
    let two_projection = alternate_layer_two_projection();
    let startup = layer_zero::alternate_startup_projection();
    let upstream = layer_zero::native_layer_zero_entries_from_projection(&startup);
    let publications = partition_owner::alternate_partition_layer_three_publications();
    let mut one = layer1_join::NativeLayerOneSession::from_alternate(&one_projection, &startup);
    let mut two =
        layer2_join::NativeLayerTwoSession::from_alternate(&two_projection, &one_projection);
    for (index, (start, residual, pre)) in upstream.iter().enumerate() {
        if matches!(start, 4 | 6) {
            one.supply_previous_layer_three_prefix(
                publications[index - 1].publication,
                &publications[index - 1].key_prefix,
            );
        }
        let entry = one.step(&(*start, residual.clone()), &(*start, pre.clone()));
        let (output_start, residual, pre) = two.step(&entry, Some(one.last_publication()));
        assert_eq!(output_start, *start);
        assert_eq!(pre.len(), residual.len() / 128);
    }
}

#[test]
#[should_panic(expected = "requires a live layer-one publication")]
fn alternate_layer_two_cannot_fall_back_to_a_canonical_owner() {
    let projection = alternate_layer_two_projection();
    let input: Tensor =
        serde_json::from_value(projection["attention"]["cases"][0]["input"].clone()).unwrap();
    let mut attention =
        layer2_attention_capture::NativeLayerTwoAttentionSession::from_alternate(&projection);
    attention.step_with_publication(&(0, input.bf16()), None);
}

#[test]
fn alternate_partition_live_upstream_and_owner_reach_final_logits() {
    let one_projection = alternate_layer_one_projection();
    let two_projection = alternate_layer_two_projection();
    let three_projection = alternate_layer_three_engram_projection();
    let startup = layer_zero::alternate_startup_projection();
    let upstream = layer_zero::native_layer_zero_entries_from_projection(&startup);
    let mut one = layer1_join::NativeLayerOneSession::from_alternate(&one_projection, &startup);
    let mut two =
        layer2_join::NativeLayerTwoSession::from_alternate(&two_projection, &one_projection);
    let mut engram = engram_capture::NativeLayerThreeEngramSession::from_alternate(
        &three_projection,
        &two_projection,
    );
    let mut owner = partition_owner::NativeAlternateLayerThreeSession::new();
    let three_fixture = alternate_partition_tail_fixture();
    let mut entries = Vec::new();
    let mut incoming = Vec::new();
    let mut publications: Vec<partition_owner::AlternateLayerThreePublication> = Vec::new();
    for (index, (start, residual, pre)) in upstream.iter().enumerate() {
        if matches!(start, 4 | 6) {
            one.supply_previous_layer_three_prefix(
                publications[index - 1].publication,
                &publications[index - 1].key_prefix,
            );
        }
        let first = one.step(&(*start, residual.clone()), &(*start, pre.clone()));
        let (two_start, stream, pre) = two.step(&first, Some(one.last_publication()));
        entries.push(engram.step(Some(&(two_start, stream))));
        incoming.push((two_start, pre));
        let inputs =
            native_layer_three_attention_inputs_from_entries(&three_fixture, &entries, &incoming);
        publications.push(owner.step(inputs.last().unwrap()));
    }
    let attention: Vec<_> = publications
        .iter()
        .map(|publication| publication.attention_output.clone())
        .collect();
    let three_tail = native_layer_three_block_tail_from_entries_with_attention(
        &three_fixture,
        Some(&entries),
        Some(&incoming),
        Some(&attention),
    );
    let four_fixture = alternate_partition_l4_fixture();
    let four_attention = alternate_l4_attention_outputs(&four_fixture, &publications, &three_tail);
    let four_tail =
        block_tail_from_supplied_attention(&four_fixture.tail, &four_attention, &three_tail);
    assert_alternate_head(&four_fixture.head, &four_tail);
}

#[test]
fn alternate_layer_three_engram_rejects_bad_stream_then_continues() {
    let projection = alternate_layer_three_engram_projection();
    let two = alternate_layer_two_projection();
    let mut tested =
        engram_capture::NativeLayerThreeEngramSession::from_alternate(&projection, &two);
    let mut control =
        engram_capture::NativeLayerThreeEngramSession::from_alternate(&projection, &two);
    for (index, case) in projection["cases"].as_array().unwrap().iter().enumerate() {
        let start = usize::try_from(case["start_pos"].as_u64().unwrap()).unwrap();
        let stream: Tensor = serde_json::from_value(case["stream"].clone()).unwrap();
        let valid = (start, stream.bf16());
        if index == 1 {
            let mut bad = valid.clone();
            bad.1[0] ^= 1;
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tested.step(Some(&bad))))
                    .is_err()
            );
        }
        assert_eq!(tested.step(Some(&valid)), control.step(Some(&valid)));
    }
}

fn alternate_layer_three_engram_projection() -> Value {
    let raw =
        include_str!("../../../../fixtures/deepseek-v41/partition-layer3-engram-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "7bbd6d11e0906d98e113075a2e9d7dd34aa7175536be30f20ebfb45aed93daa9"
    );
    serde_json::from_str(raw).unwrap()
}

#[test]
fn runtime_block_tail_rejects_invalid_inputs_without_poisoning_operands() {
    let f = fixture();
    let parameters = block_tail_parameters(&f);
    let config = &f.block_config;
    with_model(&f, false, |model| {
        let ffn = FfnSublayerReference::new(
            model,
            &parameters.ffn_norm,
            &parameters.ffn_projection,
            &parameters.ffn_scale,
            &parameters.ffn_base,
            config.copies,
            config.norm_eps,
            config.hc_sinkhorn_iters,
            config.hc_eps,
        )
        .unwrap();
        let make = |copies, iterations, epsilon| {
            BlockTailReference::new(
                ffn,
                &parameters.attn_projection,
                &parameters.attn_scale,
                &parameters.attn_base,
                copies,
                config.norm_eps,
                iterations,
                epsilon,
            )
        };
        assert!(make(1, config.hc_sinkhorn_iters, config.hc_eps).is_err());
        assert!(make(config.copies, 0, config.hc_eps).is_err());
        assert!(make(config.copies, usize::MAX, config.hc_eps).is_err());
        assert!(make(config.copies, config.hc_sinkhorn_iters, f32::NAN).is_err());
        assert!(
            BlockTailReference::new(
                ffn,
                &parameters.attn_projection[..1],
                &parameters.attn_scale,
                &parameters.attn_base,
                config.copies,
                config.norm_eps,
                config.hc_sinkhorn_iters,
                config.hc_eps,
            )
            .is_err()
        );
        let executor = make(config.copies, config.hc_sinkhorn_iters, config.hc_eps).unwrap();
        let residual = f.cases[0].block_input.bf16();
        let attention = f.cases[0].attention_output.bf16();
        let expected = executor
            .forward_token(&residual[..256], &attention[..128])
            .unwrap();
        assert!(
            executor
                .forward_token(&residual[..255], &attention[..128])
                .is_err()
        );
        assert!(
            executor
                .forward_token(&residual[..256], &attention[..127])
                .is_err()
        );
        let mut nonfinite = attention[..128].to_vec();
        nonfinite[0] = 0x7f80;
        assert!(
            executor
                .forward_token(&residual[..256], &nonfinite)
                .is_err()
        );
        let actual = executor
            .forward_token(&residual[..256], &attention[..128])
            .unwrap();
        assert_eq!(actual.ffn(), expected.ffn());
        assert_eq!(
            actual.attention_coefficients(),
            expected.attention_coefficients()
        );
        assert_eq!(
            actual.after_attention_bf16(),
            expected.after_attention_bf16()
        );
    });
}
