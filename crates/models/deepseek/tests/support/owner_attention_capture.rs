//! Shared native owner, candidate-selection, and layer-attention capture path for V4.1.
//!
//! This test joins the production index-query prefix, BF16 scorer, strict
//! selector, candidate producer, atomic compressor/key/KV owner, and attention
//! adapter. Owner/consumer layer inputs remain fixture boundaries; both producer
//! and consumer QR are computed by the native candidate-query adapter.

#![allow(
    dead_code,
    reason = "the layer-three and layer-four integration binaries use disjoint capture controls"
)]

use super::{attention_capture, candidate_capture};

#[path = "candidate_hc_capture.rs"]
mod candidate_hc_capture;

use std::num::NonZeroUsize;

use attention_capture::{
    SOURCE_LAYER, assert_diagnostic, call_frequencies, fixture as attention_fixture,
    forward_with_publication, frequencies, layout as attention_layout,
    weights as attention_weights,
};
use deepseek::{
    attention::layer::{Fp8Projection, LayerAttentionState},
    indexer::{
        cache::IndexKeyPublicationId,
        compressed_kv::{CompressedKvLayout, prepare_compressed_kv},
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{RatioOneCompressedOwner, RatioOneOwnerCall, RatioOneOwnerWeights},
        query::{
            CandidateQueryLayout, CandidateQueryWeights, IndexKeyView, IndexQueryLayout,
            IndexQueryWeights, ScoredQueryError, prepare_scored_query,
        },
        selection::{SelectionCall, SelectionGeometry, select_from_candidates},
    },
    precision::{Fp4ActivationMode, requantize_bf16_activations_e2m1},
    reduced::{LayerThreeCall, LayerThreeConfig, LayerThreeSession},
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const CAPTURE_SHA256: &str = "e27dde6ead409c74f7bb2c9e08d4cd5a2b0cfc3c9505c7d6b8908b1cd78b1cc6";
const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";

fn raw_fixture() -> Value {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-attention-reference.json"
    ))
    .expect("source attention fixture JSON");
    let source = field(&fixture, "source");
    assert_eq!(field(source, "revision").as_str(), Some(REVISION));
    assert_eq!(
        field(source, "complete_capture_sha256").as_str(),
        Some(CAPTURE_SHA256)
    );
    fixture
}

fn compressor_fixture() -> Value {
    let root: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-compressor-reference.json"
    ))
    .expect("supplementary compressor fixture");
    let source = field(&root, "source");
    assert_eq!(field(source, "revision").as_str(), Some(REVISION));
    assert_eq!(
        field(source, "complete_capture_sha256").as_str(),
        Some("2f2ff3f1734f959b33a673773cf6fe9c056fabb06a82531e5465562af5480c39")
    );
    let model = field(&root, "model");
    for (name, expected) in [
        ("batches", 1),
        ("input_dimension", 128),
        ("latent_dimension", 64),
        ("owner_layer", 3),
        ("compression_ratio", 1),
    ] {
        assert_eq!(usize_field(model, name), expected, "compressor {name}");
    }
    assert_eq!(field(model, "norm_epsilon").as_f64(), Some(1e-20));
    root
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value
        .get(name)
        .unwrap_or_else(|| panic!("missing fixture field {name}"))
}

fn usize_field(value: &Value, name: &str) -> usize {
    usize::try_from(field(value, name).as_u64().expect("unsigned fixture value"))
        .expect("fixture value fits usize")
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("source geometry is nonzero")
}

fn shape(tensor: &Value) -> Vec<usize> {
    field(tensor, "shape")
        .as_array()
        .expect("fixture shape")
        .iter()
        .map(|value| {
            usize::try_from(value.as_u64().expect("shape value")).expect("shape fits usize")
        })
        .collect()
}

fn tensor_bytes(tensor: &Value, bytes_per_element: usize) -> Vec<u8> {
    let expected_elements = shape(tensor)
        .into_iter()
        .try_fold(1_usize, usize::checked_mul)
        .expect("fixture shape product fits usize");
    assert_eq!(usize_field(tensor, "numel"), expected_elements);
    let hex = field(tensor, "storage_hex")
        .as_str()
        .expect("fixture hex storage");
    assert!(hex.len().is_multiple_of(2));
    let bytes: Vec<_> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("hex UTF-8"), 16).expect("hex byte")
        })
        .collect();
    assert_eq!(
        bytes.len(),
        expected_elements
            .checked_mul(bytes_per_element)
            .expect("fixture byte length fits usize")
    );
    let digest = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        field(tensor, "storage_sha256").as_str(),
        Some(digest.as_str())
    );
    bytes
}

fn bf16(tensor: &Value) -> Vec<u16> {
    assert_eq!(field(tensor, "dtype").as_str(), Some("torch.bfloat16"));
    tensor_bytes(tensor, 2)
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
        .collect()
}

fn fp8(tensor: &Value) -> Vec<u8> {
    assert!(matches!(
        field(tensor, "dtype").as_str(),
        Some("torch.float8_e4m3fn" | "torch.float8_e8m0fnu")
    ));
    tensor_bytes(tensor, 1)
}

fn bools(tensor: &Value) -> Vec<bool> {
    assert_eq!(field(tensor, "dtype").as_str(), Some("torch.bool"));
    tensor_bytes(tensor, 1)
        .into_iter()
        .map(|value| match value {
            0 => false,
            1 => true,
            _ => panic!("source bool storage must contain only 0 or 1"),
        })
        .collect()
}

fn i32s(tensor: &Value) -> Vec<i32> {
    assert_eq!(field(tensor, "dtype").as_str(), Some("torch.int32"));
    tensor_bytes(tensor, 4)
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().expect("i32 word")))
        .collect()
}

fn source_frequencies(
    root: &Value,
    start: usize,
    positions: usize,
    pairs: usize,
) -> Vec<deepseek::RotaryFrequency> {
    let all = field(root, "frequencies");
    let source = field(all, "fp32_pairs")
        .as_array()
        .expect("frequency values");
    assert_eq!(shape(all)[1], pairs);
    source[start * pairs..(start + positions) * pairs]
        .iter()
        .map(|pair| {
            let pair = pair.as_array().expect("complex pair");
            deepseek::RotaryFrequency::new(
                f32::from_bits(
                    u32::try_from(pair[0].as_u64().expect("real bits")).expect("real bits fit"),
                ),
                f32::from_bits(
                    u32::try_from(pair[1].as_u64().expect("imaginary bits"))
                        .expect("imaginary bits fit"),
                ),
            )
            .expect("finite source frequency")
        })
        .collect()
}

#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "keep independently captured producer and consumer operands explicit"
)]
fn generated_indices(
    root: &Value,
    raw_case: &Value,
    attention_case: &attention_capture::Case,
    layout: IndexQueryLayout,
    weights: IndexQueryWeights<'_>,
    keys: &[u16],
    publication: IndexKeyPublicationId,
    producer_attention_input: &[u16],
    consumer_attention_input: &[u16],
    candidate_projection: Option<(&Value, &str)>,
) -> Vec<i32> {
    let model = field(root, "model");
    let indexer = field(raw_case, "indexer");
    let inputs = field(indexer, "inputs");
    let operations = field(indexer, "operations");
    let start = usize_field(inputs, "start_pos");
    assert_eq!(start, attention_case.start_pos, "cross-capture call start");
    let qr = bf16(field(inputs, "qr"));
    assert_eq!(
        qr,
        attention_case.q_norm_output.bf16(),
        "source indexer QR is source attention QR at start {start}"
    );
    let x = bf16(field(inputs, "x"));
    assert_eq!(
        x,
        attention_case.input.bf16(),
        "source indexer X is attention input at start {start}"
    );
    assert_eq!(consumer_attention_input, x, "native consumer index input");
    let positions = shape(field(inputs, "x"))[1];
    let head_dimension = usize_field(model, "index_head_dim");
    let parameters = field(root, "encoded_parameters");
    let projection_codes = fp8(field(parameters, "layers.4.attn.wq_a.weight"));
    let projection_scales = fp8(field(parameters, "layers.4.attn.wq_a.scale"));
    let norm = bf16(field(parameters, "layers.4.attn.q_norm.weight"));
    let epsilon: f32 = serde_json::from_value(field(model, "norm_eps").clone())
        .expect("source normalization epsilon");
    let scored = prepare_scored_query(
        consumer_attention_input,
        &source_frequencies(
            root,
            start,
            positions,
            usize_field(model, "rope_head_dim") / 2,
        ),
        CandidateQueryWeights {
            wq_a: Fp8Projection {
                codes: &projection_codes,
                scales: &projection_scales,
            },
            q_norm: &norm,
            index: weights,
        },
        CandidateQueryLayout::new(layout, epsilon).expect("consumer QR layout"),
        IndexKeyView::new(keys, nonzero(head_dimension)).expect("consumer index keys"),
    )
    .expect("bounded production index query");
    let prepared = scored.query;
    assert_eq!(
        prepared.wq_a,
        attention_case.wq_a_output.bf16(),
        "consumer projection"
    );
    assert_eq!(prepared.qr, qr, "consumer QR");
    let query = prepared.index;
    assert_eq!(
        query.query_post_fp4,
        bf16(field(operations, "q_after_rope_fp4")),
        "start {start} index Q"
    );
    assert_eq!(
        keys,
        bf16(field(inputs, "shared_index_k_prefix")),
        "native owner prefix"
    );
    let keys_per_position = shape(field(inputs, "shared_index_k_prefix"))[1];
    assert_eq!(
        scored.dot_products,
        bf16(field(operations, "scores_einsum")),
        "consumer dots"
    );
    assert_eq!(
        scored.rectified,
        bf16(field(operations, "scores_after_relu")),
        "consumer rectified scores"
    );
    assert_eq!(
        scored.weighted,
        bf16(field(operations, "scores_weighted_per_head")),
        "consumer weighted scores"
    );
    let score_bits = scored.scores;
    assert_eq!(
        score_bits,
        bf16(field(operations, "scores_after_head_sum")),
        "start {start} head-summed scores"
    );
    let ratio = usize::try_from(
        field(model, "compress_ratios").as_array().expect("ratios")[4]
            .as_u64()
            .expect("ratio"),
    )
    .expect("ratio fits usize");
    let offset = usize_field(inputs, "offset");
    let call = SelectionCall::new(
        publication,
        0,
        SelectionGeometry::new(
            start,
            nonzero(positions),
            nonzero(keys_per_position),
            nonzero(ratio),
            offset,
        )
        .expect("source consumer selection geometry"),
    );
    let candidates = match candidate_projection {
        Some((projection, capture)) => candidate_capture::generated_candidates_from_bundle_input(
            start,
            keys,
            call,
            producer_attention_input,
            projection,
            capture,
        ),
        None => candidate_capture::generated_candidates_from_attention_input(
            start,
            keys,
            call,
            producer_attention_input,
        ),
    };
    assert_eq!(
        candidates.mask(),
        bools(field(inputs, "candidate_mask")),
        "native producer matches historical consumer mask at start {start}"
    );
    let selection = select_from_candidates(
        &score_bits,
        call,
        &candidates,
        usize_field(model, "index_topk"),
    )
    .expect("source selection adapter");
    if let Some(expected) = operations.get("scores_after_causal_mask") {
        assert_eq!(
            selection.causal_scores,
            bf16(expected),
            "consumer causal scores"
        );
    }
    assert_eq!(
        selection.masked_scores,
        bf16(field(operations, "scores_after_candidate_mask")),
        "start {start} masked scores"
    );
    let output = selection.indices;
    assert_eq!(
        output,
        i32s(field(indexer, "output_indices")),
        "start {start} selected IDs"
    );
    assert_eq!(
        i32s(field(raw_case, "compressed_indices")),
        i32s(field(indexer, "output_indices")),
        "start {start} source indexer IDs are the attention publication IDs"
    );
    output
}

/// A rejected scorer request must not make a staged owner publication visible.
fn assert_truncated_staged_score_is_rejected(
    root: &Value,
    raw_case: &Value,
    layout: IndexQueryLayout,
    weights: IndexQueryWeights<'_>,
    keys: &[u16],
) {
    let model = field(root, "model");
    let inputs = field(field(raw_case, "indexer"), "inputs");
    let start = usize_field(inputs, "start_pos");
    let x = bf16(field(inputs, "x"));
    let truncated = &x[..x.len() - 1];
    let positions = shape(field(inputs, "x"))[1];
    let parameters = field(root, "encoded_parameters");
    let projection_codes = fp8(field(parameters, "layers.4.attn.wq_a.weight"));
    let projection_scales = fp8(field(parameters, "layers.4.attn.wq_a.scale"));
    let norm = bf16(field(parameters, "layers.4.attn.q_norm.weight"));
    let epsilon: f32 = serde_json::from_value(field(model, "norm_eps").clone())
        .expect("source normalization epsilon");
    let error = prepare_scored_query(
        truncated,
        &source_frequencies(
            root,
            start,
            positions,
            usize_field(model, "rope_head_dim") / 2,
        ),
        CandidateQueryWeights {
            wq_a: Fp8Projection {
                codes: &projection_codes,
                scales: &projection_scales,
            },
            q_norm: &norm,
            index: weights,
        },
        CandidateQueryLayout::new(layout, epsilon).expect("consumer QR layout"),
        IndexKeyView::new(keys, nonzero(usize_field(model, "index_head_dim")))
            .expect("complete staged key view"),
    )
    .expect_err("shortened source input rejects before staged publication");
    assert!(matches!(
        error,
        ScoredQueryError::InputLength { actual, stride }
            if actual == truncated.len() && stride == usize_field(model, "dim")
    ));
}

struct OwnerAttentionOperands<'a> {
    raw: Value,
    attention: attention_capture::Fixture,
    owner: Value,
    compressor: Value,
    owner_inputs: Vec<(usize, Vec<u16>)>,
    candidate_oracle_inputs: Vec<(usize, Vec<u16>)>,
    candidate_projection: Option<(&'a Value, &'a str)>,
}

#[allow(
    clippy::too_many_lines,
    reason = "the source-capture join keeps owner and consumer evidence together"
)]
fn native_outputs_from_parts(
    supplied_inputs: &[(usize, Vec<u16>)],
    expected_capture_sha256: &str,
    operands: OwnerAttentionOperands<'_>,
) -> Vec<Vec<u16>> {
    let OwnerAttentionOperands {
        raw,
        attention,
        owner,
        compressor,
        owner_inputs,
        candidate_oracle_inputs,
        candidate_projection,
    } = operands;
    assert_eq!(supplied_inputs.len(), attention.cases.len());
    assert_eq!(owner_inputs.len(), attention.cases.len());
    assert_eq!(candidate_oracle_inputs.len(), attention.cases.len());
    let model = field(&raw, "model");
    let parameters = field(&raw, "encoded_parameters");
    let heads = usize_field(model, "index_n_heads");
    let head_dimension = usize_field(model, "index_head_dim");
    let index_layout = IndexQueryLayout::new(
        nonzero(1),
        nonzero(usize_field(model, "dim")),
        nonzero(usize_field(model, "q_lora_rank")),
        nonzero(heads),
        nonzero(head_dimension),
        nonzero(usize_field(model, "rope_head_dim") / 2),
    )
    .expect("source index layout");
    let wq_b_codes = fp8(field(parameters, "layers.4.attn.indexer.wq_b.weight"));
    let wq_b_scales = fp8(field(parameters, "layers.4.attn.indexer.wq_b.scale"));
    let weights_proj = bf16(field(
        parameters,
        "layers.4.attn.indexer.weights_proj.weight",
    ));
    let index_weights = IndexQueryWeights {
        wq_b_codes: &wq_b_codes,
        wq_b_scales: &wq_b_scales,
        weights_proj: &weights_proj,
    };
    assert_eq!(
        attention.cases.len(),
        field(&raw, "cases").as_array().expect("raw cases").len()
    );
    let all_frequencies = frequencies(&attention);
    let attention_weights = attention_weights(&attention.encoded_parameters);
    let mut state = LayerAttentionState::new(attention_layout(&attention.model));
    assert_eq!(
        field(field(&owner, "source"), "complete_capture_sha256").as_str(),
        Some(expected_capture_sha256)
    );
    assert_eq!(
        field(field(&owner, "source"), "revision").as_str(),
        Some(REVISION)
    );
    let owner_model = field(&owner, "model");
    assert_eq!(usize_field(owner_model, "owner_layer"), 3);
    assert_eq!(usize_field(owner_model, "batches"), 1);
    let owner_cases = field(&owner, "cases").as_array().expect("owner calls");
    assert_eq!(owner_cases.len(), attention.cases.len());
    let owner_weights = field(&owner, "weights");
    let wk = bf16(field(owner_weights, "wk"));
    let norm = bf16(field(owner_weights, "norm"));
    let key_layout = IndexKeyLayout::new(nonzero(1), nonzero(64), nonzero(64), nonzero(16), 1e-20)
        .expect("captured key layout");
    assert_eq!(usize_field(owner_model, "key_dimension"), head_dimension);
    let compressor_cases = field(&compressor, "cases")
        .as_array()
        .expect("compressor calls");
    assert_eq!(compressor_cases.len(), owner_cases.len());
    let compressor_weights = field(&compressor, "weights");
    assert_eq!(shape(field(compressor_weights, "wkv")), [64, 128]);
    assert_eq!(shape(field(compressor_weights, "norm")), [64]);
    let wkv = bf16(field(compressor_weights, "wkv"));
    let compressor_norm = bf16(field(compressor_weights, "norm"));
    let weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &norm));
    let mut key_owner = RatioOneCompressedOwner::new(
        key_layout,
        nonzero(128),
        nonzero(usize_field(owner_model, "cache_capacity")),
        3,
        &compressor_norm,
        1e-20,
    )
    .expect("bounded atomic owner");
    let mut wrong_group_detected = false;
    let mut wrong_frequency_detected = false;
    let mut outputs = Vec::with_capacity(attention.cases.len());
    for (call_id, (raw_case, attention_case)) in field(&raw, "cases")
        .as_array()
        .expect("raw cases")
        .iter()
        .zip(&attention.cases)
        .enumerate()
    {
        let (supplied_start, supplied_input) = &supplied_inputs[call_id];
        let (owner_start, owner_input) = &owner_inputs[call_id];
        let (candidate_start, candidate_input) = &candidate_oracle_inputs[call_id];
        assert_eq!(
            *supplied_start, attention_case.start_pos,
            "supplied attention start"
        );
        assert_eq!(
            *owner_start, attention_case.start_pos,
            "derived owner start"
        );
        assert_eq!(
            supplied_input,
            &attention_case.input.bf16(),
            "supplied native attention input"
        );
        let owner_case = &owner_cases[call_id];
        assert_eq!(
            usize_field(owner_case, "start_pos"),
            attention_case.start_pos
        );
        let compressor_case = &compressor_cases[call_id];
        assert_eq!(
            usize_field(compressor_case, "start_pos"),
            attention_case.start_pos
        );
        let positions = shape(field(owner_case, "latent"))[1];
        let captured_owner_input = field(compressor_case, "attention_input");
        assert_eq!(shape(captured_owner_input), [1, positions, 128]);
        assert_eq!(
            owner_input,
            &bf16(captured_owner_input),
            "derived HC owner input"
        );
        assert_eq!(
            *candidate_start, attention_case.start_pos,
            "candidate oracle start"
        );
        assert_eq!(
            owner_input, candidate_input,
            "derived HC input crosses candidate boundary"
        );
        let owner_frequencies = source_frequencies(&raw, attention_case.start_pos, positions, 16);
        let publication =
            IndexKeyPublicationId::new(3, 0, u64::try_from(call_id).expect("call ID"));
        let owner_call = RatioOneOwnerCall::new(
            publication,
            attention_case.start_pos,
            nonzero(positions),
            owner_input,
            &owner_frequencies,
            weights,
        );
        let clean = if call_id == 1 {
            assert_eq!(
                attention_case.start_pos, 5,
                "captured first decode starts at five"
            );
            let key_before = key_owner.key_prefix(0).expect("live key prefix").to_vec();
            let kv_before = key_owner.kv_prefix(0).expect("live KV prefix").to_vec();
            let metadata = (
                key_owner.epoch(),
                key_owner.next_call_id(),
                key_owner.next_position(),
                key_owner.valid_positions(),
            );
            let discarded = key_owner
                .prepare(owner_call)
                .expect("staged decode publication");
            assert_eq!(
                discarded
                    .key_prefix(0)
                    .expect("complete staged decode keys"),
                bf16(field(owner_case, "index_cache_after")),
            );
            assert_eq!(
                discarded.kv_prefix(0).expect("complete staged decode KV"),
                attention_case.compressed_kv.bf16(),
            );
            assert_truncated_staged_score_is_rejected(
                &raw,
                raw_case,
                index_layout,
                index_weights,
                discarded
                    .key_prefix(0)
                    .expect("complete staged decode keys"),
            );
            drop(discarded);
            assert_eq!(
                key_owner.key_prefix(0).expect("discarded key prefix"),
                key_before
            );
            assert_eq!(
                key_owner.kv_prefix(0).expect("discarded KV prefix"),
                kv_before
            );
            assert_eq!(
                (
                    key_owner.epoch(),
                    key_owner.next_call_id(),
                    key_owner.next_position(),
                    key_owner.valid_positions(),
                ),
                metadata,
                "discarded decode transaction is invisible",
            );
            let mut clean_owner = key_owner.clone();
            Some(
                clean_owner
                    .prepare(owner_call)
                    .expect("clean decode transaction")
                    .commit()
                    .expect("clean decode publication"),
            )
        } else {
            None
        };
        let pending = key_owner
            .prepare(owner_call)
            .expect("staged atomic owner call");
        assert_eq!(pending.publication(), publication, "staged source identity");
        let indices = generated_indices(
            &raw,
            raw_case,
            attention_case,
            index_layout,
            index_weights,
            pending.key_prefix(0).expect("complete staged key prefix"),
            pending.publication(),
            owner_input,
            supplied_input,
            candidate_projection,
        );
        assert_eq!(
            pending.kv_prefix(0).expect("complete staged KV prefix"),
            attention_case.compressed_kv.bf16(),
            "complete staged compressed-KV prefix at call {call_id}"
        );
        let prepared = pending.commit().expect("native atomic owner call");
        if let Some(clean) = clean {
            assert_eq!(prepared, clean, "decode retry matches clean publication");
        }
        assert_eq!(
            prepared.owner.projected,
            bf16(field(compressor_case, "projected"))
        );
        assert_eq!(
            prepared.owner.latent,
            bf16(field(compressor_case, "latent"))
        );
        assert_eq!(
            prepared.owner.latent,
            bf16(field(owner_case, "latent")),
            "cross-capture latent gate"
        );
        assert_eq!(
            key_owner.key_prefix(0).expect("published batch zero keys"),
            bf16(field(owner_case, "index_cache_after")),
            "published complete index-key prefix at call {call_id}"
        );
        let compressed = key_owner.kv_prefix(0).expect("batch zero compressed KV");
        assert_eq!(
            compressed,
            attention_case.compressed_kv.bf16(),
            "entire native compressed-KV prefix at call {call_id}"
        );
        let mut wrong_group = vec![0; prepared.compressed_kv.post_rope.len()];
        requantize_bf16_activations_e2m1(
            &prepared.compressed_kv.post_rope,
            positions,
            64,
            Fp4ActivationMode::Index32E8m0,
            &mut wrong_group,
        )
        .expect("wrong-group control");
        wrong_group_detected |= wrong_group != prepared.compressed_kv.post_fp4;
        if attention_case.start_pos != 0 {
            wrong_frequency_detected |= prepare_compressed_kv(
                &prepared.owner.latent,
                &source_frequencies(&raw, 0, positions, 16),
                CompressedKvLayout::new(nonzero(1), nonzero(64), nonzero(16))
                    .expect("source KV layout"),
            )
            .expect("wrong-frequency control")
            .post_fp4
                != prepared.compressed_kv.post_fp4;
        }
        let diagnostic = forward_with_publication(
            &mut state,
            supplied_input,
            attention_case.start_pos,
            0,
            u64::try_from(call_id).expect("three calls"),
            SOURCE_LAYER,
            compressed,
            &indices,
            call_frequencies(&all_frequencies, attention_case),
            attention_weights.borrowed(),
        )
        .expect("generated index publication drives attention");
        assert_eq!(
            diagnostic.qr,
            attention_case.q_norm_output.bf16(),
            "native attention QR at call {call_id}"
        );
        assert_diagnostic(attention_case, &diagnostic);
        outputs.push(diagnostic.final_output);
    }
    assert!(
        wrong_group_detected,
        "oracle distinguishes index-key quantization from KV quantization"
    );
    assert!(
        wrong_frequency_detected,
        "oracle distinguishes decode rotary positions"
    );
    outputs
}

/// Legacy wrapper retaining the independently captured source gates.
pub(super) fn native_outputs_from_ownered_inputs(
    supplied_inputs: &[(usize, Vec<u16>)],
    expected_capture_sha256: &str,
) -> Vec<Vec<u16>> {
    assert_eq!(expected_capture_sha256, CAPTURE_SHA256);
    let raw = raw_fixture();
    let attention = attention_fixture();
    let owner: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
    ))
    .expect("owner fixture");
    let candidate_oracle_inputs = attention
        .cases
        .iter()
        .map(|case| {
            (
                case.start_pos,
                candidate_capture::captured_attention_input(case.start_pos),
            )
        })
        .collect::<Vec<_>>();
    native_outputs_from_parts(
        supplied_inputs,
        expected_capture_sha256,
        OwnerAttentionOperands {
            raw,
            attention,
            owner,
            compressor: compressor_fixture(),
            owner_inputs: candidate_hc_capture::derived_inputs(),
            candidate_oracle_inputs,
            candidate_projection: None,
        },
    )
}

fn publication_bundle_operands(bundle: &Value) -> (Value, Value, &str) {
    let source = field(bundle, "source");
    assert_eq!(field(bundle, "schema_version").as_u64(), Some(1));
    let pinned: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("pinned reduced bundle metadata");
    assert_eq!(source, field(&pinned, "source"), "bundle source metadata");
    assert_eq!(field(source, "revision").as_str(), Some(REVISION));
    let capture = field(source, "complete_capture_sha256")
        .as_str()
        .expect("bundle capture");
    let projections = field(bundle, "projections");
    let projection = |name| {
        let value = field(projections, name).clone();
        let child = field(&value, "source");
        assert_eq!(field(&value, "schema_version").as_u64(), Some(1));
        assert_eq!(
            field(child, "complete_capture_sha256").as_str(),
            Some(capture)
        );
        value
    };
    for name in [
        "layer4_attention",
        "layer3_candidate",
        "layer3_index_key",
        "layer3_compressor",
    ] {
        let value = projection(name);
        assert_eq!(
            field(&value, "source"),
            &pinned["projections"][name]["source"],
            "{name} source metadata"
        );
    }
    (
        projection("layer4_attention"),
        projection("layer3_candidate"),
        capture,
    )
}

fn assert_publication_boundary(
    publication: &NativeLayerThreePublication,
    attention_case: &attention_capture::Case,
    raw_case: &Value,
    candidate: &Value,
    call_id: usize,
) {
    assert_eq!(
        publication.start_pos, attention_case.start_pos,
        "L3 publication start"
    );
    assert_eq!(
        publication.publication.source_layer(),
        SOURCE_LAYER,
        "L3 publication source"
    );
    assert_eq!(publication.publication.epoch(), 0, "L3 publication epoch");
    assert_eq!(
        publication.publication.call_id(),
        u64::try_from(call_id).expect("L4 call"),
        "L3 publication call"
    );
    let indexer = field(raw_case, "indexer");
    let source_inputs = field(indexer, "inputs");
    assert_eq!(
        publication.input,
        bf16(&candidate["cases"][call_id]["attention_input"]),
        "L3 owner input oracle"
    );
    assert_eq!(
        publication.key_prefix,
        bf16(field(source_inputs, "shared_index_k_prefix")),
        "L3 key prefix oracle"
    );
    assert_eq!(
        publication.kv_prefix,
        attention_case.compressed_kv.bf16(),
        "L3 KV prefix oracle"
    );
}

/// Consumes already committed L3 publication snapshots for L4. It never stages
/// or commits another ratio-one owner; only L4's own query and selection run.
pub(super) fn native_outputs_from_bundle_publications(
    supplied_inputs: &[(usize, Vec<u16>)],
    bundle: &Value,
    publications: &[NativeLayerThreePublication],
) -> Vec<Vec<u16>> {
    assert_eq!(
        supplied_inputs.len(),
        publications.len(),
        "L4 publication count"
    );
    let (raw, candidate, capture) = publication_bundle_operands(bundle);
    let attention: attention_capture::Fixture =
        serde_json::from_value(raw.clone()).expect("typed L4 bundle attention");
    let model = field(&raw, "model");
    let parameters = field(&raw, "encoded_parameters");
    let heads = usize_field(model, "index_n_heads");
    let head_dimension = usize_field(model, "index_head_dim");
    let index_layout = IndexQueryLayout::new(
        nonzero(1),
        nonzero(usize_field(model, "dim")),
        nonzero(usize_field(model, "q_lora_rank")),
        nonzero(heads),
        nonzero(head_dimension),
        nonzero(usize_field(model, "rope_head_dim") / 2),
    )
    .expect("L4 index layout");
    let wq_b_codes = fp8(field(parameters, "layers.4.attn.indexer.wq_b.weight"));
    let wq_b_scales = fp8(field(parameters, "layers.4.attn.indexer.wq_b.scale"));
    let weights_proj = bf16(field(
        parameters,
        "layers.4.attn.indexer.weights_proj.weight",
    ));
    let index_weights = IndexQueryWeights {
        wq_b_codes: &wq_b_codes,
        wq_b_scales: &wq_b_scales,
        weights_proj: &weights_proj,
    };
    let all_frequencies = frequencies(&attention);
    let weights = attention_weights(&attention.encoded_parameters);
    let mut state = LayerAttentionState::new(attention_layout(&attention.model));
    let raw_cases = field(&raw, "cases").as_array().expect("L4 source cases");
    assert_eq!(raw_cases.len(), attention.cases.len());
    assert_eq!(raw_cases.len(), 3, "source partition count");
    assert_eq!(
        publications.len(),
        raw_cases.len(),
        "complete publication history"
    );
    let mut outputs = Vec::with_capacity(publications.len());
    for (call_id, (((raw_case, attention_case), (start, input)), publication)) in raw_cases
        .iter()
        .zip(&attention.cases)
        .zip(supplied_inputs)
        .zip(publications)
        .enumerate()
    {
        assert_eq!(*start, attention_case.start_pos, "L4 supplied start");
        assert_publication_boundary(publication, attention_case, raw_case, &candidate, call_id);
        assert_eq!(input, &attention_case.input.bf16(), "L4 supplied input");
        let indices = generated_indices(
            &raw,
            raw_case,
            attention_case,
            index_layout,
            index_weights,
            &publication.key_prefix,
            publication.publication,
            &publication.input,
            input,
            Some((&candidate, capture)),
        );
        let diagnostic = forward_with_publication(
            &mut state,
            input,
            *start,
            0,
            u64::try_from(call_id).expect("L4 call"),
            SOURCE_LAYER,
            &publication.kv_prefix,
            &indices,
            call_frequencies(&all_frequencies, attention_case),
            weights.borrowed(),
        )
        .expect("committed L3 publication drives L4 attention");
        assert_diagnostic(attention_case, &diagnostic);
        outputs.push(diagnostic.final_output);
    }
    outputs
}

/// Runs the layer-three source-attention fixture from its native HC input,
/// native owner publication, and native producer-selected compressed IDs.
///
/// The remaining attention arithmetic stays independently constrained by the
/// narrow source fixture; this closes only the owner/producer publication seam.
pub(super) fn native_layer_three_outputs_from_ownered_inputs() -> Vec<Vec<u16>> {
    let owner_inputs = candidate_hc_capture::derived_inputs();
    native_layer_three_outputs_from_supplied_inputs(&owner_inputs)
}

/// One request-local layer-three execution and the key prefix visible to the
/// following partial layer-one call.
pub(super) struct NativeLayerThreeRun {
    pub(super) outputs: Vec<Vec<u16>>,
    pub(super) previous_call_key_prefix: Vec<u16>,
}

/// One committed L3 producer publication retained only after its attention
/// consumer completed successfully in this request-local trace.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativeLayerThreePublication {
    pub(super) start_pos: usize,
    pub(super) publication: IndexKeyPublicationId,
    pub(super) input: Vec<u16>,
    pub(super) key_prefix: Vec<u16>,
    pub(super) kv_prefix: Vec<u16>,
}

/// Numerical operands for the persistent L3 producer.  The bundle variant is
/// deliberately owned by the request publisher so every later `step` uses the
/// same checked caller-provided projections rather than falling back to a
/// legacy capture while the request is live.
struct BundlePublisherOperands {
    attention: Value,
    owner: Value,
    compressor: Value,
    candidate: Value,
    capture: String,
}

enum PublisherOperands {
    Legacy,
    Bundle(BundlePublisherOperands),
}

/// Test-private request owner for the source-shaped L3 producer and attention.
/// Each `step` advances the same coupled production cache; it never recreates
/// the preceding prefix while a later source partition is processed.
pub(super) struct NativeLayerThreePublisher {
    session: LayerThreeSession,
    next_call: usize,
    outputs: Vec<Vec<u16>>,
    inputs: Vec<(usize, Vec<u16>)>,
    publications: Vec<NativeLayerThreePublication>,
    previous_call_key_prefix: Option<Vec<u16>>,
    operands: PublisherOperands,
}

impl NativeLayerThreePublisher {
    pub(super) fn new() -> Self {
        let owner: Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
        ))
        .expect("owner fixture");
        let compressor = compressor_fixture();
        let attention = attention_capture::layer_three_fixture();
        Self {
            session: Self::session(&owner, &compressor, &attention),
            next_call: 0,
            outputs: Vec::new(),
            inputs: Vec::new(),
            publications: Vec::new(),
            previous_call_key_prefix: None,
            operands: PublisherOperands::Legacy,
        }
    }

    /// Starts the persistent L3 producer from the caller's unified reduced
    /// bundle.  Source metadata is pinned only for identity; all arithmetic
    /// operands retained below come from `bundle`.
    pub(super) fn from_bundle(bundle: &Value) -> Self {
        let operands = Self::bundle_operands(bundle);
        let attention: attention_capture::Fixture =
            serde_json::from_value(operands.attention.clone()).expect("typed bundled L3 attention");
        Self {
            session: Self::session(&operands.owner, &operands.compressor, &attention),
            next_call: 0,
            outputs: Vec::new(),
            inputs: Vec::new(),
            publications: Vec::new(),
            previous_call_key_prefix: None,
            operands: PublisherOperands::Bundle(operands),
        }
    }

    fn session(
        owner: &Value,
        compressor: &Value,
        attention: &attention_capture::Fixture,
    ) -> LayerThreeSession {
        let owner_model = field(owner, "model");
        let compressor_weights = field(compressor, "weights");
        let compressor_norm = bf16(field(compressor_weights, "norm"));
        let key_layout =
            IndexKeyLayout::new(nonzero(1), nonzero(64), nonzero(64), nonzero(16), 1e-20)
                .expect("captured key layout");
        let config = LayerThreeConfig::new(
            key_layout,
            nonzero(128),
            nonzero(usize_field(owner_model, "cache_capacity")),
            attention_layout(&attention.model),
            nonzero(attention.model.window_size),
            nonzero(1),
        )
        .expect("source-shaped layer-three session geometry");
        LayerThreeSession::new(config, 3, &compressor_norm, 1e-20)
            .expect("bounded atomic layer-three session")
    }

    /// A malformed pre-owner input poisons the composed session. Reset is the
    /// only admission path for the same first source call afterward.
    #[expect(
        clippy::too_many_lines,
        reason = "keep admission failure and exact reset replay in one control"
    )]
    fn assert_short_input_poison_and_reset(supplied: &(usize, Vec<u16>)) {
        let raw = raw_fixture();
        let attention = attention_capture::layer_three_fixture();
        let owner: Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
        ))
        .expect("owner fixture");
        let compressor = compressor_fixture();
        let owner_case = &field(&owner, "cases").as_array().expect("owner calls")[0];
        let attention_case = &attention.cases[0];
        assert_eq!(
            supplied.0, attention_case.start_pos,
            "malformed source start"
        );
        assert_eq!(
            supplied.1,
            attention_case.input.bf16(),
            "malformed source input boundary"
        );
        let positions = shape(field(owner_case, "latent"))[1];
        let owner_weights = field(&owner, "weights");
        let compressor_weights = field(&compressor, "weights");
        let wk = bf16(field(owner_weights, "wk"));
        let norm = bf16(field(owner_weights, "norm"));
        let wkv = bf16(field(compressor_weights, "wkv"));
        let weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &norm));
        let frequencies = source_frequencies(&raw, supplied.0, positions, 16);
        let attention_weights =
            attention_capture::weights_for_layer(&attention.encoded_parameters, 3);
        let mut session = Self::session(&owner, &compressor, &attention);
        let malformed = &supplied.1[..supplied.1.len() - 1];
        let malformed_result = candidate_capture::with_source_candidate_projector(|candidate| {
            session.step(LayerThreeCall::new(
                malformed,
                nonzero(positions),
                &frequencies,
                weights,
                candidate,
                attention_weights.borrowed(),
            ))
        });
        assert!(
            matches!(
                malformed_result,
                Err(deepseek::reduced::LayerThreeSessionError::Owner(_))
            ),
            "short producer input must reject at the owner boundary"
        );
        assert!(
            session.is_poisoned(),
            "failed producer call poisons live session"
        );
        assert_eq!(
            session.next_start(),
            supplied.0,
            "rejected owner input advances no position"
        );
        let retry_without_reset = candidate_capture::with_source_candidate_projector(|candidate| {
            session.step(LayerThreeCall::new(
                &supplied.1,
                nonzero(positions),
                &frequencies,
                weights,
                candidate,
                attention_weights.borrowed(),
            ))
        });
        assert!(
            matches!(
                retry_without_reset,
                Err(deepseek::reduced::LayerThreeSessionError::Poisoned)
            ),
            "same first call requires reset after malformed admission"
        );
        session.reset().expect("poisoned session reset");
        assert!(!session.is_poisoned(), "reset clears poisoned session");
        assert_eq!(
            session.next_start(),
            supplied.0,
            "reset restores first start"
        );
        let recovered = candidate_capture::with_source_candidate_projector(|candidate| {
            session.step(LayerThreeCall::new(
                &supplied.1,
                nonzero(positions),
                &frequencies,
                weights,
                candidate,
                attention_weights.borrowed(),
            ))
        })
        .expect("reset admits same first source call");
        assert_eq!(
            recovered.publication().call_id(),
            0,
            "reset restarts publication ordinal"
        );
        assert_eq!(
            recovered.publication().epoch(),
            1,
            "reset advances owner epoch"
        );
        assert_eq!(
            recovered.key_prefix(),
            bf16(field(owner_case, "index_cache_after"))
        );
        assert_eq!(recovered.kv_prefix(), attention_case.compressed_kv.bf16());
        assert_eq!(
            recovered.selected_indices(),
            attention_case.compressed_indices.i32()
        );
        assert_diagnostic(attention_case, recovered.attention());
    }

    fn bundle_operands(bundle: &Value) -> BundlePublisherOperands {
        assert_eq!(field(bundle, "schema_version").as_u64(), Some(1));
        let pinned: Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
        ))
        .expect("pinned reduced bundle metadata");
        let source = field(bundle, "source");
        assert_eq!(source, field(&pinned, "source"), "bundle source metadata");
        assert_eq!(field(source, "revision").as_str(), Some(REVISION));
        let capture = field(source, "complete_capture_sha256")
            .as_str()
            .expect("bundle capture")
            .to_owned();
        let projections = field(bundle, "projections");
        let projection = |name| {
            let value = field(projections, name).clone();
            let child_source = field(&value, "source");
            assert_eq!(field(&value, "schema_version").as_u64(), Some(1));
            assert_eq!(
                child_source,
                field(field(&pinned, "projections"), name)
                    .get("source")
                    .expect("pinned projection source"),
                "{name} source metadata"
            );
            assert_eq!(
                field(child_source, "complete_capture_sha256").as_str(),
                Some(capture.as_str()),
                "{name} capture"
            );
            value
        };
        BundlePublisherOperands {
            attention: projection("layer3_attention"),
            owner: projection("layer3_index_key"),
            compressor: projection("layer3_compressor"),
            candidate: projection("layer3_candidate"),
            capture,
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "keep source-stage assertions alongside the runtime result"
    )]
    pub(super) fn step(&mut self, supplied: &(usize, Vec<u16>)) -> Vec<u16> {
        let (raw, attention, owner, compressor, candidate) = match &self.operands {
            PublisherOperands::Legacy => {
                let owner: Value = serde_json::from_str(include_str!(
                    "../../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
                ))
                .expect("owner fixture");
                (
                    raw_fixture(),
                    attention_capture::layer_three_fixture(),
                    owner,
                    compressor_fixture(),
                    None,
                )
            }
            PublisherOperands::Bundle(operands) => (
                operands.attention.clone(),
                serde_json::from_value(operands.attention.clone())
                    .expect("typed bundled L3 attention"),
                operands.owner.clone(),
                operands.compressor.clone(),
                Some((operands.candidate.clone(), operands.capture.clone())),
            ),
        };
        let attention_case = &attention.cases[self.next_call];
        let owner_case = &field(&owner, "cases").as_array().expect("owner calls")[self.next_call];
        let compressor_case = &field(&compressor, "cases")
            .as_array()
            .expect("compressor calls")[self.next_call];
        let (start, owner_input) = supplied;
        assert_eq!(*start, attention_case.start_pos, "HC attention start");
        assert_eq!(
            owner_input,
            &attention_case.input.bf16(),
            "HC input cross-gate"
        );
        assert_eq!(usize_field(owner_case, "start_pos"), *start, "owner start");
        assert_eq!(
            usize_field(compressor_case, "start_pos"),
            *start,
            "compressor start"
        );
        assert_eq!(
            owner_input,
            &bf16(field(compressor_case, "attention_input")),
            "owner input"
        );
        let positions = shape(field(owner_case, "latent"))[1];
        let owner_weights = field(&owner, "weights");
        let compressor_weights = field(&compressor, "weights");
        let wk = bf16(field(owner_weights, "wk"));
        let norm = bf16(field(owner_weights, "norm"));
        let wkv = bf16(field(compressor_weights, "wkv"));
        let weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &norm));
        let owner_frequencies = source_frequencies(&raw, *start, positions, 16);
        assert_eq!(self.session.next_start(), *start, "live L3 session start");
        let all_attention_frequencies = frequencies(&attention);
        let attention_frequencies = call_frequencies(&all_attention_frequencies, attention_case);
        assert_eq!(
            owner_frequencies, attention_frequencies,
            "owner and attention RoPE frequency boundary"
        );
        let attention_weights =
            attention_capture::weights_for_layer(&attention.encoded_parameters, 3);
        let execute = |projector: deepseek::reduced::CandidateProjector<'_>| {
            self.session.step(LayerThreeCall::new(
                owner_input,
                nonzero(positions),
                &owner_frequencies,
                weights,
                projector,
                attention_weights.borrowed(),
            ))
        };
        let output = if let Some((candidate, capture)) = candidate.as_ref() {
            candidate_capture::with_bundle_candidate_projector(candidate, capture, execute)
        } else {
            candidate_capture::with_source_candidate_projector(execute)
        }
        .expect("native owner, candidate, and attention session step");
        let call = output.candidate().candidates().call();
        let epoch = output.publication().epoch();
        if let Some((candidate, capture)) = candidate.as_ref() {
            candidate_capture::assert_bundle_candidate_projection(
                candidate,
                capture,
                *start,
                output.key_prefix(),
                call,
                owner_input,
                epoch,
                output.candidate(),
            );
        } else {
            candidate_capture::assert_source_candidate_projection(
                *start,
                output.key_prefix(),
                call,
                owner_input,
                epoch,
                output.candidate(),
            );
        }
        let indices = output.selected_indices();
        assert_eq!(
            indices,
            attention_case.compressed_indices.i32(),
            "native producer IDs"
        );
        assert_eq!(
            output.kv_prefix(),
            attention_case.compressed_kv.bf16(),
            "native staged KV prefix"
        );
        assert_eq!(
            output.owner().owner.projected,
            bf16(field(compressor_case, "projected")),
            "native compressor projection"
        );
        assert_eq!(
            output.owner().owner.latent,
            bf16(field(compressor_case, "latent")),
            "native compressor latent"
        );
        assert_eq!(
            output.owner().owner.latent,
            bf16(field(owner_case, "latent")),
            "owner and compressor latent cross-gate"
        );
        assert_eq!(
            output.key_prefix(),
            bf16(field(owner_case, "index_cache_after")),
            "published complete index-key prefix"
        );
        assert_diagnostic(attention_case, output.attention());
        let final_output = output.attention().final_output.clone();
        self.retain_publication(*start, &output, owner_input);
        final_output
    }

    fn retain_publication(
        &mut self,
        start: usize,
        output: &deepseek::reduced::LayerThreeStepOutput,
        owner_input: &[u16],
    ) {
        self.publications.push(NativeLayerThreePublication {
            start_pos: start,
            publication: output.publication(),
            input: owner_input.to_vec(),
            key_prefix: output.key_prefix().to_vec(),
            kv_prefix: output.kv_prefix().to_vec(),
        });
        if self.next_call == 1 {
            assert_eq!(
                output.key_prefix().len(),
                6 * 64,
                "start-five layer-three key prefix"
            );
            self.previous_call_key_prefix = Some(output.key_prefix().to_vec());
        }
        self.next_call += 1;
        self.inputs.push((start, owner_input.to_vec()));
        self.outputs.push(output.attention().final_output.clone());
    }

    pub(super) fn previous_call_key_prefix(&self) -> &[u16] {
        self.previous_call_key_prefix
            .as_deref()
            .expect("start-five publication before partial L1")
    }

    pub(super) fn inputs(&self) -> &[(usize, Vec<u16>)] {
        &self.inputs
    }

    pub(super) fn publications(&self) -> &[NativeLayerThreePublication] {
        &self.publications
    }

    pub(super) fn outputs(&self) -> &[Vec<u16>] {
        &self.outputs
    }

    pub(super) fn reset_and_retry(&mut self, retry: &(usize, Vec<u16>)) {
        self.session.reset().expect("producer and attention reset");
        assert_eq!(
            self.session.next_start(),
            0,
            "reset restarts token position"
        );
        self.next_call = 0;
        self.outputs.clear();
        self.inputs.clear();
        self.publications.clear();
        self.previous_call_key_prefix = None;
        let output = self.step(retry);
        assert_eq!(
            self.inputs.len(),
            1,
            "reset retry keeps one successful input"
        );
        assert_eq!(self.inputs[0], *retry, "reset retry input history");
        assert_eq!(
            self.publications.len(),
            1,
            "reset retry keeps one publication"
        );
        assert_eq!(
            self.publications[0].start_pos, retry.0,
            "reset retry publication start"
        );
        assert!(!output.is_empty(), "reset retry computes attention output");
        assert!(
            !self.publications[0].key_prefix.is_empty(),
            "reset retry key prefix"
        );
        assert_eq!(
            self.publications[0].publication.call_id(),
            0,
            "reset retry call ID"
        );
    }
}

pub(super) fn native_layer_three_outputs_from_supplied_inputs(
    owner_inputs: &[(usize, Vec<u16>)],
) -> Vec<Vec<u16>> {
    native_layer_three_run_from_supplied_inputs(owner_inputs).outputs
}

/// Runs a bounded source sequence through one live request-local owner.
pub(super) fn native_layer_three_run_from_supplied_inputs(
    owner_inputs: &[(usize, Vec<u16>)],
) -> NativeLayerThreeRun {
    assert!(
        (2..=3).contains(&owner_inputs.len()),
        "bounded layer-three source sequence"
    );
    NativeLayerThreePublisher::assert_short_input_poison_and_reset(&owner_inputs[0]);
    let mut publisher = NativeLayerThreePublisher::new();
    for input in owner_inputs {
        publisher.step(input);
    }
    let prefix = publisher.previous_call_key_prefix().to_vec();
    let outputs = publisher.outputs().to_vec();
    publisher.reset_and_retry(&owner_inputs[0]);
    NativeLayerThreeRun {
        outputs,
        previous_call_key_prefix: prefix,
    }
}
