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
    let positions = shape(field(inputs, "x"))[1];
    let head_dimension = usize_field(model, "index_head_dim");
    let parameters = field(root, "encoded_parameters");
    let projection_codes = fp8(field(parameters, "layers.4.attn.wq_a.weight"));
    let projection_scales = fp8(field(parameters, "layers.4.attn.wq_a.scale"));
    let norm = bf16(field(parameters, "layers.4.attn.q_norm.weight"));
    let epsilon: f32 = serde_json::from_value(field(model, "norm_eps").clone())
        .expect("source normalization epsilon");
    let scored = prepare_scored_query(
        &x,
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
    let candidates = candidate_capture::generated_candidates_from_attention_input(
        start,
        keys,
        call,
        producer_attention_input,
    );
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

#[allow(
    clippy::too_many_lines,
    reason = "the source-capture join keeps owner and consumer evidence together"
)]
pub(super) fn native_outputs_from_ownered_inputs(
    supplied_inputs: &[(usize, Vec<u16>)],
    expected_capture_sha256: &str,
) -> Vec<Vec<u16>> {
    let raw = raw_fixture();
    let attention = attention_fixture();
    assert_eq!(expected_capture_sha256, CAPTURE_SHA256);
    assert_eq!(supplied_inputs.len(), attention.cases.len());
    let owner_inputs = candidate_hc_capture::derived_inputs();
    assert_eq!(owner_inputs.len(), attention.cases.len());
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
    let owner: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
    ))
    .expect("owner fixture");
    assert_eq!(
        field(field(&owner, "source"), "complete_capture_sha256").as_str(),
        Some(CAPTURE_SHA256)
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
    let compressor = compressor_fixture();
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
            owner_input,
            &candidate_capture::captured_attention_input(attention_case.start_pos),
            "derived HC input crosses historical candidate boundary"
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

/// Runs the layer-three source-attention fixture from its native HC input,
/// native owner publication, and native producer-selected compressed IDs.
///
/// The remaining attention arithmetic stays independently constrained by the
/// narrow source fixture; this closes only the owner/producer publication seam.
#[allow(
    clippy::too_many_lines,
    reason = "the source owner operands remain explicit"
)]
pub(super) fn native_layer_three_outputs_from_ownered_inputs() -> Vec<Vec<u16>> {
    let raw = raw_fixture();
    let attention = attention_capture::layer_three_fixture();
    let owner_inputs = candidate_hc_capture::derived_inputs();
    let owner: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
    ))
    .expect("owner fixture");
    let owner_model = field(&owner, "model");
    let owner_cases = field(&owner, "cases").as_array().expect("owner calls");
    assert_eq!(owner_cases.len(), attention.cases.len());
    assert_eq!(owner_inputs.len(), attention.cases.len());
    let owner_weights = field(&owner, "weights");
    let wk = bf16(field(owner_weights, "wk"));
    let norm = bf16(field(owner_weights, "norm"));
    let compressor = compressor_fixture();
    let compressor_cases = field(&compressor, "cases")
        .as_array()
        .expect("compressor calls");
    let compressor_weights = field(&compressor, "weights");
    let wkv = bf16(field(compressor_weights, "wkv"));
    let compressor_norm = bf16(field(compressor_weights, "norm"));
    let key_layout = IndexKeyLayout::new(nonzero(1), nonzero(64), nonzero(64), nonzero(16), 1e-20)
        .expect("captured key layout");
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
    let all_frequencies = frequencies(&attention);
    let attention_weights = attention_capture::weights_for_layer(&attention.encoded_parameters, 3);
    let mut state = LayerAttentionState::new(attention_layout(&attention.model));
    let mut outputs = Vec::with_capacity(attention.cases.len());
    for (call_id, ((attention_case, owner_case), compressor_case)) in attention
        .cases
        .iter()
        .zip(owner_cases)
        .zip(compressor_cases)
        .enumerate()
    {
        let (start, owner_input) = &owner_inputs[call_id];
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
        let positions = shape(field(owner_case, "latent"))[1];
        assert_eq!(
            owner_input,
            &bf16(field(compressor_case, "attention_input")),
            "owner input"
        );
        let publication =
            IndexKeyPublicationId::new(3, 0, u64::try_from(call_id).expect("call ID"));
        let owner_frequencies = source_frequencies(&raw, *start, positions, 16);
        let owner_call = RatioOneOwnerCall::new(
            publication,
            *start,
            nonzero(positions),
            owner_input,
            &owner_frequencies,
            weights,
        );
        let pending = key_owner
            .prepare(owner_call)
            .expect("staged owner publication");
        let keys = pending.key_prefix(0).expect("complete staged key prefix");
        let call = SelectionCall::new(
            pending.publication(),
            0,
            SelectionGeometry::new(
                *start,
                nonzero(positions),
                nonzero(keys.len() / 64),
                nonzero(1),
                attention_case.window_kv.shape[1],
            )
            .expect("native owner selection geometry"),
        );
        assert_eq!(
            call,
            candidate_capture::source_call(*start),
            "producer call identity"
        );
        let indices =
            candidate_capture::generated_producer_indices(*start, keys, call, owner_input);
        assert_eq!(
            indices,
            attention_case.compressed_indices.i32(),
            "native producer IDs"
        );
        assert_eq!(
            pending.kv_prefix(0).expect("complete staged KV prefix"),
            attention_case.compressed_kv.bf16(),
            "native staged KV prefix"
        );
        pending.commit().expect("owner publication commit");
        let diagnostic = forward_with_publication(
            &mut state,
            owner_input,
            *start,
            key_owner.epoch(),
            u64::try_from(call_id).expect("call ID"),
            SOURCE_LAYER,
            key_owner.kv_prefix(0).expect("published KV prefix"),
            &indices,
            call_frequencies(&all_frequencies, attention_case),
            attention_weights.borrowed(),
        )
        .expect("native owner and producer publication drives layer-three attention");
        assert_diagnostic(attention_case, &diagnostic);
        outputs.push(diagnostic.final_output);
    }
    outputs
}
