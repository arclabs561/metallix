//! Source-captured index-selection to layer-attention integration for V4.1.
//!
//! This test joins the production index-query prefix, BF16 scorer, strict
//! selector, candidate producer, atomic compressor/key/KV owner, and attention
//! adapter. Owner/consumer layer inputs remain fixture boundaries; both producer
//! and consumer QR are computed by the native candidate-query adapter.

#[path = "support/attention_capture.rs"]
mod attention_capture;

#[path = "support/candidate_capture.rs"]
mod candidate_capture;

use std::num::NonZeroUsize;

use attention_capture::{
    SOURCE_LAYER, assert_diagnostic, call_frequencies, fixture as attention_fixture,
    forward_with_publication, frequencies, layout as attention_layout,
    weights as attention_weights,
};
use deepseek::{
    attention::layer::{Fp8Projection, LayerAttentionError, LayerAttentionState},
    indexer::{
        cache::IndexKeyPublicationId,
        query::{
            CandidateQueryLayout, CandidateQueryWeights, IndexKeyView, IndexQueryLayout,
            IndexQueryWeights, prepare_scored_query,
        },
        selection::{SelectionCall, SelectionGeometry, select_from_candidates},
    },
    select_indices,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const CAPTURE_SHA256: &str = "e27dde6ead409c74f7bb2c9e08d4cd5a2b0cfc3c9505c7d6b8908b1cd78b1cc6";
const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";

fn raw_fixture() -> Value {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-attention-reference.json"
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

fn bf16_from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    u16::try_from(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16).expect("BF16 high half")
}

fn f32_from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
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

fn masked_scores(
    scores: &[u16],
    candidates: &[bool],
    start: usize,
    positions: usize,
    keys: usize,
    ratio: usize,
) -> Vec<u16> {
    assert_eq!(scores.len(), positions * keys, "score geometry");
    assert_eq!(candidates.len(), scores.len(), "candidate geometry");
    assert!(start == 0 || positions == 1, "captured causal geometry");
    let mut output = scores.to_vec();
    for position in 0..positions {
        let reachable = (start + position + 1) / ratio;
        for key in reachable..keys {
            output[position * keys + key] = bf16_from_f32(f32::NEG_INFINITY);
        }
    }
    output
        .into_iter()
        .zip(candidates)
        .map(|(score, &keep)| {
            if keep {
                score
            } else {
                bf16_from_f32(f32::NEG_INFINITY)
            }
        })
        .collect()
}

#[allow(
    clippy::too_many_lines,
    reason = "one source boundary is verified end-to-end"
)]
fn generated_indices(
    root: &Value,
    raw_case: &Value,
    attention_case: &attention_capture::Case,
    layout: IndexQueryLayout,
    weights: IndexQueryWeights<'_>,
    keys: &[u16],
    publication: IndexKeyPublicationId,
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
    let candidates = candidate_capture::generated_candidates(start, keys, call);
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

#[path = "support/owner_attention_capture.rs"]
mod owner_attention_capture;

#[test]
fn native_index_selection_drives_captured_attention_calls() {
    let attention = attention_fixture();
    let inputs: Vec<_> = attention
        .cases
        .iter()
        .map(|case| (case.start_pos, case.input.bf16()))
        .collect();
    let outputs =
        owner_attention_capture::native_outputs_from_ownered_inputs(&inputs, CAPTURE_SHA256);
    assert_eq!(outputs.len(), inputs.len());
}
fn generated_prefill_indices(raw: &Value, attention: &attention_capture::Fixture) -> Vec<i32> {
    let model = field(raw, "model");
    let parameters = field(raw, "encoded_parameters");
    let index_layout = IndexQueryLayout::new(
        nonzero(1),
        nonzero(usize_field(model, "dim")),
        nonzero(usize_field(model, "q_lora_rank")),
        nonzero(usize_field(model, "index_n_heads")),
        nonzero(usize_field(model, "index_head_dim")),
        nonzero(usize_field(model, "rope_head_dim") / 2),
    )
    .expect("source index layout");
    let wq_b_codes = fp8(field(parameters, "layers.4.attn.indexer.wq_b.weight"));
    let wq_b_scales = fp8(field(parameters, "layers.4.attn.indexer.wq_b.scale"));
    let weights_proj = bf16(field(
        parameters,
        "layers.4.attn.indexer.weights_proj.weight",
    ));
    generated_indices(
        raw,
        &field(raw, "cases").as_array().expect("raw cases")[0],
        &attention.cases[0],
        index_layout,
        IndexQueryWeights {
            wq_b_codes: &wq_b_codes,
            wq_b_scales: &wq_b_scales,
            weights_proj: &weights_proj,
        },
        &bf16(field(
            field(
                field(
                    &field(raw, "cases").as_array().expect("raw cases")[0],
                    "indexer",
                ),
                "inputs",
            ),
            "shared_index_k_prefix",
        )),
        IndexKeyPublicationId::new(3, 0, 0),
    )
}

#[test]
fn future_generated_index_is_rejected_atomically() {
    let raw = raw_fixture();
    let attention = attention_fixture();
    let generated = generated_prefill_indices(&raw, &attention);
    let mut corrupted = generated.clone();
    corrupted[0] = corrupted[0].checked_add(1).expect("small captured ID");
    let all_frequencies = frequencies(&attention);
    let attention_weights = attention_weights(&attention.encoded_parameters);
    let mut state = LayerAttentionState::new(attention_layout(&attention.model));
    let case = &attention.cases[0];
    assert!(matches!(
        forward_with_publication(
            &mut state,
            &case.input.bf16(),
            case.start_pos,
            0,
            0,
            SOURCE_LAYER,
            &case.compressed_kv.bf16(),
            &corrupted,
            call_frequencies(&all_frequencies, case),
            attention_weights.borrowed(),
        ),
        Err(LayerAttentionError::FutureCompressedIndex {
            slot: 0,
            index: 6,
            causal: 1,
        })
    ));
    let diagnostic = forward_with_publication(
        &mut state,
        &case.input.bf16(),
        case.start_pos,
        0,
        0,
        SOURCE_LAYER,
        &case.compressed_kv.bf16(),
        &generated,
        call_frequencies(&all_frequencies, case),
        attention_weights.borrowed(),
    )
    .expect("specific rejected index leaves the prefill state unchanged");
    assert_diagnostic(case, &diagnostic);
}

#[test]
fn legal_but_wrong_generated_index_changes_attention_numerics() {
    let raw = raw_fixture();
    let attention = attention_fixture();
    let mut generated = generated_prefill_indices(&raw, &attention);
    assert_eq!(generated[4], 7, "source last prefill index");
    generated[4] = 5;
    let all_frequencies = frequencies(&attention);
    let attention_weights = attention_weights(&attention.encoded_parameters);
    let mut state = LayerAttentionState::new(attention_layout(&attention.model));
    let case = &attention.cases[0];
    let diagnostic = forward_with_publication(
        &mut state,
        &case.input.bf16(),
        case.start_pos,
        0,
        0,
        SOURCE_LAYER,
        &case.compressed_kv.bf16(),
        &generated,
        call_frequencies(&all_frequencies, case),
        attention_weights.borrowed(),
    )
    .expect("causal but wrong generated index remains a valid attention publication");
    assert_ne!(
        diagnostic.final_output,
        case.output.bf16(),
        "a legal wrong generated index changes attention numerics"
    );
}

#[test]
fn removing_a_generated_candidate_changes_selection_and_attention() {
    let raw = raw_fixture();
    let attention = attention_fixture();
    let raw_case = &field(&raw, "cases").as_array().expect("cases")[0];
    let indexer = field(raw_case, "indexer");
    let inputs = field(indexer, "inputs");
    let keys = bf16(field(inputs, "shared_index_k_prefix"));
    let mut candidates =
        candidate_capture::generated_candidates(0, &keys, candidate_capture::source_call(0))
            .mask()
            .to_vec();
    let original_ids = generated_prefill_indices(&raw, &attention);
    let offset = usize_field(inputs, "offset");
    let positions = shape(field(inputs, "qr"))[1];
    let key_count = shape(field(inputs, "shared_index_k_prefix"))[1];
    let last = positions - 1;
    let selected = usize::try_from(original_ids[last]).expect("valid selected ID") - offset;
    assert!(candidates[last * key_count + selected]);
    candidates[last * key_count + selected] = false;
    // Hold scores fixed to isolate the consumer's response to this mask bit.
    // The positive join above computes these scores with the native scorer.
    let scores = bf16(field(field(indexer, "operations"), "scores_after_head_sum"));
    let masked = masked_scores(&scores, &candidates, 0, positions, key_count, 1);
    let changed_ids: Vec<i32> = (0..positions)
        .flat_map(|position| {
            select_indices(
                &masked[position * key_count..(position + 1) * key_count]
                    .iter()
                    .map(|&bits| f32_from_bf16(bits))
                    .collect::<Vec<_>>(),
                position + 1,
                1,
                offset,
            )
            .expect("remaining candidate has an unambiguous cutoff")
        })
        .collect();
    assert_eq!(&changed_ids[..last], &original_ids[..last]);
    assert_ne!(changed_ids[last], original_ids[last]);
    let case = &attention.cases[0];
    let all_frequencies = frequencies(&attention);
    let weights = attention_weights(&attention.encoded_parameters);
    let mut state = LayerAttentionState::new(attention_layout(&attention.model));
    let output = forward_with_publication(
        &mut state,
        &case.input.bf16(),
        0,
        0,
        0,
        SOURCE_LAYER,
        &case.compressed_kv.bf16(),
        &changed_ids,
        call_frequencies(&all_frequencies, case),
        weights.borrowed(),
    )
    .expect("changed selection remains a legal causal publication");
    assert_ne!(output.final_output, case.output.bf16());
}
