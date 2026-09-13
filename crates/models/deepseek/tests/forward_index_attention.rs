//! Source-captured index-selection to layer-attention integration for V4.1.
//!
//! This test joins the production index-query prefix, BF16 scorer, strict
//! selector, atomic compressor/key/KV owner, and attention adapter. The source QR,
//! owner-layer input, and candidate mask remain fixture boundaries.

#[path = "support/attention_capture.rs"]
mod attention_capture;

use std::num::NonZeroUsize;

use attention_capture::{
    SOURCE_LAYER, assert_diagnostic, call_frequencies, fixture as attention_fixture,
    forward_with_publication, frequencies, layout as attention_layout,
    weights as attention_weights,
};
use deepseek::{
    attention::layer::{LayerAttentionError, LayerAttentionState},
    indexer::{
        bf16::index_scores_bf16_reference,
        cache::IndexKeyPublicationId,
        compressed_kv::{CompressedKvLayout, prepare_compressed_kv},
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{RatioOneCompressedOwner, RatioOneOwnerCall, RatioOneOwnerWeights},
        query::{IndexQueryLayout, IndexQueryWeights, prepare_index_query},
    },
    precision::{Fp4ActivationMode, requantize_bf16_activations_e2m1},
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

fn compressor_fixture() -> Value {
    let root: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-compressor-reference.json"
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
    let positions = shape(field(inputs, "qr"))[1];
    let heads = usize_field(model, "index_n_heads");
    let head_dimension = usize_field(model, "index_head_dim");
    let query = prepare_index_query(
        &qr,
        &x,
        &source_frequencies(
            root,
            start,
            positions,
            usize_field(model, "rope_head_dim") / 2,
        ),
        weights,
        layout,
    )
    .expect("bounded production index query");
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
    let mut score_bits = Vec::with_capacity(positions * keys_per_position);
    for position in 0..positions {
        let query_start = position * heads * head_dimension;
        let weights_start = position * heads;
        let scores = index_scores_bf16_reference(
            &query.query_post_fp4[query_start..query_start + heads * head_dimension],
            keys,
            &query.scaled_head_weights[weights_start..weights_start + heads],
            nonzero(head_dimension),
        )
        .expect("bounded production BF16 score chain");
        score_bits.extend(scores.scores);
    }
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
    let masked = masked_scores(
        &score_bits,
        &bools(field(inputs, "candidate_mask")),
        start,
        positions,
        keys_per_position,
        ratio,
    );
    assert_eq!(
        masked,
        bf16(field(operations, "scores_after_candidate_mask")),
        "start {start} masked scores"
    );
    let offset = usize_field(inputs, "offset");
    let output: Vec<i32> = (0..positions)
        .flat_map(|position| {
            select_indices(
                &masked[position * keys_per_position..(position + 1) * keys_per_position]
                    .iter()
                    .map(|&bits| f32_from_bf16(bits))
                    .collect::<Vec<_>>(),
                (start + position + 1) / ratio,
                usize_field(model, "index_topk"),
                offset,
            )
            .expect("source cutoff is unambiguous")
        })
        .collect();
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

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "the source-capture join keeps owner and consumer evidence together"
)]
fn native_index_selection_drives_captured_attention_calls() {
    let raw = raw_fixture();
    let attention = attention_fixture();
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
        "../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
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
    for (call_id, (raw_case, attention_case)) in field(&raw, "cases")
        .as_array()
        .expect("raw cases")
        .iter()
        .zip(&attention.cases)
        .enumerate()
    {
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
        let input = field(compressor_case, "attention_input");
        assert_eq!(shape(input), [1, positions, 128]);
        let prepared = key_owner
            .forward(RatioOneOwnerCall::new(
                IndexKeyPublicationId::new(3, 0, u64::try_from(call_id).expect("call ID")),
                attention_case.start_pos,
                nonzero(positions),
                &bf16(input),
                &source_frequencies(&raw, attention_case.start_pos, positions, 16),
                weights,
            ))
            .expect("native atomic owner call");
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
        let keys = key_owner.key_prefix(0).expect("batch zero keys");
        assert_eq!(keys, bf16(field(owner_case, "index_cache_after")));
        let indices = generated_indices(
            &raw,
            raw_case,
            attention_case,
            index_layout,
            index_weights,
            keys,
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
            &attention_case.input.bf16(),
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
    }
    assert!(
        wrong_group_detected,
        "oracle distinguishes index-key quantization from KV quantization"
    );
    assert!(
        wrong_frequency_detected,
        "oracle distinguishes decode rotary positions"
    );
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
