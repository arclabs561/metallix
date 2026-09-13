//! Source-captured index-score and selection composition for layer four.
//!
//! The production query prefix supplies Q and signed weights. This integration
//! test calls the production BF16 scorer, applies source-shaped masks, and
//! checks the existing final selection helper. Shared keys and candidates
//! remain source-supplied; this is not a complete native indexer.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::{
        bf16::index_scores_bf16_reference,
        query::{IndexQueryLayout, IndexQueryWeights, prepare_index_query},
    },
    select_indices,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixture() -> Value {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-attention-reference.json"
    ))
    .expect("source attention fixture JSON");
    let source = field(&fixture, "source");
    assert_eq!(
        field(source, "revision").as_str(),
        Some("dba1be0a40aa45a94ad051997016db3960a90277")
    );
    assert_eq!(
        field(source, "complete_capture_sha256").as_str(),
        Some("e27dde6ead409c74f7bb2c9e08d4cd5a2b0cfc3c9505c7d6b8908b1cd78b1cc6")
    );
    let cases = field(&fixture, "cases").as_array().expect("source cases");
    assert_eq!(cases.len(), 3, "captured calls");
    assert_eq!(
        cases
            .iter()
            .map(|case| usize_field(field(field(case, "indexer"), "inputs"), "start_pos"))
            .collect::<Vec<_>>(),
        [0, 5, 6],
        "captured call starts"
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

fn frequencies(root: &Value, start: usize, positions: usize, pairs: usize) -> Vec<RotaryFrequency> {
    let all = field(root, "frequencies");
    let source = field(all, "fp32_pairs")
        .as_array()
        .expect("frequency values");
    assert_eq!(shape(all)[1], pairs);
    source[start * pairs..(start + positions) * pairs]
        .iter()
        .map(|pair| {
            let pair = pair.as_array().expect("complex pair");
            RotaryFrequency::new(
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

fn causal_mask(
    scores: &[u16],
    start: usize,
    positions: usize,
    keys: usize,
    ratio: usize,
) -> Vec<u16> {
    // The captured prefill uses per-position causal limits. Nonzero starts are
    // singleton decodes, for which the source's scalar end limit is identical.
    assert!(
        start == 0 || positions == 1,
        "fixture-local causal geometry"
    );
    let mut output = scores.to_vec();
    for position in 0..positions {
        let reachable = (start + position + 1) / ratio;
        for key in reachable..keys {
            output[position * keys + key] = bf16_from_f32(f32::NEG_INFINITY);
        }
    }
    output
}

fn candidate_mask(scores: &[u16], candidates: &[bool]) -> Vec<u16> {
    assert_eq!(scores.len(), candidates.len(), "candidate mask score shape");
    scores
        .iter()
        .zip(candidates)
        .map(|(&score, &keep)| {
            if keep {
                score
            } else {
                bf16_from_f32(f32::NEG_INFINITY)
            }
        })
        .collect()
}

struct Harness<'a> {
    root: &'a Value,
    model: &'a Value,
    layout: IndexQueryLayout,
    weights: IndexQueryWeights<'a>,
    heads: usize,
    dimension: usize,
    ratio: usize,
}

#[derive(Default)]
struct ScoreStages {
    dot_products: Vec<u16>,
    rectified: Vec<u16>,
    weighted: Vec<u16>,
    scores: Vec<u16>,
}

fn score_positions(
    query: &[u16],
    keys: &[u16],
    head_weights: &[u16],
    positions: usize,
    heads: usize,
    dimension: usize,
) -> ScoreStages {
    assert_eq!(query.len(), positions * heads * dimension);
    assert_eq!(head_weights.len(), positions * heads);
    let mut stages = ScoreStages::default();
    for position in 0..positions {
        let query_start = position * heads * dimension;
        let weight_start = position * heads;
        let diagnostic = index_scores_bf16_reference(
            &query[query_start..query_start + heads * dimension],
            keys,
            &head_weights[weight_start..weight_start + heads],
            nonzero(dimension),
        )
        .expect("bounded production BF16 score chain");
        stages.dot_products.extend(diagnostic.dot_products);
        stages.rectified.extend(diagnostic.rectified);
        stages.weighted.extend(diagnostic.weighted);
        stages.scores.extend(diagnostic.scores);
    }
    stages
}

#[allow(
    clippy::too_many_lines,
    reason = "the source fixture's ordered BF16 stages are intentionally asserted together so the first divergent boundary remains visible"
)]
fn assert_case(harness: &Harness<'_>, case: &Value) {
    let indexer = field(case, "indexer");
    let inputs = field(indexer, "inputs");
    let operations = field(indexer, "operations");
    let start = usize_field(inputs, "start_pos");
    let qr_tensor = field(inputs, "qr");
    let positions = shape(qr_tensor)[1];
    let query = prepare_index_query(
        &bf16(qr_tensor),
        &bf16(field(inputs, "x")),
        &frequencies(
            harness.root,
            start,
            positions,
            usize_field(harness.model, "rope_head_dim") / 2,
        ),
        harness.weights,
        harness.layout,
    )
    .expect("native query prefix");
    assert_eq!(
        query.query_post_fp4,
        bf16(field(operations, "q_after_rope_fp4")),
        "start {start} native Q"
    );
    let keys = bf16(field(inputs, "shared_index_k_prefix"));
    let key_positions = shape(field(inputs, "shared_index_k_prefix"))[1];
    let stages = score_positions(
        &query.query_post_fp4,
        &keys,
        &query.scaled_head_weights,
        positions,
        harness.heads,
        harness.dimension,
    );
    assert_eq!(
        stages.dot_products,
        bf16(field(operations, "scores_einsum")),
        "start {start} einsum"
    );
    assert_eq!(
        stages.rectified,
        bf16(field(operations, "scores_after_relu")),
        "start {start} ReLU"
    );
    assert_eq!(
        stages.weighted,
        bf16(field(operations, "scores_weighted_per_head")),
        "start {start} weighted"
    );
    assert_eq!(
        stages.scores,
        bf16(field(operations, "scores_after_head_sum")),
        "start {start} head sum"
    );
    let causal = causal_mask(
        &stages.scores,
        start,
        positions,
        key_positions,
        harness.ratio,
    );
    if let Some(expected) = operations.get("scores_after_causal_mask") {
        assert_eq!(causal, bf16(expected), "start {start} causal mask");
    }
    let candidates = bools(field(inputs, "candidate_mask"));
    let selected = candidate_mask(&causal, &candidates);
    let expected_selected = bf16(field(operations, "scores_after_candidate_mask"));
    assert_eq!(selected, expected_selected, "start {start} candidate mask");
    if start == 0 {
        assert!(
            causal
                .iter()
                .zip(&candidates)
                .any(|(&score, &candidate)| !candidate && f32_from_bf16(score).is_finite()),
            "prefill capture must contain a candidate-masked causal score"
        );
        assert_ne!(
            causal, selected,
            "candidate mask must be consumed after causal masking"
        );
    }
    let offset = usize_field(inputs, "offset");
    let indices: Vec<i32> = (0..positions)
        .flat_map(|position| {
            select_indices(
                &selected[position * key_positions..(position + 1) * key_positions]
                    .iter()
                    .map(|&bits| f32_from_bf16(bits))
                    .collect::<Vec<_>>(),
                (start + position + 1) / harness.ratio,
                usize_field(harness.model, "index_topk"),
                offset,
            )
            .expect("unambiguous source cutoff")
        })
        .collect();
    assert_eq!(
        indices,
        i32s(field(indexer, "output_indices")),
        "start {start} output indices"
    );
    let wrong_offset = select_indices(
        &selected[..key_positions]
            .iter()
            .map(|&bits| f32_from_bf16(bits))
            .collect::<Vec<_>>(),
        (start + 1) / harness.ratio,
        usize_field(harness.model, "index_topk"),
        0,
    )
    .expect("same source row with wrong offset is selectable");
    assert_ne!(
        &wrong_offset,
        &indices[..wrong_offset.len()],
        "start {start} offset control must affect consumed IDs"
    );
}

#[test]
fn captured_index_scores_and_selection_match_all_three_source_calls() {
    let root = fixture();
    let model = field(&root, "model");
    let parameters = field(&root, "encoded_parameters");
    let heads = usize_field(model, "index_n_heads");
    let dimension = usize_field(model, "index_head_dim");
    let ratio = usize::try_from(
        field(model, "compress_ratios").as_array().expect("ratios")[4]
            .as_u64()
            .expect("ratio"),
    )
    .expect("ratio fits usize");
    let layout = IndexQueryLayout::new(
        nonzero(1),
        nonzero(usize_field(model, "dim")),
        nonzero(usize_field(model, "q_lora_rank")),
        nonzero(heads),
        nonzero(dimension),
        nonzero(usize_field(model, "rope_head_dim") / 2),
    )
    .expect("source query layout");
    let wq_b_codes = fp8(field(parameters, "layers.4.attn.indexer.wq_b.weight"));
    let wq_b_scales = fp8(field(parameters, "layers.4.attn.indexer.wq_b.scale"));
    let weights_proj = bf16(field(
        parameters,
        "layers.4.attn.indexer.weights_proj.weight",
    ));
    let weights = IndexQueryWeights {
        wq_b_codes: &wq_b_codes,
        wq_b_scales: &wq_b_scales,
        weights_proj: &weights_proj,
    };
    let harness = Harness {
        root: &root,
        model,
        layout,
        weights,
        heads,
        dimension,
        ratio,
    };
    for case in field(&root, "cases").as_array().expect("source cases") {
        assert_case(&harness, case);
    }
}

#[test]
fn final_selection_refuses_a_source_unspecified_cutoff_tie() {
    assert!(matches!(
        select_indices(&[1.0, 1.0], 2, 1, 0),
        Err(deepseek::SelectionError::AmbiguousCutoffTie)
    ));
}
