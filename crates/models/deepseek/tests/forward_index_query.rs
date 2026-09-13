//! Source-fixture gate for the V4.1 index-query preparation prefix.
//!
//! This deliberately stops before index score/candidate composition. Expected
//! tensors are source observations, never recomputed by this test.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::query::{IndexQueryLayout, IndexQueryWeights, prepare_index_query},
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-attention-reference.json"
    ))
    .expect("source attention fixture JSON")
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value
        .get(name)
        .unwrap_or_else(|| panic!("missing source fixture field {name}"))
}

fn usize_field(value: &Value, name: &str) -> usize {
    usize::try_from(
        field(value, name)
            .as_u64()
            .expect("unsigned fixture dimension"),
    )
    .expect("fixture dimension fits usize")
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("source dimensions are nonzero")
}

fn bytes(tensor: &Value) -> Vec<u8> {
    let hex = field(tensor, "storage_hex")
        .as_str()
        .expect("fixture hex storage");
    assert!(hex.len().is_multiple_of(2), "hex byte alignment");
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("hex UTF-8"), 16).expect("hex byte")
        })
        .collect()
}

fn bf16(tensor: &Value) -> Vec<u16> {
    assert_eq!(field(tensor, "dtype").as_str(), Some("torch.bfloat16"));
    bytes(tensor)
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
        .collect()
}

fn fp8(tensor: &Value) -> Vec<u8> {
    assert!(matches!(
        field(tensor, "dtype").as_str(),
        Some("torch.float8_e4m3fn" | "torch.float8_e8m0fnu")
    ));
    bytes(tensor)
}

fn shape(tensor: &Value) -> Vec<usize> {
    field(tensor, "shape")
        .as_array()
        .expect("fixture shape")
        .iter()
        .map(|value| {
            usize::try_from(value.as_u64().expect("shape dimension")).expect("shape fits usize")
        })
        .collect()
}

fn frequencies(root: &Value, start: usize, positions: usize, pairs: usize) -> Vec<RotaryFrequency> {
    let all = field(root, "frequencies");
    let shape = shape(all);
    assert_eq!(shape[1], pairs, "source frequency pair width");
    let source = field(all, "fp32_pairs")
        .as_array()
        .expect("frequency pairs");
    assert!(
        start + positions <= shape[0],
        "call frequency slice in source table"
    );
    source[start * pairs..(start + positions) * pairs]
        .iter()
        .map(|pair| {
            let pair = pair.as_array().expect("frequency pair");
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

#[test]
fn captured_index_query_precision_boundaries_match_starts_zero_five_and_six() {
    let root = fixture();
    let model = field(&root, "model");
    let parameters = field(&root, "encoded_parameters");
    let wq_b_weight = field(parameters, "layers.4.attn.indexer.wq_b.weight");
    let wq_b_scale = field(parameters, "layers.4.attn.indexer.wq_b.scale");
    let weights_proj = field(parameters, "layers.4.attn.indexer.weights_proj.weight");
    let heads = usize_field(model, "index_n_heads");
    let head_dimension = usize_field(model, "index_head_dim");
    let rope_pairs = usize_field(model, "rope_head_dim") / 2;
    let layout = IndexQueryLayout::new(
        nonzero(1),
        nonzero(usize_field(model, "dim")),
        nonzero(usize_field(model, "q_lora_rank")),
        nonzero(heads),
        nonzero(head_dimension),
        nonzero(rope_pairs),
    )
    .expect("source model index-query geometry");
    assert_eq!(
        shape(wq_b_weight),
        [heads * head_dimension, usize_field(model, "q_lora_rank")]
    );
    assert_eq!(shape(weights_proj), [heads, usize_field(model, "dim")]);

    let starts = [0, 5, 6];
    let cases = field(&root, "cases").as_array().expect("source cases");
    assert_eq!(cases.len(), starts.len());
    for (case, start) in cases.iter().zip(starts) {
        let indexer = field(case, "indexer");
        let inputs = field(indexer, "inputs");
        let operations = field(indexer, "operations");
        assert_eq!(usize_field(inputs, "start_pos"), start);
        let qr = bf16(field(inputs, "qr"));
        let x = bf16(field(inputs, "x"));
        let positions = shape(field(inputs, "qr"))[1];
        let actual = prepare_index_query(
            &qr,
            &x,
            &frequencies(&root, start, positions, rope_pairs),
            IndexQueryWeights {
                wq_b_codes: &fp8(wq_b_weight),
                wq_b_scales: &fp8(wq_b_scale),
                weights_proj: &bf16(weights_proj),
            },
            layout,
        )
        .expect("source-shaped index query");
        assert_eq!(
            actual.query_post_fp4,
            bf16(field(operations, "q_after_rope_fp4")),
            "start {start} query FP4"
        );
        assert_eq!(
            actual.projected_head_weights,
            bf16(field(operations, "weights_proj_output")),
            "start {start} BF16 projection"
        );
        assert_eq!(
            actual.scaled_head_weights,
            bf16(field(operations, "scaled_weights")),
            "start {start} scaled weights"
        );
    }
}
