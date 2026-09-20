//! Layer-two HC/attention/FFN join into the already-qualified final suffix.
//! Earlier block residuals and incoming HC state remain captured boundaries.

use std::collections::BTreeMap;

use deepseek::hc::{
    HcCoefficients, mixing::hc_post_bf16_reference, projection::project_hc_diagnostics,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    BlockConfig, BlockControl, Coefficients, Tensor, assert_final_suffix, block_tail_from_entries,
    derive_attention_input, engram_capture, fixture, hc_coefficient_bounds, hc_projection_bounds,
    layer_three_fixture, layer2_attention_capture, layer2_ffn,
    native_layer_three_block_tail_from_entries,
};

#[derive(Deserialize)]
struct HcFixture {
    schema_version: u32,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
    cases: Vec<HcCase>,
}

#[derive(Deserialize)]
struct HcCase {
    start_pos: usize,
    residual: Tensor,
    incoming_pre: Tensor,
    attention_input: Tensor,
    after_attention_residual: Tensor,
    attention_pre: Tensor,
    attention_hc_mixes: Tensor,
    attention_coefficients: Coefficients,
}

fn hc_fixture() -> HcFixture {
    let raw = include_str!("../../../../../fixtures/deepseek-v41/layer2-hc-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "9bc422171d741516879dd3e14dd856fc1b099e9521190d4da75e1c8ba1722edf"
    );
    let fixture: HcFixture = serde_json::from_str(raw).unwrap();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.block_config.copies, 2);
    assert_eq!(fixture.block_config.hc_sinkhorn_iters, 20);
    assert_eq!(fixture.block_config.norm_eps.to_bits(), 1e-20_f32.to_bits());
    assert_eq!(fixture.block_config.hc_eps.to_bits(), 1e-6_f32.to_bits());
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| case.start_pos)
            .collect::<Vec<_>>(),
        [0, 5, 6]
    );
    fixture
}

fn native_inputs(fixture: &HcFixture) -> Vec<(usize, Vec<u16>)> {
    let norm = fixture.block_parameters["layers.2.attn_norm.weight"].bf16();
    fixture
        .cases
        .iter()
        .map(|case| {
            let residual = case.residual.bf16();
            let incoming = case.incoming_pre.fp32();
            let input: Vec<_> = residual
                .chunks_exact(256)
                .enumerate()
                .flat_map(|(position, row)| {
                    derive_attention_input(
                        row,
                        &incoming[position * 2..(position + 1) * 2],
                        &norm,
                        fixture.block_config.norm_eps,
                    )
                })
                .collect();
            assert_eq!(
                input,
                case.attention_input.bf16(),
                "native layer-two HC attention input"
            );
            (case.start_pos, input)
        })
        .collect()
}

fn check_coefficients(
    fixture: &HcFixture,
    case: &HcCase,
    position: usize,
    residual: &[u16],
) -> HcCoefficients {
    let projection = fixture.block_parameters["layers.2.hc_attn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.2.hc_attn_scale"]
        .fp32()
        .try_into()
        .unwrap();
    let base = fixture.block_parameters["layers.2.hc_attn_base"].fp32();
    let config = &fixture.block_config;
    let native = project_hc_diagnostics(
        residual,
        &projection,
        &scale,
        &base,
        config.copies,
        config.norm_eps,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .unwrap();
    let spans = hc_projection_bounds::normalized_projection_envelopes(
        residual,
        &projection,
        config.norm_eps,
    )
    .unwrap();
    let source = case.attention_hc_mixes.fp32();
    for ((span, &actual), &expected) in spans
        .iter()
        .zip(native.mixes())
        .zip(&source[position * 8..(position + 1) * 8])
    {
        assert!(
            span.contains(actual) && span.contains(expected),
            "layer-two attention HC projection envelope"
        );
    }
    let bounds = hc_coefficient_bounds::coefficient_envelopes(
        &spans
            .iter()
            .map(|span| [span.lo, span.hi])
            .collect::<Vec<_>>(),
        &scale,
        &base,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .unwrap();
    for (bounds, actual, source, width) in [
        (
            &bounds.pre,
            native.coefficients().pre(),
            case.attention_coefficients.pre.fp32(),
            2,
        ),
        (
            &bounds.post,
            native.coefficients().post(),
            case.attention_coefficients.post.fp32(),
            2,
        ),
        (
            &bounds.comb,
            native.coefficients().comb(),
            case.attention_coefficients.comb.fp32(),
            4,
        ),
    ] {
        for ((span, &actual), &expected) in bounds
            .iter()
            .zip(actual)
            .zip(&source[position * width..(position + 1) * width])
        {
            for value in [actual, expected] {
                assert!(
                    value.is_finite() && span[0] <= f64::from(value) && f64::from(value) <= span[1],
                    "layer-two attention HC coefficient envelope"
                );
            }
        }
    }
    native.coefficients().clone()
}

fn handoffs(
    fixture: &HcFixture,
    outputs: &[(usize, Vec<u16>)],
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    assert_eq!(outputs.len(), fixture.cases.len());
    fixture
        .cases
        .iter()
        .zip(outputs)
        .map(|(case, (start, attention))| {
            assert_eq!(*start, case.start_pos);
            let residual = case.residual.bf16();
            assert_eq!(attention.len() * 2, residual.len());
            let mut joined = Vec::new();
            let mut pre = Vec::new();
            for (position, row) in residual.chunks_exact(256).enumerate() {
                let coefficients = check_coefficients(fixture, case, position, row);
                let mut output = vec![0; 256];
                hc_post_bf16_reference(
                    &attention[position * 128..(position + 1) * 128],
                    row,
                    coefficients.post(),
                    coefficients.comb(),
                    &mut output,
                )
                .unwrap();
                joined.extend(output);
                pre.extend_from_slice(coefficients.pre());
            }
            assert!(
                joined == case.after_attention_residual.bf16(),
                "native layer-two attention post-mix residual at start {}",
                case.start_pos
            );
            assert_eq!(
                case.attention_pre.fp32(),
                case.attention_coefficients.pre.fp32()
            );
            (case.start_pos, joined, pre)
        })
        .collect()
}

fn through_final_suffix(entries: &layer2_ffn::AttentionEntries) {
    let layer_two = layer2_ffn::native_layer_two_entries_from_attention(Some(entries));
    let streams: Vec<_> = layer_two
        .iter()
        .map(|(start, residual, _)| (*start, residual.clone()))
        .collect();
    let engram = engram_capture::native_layer_three_block_entries_from_streams(Some(&streams));
    let pre: Vec<_> = layer_two
        .into_iter()
        .map(|(start, _, pre)| (start, pre))
        .collect();
    let third = native_layer_three_block_tail_from_entries(
        &layer_three_fixture(),
        Some(&engram),
        Some(&pre),
    );
    let fourth = fixture();
    let output =
        block_tail_from_entries(&fourth, BlockControl::NativeAttention, true, Some(&third));
    assert_final_suffix(&fourth, output);
}

#[test]
fn native_layer_two_attention_hc_ffn_reaches_final_logits() {
    let fixture = hc_fixture();
    let inputs = native_inputs(&fixture);
    let outputs = layer2_attention_capture::native_outputs_from_inputs(&inputs);
    through_final_suffix(&handoffs(&fixture, &outputs));
}

#[test]
#[should_panic(expected = "native layer-two attention post-mix residual")]
fn discarded_native_attention_fails_before_ffn() {
    let fixture = hc_fixture();
    let inputs = native_inputs(&fixture);
    let mut outputs = layer2_attention_capture::native_outputs_from_inputs(&inputs);
    outputs[0].1.fill(0);
    handoffs(&fixture, &outputs);
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(16))]

    #[test]
    fn dropping_selected_trace_token_attention_is_rejected(mask in 1_u8..128) {
        let fixture = hc_fixture();
        let inputs = native_inputs(&fixture);
        let mut outputs = layer2_attention_capture::native_outputs_from_inputs(&inputs);
        for token in 0_usize..7 {
            if mask & (1 << token) != 0 {
                let (call, position) = if token < 5 { (0, token) } else { (token - 4, 0) };
                outputs[call].1[position * 128..(position + 1) * 128].fill(0);
            }
        }
        proptest::prop_assert!(std::panic::catch_unwind(|| handoffs(&fixture, &outputs)).is_err());
    }
}
