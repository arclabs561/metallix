//! Layer-one native attention/HC/FFN join into the qualified layer-two suffix.
//!
//! The layer-one block entry remains source-captured. From its derived attention
//! input onward, this uses native owner-backed attention, HC, and FFN operators.

use std::collections::BTreeMap;

use deepseek::{
    ffn::FfnSublayerReference,
    hc::{HcCoefficients, mixing::hc_post_bf16_reference, projection::project_hc_diagnostics},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::{
    BlockConfig, Coefficients, Model, Tensor, derive_attention_input, hc_coefficient_bounds,
    hc_projection_bounds, layer1_attention_capture, layer2_join, with_model_parameters,
};

const FIXTURE_SHA256: &str = "5f31036c71b797e7195a6b93cf2d656b8744b1e6a80a04dc68007d26e326b89b";

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
    comparison_policy: ComparisonPolicy,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct ComparisonPolicy {
    route_weight_abs_error_max: f32,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    residual: Tensor,
    incoming_pre: Tensor,
    attention_input: Tensor,
    attention_output: Tensor,
    after_attention_residual: Tensor,
    attention_pre: Tensor,
    attention_coefficients: Coefficients,
    attention_hc_mixes: Tensor,
    ffn_collapsed: Tensor,
    moe_input: Tensor,
    moe_output: Tensor,
    gate_indices: Tensor,
    gate_weights: Tensor,
    ffn_coefficients: Coefficients,
    ffn_hc_mixes: Tensor,
    output: Tensor,
    next_pre: Tensor,
    layer_two_residual: Tensor,
    layer_two_incoming_pre: Tensor,
}

fn fixture() -> Fixture {
    let raw = include_str!("../../../../../fixtures/deepseek-v41/layer1-tail-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let fixture: Fixture = serde_json::from_str(raw).expect("layer-one tail fixture JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.block_config.copies, 2);
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

pub(super) type BlockEntries = [(usize, Vec<u16>, Vec<f32>)];

fn native_inputs(fixture: &Fixture, entries: Option<&BlockEntries>) -> Vec<(usize, Vec<u16>)> {
    if let Some(entries) = entries {
        assert_eq!(entries.len(), fixture.cases.len());
    }
    let norm = fixture.block_parameters["layers.1.attn_norm.weight"].bf16();
    fixture
        .cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let captured_residual = case.residual.bf16();
            let captured_incoming = case.incoming_pre.fp32();
            let (residual, incoming) = if let Some(entries) = entries {
                let (start, residual, incoming) = &entries[index];
                assert_eq!(*start, case.start_pos);
                assert_eq!(
                    residual, &captured_residual,
                    "native Engram layer-one residual at block boundary"
                );
                assert_eq!(incoming.len(), captured_incoming.len());
                assert!(incoming.iter().all(|value| value.is_finite()));
                (residual.as_slice(), incoming.as_slice())
            } else {
                (captured_residual.as_slice(), captured_incoming.as_slice())
            };
            let input = residual
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
                .collect::<Vec<_>>();
            assert_eq!(
                input,
                case.attention_input.bf16(),
                "native layer-one HC attention input"
            );
            (case.start_pos, input)
        })
        .collect()
}

fn coefficients(
    fixture: &Fixture,
    case: &Case,
    position: usize,
    residual: &[u16],
) -> HcCoefficients {
    let projection = fixture.block_parameters["layers.1.hc_attn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.1.hc_attn_scale"]
        .fp32()
        .try_into()
        .expect("three HC scales");
    let base = fixture.block_parameters["layers.1.hc_attn_base"].fp32();
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
    .expect("native layer-one attention HC");
    let projection_bounds = hc_projection_bounds::normalized_projection_envelopes(
        residual,
        &projection,
        config.norm_eps,
    )
    .expect("source-derived layer-one HC projection bounds");
    let source_mixes = case.attention_hc_mixes.fp32();
    for ((bound, &source), &actual) in projection_bounds
        .iter()
        .zip(&source_mixes[position * 8..(position + 1) * 8])
        .zip(native.mixes())
    {
        assert!(
            bound.contains(source) && bound.contains(actual),
            "layer-one attention HC projection envelope"
        );
    }
    let bounds = hc_coefficient_bounds::coefficient_envelopes(
        &projection_bounds
            .iter()
            .map(|span| [span.lo, span.hi])
            .collect::<Vec<_>>(),
        &scale,
        &base,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .expect("source-derived layer-one HC coefficient bounds");
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
        for ((bound, &actual), &source) in bounds
            .iter()
            .zip(actual)
            .zip(&source[position * width..(position + 1) * width])
        {
            assert!(
                actual.is_finite()
                    && source.is_finite()
                    && bound[0] <= f64::from(actual)
                    && f64::from(actual) <= bound[1]
                    && bound[0] <= f64::from(source)
                    && f64::from(source) <= bound[1],
                "layer-one attention HC coefficient envelope"
            );
        }
    }
    native.coefficients().clone()
}

fn attention_handoffs(
    fixture: &Fixture,
    outputs: &[(usize, Vec<u16>)],
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    assert_eq!(outputs.len(), fixture.cases.len());
    fixture
        .cases
        .iter()
        .zip(outputs)
        .map(|(case, (start, attention))| {
            assert_eq!(*start, case.start_pos);
            assert_eq!(
                attention,
                &case.attention_output.bf16(),
                "native layer-one attention output"
            );
            let residual = case.residual.bf16();
            let mut after_attention = Vec::new();
            let mut pre = Vec::new();
            for (position, row) in residual.chunks_exact(256).enumerate() {
                let coefficients = coefficients(fixture, case, position, row);
                let mut output = vec![0; 256];
                hc_post_bf16_reference(
                    &attention[position * 128..(position + 1) * 128],
                    row,
                    coefficients.post(),
                    coefficients.comb(),
                    &mut output,
                )
                .expect("native layer-one HC post mix");
                after_attention.extend(output);
                pre.extend_from_slice(coefficients.pre());
            }
            assert_eq!(
                after_attention,
                case.after_attention_residual.bf16(),
                "native layer-one attention post-mix residual"
            );
            assert_eq!(
                case.attention_pre.fp32(),
                case.attention_coefficients.pre.fp32(),
                "source layer-one attention pre receipt"
            );
            assert_eq!(pre.len(), case.attention_pre.fp32().len());
            assert!(pre.iter().all(|value| value.is_finite()));
            (case.start_pos, after_attention, pre)
        })
        .collect()
}

fn native_ffn(
    fixture: &Fixture,
    entries: &[(usize, Vec<u16>, Vec<f32>)],
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    assert_eq!(entries.len(), fixture.cases.len());
    let norm = fixture.block_parameters["layers.1.ffn_norm.weight"].bf16();
    let projection = fixture.block_parameters["layers.1.hc_ffn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.1.hc_ffn_scale"]
        .fp32()
        .try_into()
        .expect("three FFN HC scales");
    let base = fixture.block_parameters["layers.1.hc_ffn_base"].fp32();
    with_model_parameters(
        &fixture.model,
        &fixture.encoded_parameters,
        1,
        false,
        |model| {
            let ffn = FfnSublayerReference::new(
                model,
                &norm,
                &projection,
                &scale,
                &base,
                fixture.block_config.copies,
                fixture.block_config.norm_eps,
                fixture.block_config.hc_sinkhorn_iters,
                fixture.block_config.hc_eps,
            )
            .expect("native layer-one FFN");
            fixture
                .cases
                .iter()
                .zip(entries)
                .map(|(case, (start, residual, pre))| {
                    assert_eq!(*start, case.start_pos);
                    assert_eq!(residual, &case.after_attention_residual.bf16());
                    assert_eq!(pre.len(), case.attention_pre.fp32().len());
                    assert!(pre.iter().all(|value| value.is_finite()));
                    let positions = case.after_attention_residual.shape[1];
                    let mut output = Vec::new();
                    let mut next_pre = Vec::new();
                    for position in 0..positions {
                        let result = ffn
                            .forward_token(
                                &residual[position * 256..(position + 1) * 256],
                                &pre[position * 2..(position + 1) * 2],
                            )
                            .expect("native layer-one FFN token");
                        assert_eq!(
                            result.collapsed_bf16(),
                            &case.ffn_collapsed.bf16()[position * 128..(position + 1) * 128]
                        );
                        assert_eq!(
                            result.normalized_bf16(),
                            &case.moe_input.bf16()[position * 128..(position + 1) * 128]
                        );
                        assert_eq!(
                            result.moe().output_bf16(),
                            &case.moe_output.bf16()[position * 128..(position + 1) * 128]
                        );
                        assert_routes(fixture, case, position, result.moe().routes());
                        assert_ffn_envelope(fixture, case, position, &result);
                        output.extend_from_slice(result.output_bf16());
                        next_pre.extend_from_slice(result.coefficients().pre());
                    }
                    assert_eq!(
                        output,
                        case.output.bf16(),
                        "native layer-one terminal residual"
                    );
                    assert_eq!(next_pre.len(), case.next_pre.fp32().len());
                    assert!(next_pre.iter().all(|value| value.is_finite()));
                    assert_eq!(
                        case.next_pre.fp32(),
                        case.layer_two_incoming_pre.fp32(),
                        "source layer-one terminal pre feeds layer two"
                    );
                    assert_eq!(
                        output,
                        case.layer_two_residual.bf16(),
                        "native layer-one feeds layer-two residual"
                    );
                    (case.start_pos, output, next_pre)
                })
                .collect()
        },
    )
}

fn assert_routes(
    fixture: &Fixture,
    case: &Case,
    position: usize,
    routes: &[deepseek::ExpertRoute],
) {
    let ids = case.gate_indices.indices();
    let weights = case.gate_weights.fp32();
    let ids = &ids[position * 2..(position + 1) * 2];
    let weights = &weights[position * 2..(position + 1) * 2];
    assert_eq!(
        routes.len(),
        ids.len(),
        "layer-one native routed expert count"
    );
    let mut expected = ids.to_vec();
    expected.sort_unstable();
    let actual: Vec<_> = routes.iter().map(|route| route.expert_index()).collect();
    assert_eq!(actual, expected, "layer-one native selected expert set");
    for route in routes {
        let index = ids
            .iter()
            .position(|&id| id == route.expert_index())
            .expect("native selected expert is selected by layer-one source");
        assert!(
            (route.weight() - weights[index]).abs()
                <= fixture.comparison_policy.route_weight_abs_error_max,
            "layer-one route weight at position {position} expert {}",
            route.expert_index()
        );
    }
}

fn assert_ffn_envelope(
    fixture: &Fixture,
    case: &Case,
    position: usize,
    result: &deepseek::ffn::FfnDiagnostic,
) {
    let residual = case.after_attention_residual.bf16();
    let residual = &residual[position * 256..(position + 1) * 256];
    let projection = fixture.block_parameters["layers.1.hc_ffn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.1.hc_ffn_scale"]
        .fp32()
        .try_into()
        .expect("three FFN HC scales");
    let base = fixture.block_parameters["layers.1.hc_ffn_base"].fp32();
    let projection_bounds = hc_projection_bounds::normalized_projection_envelopes(
        residual,
        &projection,
        fixture.block_config.norm_eps,
    )
    .expect("source-derived layer-one FFN projection bounds");
    let observed = project_hc_diagnostics(
        residual,
        &projection,
        &scale,
        &base,
        fixture.block_config.copies,
        fixture.block_config.norm_eps,
        fixture.block_config.hc_sinkhorn_iters,
        fixture.block_config.hc_eps,
    )
    .expect("native layer-one FFN HC");
    assert_eq!(observed.coefficients(), result.coefficients());
    let source_mixes = case.ffn_hc_mixes.fp32();
    let source_mixes = &source_mixes[position * 8..(position + 1) * 8];
    for (index, ((bound, &source), &native)) in projection_bounds
        .iter()
        .zip(source_mixes)
        .zip(observed.mixes())
        .enumerate()
    {
        assert!(bound.contains(source), "layer-one source FFN mix {index}");
        assert!(bound.contains(native), "native layer-one FFN mix {index}");
    }
    let bounds = hc_coefficient_bounds::coefficient_envelopes(
        &projection_bounds
            .iter()
            .map(|span| [span.lo, span.hi])
            .collect::<Vec<_>>(),
        &scale,
        &base,
        fixture.block_config.hc_sinkhorn_iters,
        fixture.block_config.hc_eps,
    )
    .expect("source-derived layer-one FFN coefficient bounds");
    let source_pre = case.ffn_coefficients.pre.fp32();
    let source_pre = &source_pre[position * 2..(position + 1) * 2];
    for (index, ((bound, &source), &native)) in bounds
        .pre
        .iter()
        .zip(source_pre)
        .zip(result.coefficients().pre())
        .enumerate()
    {
        assert!(
            source.is_finite() && bound[0] <= f64::from(source) && f64::from(source) <= bound[1],
            "layer-one source FFN pre {index}"
        );
        assert!(
            native.is_finite() && bound[0] <= f64::from(native) && f64::from(native) <= bound[1],
            "native layer-one FFN pre at layer-two boundary {index}"
        );
    }
}

pub(super) fn native_layer_one_entries() -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    native_layer_one_entries_from_block_entries(None)
}

pub(super) fn native_layer_one_entries_from_block_entries(
    entries: Option<&BlockEntries>,
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    let fixture = fixture();
    let inputs = native_inputs(&fixture, entries);
    let outputs = layer1_attention_capture::native_outputs_from_inputs(&inputs);
    native_ffn(&fixture, &attention_handoffs(&fixture, &outputs))
}

#[test]
fn native_layer_one_attention_hc_ffn_reaches_final_logits() {
    let entries = native_layer_one_entries();
    layer2_join::native_layer_two_from_entries(&entries);
}

#[test]
#[should_panic(expected = "native layer-one attention output")]
fn discarded_native_layer_one_attention_fails_before_ffn() {
    let fixture = fixture();
    let inputs = native_inputs(&fixture, None);
    let mut outputs = layer1_attention_capture::native_outputs_from_inputs(&inputs);
    outputs[0].1.fill(0);
    attention_handoffs(&fixture, &outputs);
}
