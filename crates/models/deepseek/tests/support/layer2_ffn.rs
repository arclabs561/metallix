//! Native layer-two FFN boundary and its source-derived HC envelopes.

use super::{
    LayerTwoCase, LayerTwoFixture, hc_coefficient_bounds, hc_projection_bounds,
    with_model_parameters,
};
use deepseek::{ffn::FfnSublayerReference, hc::projection::project_hc_diagnostics};
use sha2::{Digest, Sha256};

fn layer_two_fixture() -> LayerTwoFixture {
    let source = include_str!("../../../../../fixtures/deepseek-v41/layer2-ffn-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(source.as_bytes())),
        "d92f3c1578baa205d6100febc100e62b9a26017622a32d53e484bf4ed3e62786",
        "pinned layer-two source capture"
    );
    let fixture: LayerTwoFixture = serde_json::from_str(source).unwrap();
    assert_eq!(fixture.schema_version, 1);
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

pub(super) fn native_layer_two_entries() -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    native_layer_two_entries_from_attention(None)
}

pub(super) type AttentionEntries = [(usize, Vec<u16>, Vec<f32>)];

pub(super) fn native_layer_two_entries_from_attention(
    entries: Option<&AttentionEntries>,
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    let fixture = layer_two_fixture();
    if let Some(entries) = entries {
        assert_eq!(entries.len(), fixture.cases.len());
    }
    let norm = fixture.block_parameters["layers.2.ffn_norm.weight"].bf16();
    let projection = fixture.block_parameters["layers.2.hc_ffn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.2.hc_ffn_scale"]
        .fp32()
        .try_into()
        .unwrap();
    let base = fixture.block_parameters["layers.2.hc_ffn_base"].fp32();
    with_model_parameters(
        &fixture.model,
        &fixture.encoded_parameters,
        2,
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
            .unwrap();
            fixture
                .cases
                .iter()
                .enumerate()
                .map(|(index, case)| {
                    let positions = case.after_attention_residual.shape[1];
                    let captured_residual = case.after_attention_residual.bf16();
                    let captured_pre = case.attention_pre.fp32();
                    let (residual, pre) = if let Some(entries) = entries {
                        let (start, residual, pre) = &entries[index];
                        assert_eq!(*start, case.start_pos);
                        assert_eq!(
                            residual, &captured_residual,
                            "native layer-two attention residual at FFN boundary"
                        );
                        assert_eq!(pre.len(), captured_pre.len());
                        assert!(pre.iter().all(|value| value.is_finite()));
                        (residual.as_slice(), pre.as_slice())
                    } else {
                        (captured_residual.as_slice(), captured_pre.as_slice())
                    };
                    let expected_collapsed = case.ffn_collapsed.bf16();
                    let expected_input = case.moe_input.bf16();
                    let expected_moe = case.moe_output.bf16();
                    let mut output = Vec::new();
                    let mut next = Vec::new();
                    for position in 0..positions {
                        let result = ffn
                            .forward_token(
                                &residual[position * 256..(position + 1) * 256],
                                &pre[position * 2..(position + 1) * 2],
                            )
                            .unwrap();
                        assert_eq!(
                            result.collapsed_bf16(),
                            &expected_collapsed[position * 128..(position + 1) * 128]
                        );
                        assert_eq!(
                            result.normalized_bf16(),
                            &expected_input[position * 128..(position + 1) * 128]
                        );
                        assert_eq!(
                            result.moe().output_bf16(),
                            &expected_moe[position * 128..(position + 1) * 128]
                        );
                        assert_layer_two_next_pre_envelope(&fixture, case, position, &result);
                        output.extend_from_slice(result.output_bf16());
                        next.extend_from_slice(result.coefficients().pre());
                    }
                    assert_eq!(output, case.output.bf16());
                    assert_eq!(output, case.engram_stream.bf16());
                    assert_eq!(next.len(), case.next_pre.fp32().len());
                    assert_eq!(case.ffn_coefficients.pre.fp32(), case.next_pre.fp32());
                    assert_eq!(case.next_pre.fp32(), case.layer_three_incoming_pre.fp32());
                    (case.start_pos, output, next)
                })
                .collect()
        },
    )
}

fn assert_layer_two_next_pre_envelope(
    fixture: &LayerTwoFixture,
    case: &LayerTwoCase,
    position: usize,
    result: &deepseek::ffn::FfnDiagnostic,
) {
    let residual = case.after_attention_residual.bf16();
    let residual = &residual[position * 256..(position + 1) * 256];
    let projection = fixture.block_parameters["layers.2.hc_ffn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.2.hc_ffn_scale"]
        .fp32()
        .try_into()
        .unwrap();
    let base = fixture.block_parameters["layers.2.hc_ffn_base"].fp32();
    let projection_bounds = hc_projection_bounds::normalized_projection_envelopes(
        residual,
        &projection,
        fixture.block_config.norm_eps,
    )
    .unwrap();
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
    .unwrap();
    assert_eq!(observed.coefficients(), result.coefficients());
    let source_mixes = case.ffn_hc_mixes.fp32();
    let source_mixes = &source_mixes[position * 8..(position + 1) * 8];
    assert_eq!(source_mixes.len(), projection_bounds.len());
    for (index, ((bound, &source), &native)) in projection_bounds
        .iter()
        .zip(source_mixes)
        .zip(observed.mixes())
        .enumerate()
    {
        assert!(bound.contains(source), "layer-two source ffn mix {index}");
        assert!(bound.contains(native), "layer-two native ffn mix {index}");
    }
    let mix_bounds: Vec<_> = projection_bounds
        .iter()
        .map(|bound| [bound.lo, bound.hi])
        .collect();
    let coefficients = hc_coefficient_bounds::coefficient_envelopes(
        &mix_bounds,
        &scale,
        &base,
        fixture.block_config.hc_sinkhorn_iters,
        fixture.block_config.hc_eps,
    )
    .unwrap();
    let source_pre = case.ffn_coefficients.pre.fp32();
    let source_pre = &source_pre[position * 2..(position + 1) * 2];
    for (index, ((bound, &source), &native)) in coefficients
        .pre
        .iter()
        .zip(source_pre)
        .zip(result.coefficients().pre())
        .enumerate()
    {
        assert!(
            source.is_finite() && bound[0] <= f64::from(source) && f64::from(source) <= bound[1],
            "layer-two source HC pre {index}"
        );
        assert!(
            native.is_finite() && bound[0] <= f64::from(native) && f64::from(native) <= bound[1],
            "native layer-two HC pre at layer-three boundary {index}"
        );
    }
}
