//! Borrowed block-tail operands kept alive across the complete request test.
use super::super::{
    BlockConfig, BlockTailParameters, Fixture, Model, Tensor, with_model_parameters,
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
pub(super) struct TailDefinition {
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    pub(super) block_parameters: BTreeMap<String, Tensor>,
    pub(super) block_config: BlockConfig,
}

use deepseek::{ffn::FfnSublayerReference, reduced::BlockTailReference};

pub(super) fn with_tail<R>(
    fixture: &TailDefinition,
    layer: usize,
    body: impl FnOnce(BlockTailReference<'_>) -> R,
) -> R {
    let fp32 = |name: &str| fixture.block_parameters[&format!("layers.{layer}.{name}")].fp32();
    let parameters = BlockTailParameters {
        attn_projection: fp32("hc_attn_fn"),
        attn_scale: fp32("hc_attn_scale").try_into().unwrap(),
        attn_base: fp32("hc_attn_base"),
        ffn_projection: fp32("hc_ffn_fn"),
        ffn_scale: fp32("hc_ffn_scale").try_into().unwrap(),
        ffn_base: fp32("hc_ffn_base"),
        attn_norm: fixture.block_parameters[&format!("layers.{layer}.attn_norm.weight")].bf16(),
        ffn_norm: fixture.block_parameters[&format!("layers.{layer}.ffn_norm.weight")].bf16(),
    };
    let config = &fixture.block_config;
    with_model_parameters(
        &fixture.model,
        &fixture.encoded_parameters,
        layer,
        false,
        |moe| {
            let ffn = FfnSublayerReference::new(
                moe,
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
            let tail = BlockTailReference::new(
                ffn,
                &parameters.attn_projection,
                &parameters.attn_scale,
                &parameters.attn_base,
                config.copies,
                config.norm_eps,
                config.hc_sinkhorn_iters,
                config.hc_eps,
            )
            .unwrap();
            body(tail)
        },
    )
}

/// Checks the actual request results under the existing source-derived bounds.
pub(super) fn assert_source_outputs(
    fixture: &Fixture,
    head: &super::super::HeadFixture,
    outputs: &[deepseek::reduced::RequestStepOutput],
) {
    use super::super::{agrees_with_head_oracle, hc_chain_bounds};
    let weights: Vec<_> = head
        .weight_fp32_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    assert_eq!(outputs.len(), fixture.cases.len());
    assert_eq!(outputs.len(), head.cases.len());
    for ((output, case), head_case) in outputs.iter().zip(&fixture.cases).zip(&head.cases) {
        let positions = case.input.shape[1];
        assert_eq!(case.start_pos, head_case.start_pos);
        assert_eq!(output.tails_four().len(), positions);
        assert_eq!(output.heads().len(), positions);
        assert_eq!(output.residual().len(), positions * 256);
        assert_eq!(output.incoming_pre().len(), positions * 2);
        for (position, (tail, result)) in output.tails_four().iter().zip(output.heads()).enumerate()
        {
            assert_eq!(
                &output.residual()[position * 256..(position + 1) * 256],
                tail.ffn().output_bf16()
            );
            assert_eq!(
                &output.incoming_pre()[position * 2..(position + 1) * 2],
                tail.ffn().coefficients().pre()
            );
            let terminal = hc_chain_bounds::check_position(
                fixture,
                case,
                position,
                tail.attention_coefficients(),
                tail.after_attention_bf16(),
                tail.ffn(),
            );
            let envelope = terminal.final_norm_envelope(
                &head.norm_weight_bf16,
                f32::from_bits(head.norm_epsilon_bits),
            );
            let row = position * 128..(position + 1) * 128;
            let source_normalized = &head_case.input_bf16[row.clone()];
            assert!(envelope.accepts(
                result.collapsed_bf16(),
                &head_case.collapsed_bf16[row],
                result.normalized_bf16(),
                source_normalized
            ));
            if position + 1 == positions {
                let bounds = envelope.head_bounds(source_normalized, &weights);
                assert!(agrees_with_head_oracle(
                    result.logits(),
                    &head_case.logits_fp32_bits,
                    &bounds
                ));
            }
        }
    }
}
