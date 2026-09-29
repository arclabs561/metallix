//! Alternate-partition source oracle for the fixed reduced request runtime.

use deepseek::reduced::RequestStepOutput;

use super::super::{
    HeadFixture, agrees_with_head_oracle, alternate_partition_l4_fixture, hc_chain_bounds,
};

/// Checks live 4/1/1/1 request outputs against the independently captured
/// alternate L4 tail and final-head oracle.
pub(super) fn assert_source_outputs(outputs: &[RequestStepOutput], canonical_head: &HeadFixture) {
    let alternate = alternate_partition_l4_fixture();
    let head = &alternate.head;
    let head_weight = head.head_weight.fp32();
    let canonical_weight: Vec<_> = canonical_head
        .weight_fp32_bits
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    assert_eq!(head.norm_weight.bf16(), canonical_head.norm_weight_bf16);
    assert_eq!(head_weight, canonical_weight);
    assert_eq!(
        head.norm_epsilon.to_bits(),
        canonical_head.norm_epsilon_bits
    );
    assert_eq!(head.head_weight.shape, canonical_head.weight_shape);
    assert_eq!(outputs.len(), alternate.tail.cases.len());
    assert_eq!(outputs.len(), head.cases.len());

    for ((output, tail_case), head_case) in
        outputs.iter().zip(&alternate.tail.cases).zip(&head.cases)
    {
        let positions = tail_case.input.shape[1];
        assert_eq!(tail_case.start_pos, head_case.start_pos);
        assert_eq!(output.tails_four().len(), positions);
        assert_eq!(output.heads().len(), positions);
        assert_eq!(output.residual().len(), positions * 256);
        assert_eq!(output.incoming_pre().len(), positions * 2);
        assert_eq!(head_case.norm_input.shape, [1, positions, 128]);
        assert_eq!(head_case.norm.shape, head_case.norm_input.shape);
        assert_eq!(head_case.logits.shape, [1, head.head_weight.shape[0]]);

        let source_collapsed = head_case.norm_input.bf16();
        let source_normalized = head_case.norm.bf16();
        let source_logits: Vec<_> = head_case
            .logits
            .fp32()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let mut last_bounds = Vec::new();
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
                &alternate.tail,
                tail_case,
                position,
                tail.attention_coefficients(),
                tail.after_attention_bf16(),
                tail.ffn(),
            );
            let envelope =
                terminal.final_norm_envelope(&head.norm_weight.bf16(), head.norm_epsilon);
            let row = position * 128..(position + 1) * 128;
            assert!(envelope.accepts(
                result.collapsed_bf16(),
                &source_collapsed[row.clone()],
                result.normalized_bf16(),
                &source_normalized[row.clone()],
            ));
            assert!(!envelope.accepts(
                result.collapsed_bf16(),
                &source_collapsed[row.clone()],
                &[0; 128],
                &source_normalized[row],
            ));
            last_bounds = envelope.head_bounds(
                &source_normalized[position * 128..(position + 1) * 128],
                &head_weight,
            );
        }
        let last = output
            .heads()
            .last()
            .expect("alternate request has one head row");
        assert!(
            agrees_with_head_oracle(last.logits(), &source_logits, &last_bounds),
            "alternate request final logits at start {}",
            head_case.start_pos
        );
    }
}
