//! Runtime HC-to-attention input contract independent of frozen model fixtures.

use deepseek::reduced::{AttentionInput, AttentionInputError};

#[test]
fn supplied_residual_and_pre_determine_attention_input() {
    let norm = [0x3f80, 0x3f80];
    let input = AttentionInput::new(&norm, 2, 1.0).unwrap();
    let residual = [0x3f80, 0, 0, 0];

    let output = input.forward(&residual, &[1.0, 0.0]).unwrap();
    assert_eq!(output.collapsed_bf16(), &[0x3f80, 0]);
    // RMSNorm([1, 0], eps=1) rounds 1/sqrt(1.5) to BF16 0.81640625.
    assert_eq!(output.normalized_bf16(), &[0x3f51, 0]);

    let second_copy = input.forward(&residual, &[0.0, 1.0]).unwrap();
    assert_eq!(second_copy.collapsed_bf16(), &[0, 0]);
    assert_eq!(second_copy.normalized_bf16(), &[0, 0]);
}

#[test]
fn rejected_runtime_input_leaves_component_reusable() {
    let norm = [0x3f80, 0x3f80];
    let input = AttentionInput::new(&norm, 2, 1.0).unwrap();
    let residual = [0x3f80, 0, 0, 0];
    let expected = input.forward(&residual, &[1.0, 0.0]).unwrap();

    assert!(matches!(
        input.forward(&residual[..3], &[1.0, 0.0]),
        Err(AttentionInputError::Length {
            field: "residual",
            ..
        })
    ));
    assert!(matches!(
        input.forward(&residual, &[f32::NAN, 0.0]),
        Err(AttentionInputError::HcMix(_))
    ));
    assert!(matches!(
        input.forward(&[0x7f80, 0, 0, 0], &[1.0, 0.0]),
        Err(AttentionInputError::HcMix(_))
    ));
    assert_eq!(input.forward(&residual, &[1.0, 0.0]).unwrap(), expected);
}

#[test]
fn invalid_static_operands_and_geometry_are_rejected() {
    assert!(matches!(
        AttentionInput::new(&[], 2, 1.0),
        Err(AttentionInputError::EmptyWidth)
    ));
    assert!(matches!(
        AttentionInput::new(&[0x3f80], 0, 1.0),
        Err(AttentionInputError::EmptyCopies)
    ));
    assert!(matches!(
        AttentionInput::new(&[0x3f80], usize::MAX, 1.0),
        Err(AttentionInputError::CopyCountTooLarge { .. })
    ));
    assert!(matches!(
        AttentionInput::new(&[0x3f80], 1, 0.0),
        Err(AttentionInputError::InvalidEpsilon)
    ));
    assert!(matches!(
        AttentionInput::new(&[0x7f80], 1, 1.0),
        Err(AttentionInputError::NonFiniteNormWeight { element: 0 })
    ));
}
