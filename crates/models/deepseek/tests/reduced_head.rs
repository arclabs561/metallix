//! Runtime head contract independent of frozen model fixtures.
use deepseek::reduced::FinalHead;

#[test]
fn supplied_residual_pre_and_weights_determine_logits() {
    let norm = [0x3f80, 0x3f80];
    let weights = [1.0, 0.0, 0.0, 1.0];
    let head = FinalHead::new(&norm, &weights, 2, 2, 1.0).unwrap();
    let residual = [0x3f80, 0, 0, 0];
    let output = head.forward(&residual, &[1.0, 0.0]).unwrap();
    assert_eq!(output.collapsed_bf16(), &[0x3f80, 0]);
    // RMSNorm([1,0], eps=1) rounds 1/sqrt(1.5) to BF16 0.81640625.
    assert_eq!(output.normalized_bf16(), &[0x3f51, 0]);
    assert_eq!(output.logits(), &[0.816_406_25, 0.0]);
    assert_eq!(
        head.forward(&residual, &[0.0, 1.0]).unwrap().logits(),
        &[0.0, 0.0]
    );
    let reversed = FinalHead::new(&norm, &[0.0, 1.0, 1.0, 0.0], 2, 2, 1.0).unwrap();
    assert_eq!(
        reversed.forward(&residual, &[1.0, 0.0]).unwrap().logits(),
        &[0.0, 0.816_406_25]
    );
}

#[test]
fn malformed_inputs_do_not_affect_subsequent_execution() {
    let norm = [0x3f80, 0x3f80];
    let weights = [1.0, 0.0];
    let head = FinalHead::new(&norm, &weights, 1, 2, 1.0).unwrap();
    let residual = [0x3f80, 0, 0, 0];
    let expected = head.forward(&residual, &[1.0, 0.0]).unwrap();
    assert!(head.forward(&residual[..3], &[1.0, 0.0]).is_err());
    assert!(head.forward(&residual, &[1.0]).is_err());
    assert!(head.forward(&residual, &[f32::NAN, 0.0]).is_err());
    assert!(head.forward(&[0x7f80, 0, 0, 0], &[1.0, 0.0]).is_err());
    assert_eq!(
        head.forward(&residual, &[1.0, 0.0]).unwrap().logits(),
        expected.logits()
    );
}

#[test]
fn invalid_weights_and_unbounded_geometry_are_rejected() {
    assert!(FinalHead::new(&[], &[], 1, 2, 1.0).is_err());
    assert!(FinalHead::new(&[0x3f80], &[1.0], 1, 0, 1.0).is_err());
    assert!(FinalHead::new(&[0x3f80], &[1.0], usize::MAX, 2, 1.0).is_err());
    assert!(FinalHead::new(&[0x3f80], &[1.0], 1, usize::MAX, 1.0).is_err());
    assert!(FinalHead::new(&[0x3f80], &[1.0], 1, 2, f32::NAN).is_err());
    assert!(FinalHead::new(&[0x3f80], &[1.0], 1, 2, 0.0).is_err());
    assert!(FinalHead::new(&[0x7f80], &[1.0], 1, 2, 1.0).is_err());
    assert!(FinalHead::new(&[0x3f80], &[f32::NAN], 1, 2, 1.0).is_err());
    assert!(FinalHead::new(&[0x3f80], &[], 1, 2, 1.0).is_err());
}
