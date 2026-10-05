//! Runtime head contract independent of frozen model fixtures.
use deepseek::reduced::{FinalHead, FinalHeadError, HeadWeights};

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

/// Deterministic finite BF16 bits spanning signs and several binades.
fn bf16_values(count: usize, seed: u32) -> Vec<u16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Sign, exponent 120..=134 (about 2^-7..2^7), any mantissa.
            let sign = u16::from(state >> 31 == 1) << 15;
            let exponent = u16::try_from(120 + (state >> 16) % 15).unwrap() << 7;
            let mantissa = u16::try_from(state & 0x7f).unwrap();
            sign | exponent | mantissa
        })
        .collect()
}

#[test]
fn bf16_head_weights_produce_the_fp32_logits_bit_for_bit() {
    for (vocabulary, width, copies) in [(1, 32, 1), (7, 64, 2), (33, 160, 4)] {
        let norm = bf16_values(width, 7);
        let head_bf16 = bf16_values(vocabulary * width, 11);
        let head_f32: Vec<f32> = head_bf16
            .iter()
            .map(|&bits| f32::from_bits(u32::from(bits) << 16))
            .collect();
        let residual = bf16_values(copies * width, 13);
        let pre: Vec<f32> = [1.0, 0.5, 0.25, 0.125][..copies].to_vec();
        let fp32 = FinalHead::new(&norm, &head_f32, vocabulary, copies, 1e-6).unwrap();
        let bf16 = FinalHead::with_weights(
            &norm,
            HeadWeights::Bf16(&head_bf16),
            vocabulary,
            copies,
            1e-6,
        )
        .unwrap();
        let (expected, actual) = (
            fp32.forward(&residual, &pre).unwrap(),
            bf16.forward(&residual, &pre).unwrap(),
        );
        assert_eq!(expected.normalized_bf16(), actual.normalized_bf16());
        assert!(
            expected
                .logits()
                .iter()
                .map(|value| value.to_bits())
                .eq(actual.logits().iter().map(|value| value.to_bits())),
            "{vocabulary}x{width}"
        );
    }
}

#[test]
fn bf16_head_caps_and_weights_are_checked_per_variant() {
    let norm = [0x3f80; 32];
    // 2^22 / 32 + 1 rows exceed the FP32 weight cap but fit the BF16 one.
    let vocabulary = (1 << 22) / 32 + 1;
    assert!(matches!(
        FinalHead::new(&norm, &[], vocabulary, 1, 1.0),
        Err(FinalHeadError::ElementLimit { .. })
    ));
    assert!(matches!(
        FinalHead::with_weights(&norm, HeadWeights::Bf16(&[]), vocabulary, 1, 1.0),
        Err(FinalHeadError::Length { expected, .. }) if expected == vocabulary * 32
    ));
    assert!(matches!(
        FinalHead::with_weights(&norm, HeadWeights::Bf16(&[]), (1 << 30) / 32 + 1, 1, 1.0),
        Err(FinalHeadError::ElementLimit { .. })
    ));
    let mut weights = vec![0x3f80; 32];
    weights[5] = 0x7fc0;
    assert!(matches!(
        FinalHead::with_weights(&norm, HeadWeights::Bf16(&weights), 1, 1, 1.0),
        Err(FinalHeadError::NonFiniteHeadWeight { element: 5 })
    ));
}
