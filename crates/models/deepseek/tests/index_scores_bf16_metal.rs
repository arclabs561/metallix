//! Bounded device scorer admission and finite-result contracts.
#![cfg(feature = "metal")]

use std::num::NonZeroUsize;

use deepseek::indexer::bf16::{
    Bf16IndexScoreError, Bf16MetalScoreError, index_scores_bf16_metal, index_scores_bf16_reference,
};

fn dimension(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

#[test]
fn rejects_malformed_and_nonfinite_inputs() {
    assert!(matches!(
        index_scores_bf16_metal(&[], &[0x3f80], &[0x3f80], dimension(1)),
        Err(Bf16MetalScoreError::Reference(
            Bf16IndexScoreError::EmptyInput { field: "query" }
        ))
    ));
    assert!(matches!(
        index_scores_bf16_metal(&[0; 3], &[0; 2], &[0], dimension(2)),
        Err(Bf16MetalScoreError::Reference(
            Bf16IndexScoreError::QueryShape { .. }
        ))
    ));
    assert!(matches!(
        index_scores_bf16_metal(&[0; 2], &[0; 3], &[0], dimension(2)),
        Err(Bf16MetalScoreError::Reference(
            Bf16IndexScoreError::KeyShape { .. }
        ))
    ));
    assert!(matches!(
        index_scores_bf16_metal(&[0; 2], &[0; 2], &[0; 2], dimension(2)),
        Err(Bf16MetalScoreError::Reference(
            Bf16IndexScoreError::HeadWeightCount { .. }
        ))
    ));
    assert!(matches!(
        index_scores_bf16_metal(&[0x7fc1], &[0x3f80], &[0x3f80], dimension(1)),
        Err(Bf16MetalScoreError::Reference(
            Bf16IndexScoreError::NonFiniteInput {
                field: "query",
                position: 0
            }
        ))
    ));
    assert!(matches!(
        index_scores_bf16_metal(
            &[0x3f80; 4097],
            &[0x3f80; 4096],
            &[0x3f80; 4097],
            dimension(1)
        ),
        Err(Bf16MetalScoreError::Reference(
            Bf16IndexScoreError::WorkloadTooLarge { .. }
        ))
    ));
}

#[test]
fn device_rejects_overflow_then_accepts_a_fresh_finite_call() {
    // Exercise three distinct boundaries: dot, signed multiplication, head sum.
    for (query, keys, weights, stage) in [
        (
            vec![0x7f7f],
            vec![0x4000],
            vec![0x3f80],
            Some("dot product"),
        ),
        (vec![0x7f7f], vec![0x3f80], vec![0x4000], Some("weighted")),
        (vec![0x7f7f; 2], vec![0x3f80], vec![0x3f80; 2], None),
    ] {
        let error = index_scores_bf16_metal(&query, &keys, &weights, dimension(1))
            .expect_err("overflow must not escape as a successful diagnostic");
        match (error, stage) {
            (
                Bf16MetalScoreError::Reference(Bf16IndexScoreError::NonFiniteIntermediate {
                    stage: actual,
                    ..
                }),
                Some(expected),
            ) => assert_eq!(actual, expected),
            (
                Bf16MetalScoreError::Reference(Bf16IndexScoreError::NonFiniteOutput {
                    position: 0,
                }),
                None,
            ) => {}
            (error, stage) => panic!("wrong rejection for {stage:?}: {error:?}"),
        }
        // The scorer has no request state; an error must not poison the device
        // for subsequent finite calls. Signed heads cancel exactly here.
        let finite = index_scores_bf16_metal(
            &[0x3f80, 0x4000],
            &[0x4040, 0x4080],
            &[0x4000, 0xbf80],
            dimension(1),
        )
        .unwrap();
        let reference = index_scores_bf16_reference(
            &[0x3f80, 0x4000],
            &[0x4040, 0x4080],
            &[0x4000, 0xbf80],
            dimension(1),
        )
        .unwrap();
        assert_eq!(finite, reference);
        assert_eq!(finite.scores, [0, 0]);
    }
}
