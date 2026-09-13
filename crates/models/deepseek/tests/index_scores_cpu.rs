//! CPU scoring and its causal/candidate-selection boundary, without MLX.

use deepseek::{IndexScoreError, candidate_mask, index_scores_reference, select_indices};
use serde::Deserialize;
use std::num::NonZeroUsize;

fn dim(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

#[test]
fn relu_precedes_signed_weighting() {
    let scores = index_scores_reference(
        &[1.0, -1.0, 2.0, 1.0],
        &[3.0, 1.0, -1.0, 2.0],
        &[2.0, -1.0],
        dim(2),
    )
    .unwrap();
    assert_eq!(scores, [-3.0, 0.0]);
}

#[test]
fn reductions_have_explicit_ascending_order() {
    assert_eq!(
        index_scores_reference(&[1e20, -1e20, 1.0], &[1.0; 3], &[1.0], dim(3)).unwrap(),
        [1.0]
    );
    assert_eq!(
        index_scores_reference(&[1.0; 3], &[1.0], &[1e20, -1e20, 1.0], dim(1)).unwrap(),
        [1.0]
    );
}

#[test]
fn rejects_invalid_shapes_and_nonfinite_inputs() {
    assert!(matches!(
        index_scores_reference(&[], &[1.0], &[1.0], dim(1)),
        Err(IndexScoreError::EmptyInput { field: "query" })
    ));
    assert!(matches!(
        index_scores_reference(&[1.0], &[], &[1.0], dim(1)),
        Err(IndexScoreError::EmptyInput { field: "keys" })
    ));
    assert!(matches!(
        index_scores_reference(&[1.0], &[1.0], &[], dim(1)),
        Err(IndexScoreError::EmptyInput {
            field: "head_weights"
        })
    ));
    assert!(matches!(
        index_scores_reference(&[1.0; 3], &[1.0; 2], &[1.0], dim(2)),
        Err(IndexScoreError::QueryShape { .. })
    ));
    assert!(matches!(
        index_scores_reference(&[1.0; 2], &[1.0; 3], &[1.0], dim(2)),
        Err(IndexScoreError::KeyShape { .. })
    ));
    assert!(matches!(
        index_scores_reference(&[1.0; 2], &[1.0], &[1.0], dim(1)),
        Err(IndexScoreError::HeadWeightCount { .. })
    ));
    for (q, k, w, field) in [
        (f32::NAN, 1.0, 1.0, "query"),
        (1.0, f32::INFINITY, 1.0, "keys"),
        (1.0, 1.0, f32::NEG_INFINITY, "head_weights"),
    ] {
        assert!(matches!(
            index_scores_reference(&[q], &[k], &[w], dim(1)),
            Err(IndexScoreError::NonFiniteInput { field: actual, position: 0 }) if actual == field
        ));
    }
}

#[test]
fn rejects_overflow_before_relu_can_hide_it() {
    // A negative overflowing dot must fail, not disappear under ReLU.
    for (query, keys, weights) in [
        (vec![-f32::MAX], vec![2.0], vec![1.0]),
        (vec![f32::MAX; 2], vec![1.0; 2], vec![1.0]),
        (vec![f32::MAX], vec![1.0], vec![2.0]),
    ] {
        assert!(matches!(
            index_scores_reference(&query, &keys, &weights, dim(query.len())),
            Err(IndexScoreError::NonFiniteIntermediate {
                head: 0,
                position: 0
            })
        ));
    }
    assert!(matches!(
        index_scores_reference(&[1.0; 2], &[1.0], &[f32::MAX; 2], dim(1)),
        Err(IndexScoreError::NonFiniteOutput { position: 0 })
    ));
}

#[test]
fn bounds_scalar_work_not_just_score_matrix_size() {
    // Only 65*65 score cells, but 65*65*4097 scalar terms.
    assert!(matches!(
        index_scores_reference(
            &vec![1.0; 65 * 4097],
            &vec![1.0; 65 * 4097],
            &[1.0; 65],
            dim(4097)
        ),
        Err(IndexScoreError::ScalarWorkloadTooLarge { .. })
    ));
}

#[test]
fn scoring_joins_candidate_mask_and_position_sorted_selection() {
    let mut scores =
        index_scores_reference(&[1.0], &[9.0, 2.0, 1.0, 100.0], &[1.0], dim(1)).unwrap();
    // The future key must be masked before candidate-block selection.
    scores[3] = f32::NEG_INFINITY;
    let keep = candidate_mask(&scores, 3, 2, dim(1)).unwrap();
    assert_eq!(keep, [true, false, true, false]);
    for (score, keep) in scores.iter_mut().zip(keep) {
        if !keep {
            *score = f32::NEG_INFINITY;
        }
    }
    assert_eq!(select_indices(&scores, 3, 2, 6).unwrap(), [6, 8]);
}

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    query: Vec<f32>,
    keys: Vec<f32>,
    head_weights: Vec<f32>,
    head_dim: usize,
    expected_scores: Vec<f32>,
}

#[test]
fn matches_existing_pinned_source_score_fixture() {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/index-score-reference.json"
    ))
    .unwrap();
    for case in fixture.cases {
        let scores = index_scores_reference(
            &case.query,
            &case.keys,
            &case.head_weights,
            dim(case.head_dim),
        )
        .unwrap();
        assert_eq!(scores.len(), case.expected_scores.len());
        for (actual, expected) in scores.iter().zip(&case.expected_scores) {
            // Retain the existing Metal fixture consumer's policy, unchanged.
            let tolerance = if case.name == "configured_heads_and_dimension" {
                1e-4_f32 + 1e-5_f32 * expected.abs()
            } else {
                0.0
            };
            assert!(
                (actual - expected).abs() <= tolerance,
                "{}: {actual} != {expected}",
                case.name
            );
        }
    }
}
