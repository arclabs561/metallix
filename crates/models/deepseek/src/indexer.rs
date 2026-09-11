//! FP32 Metal qualification for the V4.1 indexer's score-reduction core.
//!
//! This follows the score sequence in the pinned upstream
//! [`Indexer.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L558):
//! query/key dot product, `ReLU`, signed per-head weighting, then head sum.
//! Inputs are already post-RoPE FP32 operands. This is not official BF16/FP4
//! parity, quantization, candidate selection, or a complete indexer.

use std::num::NonZeroUsize;

use mlx_rs::{Array, StreamOrDevice, ops};
use thiserror::Error;

/// Largest permitted `[heads, positions]` core score matrix for this diagnostic.
///
/// This bounds the 64 MiB FP32 matrix itself, not MLX temporary allocations or
/// peak GPU memory.
pub const MAX_INDEX_SCORE_ELEMENTS: usize = 16 * 1024 * 1024;

/// Errors from the bounded V4.1 index-score qualification operation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum IndexScoreError {
    /// The head dimension cannot be represented in MLX's signed shape type.
    #[error("{field} does not fit MLX's shape representation")]
    DimensionOutOfRange { field: &'static str },
    /// The query cannot be divided into complete heads.
    #[error("query length {actual} is not divisible by head dimension {head_dim}")]
    QueryShape { actual: usize, head_dim: usize },
    /// The key buffer cannot be divided into complete key positions.
    #[error("key length {actual} is not divisible by head dimension {head_dim}")]
    KeyShape { actual: usize, head_dim: usize },
    /// There is no query head, key position, or head weight to score.
    #[error("{field} must not be empty")]
    EmptyInput { field: &'static str },
    /// The supplied weights do not provide one signed weight per query head.
    #[error("head weight count {actual} does not equal query head count {expected}")]
    HeadWeightCount { expected: usize, actual: usize },
    /// The core score matrix would exceed this diagnostic's explicit bound.
    #[error(
        "index-score core matrix {heads} heads × {positions} positions exceeds {max_elements} elements"
    )]
    WorkloadTooLarge {
        heads: usize,
        positions: usize,
        max_elements: usize,
    },
    /// A caller supplied a non-finite FP32 operand.
    #[error("{field} at position {position} is not finite")]
    NonFiniteInput {
        field: &'static str,
        position: usize,
    },
    /// MLX could not construct, evaluate, or read back the GPU graph.
    #[error("MLX Metal index-score evaluation failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// GPU readback did not produce finite scores.
    #[error("GPU index score at position {position} is not finite")]
    NonFiniteOutput { position: usize },
}

/// Computes one query's V4.1 index scores on the GPU from post-RoPE FP32 inputs.
///
/// `query` is row-major `[heads, head_dim]`; `keys` is row-major
/// `[positions, head_dim]`; and `head_weights` supplies one signed scalar per
/// head. The operation is GPU-only after CPU validation: `q @ kᵀ`, `ReLU`,
/// signed head weighting, and reduction across heads.
pub fn index_scores_f32(
    query: &[f32],
    keys: &[f32],
    head_weights: &[f32],
    head_dim: NonZeroUsize,
) -> Result<Vec<f32>, IndexScoreError> {
    let dim = head_dim.get();
    if query.is_empty() {
        return Err(IndexScoreError::EmptyInput { field: "query" });
    }
    if keys.is_empty() {
        return Err(IndexScoreError::EmptyInput { field: "keys" });
    }
    if head_weights.is_empty() {
        return Err(IndexScoreError::EmptyInput {
            field: "head_weights",
        });
    }
    if !query.len().is_multiple_of(dim) {
        return Err(IndexScoreError::QueryShape {
            actual: query.len(),
            head_dim: dim,
        });
    }
    if !keys.len().is_multiple_of(dim) {
        return Err(IndexScoreError::KeyShape {
            actual: keys.len(),
            head_dim: dim,
        });
    }
    let heads = query.len() / dim;
    let positions = keys.len() / dim;
    if head_weights.len() != heads {
        return Err(IndexScoreError::HeadWeightCount {
            expected: heads,
            actual: head_weights.len(),
        });
    }
    if heads
        .checked_mul(positions)
        .is_none_or(|elements| elements > MAX_INDEX_SCORE_ELEMENTS)
    {
        return Err(IndexScoreError::WorkloadTooLarge {
            heads,
            positions,
            max_elements: MAX_INDEX_SCORE_ELEMENTS,
        });
    }
    validate_finite(query, "query")?;
    validate_finite(keys, "keys")?;
    validate_finite(head_weights, "head_weights")?;

    let heads_i32 = as_i32(heads, "heads")?;
    let positions_i32 = as_i32(positions, "positions")?;
    let dim_i32 = as_i32(dim, "head_dim")?;
    let stream = StreamOrDevice::gpu();
    let query = Array::from_slice(query, &[heads_i32, dim_i32]);
    let keys = Array::from_slice(keys, &[positions_i32, dim_i32]);
    let weights = Array::from_slice(head_weights, &[heads_i32, 1]);
    let dot = query.matmul_device(&keys.transpose_device(&stream)?, &stream)?;
    let zero = Array::from_slice(&[0.0_f32], &[]);
    let rectified = ops::maximum_device(&dot, &zero, &stream)?;
    let weighted = rectified.multiply_device(&weights, &stream)?;
    let scores = weighted.sum_axis_device(0, false, &stream)?;
    scores.eval()?;
    let output = scores.as_slice::<f32>().to_vec();
    for (position, &score) in output.iter().enumerate() {
        if !score.is_finite() {
            return Err(IndexScoreError::NonFiniteOutput { position });
        }
    }
    Ok(output)
}

fn as_i32(value: usize, field: &'static str) -> Result<i32, IndexScoreError> {
    i32::try_from(value).map_err(|_| IndexScoreError::DimensionOutOfRange { field })
}

fn validate_finite(values: &[f32], field: &'static str) -> Result<(), IndexScoreError> {
    values
        .iter()
        .position(|value| !value.is_finite())
        .map_or(Ok(()), |position| {
            Err(IndexScoreError::NonFiniteInput { field, position })
        })
}

#[cfg(test)]
mod tests {
    use super::{IndexScoreError, index_scores_f32};
    use serde::Deserialize;
    use std::num::NonZeroUsize;

    fn dim(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    #[derive(Debug, Deserialize)]
    struct Fixture {
        schema_version: u8,
        source: FixtureSource,
        cases: Vec<FixtureCase>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureSource {
        revision: String,
        sha256: String,
        symbol: String,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureCase {
        name: String,
        query: Vec<f32>,
        keys: Vec<f32>,
        head_weights: Vec<f32>,
        head_dim: usize,
        expected_scores: Vec<f32>,
    }

    #[test]
    fn rejects_invalid_cpu_inputs_before_gpu_work() {
        assert!(matches!(
            index_scores_f32(&[], &[1.0], &[1.0], dim(1)),
            Err(IndexScoreError::EmptyInput { field: "query" })
        ));
        assert!(matches!(
            index_scores_f32(&[1.0, 2.0, 3.0], &[1.0, 2.0], &[1.0], dim(2)),
            Err(IndexScoreError::QueryShape { .. })
        ));
        assert!(matches!(
            index_scores_f32(&[1.0, 2.0], &[1.0, 2.0, 3.0], &[1.0], dim(2)),
            Err(IndexScoreError::KeyShape { .. })
        ));
        assert!(matches!(
            index_scores_f32(&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0], dim(2)),
            Err(IndexScoreError::HeadWeightCount { .. })
        ));
        assert!(matches!(
            index_scores_f32(&[f32::NAN], &[1.0], &[1.0], dim(1)),
            Err(IndexScoreError::NonFiniteInput { field: "query", .. })
        ));
    }

    #[test]
    fn rejects_a_large_core_matrix_before_gpu_work() {
        let query = vec![1.0; 4_097];
        let keys = vec![1.0; 4_096];
        let head_weights = vec![1.0; 4_097];
        assert!(matches!(
            index_scores_f32(&query, &keys, &head_weights, dim(1)),
            Err(IndexScoreError::WorkloadTooLarge {
                heads: 4_097,
                positions: 4_096,
                max_elements: super::MAX_INDEX_SCORE_ELEMENTS,
            })
        ));
    }

    #[test]
    fn gpu_scores_relu_before_signed_weighting_and_summing_heads() {
        let _guard = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let scores = index_scores_f32(
            &[1.0, -1.0, 2.0, 1.0],
            &[3.0, 1.0, -1.0, 2.0],
            &[2.0, -1.0],
            dim(2),
        )
        .unwrap();
        // Dot products are [2, -3] and [7, 0]; ReLU then signed weighting
        // produces [2*2 + 7*(-1), 0] = [-3, 0].
        assert_eq!(scores, [-3.0, 0.0]);
    }

    #[test]
    fn matches_pinned_official_cpu_score_fixture() {
        let _guard = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/index-score-reference.json"
        ))
        .expect("fixture JSON is valid");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.source.revision,
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        assert_eq!(
            fixture.source.sha256,
            "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65"
        );
        assert_eq!(fixture.source.symbol, "Indexer.forward:index_score");
        assert_eq!(fixture.cases.len(), 5);

        for case in fixture.cases {
            let actual = index_scores_f32(
                &case.query,
                &case.keys,
                &case.head_weights,
                dim(case.head_dim),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(actual.len(), case.expected_scores.len(), "{}", case.name);
            for (actual, expected) in actual.iter().zip(&case.expected_scores) {
                if case.name == "configured_heads_and_dimension" {
                    let tolerance = 1e-4_f32 + 1e-5_f32 * expected.abs();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "{}: actual {actual}, expected {expected}, tolerance {tolerance}",
                        case.name
                    );
                } else {
                    assert!(
                        exact_except_zero_sign(*actual, *expected),
                        "{}: actual {actual:?}, expected {expected:?}",
                        case.name
                    );
                }
            }
        }
    }

    fn exact_except_zero_sign(actual: f32, expected: f32) -> bool {
        actual.to_bits() == expected.to_bits()
            || (actual.abs().to_bits() == 0 && expected.abs().to_bits() == 0)
    }
}
