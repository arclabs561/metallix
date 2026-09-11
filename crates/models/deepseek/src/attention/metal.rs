//! Bounded FP32 Metal diagnostic for V4.1 sparse attention.
//!
//! Sparse indices are deliberately gathered on the host before each small MLX
//! graph. This keeps duplicate and all-masked semantics explicit while
//! qualifying the score, sink-denominator softmax, and value reduction on
//! Metal. It is not device-resident index selection, BF16/TileLang parity,
//! throughput evidence, or a decoder implementation.

use mlx_rs::{
    Array, StreamOrDevice,
    ops::{self, indexing::TryIndexOp},
};
use thiserror::Error;

use super::{SparseAttentionError, SparseAttentionLayout, product, validate_inputs};

/// Largest gathered-KV or returned-output vector accepted by this diagnostic.
///
/// This is an element-count guard, not a bound on the index vector, MLX
/// temporaries, allocator retention, or process memory.
pub const MAX_SPARSE_ATTENTION_METAL_ELEMENTS: usize = 1_048_576;

/// Largest absolute finite logit admitted to the FP32 diagnostic.
///
/// The CPU reference is intentionally FP64 and accepts much larger finite
/// scores. This separate envelope avoids treating an FP32-device overflow as
/// a sparse-attention semantic result.
const MAX_ABSOLUTE_LOGIT: f64 = 80.0;

/// Conservative headroom below the largest finite FP32 accumulator.
///
/// This is a diagnostic admission rule, not a model-wide numerical limit.
const MAX_ABSOLUTE_FP32_ACCUMULATION: f64 = f32::MAX as f64 / 16.0;

/// Errors from the bounded FP32 Metal sparse-attention diagnostic.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SparseAttentionMetalError {
    /// The shared sparse-attention input contract was invalid before GPU work.
    #[error("invalid sparse-attention input: {0}")]
    Input(#[from] SparseAttentionError),
    /// The caller's explicit operation-count limit was exceeded.
    #[error("sparse-attention work estimate {required} exceeds maximum {maximum}")]
    WorkloadTooLarge { required: usize, maximum: usize },
    /// A per-row host gather or complete output exceeds this diagnostic limit.
    #[error("sparse-attention {field} has {elements} FP32 elements, maximum {maximum}")]
    ElementLimit {
        field: &'static str,
        elements: usize,
        maximum: usize,
    },
    /// A Metal shape cannot be represented by MLX's signed dimensions.
    #[error("{field} does not fit MLX's shape representation")]
    DimensionOutOfRange { field: &'static str },
    /// A host staging allocation could not be reserved.
    #[error("could not reserve {elements} host-gather FP32 elements")]
    AllocationFailed { elements: usize },
    /// A finite input would leave the explicitly bounded FP32 logit envelope.
    #[error("{field} at index {index} exceeds the FP32 diagnostic precision envelope")]
    PrecisionEnvelope { field: &'static str, index: usize },
    /// MLX could not construct, evaluate, or read back the Metal graph.
    #[error("MLX Metal sparse-attention evaluation failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
    /// GPU readback contained a non-finite scalar.
    #[error("GPU sparse-attention output at scalar index {index} is not finite")]
    NonFiniteOutput { index: usize },
}

/// Runs a bounded FP32 Metal sparse-attention diagnostic.
///
/// Inputs use [`SparseAttentionLayout`]'s row-major layout. Validation occurs
/// before host gathering or MLX allocation. `max_work` bounds the conservative
/// query-key scalar-product count `batch * query * head * slot * dimension`.
/// It deliberately does not account for CPU preflight or the later
/// probability-value reduction, and is not a live-memory or process-memory
/// bound. Separate fixed element limits bound only the gathered-KV and output
/// vectors; index staging, MLX temporaries, and allocator retention remain out
/// of scope.
///
/// Valid indices are gathered on the host in slot order, so duplicates are
/// retained. Each nonempty row/head then performs FP32 `q @ kv^T`, appends the
/// sink as a denominator-only logit, performs a stable Metal softmax, and
/// reduces probabilities against the same gathered KV. All-masked rows remain
/// zero without submitting an undefined softmax. The precision envelope is
/// deliberately narrower than the CPU FP64 mathematical reference.
pub fn sparse_attention_metal_f32(
    query: &[f32],
    shared_kv: &[f32],
    attn_sink: &[f32],
    indices: &[i32],
    scale: f32,
    layout: SparseAttentionLayout,
    max_work: usize,
) -> Result<Vec<f32>, SparseAttentionMetalError> {
    validate_inputs(query, shared_kv, attn_sink, indices, scale, layout)?;

    let batches = layout.batches.get();
    let queries = layout.query_positions.get();
    let heads = layout.heads.get();
    let dimensions = layout.dimensions.get();
    let keys = layout.key_positions.get();
    let slots = layout.sparse_slots.get();
    let work = product(&[batches, queries, heads, slots, dimensions])?;
    if work > max_work {
        return Err(SparseAttentionMetalError::WorkloadTooLarge {
            required: work,
            maximum: max_work,
        });
    }
    let output_len = layout.output_len()?;
    if output_len > MAX_SPARSE_ATTENTION_METAL_ELEMENTS {
        return Err(SparseAttentionMetalError::ElementLimit {
            field: "output",
            elements: output_len,
            maximum: MAX_SPARSE_ATTENTION_METAL_ELEMENTS,
        });
    }
    let gather_elements = slots
        .checked_mul(dimensions)
        .ok_or(SparseAttentionError::LayoutOverflow)?;
    if gather_elements > MAX_SPARSE_ATTENTION_METAL_ELEMENTS {
        return Err(SparseAttentionMetalError::ElementLimit {
            field: "host gather",
            elements: gather_elements,
            maximum: MAX_SPARSE_ATTENTION_METAL_ELEMENTS,
        });
    }
    validate_precision_envelope(query, shared_kv, attn_sink, indices, scale, layout)?;

    let dimensions_i32 = as_i32(dimensions, "dimensions")?;
    let mut output = Vec::new();
    output.try_reserve_exact(output_len).map_err(|_| {
        SparseAttentionMetalError::AllocationFailed {
            elements: output_len,
        }
    })?;
    output.resize(output_len, 0.0);
    let stream = StreamOrDevice::gpu();

    for batch in 0..batches {
        for query_position in 0..queries {
            let index_base = (batch * queries + query_position) * slots;
            let valid_indices: Vec<_> = indices[index_base..index_base + slots]
                .iter()
                .copied()
                .filter(|index| *index >= 0)
                .filter_map(|index| usize::try_from(index).ok())
                .collect();
            if valid_indices.is_empty() {
                continue;
            }
            let gathered_len = valid_indices
                .len()
                .checked_mul(dimensions)
                .ok_or(SparseAttentionError::LayoutOverflow)?;
            let mut gathered = Vec::new();
            gathered.try_reserve_exact(gathered_len).map_err(|_| {
                SparseAttentionMetalError::AllocationFailed {
                    elements: gathered_len,
                }
            })?;
            for key_index in valid_indices.iter().copied() {
                let key_base = (batch * keys + key_index) * dimensions;
                gathered.extend_from_slice(&shared_kv[key_base..key_base + dimensions]);
            }
            let valid_i32 = as_i32(valid_indices.len(), "valid sparse slots")?;
            let values = Array::from_slice(&gathered, &[valid_i32, dimensions_i32]);

            for (head, &sink) in attn_sink.iter().enumerate() {
                let query_base = ((batch * queries + query_position) * heads + head) * dimensions;
                let query = Array::from_slice(
                    &query[query_base..query_base + dimensions],
                    &[1, dimensions_i32],
                );
                let scores = query.matmul_device(&values.transpose_device(&stream)?, &stream)?;
                let scale = Array::from_slice(&[scale], &[]);
                let scores = scores.multiply_device(&scale, &stream)?;
                let sink = Array::from_slice(&[sink], &[1, 1]);
                let logits = ops::concatenate_axis_device(&[&scores, &sink], 1, &stream)?;
                let probabilities = ops::softmax_axis_device(&logits, 1, true, &stream)?;
                let probabilities = probabilities
                    .try_index_device((0_i32, 0..valid_i32), &stream)?
                    .reshape_device(&[1, valid_i32], &stream)?;
                let row = probabilities.matmul_device(&values, &stream)?;
                row.eval()?;
                let row = row.as_slice::<f32>();
                for (dimension, &value) in row.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(SparseAttentionMetalError::NonFiniteOutput {
                            index: query_base + dimension,
                        });
                    }
                    output[query_base + dimension] = value;
                }
            }
        }
    }
    Ok(output)
}

fn validate_precision_envelope(
    query: &[f32],
    shared_kv: &[f32],
    attn_sink: &[f32],
    indices: &[i32],
    scale: f32,
    layout: SparseAttentionLayout,
) -> Result<(), SparseAttentionMetalError> {
    for (index, &sink) in attn_sink.iter().enumerate() {
        if f64::from(sink).abs() > MAX_ABSOLUTE_LOGIT {
            return Err(SparseAttentionMetalError::PrecisionEnvelope {
                field: "attn_sink",
                index,
            });
        }
    }
    let batches = layout.batches.get();
    let queries = layout.query_positions.get();
    let heads = layout.heads.get();
    let dimensions = layout.dimensions.get();
    let keys = layout.key_positions.get();
    let slots = layout.sparse_slots.get();
    for batch in 0..batches {
        for query_position in 0..queries {
            let index_base = (batch * queries + query_position) * slots;
            for head in 0..heads {
                let query_base = ((batch * queries + query_position) * heads + head) * dimensions;
                for slot in 0..slots {
                    let key_index = indices[index_base + slot];
                    if key_index < 0 {
                        continue;
                    }
                    let Some(key_index) = usize::try_from(key_index).ok() else {
                        continue;
                    };
                    let key_base = (batch * keys + key_index) * dimensions;
                    let score = checked_score(
                        query,
                        query_base,
                        shared_kv,
                        key_base,
                        dimensions,
                        scale,
                        index_base + slot,
                    )?;
                    if !score.is_finite() || score.abs() > MAX_ABSOLUTE_LOGIT {
                        return Err(SparseAttentionMetalError::PrecisionEnvelope {
                            field: "scaled query-KV score",
                            index: index_base + slot,
                        });
                    }
                }
            }
        }
    }
    Ok(())
}

fn checked_score(
    query: &[f32],
    query_base: usize,
    shared_kv: &[f32],
    key_base: usize,
    dimensions: usize,
    scale: f32,
    index: usize,
) -> Result<f64, SparseAttentionMetalError> {
    let mut accumulation = 0.0_f64;
    let mut absolute_accumulation = 0.0_f64;
    for dimension in 0..dimensions {
        let product =
            f64::from(query[query_base + dimension]) * f64::from(shared_kv[key_base + dimension]);
        if !product.is_finite() || product.abs() > MAX_ABSOLUTE_FP32_ACCUMULATION {
            return Err(SparseAttentionMetalError::PrecisionEnvelope {
                field: "query-KV FP32 product",
                index,
            });
        }
        absolute_accumulation += product.abs();
        if !absolute_accumulation.is_finite()
            || absolute_accumulation > MAX_ABSOLUTE_FP32_ACCUMULATION
        {
            return Err(SparseAttentionMetalError::PrecisionEnvelope {
                field: "query-KV FP32 absolute accumulation",
                index,
            });
        }
        accumulation += product;
    }
    Ok(accumulation * f64::from(scale))
}

fn as_i32(value: usize, field: &'static str) -> Result<i32, SparseAttentionMetalError> {
    i32::try_from(value).map_err(|_| SparseAttentionMetalError::DimensionOutOfRange { field })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{SparseAttentionMetalError, sparse_attention_metal_f32};
    use crate::attention::SparseAttentionError;
    use crate::{GPU_TEST_LOCK, SparseAttentionLayout, sparse_attention_reference};

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test dimensions are nonzero")
    }

    fn layout(
        batches: usize,
        queries: usize,
        heads: usize,
        dimensions: usize,
        keys: usize,
        slots: usize,
    ) -> SparseAttentionLayout {
        SparseAttentionLayout::new(
            nonzero(batches),
            nonzero(queries),
            nonzero(heads),
            nonzero(dimensions),
            nonzero(keys),
            nonzero(slots),
        )
        .expect("small test layout")
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 0.000_1_f32 + 0.000_01_f32 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "scalar {index}: actual {actual}, expected {expected}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn metal_matches_reference_for_multibatch_duplicates_masks_and_sink() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let layout = layout(2, 2, 2, 2, 3, 3);
        let query = [
            0.5, -0.25, 0.2, 0.75, -0.4, 0.1, 0.8, -0.2, 0.3, 0.4, -0.6, 0.5, 0.7, -0.3, 0.15, 0.9,
        ];
        let shared_kv = [
            0.1, 0.2, 0.3, -0.4, 0.5, 0.6, -0.2, 0.7, 0.8, -0.9, 0.4, 0.15,
        ];
        // The final row is all masked; the first duplicates key zero.
        let indices = [0, 0, 2, 1, -1, 2, 2, -1, 0, -1, -1, -1];
        let sink = [0.25, -0.5];
        let expected = sparse_attention_reference(&query, &shared_kv, &sink, &indices, 0.5, layout)
            .expect("valid CPU reference");
        let actual =
            sparse_attention_metal_f32(&query, &shared_kv, &sink, &indices, 0.5, layout, 1_000)
                .expect("valid Metal diagnostic");
        assert_close(&actual, &expected);
        assert_eq!(&actual[12..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn invalid_inputs_and_work_budget_fail_before_gpu_allocation() {
        let unit = layout(1, 1, 1, 1, 1, 1);
        assert!(matches!(
            sparse_attention_metal_f32(&[1.0], &[1.0], &[0.0], &[0], 0.0, unit, 1),
            Err(SparseAttentionMetalError::Input(_))
        ));
        assert!(matches!(
            sparse_attention_metal_f32(&[1.0], &[1.0], &[0.0], &[0], 1.0, unit, 0),
            Err(SparseAttentionMetalError::WorkloadTooLarge {
                required: 1,
                maximum: 0
            })
        ));
    }

    #[test]
    fn rejects_finite_scores_outside_the_fp32_precision_envelope() {
        let unit = layout(1, 1, 1, 1, 1, 1);
        assert!(matches!(
            sparse_attention_metal_f32(&[1.0e20], &[1.0e20], &[0.0], &[0], 1.0, unit, 1,),
            Err(SparseAttentionMetalError::PrecisionEnvelope {
                field: "query-KV FP32 product",
                index: 0
            })
        ));
        assert!(matches!(
            sparse_attention_metal_f32(
                &[1.0e20, 1.0e20],
                &[1.0e20, -1.0e20],
                &[0.0],
                &[0],
                1.0,
                layout(1, 1, 1, 2, 1, 1),
                2,
            ),
            Err(SparseAttentionMetalError::PrecisionEnvelope {
                field: "query-KV FP32 product",
                index: 0
            })
        ));
        assert!(matches!(
            sparse_attention_metal_f32(&[1.0e20], &[1.0e20], &[0.0], &[0], 1.0e-40, unit, 1,),
            Err(SparseAttentionMetalError::PrecisionEnvelope {
                field: "query-KV FP32 product",
                index: 0
            })
        ));
        let query = vec![1.0e18_f32; 32];
        let shared_kv: Vec<_> = (0_usize..32)
            .map(|dimension| {
                if dimension.is_multiple_of(2) {
                    1.0e18
                } else {
                    -1.0e18
                }
            })
            .collect();
        assert!(matches!(
            sparse_attention_metal_f32(
                &query,
                &shared_kv,
                &[0.0],
                &[0],
                1.0,
                layout(1, 1, 1, 32, 1, 1),
                32,
            ),
            Err(SparseAttentionMetalError::PrecisionEnvelope {
                field: "query-KV FP32 absolute accumulation",
                index: 0
            })
        ));
    }

    #[test]
    fn metal_sink_denominator_has_the_hand_checked_duplicate_result() {
        let _guard = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let single = sparse_attention_metal_f32(
            &[0.0],
            &[2.0],
            &[0.0],
            &[0],
            1.0,
            layout(1, 1, 1, 1, 1, 1),
            1,
        )
        .expect("single valid slot");
        let duplicate = sparse_attention_metal_f32(
            &[0.0],
            &[2.0],
            &[0.0],
            &[0, 0],
            1.0,
            layout(1, 1, 1, 1, 1, 2),
            2,
        )
        .expect("duplicate valid slots");
        assert_close(&single, &[1.0]);
        assert_close(&duplicate, &[4.0 / 3.0]);
    }

    #[test]
    fn rejects_length_nonfinite_index_and_gather_limit_before_gpu_work() {
        let unit = layout(1, 1, 1, 1, 1, 1);
        assert!(matches!(
            sparse_attention_metal_f32(&[], &[1.0], &[0.0], &[0], 1.0, unit, 1),
            Err(SparseAttentionMetalError::Input(
                SparseAttentionError::LengthMismatch { field: "query", .. }
            ))
        ));
        assert!(matches!(
            sparse_attention_metal_f32(&[f32::NAN], &[1.0], &[0.0], &[0], 1.0, unit, 1),
            Err(SparseAttentionMetalError::Input(
                SparseAttentionError::NonFiniteValue {
                    field: "query",
                    index: 0
                }
            ))
        ));
        assert!(matches!(
            sparse_attention_metal_f32(&[1.0], &[1.0], &[0.0], &[-2], 1.0, unit, 1),
            Err(SparseAttentionMetalError::Input(
                SparseAttentionError::InvalidIndex {
                    slot: 0,
                    index: -2,
                    ..
                }
            ))
        ));
        let slots = super::MAX_SPARSE_ATTENTION_METAL_ELEMENTS + 1;
        let indices = vec![-1_i32; slots];
        assert!(matches!(
            sparse_attention_metal_f32(
                &[1.0],
                &[1.0],
                &[0.0],
                &indices,
                1.0,
                layout(1, 1, 1, 1, 1, slots),
                slots,
            ),
            Err(SparseAttentionMetalError::ElementLimit {
                field: "host gather",
                elements,
                maximum: super::MAX_SPARSE_ATTENTION_METAL_ELEMENTS,
            }) if elements == slots
        ));
    }
}
