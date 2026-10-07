//! Source-compatible BF16 staging for one V4.1 index-score query.
//!
//! This narrow reference owns only the numerical chain after query preparation:
//! BF16 dot products, `ReLU`, signed per-head weighting, and head reduction.
//! Causal masking, candidate filtering, cache publication, and final selection
//! remain caller-owned boundaries.

use std::num::NonZeroUsize;

use thiserror::Error;

#[cfg(feature = "metal")]
use mlx_rs::{Array, Dtype, StreamOrDevice, ops, transforms};

use crate::precision::{
    Bf16LinearError, MAX_BF16_LINEAR_ELEMENTS, bf16_linear_reference, bf16_to_f32, f32_to_bf16_rne,
};

use super::{MAX_INDEX_REFERENCE_TERMS, MAX_INDEX_SCORE_ELEMENTS};

/// The BF16 intermediates from one source-compatible V4.1 index-score query.
///
/// The first three fields are row-major `[heads, positions]`; [`Self::scores`]
/// is the final `[positions]` BF16 head reduction.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct Bf16IndexScoreDiagnostic {
    /// BF16 query/key dot products before rectification.
    pub dot_products: Vec<u16>,
    /// BF16 `ReLU(dot_products)`, preserving a negative-zero input bit pattern.
    pub rectified: Vec<u16>,
    /// BF16 signed per-head products after rectification.
    pub weighted: Vec<u16>,
    /// BF16 scores after ascending FP32 head reduction and one final narrowing.
    pub scores: Vec<u16>,
}

/// Errors from the bounded BF16 V4.1 index-score reference.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum Bf16IndexScoreError {
    /// A required input has no elements.
    #[error("BF16 index-score {field} must not be empty")]
    EmptyInput {
        /// The rejected input name.
        field: &'static str,
    },
    /// The query does not contain an integral number of heads.
    #[error("BF16 index-score query length {actual} is not divisible by head dimension {head_dim}")]
    QueryShape {
        /// The supplied BF16 element count.
        actual: usize,
        /// The requested per-head BF16 width.
        head_dim: usize,
    },
    /// The key storage does not contain an integral number of key positions.
    #[error("BF16 index-score key length {actual} is not divisible by head dimension {head_dim}")]
    KeyShape {
        /// The supplied BF16 element count.
        actual: usize,
        /// The requested per-key BF16 width.
        head_dim: usize,
    },
    /// The signed head-weight count does not equal the query-head count.
    #[error(
        "BF16 index-score head-weight count {actual} does not equal query-head count {expected}"
    )]
    HeadWeightCount {
        /// The number of heads derived from the query shape.
        expected: usize,
        /// The supplied weight count.
        actual: usize,
    },
    /// Checked shape or work arithmetic overflowed `usize`.
    #[error("BF16 index-score shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// The derived quantity that overflowed.
        field: &'static str,
    },
    /// The logical `[heads, positions]` score matrix exceeds the shared cap.
    #[error(
        "BF16 index-score matrix {heads} heads × {positions} positions exceeds {max_elements} elements"
    )]
    WorkloadTooLarge {
        /// Query heads.
        heads: usize,
        /// Key positions.
        positions: usize,
        /// Shared score-matrix limit.
        max_elements: usize,
    },
    /// Scalar dot-product work exceeds the shared CPU reference cap.
    #[error("BF16 index-score scalar work exceeds {max_terms} terms")]
    ScalarWorkloadTooLarge {
        /// Shared scalar-work limit.
        max_terms: usize,
    },
    /// A supplied BF16 value denotes NaN or infinity.
    #[error("nonfinite BF16 {field} at position {position}")]
    NonFiniteInput {
        /// The rejected input name.
        field: &'static str,
        /// Flat BF16 storage position.
        position: usize,
    },
    /// A bounded staging buffer could not be reserved.
    #[error("could not allocate {elements} BF16 index-score {field} elements")]
    AllocationFailed {
        /// The staging-buffer name.
        field: &'static str,
        /// Requested BF16 elements.
        elements: usize,
    },
    /// The underlying BF16 dot-product stage rejected the validated request.
    #[error("BF16 index-score dot-product stage failed: {0}")]
    Linear(#[from] Bf16LinearError),
    /// A finite BF16 input overflowed while forming a staged scalar result.
    #[error("nonfinite BF16 index-score {stage} at head {head}, position {position}")]
    NonFiniteIntermediate {
        /// The scalar stage that overflowed.
        stage: &'static str,
        /// Query-head index.
        head: usize,
        /// Key-position index.
        position: usize,
    },
    /// A final per-position head reduction or BF16 narrowing was nonfinite.
    #[error("nonfinite BF16 index score at position {position}")]
    NonFiniteOutput {
        /// Final key-position index.
        position: usize,
    },
}

/// Errors from the bounded Metal BF16 score-stage qualification.
#[cfg(feature = "metal")]
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Bf16MetalScoreError {
    /// Shared BF16 shape, finite-input, allocation, or stage validation failed.
    #[error(transparent)]
    Reference(#[from] Bf16IndexScoreError),
    /// MLX could not construct, evaluate, or read the Metal graph.
    #[error("MLX Metal BF16 index-score evaluation failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
}

/// Computes source-compatible BF16 V4.1 index scores for one query position.
///
/// `query` is row-major `[heads, head_dim]`, `keys` is row-major
/// `[positions, head_dim]`, and `head_weights` supplies one already-scaled,
/// signed BF16 scalar per head. The dot product, rectification, signed product,
/// and final head sum each retain their observed BF16 boundary. Head summation
/// is scalar FP32 in ascending head order, narrowed once to BF16.
///
/// This is not a cache, mask, selection, generic tensor, or backend API.
/// It is a scalar precision-staging reference qualified against the synthetic
/// source fixture, not a bit-parity claim for arbitrary `PyTorch`/hardware
/// GEMM reductions or a GPU/throughput qualification.
/// Inputs must be finite; all shape/work checks, including the effective
/// [`MAX_BF16_LINEAR_ELEMENTS`] per-buffer limit of the reused dot primitive,
/// complete before any output buffer allocation.
///
/// # Errors
///
/// * [`Bf16IndexScoreError::EmptyInput`], [`Bf16IndexScoreError::QueryShape`],
///   [`Bf16IndexScoreError::KeyShape`] and
///   [`Bf16IndexScoreError::HeadWeightCount`] when the inputs do not form
///   whole heads and key positions with one weight per head.
/// * [`Bf16IndexScoreError::ShapeOverflow`],
///   [`Bf16IndexScoreError::WorkloadTooLarge`] and
///   [`Bf16IndexScoreError::ScalarWorkloadTooLarge`] past the work caps.
/// * [`Bf16IndexScoreError::NonFiniteInput`] for a NaN or infinite input.
/// * [`Bf16IndexScoreError::AllocationFailed`] when staging cannot be
///   reserved.
/// * [`Bf16IndexScoreError::Linear`],
///   [`Bf16IndexScoreError::NonFiniteIntermediate`] and
///   [`Bf16IndexScoreError::NonFiniteOutput`] when a stage overflows.
pub fn index_scores_bf16_reference(
    query: &[u16],
    keys: &[u16],
    head_weights: &[u16],
    head_dim: NonZeroUsize,
) -> Result<Bf16IndexScoreDiagnostic, Bf16IndexScoreError> {
    let shape = Shape::new(query, keys, head_weights, head_dim)?;
    validate_finite(query, "query")?;
    validate_finite(keys, "keys")?;
    validate_finite(head_weights, "head_weights")?;

    let mut dot_products = reserve(shape.matrix_elements, "dot_products")?;
    bf16_linear_reference(
        query,
        keys,
        shape.heads,
        shape.dimension,
        shape.positions,
        &mut dot_products,
    )?;

    let mut rectified = reserve(shape.matrix_elements, "rectified")?;
    for (flat, (&dot, output)) in dot_products.iter().zip(&mut rectified).enumerate() {
        let value = bf16_to_f32(dot);
        if !value.is_finite() {
            return Err(shape.intermediate_error("dot product", flat));
        }
        *output = relu_bf16(dot);
    }

    let mut weighted = reserve(shape.matrix_elements, "weighted")?;
    for (head, &weight_bits) in head_weights.iter().enumerate() {
        let weight = bf16_to_f32(weight_bits);
        for position in 0..shape.positions {
            let flat = head * shape.positions + position;
            let product = bf16_to_f32(rectified[flat]) * weight;
            if !product.is_finite() {
                return Err(Bf16IndexScoreError::NonFiniteIntermediate {
                    stage: "weighted product",
                    head,
                    position,
                });
            }
            let bits = f32_to_bf16_rne(product);
            if !bf16_to_f32(bits).is_finite() {
                return Err(Bf16IndexScoreError::NonFiniteIntermediate {
                    stage: "weighted BF16 output",
                    head,
                    position,
                });
            }
            weighted[flat] = bits;
        }
    }

    let mut scores = reserve(shape.positions, "scores")?;
    for (position, output) in scores.iter_mut().enumerate() {
        let mut sum = 0.0_f32;
        for head in 0..shape.heads {
            sum += bf16_to_f32(weighted[head * shape.positions + position]);
            if !sum.is_finite() {
                return Err(Bf16IndexScoreError::NonFiniteIntermediate {
                    stage: "head sum",
                    head,
                    position,
                });
            }
        }
        let bits = f32_to_bf16_rne(sum);
        if !bf16_to_f32(bits).is_finite() {
            return Err(Bf16IndexScoreError::NonFiniteOutput { position });
        }
        *output = bits;
    }

    Ok(Bf16IndexScoreDiagnostic {
        dot_products,
        rectified,
        weighted,
        scores,
    })
}

/// Qualifies the V4.1 BF16 index-score stages on Metal for one prepared query.
///
/// This is a bounded device path, not a general backend replacement or arbitrary
/// hardware-parity claim. `query` is `[heads, head_dim]`, `keys` is
/// `[positions, head_dim]`, and `head_weights` is `[heads]`. It retains the
/// scalar reference's shared CPU work and BF16-buffer caps, keeps BF16 after
/// dot, rectification, and signed weighting, promotes only the head reduction
/// to FP32, and narrows its result back to BF16. The scalar reference remains
/// the source staging authority.
///
/// # Errors
///
/// Returns [`Bf16MetalScoreError::Reference`] for any input the scalar
/// reference rejects, or a non-finite device result, and
/// [`Bf16MetalScoreError::Mlx`] when MLX cannot build, evaluate or read the
/// graph.
#[cfg(feature = "metal")]
pub fn index_scores_bf16_metal(
    query: &[u16],
    keys: &[u16],
    head_weights: &[u16],
    head_dim: NonZeroUsize,
) -> Result<Bf16IndexScoreDiagnostic, Bf16MetalScoreError> {
    let shape = Shape::new(query, keys, head_weights, head_dim)?;
    validate_finite(query, "query")?;
    validate_finite(keys, "keys")?;
    validate_finite(head_weights, "head_weights")?;
    let heads = metal_dimension(shape.heads, "heads")?;
    let positions = metal_dimension(shape.positions, "positions")?;
    let dimension = metal_dimension(shape.dimension, "head dimension")?;
    let _device = crate::device_lock();
    let stream = StreamOrDevice::gpu();
    let query = metal_bf16_array(query, &[heads, dimension], &stream)?;
    let keys = metal_bf16_array(keys, &[positions, dimension], &stream)?;
    let weights = metal_bf16_array(head_weights, &[heads, 1], &stream)?;
    let zero = metal_bf16_array(&[0], &[], &stream)?;
    let dot = query.matmul_device(keys.transpose_device(&stream)?, &stream)?;
    let negative = dot.lt_device(&zero, &stream)?;
    let rectified = ops::r#where_device(&negative, &zero, &dot, &stream)?;
    let weighted = rectified.multiply_device(&weights, &stream)?;
    let scores = weighted
        .as_dtype_device(Dtype::Float32, &stream)?
        .sum_axis_device(0, false, &stream)?
        .as_dtype_device(Dtype::Bfloat16, &stream)?;
    // Each Uint16 view is an observable BF16 stage boundary. Evaluate them
    // together before host reads so MLX completes one graph while retaining
    // all four staged diagnostics and their original narrowing points.
    let dot_words = metal_bf16_words(&dot, &stream)?;
    let rectified_words = metal_bf16_words(&rectified, &stream)?;
    let weighted_words = metal_bf16_words(&weighted, &stream)?;
    let score_words = metal_bf16_words(&scores, &stream)?;
    transforms::eval([&dot_words, &rectified_words, &weighted_words, &score_words])?;

    // Preserve the source-stage error order after the shared device barrier.
    let dot_products = metal_bf16_read(&dot_words, shape.matrix_elements, "dot_products")?;
    validate_metal_matrix(&dot_products, shape, "dot product")?;
    let rectified_bits = metal_bf16_read(&rectified_words, shape.matrix_elements, "rectified")?;
    validate_metal_matrix(&rectified_bits, shape, "rectified")?;
    let weighted_bits = metal_bf16_read(&weighted_words, shape.matrix_elements, "weighted")?;
    validate_metal_matrix(&weighted_bits, shape, "weighted")?;
    let scores = metal_bf16_read(&score_words, shape.positions, "scores")?;
    validate_metal_scores(&scores)?;
    Ok(Bf16IndexScoreDiagnostic {
        dot_products,
        rectified: rectified_bits,
        weighted: weighted_bits,
        scores,
    })
}

#[cfg(feature = "metal")]
fn metal_bf16_array(
    values: &[u16],
    shape: &[i32],
    stream: &StreamOrDevice,
) -> Result<Array, Bf16MetalScoreError> {
    Ok(Array::from_slice(values, shape).view_dtype_device(Dtype::Bfloat16, stream)?)
}

#[cfg(feature = "metal")]
fn metal_bf16_read(
    words: &Array,
    elements: usize,
    field: &'static str,
) -> Result<Vec<u16>, Bf16MetalScoreError> {
    let words = words.as_slice::<u16>();
    if words.len() != elements {
        return Err(Bf16IndexScoreError::Linear(Bf16LinearError::Length {
            field,
            actual: words.len(),
            expected: elements,
        })
        .into());
    }
    let mut output = reserve(elements, field)?;
    output.copy_from_slice(words);
    Ok(output)
}

/// Returns an observable uint16 view for one BF16 stage before joint evaluation.
#[cfg(feature = "metal")]
fn metal_bf16_words(array: &Array, stream: &StreamOrDevice) -> Result<Array, Bf16MetalScoreError> {
    Ok(array.view_dtype_device(Dtype::Uint16, stream)?)
}

#[cfg(feature = "metal")]
fn validate_metal_matrix(
    values: &[u16],
    shape: Shape,
    stage: &'static str,
) -> Result<(), Bf16MetalScoreError> {
    if let Some(flat) = values
        .iter()
        .position(|&bits| !bf16_to_f32(bits).is_finite())
    {
        return Err(shape.intermediate_error(stage, flat).into());
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn validate_metal_scores(values: &[u16]) -> Result<(), Bf16MetalScoreError> {
    if let Some(position) = values
        .iter()
        .position(|&bits| !bf16_to_f32(bits).is_finite())
    {
        return Err(Bf16IndexScoreError::NonFiniteOutput { position }.into());
    }
    Ok(())
}

#[cfg(feature = "metal")]
fn metal_dimension(value: usize, field: &'static str) -> Result<i32, Bf16MetalScoreError> {
    i32::try_from(value).map_err(|_| Bf16IndexScoreError::ShapeOverflow { field }.into())
}

#[derive(Clone, Copy)]
struct Shape {
    heads: usize,
    positions: usize,
    dimension: usize,
    matrix_elements: usize,
}

impl Shape {
    fn new(
        query: &[u16],
        keys: &[u16],
        head_weights: &[u16],
        head_dim: NonZeroUsize,
    ) -> Result<Self, Bf16IndexScoreError> {
        if query.is_empty() {
            return Err(Bf16IndexScoreError::EmptyInput { field: "query" });
        }
        if keys.is_empty() {
            return Err(Bf16IndexScoreError::EmptyInput { field: "keys" });
        }
        if head_weights.is_empty() {
            return Err(Bf16IndexScoreError::EmptyInput {
                field: "head_weights",
            });
        }
        let dimension = head_dim.get();
        if !query.len().is_multiple_of(dimension) {
            return Err(Bf16IndexScoreError::QueryShape {
                actual: query.len(),
                head_dim: dimension,
            });
        }
        if !keys.len().is_multiple_of(dimension) {
            return Err(Bf16IndexScoreError::KeyShape {
                actual: keys.len(),
                head_dim: dimension,
            });
        }
        let heads = query.len() / dimension;
        let positions = keys.len() / dimension;
        if head_weights.len() != heads {
            return Err(Bf16IndexScoreError::HeadWeightCount {
                expected: heads,
                actual: head_weights.len(),
            });
        }
        let matrix_elements =
            heads
                .checked_mul(positions)
                .ok_or(Bf16IndexScoreError::ShapeOverflow {
                    field: "score matrix",
                })?;
        if matrix_elements > MAX_INDEX_SCORE_ELEMENTS {
            return Err(Bf16IndexScoreError::WorkloadTooLarge {
                heads,
                positions,
                max_elements: MAX_INDEX_SCORE_ELEMENTS,
            });
        }
        let work =
            matrix_elements
                .checked_mul(dimension)
                .ok_or(Bf16IndexScoreError::ShapeOverflow {
                    field: "scalar work",
                })?;
        if work > MAX_INDEX_REFERENCE_TERMS {
            return Err(Bf16IndexScoreError::ScalarWorkloadTooLarge {
                max_terms: MAX_INDEX_REFERENCE_TERMS,
            });
        }
        for (field, elements) in [
            ("activations", query.len()),
            ("weights", keys.len()),
            ("output", matrix_elements),
        ] {
            if elements > MAX_BF16_LINEAR_ELEMENTS {
                return Err(Bf16IndexScoreError::Linear(Bf16LinearError::ElementLimit {
                    field,
                    elements,
                    maximum: MAX_BF16_LINEAR_ELEMENTS,
                }));
            }
        }
        Ok(Self {
            heads,
            positions,
            dimension,
            matrix_elements,
        })
    }

    fn intermediate_error(self, stage: &'static str, flat: usize) -> Bf16IndexScoreError {
        Bf16IndexScoreError::NonFiniteIntermediate {
            stage,
            head: flat / self.positions,
            position: flat % self.positions,
        }
    }
}

fn validate_finite(values: &[u16], field: &'static str) -> Result<(), Bf16IndexScoreError> {
    values
        .iter()
        .position(|&bits| !bf16_to_f32(bits).is_finite())
        .map_or(Ok(()), |position| {
            Err(Bf16IndexScoreError::NonFiniteInput { field, position })
        })
}

fn relu_bf16(bits: u16) -> u16 {
    // Pinned Torch BF16 `relu_` retains -0.0 but clears strictly negative inputs.
    if bf16_to_f32(bits) < 0.0 { 0 } else { bits }
}

fn reserve(elements: usize, field: &'static str) -> Result<Vec<u16>, Bf16IndexScoreError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| Bf16IndexScoreError::AllocationFailed { field, elements })?;
    output.resize(elements, 0);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        Bf16IndexScoreError, Bf16LinearError, MAX_BF16_LINEAR_ELEMENTS,
        index_scores_bf16_reference, relu_bf16,
    };

    fn dim(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("nonzero test dimension")
    }

    #[test]
    fn exact_known_bf16_stages_keep_the_signed_weighting_order() {
        let actual = index_scores_bf16_reference(
            &[0x3f80, 0x4000],
            &[0x4040, 0x4080],
            &[0x4000, 0xbf80],
            dim(1),
        )
        .expect("finite source-shaped score chain");
        assert_eq!(actual.dot_products, [0x4040, 0x4080, 0x40c0, 0x4100]);
        assert_eq!(actual.rectified, [0x4040, 0x4080, 0x40c0, 0x4100]);
        assert_eq!(actual.weighted, [0x40c0, 0x4100, 0xc0c0, 0xc100]);
        assert_eq!(actual.scores, [0, 0]);
    }

    #[test]
    fn relu_preserves_negative_zero_but_rectifies_negative_values() {
        assert_eq!(
            [0x8000, 0, 0xbf80, 0x3f80].map(relu_bf16),
            [0x8000, 0, 0, 0x3f80]
        );
    }

    #[test]
    fn rejects_misaligned_queries_keys_and_head_weights() {
        assert!(matches!(
            index_scores_bf16_reference(&[0; 3], &[0; 2], &[0], dim(2)),
            Err(Bf16IndexScoreError::QueryShape {
                actual: 3,
                head_dim: 2
            })
        ));
        assert!(matches!(
            index_scores_bf16_reference(&[0; 2], &[0; 3], &[0], dim(2)),
            Err(Bf16IndexScoreError::KeyShape {
                actual: 3,
                head_dim: 2
            })
        ));
        assert!(matches!(
            index_scores_bf16_reference(&[0; 4], &[0; 2], &[0], dim(2)),
            Err(Bf16IndexScoreError::HeadWeightCount {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    fn rejects_late_nonfinite_keys_and_signed_head_weights() {
        assert!(matches!(
            index_scores_bf16_reference(&[0x3f80], &[0x3f80, 0x7fc1], &[0x3f80], dim(1)),
            Err(Bf16IndexScoreError::NonFiniteInput {
                field: "keys",
                position: 1
            })
        ));
        assert!(matches!(
            index_scores_bf16_reference(&[0x3f80; 2], &[0x3f80], &[0x3f80, 0xff80], dim(1)),
            Err(Bf16IndexScoreError::NonFiniteInput {
                field: "head_weights",
                position: 1
            })
        ));
    }

    #[test]
    fn rejects_nonfinite_and_overflowing_bf16_stages() {
        assert!(matches!(
            index_scores_bf16_reference(&[0x7f80], &[0x3f80], &[0x3f80], dim(1)),
            Err(Bf16IndexScoreError::NonFiniteInput {
                field: "query",
                position: 0
            })
        ));
        assert!(matches!(
            index_scores_bf16_reference(&[0x7f7f], &[0x4000], &[0x3f80], dim(1)),
            Err(Bf16IndexScoreError::Linear(_))
        ));
        assert!(matches!(
            index_scores_bf16_reference(&[0x7f7f], &[0x3f80], &[0x4000], dim(1)),
            Err(Bf16IndexScoreError::NonFiniteIntermediate {
                stage: "weighted product",
                ..
            })
        ));
        assert!(matches!(
            index_scores_bf16_reference(&[0x7f78], &[0x3f80], &[0x3f84], dim(1)),
            Err(Bf16IndexScoreError::NonFiniteIntermediate {
                stage: "weighted BF16 output",
                ..
            })
        ));
        assert!(matches!(
            index_scores_bf16_reference(&[0x7f7f, 0x7b00], &[0x3f80], &[0x3f80, 0x3f80], dim(1)),
            Err(Bf16IndexScoreError::NonFiniteOutput { position: 0 })
        ));
        assert!(matches!(
            index_scores_bf16_reference(&[0x7f7f; 2], &[0x3f80], &[0x3f80; 2], dim(1)),
            Err(Bf16IndexScoreError::NonFiniteIntermediate {
                stage: "head sum",
                head: 1,
                position: 0,
            })
        ));
    }

    #[test]
    fn rejects_shapes_and_shared_caps_before_staging() {
        let scalar_work_inputs = vec![0x3f80; 65 * 4_097];
        assert!(matches!(
            index_scores_bf16_reference(&[], &[0x3f80], &[0x3f80], dim(1)),
            Err(Bf16IndexScoreError::EmptyInput { field: "query" })
        ));
        assert!(matches!(
            index_scores_bf16_reference(
                &[0x3f80; 4_097],
                &[0x3f80; 4_096],
                &[0x3f80; 4_097],
                dim(1)
            ),
            Err(Bf16IndexScoreError::WorkloadTooLarge { .. })
        ));
        assert!(matches!(
            index_scores_bf16_reference(
                &scalar_work_inputs,
                &scalar_work_inputs,
                &[0x3f80; 65],
                dim(4_097)
            ),
            Err(Bf16IndexScoreError::ScalarWorkloadTooLarge { .. })
        ));
        assert!(matches!(
            index_scores_bf16_reference(
                &[0x3f80],
                &vec![0x3f80; MAX_BF16_LINEAR_ELEMENTS + 1],
                &[0x3f80],
                dim(1)
            ),
            Err(Bf16IndexScoreError::Linear(Bf16LinearError::ElementLimit {
                field: "weights",
                ..
            }))
        ));
    }
}
