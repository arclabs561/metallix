//! Bounded FP32-input mathematical reference for V4.1 sparse attention.
//!
//! This is deliberately the semantic operation behind the pinned
//! `sparse_attn_kernel`, not an execution of that TileLang/CUDA kernel.  In
//! particular it does not model BF16 query/KV storage, BF16 attention-probability
//! rounding before the numerator GEMM, block-64 online reduction, or device
//! scheduling.  It gives later kernels a small, validated numerical oracle.

use std::num::NonZeroUsize;

use thiserror::Error;

#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use metal::{SparseAttentionMetalError, sparse_attention_metal_f32};

#[cfg(all(test, feature = "metal"))]
mod composition_tests;

#[cfg(test)]
mod preparation_tests;

/// Validated dense buffers and sparse-index shape for one attention call.
///
/// Query values are contiguous `[batch, query_position, head, dimension]`;
/// shared key/value vectors are `[batch, key_position, dimension]`; indices
/// are `[batch, query_position, sparse_slot]`; and sink logits are per head.
/// The upstream caller owns causal masking: this layout intentionally assigns
/// no temporal meaning to an index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseAttentionLayout {
    batches: NonZeroUsize,
    query_positions: NonZeroUsize,
    heads: NonZeroUsize,
    dimensions: NonZeroUsize,
    key_positions: NonZeroUsize,
    sparse_slots: NonZeroUsize,
}

impl SparseAttentionLayout {
    /// Creates a bounded sparse-attention layout.
    ///
    /// # Errors
    ///
    /// Returns [`SparseAttentionError::LayoutOverflow`] if a required buffer
    /// length cannot be represented by `usize`.
    pub fn new(
        batches: NonZeroUsize,
        query_positions: NonZeroUsize,
        heads: NonZeroUsize,
        dimensions: NonZeroUsize,
        key_positions: NonZeroUsize,
        sparse_slots: NonZeroUsize,
    ) -> Result<Self, SparseAttentionError> {
        let layout = Self {
            batches,
            query_positions,
            heads,
            dimensions,
            key_positions,
            sparse_slots,
        };
        let _ = layout.query_len()?;
        let _ = layout.kv_len()?;
        let _ = layout.index_len()?;
        let _ = layout.output_len()?;
        Ok(layout)
    }

    fn query_len(self) -> Result<usize, SparseAttentionError> {
        product(&[
            self.batches.get(),
            self.query_positions.get(),
            self.heads.get(),
            self.dimensions.get(),
        ])
    }

    fn kv_len(self) -> Result<usize, SparseAttentionError> {
        product(&[
            self.batches.get(),
            self.key_positions.get(),
            self.dimensions.get(),
        ])
    }

    fn index_len(self) -> Result<usize, SparseAttentionError> {
        product(&[
            self.batches.get(),
            self.query_positions.get(),
            self.sparse_slots.get(),
        ])
    }

    fn output_len(self) -> Result<usize, SparseAttentionError> {
        self.query_len()
    }
}

fn product(values: &[usize]) -> Result<usize, SparseAttentionError> {
    values.iter().try_fold(1_usize, |total, value| {
        total
            .checked_mul(*value)
            .ok_or(SparseAttentionError::LayoutOverflow)
    })
}

/// Errors from the bounded sparse-attention mathematical reference.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum SparseAttentionError {
    /// A layout-derived buffer length cannot be represented by `usize`.
    #[error("sparse-attention layout element count overflows usize")]
    LayoutOverflow,
    /// An input buffer's actual length differs from its validated layout.
    #[error("{field} contains {actual} values, expected {expected}")]
    LengthMismatch {
        /// Logical input buffer name.
        field: &'static str,
        /// Received element count.
        actual: usize,
        /// Required element count.
        expected: usize,
    },
    /// The softmax multiplier is not finite and strictly positive.
    #[error("attention softmax scale must be finite and strictly positive")]
    InvalidScale,
    /// A floating-point input contains NaN or an infinity.
    #[error("{field} value at scalar index {index} is not finite")]
    NonFiniteValue {
        /// Logical input buffer name.
        field: &'static str,
        /// Flat scalar index.
        index: usize,
    },
    /// A sparse slot is neither the `-1` empty sentinel nor a key position.
    #[error("sparse index at slot {slot} is {index}, expected -1 or 0..{key_positions}")]
    InvalidIndex {
        /// Flat index-buffer slot.
        slot: usize,
        /// Received value.
        index: i32,
        /// Exclusive valid key-position bound.
        key_positions: usize,
    },
    /// The result allocation could not be reserved.
    #[error("could not allocate {elements} sparse-attention output scalars")]
    AllocationFailed {
        /// Requested output scalar count.
        elements: usize,
    },
}

/// Computes the sparse-attention mathematical reference from FP32 inputs.
///
/// For each query/head, every nonnegative index slot contributes one score and
/// one numerator vector. Repeated indices deliberately contribute repeatedly,
/// matching the upstream gathering kernel. `-1` slots contribute neither. The
/// per-head sink logit contributes only to the softmax denominator, never an
/// output vector. A row containing only `-1` slots returns zero, matching the
/// pinned kernel's explicit all-masked convention.
///
/// Dot products, maxima, exponentials, and accumulations use FP64 so this is
/// a stable mathematical reference for FP32 *inputs*; final output scalars are
/// narrowed to FP32. It is not BF16 or CUDA-kernel parity.
///
/// # Errors
///
/// Returns [`SparseAttentionError`] before allocation or calculation for an
/// incompatible layout, non-finite input, invalid index, or invalid scale.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the public contract intentionally narrows stable FP64 reference output to FP32"
)]
pub fn sparse_attention_reference(
    query: &[f32],
    shared_kv: &[f32],
    attn_sink: &[f32],
    indices: &[i32],
    scale: f32,
    layout: SparseAttentionLayout,
) -> Result<Vec<f32>, SparseAttentionError> {
    validate_inputs(query, shared_kv, attn_sink, indices, scale, layout)?;
    let output_len = layout.output_len()?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_len)
        .map_err(|_| SparseAttentionError::AllocationFailed {
            elements: output_len,
        })?;
    output.resize(output_len, 0.0);

    let batches = layout.batches.get();
    let queries = layout.query_positions.get();
    let heads = layout.heads.get();
    let dimensions = layout.dimensions.get();
    let slots = layout.sparse_slots.get();
    let scale = f64::from(scale);

    for batch in 0..batches {
        for query_position in 0..queries {
            let index_base = (batch * queries + query_position) * slots;
            for (head, sink) in attn_sink.iter().copied().enumerate() {
                let query_base = ((batch * queries + query_position) * heads + head) * dimensions;
                let mut max_score = f64::NEG_INFINITY;
                let mut has_key = false;
                for slot in 0..slots {
                    let key_index = indices[index_base + slot];
                    if key_index < 0 {
                        continue;
                    }
                    has_key = true;
                    let Ok(key_index) = usize::try_from(key_index) else {
                        continue;
                    };
                    let key_base = (batch * layout.key_positions.get() + key_index) * dimensions;
                    let score = dot(query, query_base, shared_kv, key_base, dimensions) * scale;
                    max_score = max_score.max(score);
                }
                if !has_key {
                    continue;
                }
                let sink = f64::from(sink);
                let normalizer_max = max_score.max(sink);
                let mut denominator = (sink - normalizer_max).exp();
                for slot in 0..slots {
                    let key_index = indices[index_base + slot];
                    if key_index < 0 {
                        continue;
                    }
                    let Ok(key_index) = usize::try_from(key_index) else {
                        continue;
                    };
                    let key_base = (batch * layout.key_positions.get() + key_index) * dimensions;
                    let score = dot(query, query_base, shared_kv, key_base, dimensions) * scale;
                    denominator += (score - normalizer_max).exp();
                }
                let output_base = query_base;
                for dimension in 0..dimensions {
                    let mut numerator = 0.0_f64;
                    for slot in 0..slots {
                        let key_index = indices[index_base + slot];
                        if key_index < 0 {
                            continue;
                        }
                        let Ok(key_index) = usize::try_from(key_index) else {
                            continue;
                        };
                        let key_base =
                            (batch * layout.key_positions.get() + key_index) * dimensions;
                        let score = dot(query, query_base, shared_kv, key_base, dimensions) * scale;
                        numerator += (score - normalizer_max).exp()
                            * f64::from(shared_kv[key_base + dimension]);
                    }
                    output[output_base + dimension] = (numerator / denominator) as f32;
                }
            }
        }
    }
    Ok(output)
}

fn dot(left: &[f32], left_base: usize, right: &[f32], right_base: usize, dimensions: usize) -> f64 {
    let mut total = 0.0_f64;
    for dimension in 0..dimensions {
        total += f64::from(left[left_base + dimension]) * f64::from(right[right_base + dimension]);
    }
    total
}

fn validate_inputs(
    query: &[f32],
    shared_kv: &[f32],
    attn_sink: &[f32],
    indices: &[i32],
    scale: f32,
    layout: SparseAttentionLayout,
) -> Result<(), SparseAttentionError> {
    validate_length("query", query.len(), layout.query_len()?)?;
    validate_length("shared_kv", shared_kv.len(), layout.kv_len()?)?;
    validate_length("attn_sink", attn_sink.len(), layout.heads.get())?;
    validate_length("indices", indices.len(), layout.index_len()?)?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(SparseAttentionError::InvalidScale);
    }
    validate_finite("query", query)?;
    validate_finite("shared_kv", shared_kv)?;
    validate_finite("attn_sink", attn_sink)?;
    let key_positions = layout.key_positions.get();
    for (slot, index) in indices.iter().copied().enumerate() {
        if index < -1 || usize::try_from(index).is_ok_and(|index| index >= key_positions) {
            return Err(SparseAttentionError::InvalidIndex {
                slot,
                index,
                key_positions,
            });
        }
    }
    Ok(())
}

fn validate_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SparseAttentionError> {
    if actual == expected {
        Ok(())
    } else {
        Err(SparseAttentionError::LengthMismatch {
            field,
            actual,
            expected,
        })
    }
}

fn validate_finite(field: &'static str, values: &[f32]) -> Result<(), SparseAttentionError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(SparseAttentionError::NonFiniteValue { field, index });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{SparseAttentionError, SparseAttentionLayout, sparse_attention_reference};
    use std::num::NonZeroUsize;

    use serde::Deserialize;

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
        batches: usize,
        query_positions: usize,
        heads: usize,
        dimensions: usize,
        key_positions: usize,
        sparse_slots: usize,
        scale: f32,
        query: Vec<f32>,
        shared_kv: Vec<f32>,
        attn_sink: Vec<f32>,
        indices: Vec<i32>,
        expected_output: Vec<f32>,
    }

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
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "scalar {index}: actual {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn matches_independent_pinned_sparse_attention_fixture() {
        let fixture: Fixture = serde_json::from_str(include_str!(
            "../../../../fixtures/deepseek-v41/sparse-attention-reference.json"
        ))
        .expect("fixture JSON is valid");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(
            fixture.source.revision,
            "dba1be0a40aa45a94ad051997016db3960a90277"
        );
        assert_eq!(
            fixture.source.sha256,
            "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
        );
        assert_eq!(fixture.source.symbol, "sparse_attn_kernel");
        for case in fixture.cases {
            let actual = sparse_attention_reference(
                &case.query,
                &case.shared_kv,
                &case.attn_sink,
                &case.indices,
                case.scale,
                layout(
                    case.batches,
                    case.query_positions,
                    case.heads,
                    case.dimensions,
                    case.key_positions,
                    case.sparse_slots,
                ),
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
            assert_eq!(
                actual.len(),
                case.expected_output.len(),
                "{} length",
                case.name
            );
            for (index, (actual, expected)) in actual.iter().zip(&case.expected_output).enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 0.000_001,
                    "{} scalar {index}: actual {actual}, expected {expected}",
                    case.name
                );
            }
        }
    }

    #[test]
    fn duplicate_slots_count_twice() {
        let one = sparse_attention_reference(
            &[1.0],
            &[2.0, 8.0],
            &[0.0],
            &[0],
            1.0,
            layout(1, 1, 1, 1, 2, 1),
        )
        .unwrap();
        let duplicated = sparse_attention_reference(
            &[1.0],
            &[2.0, 8.0],
            &[0.0],
            &[0, 0],
            1.0,
            layout(1, 1, 1, 1, 2, 2),
        )
        .unwrap();
        assert!(duplicated[0] > one[0]);
        assert_close(&duplicated, &[1.873_242]);
    }

    #[test]
    fn masked_slots_contribute_nothing_and_all_masked_is_zero() {
        let masked = sparse_attention_reference(
            &[1.0, -1.0],
            &[2.0, 3.0, 5.0, 7.0],
            &[0.0],
            &[1, -1],
            1.0,
            layout(1, 1, 1, 2, 2, 2),
        )
        .unwrap();
        assert_close(&masked, &[0.596_015, 0.834_421]);
        let all_masked = sparse_attention_reference(
            &[1.0, -1.0],
            &[2.0, 3.0, 5.0, 7.0],
            &[123.0],
            &[-1, -1],
            1.0,
            layout(1, 1, 1, 2, 2, 2),
        )
        .unwrap();
        assert_close(&all_masked, &[0.0, 0.0]);
    }

    #[test]
    fn sink_is_denominator_only_and_can_dominate() {
        let output = sparse_attention_reference(
            &[1.0],
            &[4.0],
            &[30.0],
            &[0],
            1.0,
            layout(1, 1, 1, 1, 1, 1),
        )
        .unwrap();
        assert!(output[0] > 0.0 && output[0] < 0.000_000_1);
    }

    #[test]
    fn stable_for_large_finite_scores() {
        let output = sparse_attention_reference(
            &[1.0e20],
            &[1.0e20, -1.0e20],
            &[0.0],
            &[0, 1],
            1.0,
            layout(1, 1, 1, 1, 2, 2),
        )
        .unwrap();
        assert_close(&output, &[1.0e20]);
    }

    #[test]
    fn validates_the_entire_contract_before_calculation() {
        let shape = layout(1, 1, 1, 1, 1, 1);
        assert_eq!(
            sparse_attention_reference(&[1.0], &[1.0], &[0.0], &[0], 0.0, shape),
            Err(SparseAttentionError::InvalidScale)
        );
        assert_eq!(
            sparse_attention_reference(&[f32::NAN], &[1.0], &[0.0], &[0], 1.0, shape),
            Err(SparseAttentionError::NonFiniteValue {
                field: "query",
                index: 0
            })
        );
        assert_eq!(
            sparse_attention_reference(&[1.0], &[1.0], &[0.0], &[-2], 1.0, shape),
            Err(SparseAttentionError::InvalidIndex {
                slot: 0,
                index: -2,
                key_positions: 1
            })
        );
        assert_eq!(
            sparse_attention_reference(&[1.0], &[1.0], &[0.0], &[1], 1.0, shape),
            Err(SparseAttentionError::InvalidIndex {
                slot: 0,
                index: 1,
                key_positions: 1
            })
        );
    }
}
