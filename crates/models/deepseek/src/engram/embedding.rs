//! Scalar lookup for a single-rank FP8 Engram table.
//!
//! Rows are dequantized with row-local E8M0 scales and narrowed to BF16.
//! Out-of-table IDs produce zero rows, matching the pinned source's masking.
//! This does not perform distributed reduction or allocate a checkpoint table.

use thiserror::Error;

use crate::precision::{decode_e4m3fn, decode_e8m0, f32_to_bf16_rne};

const MAX_ELEMENTS: usize = 1 << 20;

/// Validated geometry of one bounded, unsharded Engram embedding table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngramEmbeddingLayout {
    rows: usize,
    width: usize,
    group: usize,
    elements: usize,
}

impl EngramEmbeddingLayout {
    /// Validates dimensions, scale grouping, and the scalar reference size cap.
    ///
    /// # Errors
    /// Returns [`EngramEmbeddingError`] for empty, ungrouped, or oversized shapes.
    pub fn new(rows: usize, width: usize, group: usize) -> Result<Self, EngramEmbeddingError> {
        if rows == 0 || width == 0 {
            return Err(EngramEmbeddingError::EmptyDimension);
        }
        if group != 32 || !width.is_multiple_of(group) {
            return Err(EngramEmbeddingError::InvalidGroup { width, group });
        }
        Ok(Self {
            rows,
            width,
            group,
            elements: bounded_product(rows, width)?,
        })
    }
}

/// Invalid table geometry, storage, or selected-row arithmetic.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum EngramEmbeddingError {
    /// A table dimension is zero.
    #[error("Engram embedding dimensions must be nonzero")]
    EmptyDimension,
    /// Only the pinned source's complete 32-element scale groups are supported.
    #[error("Engram embedding width {width} is incompatible with group {group}")]
    InvalidGroup { width: usize, group: usize },
    /// A derived shape overflowed or exceeded the scalar reference cap.
    #[error("Engram embedding shape exceeds the bounded scalar reference")]
    ElementLimit,
    /// A supplied buffer does not match its declared dimensions.
    #[error("Engram embedding {field} length {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// A selected value or its scale is nonfinite, or BF16 narrowing overflows.
    #[error("Engram embedding selected element {element} is nonfinite or overflows BF16")]
    NonFinite { element: usize },
}

fn bounded_product(left: usize, right: usize) -> Result<usize, EngramEmbeddingError> {
    left.checked_mul(right)
        .filter(|&value| value <= MAX_ELEMENTS)
        .ok_or(EngramEmbeddingError::ElementLimit)
}

fn selected_row(id: i64, rows: usize) -> Option<usize> {
    usize::try_from(id).ok().filter(|&row| row < rows)
}

fn decoded_value(
    codes: &[u8],
    scales: &[u8],
    index: usize,
    group: usize,
) -> Result<u16, EngramEmbeddingError> {
    let value = decode_e4m3fn(codes[index]) * decode_e8m0(scales[index / group]);
    let bits = f32_to_bf16_rne(value);
    if !value.is_finite() || bits & 0x7f80 == 0x7f80 {
        Err(EngramEmbeddingError::NonFinite { element: index })
    } else {
        Ok(bits)
    }
}

/// Looks up flattened hash IDs in an FP8 table and writes BF16 rows.
///
/// `codes` is `[table_rows, width]`, `scales` is
/// `[table_rows, width / group]`, and `output` is `[ids.len(), width]`.
/// IDs outside the table, including negative mask sentinels, produce zero rows.
/// Only selected rows are decoded; unused table values do not affect a lookup.
///
/// # Errors
/// Shape and selected-value validation finishes before any output is written.
/// Errors leave `output` unchanged. No allocation occurs in this operation.
pub fn engram_embedding_bf16_reference(
    ids: &[i64],
    codes: &[u8],
    scales: &[u8],
    layout: EngramEmbeddingLayout,
    output: &mut [u16],
) -> Result<(), EngramEmbeddingError> {
    let output_elements = bounded_product(ids.len(), layout.width)?;
    for (field, actual, expected) in [
        ("codes", codes.len(), layout.elements),
        ("scales", scales.len(), layout.elements / layout.group),
        ("output", output.len(), output_elements),
    ] {
        if actual != expected {
            return Err(EngramEmbeddingError::Length {
                field,
                actual,
                expected,
            });
        }
    }
    for &id in ids {
        if let Some(row) = selected_row(id, layout.rows) {
            for index in row * layout.width..(row + 1) * layout.width {
                decoded_value(codes, scales, index, layout.group)?;
            }
        }
    }
    for (&id, destination) in ids.iter().zip(output.chunks_exact_mut(layout.width)) {
        if let Some(row) = selected_row(id, layout.rows) {
            for (feature, value) in destination.iter_mut().enumerate() {
                *value = decoded_value(codes, scales, row * layout.width + feature, layout.group)?;
            }
        } else {
            destination.fill(0);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn late_invalid_selected_value_preserves_entire_output() {
        let layout = EngramEmbeddingLayout::new(2, 32, 32).unwrap();
        let mut codes = vec![0x38; 64]; // E4M3FN one.
        codes[63] = 0x7f;
        let mut output = vec![0x1234; 64];
        assert!(
            engram_embedding_bf16_reference(&[0, 1], &codes, &[127; 2], layout, &mut output)
                .is_err()
        );
        assert_eq!(output, vec![0x1234; 64]);
        assert!(
            engram_embedding_bf16_reference(&[0], &codes, &[127; 2], layout, &mut output[..32])
                .is_ok()
        );
    }

    #[test]
    fn malformed_shapes_and_overflow_leave_output_unchanged() {
        assert!(EngramEmbeddingLayout::new(usize::MAX, 32, 32).is_err());
        assert!(EngramEmbeddingLayout::new(1, 33, 32).is_err());
        assert!(EngramEmbeddingLayout::new(1, 32, 0).is_err());
        assert!(EngramEmbeddingLayout::new(1, 128, 128).is_err());
        let layout = EngramEmbeddingLayout::new(1, 32, 32).unwrap();
        let mut output = [7; 32];
        for (codes, scales) in [
            (vec![0x38; 31], vec![127]),
            (vec![0x7e; 32], vec![254]),
            (vec![0x38; 32], vec![255]),
        ] {
            assert!(
                engram_embedding_bf16_reference(&[0], &codes, &scales, layout, &mut output)
                    .is_err()
            );
            assert_eq!(output, [7; 32]);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]
        #[test]
        fn selected_corruption_is_atomic_after_arbitrary_valid_rows(
            prefix in prop::collection::vec(-2_i64..3, 0..24),
            feature in 0_usize..64,
            sentinel in any::<u16>(),
            corrupt_scale in any::<bool>(),
        ) {
            let layout = EngramEmbeddingLayout::new(4, 64, 32).unwrap();
            let mut codes = vec![0x38; 256];
            let mut scales = vec![127; 8];
            if corrupt_scale {
                scales[6 + feature / 32] = 255;
            } else {
                codes[192 + feature] = 0x7f;
            }
            let mut ids = prefix;
            // The corrupt row is reached after any mix of valid and masked rows.
            ids.push(3);
            let original = vec![sentinel; ids.len() * 64];
            let mut output = original.clone();
            prop_assert!(engram_embedding_bf16_reference(
                &ids, &codes, &scales, layout, &mut output
            ).is_err());
            prop_assert_eq!(&output, &original);
            // Masking the bad row makes the same table acceptable.
            *ids.last_mut().unwrap() = i64::MAX;
            engram_embedding_bf16_reference(&ids, &codes, &scales, layout, &mut output).unwrap();
            prop_assert!(output[output.len() - 64..].iter().all(|&bits| bits == 0));
        }

        #[test]
        fn selected_rows_obey_mask_order_and_row_local_scales(
            ids in prop::collection::vec(-3_i64..7, 0..24),
            exponents in prop::collection::vec(-4_i32..5, 8),
        ) {
            let group = 32;
            let width = group * 2;
            let layout = EngramEmbeddingLayout::new(4, width, group).unwrap();
            let scales: Vec<_> = exponents.iter().map(|&e| u8::try_from(e + 127).unwrap()).collect();
            let mut output = vec![0xffff; ids.len() * width];
            engram_embedding_bf16_reference(&ids, &vec![0x38; 4 * width], &scales, layout, &mut output).unwrap();
            for (&id, row) in ids.iter().zip(output.chunks_exact(width)) {
                for (feature, &bits) in row.iter().enumerate() {
                    let expected = if (0..4).contains(&id) {
                        // Powers of two have exact BF16 encodings; independent
                        // from the production decoder and narrowing helper.
                        u16::try_from(exponents[usize::try_from(id).unwrap() * 2 + feature / group] + 127).unwrap() << 7
                    } else { 0 };
                    prop_assert_eq!(bits, expected);
                }
            }
        }
    }
}
