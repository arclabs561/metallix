//! Bounded scalar projection that produces one V4.1 Hyper-Connection mix row.
//!
//! BF16 residual storage is promoted to FP32. RMS is measured over its entire
//! flattened copy-width row; each completed FP32 projection dot is then scaled
//! by reciprocal RMS before the coefficient split. This is not a GPU or `F.linear`
//! reduction-order parity claim.

use thiserror::Error;

use super::{HcCoefficients, HcError, split_hc_coefficients};

const MAX_COPIES: usize = 16;
const MAX_WIDTH: usize = 16_384;
const MAX_MATRIX_ELEMENTS: usize = 1 << 20;

/// An invalid normalized Hyper-Connection projection request or scalar result.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum HcProjectionError {
    /// The flattened residual row is empty.
    #[error("Hyper-Connection projection requires a nonempty residual row")]
    EmptyResidual,
    /// The copy count is outside the bounded scalar-reference range.
    #[error("Hyper-Connection projection copy count {copies} is outside 1 through {max_copies}")]
    InvalidCopies {
        /// Requested copy count.
        copies: usize,
        /// Largest accepted copy count.
        max_copies: usize,
    },
    /// The flattened residual length is not an exact copy-width product.
    #[error("Hyper-Connection residual length {length} is not divisible by {copies} copies")]
    ResidualShape {
        /// Supplied flattened BF16 residual length.
        length: usize,
        /// Requested copy count.
        copies: usize,
    },
    /// The inferred per-copy hidden width exceeds the scalar-reference bound.
    #[error("Hyper-Connection width {width} exceeds maximum {max_width}")]
    WidthTooLarge {
        /// Inferred per-copy hidden width.
        width: usize,
        /// Largest accepted per-copy width.
        max_width: usize,
    },
    /// Checked projection shape arithmetic overflowed.
    #[error("Hyper-Connection projection shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// Derived shape role.
        field: &'static str,
    },
    /// The projection matrix exceeds the explicit scalar-work bound.
    #[error("Hyper-Connection projection has {elements} elements, maximum is {max_elements}")]
    MatrixTooLarge {
        /// Requested FP32 matrix element count.
        elements: usize,
        /// Largest accepted FP32 matrix element count.
        max_elements: usize,
    },
    /// The FP32 projection matrix has an unexpected exact length.
    #[error("Hyper-Connection projection length is {actual}, expected {expected}")]
    ProjectionLength {
        /// Supplied FP32 projection count.
        actual: usize,
        /// Required row-major matrix count.
        expected: usize,
    },
    /// The residual normalization epsilon must be finite and positive.
    #[error("Hyper-Connection normalization epsilon must be finite and positive")]
    InvalidNormEpsilon,
    /// A BF16 residual or FP32 projection value was non-finite.
    #[error("Hyper-Connection {field} at index {index} is non-finite")]
    NonFiniteInput {
        /// Input buffer role.
        field: &'static str,
        /// Flat item index.
        index: usize,
    },
    /// A scalar normalization, dot, or post-dot scale result overflowed.
    #[error("Hyper-Connection projection scalar overflowed at {stage}, index {index}")]
    ValueOverflow {
        /// Named scalar stage.
        stage: &'static str,
        /// Flat residual, row, or matrix index.
        index: usize,
    },
    /// The downstream pinned coefficient split rejected the resulting controls or mixes.
    #[error(transparent)]
    Coefficients(#[from] HcError),
}

/// Derives V4.1 Hyper-Connection coefficients from a BF16 residual and FP32 projection.
///
/// `residual` is row-major `[copies, width]` BF16 storage and `projection` is
/// row-major FP32 `[(2 + copies) * copies, copies * width]`. The FP32 scale is
/// applied after each full FP32 dot, matching `F.linear(x, hc_fn) * rsqrt` in
/// the pinned source. This API returns coefficients only and never reads a
/// checkpoint or mutates caller-owned output.
///
/// # Errors
///
/// Returns [`HcProjectionError`] before returning coefficients if shape,
/// scalar-control, finite-input, normalization, projection, or coefficient
/// split invariants fail.
#[allow(
    clippy::too_many_arguments,
    reason = "the direct residual, projection, and coefficient roles remain explicit"
)]
pub fn project_hc_coefficients(
    residual: &[u16],
    projection: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    copies: usize,
    norm_epsilon: f32,
    sinkhorn_iterations: usize,
    hc_epsilon: f32,
) -> Result<HcCoefficients, HcProjectionError> {
    if !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
        return Err(HcProjectionError::InvalidNormEpsilon);
    }
    let shape = Shape::validate(
        residual,
        projection,
        scale,
        base,
        copies,
        sinkhorn_iterations,
        hc_epsilon,
    )?;
    let mut sum_squares = 0.0_f32;
    for (index, &bits) in residual.iter().enumerate() {
        let value = bf16_to_f32(bits);
        finite_input("residual", index, value)?;
        let square = value * value;
        finite(square, "square", index)?;
        sum_squares += square;
        finite(sum_squares, "sum", index)?;
    }
    let mean = sum_squares / shape.residual_len_f32;
    finite(mean, "mean", 0)?;
    let variance = mean + norm_epsilon;
    finite(variance, "variance", 0)?;
    let inverse_rms = variance.sqrt().recip();
    finite(inverse_rms, "rsqrt", 0)?;

    let mut mixes = Vec::with_capacity(shape.mix_rows);
    for row in 0..shape.mix_rows {
        let projection_row = &projection[row * shape.residual_len..(row + 1) * shape.residual_len];
        let mut dot = 0.0_f32;
        for (column, (&bits, &weight)) in residual.iter().zip(projection_row).enumerate() {
            let term = bf16_to_f32(bits) * weight;
            finite(term, "dot_product", row * shape.residual_len + column)?;
            dot += term;
            finite(dot, "dot_sum", row)?;
        }
        let mix = dot * inverse_rms;
        finite(mix, "post_dot_norm", row)?;
        mixes.push(mix);
    }
    Ok(split_hc_coefficients(
        &mixes,
        scale,
        base,
        copies,
        sinkhorn_iterations,
        hc_epsilon,
    )?)
}

#[derive(Clone, Copy)]
struct Shape {
    residual_len: usize,
    residual_len_f32: f32,
    mix_rows: usize,
}

impl Shape {
    fn validate(
        residual: &[u16],
        projection: &[f32],
        scale: &[f32; 3],
        base: &[f32],
        copies: usize,
        sinkhorn_iterations: usize,
        hc_epsilon: f32,
    ) -> Result<Self, HcProjectionError> {
        if residual.is_empty() {
            return Err(HcProjectionError::EmptyResidual);
        }
        if copies == 0 || copies > MAX_COPIES {
            return Err(HcProjectionError::InvalidCopies {
                copies,
                max_copies: MAX_COPIES,
            });
        }
        if !residual.len().is_multiple_of(copies) {
            return Err(HcProjectionError::ResidualShape {
                length: residual.len(),
                copies,
            });
        }
        let width = residual.len() / copies;
        if width > MAX_WIDTH {
            return Err(HcProjectionError::WidthTooLarge {
                width,
                max_width: MAX_WIDTH,
            });
        }
        let mix_rows = copies
            .checked_add(2)
            .ok_or(HcProjectionError::ShapeOverflow { field: "mix_rows" })?
            .checked_mul(copies)
            .ok_or(HcProjectionError::ShapeOverflow { field: "mix_rows" })?;
        let matrix_elements =
            mix_rows
                .checked_mul(residual.len())
                .ok_or(HcProjectionError::ShapeOverflow {
                    field: "projection",
                })?;
        if matrix_elements > MAX_MATRIX_ELEMENTS {
            return Err(HcProjectionError::MatrixTooLarge {
                elements: matrix_elements,
                max_elements: MAX_MATRIX_ELEMENTS,
            });
        }
        if projection.len() != matrix_elements {
            return Err(HcProjectionError::ProjectionLength {
                actual: projection.len(),
                expected: matrix_elements,
            });
        }
        if base.len() != mix_rows {
            return Err(HcProjectionError::Coefficients(HcError::Length {
                field: "base",
                expected: mix_rows,
                actual: base.len(),
            }));
        }
        if sinkhorn_iterations == 0 || sinkhorn_iterations > 64 {
            return Err(HcProjectionError::Coefficients(
                HcError::InvalidIterations {
                    iterations: sinkhorn_iterations,
                    max_iterations: 64,
                },
            ));
        }
        if !hc_epsilon.is_finite() || hc_epsilon <= 0.0 {
            return Err(HcProjectionError::Coefficients(HcError::InvalidEpsilon));
        }
        for (index, &value) in scale.iter().enumerate() {
            finite_input("scale", index, value)?;
        }
        for (index, &value) in base.iter().enumerate() {
            finite_input("base", index, value)?;
        }
        for (index, &value) in projection.iter().enumerate() {
            finite_input("projection", index, value)?;
        }
        // Matrix bound implies this is below 2^24, so the FP32 divisor is exact.
        #[allow(
            clippy::cast_precision_loss,
            reason = "validated matrix bound keeps the residual length below 2^24"
        )]
        let residual_len_f32 = residual.len() as f32;
        Ok(Self {
            residual_len: residual.len(),
            residual_len_f32,
            mix_rows,
        })
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn finite_input(field: &'static str, index: usize, value: f32) -> Result<(), HcProjectionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(HcProjectionError::NonFiniteInput { field, index })
    }
}

fn finite(value: f32, stage: &'static str, index: usize) -> Result<(), HcProjectionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(HcProjectionError::ValueOverflow { stage, index })
    }
}

#[cfg(test)]
mod tests {
    use super::{HcProjectionError, project_hc_coefficients};
    use crate::hc::split_hc_coefficients;

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }

    #[test]
    fn projects_hand_staged_mixes_before_the_existing_split() {
        let residual = [0x4000_u16, 0xc000]; // BF16 [2, -2], mean square 4, inverse RMS 1/2.
        let mut projection = [0.0_f32; 16]; // 8 output rows by flattened width 2.
        projection[5 * 2] = 2.0;
        projection[6 * 2] = -1.0;
        projection[7 * 2] = 1.0;
        let actual = project_hc_coefficients(
            &residual,
            &projection,
            &[1.0, 0.5, 2.0],
            &[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            2,
            1.0e-20,
            1,
            0.1,
        )
        .expect("finite hand-staged HC projection");
        let expected = split_hc_coefficients(
            &[0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -1.0, 1.0],
            &[1.0, 0.5, 2.0],
            &[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            2,
            1,
            0.1,
        )
        .expect("split of hand-derived projection logits");
        assert_eq!(actual.copies(), expected.copies());
        for (actual, expected) in actual.pre().iter().zip(expected.pre()) {
            assert_close(*actual, *expected);
        }
        for (actual, expected) in actual.post().iter().zip(expected.post()) {
            assert_close(*actual, *expected);
        }
        for (actual, expected) in actual.comb().iter().zip(expected.comb()) {
            assert_close(*actual, *expected);
        }
    }

    #[test]
    fn applies_normalization_after_the_cancelling_dot() {
        // The full dot is seven. Scaling it after the dot and scaling each
        // input first differ by one FP32 ULP. The affine scale magnifies that
        // difference so it remains observable through the sigmoid split.
        let residual = [0x40a0_u16, 0x3f80, 0xbf80, 0x4080]; // BF16 [5, 1, -1, 4]
        let mut projection = [0.0_f32; 32]; // hc=2 gives eight output rows by width four.
        projection[..4].copy_from_slice(&[3.0, 1.0, 1.0, -2.0]);
        let correct_mix = f32::from_bits(0x4008_a383);
        let mut base = [0.0_f32; 8];
        base[0] = -correct_mix * 8_388_608.0;
        let actual = project_hc_coefficients(
            &residual,
            &projection,
            &[8_388_608.0, 0.0, 0.0],
            &base,
            2,
            1.0e-20,
            1,
            0.5,
        )
        .expect("finite post-dot normalization");
        assert_eq!(actual.pre()[0].to_bits(), 1.0_f32.to_bits());

        let inverse = (10.75_f32 + 1.0e-20_f32).sqrt().recip();
        let mut pre_normalized_dot = 0.0_f32;
        for (&value, &weight) in [5.0_f32, 1.0, -1.0, 4.0].iter().zip(&projection[..4]) {
            pre_normalized_dot += (value * inverse) * weight;
        }
        let misplaced = split_hc_coefficients(
            &[pre_normalized_dot, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[8_388_608.0, 0.0, 0.0],
            &base,
            2,
            1,
            0.5,
        )
        .expect("finite misplaced normalization mutant");
        assert_ne!(actual.pre()[0].to_bits(), misplaced.pre()[0].to_bits());
    }

    #[test]
    fn rejects_malformed_nonfinite_and_overflowing_requests() {
        assert!(matches!(
            project_hc_coefficients(&[], &[], &[0.0; 3], &[], 1, 1.0, 1, 1.0),
            Err(HcProjectionError::EmptyResidual)
        ));
        assert!(matches!(
            project_hc_coefficients(&[0x3f80], &[], &[0.0; 3], &[], 2, 1.0, 1, 1.0),
            Err(HcProjectionError::ResidualShape { .. })
        ));
        assert!(matches!(
            project_hc_coefficients(&[0x3f80], &[], &[0.0; 3], &[0.0; 3], 1, 1.0, 1, 1.0),
            Err(HcProjectionError::ProjectionLength { .. })
        ));
        assert!(matches!(
            project_hc_coefficients(&[0x7f80], &[0.0; 3], &[0.0; 3], &[0.0; 3], 1, 1.0, 1, 1.0),
            Err(HcProjectionError::NonFiniteInput {
                field: "residual",
                index: 0
            })
        ));
        assert!(matches!(
            project_hc_coefficients(
                &[0x4000],
                &[f32::MAX, 0.0, 0.0],
                &[2.0; 3],
                &[0.0; 3],
                1,
                1.0,
                1,
                1.0
            ),
            Err(HcProjectionError::ValueOverflow {
                stage: "dot_product",
                index: 0
            })
        ));
    }
}
