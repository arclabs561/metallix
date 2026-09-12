//! Bounded scalar BF16 `RMSNorm` reference for V4.1 text sublayers.
//!
//! This follows the pinned `RMSNorm.forward` sequence: promote BF16 storage to
//! FP32, square and mean, add epsilon, reciprocal-square-root, multiply the
//! normalized input by the learned weight, then round back to BF16. It is not
//! a checkpoint reader, GPU kernel, or reduction-order parity claim.

use thiserror::Error;

/// Maximum hidden width accepted by one scalar `RMSNorm` call.
pub const MAX_RMS_NORM_WIDTH: usize = 16_384;

/// An invalid BF16 `RMSNorm` reference request or non-finite scalar result.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum RmsNormError {
    /// `RMSNorm` needs at least one hidden element.
    #[error("RMSNorm requires a nonempty hidden row")]
    EmptyInput,
    /// The hidden row exceeds the explicit scalar-work bound.
    #[error("RMSNorm width {width} exceeds maximum {max_width}")]
    WidthTooLarge {
        /// Supplied hidden width.
        width: usize,
        /// Maximum accepted hidden width.
        max_width: usize,
    },
    /// A caller-supplied buffer has an unexpected exact length.
    #[error("RMSNorm {field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Required element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// The RMS epsilon must be finite and positive.
    #[error("RMSNorm epsilon must be finite and positive")]
    InvalidEpsilon,
    /// A BF16 input bit pattern represents infinity or NaN.
    #[error("RMSNorm BF16 input at element {element} is non-finite")]
    NonFiniteInput {
        /// Flat input element index.
        element: usize,
    },
    /// A BF16 learned weight bit pattern represents infinity or NaN.
    #[error("RMSNorm BF16 weight at element {element} is non-finite")]
    NonFiniteWeight {
        /// Flat learned-weight element index.
        element: usize,
    },
    /// A scalar FP32 intermediate or BF16-rounded result could not remain finite.
    #[error("RMSNorm scalar result overflowed at {stage}, element {element}")]
    ValueOverflow {
        /// Named scalar stage that became non-finite.
        stage: &'static str,
        /// Input element responsible, or zero for row-wide stages.
        element: usize,
    },
}

/// Computes a scalar BF16 `RMSNorm` reference into an exact caller-owned buffer.
///
/// `input_bf16`, `weight_bf16`, and `output_bf16` are equal-width vectors.
/// Inputs and weights are BF16 storage bits. All arithmetic before the final
/// conversion uses ordered scalar FP32 operations. Validation and the full
/// result pass complete before any `output_bf16` slot changes.
///
/// # Errors
///
/// Returns [`RmsNormError`] if a buffer shape, scalar input, epsilon, or
/// intermediate is invalid. In every error case `output_bf16` is unchanged.
pub fn rms_norm_bf16_reference(
    input_bf16: &[u16],
    weight_bf16: &[u16],
    epsilon: f32,
    output_bf16: &mut [u16],
) -> Result<(), RmsNormError> {
    let width = validate_shape(input_bf16, weight_bf16, epsilon, output_bf16)?;
    let mut sum_squares = 0.0_f32;
    for (element, &bits) in input_bf16.iter().enumerate() {
        let value = bf16_to_f32(bits);
        if !value.is_finite() {
            return Err(RmsNormError::NonFiniteInput { element });
        }
        let square = value * value;
        if !square.is_finite() {
            return Err(RmsNormError::ValueOverflow {
                stage: "square",
                element,
            });
        }
        sum_squares += square;
        if !sum_squares.is_finite() {
            return Err(RmsNormError::ValueOverflow {
                stage: "sum",
                element,
            });
        }
    }
    for (element, &bits) in weight_bf16.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(RmsNormError::NonFiniteWeight { element });
        }
    }

    let width_f32 = f32::from(
        u16::try_from(width).map_err(|_| RmsNormError::WidthTooLarge {
            width,
            max_width: MAX_RMS_NORM_WIDTH,
        })?,
    );
    let mean = sum_squares / width_f32;
    if !mean.is_finite() {
        return Err(RmsNormError::ValueOverflow {
            stage: "mean",
            element: 0,
        });
    }
    let variance = mean + epsilon;
    if !variance.is_finite() {
        return Err(RmsNormError::ValueOverflow {
            stage: "variance",
            element: 0,
        });
    }
    let inverse_rms = variance.sqrt().recip();
    if !inverse_rms.is_finite() {
        return Err(RmsNormError::ValueOverflow {
            stage: "rsqrt",
            element: 0,
        });
    }

    let mut result = Vec::with_capacity(width);
    for (element, (&input_bits, &weight_bits)) in input_bf16.iter().zip(weight_bf16).enumerate() {
        let normalized = bf16_to_f32(input_bits) * inverse_rms;
        if !normalized.is_finite() {
            return Err(RmsNormError::ValueOverflow {
                stage: "normalize",
                element,
            });
        }
        let weighted = normalized * bf16_to_f32(weight_bits);
        if !weighted.is_finite() {
            return Err(RmsNormError::ValueOverflow {
                stage: "weight",
                element,
            });
        }
        let bits = f32_to_bf16_rne(weighted);
        if !bf16_to_f32(bits).is_finite() {
            return Err(RmsNormError::ValueOverflow {
                stage: "bf16",
                element,
            });
        }
        result.push(bits);
    }
    output_bf16.copy_from_slice(&result);
    Ok(())
}

fn validate_shape(
    input_bf16: &[u16],
    weight_bf16: &[u16],
    epsilon: f32,
    output_bf16: &[u16],
) -> Result<usize, RmsNormError> {
    let width = input_bf16.len();
    if width == 0 {
        return Err(RmsNormError::EmptyInput);
    }
    if width > MAX_RMS_NORM_WIDTH {
        return Err(RmsNormError::WidthTooLarge {
            width,
            max_width: MAX_RMS_NORM_WIDTH,
        });
    }
    for (field, actual) in [("weight", weight_bf16.len()), ("output", output_bf16.len())] {
        if actual != width {
            return Err(RmsNormError::Length {
                field,
                expected: width,
                actual,
            });
        }
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(RmsNormError::InvalidEpsilon);
    }
    Ok(width)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    u16::try_from(rounded >> 16).expect("an FP32 high half always fits BF16 storage")
}

#[cfg(test)]
mod tests {
    use super::{MAX_RMS_NORM_WIDTH, RmsNormError, rms_norm_bf16_reference};

    #[test]
    fn matches_hand_staged_bf16_rmsnorm() {
        let mut output = [0_u16; 4];
        rms_norm_bf16_reference(
            &[0x4000, 0xc000, 0x4000, 0xc000], // [2, -2, 2, -2]
            &[0x3f00, 0xc000, 0x3fc0, 0xbf00], // [0.5, -2, 1.5, -0.5]
            1.0e-20,
            &mut output,
        )
        .expect("finite exact RMSNorm staging");
        assert_eq!(output, [0x3f00, 0x4000, 0x3fc0, 0x3f00]);
    }

    #[test]
    fn rounds_the_fp32_normalized_value_to_bf16() {
        let mut output = [0_u16; 1];
        rms_norm_bf16_reference(&[0x3f80], &[0x3f80], 1.0, &mut output)
            .expect("finite rounding case");
        // FP32 1 / sqrt(2) rounds to BF16 0.70703125.
        assert_eq!(output, [0x3f35]);
    }

    #[test]
    fn rejects_invalid_requests_without_changing_output() {
        let mut output = [0xdead_u16; 2];
        assert_eq!(
            rms_norm_bf16_reference(&[0x3f80, 0x3f80], &[0x3f80, 0x7fc0], 1.0, &mut output),
            Err(RmsNormError::NonFiniteWeight { element: 1 })
        );
        assert_eq!(output, [0xdead, 0xdead]);
        assert_eq!(
            rms_norm_bf16_reference(&[], &[], 1.0, &mut []),
            Err(RmsNormError::EmptyInput)
        );
        assert!(matches!(
            rms_norm_bf16_reference(&[0x3f80], &[0x3f80], 1.0, &mut []),
            Err(RmsNormError::Length {
                field: "output",
                ..
            })
        ));
        assert!(matches!(
            rms_norm_bf16_reference(&[0x3f80], &[0x3f80], 0.0, &mut [0]),
            Err(RmsNormError::InvalidEpsilon)
        ));
        assert!(matches!(
            rms_norm_bf16_reference(&[0x7f80], &[0x3f80], 1.0, &mut [0]),
            Err(RmsNormError::NonFiniteInput { element: 0 })
        ));
        assert!(matches!(
            rms_norm_bf16_reference(&[0x7f7f], &[0x3f80], 1.0, &mut [0]),
            Err(RmsNormError::ValueOverflow {
                stage: "square",
                ..
            })
        ));
        let wide = vec![0_u16; MAX_RMS_NORM_WIDTH + 1];
        assert!(matches!(
            rms_norm_bf16_reference(&wide, &wide, 1.0, &mut wide.clone()),
            Err(RmsNormError::WidthTooLarge { .. })
        ));
    }

    #[test]
    fn late_bf16_overflow_preserves_every_output_slot() {
        // The second weighted FP32 value is finite but above the last finite
        // BF16 rounding boundary. Even the first valid slot must stay untouched.
        let mut output = [0xdead_u16; 2];
        assert_eq!(
            rms_norm_bf16_reference(
                &[0x3f7e, 0x3f80], // [0.9921875, 1]
                &[0x0000, 0x7f7f], // zero and BF16 maximum
                1.0e-20,
                &mut output,
            ),
            Err(RmsNormError::ValueOverflow {
                stage: "bf16",
                element: 1,
            })
        );
        assert_eq!(output, [0xdead, 0xdead]);
    }
}
