//! Bounded scalar BF16 → E2M1 → BF16 activation reconstruction.
//!
//! Follows the pinned V4.1 FP4 scale paths with software RNE encoders, not
//! GPU cast/reduction parity, packed cache storage or checkpoint conversion.

use super::{bf16_to_f32, decode_e4m3fn, decode_e8m0, decode_nibble, f32_to_bf16_rne};
use thiserror::Error;

/// Maximum input/output element count in one scalar FP4 reconstruction call.
pub const MAX_FP4_ACTIVATION_ELEMENTS: usize = 1 << 20;

/// Source-specific FP4 activation scale and grouping modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Fp4ActivationMode {
    /// Compressed KV: 16-element groups with E4M3 scales.
    CompressedKv16E4m3,
    /// Index queries and keys: 32-element groups with power-of-two E8M0 scales.
    Index32E8m0,
}

impl Fp4ActivationMode {
    const fn block_size(self) -> usize {
        match self {
            Self::CompressedKv16E4m3 => 16,
            Self::Index32E8m0 => 32,
        }
    }
}

/// Invalid inputs or failed scalar FP4 reconstruction; caller output is unchanged.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum Fp4ActivationError {
    /// Matrix dimensions must be nonzero.
    #[error("rows and width must be nonzero")]
    EmptyShape,
    /// A row does not contain complete scale groups.
    #[error("FP4 width {width} is not divisible by block size {block_size}")]
    PartialBlock {
        /// Supplied row width.
        width: usize,
        /// Required scale group width.
        block_size: usize,
    },
    /// Matrix element count overflowed addressable memory.
    #[error("FP4 element count overflow")]
    ShapeOverflow,
    /// The requested shape exceeds the bounded reference's work limit.
    #[error("FP4 element count {elements} exceeds {MAX_FP4_ACTIVATION_ELEMENTS}")]
    ElementLimit {
        /// Required element count.
        elements: usize,
    },
    /// Caller storage does not match the matrix shape.
    #[error("FP4 {field} length is {actual}, expected {expected}")]
    Length {
        /// Input or output buffer.
        field: &'static str,
        /// Supplied element count.
        actual: usize,
        /// Required element count.
        expected: usize,
    },
    /// An input BF16 word represents NaN or infinity.
    #[error("FP4 BF16 input at {index} is non-finite")]
    NonFiniteInput {
        /// Flat input index.
        index: usize,
    },
    /// The raw compressed-KV scale exceeds the supported finite E4M3 range.
    #[error("FP4 E4M3 scale exceeds finite supported range")]
    E4m3ScaleOverflow,
    /// The index scale cannot be represented as a finite E8M0 power.
    #[error("FP4 E8M0 scale exceeds finite supported range")]
    E8m0ScaleOverflow,
    /// Fallible reservation of the temporary output buffer failed.
    #[error("FP4 could not allocate {elements} temporary BF16 elements")]
    AllocationFailed {
        /// Requested temporary element count.
        elements: usize,
    },
    /// FP32 reconstruction or its final BF16 narrowing overflowed.
    #[error("FP4 reconstruction overflowed at {index}")]
    ReconstructionOverflow {
        /// Flat output index.
        index: usize,
    },
}

/// Quantizes finite BF16 rows through logical E2M1 and reconstructs BF16.
///
/// Software RNE for E2M1/E4M3 is a scalar test assumption; hardware cast
/// tie behavior is not qualified. The caller output remains unchanged unless
/// every input, scale, code, and reconstructed BF16 is finite.
///
/// `input` and `output` are row-major `[rows, width]` BF16 storage words.
/// Width must be divisible by the mode's group size. This reference uses one
/// fallibly allocated BF16 staging buffer; it does not return packed FP4 data.
/// Raw E4M3 scales above 448 are rejected, not saturated.
///
/// # Errors
///
/// Returns [`Fp4ActivationError`] for invalid shape, nonfinite input, excessive
/// work, allocation failure, unsupported scales or nonfinite reconstruction.
/// Caller output is unchanged for every returned error.
///
/// # Example
///
/// ```
/// use deepseek::precision::{Fp4ActivationMode, requantize_bf16_activations_e2m1};
/// let input = [0x4110_u16; 32]; // BF16 value 9
/// let mut output = [0_u16; 32];
/// requantize_bf16_activations_e2m1(
///     &input, 1, 32, Fp4ActivationMode::Index32E8m0, &mut output,
/// )?;
/// assert_eq!(output, [0x4100_u16; 32]); // BF16 value 8
/// # Ok::<(), deepseek::precision::Fp4ActivationError>(())
/// ```
pub fn requantize_bf16_activations_e2m1(
    input: &[u16],
    rows: usize,
    width: usize,
    mode: Fp4ActivationMode,
    output: &mut [u16],
) -> Result<(), Fp4ActivationError> {
    if rows == 0 || width == 0 {
        return Err(Fp4ActivationError::EmptyShape);
    }
    let block_size = mode.block_size();
    if !width.is_multiple_of(block_size) {
        return Err(Fp4ActivationError::PartialBlock { width, block_size });
    }
    let elements = rows
        .checked_mul(width)
        .ok_or(Fp4ActivationError::ShapeOverflow)?;
    if elements > MAX_FP4_ACTIVATION_ELEMENTS {
        return Err(Fp4ActivationError::ElementLimit { elements });
    }
    check_length("input", input.len(), elements)?;
    check_length("output", output.len(), elements)?;
    for (index, &bits) in input.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(Fp4ActivationError::NonFiniteInput { index });
        }
    }

    let groups = elements / block_size;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| Fp4ActivationError::AllocationFailed { elements })?;
    for group in 0..groups {
        let start = group * block_size;
        let end = start + block_size;
        let scale = scale_for(&input[start..end], mode)?;
        for (offset, &bits) in input[start..end].iter().enumerate() {
            let code = encode_e2m1_rne((bf16_to_f32(bits) / scale.1).clamp(-6.0, 6.0));
            let reconstructed = decode_nibble(code) * scale.1;
            let result = f32_to_bf16_rne(reconstructed);
            if !bf16_to_f32(result).is_finite() {
                return Err(Fp4ActivationError::ReconstructionOverflow {
                    index: start + offset,
                });
            }
            values.push(result);
        }
    }
    output.copy_from_slice(&values);
    Ok(())
}

fn scale_for(group: &[u16], mode: Fp4ActivationMode) -> Result<(u8, f32), Fp4ActivationError> {
    let amax = group
        .iter()
        .map(|&bits| bf16_to_f32(bits).abs())
        .fold(0.0_f32, f32::max);
    match mode {
        Fp4ActivationMode::CompressedKv16E4m3 => {
            let raw = amax.max(6.0 * 2.0_f32.powi(-9)) / 6.0;
            if raw > 448.0 {
                return Err(Fp4ActivationError::E4m3ScaleOverflow);
            }
            let code = encode_e4m3_rne(raw).ok_or(Fp4ActivationError::E4m3ScaleOverflow)?;
            let scale = decode_e4m3fn(code);
            if !scale.is_finite() || scale <= 0.0 {
                return Err(Fp4ActivationError::E4m3ScaleOverflow);
            }
            Ok((code, scale))
        }
        Fp4ActivationMode::Index32E8m0 => {
            // Pinned `fast_round_scale` receives `amax * fp4_max_inv`; keep
            // its FP32 reciprocal multiply rather than algebraically dividing.
            let raw = amax.max(6.0 * 2.0_f32.powi(-126)) * (1.0_f32 / 6.0);
            let exponent = ceil_log2(raw).ok_or(Fp4ActivationError::E8m0ScaleOverflow)?;
            if exponent > 127 {
                return Err(Fp4ActivationError::E8m0ScaleOverflow);
            }
            let code = u8::try_from(exponent + 127).expect("validated E8M0 exponent");
            let scale = decode_e8m0(code);
            if !scale.is_finite() {
                return Err(Fp4ActivationError::E8m0ScaleOverflow);
            }
            Ok((code, scale))
        }
    }
}

fn ceil_log2(value: f32) -> Option<i32> {
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let bits = value.to_bits();
    let exponent = i32::try_from((bits >> 23) & 0xff).ok()? - 127;
    Some(exponent + i32::from((bits & 0x7f_ffff) != 0))
}

fn encode_e2m1_rne(value: f32) -> u8 {
    nearest_code(value, 0_u8..=15, decode_nibble)
}

fn encode_e4m3_rne(value: f32) -> Option<u8> {
    if !value.is_finite() || value < 0.0 || value > 448.0 {
        return None;
    }
    Some(nearest_code(value, 0_u8..=126, decode_e4m3fn))
}

fn nearest_code(value: f32, codes: impl Iterator<Item = u8>, decode: impl Fn(u8) -> f32) -> u8 {
    codes
        .filter(|&code| {
            let decoded = decode(code);
            decoded.is_finite() && value.is_sign_negative() == decoded.is_sign_negative()
        })
        .min_by(|&left, &right| {
            let left_distance = (f64::from(decode(left)) - f64::from(value)).abs();
            let right_distance = (f64::from(decode(right)) - f64::from(value)).abs();
            left_distance
                .total_cmp(&right_distance)
                .then_with(|| (left & 1).cmp(&(right & 1)))
        })
        .expect("finite codebook")
}

fn check_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), Fp4ActivationError> {
    if actual == expected {
        Ok(())
    } else {
        Err(Fp4ActivationError::Length {
            field,
            actual,
            expected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::precision::decode_e2m1;
    #[test]
    fn e2m1_software_rne_covers_ties_and_adjacent_fp32_inputs() {
        for code in 0_u8..=15 {
            let value = decode_e2m1(code).expect("code");
            assert_eq!(encode_e2m1_rne(value), code);
        }
        for low_code in 0_u8..7 {
            let low = decode_e2m1(low_code).expect("low code");
            let high = decode_e2m1(low_code + 1).expect("high code");
            let midpoint = low.midpoint(high);
            let expected = if low_code & 1 == 0 {
                low_code
            } else {
                low_code + 1
            };
            assert_eq!(encode_e2m1_rne(midpoint), expected);
            assert_eq!(encode_e2m1_rne(midpoint.next_down()), low_code);
            assert_eq!(encode_e2m1_rne(midpoint.next_up()), low_code + 1);

            let negative_low = low_code + 8;
            let negative_high = negative_low + 1;
            let negative_midpoint = -midpoint;
            let negative_expected = if negative_low & 1 == 0 {
                negative_low
            } else {
                negative_high
            };
            assert_eq!(encode_e2m1_rne(negative_midpoint), negative_expected);
            assert_eq!(encode_e2m1_rne(negative_midpoint.next_up()), negative_low);
            assert_eq!(
                encode_e2m1_rne(negative_midpoint.next_down()),
                negative_high
            );
        }
    }

    #[test]
    fn e4m3_scale_encoder_covers_every_positive_code_and_rounding_boundary() {
        for code in 0_u8..=126 {
            assert_eq!(encode_e4m3_rne(decode_e4m3fn(code)), Some(code));
        }
        for lower in 0_u8..126 {
            let midpoint = decode_e4m3fn(lower).midpoint(decode_e4m3fn(lower + 1));
            let even = if lower & 1 == 0 { lower } else { lower + 1 };
            assert_eq!(encode_e4m3_rne(midpoint), Some(even));
            assert_eq!(encode_e4m3_rne(midpoint.next_down()), Some(lower));
            assert_eq!(encode_e4m3_rne(midpoint.next_up()), Some(lower + 1));
        }
        for unsupported in [448.0_f32.next_up(), f32::INFINITY, f32::NAN, -1.0] {
            assert_eq!(encode_e4m3_rne(unsupported), None);
        }
    }

    #[test]
    fn source_fixture_scale_values_and_known_codes_are_preserved() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/fp4-activation-reference.json"
        ))
        .expect("independent CPU oracle");
        for case in fixture["cases"].as_array().expect("cases") {
            let mode = match case["mode"].as_str().expect("mode") {
                "compressed_kv" => Fp4ActivationMode::CompressedKv16E4m3,
                "index" => Fp4ActivationMode::Index32E8m0,
                mode => panic!("unexpected mode {mode}"),
            };
            let input: Vec<u16> =
                serde_json::from_value(case["input_bf16"].clone()).expect("BF16 input");
            let (_, scale) = scale_for(&input, mode).expect("finite scale");
            let expected: u32 =
                serde_json::from_value(case["scale_f32_bits"].clone()).expect("scale bits");
            assert_eq!(scale.to_bits(), expected, "{}", case["name"]);
        }
        assert_eq!(
            scale_for(&[0; 32], Fp4ActivationMode::Index32E8m0)
                .expect("zero")
                .0,
            1
        );
        assert_eq!(
            scale_for(&[0x40c0; 16], Fp4ActivationMode::CompressedKv16E4m3)
                .expect("six")
                .0,
            0x38
        );
        assert_eq!([6.0, -6.0, 1.0, -0.0].map(encode_e2m1_rne), [7, 15, 2, 8]);
    }
}
