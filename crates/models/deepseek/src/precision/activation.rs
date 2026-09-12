//! Scalar activation preparation for the pinned V4.1 power-of-two FP8 path.
//!
//! This accepts BF16 storage bits, promotes them exactly to FP32, and produces
//! E4M3FN codes with E8M0 power-of-two scales. Its software round-to-nearest,
//! ties-to-even encoder is a CPU reference assumption, not `TileLang`, CUDA, or
//! Tensor Core parity.

use thiserror::Error;

use super::{ActivationGroup, decode_e4m3fn, decode_e8m0};

const FP8_MAX: f32 = 448.0;
const FP8_MAX_INV: f32 = 1.0 / FP8_MAX;
const AMAX_FLOOR: f32 = 1e-4;

/// An invalid BF16 activation input or output buffer for scalar FP8 preparation.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum ActivationQuantError {
    /// Matrix dimensions require at least one row and reduction element.
    #[error("activation quantization dimensions must both be nonzero")]
    EmptyDimension,
    /// The reduction dimension cannot be grouped as required by the runtime path.
    #[error(
        "reduction dimension {reduction} is not divisible by activation group {activation_group}"
    )]
    IncompleteActivationGroup {
        /// Caller-supplied reduction width.
        reduction: usize,
        /// Selected activation scale group width.
        activation_group: usize,
    },
    /// A supplied buffer has an unexpected exact length.
    #[error("{field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Required elements or bytes.
        expected: usize,
        /// Actual supplied elements or bytes.
        actual: usize,
    },
    /// Matrix shape arithmetic did not fit addressable memory.
    #[error("activation quantization shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// Failing derived buffer role.
        field: &'static str,
    },
    /// A BF16 input denotes NaN or infinity.
    #[error("nonfinite BF16 activation at element {element}")]
    NonFiniteInput {
        /// Flat BF16-input index.
        element: usize,
    },
}

/// Quantizes BF16 activation storage bits into E4M3FN codes and E8M0 scales.
///
/// `input_bf16` and `codes` are `[rows, reduction]`; `scales` is
/// `[rows, reduction / activation_group]`. The implementation covers only the
/// pinned non-inplace path with power-of-two E8M0 scales. For each group it
/// promotes BF16 bits exactly to FP32, obtains `amax = max(abs(input), 1e-4)`,
/// computes `2^ceil(log2(amax * (1 / 448)))` using FP32 arithmetic, stores that
/// exact power as E8M0, then clamps and encodes each normalized value as E4M3FN.
///
/// E4M3FN encoding is software round-to-nearest, ties-to-even. The pinned
/// `TileLang` source does not itself establish that this is hardware cast parity.
/// All validation completes before either output buffer is written.
pub fn quantize_bf16_activations_e4m3fn(
    input_bf16: &[u16],
    rows: usize,
    reduction: usize,
    activation_group: ActivationGroup,
    codes: &mut [u8],
    scales: &mut [u8],
) -> Result<(), ActivationQuantError> {
    let shape = ActivationShape::new(rows, reduction, activation_group)?;
    shape.validate_lengths(input_bf16, codes, scales)?;
    for (element, &bits) in input_bf16.iter().enumerate() {
        if !decode_bf16(bits).is_finite() {
            return Err(ActivationQuantError::NonFiniteInput { element });
        }
    }

    for row in 0..shape.rows {
        for group in 0..shape.groups_per_row {
            let input_start = row * shape.reduction + group * shape.activation_group;
            let input_end = input_start + shape.activation_group;
            let input = &input_bf16[input_start..input_end];
            let (scale, scale_code) = group_scale(input);
            let scale_index = row * shape.groups_per_row + group;
            scales[scale_index] = scale_code;
            for (offset, &bits) in input.iter().enumerate() {
                let normalized = (decode_bf16(bits) / scale).clamp(-FP8_MAX, FP8_MAX);
                codes[input_start + offset] = encode_e4m3fn_rne(normalized);
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct ActivationShape {
    rows: usize,
    reduction: usize,
    activation_group: usize,
    groups_per_row: usize,
}

impl ActivationShape {
    fn new(
        rows: usize,
        reduction: usize,
        activation_group: ActivationGroup,
    ) -> Result<Self, ActivationQuantError> {
        if rows == 0 || reduction == 0 {
            return Err(ActivationQuantError::EmptyDimension);
        }
        let activation_group = activation_group.elements();
        if !reduction.is_multiple_of(activation_group) {
            return Err(ActivationQuantError::IncompleteActivationGroup {
                reduction,
                activation_group,
            });
        }
        Ok(Self {
            rows,
            reduction,
            activation_group,
            groups_per_row: reduction / activation_group,
        })
    }

    fn validate_lengths(
        self,
        input_bf16: &[u16],
        codes: &[u8],
        scales: &[u8],
    ) -> Result<(), ActivationQuantError> {
        check_length(
            "input_bf16",
            self.rows.checked_mul(self.reduction),
            input_bf16.len(),
        )?;
        check_length("codes", self.rows.checked_mul(self.reduction), codes.len())?;
        check_length(
            "scales",
            self.rows.checked_mul(self.groups_per_row),
            scales.len(),
        )
    }
}

fn check_length(
    field: &'static str,
    expected: Option<usize>,
    actual: usize,
) -> Result<(), ActivationQuantError> {
    let expected = expected.ok_or(ActivationQuantError::ShapeOverflow { field })?;
    if actual != expected {
        return Err(ActivationQuantError::Length {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn decode_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn group_scale(input: &[u16]) -> (f32, u8) {
    let mut amax = AMAX_FLOOR;
    for &bits in input {
        amax = amax.max(decode_bf16(bits).abs());
    }
    let scaled_amax = amax * FP8_MAX_INV;
    let exponent = ceil_log2_normal(scaled_amax);
    let biased_exponent = exponent + 127;
    let computed_scale = f32::from_bits(
        u32::try_from(biased_exponent)
            .expect("finite BF16 activation scale exponents are nonnegative")
            << 23,
    );
    let scale_code = u8::try_from(biased_exponent)
        .expect("finite BF16 activation scales always fit the E8M0 exponent range");
    debug_assert_eq!(decode_e8m0(scale_code).to_bits(), computed_scale.to_bits());
    (computed_scale, scale_code)
}

fn ceil_log2_normal(value: f32) -> i32 {
    debug_assert!(value.is_finite() && value.is_normal() && value.is_sign_positive());
    let bits = value.to_bits();
    let exponent =
        i32::try_from((bits >> 23) & 0xff).expect("an FP32 exponent field always fits i32") - 127;
    exponent + i32::from(bits & 0x007f_ffff != 0)
}

fn encode_e4m3fn_rne(value: f32) -> u8 {
    debug_assert!(value.is_finite() && (-FP8_MAX..=FP8_MAX).contains(&value));
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = f64::from(value.abs());
    let mut best_code = 0_u8;
    let mut best_distance = magnitude;
    for code in 1_u8..=0x7e {
        let distance = (magnitude - f64::from(decode_e4m3fn(code))).abs();
        if distance < best_distance
            || (distance.to_bits() == best_distance.to_bits() && code & 1 == 0)
        {
            best_code = code;
            best_distance = distance;
        }
    }
    sign | best_code
}

#[cfg(test)]
mod tests {
    use super::{
        ActivationGroup, ActivationQuantError, decode_bf16, decode_e4m3fn, encode_e4m3fn_rne,
        quantize_bf16_activations_e4m3fn,
    };
    use crate::precision::fp4_linear_runtime_f32;

    const BF16_ZERO: u16 = 0x0000;

    fn assert_bits_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
            "actual {actual:?} differs from expected {expected:?}"
        );
    }

    #[test]
    fn bf16_promotion_preserves_exact_top_bits_and_signed_zero() {
        assert_bits_eq(
            &[
                decode_bf16(0x3f80),
                decode_bf16(0x8000),
                decode_bf16(0x7f7f),
            ],
            &[1.0, -0.0, f32::from_bits(0x7f7f_0000)],
        );
    }

    #[test]
    fn rne_round_trips_every_finite_code_and_all_adjacent_midpoints() {
        for code in 0_u8..=0x7e {
            let value = decode_e4m3fn(code);
            assert_eq!(encode_e4m3fn_rne(value), code, "positive code {code:#04x}");
            assert_eq!(
                encode_e4m3fn_rne(-value),
                code | 0x80,
                "negative code {code:#04x}"
            );
        }
        for lower in 0_u8..0x7e {
            let upper = lower + 1;
            let midpoint = decode_e4m3fn(lower).midpoint(decode_e4m3fn(upper));
            let expected = if lower & 1 == 0 { lower } else { upper };
            assert_eq!(
                encode_e4m3fn_rne(midpoint),
                expected,
                "positive midpoint {lower:#04x}"
            );
            let below = f32::from_bits(midpoint.to_bits() - 1);
            let above = f32::from_bits(midpoint.to_bits() + 1);
            assert_eq!(
                encode_e4m3fn_rne(below),
                lower,
                "below midpoint {lower:#04x}"
            );
            assert_eq!(
                encode_e4m3fn_rne(above),
                upper,
                "above midpoint {lower:#04x}"
            );
            assert_eq!(
                encode_e4m3fn_rne(-midpoint),
                expected | 0x80,
                "negative midpoint {lower:#04x}"
            );
            assert_eq!(
                encode_e4m3fn_rne(-below),
                lower | 0x80,
                "negative below midpoint {lower:#04x}"
            );
            assert_eq!(
                encode_e4m3fn_rne(-above),
                upper | 0x80,
                "negative above midpoint {lower:#04x}"
            );
        }
    }

    #[test]
    fn maximum_bf16_uses_representable_scale_and_preserves_sign() {
        let mut input = [0_u16; 32];
        input[0] = 0x7f7f;
        input[1] = 0xff7f;
        let mut codes = [0x55; 32];
        let mut scales = [0x55];
        quantize_bf16_activations_e4m3fn(
            &input,
            1,
            32,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )
        .expect("finite BF16 extrema");
        assert_eq!(scales, [247]);
        assert_eq!(&codes[..2], &[0x78, 0xf8]);
        input[31] = 0x7fc0;
        let prior_codes = codes;
        let prior_scales = scales;
        assert_eq!(
            quantize_bf16_activations_e4m3fn(
                &input,
                1,
                32,
                ActivationGroup::Elements32,
                &mut codes,
                &mut scales,
            ),
            Err(ActivationQuantError::NonFiniteInput { element: 31 })
        );
        assert_eq!(codes, prior_codes);
        assert_eq!(scales, prior_scales);
    }

    #[test]
    fn floor_scale_and_hand_rne_boundaries_are_exact() {
        let mut input = [BF16_ZERO; 32];
        input[1] = 0x8000;
        let mut codes = [0x55; 32];
        let mut scales = [0x55; 1];
        quantize_bf16_activations_e4m3fn(
            &input,
            1,
            32,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )
        .expect("finite complete group");
        assert_eq!(scales, [105]);
        assert_eq!(codes[0], 0x00);
        assert_eq!(codes[1], 0x80);

        let mut boundary = [BF16_ZERO; 32];
        boundary[..7].copy_from_slice(&[0x43e0, 0x43d8, 0x3f88, 0x3f98, 0x3a80, 0xba80, 0x3c70]);
        quantize_bf16_activations_e4m3fn(
            &boundary,
            1,
            32,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )
        .expect("finite RNE boundaries");
        assert_eq!(scales, [127]);
        assert_eq!(&codes[..7], &[0x7e, 0x7e, 0x38, 0x3a, 0x00, 0x80, 0x08]);
    }

    #[test]
    fn rows_and_groups_have_independent_power_of_two_scales() {
        let mut input = [BF16_ZERO; 256];
        input[0] = 0x4360; // 224
        input[32] = 0x4361; // 225
        input[128] = 0x43e0; // 448
        input[160] = 0x4360; // 224
        let mut codes = [0_u8; 256];
        let mut scales = [0_u8; 8];
        quantize_bf16_activations_e4m3fn(
            &input,
            2,
            128,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )
        .expect("two independent rows and groups");
        assert_eq!(scales, [126, 127, 105, 105, 127, 126, 105, 105]);
        assert_eq!(codes[0], 0x7e);
        assert_eq!(codes[32], 0x76);
        assert_eq!(codes[128], 0x7e);
        assert_eq!(codes[160], 0x7e);
    }

    #[test]
    fn composed_fp8_activation_and_fp4_linear_scalar_reference() {
        let mut input = [0x3f80_u16; 64];
        input[32..].fill(0x4000);
        let mut activation_codes = [0_u8; 64];
        let mut activation_scales = [0_u8; 2];
        quantize_bf16_activations_e4m3fn(
            &input,
            2,
            32,
            ActivationGroup::Elements32,
            &mut activation_codes,
            &mut activation_scales,
        )
        .expect("finite BF16 rows");
        assert_eq!(activation_codes, [0x78; 64]);
        assert_eq!(activation_scales, [119, 120]);

        let mut output = [0.0_f32; 4];
        let mut weights = [0_u8; 32];
        weights[..16].fill(0x11);
        weights[16..].fill(0x22);
        fp4_linear_runtime_f32(
            &activation_codes,
            &activation_scales,
            &weights,
            &[127, 127],
            2,
            32,
            2,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect("scalar runtime composition");
        assert_bits_eq(&output, &[16.0, 32.0, 32.0, 64.0]);
    }

    #[test]
    fn group_128_shares_one_scale_and_errors_are_output_atomic() {
        let mut input = [BF16_ZERO; 128];
        input[0] = 0x4360;
        input[32] = 0x4361;
        input[64] = 0x43e0;
        let mut codes = [0xa5; 128];
        let mut scales = [0xa5; 1];
        quantize_bf16_activations_e4m3fn(
            &input,
            1,
            128,
            ActivationGroup::Elements128,
            &mut codes,
            &mut scales,
        )
        .expect("one G128 scale");
        assert_eq!(scales, [127]);
        assert_eq!(codes[0], 0x76);
        assert_eq!(codes[32], 0x76);
        assert_eq!(codes[64], 0x7e);

        input[127] = 0x7fc0;
        codes.fill(0xa5);
        scales.fill(0xa5);
        let error = quantize_bf16_activations_e4m3fn(
            &input,
            1,
            128,
            ActivationGroup::Elements128,
            &mut codes,
            &mut scales,
        )
        .expect_err("NaN BF16 input");
        assert_eq!(error, ActivationQuantError::NonFiniteInput { element: 127 });
        assert_eq!(codes, [0xa5; 128]);
        assert_eq!(scales, [0xa5]);

        input[127] = 0x7f80;
        let error = quantize_bf16_activations_e4m3fn(
            &input,
            1,
            128,
            ActivationGroup::Elements128,
            &mut codes,
            &mut scales,
        )
        .expect_err("infinite BF16 input");
        assert_eq!(error, ActivationQuantError::NonFiniteInput { element: 127 });
        assert_eq!(codes, [0xa5; 128]);
        assert_eq!(scales, [0xa5]);
    }

    #[test]
    fn rejects_lengths_incomplete_groups_and_shape_overflow_before_writes() {
        let mut codes = [0x55; 32];
        let mut scales = [0x55; 1];
        let error = quantize_bf16_activations_e4m3fn(
            &[BF16_ZERO; 32],
            1,
            32,
            ActivationGroup::Elements32,
            &mut codes,
            &mut [],
        )
        .expect_err("missing scale");
        assert_eq!(
            error,
            ActivationQuantError::Length {
                field: "scales",
                expected: 1,
                actual: 0,
            }
        );
        assert_eq!(codes, [0x55; 32]);
        assert_eq!(scales, [0x55]);

        let error = quantize_bf16_activations_e4m3fn(
            &[BF16_ZERO; 64],
            1,
            64,
            ActivationGroup::Elements128,
            &mut codes,
            &mut scales,
        )
        .expect_err("incomplete G128 group");
        assert!(matches!(
            error,
            ActivationQuantError::IncompleteActivationGroup { .. }
        ));
        assert_eq!(codes, [0x55; 32]);
        assert_eq!(scales, [0x55]);

        let error = quantize_bf16_activations_e4m3fn(
            &[],
            usize::MAX,
            32,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )
        .expect_err("rows times reduction cannot fit usize");
        assert_eq!(
            error,
            ActivationQuantError::ShapeOverflow {
                field: "input_bf16"
            }
        );
        assert_eq!(codes, [0x55; 32]);
        assert_eq!(scales, [0x55]);
    }
}
