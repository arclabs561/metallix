//! Exact scalar expansion for the V4.1 reference's narrow floating-point types.
//!
//! Encodings follow OCP Microscaling Formats v1.0, sections 5.3 and 5.4.
//! These are decode-only references, not quantizers or packed tensor loaders.
//! Nibble order, scale association, block layout and arithmetic rounding remain
//! separate checkpoint/kernel contracts. NaNs are returned, not sanitized.

/// Expands an E2M1 sign/exponent/mantissa nibble, preserving signed zero.
///
/// Returns `None` if any upper four bits are set. This function deliberately
/// does not choose which nibble of a packed checkpoint byte comes first.
#[must_use]
pub fn decode_e2m1(nibble: u8) -> Option<f32> {
    if nibble > 0x0f {
        return None;
    }
    let magnitude = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(nibble & 7)];
    Some(if nibble & 8 == 0 {
        magnitude
    } else {
        -magnitude
    })
}

/// Expands E4M3FN to FP32, including signed zero and subnormals.
///
/// Bytes `0x7f` and `0xff` produce NaN. No encoding represents infinity.
/// NaN payload and sign are unspecified.
#[must_use]
pub fn decode_e4m3fn(byte: u8) -> f32 {
    let magnitude = byte & 0x7f;
    if magnitude == 0x7f {
        return f32::NAN;
    }
    let exponent = magnitude >> 3;
    let mantissa = magnitude & 7;
    let value = if exponent == 0 {
        f32::from(mantissa) / 512.0
    } else {
        // E4M3 bias 7 -> FP32 bias 127; all normal values expand exactly.
        f32::from_bits(((u32::from(exponent) + 120) << 23) | (u32::from(mantissa) << 20))
    };
    if byte & 0x80 == 0 { value } else { -value }
}

/// Expands an unsigned E8M0 scale to FP32.
///
/// Byte zero means 2^-127, not zero; `0xff` means NaN. All other codes
/// represent positive powers of two, including `0xfe` = 2^127.
#[must_use]
pub fn decode_e8m0(byte: u8) -> f32 {
    match byte {
        0 => f32::from_bits(0x0040_0000),
        255 => f32::NAN,
        _ => f32::from_bits(u32::from(byte) << 23),
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_e2m1, decode_e4m3fn, decode_e8m0};

    #[test]
    fn e2m1_exhaustive_codes_and_rejected_non_nibbles() {
        // Independently enumerate the complete signed scalar codebook.
        let expected = [
            0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        for byte in 0_u8..=255 {
            if let Some(value) = expected.get(usize::from(byte)) {
                assert_eq!(
                    decode_e2m1(byte).expect("nibble").to_bits(),
                    value.to_bits()
                );
            } else {
                assert!(decode_e2m1(byte).is_none());
            }
        }
    }

    #[test]
    fn e4m3fn_all_codes_match_scalar_equation() {
        for byte in 0_u8..=255 {
            let actual = decode_e4m3fn(byte);
            if byte & 0x7f == 0x7f {
                assert!(actual.is_nan());
                continue;
            }
            let exponent = i32::from((byte >> 3) & 15);
            let fraction = f32::from(byte & 7) / 8.0;
            let positive = if exponent == 0 {
                2.0_f32.powi(-6) * fraction
            } else {
                2.0_f32.powi(exponent - 7) * (1.0 + fraction)
            };
            let expected = if byte < 128 { positive } else { -positive };
            assert_eq!(actual.to_bits(), expected.to_bits(), "code {byte:#04x}");
        }
        for (byte, expected) in [(0x01, 1.0_f32 / 512.0), (0x08, 1.0 / 64.0), (0x7e, 448.0)] {
            assert_eq!(decode_e4m3fn(byte).to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn e8m0_all_codes_match_exponent_equation() {
        // Start at 2^-127 and double: independent of FP32 bit assembly.
        let mut expected = f32::MIN_POSITIVE / 2.0;
        for byte in 0_u8..=254 {
            assert_eq!(
                decode_e8m0(byte).to_bits(),
                expected.to_bits(),
                "code {byte}"
            );
            if byte < 254 {
                expected *= 2.0;
            }
        }
        assert_eq!(decode_e8m0(127).to_bits(), 1.0_f32.to_bits());
        assert!(decode_e8m0(255).is_nan());
    }
}
