//! Reference expansion for the V4.1 runtime's narrow floating-point types.
//!
//! Encodings follow OCP Microscaling Formats v1.0, sections 5.3 and 5.4.
//! These are decode, activation-quantization and scalar-arithmetic references,
//! not checkpoint converters or loaders.
//! Runtime pairs and contiguous 32-element scaled blocks are supported; mapping
//! checkpoint bytes to that runtime layout remains a separate contract.
//! Scalar decoders return NaNs; block expansion and linear arithmetic reject
//! non-finite results. Linear references preserve their specified per-group
//! dot/scale placement, not hardware reduction or BF16 output rounding.
//!
//! Everything here is scalar CPU code with no model or GPU dependency. It is
//! an independent oracle that faster kernels are tested against, so it keeps a
//! stated evaluation order rather than the fastest one.
//!
//! # Overview
//!
//! * Scalar codes: [`decode_e2m1`], [`decode_e2m1x2`], [`decode_e4m3fn`] and
//!   [`decode_e8m0`] expand one code to FP32. [`bf16_to_f32`] and
//!   [`f32_to_bf16_rne`] convert BF16 storage bits.
//! * Scaled blocks: [`expand_e2m1x2_blocks32`] expands packed FP4 values with
//!   one E8M0 scale per 32 elements.
//! * Activation quantization: [`quantize_bf16_activations_e4m3fn`] produces
//!   E4M3FN codes and E8M0 scales. [`requantize_bf16_activations_e4m3fn`] and
//!   [`requantize_bf16_activations_e2m1`] quantize and reconstruct BF16 in one
//!   call, which models the precision an in-place quantization keeps.
//! * Linear layers: [`fp4_linear_runtime_f32`] (FP8 activations, FP4 weights)
//!   and [`fp8_linear_runtime_f32`] (FP8 both) evaluate the runtime's grouped
//!   equations. [`bf16_linear_reference`] and [`fp32_linear_reference`] are
//!   plain matrix products in a fixed scalar order.
//!
//! # Example: expanding one FP4 block
//!
//! A runtime FP4 block is 16 bytes holding 32 E2M1 values, low nibble first,
//! with one E8M0 scale. Scale code 128 is 2.0.
//!
//! ```
//! use blockfloat::expand_e2m1x2_blocks32;
//!
//! // Low nibble 1 is 0.5 and high nibble 2 is 1.0, before scaling.
//! let packed = [0x21_u8; 16];
//! let mut output = [0.0_f32; 32];
//! expand_e2m1x2_blocks32(&packed, &[128], &mut output)?;
//! assert_eq!(&output[..4], &[1.0, 2.0, 1.0, 2.0]);
//! # Ok::<(), blockfloat::BlockDecodeError>(())
//! ```
//!
//! # Conventions
//!
//! These hold for every function in the crate and are not repeated on each:
//!
//! * Matrices are flat row-major slices, with shapes written `[rows, columns]`.
//!   Linear weights are `[outputs, reduction]`, and a linear result is
//!   `activations @ weights.transpose()`, shaped `[rows, outputs]`.
//! * Encoded values travel as raw storage: `u8` codes for E2M1 pairs, E4M3FN
//!   and E8M0, and `u16` bits for BF16.
//! * A fallible function validates its input and computes its whole result
//!   before it writes a caller buffer, so an error leaves every output buffer
//!   unchanged. An error that points at an element carries its flat index.
//! * The BF16, FP32 and requantization references bound their buffers with a
//!   `MAX_*` constant and return an error past it rather than run unbounded.
//! * Rounding to E4M3FN, E2M1 and BF16 is software round-to-nearest, ties to
//!   even. That is a reference choice, not a claim of parity with a GPU cast.
//!
//! "Pinned" in item docs refers to the
//! [DeepSeek-V4.1 inference kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py)
//! at that commit.

#![deny(missing_docs)]
// The workspace allows this lint; crates opt in once their docs are complete.
#![warn(clippy::missing_errors_doc)]

mod blocks;
pub use blocks::{BlockDecodeError, expand_e2m1x2_blocks32};
mod bf16_linear;
pub use bf16_linear::{Bf16LinearError, MAX_BF16_LINEAR_ELEMENTS, bf16_linear_reference};
pub use bf16_linear::{bf16_to_f32, f32_to_bf16_rne};
mod fp32_linear;
pub use fp32_linear::{Fp32LinearError, MAX_FP32_LINEAR_ELEMENTS, fp32_linear_reference};
mod activation;
pub use activation::{ActivationQuantError, quantize_bf16_activations_e4m3fn};
mod linear;
pub use linear::{
    ActivationGroup, Fp4LinearError, fp4_linear_runtime_f32, fp4_linear_runtime_f32_owned,
};
mod fp8_linear;
pub use fp8_linear::{Fp8LinearError, fp8_linear_runtime_f32};

mod roundtrip;
pub use roundtrip::{
    ActivationRoundtripError, MAX_ACTIVATION_ROUNDTRIP_ELEMENTS, requantize_bf16_activations_e4m3fn,
};
mod fp4_activation;
pub use fp4_activation::{
    Fp4ActivationError, Fp4ActivationMode, MAX_FP4_ACTIVATION_ELEMENTS,
    requantize_bf16_activations_e2m1,
};

/// Expands an E2M1 sign/exponent/mantissa nibble, preserving signed zero.
///
/// Returns `None` if any upper four bits are set. This function deliberately
/// does not choose which nibble of a packed checkpoint byte comes first.
///
/// # Example
///
/// ```
/// use blockfloat::decode_e2m1;
///
/// assert_eq!(decode_e2m1(0b0111), Some(6.0));
/// assert_eq!(decode_e2m1(0b1000).map(f32::to_bits), Some((-0.0_f32).to_bits()));
/// assert_eq!(decode_e2m1(0x10), None);
/// ```
#[must_use]
pub fn decode_e2m1(nibble: u8) -> Option<f32> {
    if nibble > 0x0f {
        return None;
    }
    Some(decode_nibble(nibble))
}

fn decode_nibble(nibble: u8) -> f32 {
    let magnitude = [0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(nibble & 7)];
    if nibble & 8 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Expands a `PyTorch` `E2M1x2` runtime byte in logical element order.
///
/// The low nibble is the first element, the high nibble the second. This
/// describes the typed runtime representation, not an uninspected checkpoint.
/// See [PyTorch's pinned encoding definition](https://github.com/pytorch/pytorch/blob/84e524623ea4754a748936bf1ba6ecaaa92c3ae6/torch/headeronly/util/Float4_e2m1fn_x2.h).
///
/// # Example
///
/// ```
/// use blockfloat::decode_e2m1x2;
///
/// // Low nibble 2 is 1.0; high nibble 7 is 6.0.
/// assert_eq!(decode_e2m1x2(0x72), [1.0, 6.0]);
/// ```
#[must_use]
pub fn decode_e2m1x2(byte: u8) -> [f32; 2] {
    [decode_nibble(byte & 15), decode_nibble(byte >> 4)]
}

/// Expands E4M3FN to FP32, including signed zero and subnormals.
///
/// Bytes `0x7f` and `0xff` produce NaN. No encoding represents infinity.
/// NaN payload and sign are unspecified.
///
/// # Example
///
/// ```
/// use blockfloat::decode_e4m3fn;
///
/// assert_eq!(decode_e4m3fn(0x38), 1.0);
/// assert_eq!(decode_e4m3fn(0x7e), 448.0); // the largest finite value
/// assert_eq!(decode_e4m3fn(0x01), 1.0 / 512.0); // the smallest subnormal
/// assert!(decode_e4m3fn(0xff).is_nan());
/// ```
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
///
/// # Example
///
/// ```
/// use blockfloat::decode_e8m0;
///
/// assert_eq!(decode_e8m0(127), 1.0);
/// assert_eq!(decode_e8m0(128), 2.0);
/// assert_eq!(decode_e8m0(0), f32::MIN_POSITIVE / 2.0); // 2^-127, a subnormal
/// assert!(decode_e8m0(0xff).is_nan());
/// ```
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
