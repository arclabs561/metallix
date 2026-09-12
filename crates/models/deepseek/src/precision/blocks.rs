//! Caller-buffer expansion of contiguous runtime FP4 blocks, not file I/O.

use thiserror::Error;

use super::{decode_e2m1x2, decode_e8m0};

/// An invalid packed runtime block or output buffer.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum BlockDecodeError {
    /// Packed bytes must contain at least one complete 32-element block.
    #[error("packed FP4 bytes must be nonempty and divisible by 16")]
    PackedLength,
    /// Each block requires one E8M0 scale.
    #[error("expected one E8M0 scale per 16 packed bytes")]
    ScaleLength,
    /// The output must hold exactly two FP32 elements per packed byte.
    #[error("output length must equal twice the packed byte length")]
    OutputLength,
    /// The E8M0 code denotes NaN.
    #[error("nonfinite E8M0 scale in block {block}")]
    NonFiniteScale { block: usize },
    /// Applying a finite scale exceeds the FP32 output range.
    #[error("scaled FP4 value exceeds FP32 range at element {element}")]
    ValueOverflow { element: usize },
}

/// Expands contiguous 32-element `E2M1x2` runtime blocks with E8M0 scales.
///
/// Each 16-byte block uses the corresponding scale; low nibbles precede high
/// nibbles. This matches the pinned V4.1 FP4 linear runtime for complete K
/// groups. Callers still establish tensor shape, row boundaries, scale order,
/// and file-to-runtime transformations. No padding or swizzle is inferred.
///
/// Allocates nothing. The caller owns the output buffer and its memory budget.
/// All input and range checks complete before any output is changed.
/// Multiplication uses FP32, not the upstream fused GEMM's accumulator order.
///
/// # Errors
///
/// Returns [`BlockDecodeError`] for incomplete blocks, mismatched buffers,
/// nonfinite scales or scaled-value overflow. Output is unchanged on error.
pub fn expand_e2m1x2_blocks32(
    packed: &[u8],
    scales: &[u8],
    output: &mut [f32],
) -> Result<(), BlockDecodeError> {
    if packed.is_empty() || !packed.len().is_multiple_of(16) {
        return Err(BlockDecodeError::PackedLength);
    }
    if scales.len() != packed.len() / 16 {
        return Err(BlockDecodeError::ScaleLength);
    }
    if packed.len().checked_mul(2) != Some(output.len()) {
        return Err(BlockDecodeError::OutputLength);
    }
    for (block, (&scale_code, bytes)) in scales.iter().zip(packed.chunks_exact(16)).enumerate() {
        let scale = decode_e8m0(scale_code);
        if !scale.is_finite() {
            return Err(BlockDecodeError::NonFiniteScale { block });
        }
        for (pair, &byte) in bytes.iter().enumerate() {
            for (lane, value) in decode_e2m1x2(byte).into_iter().enumerate() {
                if !(value * scale).is_finite() {
                    return Err(BlockDecodeError::ValueOverflow {
                        element: block * 32 + pair * 2 + lane,
                    });
                }
            }
        }
    }
    for ((bytes, &scale_code), destination) in packed
        .chunks_exact(16)
        .zip(scales)
        .zip(output.chunks_exact_mut(32))
    {
        let scale = decode_e8m0(scale_code);
        for (&byte, pair) in bytes.iter().zip(destination.chunks_exact_mut(2)) {
            let [first, second] = decode_e2m1x2(byte);
            pair[0] = first * scale;
            pair[1] = second * scale;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BlockDecodeError, expand_e2m1x2_blocks32};
    use crate::precision::decode_e2m1x2;

    #[test]
    fn all_packed_pairs_preserve_lane_order_and_signed_zero() {
        let values = [
            0.0_f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
            -6.0,
        ];
        for byte in 0_u8..=255 {
            let pair = decode_e2m1x2(byte);
            assert_eq!(pair[0].to_bits(), values[usize::from(byte % 16)].to_bits());
            assert_eq!(pair[1].to_bits(), values[usize::from(byte / 16)].to_bits());
        }
    }

    #[test]
    fn scale_changes_only_at_complete_block_boundaries() {
        let mut bytes = [0x80; 32];
        bytes[0] = 0xe1;
        bytes[1] = 0x4b;
        bytes[16] = 0xe1;
        bytes[17] = 0x4b;
        let mut output = [99.0; 64];
        expand_e2m1x2_blocks32(&bytes, &[127, 128], &mut output).expect("two valid blocks");
        for (offset, expected) in [
            (0, [0.5_f32, -4.0, -1.5, 2.0]),
            (32, [1.0, -8.0, -3.0, 4.0]),
        ] {
            for (actual, expected) in output[offset..offset + 4].iter().zip(expected) {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
        assert_eq!(output[30].to_bits(), 0.0_f32.to_bits());
        assert_eq!(output[31].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(output[63].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn every_failure_preserves_the_whole_output() {
        let cases: &[(&[u8], &[u8], usize, BlockDecodeError)] = &[
            (&[], &[], 0, BlockDecodeError::PackedLength),
            (&[0; 15], &[127], 30, BlockDecodeError::PackedLength),
            (&[0; 16], &[], 32, BlockDecodeError::ScaleLength),
            (&[0; 16], &[127], 31, BlockDecodeError::OutputLength),
            (
                &[0; 32],
                &[127, 255],
                64,
                BlockDecodeError::NonFiniteScale { block: 1 },
            ),
            (
                &[0x77; 32],
                &[127, 254],
                64,
                BlockDecodeError::ValueOverflow { element: 32 },
            ),
        ];
        for &(bytes, scales, length, expected) in cases {
            let mut output = vec![42.0_f32; length];
            assert_eq!(
                expand_e2m1x2_blocks32(bytes, scales, &mut output),
                Err(expected)
            );
            assert!(
                output
                    .iter()
                    .all(|value| value.to_bits() == 42.0_f32.to_bits())
            );
        }
    }

    #[test]
    fn minimum_scale_keeps_subnormal_values() {
        let mut output = [0.0; 32];
        expand_e2m1x2_blocks32(&[0x91; 16], &[0], &mut output).expect("finite subnormals");
        let expected = f32::MIN_POSITIVE / 4.0;
        assert_eq!(output[0].to_bits(), expected.to_bits());
        assert_eq!(output[1].to_bits(), (-expected).to_bits());
    }
}
