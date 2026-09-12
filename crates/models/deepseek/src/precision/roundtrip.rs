//! Scalar BF16 → FP8 → BF16 reference for the pinned in-place activation path.
//!
//! This composes the existing software encoder with power-of-two dequantization.
//! It models the resulting BF16 values, not physical FP8 cache storage or GPU
//! cast parity. Nonfinite reconstruction is rejected rather than published.

use thiserror::Error;

use super::{
    ActivationGroup, ActivationQuantError, decode_e4m3fn, decode_e8m0,
    quantize_bf16_activations_e4m3fn,
};

/// Maximum number of activations in one bounded software round trip.
pub const MAX_ACTIVATION_ROUNDTRIP_ELEMENTS: usize = 1 << 20;

/// An invalid or nonfinite FP8 activation round trip.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum ActivationRoundtripError {
    /// The request exceeds the reference's allocation and work limit.
    #[error("activation round trip has {elements} elements, maximum is {maximum}")]
    TooManyElements {
        /// Supplied input element count.
        elements: usize,
        /// Largest accepted input element count.
        maximum: usize,
    },
    /// Caller output must have exactly one slot per input element.
    #[error("activation round-trip output length is {actual}, expected {expected}")]
    OutputLength {
        /// Supplied output element count.
        actual: usize,
        /// Required output element count.
        expected: usize,
    },
    /// The existing quantizer rejected the input shape or values.
    #[error(transparent)]
    Quantization(#[from] ActivationQuantError),
    /// FP32 dequantization or the final BF16 cast overflowed.
    #[error("activation reconstruction overflowed at element {element}")]
    ReconstructionOverflow {
        /// Flat input/output element index.
        element: usize,
    },
}

/// Quantizes BF16 activations to E4M3FN/E8M0 and reconstructs BF16 values.
///
/// Shapes and scale grouping match [`quantize_bf16_activations_e4m3fn`]. This
/// is the scalar equivalent of pinned `act_quant(..., inplace=True)` with
/// power-of-two scales. Unlike that kernel, it rejects nonfinite reconstructed
/// results. The entire result is validated before caller output is changed.
///
/// # Errors
///
/// Returns [`ActivationRoundtripError`] for invalid shapes, nonfinite input,
/// an oversized request, or reconstruction overflow; output remains unchanged.
pub fn requantize_bf16_activations_e4m3fn(
    input: &[u16],
    rows: usize,
    reduction: usize,
    group: ActivationGroup,
    output: &mut [u16],
) -> Result<(), ActivationRoundtripError> {
    if input.len() > MAX_ACTIVATION_ROUNDTRIP_ELEMENTS {
        return Err(ActivationRoundtripError::TooManyElements {
            elements: input.len(),
            maximum: MAX_ACTIVATION_ROUNDTRIP_ELEMENTS,
        });
    }
    if output.len() != input.len() {
        return Err(ActivationRoundtripError::OutputLength {
            actual: output.len(),
            expected: input.len(),
        });
    }
    let mut codes = vec![0_u8; input.len()];
    let mut scales = vec![0_u8; input.len() / group.elements()];
    quantize_bf16_activations_e4m3fn(input, rows, reduction, group, &mut codes, &mut scales)?;

    let mut result = Vec::with_capacity(input.len());
    for (element, &code) in codes.iter().enumerate() {
        let value = decode_e4m3fn(code) * decode_e8m0(scales[element / group.elements()]);
        if !value.is_finite() {
            return Err(ActivationRoundtripError::ReconstructionOverflow { element });
        }
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        let [_, _, low, high] = rounded.to_le_bytes();
        let bf16 = u16::from_le_bytes([low, high]);
        if !f32::from_bits(u32::from(bf16) << 16).is_finite() {
            return Err(ActivationRoundtripError::ReconstructionOverflow { element });
        }
        result.push(bf16);
    }
    output.copy_from_slice(&result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        ActivationGroup, ActivationRoundtripError, MAX_ACTIVATION_ROUNDTRIP_ELEMENTS,
        requantize_bf16_activations_e4m3fn,
    };

    #[test]
    fn reconstructs_each_rows_own_scale_and_preserves_signed_zero() {
        let mut input = [0x3f22; 64]; // 0.6328125 rounds through FP8 to 0.625.
        input[32..].fill(0x4301); // 129 rounds through FP8 to 128.
        input[0] = 0x0000;
        input[1] = 0x8000;
        let mut output = [0_u16; 64];
        requantize_bf16_activations_e4m3fn(&input, 2, 32, ActivationGroup::Elements32, &mut output)
            .expect("finite independent row scales");
        assert_eq!(&output[..2], &[0x0000, 0x8000]);
        assert_eq!(&output[2..32], &[0x3f20; 30]);
        assert_eq!(&output[32..], &[0x4300; 32]);
    }

    #[test]
    fn supports_g128_and_rejects_malformed_shapes_atomically() {
        let input = [0x4301; 256];
        let mut output = [0xdead; 256];
        requantize_bf16_activations_e4m3fn(
            &input,
            2,
            128,
            ActivationGroup::Elements128,
            &mut output,
        )
        .expect("two G128 rows");
        assert_eq!(output, [0x4300; 256]);

        output.fill(0xdead);
        assert!(matches!(
            requantize_bf16_activations_e4m3fn(
                &input,
                4,
                64,
                ActivationGroup::Elements128,
                &mut output,
            ),
            Err(ActivationRoundtripError::Quantization(_))
        ));
        assert_eq!(output, [0xdead; 256]);
        assert!(matches!(
            requantize_bf16_activations_e4m3fn(
                &input,
                2,
                128,
                ActivationGroup::Elements128,
                &mut output[..255],
            ),
            Err(ActivationRoundtripError::OutputLength {
                actual: 255,
                expected: 256
            })
        ));
        assert_eq!(output, [0xdead; 256]);
    }

    #[test]
    fn late_reconstruction_overflow_preserves_every_output_slot() {
        let mut input = [0x3f80; 64];
        input[63] = 0x7f7f; // Maximum finite BF16 rounds above FP32 range via FP8.
        let mut output = [0xdead; 64];
        assert_eq!(
            requantize_bf16_activations_e4m3fn(
                &input,
                2,
                32,
                ActivationGroup::Elements32,
                &mut output
            ),
            Err(ActivationRoundtripError::ReconstructionOverflow { element: 63 })
        );
        assert_eq!(output, [0xdead; 64]);
    }

    #[test]
    fn rejects_invalid_input_and_budget_without_publishing_output() {
        let input = vec![0_u16; MAX_ACTIVATION_ROUNDTRIP_ELEMENTS + 1];
        assert!(matches!(
            requantize_bf16_activations_e4m3fn(
                &input,
                1,
                input.len(),
                ActivationGroup::Elements32,
                &mut []
            ),
            Err(ActivationRoundtripError::TooManyElements { .. })
        ));
        let mut input = [0x3f80; 32];
        input[31] = 0x7f80;
        let mut output = [0xdead; 32];
        assert!(matches!(
            requantize_bf16_activations_e4m3fn(
                &input,
                1,
                32,
                ActivationGroup::Elements32,
                &mut output
            ),
            Err(ActivationRoundtripError::Quantization(_))
        ));
        assert_eq!(output, [0xdead; 32]);
    }
}
