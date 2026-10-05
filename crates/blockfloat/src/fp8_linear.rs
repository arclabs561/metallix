//! Scalar FP8 linear reference over already-quantized runtime buffers.
//!
//! This implements the pinned per-group FP8 equation in FP32. It is not a
//! checkpoint decoder or a `TileLang`, CUDA, Tensor Core, or BF16-output oracle.

use thiserror::Error;

use super::{ActivationGroup, decode_e4m3fn, decode_e8m0};

/// An invalid direct-runtime FP8 linear input or FP32 result.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum Fp8LinearError {
    /// Matrix dimensions require at least one row, output, and reduction element.
    #[error("FP8 linear dimensions must all be nonzero")]
    EmptyDimension,
    /// The reduction dimension cannot be grouped as required by the runtime equation.
    #[error(
        "reduction dimension {reduction} is not divisible by activation group {activation_group}"
    )]
    IncompleteActivationGroup {
        /// Caller-supplied reduction width.
        reduction: usize,
        /// Selected activation scale group width.
        activation_group: usize,
    },
    /// A supplied direct-runtime buffer has an unexpected exact length.
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
    #[error("FP8 linear shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// Failing derived buffer role.
        field: &'static str,
    },
    /// An E4M3FN activation code denotes NaN.
    #[error("nonfinite E4M3FN activation at element {element}")]
    NonFiniteActivation {
        /// Flat activation-code index.
        element: usize,
    },
    /// An E4M3FN weight code denotes NaN.
    #[error("nonfinite E4M3FN weight at element {element}")]
    NonFiniteWeight {
        /// Flat weight-code index.
        element: usize,
    },
    /// An activation E8M0 scale code denotes NaN.
    #[error("nonfinite activation E8M0 scale at index {index}")]
    NonFiniteActivationScale {
        /// Flat activation-scale index.
        index: usize,
    },
    /// A weight E8M0 scale code denotes NaN.
    #[error("nonfinite weight E8M0 scale at index {index}")]
    NonFiniteWeightScale {
        /// Flat weight-scale index.
        index: usize,
    },
    /// A scaled group contribution or final accumulation cannot remain FP32.
    #[error("FP32 FP8 linear result overflowed at row {row}, output {output}")]
    ValueOverflow {
        /// Logical activation row.
        row: usize,
        /// Logical output row of the transposed weight matrix.
        output: usize,
    },
}

/// Computes a scalar FP32 reference for the pinned FP8 activation × FP8 weight equation.
///
/// `activation_codes` is E4M3FN `[rows, reduction]`; `activation_scales` is
/// E8M0 `[rows, reduction / activation_group]`; `weight_codes` is E4M3FN
/// `[outputs, reduction]`; and `weight_scales` is E8M0
/// `[ceil(outputs / activation_group), reduction / activation_group]`. Each
/// full activation group is dotted in FP32, then multiplied by its activation
/// and shared output-group weight scale before FP32 accumulation. `output` is
/// `[rows, outputs]`.
///
/// The result is explicitly scalar FP32. This makes no hardware reduction or
/// BF16 cast claim. Validation and numerical checks complete before output
/// writes, so errors leave `output` unchanged.
#[allow(
    clippy::too_many_arguments,
    reason = "the direct runtime-buffer contract keeps each shape and scale role explicit"
)]
pub fn fp8_linear_runtime_f32(
    activation_codes: &[u8],
    activation_scales: &[u8],
    weight_codes: &[u8],
    weight_scales: &[u8],
    rows: usize,
    reduction: usize,
    outputs: usize,
    activation_group: ActivationGroup,
    output: &mut [f32],
) -> Result<(), Fp8LinearError> {
    let shape = LinearShape::new(rows, reduction, outputs, activation_group)?;
    shape.validate_lengths(
        activation_codes,
        activation_scales,
        weight_codes,
        weight_scales,
        output,
    )?;
    validate_codes(
        activation_codes,
        activation_scales,
        weight_codes,
        weight_scales,
    )?;

    for row in 0..shape.rows {
        for column in 0..shape.outputs {
            let _ = shape.compute(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                row,
                column,
            )?;
        }
    }
    for row in 0..shape.rows {
        let destination = &mut output[row * shape.outputs..(row + 1) * shape.outputs];
        for (column, slot) in destination.iter_mut().enumerate() {
            *slot = shape.compute(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                row,
                column,
            )?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct LinearShape {
    rows: usize,
    reduction: usize,
    outputs: usize,
    activation_group: usize,
    activation_scales_per_row: usize,
    output_scale_groups: usize,
}

impl LinearShape {
    fn new(
        rows: usize,
        reduction: usize,
        outputs: usize,
        activation_group: ActivationGroup,
    ) -> Result<Self, Fp8LinearError> {
        if rows == 0 || reduction == 0 || outputs == 0 {
            return Err(Fp8LinearError::EmptyDimension);
        }
        let activation_group = activation_group.elements();
        if !reduction.is_multiple_of(activation_group) {
            return Err(Fp8LinearError::IncompleteActivationGroup {
                reduction,
                activation_group,
            });
        }
        let activation_scales_per_row = reduction / activation_group;
        let output_scale_groups = outputs.div_ceil(activation_group);
        Ok(Self {
            rows,
            reduction,
            outputs,
            activation_group,
            activation_scales_per_row,
            output_scale_groups,
        })
    }

    fn validate_lengths(
        self,
        activation_codes: &[u8],
        activation_scales: &[u8],
        weight_codes: &[u8],
        weight_scales: &[u8],
        output: &[f32],
    ) -> Result<(), Fp8LinearError> {
        check_length(
            "activation_codes",
            self.rows.checked_mul(self.reduction),
            activation_codes.len(),
        )?;
        check_length(
            "activation_scales",
            self.rows.checked_mul(self.activation_scales_per_row),
            activation_scales.len(),
        )?;
        check_length(
            "weight_codes",
            self.outputs.checked_mul(self.reduction),
            weight_codes.len(),
        )?;
        check_length(
            "weight_scales",
            self.output_scale_groups
                .checked_mul(self.activation_scales_per_row),
            weight_scales.len(),
        )?;
        check_length("output", self.rows.checked_mul(self.outputs), output.len())
    }

    fn compute(
        self,
        activation_codes: &[u8],
        activation_scales: &[u8],
        weight_codes: &[u8],
        weight_scales: &[u8],
        row: usize,
        column: usize,
    ) -> Result<f32, Fp8LinearError> {
        let mut accumulated = 0.0_f32;
        for group in 0..self.activation_scales_per_row {
            let start = group * self.activation_group;
            let mut dot = 0.0_f32;
            for offset in 0..self.activation_group {
                dot += decode_e4m3fn(activation_codes[row * self.reduction + start + offset])
                    * decode_e4m3fn(weight_codes[column * self.reduction + start + offset]);
            }
            let activation_scale =
                decode_e8m0(activation_scales[row * self.activation_scales_per_row + group]);
            let weight_scale = decode_e8m0(
                weight_scales
                    [(column / self.activation_group) * self.activation_scales_per_row + group],
            );
            let scaled = dot * activation_scale * weight_scale;
            if !scaled.is_finite() {
                return Err(Fp8LinearError::ValueOverflow {
                    row,
                    output: column,
                });
            }
            accumulated += scaled;
            if !accumulated.is_finite() {
                return Err(Fp8LinearError::ValueOverflow {
                    row,
                    output: column,
                });
            }
        }
        Ok(accumulated)
    }
}

fn check_length(
    field: &'static str,
    expected: Option<usize>,
    actual: usize,
) -> Result<(), Fp8LinearError> {
    let expected = expected.ok_or(Fp8LinearError::ShapeOverflow { field })?;
    if actual != expected {
        return Err(Fp8LinearError::Length {
            field,
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_codes(
    activation_codes: &[u8],
    activation_scales: &[u8],
    weight_codes: &[u8],
    weight_scales: &[u8],
) -> Result<(), Fp8LinearError> {
    for (element, &code) in activation_codes.iter().enumerate() {
        if !decode_e4m3fn(code).is_finite() {
            return Err(Fp8LinearError::NonFiniteActivation { element });
        }
    }
    for (element, &code) in weight_codes.iter().enumerate() {
        if !decode_e4m3fn(code).is_finite() {
            return Err(Fp8LinearError::NonFiniteWeight { element });
        }
    }
    for (index, &code) in activation_scales.iter().enumerate() {
        if !decode_e8m0(code).is_finite() {
            return Err(Fp8LinearError::NonFiniteActivationScale { index });
        }
    }
    for (index, &code) in weight_scales.iter().enumerate() {
        if !decode_e8m0(code).is_finite() {
            return Err(Fp8LinearError::NonFiniteWeightScale { index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ActivationGroup, Fp8LinearError, fp8_linear_runtime_f32};

    const ONE: u8 = 0x38;
    const NEG_ONE: u8 = 0xb8;

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
    fn computes_multirow_two_block_transposed_fp8_equation() {
        let mut activations = [ONE; 128];
        activations[64..].fill(NEG_ONE);
        let mut weights = [0_u8; 128];
        weights[..64].fill(ONE);
        weights[64..].fill(0x40); // +2
        let mut output = [0.0; 4];
        fp8_linear_runtime_f32(
            &activations,
            &[127, 128, 129, 130],
            &weights,
            &[127, 126],
            2,
            64,
            2,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect("complete scalar groups");
        // Row 0: 32*1*1 + 32*2*(1/2) = 64; output 1 doubles both terms.
        // Row 1 negates the projections with distinct activation scales 4 and 8.
        assert_bits_eq(&output, &[64.0, 128.0, -256.0, -512.0]);
    }

    #[test]
    fn weight_scales_change_only_at_the_pinned_output_group_boundary() {
        for (group, outputs, reduction, expected_first, expected_last) in [
            (ActivationGroup::Elements32, 33, 32, 32.0_f32, 64.0_f32),
            (ActivationGroup::Elements128, 129, 128, 128.0_f32, 256.0_f32),
        ] {
            let activations = vec![ONE; reduction];
            let weights = vec![ONE; outputs * reduction];
            let mut scales =
                vec![127_u8; outputs.div_ceil(group.elements()) * (reduction / group.elements())];
            let last_scale_group = scales.len() - 1;
            scales[last_scale_group] = 128;
            let mut output = vec![0.0_f32; outputs];
            fp8_linear_runtime_f32(
                &activations,
                &vec![127; reduction / group.elements()],
                &weights,
                &scales,
                1,
                reduction,
                outputs,
                group,
                &mut output,
            )
            .expect("output scale groups");
            assert_eq!(output[0].to_bits(), expected_first.to_bits());
            assert_eq!(output[outputs - 2].to_bits(), expected_first.to_bits());
            assert_eq!(output[outputs - 1].to_bits(), expected_last.to_bits());
        }
    }

    #[test]
    fn g32_weight_scale_grid_uses_both_output_and_reduction_coordinates() {
        let activations = [ONE; 64];
        let weights = [ONE; 33 * 64];
        let mut output = [0.0_f32; 33];
        fp8_linear_runtime_f32(
            &activations,
            &[127, 128],
            &weights,
            // [output-scale group, reduction-scale group] = [[1, 1/2], [4, 2]].
            &[127, 126, 129, 128],
            1,
            64,
            33,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect("complete two-dimensional scale grid");
        assert_eq!(output[0].to_bits(), 64.0_f32.to_bits());
        assert_eq!(output[31].to_bits(), 64.0_f32.to_bits());
        assert_eq!(output[32].to_bits(), 256.0_f32.to_bits());
    }

    #[test]
    fn rejects_short_weight_buffer_atomically() {
        let mut output = [7.0_f32; 2];
        let weights = [ONE; 63];
        let error = fp8_linear_runtime_f32(
            &[ONE; 32],
            &[127],
            &weights,
            &[127],
            1,
            32,
            2,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("weight length is exact");
        assert_eq!(
            error,
            Fp8LinearError::Length {
                field: "weight_codes",
                expected: 64,
                actual: 63,
            }
        );
        assert_bits_eq(&output, &[7.0, 7.0]);
    }

    #[test]
    fn rejects_other_malformed_buffers_atomically() {
        let mut output = [7.0_f32; 2];
        for (
            activation_codes,
            activation_scales,
            weight_codes,
            weight_scales,
            field,
            expected,
            actual,
        ) in [
            (
                &[ONE; 31][..],
                &[127][..],
                &[ONE; 64][..],
                &[127][..],
                "activation_codes",
                32,
                31,
            ),
            (
                &[ONE; 32][..],
                &[][..],
                &[ONE; 64][..],
                &[127][..],
                "activation_scales",
                1,
                0,
            ),
            (
                &[ONE; 32][..],
                &[127][..],
                &[ONE; 64][..],
                &[][..],
                "weight_scales",
                1,
                0,
            ),
        ] {
            let error = fp8_linear_runtime_f32(
                activation_codes,
                activation_scales,
                weight_codes,
                weight_scales,
                1,
                32,
                2,
                ActivationGroup::Elements32,
                &mut output,
            )
            .expect_err("every direct runtime buffer has an exact length");
            assert_eq!(
                error,
                Fp8LinearError::Length {
                    field,
                    expected,
                    actual,
                }
            );
            assert_bits_eq(&output, &[7.0, 7.0]);
        }

        let mut short_output = [7.0_f32; 1];
        let error = fp8_linear_runtime_f32(
            &[ONE; 32],
            &[127],
            &[ONE; 64],
            &[127],
            1,
            32,
            2,
            ActivationGroup::Elements32,
            &mut short_output,
        )
        .expect_err("output length is exact");
        assert_eq!(
            error,
            Fp8LinearError::Length {
                field: "output",
                expected: 2,
                actual: 1,
            }
        );
        assert_bits_eq(&short_output, &[7.0]);
    }

    #[test]
    fn rejects_nonfinite_codes_atomically() {
        let mut output = [7.0_f32; 2];
        for (activations, weights, activation_scales, weight_scales, expected) in [
            (
                &[0x7f; 32][..],
                &[ONE; 64][..],
                &[127][..],
                &[127][..],
                "activation",
            ),
            (
                &[ONE; 32][..],
                &[0x7f; 64][..],
                &[127][..],
                &[127][..],
                "weight",
            ),
            (
                &[ONE; 32][..],
                &[ONE; 64][..],
                &[255][..],
                &[127][..],
                "activation scale",
            ),
            (
                &[ONE; 32][..],
                &[ONE; 64][..],
                &[127][..],
                &[255][..],
                "weight scale",
            ),
        ] {
            let error = fp8_linear_runtime_f32(
                activations,
                activation_scales,
                weights,
                weight_scales,
                1,
                32,
                2,
                ActivationGroup::Elements32,
                &mut output,
            )
            .expect_err(expected);
            assert!(matches!(
                error,
                Fp8LinearError::NonFiniteActivation { .. }
                    | Fp8LinearError::NonFiniteWeight { .. }
                    | Fp8LinearError::NonFiniteActivationScale { .. }
                    | Fp8LinearError::NonFiniteWeightScale { .. }
            ));
            assert_bits_eq(&output, &[7.0, 7.0]);
        }
    }

    #[test]
    fn rejects_invalid_shapes_and_late_overflow_atomically() {
        let mut output = [7.0_f32; 2];
        let error = fp8_linear_runtime_f32(
            &[ONE; 32],
            &[127],
            &[ONE; 32],
            &[127],
            1,
            32,
            1,
            ActivationGroup::Elements128,
            &mut output[..1],
        )
        .expect_err("incomplete G128");
        assert!(matches!(
            error,
            Fp8LinearError::IncompleteActivationGroup { .. }
        ));

        let error = fp8_linear_runtime_f32(
            &[],
            &[],
            &[],
            &[],
            usize::MAX,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output[..1],
        )
        .expect_err("shape multiplication overflow");
        assert_eq!(
            error,
            Fp8LinearError::ShapeOverflow {
                field: "activation_codes"
            }
        );

        let mut overflow_weights = [0_u8; 64];
        overflow_weights[32..].fill(0x7e);
        let error = fp8_linear_runtime_f32(
            &[ONE; 32],
            &[127],
            &overflow_weights,
            &[254],
            1,
            32,
            2,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("second output overflows after first is finite");
        assert_eq!(error, Fp8LinearError::ValueOverflow { row: 0, output: 1 });
        assert_bits_eq(&output, &[7.0, 7.0]);
    }
}
