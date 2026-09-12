//! Scalar FP4 linear reference over already-packed runtime buffers.
//!
//! This reproduces the pinned runtime equation for complete K groups: one
//! unscaled 32-term dot, followed by activation and weight scale application.
//! It is neither a checkpoint decoder nor a CUDA, Tensor Core, or BF16-output
//! parity oracle.

use thiserror::Error;

use super::{decode_e2m1x2, decode_e4m3fn, decode_e8m0};

const WEIGHT_GROUP: usize = 32;

/// Activation-scale grouping accepted by the pinned FP4 linear runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActivationGroup {
    /// One activation scale for every 32 reduction elements.
    Elements32,
    /// One activation scale shared across four 32-element reduction blocks.
    Elements128,
}

impl ActivationGroup {
    const fn elements(self) -> usize {
        match self {
            Self::Elements32 => 32,
            Self::Elements128 => 128,
        }
    }
}

/// An invalid direct-runtime FP4 linear input or FP32 result.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum Fp4LinearError {
    /// Matrix dimensions require at least one row, output, and reduction element.
    #[error("FP4 linear dimensions must all be nonzero")]
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
    #[error("FP4 linear shape arithmetic overflowed for {field}")]
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
    /// A scaled block contribution or final accumulation cannot remain FP32.
    #[error("FP32 FP4 linear result overflowed at row {row}, output {output}")]
    ValueOverflow {
        /// Logical activation row.
        row: usize,
        /// Logical output row of the transposed weight matrix.
        output: usize,
    },
}

/// Computes a scalar FP32 reference for the pinned FP8-activation × FP4-weight equation.
///
/// `activation_codes` holds E4M3FN runtime codes shaped `[rows, reduction]`.
/// `activation_scales` holds E8M0 runtime codes shaped
/// `[rows, reduction / activation_group]`. `weight_codes` holds low-nibble-first
/// `E2M1x2` bytes shaped `[outputs, reduction / 2]`; its logical matrix is
/// `[outputs, reduction]`. `weight_scales` holds E8M0 codes shaped
/// `[outputs, reduction / 32]`. The supplied `output` is `[rows, outputs]` and
/// receives `activation @ weight.transpose()`.
///
/// Every complete 32-element dot is formed before both scales are multiplied
/// in, matching the pinned kernel's stated scale placement. This function uses
/// scalar FP32 operations only. It neither quantizes activations, decodes a
/// checkpoint file, nor establishes CUDA reduction order or BF16 output parity.
///
/// All validation and overflow checks finish before output writes begin, so an
/// error leaves `output` unchanged.
#[allow(
    clippy::too_many_arguments,
    reason = "the direct runtime-buffer contract keeps each shape and scale role explicit"
)]
pub fn fp4_linear_runtime_f32(
    activation_codes: &[u8],
    activation_scales: &[u8],
    weight_codes: &[u8],
    weight_scales: &[u8],
    rows: usize,
    reduction: usize,
    outputs: usize,
    activation_group: ActivationGroup,
    output: &mut [f32],
) -> Result<(), Fp4LinearError> {
    let shape = LinearShape::new(rows, reduction, outputs, activation_group)?;
    shape.validate_lengths(
        activation_codes,
        activation_scales,
        weight_codes,
        weight_scales,
        output,
    )?;
    validate_codes(activation_codes, activation_scales, weight_scales)?;

    // Validate all numerical results before the write pass. The write pass has
    // identical, infallible scalar arithmetic over inputs established above.
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
    weight_blocks_per_row: usize,
}

impl LinearShape {
    fn new(
        rows: usize,
        reduction: usize,
        outputs: usize,
        activation_group: ActivationGroup,
    ) -> Result<Self, Fp4LinearError> {
        if rows == 0 || reduction == 0 || outputs == 0 {
            return Err(Fp4LinearError::EmptyDimension);
        }
        let activation_group = activation_group.elements();
        if !reduction.is_multiple_of(activation_group) {
            return Err(Fp4LinearError::IncompleteActivationGroup {
                reduction,
                activation_group,
            });
        }
        Ok(Self {
            rows,
            reduction,
            outputs,
            activation_group,
            activation_scales_per_row: reduction / activation_group,
            weight_blocks_per_row: reduction / WEIGHT_GROUP,
        })
    }

    fn validate_lengths(
        self,
        activation_codes: &[u8],
        activation_scales: &[u8],
        weight_codes: &[u8],
        weight_scales: &[u8],
        output: &[f32],
    ) -> Result<(), Fp4LinearError> {
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
            self.outputs.checked_mul(self.reduction / 2),
            weight_codes.len(),
        )?;
        check_length(
            "weight_scales",
            self.outputs.checked_mul(self.weight_blocks_per_row),
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
    ) -> Result<f32, Fp4LinearError> {
        let mut accumulated = 0.0_f32;
        for block in 0..self.weight_blocks_per_row {
            let activation_base = row * self.reduction + block * WEIGHT_GROUP;
            let weight_base = column * (self.reduction / 2) + block * (WEIGHT_GROUP / 2);
            let mut dot = 0.0_f32;
            for pair in 0..(WEIGHT_GROUP / 2) {
                let [low, high] = decode_e2m1x2(weight_codes[weight_base + pair]);
                dot += decode_e4m3fn(activation_codes[activation_base + pair * 2]) * low;
                dot += decode_e4m3fn(activation_codes[activation_base + pair * 2 + 1]) * high;
            }
            let activation_scale_index = row * self.activation_scales_per_row
                + (block * WEIGHT_GROUP) / self.activation_group;
            let weight_scale_index = column * self.weight_blocks_per_row + block;
            let scaled = dot
                * decode_e8m0(activation_scales[activation_scale_index])
                * decode_e8m0(weight_scales[weight_scale_index]);
            if !scaled.is_finite() {
                return Err(Fp4LinearError::ValueOverflow {
                    row,
                    output: column,
                });
            }
            accumulated += scaled;
            if !accumulated.is_finite() {
                return Err(Fp4LinearError::ValueOverflow {
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
) -> Result<(), Fp4LinearError> {
    let expected = expected.ok_or(Fp4LinearError::ShapeOverflow { field })?;
    if actual != expected {
        return Err(Fp4LinearError::Length {
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
    weight_scales: &[u8],
) -> Result<(), Fp4LinearError> {
    for (element, &code) in activation_codes.iter().enumerate() {
        if !decode_e4m3fn(code).is_finite() {
            return Err(Fp4LinearError::NonFiniteActivation { element });
        }
    }
    for (index, &code) in activation_scales.iter().enumerate() {
        if !decode_e8m0(code).is_finite() {
            return Err(Fp4LinearError::NonFiniteActivationScale { index });
        }
    }
    for (index, &code) in weight_scales.iter().enumerate() {
        if !decode_e8m0(code).is_finite() {
            return Err(Fp4LinearError::NonFiniteWeightScale { index });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ActivationGroup, Fp4LinearError, fp4_linear_runtime_f32};

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
    fn computes_transposed_rows_with_distinct_32_element_scales() {
        let activations = [ONE; 64];
        let activation_scales = [127, 128];
        let mut weights = [0x11; 64];
        weights[32..].fill(0x22);
        let weight_scales = [127, 126, 128, 127];
        let mut output = [0.0; 2];

        fp4_linear_runtime_f32(
            &activations,
            &activation_scales,
            &weights,
            &weight_scales,
            1,
            64,
            2,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect("complete runtime groups");

        // Output 0: 16*1*1 + 16*2*(1/2) = 32.
        // Output 1: 32*1*2 + 32*2*1 = 128.
        assert_bits_eq(&output, &[32.0, 128.0]);
    }

    #[test]
    fn group_128_shares_one_activation_scale_and_preserves_packed_signs() {
        let mut activations = [0_u8; 128];
        activations[0] = ONE;
        activations[1] = NEG_ONE;
        let mut weights = [0x00; 64];
        weights[0] = 0xe1;
        let mut output = [0.0; 1];

        fp4_linear_runtime_f32(
            &activations,
            &[128],
            &weights,
            &[127, 127, 127, 127],
            1,
            128,
            1,
            ActivationGroup::Elements128,
            &mut output,
        )
        .expect("complete runtime groups");

        // Low/high E2M1 lanes are +0.5 and -4.0: 1*0.5 + (-1)*(-4) = 4.5.
        // The one G=128 activation scale is 2.
        assert_bits_eq(&output, &[9.0]);
    }

    #[test]
    fn rows_have_independent_activation_code_and_scale_strides() {
        let mut activations = [ONE; 256];
        activations[128..].fill(NEG_ONE);
        let mut weights = [0_u8; 128];
        weights[..64].fill(0x11);
        weights[64..].fill(0x22);
        let mut output = [0.0; 4];

        fp4_linear_runtime_f32(
            &activations,
            &[127, 128, 129, 130, 131, 132, 133, 134],
            &weights,
            &[127, 128, 129, 130, 131, 132, 133, 134],
            2,
            128,
            2,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect("independent runtime rows");

        assert_bits_eq(&output, &[1_360.0, 43_520.0, -21_760.0, -696_320.0]);
    }

    #[test]
    fn group_128_reuses_one_scale_across_four_distinct_weight_blocks() {
        let activations = [ONE; 128];
        let mut weights = [0_u8; 64];
        for (block, code) in [0x11, 0x22, 0x33, 0x44].into_iter().enumerate() {
            weights[block * 16..(block + 1) * 16].fill(code);
        }
        let weight_scales = [127, 128, 126, 129];
        let mut group_128 = [0.0; 1];
        let mut group_32 = [0.0; 1];

        fp4_linear_runtime_f32(
            &activations,
            &[128],
            &weights,
            &weight_scales,
            1,
            128,
            1,
            ActivationGroup::Elements128,
            &mut group_128,
        )
        .expect("one G128 activation scale");
        fp4_linear_runtime_f32(
            &activations,
            &[128, 128, 128, 128],
            &weights,
            &weight_scales,
            1,
            128,
            1,
            ActivationGroup::Elements32,
            &mut group_32,
        )
        .expect("equivalent G32 scales");

        // 2 * (16*1 + 32*2 + 48*(1/2) + 64*4).
        assert_bits_eq(&group_128, &[720.0]);
        assert_bits_eq(&group_32, &group_128);
    }

    #[test]
    fn failures_are_output_atomic_and_lengths_are_exact() {
        let activations = [ONE; 32];
        let weights = [0x11; 16];
        let mut output = [42.0; 1];
        let error = fp4_linear_runtime_f32(
            &activations,
            &[127],
            &weights,
            &[255],
            1,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("NaN scale");
        assert_eq!(error, Fp4LinearError::NonFiniteWeightScale { index: 0 });
        assert_bits_eq(&output, &[42.0]);

        let error = fp4_linear_runtime_f32(
            &activations,
            &[],
            &weights,
            &[127],
            1,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("missing scale");
        assert_eq!(
            error,
            Fp4LinearError::Length {
                field: "activation_scales",
                expected: 1,
                actual: 0,
            }
        );
        assert_bits_eq(&output, &[42.0]);
    }

    #[test]
    fn rejects_incomplete_groups_and_later_result_overflow_before_writing() {
        let mut output = [7.0; 2];
        assert!(matches!(
            fp4_linear_runtime_f32(
                &[ONE; 64],
                &[127],
                &[0x11; 32],
                &[127, 127],
                1,
                64,
                1,
                ActivationGroup::Elements128,
                &mut output,
            ),
            Err(Fp4LinearError::IncompleteActivationGroup { .. })
        ));
        assert_bits_eq(&output, &[7.0, 7.0]);

        let error = fp4_linear_runtime_f32(
            &[ONE; 32],
            &[127],
            &[0x11; 32],
            &[127, 254],
            1,
            32,
            2,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("second output's scaled block exceeds FP32");
        assert_eq!(error, Fp4LinearError::ValueOverflow { row: 0, output: 1 });
        assert_bits_eq(&output, &[7.0, 7.0]);
    }

    #[test]
    fn finite_blocks_can_overflow_only_when_accumulated() {
        let mut output = [7.0];
        // Each dot is 32 * 448 * 6 = 86016. Multiplying by 2^111 is
        // finite, but adding two such block contributions exceeds FP32.
        let error = fp4_linear_runtime_f32(
            &[0x7e; 64],
            &[238, 238],
            &[0x77; 32],
            &[127, 127],
            1,
            64,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("finite block contributions overflow on accumulation");
        assert_eq!(error, Fp4LinearError::ValueOverflow { row: 0, output: 0 });
        assert_bits_eq(&output, &[7.0]);
    }

    #[test]
    fn preserves_post_dot_scale_order_and_accepts_cancelling_large_scale_blocks() {
        let mut output = [5.0; 1];
        let error = fp4_linear_runtime_f32(
            &[ONE; 32],
            &[254],
            &[0x77; 16],
            &[0],
            1,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("dot times activation scale overflows before the weight scale rescales it");
        assert_eq!(error, Fp4LinearError::ValueOverflow { row: 0, output: 0 });
        assert_bits_eq(&output, &[5.0]);

        let mut cancelling_activations = [ONE; 32];
        for value in cancelling_activations.iter_mut().skip(1).step_by(2) {
            *value = NEG_ONE;
        }
        fp4_linear_runtime_f32(
            &cancelling_activations,
            &[127],
            &[0x77; 16],
            &[254],
            1,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect("unscaled cancellation occurs before the enormous weight scale");
        assert_bits_eq(&output, &[0.0]);
    }

    #[test]
    fn rejects_nonfinite_activation_and_scale_and_detects_shape_overflow() {
        let mut output = [11.0; 1];
        let activation_nan = fp4_linear_runtime_f32(
            &[0x7f; 32],
            &[127],
            &[0x11; 16],
            &[127],
            1,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("E4M3FN NaN activation");
        assert_eq!(
            activation_nan,
            Fp4LinearError::NonFiniteActivation { element: 0 }
        );
        assert_bits_eq(&output, &[11.0]);

        let scale_nan = fp4_linear_runtime_f32(
            &[ONE; 32],
            &[255],
            &[0x11; 16],
            &[127],
            1,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("E8M0 NaN activation scale");
        assert_eq!(
            scale_nan,
            Fp4LinearError::NonFiniteActivationScale { index: 0 }
        );
        assert_bits_eq(&output, &[11.0]);

        let overflow = fp4_linear_runtime_f32(
            &[],
            &[],
            &[],
            &[],
            usize::MAX,
            32,
            1,
            ActivationGroup::Elements32,
            &mut output,
        )
        .expect_err("rows times reduction cannot fit usize");
        assert_eq!(
            overflow,
            Fp4LinearError::ShapeOverflow {
                field: "activation_codes"
            }
        );
        assert_bits_eq(&output, &[11.0]);
    }
}
