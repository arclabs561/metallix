//! Bounded scalar FP32 linear reference.
//!
//! Activations are `[rows, reduction]`, weights are `[outputs, reduction]`,
//! and output is `[rows, outputs]`. This uses an explicit scalar FP32
//! multiply, then an ascending-reduction-index FP32 sum. It is not a
//! `PyTorch` or Metal reduction-order parity reference.

use thiserror::Error;

/// Largest one-buffer FP32 linear input, weight, or output accepted here.
pub const MAX_FP32_LINEAR_ELEMENTS: usize = 1 << 20;
const MAX_FP32_LINEAR_WORK: usize = 1 << 24;

/// An invalid bounded scalar FP32 linear request.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum Fp32LinearError {
    /// Rows, reduction width, and output width must all be nonzero.
    #[error("FP32 linear dimensions must all be nonzero")]
    EmptyDimension,
    /// A derived buffer or work count overflowed `usize` for this field.
    #[error("FP32 linear shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// The derived count that overflowed.
        field: &'static str,
    },
    /// This bounded buffer exceeds the scalar reference's fixed limit.
    #[error("FP32 linear {field} has {elements} elements, maximum is {maximum}")]
    ElementLimit {
        /// The bounded buffer name.
        field: &'static str,
        /// The requested element count.
        elements: usize,
        /// The fixed maximum element count.
        maximum: usize,
    },
    /// The scalar multiply-accumulate count exceeds this reference's fixed limit.
    #[error("FP32 linear work estimate {elements} exceeds maximum {maximum}")]
    WorkLimit {
        /// The requested scalar multiply-accumulate count.
        elements: usize,
        /// The fixed maximum work count.
        maximum: usize,
    },
    /// A bounded result vector could not be reserved.
    #[error("could not allocate {elements} FP32 linear result elements")]
    AllocationFailed {
        /// The result element count that could not be reserved.
        elements: usize,
    },
    /// A caller buffer has an unexpected exact length.
    #[error("FP32 linear {field} length is {actual}, expected {expected}")]
    Length {
        /// The caller buffer name.
        field: &'static str,
        /// The supplied element count.
        actual: usize,
        /// The shape-derived required element count.
        expected: usize,
    },
    /// An FP32 activation denotes NaN or infinity.
    #[error("nonfinite FP32 activation at element {element}")]
    NonFiniteActivation {
        /// The activation element index.
        element: usize,
    },
    /// An FP32 weight denotes NaN or infinity.
    #[error("nonfinite FP32 weight at element {element}")]
    NonFiniteWeight {
        /// The weight element index.
        element: usize,
    },
    /// A scalar FP32 product or sum overflowed to a nonfinite value.
    #[error("FP32 linear overflowed at {stage}, row {row}, output {output}")]
    ValueOverflow {
        /// Whether the overflow occurred in the product or sum stage.
        stage: &'static str,
        /// The output row being calculated.
        row: usize,
        /// The output column being calculated.
        output: usize,
    },
}

/// Multiplies FP32 activations `[rows, reduction]` by FP32 weights
/// `[outputs, reduction]` and writes FP32 `[rows, outputs]`.
///
/// Every input, each scalar product, and each scalar sum must be finite. The
/// result is staged before copying to `output`, so every error leaves the
/// caller's output untouched.
///
/// ```
/// use deepseek::precision::fp32_linear_reference;
/// let mut output = [0.0; 4];
/// fp32_linear_reference(
///     &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0],
///     2, 2, 2, &mut output,
/// )?;
/// assert_eq!(output, [17.0, 23.0, 39.0, 53.0]);
/// # Ok::<(), deepseek::precision::Fp32LinearError>(())
/// ```
pub fn fp32_linear_reference(
    activations: &[f32],
    weights: &[f32],
    rows: usize,
    reduction: usize,
    outputs: usize,
    output: &mut [f32],
) -> Result<(), Fp32LinearError> {
    let shape = Shape::new(rows, reduction, outputs)?;
    shape.validate(activations, weights, output)?;
    for (element, &activation) in activations.iter().enumerate() {
        if !activation.is_finite() {
            return Err(Fp32LinearError::NonFiniteActivation { element });
        }
    }
    for (element, &weight) in weights.iter().enumerate() {
        if !weight.is_finite() {
            return Err(Fp32LinearError::NonFiniteWeight { element });
        }
    }

    let mut result = Vec::new();
    result
        .try_reserve_exact(shape.output)
        .map_err(|_| Fp32LinearError::AllocationFailed {
            elements: shape.output,
        })?;
    for row in 0..rows {
        for column in 0..outputs {
            let mut sum = 0.0_f32;
            for reduction_index in 0..reduction {
                let product = activations[row * reduction + reduction_index]
                    * weights[column * reduction + reduction_index];
                if !product.is_finite() {
                    return Err(Fp32LinearError::ValueOverflow {
                        stage: "product",
                        row,
                        output: column,
                    });
                }
                sum += product;
                if !sum.is_finite() {
                    return Err(Fp32LinearError::ValueOverflow {
                        stage: "sum",
                        row,
                        output: column,
                    });
                }
            }
            result.push(sum);
        }
    }
    output.copy_from_slice(&result);
    Ok(())
}

#[derive(Clone, Copy)]
struct Shape {
    activations: usize,
    weights: usize,
    output: usize,
}

impl Shape {
    fn new(rows: usize, reduction: usize, outputs: usize) -> Result<Self, Fp32LinearError> {
        if rows == 0 || reduction == 0 || outputs == 0 {
            return Err(Fp32LinearError::EmptyDimension);
        }
        let activation_elements = checked_product(rows, reduction, "activations")?;
        let weight_elements = checked_product(outputs, reduction, "weights")?;
        let output_elements = checked_product(rows, outputs, "output")?;
        let work = checked_product(output_elements, reduction, "work")?;
        for (field, elements) in [
            ("activations", activation_elements),
            ("weights", weight_elements),
            ("output", output_elements),
        ] {
            if elements > MAX_FP32_LINEAR_ELEMENTS {
                return Err(Fp32LinearError::ElementLimit {
                    field,
                    elements,
                    maximum: MAX_FP32_LINEAR_ELEMENTS,
                });
            }
        }
        if work > MAX_FP32_LINEAR_WORK {
            return Err(Fp32LinearError::WorkLimit {
                elements: work,
                maximum: MAX_FP32_LINEAR_WORK,
            });
        }
        Ok(Self {
            activations: activation_elements,
            weights: weight_elements,
            output: output_elements,
        })
    }

    fn validate(
        self,
        activations: &[f32],
        weights: &[f32],
        output: &[f32],
    ) -> Result<(), Fp32LinearError> {
        for (field, actual, expected) in [
            ("activations", activations.len(), self.activations),
            ("weights", weights.len(), self.weights),
            ("output", output.len(), self.output),
        ] {
            if actual != expected {
                return Err(Fp32LinearError::Length {
                    field,
                    actual,
                    expected,
                });
            }
        }
        Ok(())
    }
}

fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, Fp32LinearError> {
    left.checked_mul(right)
        .ok_or(Fp32LinearError::ShapeOverflow { field })
}

#[cfg(test)]
mod tests {
    use super::{Fp32LinearError, MAX_FP32_LINEAR_ELEMENTS, fp32_linear_reference};

    #[test]
    fn calculates_non_square_row_and_output_orientation() {
        let mut output = [0.0_f32; 4];
        fp32_linear_reference(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[1.0, 0.0, -1.0, 2.0, 1.0, 0.0],
            2,
            3,
            2,
            &mut output,
        )
        .expect("finite scalar reference");
        assert_eq!(
            output.map(f32::to_bits),
            [-2.0_f32, 4.0, -2.0, 13.0].map(f32::to_bits)
        );
    }

    #[test]
    fn preserves_fp32_precision_without_bf16_narrowing() {
        let mut output = [0.0_f32];
        fp32_linear_reference(&[1.0, 1.0 / 256.0], &[1.0, 1.0], 1, 2, 1, &mut output)
            .expect("finite scalar reference");
        assert_eq!(output[0].to_bits(), (1.0_f32 + 1.0 / 256.0).to_bits());
    }

    #[test]
    fn accumulates_in_ascending_reduction_order() {
        let mut output = [1.0_f32];
        fp32_linear_reference(
            &[1.0e20, -1.0e20, 1.0],
            &[1.0, 1.0, 1.0],
            1,
            3,
            1,
            &mut output,
        )
        .expect("finite scalar reference");
        // Ascending order cancels first, then retains 1; reverse order loses it.
        assert_eq!(output.map(f32::to_bits), [1.0_f32.to_bits()]);
    }

    #[test]
    fn rejects_empty_lengths_limits_work_and_overflow_without_allocation() {
        let mut output = [123.0_f32];
        assert_eq!(
            fp32_linear_reference(&[], &[], 0, 1, 1, &mut output),
            Err(Fp32LinearError::EmptyDimension)
        );
        assert!(matches!(
            fp32_linear_reference(&[], &[], 1, 1, 1, &mut output),
            Err(Fp32LinearError::Length {
                field: "activations",
                ..
            })
        ));
        assert!(matches!(
            fp32_linear_reference(&[], &[], MAX_FP32_LINEAR_ELEMENTS + 1, 1, 1, &mut output),
            Err(Fp32LinearError::ElementLimit {
                field: "activations",
                ..
            })
        ));
        assert!(matches!(
            fp32_linear_reference(&[], &[], 512, 32, 1_025, &mut output),
            Err(Fp32LinearError::WorkLimit { .. })
        ));
        assert!(matches!(
            fp32_linear_reference(&[], &[], usize::MAX, 2, 1, &mut output),
            Err(Fp32LinearError::ShapeOverflow {
                field: "activations"
            })
        ));
        assert_eq!(output.map(f32::to_bits), [123.0_f32.to_bits()]);
    }

    #[test]
    fn rejects_nonfinite_inputs_and_late_overflow_without_touching_output() {
        let mut output = [123.0_f32];
        assert_eq!(
            fp32_linear_reference(&[f32::NAN], &[1.0], 1, 1, 1, &mut output),
            Err(Fp32LinearError::NonFiniteActivation { element: 0 })
        );
        assert_eq!(
            fp32_linear_reference(&[1.0], &[f32::INFINITY], 1, 1, 1, &mut output),
            Err(Fp32LinearError::NonFiniteWeight { element: 0 })
        );
        assert_eq!(
            fp32_linear_reference(&[f32::MAX], &[f32::MAX], 1, 1, 1, &mut output),
            Err(Fp32LinearError::ValueOverflow {
                stage: "product",
                row: 0,
                output: 0,
            })
        );
        assert_eq!(
            fp32_linear_reference(&[f32::MAX, f32::MAX], &[1.0, 1.0], 1, 2, 1, &mut output,),
            Err(Fp32LinearError::ValueOverflow {
                stage: "sum",
                row: 0,
                output: 0,
            })
        );
        assert_eq!(output.map(f32::to_bits), [123.0_f32.to_bits()]);
        let mut two_columns = [123.0_f32, 456.0];
        assert_eq!(
            fp32_linear_reference(&[2.0], &[1.0, f32::MAX], 1, 1, 2, &mut two_columns),
            Err(Fp32LinearError::ValueOverflow {
                stage: "product",
                row: 0,
                output: 1
            })
        );
        assert_eq!(
            two_columns.map(f32::to_bits),
            [123.0_f32, 456.0].map(f32::to_bits)
        );
    }
}
