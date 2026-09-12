//! Bounded scalar BF16 linear reference.
//!
//! This establishes a portable staging contract for BF16 matrices: inputs and
//! weights widen to FP32, products and the reduction accumulate in scalar
//! FP32 order, then every result rounds to BF16. It is not a CUDA, Metal, or
//! `PyTorch` GEMM reduction-order oracle.

use thiserror::Error;

/// Largest one-buffer BF16 linear input, weight, or output accepted here.
pub const MAX_BF16_LINEAR_ELEMENTS: usize = 1 << 20;
const MAX_BF16_LINEAR_WORK: usize = 1 << 24;

/// An invalid bounded scalar BF16 linear request.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum Bf16LinearError {
    /// Rows, reduction width, and output width must all be nonzero.
    #[error("BF16 linear dimensions must all be nonzero")]
    EmptyDimension,
    /// A derived buffer or work count overflowed `usize`.
    #[error("BF16 linear shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    /// One bounded buffer exceeds this scalar reference's fixed limit.
    #[error("BF16 linear {field} has {elements} elements, maximum is {maximum}")]
    ElementLimit {
        field: &'static str,
        elements: usize,
        maximum: usize,
    },
    /// The scalar multiply-accumulate count exceeds this reference's fixed limit.
    #[error("BF16 linear work estimate {elements} exceeds maximum {maximum}")]
    WorkLimit { elements: usize, maximum: usize },
    /// A bounded result vector could not be reserved.
    #[error("could not allocate {elements} BF16 linear result elements")]
    AllocationFailed { elements: usize },
    /// A direct-runtime buffer has an unexpected exact length.
    #[error("BF16 linear {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// A BF16 activation denotes NaN or infinity.
    #[error("nonfinite BF16 activation at element {element}")]
    NonFiniteActivation { element: usize },
    /// A BF16 weight denotes NaN or infinity.
    #[error("nonfinite BF16 weight at element {element}")]
    NonFiniteWeight { element: usize },
    /// A scalar FP32 intermediate or the final BF16 narrowing was nonfinite.
    #[error("BF16 linear overflowed at {stage}, row {row}, output {output}")]
    ValueOverflow {
        stage: &'static str,
        row: usize,
        output: usize,
    },
}

/// Multiplies BF16 activations `[rows, reduction]` by BF16 weights
/// `[outputs, reduction]` and writes BF16 `[rows, outputs]`.
///
/// Input and weight storage widen exactly to FP32. Each product and scalar
/// accumulation is FP32, then the completed row/output value is rounded to
/// nearest-even BF16. All validation and all result calculation finish before
/// `output` changes, so every error leaves it untouched.
///
/// This is a precision staging reference used by V4.1 composition tests; it
/// does not qualify hardware GEMM reduction order or throughput.
pub fn bf16_linear_reference(
    activations: &[u16],
    weights: &[u16],
    rows: usize,
    reduction: usize,
    outputs: usize,
    output: &mut [u16],
) -> Result<(), Bf16LinearError> {
    let shape = Shape::new(rows, reduction, outputs)?;
    shape.validate(activations, weights, output)?;
    for (element, &bits) in activations.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(Bf16LinearError::NonFiniteActivation { element });
        }
    }
    for (element, &bits) in weights.iter().enumerate() {
        if !bf16_to_f32(bits).is_finite() {
            return Err(Bf16LinearError::NonFiniteWeight { element });
        }
    }

    let mut result = Vec::new();
    result
        .try_reserve_exact(shape.output)
        .map_err(|_| Bf16LinearError::AllocationFailed {
            elements: shape.output,
        })?;
    for row in 0..rows {
        for column in 0..outputs {
            let mut sum = 0.0_f32;
            for reduction_index in 0..reduction {
                let product = bf16_to_f32(activations[row * reduction + reduction_index])
                    * bf16_to_f32(weights[column * reduction + reduction_index]);
                if !product.is_finite() {
                    return Err(Bf16LinearError::ValueOverflow {
                        stage: "product",
                        row,
                        output: column,
                    });
                }
                sum += product;
                if !sum.is_finite() {
                    return Err(Bf16LinearError::ValueOverflow {
                        stage: "sum",
                        row,
                        output: column,
                    });
                }
            }
            let bits = f32_to_bf16_rne(sum);
            if !bf16_to_f32(bits).is_finite() {
                return Err(Bf16LinearError::ValueOverflow {
                    stage: "BF16 output",
                    row,
                    output: column,
                });
            }
            result.push(bits);
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
    fn new(rows: usize, reduction: usize, outputs: usize) -> Result<Self, Bf16LinearError> {
        if rows == 0 || reduction == 0 || outputs == 0 {
            return Err(Bf16LinearError::EmptyDimension);
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
            if elements > MAX_BF16_LINEAR_ELEMENTS {
                return Err(Bf16LinearError::ElementLimit {
                    field,
                    elements,
                    maximum: MAX_BF16_LINEAR_ELEMENTS,
                });
            }
        }
        if work > MAX_BF16_LINEAR_WORK {
            return Err(Bf16LinearError::WorkLimit {
                elements: work,
                maximum: MAX_BF16_LINEAR_WORK,
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
        activations: &[u16],
        weights: &[u16],
        output: &[u16],
    ) -> Result<(), Bf16LinearError> {
        for (field, actual, expected) in [
            ("activations", activations.len(), self.activations),
            ("weights", weights.len(), self.weights),
            ("output", output.len(), self.output),
        ] {
            if actual != expected {
                return Err(Bf16LinearError::Length {
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
) -> Result<usize, Bf16LinearError> {
    left.checked_mul(right)
        .ok_or(Bf16LinearError::ShapeOverflow { field })
}

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(crate) fn f32_to_bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    u16::try_from(rounded >> 16).expect("an FP32 high half always fits BF16 storage")
}

#[cfg(test)]
mod tests {
    use super::{Bf16LinearError, bf16_linear_reference};

    #[test]
    fn stages_fp32_accumulation_then_one_bf16_rounding() {
        let mut output = [0_u16; 2];
        bf16_linear_reference(
            &[0x3f80, 0x4000, 0xc040, 0x3f80],
            &[0x3f80, 0x3f80],
            2,
            2,
            1,
            &mut output,
        )
        .expect("finite scalar staging");
        assert_eq!(output, [0x4040, 0xc000]); // 3, -2
    }

    #[test]
    fn rejects_late_nonfinite_weight_without_touching_output() {
        let mut output = [0xdead_u16];
        assert_eq!(
            bf16_linear_reference(&[0x3f80], &[0x7f80], 1, 1, 1, &mut output),
            Err(Bf16LinearError::NonFiniteWeight { element: 0 })
        );
        assert_eq!(output, [0xdead]);
    }

    #[test]
    fn rejects_shape_work_values_and_lengths_without_touching_output() {
        let mut output = [0xdead_u16];
        assert!(matches!(
            bf16_linear_reference(&[], &[], usize::MAX, 2, 1, &mut output),
            Err(Bf16LinearError::ShapeOverflow {
                field: "activations"
            })
        ));
        assert!(matches!(
            bf16_linear_reference(&[], &[], 512, 32, 1_025, &mut output),
            Err(Bf16LinearError::WorkLimit { .. })
        ));
        assert!(matches!(
            bf16_linear_reference(&[0x7fc0], &[0x3f80], 1, 1, 1, &mut output),
            Err(Bf16LinearError::NonFiniteActivation { element: 0 })
        ));
        assert!(matches!(
            bf16_linear_reference(&[], &[0x3f80], 1, 1, 1, &mut output),
            Err(Bf16LinearError::Length {
                field: "activations",
                ..
            })
        ));
        assert_eq!(output, [0xdead]);
    }

    #[test]
    fn distinguishes_product_and_final_bf16_overflow() {
        let mut output = [0xdead_u16];
        assert_eq!(
            bf16_linear_reference(&[0x7f7f], &[0x7f7f], 1, 1, 1, &mut output),
            Err(Bf16LinearError::ValueOverflow {
                stage: "product",
                row: 0,
                output: 0,
            })
        );
        // max BF16 plus 2^119 is finite FP32 but exactly half-way from the
        // finite BF16 maximum to infinity; ties-to-even selects infinity.
        assert_eq!(
            bf16_linear_reference(&[0x7f7f, 0x7b00], &[0x3f80, 0x3f80], 1, 2, 1, &mut output,),
            Err(Bf16LinearError::ValueOverflow {
                stage: "BF16 output",
                row: 0,
                output: 0,
            })
        );
        assert_eq!(output, [0xdead]);
    }
}
