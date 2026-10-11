//! Bounded scalar BF16 linear reference.
//!
//! This establishes a portable staging contract for BF16 matrices: inputs and
//! weights widen to FP32, products and the reduction accumulate in scalar
//! FP32 order, then every result rounds to BF16. It is not a CUDA, Metal, or
//! `PyTorch` GEMM reduction-order oracle.

use thiserror::Error;

/// Largest one-buffer BF16 linear input, weight, or output accepted here.
/// Admits one real V4.1 `wo_a` group (1024 x 4096 BF16 weights).
pub const MAX_BF16_LINEAR_ELEMENTS: usize = 1 << 23;
const MAX_BF16_LINEAR_WORK: usize = 1 << 30;

/// An invalid bounded scalar BF16 linear request.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum Bf16LinearError {
    /// Rows, reduction width, and output width must all be nonzero.
    #[error("BF16 linear dimensions must all be nonzero")]
    EmptyDimension,
    /// A derived buffer or work count overflowed `usize`.
    #[error("BF16 linear shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// The derived count that overflowed: a buffer name or `"work"`.
        field: &'static str,
    },
    /// One bounded buffer exceeds this scalar reference's fixed limit.
    #[error("BF16 linear {field} has {elements} elements, maximum is {maximum}")]
    ElementLimit {
        /// The bounded buffer name.
        field: &'static str,
        /// The requested element count.
        elements: usize,
        /// The fixed maximum, [`MAX_BF16_LINEAR_ELEMENTS`].
        maximum: usize,
    },
    /// The scalar multiply-accumulate count exceeds this reference's fixed limit.
    #[error("BF16 linear work estimate {elements} exceeds maximum {maximum}")]
    WorkLimit {
        /// The requested multiply-accumulate count, `rows * outputs * reduction`.
        elements: usize,
        /// The fixed maximum work count.
        maximum: usize,
    },
    /// A bounded result vector could not be reserved.
    #[error("could not allocate {elements} BF16 linear result elements")]
    AllocationFailed {
        /// The result element count that could not be reserved.
        elements: usize,
    },
    /// A direct-runtime buffer has an unexpected exact length.
    #[error("BF16 linear {field} length is {actual}, expected {expected}")]
    Length {
        /// The caller buffer name.
        field: &'static str,
        /// The supplied element count.
        actual: usize,
        /// The shape-derived required element count.
        expected: usize,
    },
    /// A BF16 activation denotes NaN or infinity.
    #[error("nonfinite BF16 activation at element {element}")]
    NonFiniteActivation {
        /// Flat index into `activations`.
        element: usize,
    },
    /// A BF16 weight denotes NaN or infinity.
    #[error("nonfinite BF16 weight at element {element}")]
    NonFiniteWeight {
        /// Flat index into `weights`.
        element: usize,
    },
    /// A scalar FP32 intermediate or the final BF16 narrowing was nonfinite.
    #[error("BF16 linear overflowed at {stage}, row {row}, output {output}")]
    ValueOverflow {
        /// `"product"`, `"sum"` or `"BF16 output"`.
        stage: &'static str,
        /// The activation row being calculated.
        row: usize,
        /// The output column being calculated.
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
///
/// # Errors
///
/// `output` is unchanged on every error.
///
/// * [`Bf16LinearError::EmptyDimension`] when any dimension is 0.
/// * [`Bf16LinearError::ShapeOverflow`] when a buffer size or the work count
///   does not fit in `usize`.
/// * [`Bf16LinearError::ElementLimit`] when a buffer exceeds
///   [`MAX_BF16_LINEAR_ELEMENTS`], and [`Bf16LinearError::WorkLimit`] when
///   `rows * outputs * reduction` exceeds 2^30.
/// * [`Bf16LinearError::Length`] when a buffer does not match its shape.
/// * [`Bf16LinearError::NonFiniteActivation`] and
///   [`Bf16LinearError::NonFiniteWeight`] for a NaN or infinite input.
/// * [`Bf16LinearError::AllocationFailed`] when the staged result cannot be
///   reserved.
/// * [`Bf16LinearError::ValueOverflow`] when a product, a running sum or the
///   rounded BF16 result is not finite.
///
/// # Example
///
/// ```
/// use blockfloat::{bf16_linear_reference, bf16_to_f32};
///
/// // [1, 2] dotted with [3, 4] is 11.
/// let mut output = [0_u16; 1];
/// bf16_linear_reference(&[0x3f80, 0x4000], &[0x4040, 0x4080], 1, 2, 1, &mut output)?;
/// assert_eq!(bf16_to_f32(output[0]), 11.0);
/// # Ok::<(), blockfloat::Bf16LinearError>(())
/// ```
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
    for (row, activation) in activations.chunks_exact(reduction).enumerate() {
        for (column, weight) in weights.chunks_exact(reduction).enumerate() {
            let mut sum = 0.0_f32;
            for (&activation, &weight) in activation.iter().zip(weight) {
                let product = bf16_to_f32(activation) * bf16_to_f32(weight);
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

/// Widens BF16 storage bits to FP32 exactly.
///
/// # Example
///
/// ```
/// use blockfloat::bf16_to_f32;
///
/// assert_eq!(bf16_to_f32(0x3f80), 1.0);
/// assert_eq!(bf16_to_f32(0xc000), -2.0);
/// ```
#[must_use]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Rounds FP32 to BF16 storage bits, nearest with ties to even.
///
/// Finite values can round to infinity at the BF16 range boundary; callers
/// that need a finite result check it, as [`bf16_linear_reference`] does.
///
/// # Panics
///
/// Never: the high half of an FP32 word always fits in `u16`.
///
/// # Example
///
/// Values halfway between two BF16 neighbors round to the one with an even
/// last bit.
///
/// ```
/// use blockfloat::f32_to_bf16_rne;
///
/// assert_eq!(f32_to_bf16_rne(1.0), 0x3f80);
/// // 1 + 2^-8 lies halfway between 0x3f80 and 0x3f81: round down to even.
/// assert_eq!(f32_to_bf16_rne(1.0 + 2.0_f32.powi(-8)), 0x3f80);
/// // 1 + 3 * 2^-8 lies halfway between 0x3f81 and 0x3f82: round up to even.
/// assert_eq!(f32_to_bf16_rne(1.0 + 3.0 * 2.0_f32.powi(-8)), 0x3f82);
/// ```
#[must_use]
pub fn f32_to_bf16_rne(value: f32) -> u16 {
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
            bf16_linear_reference(&[], &[], 2_048, 4_096, 129, &mut output),
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
