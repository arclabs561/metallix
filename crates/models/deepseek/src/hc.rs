//! Bounded scalar reference for the V4.1 Hyper-Connection coefficient split.
//!
//! This follows the pinned `hc_split_sinkhorn_kernel` affine split, stable row
//! softmax, and finite Sinkhorn normalizations in ordered scalar FP32. It is
//! not a GPU-kernel or reduction-bit parity claim and does not project mixes.

use thiserror::Error;

pub mod mixing;

/// One token's Hyper-Connection coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct HcCoefficients {
    copies: usize,
    pre: Vec<f32>,
    post: Vec<f32>,
    comb: Vec<f32>,
}

impl HcCoefficients {
    /// Returns the Hyper-Connection copy count.
    #[must_use]
    pub const fn copies(&self) -> usize {
        self.copies
    }

    /// Returns the pre-collapse coefficient for each copy.
    #[must_use]
    pub fn pre(&self) -> &[f32] {
        &self.pre
    }

    /// Returns the post-expansion coefficient for each copy.
    #[must_use]
    pub fn post(&self) -> &[f32] {
        &self.post
    }

    /// Returns row-major combination coefficients indexed `[source_copy, destination_copy]`.
    #[must_use]
    pub fn comb(&self) -> &[f32] {
        &self.comb
    }

    /// Returns one combination coefficient indexed `[source_copy, destination_copy]`.
    #[must_use]
    pub fn comb_at(&self, source_copy: usize, destination_copy: usize) -> Option<f32> {
        if source_copy >= self.copies || destination_copy >= self.copies {
            return None;
        }
        source_copy
            .checked_mul(self.copies)
            .and_then(|offset| offset.checked_add(destination_copy))
            .and_then(|index| self.comb.get(index))
            .copied()
    }
}

/// An invalid Hyper-Connection coefficient split request or scalar result.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum HcError {
    /// The copy count is outside the bounded scalar-reference range.
    #[error("Hyper-Connection copy count {copies} is outside 1 through {max_copies}")]
    InvalidCopies {
        /// Requested copy count.
        copies: usize,
        /// Largest accepted copy count.
        max_copies: usize,
    },
    /// The Sinkhorn iteration count is outside the bounded scalar-reference range.
    #[error("Hyper-Connection iteration count {iterations} is outside 1 through {max_iterations}")]
    InvalidIterations {
        /// Requested iteration count.
        iterations: usize,
        /// Largest accepted iteration count.
        max_iterations: usize,
    },
    /// The stabilization epsilon must be finite and positive.
    #[error("Hyper-Connection epsilon must be finite and positive")]
    InvalidEpsilon,
    /// A supplied input buffer has an unexpected exact length.
    #[error("Hyper-Connection {field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Required item count.
        expected: usize,
        /// Actual item count.
        actual: usize,
    },
    /// Checked shape arithmetic overflowed.
    #[error("Hyper-Connection shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// Derived buffer role.
        field: &'static str,
    },
    /// A supplied scalar was NaN or infinite.
    #[error("Hyper-Connection {field} at index {index} is non-finite")]
    NonFiniteInput {
        /// Input buffer role.
        field: &'static str,
        /// Flat item index.
        index: usize,
    },
    /// An affine, sum, denominator, or coefficient could not remain finite.
    #[error("Hyper-Connection scalar overflowed at {stage}, index {index}")]
    ValueOverflow {
        /// Named scalar stage.
        stage: &'static str,
        /// Flat copy or matrix index associated with the value.
        index: usize,
    },
}

/// Splits one scalar V4.1 Hyper-Connection mix row into pre, post, and comb coefficients.
///
/// `mixes` and `base` are exact length `(2 + copies) * copies`; `scale` holds
/// pre, post, and comb affine scales. `comb` initially applies row-wise stable
/// softmax plus `epsilon`, then column normalization, followed by alternating
/// row and column normalization for the remaining iterations. Stable sigmoid
/// is mathematically equivalent to the pinned sigmoid but does not claim GPU
/// instruction-level parity. Epsilon and finite iterations intentionally do
/// not promise an exactly doubly-stochastic `comb` matrix.
///
/// # Errors
///
/// Returns [`HcError`] when shape, finite-input, affine, reduction, or output
/// invariants fail before a coefficient object is returned.
pub fn split_hc_coefficients(
    mixes: &[f32],
    scale: &[f32; 3],
    base: &[f32],
    copies: usize,
    sinkhorn_iterations: usize,
    epsilon: f32,
) -> Result<HcCoefficients, HcError> {
    let shape = Shape::validate(mixes, scale, base, copies, sinkhorn_iterations, epsilon)?;

    let mut pre = Vec::with_capacity(copies);
    let mut post = Vec::with_capacity(copies);
    for copy in 0..copies {
        let pre_affine = affine(mixes[copy], scale[0], base[copy], "pre_affine", copy)?;
        let value = sigmoid(pre_affine) + epsilon;
        finite(value, "pre", copy)?;
        pre.push(value);

        let post_index = copy + copies;
        let affine = affine(
            mixes[post_index],
            scale[1],
            base[post_index],
            "post_affine",
            copy,
        )?;
        let value = 2.0 * sigmoid(affine);
        finite(value, "post", copy)?;
        post.push(value);
    }

    let comb_offset = copies.checked_mul(2).ok_or(HcError::ShapeOverflow {
        field: "comb_offset",
    })?;
    let mut comb = Vec::with_capacity(shape.comb_elements);
    for index in 0..shape.comb_elements {
        comb.push(affine(
            mixes[comb_offset + index],
            scale[2],
            base[comb_offset + index],
            "comb_affine",
            index,
        )?);
    }
    row_softmax_plus_epsilon(&mut comb, copies, epsilon)?;
    normalize_columns(&mut comb, copies, epsilon)?;
    for _ in 1..sinkhorn_iterations {
        normalize_rows(&mut comb, copies, epsilon)?;
        normalize_columns(&mut comb, copies, epsilon)?;
    }
    for (index, &value) in comb.iter().enumerate() {
        finite(value, "comb", index)?;
    }

    Ok(HcCoefficients {
        copies,
        pre,
        post,
        comb,
    })
}

#[derive(Clone, Copy)]
struct Shape {
    comb_elements: usize,
}

impl Shape {
    fn validate(
        mixes: &[f32],
        scale: &[f32; 3],
        base: &[f32],
        copies: usize,
        sinkhorn_iterations: usize,
        epsilon: f32,
    ) -> Result<Self, HcError> {
        const MAX_COPIES: usize = 16;
        const MAX_ITERATIONS: usize = 64;
        if copies == 0 || copies > MAX_COPIES {
            return Err(HcError::InvalidCopies {
                copies,
                max_copies: MAX_COPIES,
            });
        }
        if sinkhorn_iterations == 0 || sinkhorn_iterations > MAX_ITERATIONS {
            return Err(HcError::InvalidIterations {
                iterations: sinkhorn_iterations,
                max_iterations: MAX_ITERATIONS,
            });
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(HcError::InvalidEpsilon);
        }
        let prefixed = copies
            .checked_add(2)
            .ok_or(HcError::ShapeOverflow { field: "mixes" })?;
        let expected = prefixed
            .checked_mul(copies)
            .ok_or(HcError::ShapeOverflow { field: "mixes" })?;
        let comb_elements = copies
            .checked_mul(copies)
            .ok_or(HcError::ShapeOverflow { field: "comb" })?;
        check_length("mixes", mixes.len(), expected)?;
        check_length("base", base.len(), expected)?;
        for (index, &value) in scale.iter().enumerate() {
            finite_input("scale", index, value)?;
        }
        for (index, &value) in mixes.iter().enumerate() {
            finite_input("mixes", index, value)?;
        }
        for (index, &value) in base.iter().enumerate() {
            finite_input("base", index, value)?;
        }
        Ok(Self { comb_elements })
    }
}

fn check_length(field: &'static str, actual: usize, expected: usize) -> Result<(), HcError> {
    if actual == expected {
        Ok(())
    } else {
        Err(HcError::Length {
            field,
            expected,
            actual,
        })
    }
}

fn finite_input(field: &'static str, index: usize, value: f32) -> Result<(), HcError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(HcError::NonFiniteInput { field, index })
    }
}

fn finite(value: f32, stage: &'static str, index: usize) -> Result<(), HcError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(HcError::ValueOverflow { stage, index })
    }
}

fn affine(
    mix: f32,
    scale: f32,
    base: f32,
    stage: &'static str,
    index: usize,
) -> Result<f32, HcError> {
    let product = mix * scale;
    finite(product, stage, index)?;
    let result = product + base;
    finite(result, stage, index)?;
    Ok(result)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

fn row_softmax_plus_epsilon(comb: &mut [f32], copies: usize, epsilon: f32) -> Result<(), HcError> {
    for source in 0..copies {
        let row_start = source * copies;
        let row = &mut comb[row_start..row_start + copies];
        let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        finite(maximum, "row_max", source)?;
        let mut sum = 0.0_f32;
        for (destination, value) in row.iter_mut().enumerate() {
            *value = (*value - maximum).exp();
            sum += *value;
            finite(sum, "row_sum", row_start + destination)?;
        }
        for (destination, value) in row.iter_mut().enumerate() {
            *value = *value / sum + epsilon;
            finite(*value, "row_softmax", row_start + destination)?;
        }
    }
    Ok(())
}

fn normalize_rows(comb: &mut [f32], copies: usize, epsilon: f32) -> Result<(), HcError> {
    for source in 0..copies {
        let row_start = source * copies;
        let row = &mut comb[row_start..row_start + copies];
        let mut sum = 0.0_f32;
        for (destination, &value) in row.iter().enumerate() {
            sum += value;
            finite(sum, "row_sum", row_start + destination)?;
        }
        let denominator = sum + epsilon;
        finite(denominator, "row_denominator", source)?;
        for (destination, value) in row.iter_mut().enumerate() {
            *value /= denominator;
            finite(*value, "row", row_start + destination)?;
        }
    }
    Ok(())
}

fn normalize_columns(comb: &mut [f32], copies: usize, epsilon: f32) -> Result<(), HcError> {
    for destination in 0..copies {
        let mut sum = 0.0_f32;
        for source in 0..copies {
            sum += comb[source * copies + destination];
            finite(sum, "column_sum", destination)?;
        }
        let denominator = sum + epsilon;
        finite(denominator, "column_denominator", destination)?;
        for source in 0..copies {
            let index = source * copies + destination;
            comb[index] /= denominator;
            finite(comb[index], "column", index)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{HcError, split_hc_coefficients};

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
    }

    #[test]
    fn all_zero_two_copy_mixes_match_hand_staged_sinkhorn_iterations() {
        let mixes = [0.0; 8];
        let base = [0.0; 8];
        let first = split_hc_coefficients(&mixes, &[0.0; 3], &base, 2, 1, 0.5)
            .expect("finite all-zero HC split");
        assert_eq!(first.pre(), [1.0, 1.0]);
        assert_eq!(first.post(), [1.0, 1.0]);
        for value in first.comb() {
            assert_close(*value, 0.4); // ((1/2 + 1/2) / (2 + 1/2)).
        }

        let second = split_hc_coefficients(&mixes, &[0.0; 3], &base, 2, 2, 0.5)
            .expect("finite second Sinkhorn iteration");
        for value in second.comb() {
            assert_close(*value, 8.0 / 29.0); // (2/5)/(2*(2/5) + 1/2), then column-normalize.
        }
    }

    #[test]
    fn affine_split_and_comb_axes_are_source_then_destination() {
        let coefficients = split_hc_coefficients(
            &[1.0, -1.0, 2.0, -2.0, 0.0, 2.0, -1.0, 1.0],
            &[1.0, 0.5, 2.0],
            &[0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            2,
            1,
            0.1,
        )
        .expect("finite asymmetric HC split");
        assert_close(coefficients.pre()[0], 1.0 / (1.0 + (-1.0_f32).exp()) + 0.1);
        assert_close(coefficients.pre()[1], 0.6);
        assert_close(coefficients.post()[0], 2.0 / (1.0 + (-1.0_f32).exp()));
        assert_close(coefficients.post()[1], 2.0 / (1.0 + 2.0_f32.exp()));
        assert!(
            coefficients.comb_at(0, 1).expect("in bounds")
                > coefficients.comb_at(1, 0).expect("in bounds")
        );
        assert_eq!(coefficients.comb_at(2, 0), None);
        assert_eq!(coefficients.comb_at(0, 2), None);
    }

    #[test]
    fn rejects_malformed_nonfinite_and_overflowing_requests() {
        assert!(matches!(
            split_hc_coefficients(&[], &[0.0; 3], &[], 0, 1, 1.0),
            Err(HcError::InvalidCopies { .. })
        ));
        assert!(matches!(
            split_hc_coefficients(&[0.0; 8], &[0.0; 3], &[0.0; 8], 2, 0, 1.0),
            Err(HcError::InvalidIterations { .. })
        ));
        assert!(matches!(
            split_hc_coefficients(&[0.0; 7], &[0.0; 3], &[0.0; 8], 2, 1, 1.0),
            Err(HcError::Length { field: "mixes", .. })
        ));
        assert!(matches!(
            split_hc_coefficients(&[f32::NAN; 8], &[0.0; 3], &[0.0; 8], 2, 1, 1.0),
            Err(HcError::NonFiniteInput {
                field: "mixes",
                index: 0
            })
        ));
        assert!(matches!(
            split_hc_coefficients(&[f32::MAX; 8], &[2.0; 3], &[0.0; 8], 2, 1, 1.0),
            Err(HcError::ValueOverflow {
                stage: "pre_affine",
                index: 0,
            })
        ));
    }
}
