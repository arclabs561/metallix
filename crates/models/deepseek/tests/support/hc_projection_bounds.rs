//! Test-only arithmetic envelope for normalized Hyper-Connection projections.
//!
//! This is a compatibility check against one declared FP32 target arithmetic
//! model.  It does not claim that Torch `rsqrt`, a BLAS kernel, or another
//! backend universally implements that model.  In particular, callers may
//! check both a captured source mix and a native mix against this envelope,
//! but must not turn a successful check into a Torch accuracy guarantee.

use std::fmt;

const F32_UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;
const F32_HALF_MIN_SUBNORMAL: f64 = f64::from_bits(0x3690_0000_0000_0000);
const MAX_TERMS: usize = 1 << 20;

/// A closed, outward-rounded finite FP64 interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Interval {
    /// Lower endpoint.
    pub(crate) lo: f64,
    /// Upper endpoint.
    pub(crate) hi: f64,
}

impl Interval {
    /// Returns whether a finite observed scalar is contained by this interval.
    pub(crate) fn contains(self, value: f32) -> bool {
        let value = f64::from(value);
        value.is_finite() && self.lo <= value && value <= self.hi
    }
}

/// A target-arithmetic precondition that prevents a relative-error proof.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BoundsError {
    /// The residual row has no elements.
    EmptyResidual,
    /// The projection does not contain a whole, nonempty number of rows.
    ProjectionShape,
    /// The gamma bound would be invalid or too broad for this test helper.
    TooManyTerms,
    /// Epsilon is not finite, positive, and normal in the target FP32 model.
    InvalidEpsilon,
    /// An input or required target intermediate is not finite and normal.
    NonNormal { field: &'static str, index: usize },
    /// A derived FP64 endpoint is not finite.
    NonFiniteEnvelope,
}

impl fmt::Display for BoundsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyResidual => formatter.write_str("normalized HC projection needs a residual"),
            Self::ProjectionShape => {
                formatter.write_str("projection must contain whole nonempty rows")
            }
            Self::TooManyTerms => {
                formatter.write_str("projection term count is outside gamma bound")
            }
            Self::InvalidEpsilon => {
                formatter.write_str("epsilon must be finite, positive, and normal")
            }
            Self::NonNormal { field, index } => {
                write!(
                    formatter,
                    "{field} at index {index} is outside normal FP32 domain"
                )
            }
            Self::NonFiniteEnvelope => {
                formatter.write_str("projection envelope endpoint is non-finite")
            }
        }
    }
}

impl std::error::Error for BoundsError {}

/// Returns one envelope per projection row for the declared target model.
///
/// The target has normal finite BF16-decoded FP32 inputs and FP32 weights; a
/// dot-product absolute error bounded by `gamma(2n) * sum(abs(x*w))` plus a
/// gradual-underflow term; a positive sum-of-squares with the same bound;
/// rounded FP32 divide and add with half-min-subnormal absolute terms;
/// correctly-rounded FP32 square-root followed by reciprocal; and a rounded
/// FP32 final multiply with a half-min-subnormal absolute term. Endpoints are
/// evaluated in FP64 with an extra step outward after each endpoint operation.
///
/// The helper fails closed when relative-error reasoning could be invalid,
/// including subnormal point inputs, out-of-range interval endpoints, or
/// normalization variance.  It deliberately does not take an observed source
/// discrepancy as input and has no source-output-calibrated budget.
pub(crate) fn normalized_projection_envelopes(
    residual_bf16: &[u16],
    projection: &[f32],
    epsilon: f32,
) -> Result<Vec<Interval>, BoundsError> {
    let points: Vec<_> = residual_bf16
        .iter()
        .copied()
        .map(bf16_to_f32)
        .enumerate()
        .map(|(index, value)| {
            normal(value, "residual", index)?;
            Ok(value)
        })
        .collect::<Result<_, BoundsError>>()?;
    validate_point_intermediates(&points, projection)?;
    let residual: Vec<_> = points
        .into_iter()
        .map(|value| {
            let value = f64::from(value);
            [value, value]
        })
        .collect();
    normalized_projection_interval_envelopes(&residual, projection, epsilon)
}

/// Returns target-arithmetic projection envelopes for interval residual input.
///
/// Each `[lo, hi]` is an outward BF16-compatible residual enclosure from a
/// preceding stage.  Projection weights remain exact FP32 checkpoint values.
/// The resulting envelope therefore propagates input uncertainty without
/// calibrating itself from an observed source discrepancy.  Intervals must be
/// finite and lie in the target FP32 normal range at their nonzero endpoints;
/// a sign-spanning interval is supported and squares it to `[0, max(lo²,hi²)]`.
/// The interval form bounds gradual underflow explicitly, but assumes IEEE
/// round-to-nearest gradual-underflow arithmetic, not a flush-to-zero backend.
/// It rejects any row whose conservative FP32 prefix envelope could overflow.
pub(crate) fn normalized_projection_interval_envelopes(
    residual: &[[f64; 2]],
    projection: &[f32],
    epsilon: f32,
) -> Result<Vec<Interval>, BoundsError> {
    let terms = residual.len();
    if terms == 0 {
        return Err(BoundsError::EmptyResidual);
    }
    if terms > MAX_TERMS || projection.is_empty() || !projection.len().is_multiple_of(terms) {
        return Err(BoundsError::ProjectionShape);
    }
    let operations = terms.checked_mul(2).ok_or(BoundsError::TooManyTerms)?;
    let gamma = gamma(operations)?;
    let underflow = underflow_error(operations)?;
    if !epsilon.is_finite() || epsilon <= 0.0 || !epsilon.is_normal() {
        return Err(BoundsError::InvalidEpsilon);
    }

    for (index, &value) in residual.iter().enumerate() {
        normal_interval(value, "residual", index)?;
    }
    let squares = positive_sum_squares(residual, gamma, underflow)?;
    let variance = normalized_variance(squares, terms, f64::from(epsilon), gamma, underflow)?;
    let inverse_rms = inverse_rms_envelope(variance)?;

    projection
        .chunks_exact(terms)
        .enumerate()
        .map(|(row, weights)| {
            dot_times_inverse_rms(residual, weights, inverse_rms, gamma, underflow, row)
        })
        .collect()
}

fn gamma(operations: usize) -> Result<f64, BoundsError> {
    if operations == 0 || operations > MAX_TERMS.saturating_mul(2) {
        return Err(BoundsError::TooManyTerms);
    }
    let product = bounded_usize_f64(operations)? * F32_UNIT_ROUNDOFF;
    if !product.is_finite() || product >= 1.0 {
        return Err(BoundsError::TooManyTerms);
    }
    Ok(product / (1.0 - product))
}

fn underflow_error(operations: usize) -> Result<f64, BoundsError> {
    let operations = bounded_usize_f64(operations)?;
    let nu = operations * F32_UNIT_ROUNDOFF;
    if !nu.is_finite() || nu >= 1.0 {
        return Err(BoundsError::TooManyTerms);
    }
    let error = (operations * F32_HALF_MIN_SUBNORMAL) / (1.0 - nu);
    if error.is_finite() {
        Ok(up(error))
    } else {
        Err(BoundsError::NonFiniteEnvelope)
    }
}

fn positive_sum_squares(
    residual: &[[f64; 2]],
    gamma: f64,
    underflow: f64,
) -> Result<Interval, BoundsError> {
    let mut sum = Interval { lo: 0.0, hi: 0.0 };
    for (index, &[lo, hi]) in residual.iter().enumerate() {
        let square = if lo <= 0.0 && hi >= 0.0 {
            Interval {
                lo: 0.0,
                hi: up((lo * lo).max(hi * hi)),
            }
        } else {
            multiply(Interval { lo, hi }, Interval { lo, hi })?
        };
        if !square.hi.is_finite() || square.hi > f64::from(f32::MAX) {
            return Err(BoundsError::NonNormal {
                field: "square",
                index,
            });
        }
        sum = add_nonnegative(sum, square);
        let rounded_prefix = gamma_nonnegative(sum, gamma, underflow)?;
        if rounded_prefix.hi > f64::from(f32::MAX) {
            return Err(BoundsError::NonNormal {
                field: "square_sum",
                index,
            });
        }
    }
    Ok(sum)
}

fn normalized_variance(
    squares: Interval,
    terms: usize,
    epsilon: f64,
    gamma: f64,
    underflow: f64,
) -> Result<Interval, BoundsError> {
    let summed = gamma_nonnegative(squares, gamma, underflow)?;
    let divided = rounded_nonnegative(divide_positive(summed, bounded_usize_f64(terms)?)?);
    let variance = rounded_nonnegative(add_nonnegative(divided, Interval::point(epsilon)?));
    if variance.lo <= f64::from(f32::MIN_POSITIVE)
        || variance.hi > f64::from(f32::MAX)
        || !variance.lo.is_finite()
        || !variance.hi.is_finite()
    {
        return Err(BoundsError::NonNormal {
            field: "variance",
            index: 0,
        });
    }
    Ok(variance)
}

fn inverse_rms_envelope(variance: Interval) -> Result<Interval, BoundsError> {
    // `rsqrt` is modeled only as correctly-rounded sqrt followed by correctly-
    // rounded reciprocal.  A backend with a distinct rsqrt approximation must
    // supply a separate validated envelope before using this one.
    let ideal = Interval {
        lo: down(1.0 / variance.hi.sqrt()),
        hi: up(1.0 / variance.lo.sqrt()),
    };
    // If sqrt has relative error +/-u, its reciprocal contributes reciprocal
    // factors 1/(1+u) and 1/(1-u).  The following reciprocal rounding then
    // contributes (1-u) and (1+u), respectively.
    finite_interval(Interval {
        lo: down(ideal.lo * (1.0 - F32_UNIT_ROUNDOFF) / (1.0 + F32_UNIT_ROUNDOFF)),
        hi: up(ideal.hi * (1.0 + F32_UNIT_ROUNDOFF) / (1.0 - F32_UNIT_ROUNDOFF)),
    })
}

fn dot_times_inverse_rms(
    residual: &[[f64; 2]],
    weights: &[f32],
    inverse_rms: Interval,
    gamma: f64,
    underflow: f64,
    row: usize,
) -> Result<Interval, BoundsError> {
    let mut dot = Interval { lo: 0.0, hi: 0.0 };
    let mut magnitude = Interval { lo: 0.0, hi: 0.0 };
    for (column, (&value, &weight)) in residual.iter().zip(weights).enumerate() {
        normal(weight, "projection", row * residual.len() + column)?;
        let product_interval = multiply(
            Interval {
                lo: value[0],
                hi: value[1],
            },
            Interval::point(f64::from(weight))?,
        )?;
        if abs_upper(product_interval) > f64::from(f32::MAX) {
            return Err(BoundsError::NonNormal {
                field: "dot_product",
                index: row * residual.len() + column,
            });
        }
        dot = add(dot, product_interval);
        magnitude = add_nonnegative(magnitude, Interval::point(abs_upper(product_interval))?);
        let rounded_prefix = gamma_signed(dot, magnitude.hi, gamma, underflow)?;
        if rounded_prefix.lo < -f64::from(f32::MAX) || rounded_prefix.hi > f64::from(f32::MAX) {
            return Err(BoundsError::NonNormal {
                field: "dot_sum",
                index: row * residual.len() + column,
            });
        }
    }
    // A different legal reduction tree can group all same-sign terms before
    // cancellation.  Bounding the rounded total absolute magnitude therefore
    // bounds every subset, not merely this left-to-right diagnostic prefix.
    if gamma_nonnegative(magnitude, gamma, underflow)?.hi > f64::from(f32::MAX) {
        return Err(BoundsError::NonNormal {
            field: "dot_magnitude",
            index: row,
        });
    }
    let dot = gamma_signed(dot, magnitude.hi, gamma, underflow)?;
    let result = multiply(dot, inverse_rms)?;
    // One final normal FP32 multiplication rounding.
    let result = rounded_signed(result);
    finite_interval(result)
}

fn validate_point_intermediates(residual: &[f32], projection: &[f32]) -> Result<(), BoundsError> {
    let mut square_sum = 0.0_f32;
    for (index, &value) in residual.iter().enumerate() {
        let square = value * value;
        normal(square, "square", index)?;
        square_sum += square;
        if !square_sum.is_normal() {
            return Err(BoundsError::NonNormal {
                field: "square_sum",
                index,
            });
        }
    }
    for (row, weights) in projection.chunks_exact(residual.len()).enumerate() {
        let mut dot_sum = 0.0_f32;
        for (column, (&value, &weight)) in residual.iter().zip(weights).enumerate() {
            normal(weight, "projection", row * residual.len() + column)?;
            let product = value * weight;
            normal(product, "dot_product", row * residual.len() + column)?;
            dot_sum += product;
            if !(dot_sum.is_normal() || dot_sum == 0.0) {
                return Err(BoundsError::NonNormal {
                    field: "dot_sum",
                    index: row * residual.len() + column,
                });
            }
        }
    }
    Ok(())
}

fn gamma_nonnegative(value: Interval, gamma: f64, underflow: f64) -> Result<Interval, BoundsError> {
    if value.lo < 0.0 || !gamma.is_finite() || !underflow.is_finite() {
        return Err(BoundsError::NonFiniteEnvelope);
    }
    let error = up(gamma * value.hi + underflow);
    finite_interval(Interval {
        lo: down(value.lo - error).max(0.0),
        hi: up(value.hi + error),
    })
}

fn gamma_signed(
    value: Interval,
    magnitude: f64,
    gamma: f64,
    underflow: f64,
) -> Result<Interval, BoundsError> {
    if magnitude < 0.0 || !magnitude.is_finite() || !gamma.is_finite() || !underflow.is_finite() {
        return Err(BoundsError::NonFiniteEnvelope);
    }
    let error = up(gamma * magnitude + underflow);
    finite_interval(Interval {
        lo: down(value.lo - error),
        hi: up(value.hi + error),
    })
}

fn rounded_nonnegative(value: Interval) -> Interval {
    let error = up(F32_UNIT_ROUNDOFF * value.hi + F32_HALF_MIN_SUBNORMAL);
    Interval {
        lo: down(value.lo - error).max(0.0),
        hi: up(value.hi + error),
    }
}

fn rounded_signed(value: Interval) -> Interval {
    let error = up(F32_UNIT_ROUNDOFF * abs_upper(value) + F32_HALF_MIN_SUBNORMAL);
    Interval {
        lo: down(value.lo - error),
        hi: up(value.hi + error),
    }
}

fn divide_positive(value: Interval, divisor: f64) -> Result<Interval, BoundsError> {
    if value.lo < 0.0 || divisor <= 0.0 || !divisor.is_finite() {
        return Err(BoundsError::NonFiniteEnvelope);
    }
    finite_interval(Interval {
        lo: down(value.lo / divisor).max(0.0),
        hi: up(value.hi / divisor),
    })
}

fn add(left: Interval, right: Interval) -> Interval {
    Interval {
        lo: down(left.lo + right.lo),
        hi: up(left.hi + right.hi),
    }
}

fn add_nonnegative(left: Interval, right: Interval) -> Interval {
    debug_assert!(left.lo >= 0.0 && right.lo >= 0.0);
    Interval {
        lo: down(left.lo + right.lo).max(0.0),
        hi: up(left.hi + right.hi),
    }
}

fn multiply(left: Interval, right: Interval) -> Result<Interval, BoundsError> {
    let products = [
        left.lo * right.lo,
        left.lo * right.hi,
        left.hi * right.lo,
        left.hi * right.hi,
    ];
    if products.iter().any(|value| !value.is_finite()) {
        return Err(BoundsError::NonFiniteEnvelope);
    }
    let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    finite_interval(Interval {
        lo: down(lo),
        hi: up(hi),
    })
}

impl Interval {
    fn point(value: f64) -> Result<Self, BoundsError> {
        finite_interval(Self {
            lo: down(value),
            hi: up(value),
        })
    }
}

fn finite_interval(interval: Interval) -> Result<Interval, BoundsError> {
    if interval.lo.is_finite() && interval.hi.is_finite() && interval.lo <= interval.hi {
        Ok(interval)
    } else {
        Err(BoundsError::NonFiniteEnvelope)
    }
}

fn normal(value: f32, field: &'static str, index: usize) -> Result<(), BoundsError> {
    if value.is_normal() {
        Ok(())
    } else {
        Err(BoundsError::NonNormal { field, index })
    }
}

fn normal_interval(value: [f64; 2], field: &'static str, index: usize) -> Result<(), BoundsError> {
    let [lo, hi] = value;
    if !lo.is_finite() || !hi.is_finite() || lo > hi {
        return Err(BoundsError::NonNormal { field, index });
    }
    for endpoint in [lo, hi] {
        if endpoint != 0.0
            && (endpoint.abs() < f64::from(f32::MIN_POSITIVE)
                || endpoint.abs() > f64::from(f32::MAX))
        {
            return Err(BoundsError::NonNormal { field, index });
        }
    }
    Ok(())
}

fn abs_upper(value: Interval) -> f64 {
    value.lo.abs().max(value.hi.abs())
}

fn bounded_usize_f64(value: usize) -> Result<f64, BoundsError> {
    u32::try_from(value)
        .map(f64::from)
        .map_err(|_| BoundsError::TooManyTerms)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn down(value: f64) -> f64 {
    value.next_down()
}

fn up(value: f64) -> f64 {
    value.next_up()
}

#[cfg(test)]
mod tests {
    use super::{
        BoundsError, normalized_projection_envelopes, normalized_projection_interval_envelopes,
    };

    #[test]
    fn one_term_target_mix_is_included() {
        let envelopes = normalized_projection_envelopes(&[0x3f80], &[2.0], 1.0).unwrap();
        let target = 2.0_f32 * ((1.0_f32 + 1.0_f32).sqrt().recip());
        assert!(envelopes[0].contains(target));
    }

    #[test]
    fn cancellation_remains_inside_an_absolute_dot_envelope() {
        let envelopes =
            normalized_projection_envelopes(&[0x3f80, 0xbf80], &[1.0, 1.0], 1.0).unwrap();
        assert!(envelopes[0].contains(0.0));
    }

    #[test]
    fn sign_spanning_interval_squares_to_zero_lower_bound() {
        let envelopes =
            normalized_projection_interval_envelopes(&[[-1.0, 1.0]], &[1.0], 1.0).unwrap();
        assert!(envelopes[0].contains(0.0));
    }

    #[test]
    fn tiny_normal_sign_spanning_interval_covers_zero_after_square_underflow() {
        let tiny = f64::from(f32::MIN_POSITIVE);
        let envelopes =
            normalized_projection_interval_envelopes(&[[-tiny, tiny]], &[1.0], 1.0).unwrap();
        assert!(envelopes[0].contains(0.0));
    }

    #[test]
    fn rejects_square_prefix_overflow_before_mean_can_restore_finiteness() {
        let residual = [[1.0e19, 1.0e19]; 4];
        assert!(matches!(
            normalized_projection_interval_envelopes(&residual, &[1.0; 4], 1.0),
            Err(BoundsError::NonNormal {
                field: "square_sum",
                ..
            })
        ));
    }

    #[test]
    fn rejects_dot_prefix_overflow_before_later_cancellation() {
        let residual = [[1.0, 1.0]; 8];
        let weights = [
            1.0e38, 1.0e38, 1.0e38, 1.0e38, -1.0e38, -1.0e38, -1.0e38, -1.0e38,
        ];
        assert!(matches!(
            normalized_projection_interval_envelopes(&residual, &weights, 1.0),
            Err(BoundsError::NonNormal {
                field: "dot_sum",
                ..
            })
        ));
    }

    #[test]
    fn rejects_safe_ascending_dot_when_another_reduction_tree_overflows() {
        let residual = [[1.0, 1.0]; 8];
        let weights = [
            1.0e38, -1.0e38, 1.0e38, -1.0e38, 1.0e38, -1.0e38, 1.0e38, -1.0e38,
        ];
        assert!(matches!(
            normalized_projection_interval_envelopes(&residual, &weights, 1.0),
            Err(BoundsError::NonNormal {
                field: "dot_magnitude",
                ..
            })
        ));
    }

    #[test]
    fn value_beyond_outward_endpoint_is_rejected() {
        let envelope = normalized_projection_envelopes(&[0x3f80], &[2.0], 1.0).unwrap()[0];
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the test deliberately chooses the next representable FP32 beyond an FP64 endpoint"
        )]
        let outside = (envelope.hi as f32).next_up();
        assert!(!envelope.contains(outside));
    }

    #[test]
    fn subnormal_and_invalid_inputs_fail_closed() {
        assert!(matches!(
            normalized_projection_envelopes(&[1], &[1.0], 1.0),
            Err(BoundsError::NonNormal {
                field: "residual",
                ..
            })
        ));
        assert!(matches!(
            normalized_projection_envelopes(&[0x3f80], &[f32::from_bits(1)], 1.0),
            Err(BoundsError::NonNormal {
                field: "projection",
                ..
            })
        ));
        assert!(matches!(
            normalized_projection_envelopes(&[0x3f80], &[1.0], 0.0),
            Err(BoundsError::InvalidEpsilon)
        ));
    }

    #[test]
    fn point_subnormal_products_and_squares_fail_closed() {
        assert!(matches!(
            normalized_projection_envelopes(&[0x0080], &[1.0], 1.0),
            Err(BoundsError::NonNormal {
                field: "square",
                ..
            })
        ));
        assert!(matches!(
            normalized_projection_envelopes(&[0x3f00], &[f32::MIN_POSITIVE], 1.0),
            Err(BoundsError::NonNormal {
                field: "dot_product",
                ..
            })
        ));
    }
}
