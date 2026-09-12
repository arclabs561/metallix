//! Bounded scalar qualification of the V4.1 Flash text `MoE` routing rule.
//!
//! This accepts already-projected gate logits for one text token. It mirrors
//! the pinned `Gate.forward` sqrt-softplus score, correction-bias selection,
//! and optional Top-K normalization. It is not a gate projection, a vision
//! bias path, a full `MoE` execution, or a `PyTorch` Top-K tie-order oracle.

use thiserror::Error;

/// Maximum number of routed experts accepted for one bounded score row.
pub const MAX_FLASH_ROUTING_WIDTH: usize = 4_096;

/// One selected routed expert with its final multiplicative route weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExpertRoute {
    expert_index: usize,
    weight: f32,
}

impl ExpertRoute {
    /// Returns the canonical routed-expert index.
    #[must_use]
    pub const fn expert_index(self) -> usize {
        self.expert_index
    }

    /// Returns the final scalar route weight after any Top-K normalization.
    #[must_use]
    pub const fn weight(self) -> f32 {
        self.weight
    }
}

/// An invalid bounded V4.1 Flash routing request or non-finite scalar result.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum FlashRoutingError {
    /// The expert score row is empty.
    #[error("Flash routing requires at least one expert")]
    EmptyExperts,
    /// The correction-bias row must correspond exactly to the logit row.
    #[error("Flash routing bias length is {bias}, expected {logits}")]
    BiasLength {
        /// Supplied correction-bias count.
        bias: usize,
        /// Supplied logit count.
        logits: usize,
    },
    /// The score row exceeds the helper's explicit work bound.
    #[error("Flash routing width {width} exceeds maximum {max_width}")]
    WidthTooLarge {
        /// Supplied expert count.
        width: usize,
        /// Maximum accepted expert count.
        max_width: usize,
    },
    /// The requested Top-K is not in the closed expert-count range.
    #[error("Flash routing Top-K {top_k} is invalid for {experts} experts")]
    InvalidTopK {
        /// Requested selected-expert count.
        top_k: usize,
        /// Available expert count.
        experts: usize,
    },
    /// The gate temperature must be finite and strictly positive.
    #[error("Flash routing gate temperature must be finite and positive")]
    InvalidGateTemperature,
    /// The route scale must be finite and strictly positive.
    #[error("Flash routing route scale must be finite and positive")]
    InvalidRouteScale,
    /// A supplied projected gate logit was non-finite.
    #[error("Flash routing logit at expert {expert_index} is non-finite")]
    NonFiniteLogit {
        /// Index of the invalid expert logit.
        expert_index: usize,
    },
    /// A supplied correction bias was non-finite.
    #[error("Flash routing bias at expert {expert_index} is non-finite")]
    NonFiniteBias {
        /// Index of the invalid correction bias.
        expert_index: usize,
    },
    /// Temperature scaling, sqrt-softplus scoring, or biased selection overflowed.
    #[error("Flash routing scalar computation overflowed at expert {expert_index}")]
    ValueOverflow {
        /// Expert whose scalar intermediate could not remain finite.
        expert_index: usize,
    },
    /// Equal biased scores straddle the requested Top-K cutoff.
    #[error("equal biased scores straddle the Flash routing Top-K cutoff")]
    AmbiguousCutoffTie,
}

/// Selects bounded V4.1 Flash text routes from one row of projected gate logits.
///
/// This is the pinned non-softmax/non-sigmoid path: it divides each supplied
/// logit by `gate_temperature`, applies `sqrt(softplus(.))`, adds `bias` only
/// while selecting Top-K experts, gathers the original unbiased scores, then
/// optionally normalizes them when `top_k > 1` before multiplying by
/// `route_scale`. The softplus threshold is the `PyTorch` default `20`.
///
/// A tie at the selection cutoff is rejected because `PyTorch` does not offer
/// a portable tie order. Ties wholly inside the selected set are accepted.
/// Their normalization follows this helper's scalar score-ranked traversal,
/// not a reduction-bit parity claim. Returned routes are then in ascending
/// expert-index order, matching the pinned `MoE` expert iteration order.
///
/// This does not project gate weights, choose a vision bias, execute experts,
/// or establish full-model or hardware parity.
///
/// # Errors
///
/// Returns [`FlashRoutingError`] for malformed, over-bounded, non-finite, or
/// numerically overflowing inputs, and for a tie crossing the Top-K cutoff.
pub fn flash_sqrt_softplus_routes(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    gate_temperature: f32,
    normalize_top_k: bool,
    route_scale: f32,
) -> Result<Vec<ExpertRoute>, FlashRoutingError> {
    validate_request(logits, bias, top_k, gate_temperature, route_scale)?;

    let mut ranked = Vec::with_capacity(logits.len());
    for (expert_index, (&logit, &correction_bias)) in logits.iter().zip(bias).enumerate() {
        let scaled = logit / gate_temperature;
        if !scaled.is_finite() {
            return Err(FlashRoutingError::ValueOverflow { expert_index });
        }
        let score = sqrt_softplus(scaled);
        if !score.is_finite() {
            return Err(FlashRoutingError::ValueOverflow { expert_index });
        }
        let selection_score = score + correction_bias;
        if !selection_score.is_finite() {
            return Err(FlashRoutingError::ValueOverflow { expert_index });
        }
        ranked.push(Candidate {
            expert_index,
            score,
            selection_score,
            weight: 0.0,
        });
    }

    ranked.sort_unstable_by(|left, right| right.selection_score.total_cmp(&left.selection_score));
    if top_k < ranked.len()
        && scores_equal(
            ranked[top_k - 1].selection_score,
            ranked[top_k].selection_score,
        )
    {
        return Err(FlashRoutingError::AmbiguousCutoffTie);
    }
    ranked.truncate(top_k);

    // This order is intentionally the score-ranked selection traversal. The
    // later expert-index sort is only the `MoE` execution presentation order.
    let denominator = if normalize_top_k && top_k > 1 {
        let score_sum = ranked.iter().map(|candidate| candidate.score).sum::<f32>();
        let denominator = score_sum + 1.0e-20_f32;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(FlashRoutingError::ValueOverflow {
                expert_index: ranked[0].expert_index,
            });
        }
        Some(denominator)
    } else {
        None
    };

    for candidate in &mut ranked {
        let normalized = denominator.map_or(candidate.score, |value| candidate.score / value);
        candidate.weight = normalized * route_scale;
        if !candidate.weight.is_finite() {
            return Err(FlashRoutingError::ValueOverflow {
                expert_index: candidate.expert_index,
            });
        }
    }
    ranked.sort_unstable_by_key(|candidate| candidate.expert_index);
    Ok(ranked
        .into_iter()
        .map(|candidate| ExpertRoute {
            expert_index: candidate.expert_index,
            weight: candidate.weight,
        })
        .collect())
}

#[derive(Clone, Copy)]
struct Candidate {
    expert_index: usize,
    score: f32,
    selection_score: f32,
    weight: f32,
}

fn validate_request(
    logits: &[f32],
    bias: &[f32],
    top_k: usize,
    gate_temperature: f32,
    route_scale: f32,
) -> Result<(), FlashRoutingError> {
    if logits.is_empty() {
        return Err(FlashRoutingError::EmptyExperts);
    }
    if bias.len() != logits.len() {
        return Err(FlashRoutingError::BiasLength {
            bias: bias.len(),
            logits: logits.len(),
        });
    }
    if logits.len() > MAX_FLASH_ROUTING_WIDTH {
        return Err(FlashRoutingError::WidthTooLarge {
            width: logits.len(),
            max_width: MAX_FLASH_ROUTING_WIDTH,
        });
    }
    if top_k == 0 || top_k > logits.len() {
        return Err(FlashRoutingError::InvalidTopK {
            top_k,
            experts: logits.len(),
        });
    }
    if !gate_temperature.is_finite() || gate_temperature <= 0.0 {
        return Err(FlashRoutingError::InvalidGateTemperature);
    }
    if !route_scale.is_finite() || route_scale <= 0.0 {
        return Err(FlashRoutingError::InvalidRouteScale);
    }
    for (expert_index, (&logit, &correction_bias)) in logits.iter().zip(bias).enumerate() {
        if !logit.is_finite() {
            return Err(FlashRoutingError::NonFiniteLogit { expert_index });
        }
        if !correction_bias.is_finite() {
            return Err(FlashRoutingError::NonFiniteBias { expert_index });
        }
    }
    Ok(())
}

fn sqrt_softplus(value: f32) -> f32 {
    let softplus = if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    };
    softplus.sqrt()
}

fn scores_equal(left: f32, right: f32) -> bool {
    left.to_bits() == right.to_bits()
        || (left.abs().to_bits() == 0.0_f32.to_bits() && right.abs().to_bits() == 0.0_f32.to_bits())
}

#[cfg(test)]
mod tests {
    use super::{
        FlashRoutingError, MAX_FLASH_ROUTING_WIDTH, flash_sqrt_softplus_routes, sqrt_softplus,
    };

    #[test]
    fn correction_bias_changes_selection_but_not_gathered_weight() {
        let routes =
            flash_sqrt_softplus_routes(&[1.0, 1.0, 0.0], &[0.0, 3.0, 0.0], 1, 1.0, true, 1.0)
                .expect("finite bounded route");
        assert_eq!(routes[0].expert_index(), 1);
        assert_eq!(routes[0].weight().to_bits(), sqrt_softplus(1.0).to_bits());
    }

    #[test]
    fn top_one_skips_normalization_and_routes_are_canonicalized() {
        let routes =
            flash_sqrt_softplus_routes(&[0.0, 2.0, 1.0], &[0.0, 0.0, 4.0], 2, 1.0, true, 2.5)
                .expect("finite bounded route");
        assert_eq!(
            routes
                .iter()
                .map(|route| route.expert_index())
                .collect::<Vec<_>>(),
            [1, 2]
        );

        let one = flash_sqrt_softplus_routes(&[0.0, 1.0], &[0.0, 0.0], 1, 1.0, true, 2.5)
            .expect("finite Top-1 route");
        assert_eq!(one[0].expert_index(), 1);
        assert_eq!(
            one[0].weight().to_bits(),
            (sqrt_softplus(1.0) * 2.5).to_bits()
        );
    }

    #[test]
    fn stable_score_floor_and_large_finite_score_remain_finite() {
        let routes =
            flash_sqrt_softplus_routes(&[-f32::MAX, f32::MAX], &[0.0, 0.0], 2, 1.0, true, 1.0)
                .expect("finite extreme scores");
        assert_eq!(routes[0].expert_index(), 0);
        assert_eq!(routes[0].weight().to_bits(), 0.0_f32.to_bits());
        assert!(routes[1].weight().is_finite());
        assert!(routes[1].weight() > 0.0);
    }

    #[test]
    fn rejects_overflow_in_temperature_and_final_route_scaling() {
        for (logit, temperature, scale) in [
            (f32::MAX, f32::from_bits(1), 1.0),
            (f32::MAX, 1.0, f32::MAX),
        ] {
            assert_eq!(
                flash_sqrt_softplus_routes(&[logit], &[0.0], 1, temperature, false, scale),
                Err(FlashRoutingError::ValueOverflow { expert_index: 0 })
            );
        }
        assert_eq!(
            flash_sqrt_softplus_routes(&[0.0], &[f32::NAN], 1, 1.0, true, 1.0),
            Err(FlashRoutingError::NonFiniteBias { expert_index: 0 })
        );
        assert_eq!(
            flash_sqrt_softplus_routes(&[0.0], &[0.0], 1, 1.0, true, 0.0),
            Err(FlashRoutingError::InvalidRouteScale)
        );
    }

    #[test]
    fn rejects_malformed_nonfinite_and_ambiguous_requests() {
        assert!(matches!(
            flash_sqrt_softplus_routes(&[], &[], 1, 1.0, false, 1.0),
            Err(FlashRoutingError::EmptyExperts)
        ));
        assert!(matches!(
            flash_sqrt_softplus_routes(&[0.0], &[], 1, 1.0, false, 1.0),
            Err(FlashRoutingError::BiasLength { .. })
        ));
        assert!(matches!(
            flash_sqrt_softplus_routes(&[0.0], &[0.0], 0, 1.0, false, 1.0),
            Err(FlashRoutingError::InvalidTopK { .. })
        ));
        assert!(matches!(
            flash_sqrt_softplus_routes(&[0.0], &[0.0], 1, 0.0, false, 1.0),
            Err(FlashRoutingError::InvalidGateTemperature)
        ));
        assert!(matches!(
            flash_sqrt_softplus_routes(&[f32::NAN], &[0.0], 1, 1.0, false, 1.0),
            Err(FlashRoutingError::NonFiniteLogit { .. })
        ));
        assert!(matches!(
            flash_sqrt_softplus_routes(&[1.0, 1.0], &[0.0, 0.0], 1, 1.0, false, 1.0),
            Err(FlashRoutingError::AmbiguousCutoffTie)
        ));
        let too_wide = vec![0.0; MAX_FLASH_ROUTING_WIDTH + 1];
        assert!(matches!(
            flash_sqrt_softplus_routes(&too_wide, &too_wide, 1, 1.0, false, 1.0),
            Err(FlashRoutingError::WidthTooLarge { .. })
        ));
    }
}
