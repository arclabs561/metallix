//! Bounded scalar qualification of the V4.1 Flash text `MoE` routing rule.
//!
//! The raw-logit helper and BF16 gate-projection wrapper mirror the pinned
//! `Gate.forward` sqrt-softplus score, correction-bias selection, and optional
//! Top-K normalization. They are not a vision-bias path, a full `MoE`
//! execution, or a `PyTorch` Top-K tie-order oracle.

use thiserror::Error;

/// Maximum number of routed experts accepted for one bounded score row.
pub const MAX_FLASH_ROUTING_WIDTH: usize = 4_096;

/// Maximum BF16 gate-matrix elements accepted by one scalar projection call.
pub const MAX_FLASH_GATE_PROJECTION_ELEMENTS: usize = 1 << 22;

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

/// An invalid bounded BF16 V4.1 Flash gate-projection request.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum FlashGateProjectionError {
    /// The supplied hidden width is zero.
    #[error("Flash gate projection requires a nonzero hidden width")]
    EmptyHiddenWidth,
    /// The BF16 hidden row does not have its declared width.
    #[error("Flash gate hidden length is {actual}, expected {expected}")]
    HiddenLength {
        /// Supplied BF16 hidden-value count.
        actual: usize,
        /// Declared hidden width.
        expected: usize,
    },
    /// The requested gate-matrix shape cannot be represented safely.
    #[error("Flash gate projection shape overflowed for gate weights")]
    ShapeOverflow,
    /// The gate matrix exceeds the helper's explicit scalar-work bound.
    #[error("Flash gate projection has {elements} elements, maximum is {max_elements}")]
    WorkTooLarge {
        /// Requested `[experts, hidden_width]` element count.
        elements: usize,
        /// Maximum accepted scalar multiply count.
        max_elements: usize,
    },
    /// The BF16 gate matrix does not have its declared row-major shape.
    #[error("Flash gate weight length is {actual}, expected {expected}")]
    GateWeightLength {
        /// Supplied BF16 gate-weight count.
        actual: usize,
        /// Required `[experts, hidden_width]` element count.
        expected: usize,
    },
    /// A BF16 hidden value represents infinity or NaN.
    #[error("Flash gate hidden BF16 value at element {element} is non-finite")]
    NonFiniteHidden {
        /// Flat hidden-row element index.
        element: usize,
    },
    /// A BF16 gate weight represents infinity or NaN.
    #[error("Flash gate BF16 weight at expert {expert_index}, hidden {hidden_index} is non-finite")]
    NonFiniteWeight {
        /// Row-major gate expert index.
        expert_index: usize,
        /// Reduction-axis index within the expert row.
        hidden_index: usize,
    },
    /// A scalar FP32 gate-dot product overflowed.
    #[error("Flash gate FP32 dot overflowed at expert {expert_index}, hidden {hidden_index}")]
    ProjectionOverflow {
        /// Row-major gate expert index.
        expert_index: usize,
        /// Reduction-axis term that overflowed a product or accumulation.
        hidden_index: usize,
    },
    /// The routing request was invalid after a finite gate projection.
    #[error(transparent)]
    Routing(#[from] FlashRoutingError),
}

/// Projects one BF16 hidden row through BF16 V4.1 Flash gate weights and routes it.
///
/// `hidden_bf16` is `[hidden_width]` and `gate_weights_bf16` is a row-major
/// `[experts, hidden_width]` matrix. Both BF16 bit patterns are promoted
/// exactly to scalar FP32 before every multiply and accumulation, matching the
/// pinned `Gate.forward` use of `x.float()` and `self.weight.float()` before
/// `F.linear`. The pinned normal Flash source sets `torch`'s default storage
/// type to BF16 and the captured shard header records
/// `layers.6.ffn.gate.weight` as BF16 `[384, 5120]`.
///
/// This is a bounded software FP32 dot reference. It does not identify a
/// checkpoint revision or shard, load tensors, establish `F.linear` kernel
/// reduction parity, project vision routing, or execute a full `MoE` block.
///
/// # Errors
///
/// Returns [`FlashGateProjectionError`] before producing routes if the exact
/// shapes, BF16 values, scalar-work budget, dot products, or routing request
/// are invalid.
#[allow(
    clippy::too_many_arguments,
    reason = "the gate and routing shapes are explicit direct-runtime roles"
)]
pub fn flash_bf16_gate_routes(
    hidden_bf16: &[u16],
    gate_weights_bf16: &[u16],
    experts: usize,
    hidden_width: usize,
    bias: &[f32],
    top_k: usize,
    gate_temperature: f32,
    normalize_top_k: bool,
    route_scale: f32,
) -> Result<Vec<ExpertRoute>, FlashGateProjectionError> {
    let gate_elements = validate_gate_shape(hidden_bf16, gate_weights_bf16, experts, hidden_width)?;
    validate_routing_parameters(experts, bias, top_k, gate_temperature, route_scale)?;
    let hidden: Vec<f32> = hidden_bf16
        .iter()
        .enumerate()
        .map(|(element, &bits)| {
            let value = bf16_to_f32(bits);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(FlashGateProjectionError::NonFiniteHidden { element })
            }
        })
        .collect::<Result<_, _>>()?;
    for (element, &bits) in gate_weights_bf16.iter().enumerate().take(gate_elements) {
        if !bf16_to_f32(bits).is_finite() {
            return Err(FlashGateProjectionError::NonFiniteWeight {
                expert_index: element / hidden_width,
                hidden_index: element % hidden_width,
            });
        }
    }

    let mut logits = Vec::with_capacity(experts);
    for expert_index in 0..experts {
        let row =
            &gate_weights_bf16[expert_index * hidden_width..(expert_index + 1) * hidden_width];
        let mut dot = 0.0_f32;
        for (hidden_index, (&input, &weight_bits)) in hidden.iter().zip(row).enumerate() {
            let term = input * bf16_to_f32(weight_bits);
            if !term.is_finite() {
                return Err(FlashGateProjectionError::ProjectionOverflow {
                    expert_index,
                    hidden_index,
                });
            }
            dot += term;
            if !dot.is_finite() {
                return Err(FlashGateProjectionError::ProjectionOverflow {
                    expert_index,
                    hidden_index,
                });
            }
        }
        logits.push(dot);
    }
    Ok(flash_sqrt_softplus_routes(
        &logits,
        bias,
        top_k,
        gate_temperature,
        normalize_top_k,
        route_scale,
    )?)
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
    validate_routing_parameters(logits.len(), bias, top_k, gate_temperature, route_scale)?;
    for (expert_index, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            return Err(FlashRoutingError::NonFiniteLogit { expert_index });
        }
    }
    Ok(())
}

fn validate_routing_parameters(
    experts: usize,
    bias: &[f32],
    top_k: usize,
    gate_temperature: f32,
    route_scale: f32,
) -> Result<(), FlashRoutingError> {
    if experts == 0 {
        return Err(FlashRoutingError::EmptyExperts);
    }
    if bias.len() != experts {
        return Err(FlashRoutingError::BiasLength {
            bias: bias.len(),
            logits: experts,
        });
    }
    if experts > MAX_FLASH_ROUTING_WIDTH {
        return Err(FlashRoutingError::WidthTooLarge {
            width: experts,
            max_width: MAX_FLASH_ROUTING_WIDTH,
        });
    }
    if top_k == 0 || top_k > experts {
        return Err(FlashRoutingError::InvalidTopK { top_k, experts });
    }
    if !gate_temperature.is_finite() || gate_temperature <= 0.0 {
        return Err(FlashRoutingError::InvalidGateTemperature);
    }
    if !route_scale.is_finite() || route_scale <= 0.0 {
        return Err(FlashRoutingError::InvalidRouteScale);
    }
    for (expert_index, &correction_bias) in bias.iter().enumerate() {
        if !correction_bias.is_finite() {
            return Err(FlashRoutingError::NonFiniteBias { expert_index });
        }
    }
    Ok(())
}

fn validate_gate_shape(
    hidden_bf16: &[u16],
    gate_weights_bf16: &[u16],
    experts: usize,
    hidden_width: usize,
) -> Result<usize, FlashGateProjectionError> {
    if hidden_width == 0 {
        return Err(FlashGateProjectionError::EmptyHiddenWidth);
    }
    if hidden_bf16.len() != hidden_width {
        return Err(FlashGateProjectionError::HiddenLength {
            actual: hidden_bf16.len(),
            expected: hidden_width,
        });
    }
    let gate_elements = experts
        .checked_mul(hidden_width)
        .ok_or(FlashGateProjectionError::ShapeOverflow)?;
    if gate_elements > MAX_FLASH_GATE_PROJECTION_ELEMENTS {
        return Err(FlashGateProjectionError::WorkTooLarge {
            elements: gate_elements,
            max_elements: MAX_FLASH_GATE_PROJECTION_ELEMENTS,
        });
    }
    if gate_weights_bf16.len() != gate_elements {
        return Err(FlashGateProjectionError::GateWeightLength {
            actual: gate_weights_bf16.len(),
            expected: gate_elements,
        });
    }
    Ok(gate_elements)
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
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
        FlashGateProjectionError, FlashRoutingError, MAX_FLASH_GATE_PROJECTION_ELEMENTS,
        MAX_FLASH_ROUTING_WIDTH, flash_bf16_gate_routes, flash_sqrt_softplus_routes, sqrt_softplus,
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

    #[test]
    fn bf16_gate_projection_computes_logits_before_routing() {
        let hidden = [0x3f80_u16; 3]; // BF16 [1, 1, 1]
        let weights = [
            0x0000, 0x0000, 0x0000, // expert 0: 0
            0x3f80, 0x0000, 0x0000, // expert 1: 1
            0x3f80, 0x3f80, 0x3f80, // expert 2: 3
        ];
        let routes = flash_bf16_gate_routes(
            &hidden,
            &weights,
            3,
            3,
            &[0.0, 0.0, -10.0],
            2,
            1.0,
            true,
            1.0,
        )
        .expect("finite BF16 gate matrix");
        assert_eq!(
            routes
                .iter()
                .map(|route| route.expert_index())
                .collect::<Vec<_>>(),
            [0, 1]
        );
        let score0 = sqrt_softplus(0.0);
        let score1 = sqrt_softplus(1.0);
        let denominator = score1 + score0 + 1.0e-20_f32;
        assert_eq!(
            routes[0].weight().to_bits(),
            (score0 / denominator).to_bits()
        );
        assert_eq!(
            routes[1].weight().to_bits(),
            (score1 / denominator).to_bits()
        );
    }

    #[test]
    fn bf16_gate_keeps_products_and_accumulation_in_fp32() {
        // 1 + 2^-8 - 1 = 2^-8. BF16 round-to-nearest-even after each
        // addition would instead erase the half-ULP and give zero.
        let accumulation = flash_bf16_gate_routes(
            &[0x3f80; 3],
            &[0x3f80, 0x3b80, 0xbf80, 0x3b00, 0, 0],
            2,
            3,
            &[0.0; 2],
            1,
            2.0_f32.powi(-8),
            false,
            1.0,
        )
        .expect("FP32 accumulation preserves the cancellation residual");

        // (1 + 2^-7)^2 - (1 + 2^-6) = 2^-14. Rounding the first
        // product to BF16 would erase that residual before subtraction.
        let product = flash_bf16_gate_routes(
            &[0x3f81, 0x3f80],
            &[0x3f81, 0xbf82, 0, 0x3800],
            2,
            2,
            &[0.0; 2],
            1,
            2.0_f32.powi(-14),
            false,
            1.0,
        )
        .expect("FP32 multiplication preserves the cancellation residual");

        // Both exact scaled rows are [1, 0.5]. The deliberately separate
        // f64 score formula checks the selected unnormalized route weight.
        let expected_weight = (1.0_f64.exp() + 1.0).ln().sqrt();
        for routes in [&accumulation, &product] {
            assert_eq!(routes.len(), 1);
            assert_eq!(routes[0].expert_index(), 0);
            assert!((f64::from(routes[0].weight()) - expected_weight).abs() < 1.0e-7);
        }
        let prematurely_rounded =
            flash_sqrt_softplus_routes(&[0.0, 0.5], &[0.0; 2], 1, 1.0, false, 1.0)
                .expect("premature BF16 rounding changes the winner without a tie");
        assert_eq!(prematurely_rounded[0].expert_index(), 1);
    }

    #[test]
    fn bf16_gate_projection_rejects_shapes_nonfinite_values_and_overflow() {
        assert!(matches!(
            flash_bf16_gate_routes(&[0x3f80], &[], 1, 2, &[0.0], 1, 1.0, false, 1.0),
            Err(FlashGateProjectionError::HiddenLength { .. })
        ));
        assert!(matches!(
            flash_bf16_gate_routes(&[0x3f80], &[], 1, 1, &[0.0], 1, 1.0, false, 1.0),
            Err(FlashGateProjectionError::GateWeightLength { .. })
        ));
        let oversized_hidden = vec![0x0000; 1_025];
        let oversized = flash_bf16_gate_routes(
            &oversized_hidden,
            &[],
            4_096,
            oversized_hidden.len(),
            &[0.0; 4_096],
            1,
            1.0,
            false,
            1.0,
        )
        .expect_err("oversized gate matrix must not be traversed");
        assert_eq!(
            oversized,
            FlashGateProjectionError::WorkTooLarge {
                elements: MAX_FLASH_GATE_PROJECTION_ELEMENTS + 4_096,
                max_elements: MAX_FLASH_GATE_PROJECTION_ELEMENTS,
            }
        );
        assert!(matches!(
            flash_bf16_gate_routes(&[0x7f80], &[0x3f80], 1, 1, &[0.0], 1, 1.0, false, 1.0),
            Err(FlashGateProjectionError::NonFiniteHidden { element: 0 })
        ));
        assert!(matches!(
            flash_bf16_gate_routes(&[0x3f80], &[0x7fc0], 1, 1, &[0.0], 1, 1.0, false, 1.0),
            Err(FlashGateProjectionError::NonFiniteWeight {
                expert_index: 0,
                hidden_index: 0,
            })
        ));
        assert!(matches!(
            flash_bf16_gate_routes(
                &[0x7f7f, 0x7f7f],
                &[0x3f80, 0x3f80],
                1,
                2,
                &[0.0],
                1,
                1.0,
                false,
                1.0,
            ),
            Err(FlashGateProjectionError::ProjectionOverflow {
                expert_index: 0,
                hidden_index: 1,
            })
        ));
    }
}
