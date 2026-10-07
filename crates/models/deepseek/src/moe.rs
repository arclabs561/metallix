//! Bounded one-token reference composition for the V4.1 text `MoE`.
//!
//! This preserves the pinned routed and shared expert order over encoded
//! FP4/FP8 weights. It is a scalar CPU reference for reduced-graph comparison,
//! not a scheduler, checkpoint loader, production serving path, or hardware
//! reduction-parity claim.

use thiserror::Error;

use crate::{
    precision::{
        ActivationGroup, ActivationQuantError, Fp4LinearError, Fp8ForwardError, Fp8LinearError,
        bf16_to_f32, f32_to_bf16_rne, fp4_linear_runtime_f32_owned, fp8_linear_f32,
        quantize_bf16_activations_e4m3fn,
    },
    routing::{ExpertRoute, FlashGateProjectionError, flash_bf16_gate_routes},
};

const GROUP_WIDTH: usize = 32;
/// Largest accepted per-token vector in this bounded reference.
pub(crate) const MAX_MOE_ELEMENTS: usize = 1 << 22;
/// Largest logical projection-term count across selected and shared experts.
///
/// Scalar linear leaves perform validation and write passes, so this is not an
/// exact instruction count. Gate projection and activation preparation add work.
pub(crate) const MAX_MOE_WORK: usize = 1 << 28;

/// Static V4.1 `MoE` geometry and routing parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoEConfig {
    hidden_width: usize,
    intermediate_width: usize,
    swiglu_limit: f32,
    top_k: usize,
    gate_temperature: f32,
    normalize_top_k: bool,
    route_scale: f32,
}

impl MoEConfig {
    /// Validates the dimensions and scalar parameters required by one token.
    ///
    /// # Errors
    ///
    /// * [`MoEError::InvalidWidth`] for a zero, unaligned or oversized width.
    /// * [`MoEError::InvalidSwiGluLimit`], [`MoEError::InvalidTopK`],
    ///   [`MoEError::InvalidGateTemperature`] and
    ///   [`MoEError::InvalidRouteScale`] for a scalar outside its domain.
    #[allow(
        clippy::too_many_arguments,
        reason = "the source configuration has these independent scalar roles"
    )]
    pub fn new(
        hidden_width: usize,
        intermediate_width: usize,
        swiglu_limit: f32,
        top_k: usize,
        gate_temperature: f32,
        normalize_top_k: bool,
        route_scale: f32,
    ) -> Result<Self, MoEError> {
        validate_width("hidden", hidden_width)?;
        validate_width("intermediate", intermediate_width)?;
        if !swiglu_limit.is_finite() || swiglu_limit < 0.0 {
            return Err(MoEError::InvalidSwiGluLimit);
        }
        if top_k == 0 {
            return Err(MoEError::InvalidTopK);
        }
        if !gate_temperature.is_finite() || gate_temperature <= 0.0 {
            return Err(MoEError::InvalidGateTemperature);
        }
        if !route_scale.is_finite() || route_scale <= 0.0 {
            return Err(MoEError::InvalidRouteScale);
        }
        Ok(Self {
            hidden_width,
            intermediate_width,
            swiglu_limit,
            top_k,
            gate_temperature,
            normalize_top_k,
            route_scale,
        })
    }
}

/// Borrowed packed E2M1 routed-expert projections with E8M0 scales.
#[derive(Clone, Copy, Debug)]
pub struct Fp4ExpertWeights<'a> {
    w1_codes: &'a [u8],
    w1_scales: &'a [u8],
    w2_codes: &'a [u8],
    w2_scales: &'a [u8],
    w3_codes: &'a [u8],
    w3_scales: &'a [u8],
    hidden_width: usize,
    intermediate_width: usize,
}

impl<'a> Fp4ExpertWeights<'a> {
    /// Validates geometry and exact buffer lengths for one packed FP4 expert.
    /// Numerical scale validation occurs when the expert is executed.
    ///
    /// # Errors
    ///
    /// * [`MoEError::InvalidWidth`] for a zero, unaligned or oversized width.
    /// * [`MoEError::Length`] when a code or scale buffer does not match the
    ///   geometry, and [`MoEError::ShapeOverflow`] when that geometry does not
    ///   fit in `usize`.
    #[allow(
        clippy::too_many_arguments,
        reason = "each encoded projection owns distinct code and scale storage"
    )]
    pub fn new(
        hidden_width: usize,
        intermediate_width: usize,
        w1_codes: &'a [u8],
        w1_scales: &'a [u8],
        w2_codes: &'a [u8],
        w2_scales: &'a [u8],
        w3_codes: &'a [u8],
        w3_scales: &'a [u8],
    ) -> Result<Self, MoEError> {
        validate_width("hidden", hidden_width)?;
        validate_width("intermediate", intermediate_width)?;
        validate_fp4_matrix("w1", w1_codes, w1_scales, intermediate_width, hidden_width)?;
        validate_fp4_matrix("w2", w2_codes, w2_scales, hidden_width, intermediate_width)?;
        validate_fp4_matrix("w3", w3_codes, w3_scales, intermediate_width, hidden_width)?;
        Ok(Self {
            w1_codes,
            w1_scales,
            w2_codes,
            w2_scales,
            w3_codes,
            w3_scales,
            hidden_width,
            intermediate_width,
        })
    }

    /// Executes one checked routed FP4 expert over a BF16 hidden token.
    ///
    /// This is the routed-expert leaf only: it does not select a route, add a
    /// shared expert, or accumulate across experts. A supplied route weight is
    /// applied within the source `SwiGLU` stage before W2, matching
    /// [`MoEReference::forward_token`].
    ///
    /// # Errors
    ///
    /// * [`MoEError::Length`] when `input_bf16` is not one hidden-width token.
    /// * [`MoEError::InvalidSwiGluLimit`] and [`MoEError::InvalidRouteWeight`]
    ///   for a control outside its domain.
    /// * [`MoEError::WorkOverflow`] and [`MoEError::WorkTooLarge`] when the
    ///   expert's work does not fit the bounded reference.
    /// * [`MoEError::NonFinite`] and [`MoEError::Bf16Overflow`] when a stage
    ///   leaves the finite FP32 or BF16 range.
    /// * [`MoEError::Activation`], [`MoEError::Fp4`] and [`MoEError::Fp8`] when
    ///   a quantization or projection leaf rejects its input or overflows.
    /// * [`MoEError::Allocation`] when a temporary buffer cannot be reserved.
    pub fn forward_token(
        &self,
        input_bf16: &[u16],
        swiglu_limit: f32,
        route_weight: Option<f32>,
    ) -> Result<Vec<u16>, MoEError> {
        if input_bf16.len() != self.hidden_width {
            return Err(MoEError::Length {
                field: "input_bf16",
                actual: input_bf16.len(),
                expected: self.hidden_width,
            });
        }
        if !swiglu_limit.is_finite() || swiglu_limit < 0.0 {
            return Err(MoEError::InvalidSwiGluLimit);
        }
        if route_weight.is_some_and(|weight| !weight.is_finite() || weight < 0.0) {
            return Err(MoEError::InvalidRouteWeight);
        }
        validate_expert_work(self.hidden_width, self.intermediate_width)?;
        for (element, &bits) in input_bf16.iter().enumerate() {
            if !bf16_to_f32(bits).is_finite() {
                return Err(MoEError::NonFinite {
                    stage: "FP4 expert input",
                    element,
                });
            }
        }
        project_fp4_expert(input_bf16, self, swiglu_limit, route_weight)
    }
}

/// Borrowed E4M3 shared-expert projections with E8M0 scales.
#[derive(Clone, Copy, Debug)]
pub struct Fp8ExpertWeights<'a> {
    w1_codes: &'a [u8],
    w1_scales: &'a [u8],
    w2_codes: &'a [u8],
    w2_scales: &'a [u8],
    w3_codes: &'a [u8],
    w3_scales: &'a [u8],
    hidden_width: usize,
    intermediate_width: usize,
}

impl<'a> Fp8ExpertWeights<'a> {
    /// Validates geometry and exact buffer lengths for the FP8 shared expert.
    /// Numerical code and scale validation occurs during execution.
    ///
    /// # Errors
    ///
    /// * [`MoEError::InvalidWidth`] for a zero, unaligned or oversized width.
    /// * [`MoEError::Length`] when a code or scale buffer does not match the
    ///   geometry, and [`MoEError::ShapeOverflow`] when that geometry does not
    ///   fit in `usize`.
    #[allow(
        clippy::too_many_arguments,
        reason = "each encoded projection owns distinct code and scale storage"
    )]
    pub fn new(
        hidden_width: usize,
        intermediate_width: usize,
        w1_codes: &'a [u8],
        w1_scales: &'a [u8],
        w2_codes: &'a [u8],
        w2_scales: &'a [u8],
        w3_codes: &'a [u8],
        w3_scales: &'a [u8],
    ) -> Result<Self, MoEError> {
        validate_width("hidden", hidden_width)?;
        validate_width("intermediate", intermediate_width)?;
        validate_fp8_matrix("w1", w1_codes, w1_scales, intermediate_width, hidden_width)?;
        validate_fp8_matrix("w2", w2_codes, w2_scales, hidden_width, intermediate_width)?;
        validate_fp8_matrix("w3", w3_codes, w3_scales, intermediate_width, hidden_width)?;
        Ok(Self {
            w1_codes,
            w1_scales,
            w2_codes,
            w2_scales,
            w3_codes,
            w3_scales,
            hidden_width,
            intermediate_width,
        })
    }
}

/// A bounded source-order `MoE` reference over one BF16 hidden token.
#[derive(Clone, Copy, Debug)]
pub struct MoEReference<'a> {
    config: MoEConfig,
    gate_bf16: &'a [u16],
    bias: &'a [f32],
    routed: RoutedExperts<'a>,
    shared: Fp8ExpertWeights<'a>,
}

/// Supplies routed experts after routing selects them.
///
/// Implementations own (or borrow) the packed bytes and lend them to `run`
/// for the duration of one expert call, so a caller can hold only a bounded
/// working set of a checkpoint's experts.
pub trait RoutedExpertSource {
    /// Runs `run` with routed expert `index`'s weights, or fails if it is unavailable.
    ///
    /// # Errors
    ///
    /// Returns the source's error or the error `run` returns.
    fn with_expert(
        &self,
        index: usize,
        run: &mut dyn FnMut(Fp4ExpertWeights<'_>) -> Result<Vec<u16>, MoEError>,
    ) -> Result<Vec<u16>, MoEError>;
}

#[derive(Clone, Copy, Debug)]
enum RoutedExperts<'a> {
    Dense(&'a [Fp4ExpertWeights<'a>]),
    Sparse(&'a [Option<Fp4ExpertWeights<'a>>]),
}

impl<'a> RoutedExperts<'a> {
    const fn len(self) -> usize {
        match self {
            Self::Dense(experts) => experts.len(),
            Self::Sparse(experts) => experts.len(),
        }
    }

    fn geometry_matches(self, config: MoEConfig) -> bool {
        match self {
            Self::Dense(experts) => experts.iter().all(|expert| {
                expert.hidden_width == config.hidden_width
                    && expert.intermediate_width == config.intermediate_width
            }),
            Self::Sparse(experts) => experts.iter().flatten().all(|expert| {
                expert.hidden_width == config.hidden_width
                    && expert.intermediate_width == config.intermediate_width
            }),
        }
    }

    fn selected(self, index: usize) -> Result<Fp4ExpertWeights<'a>, MoEError> {
        match self {
            Self::Dense(experts) => experts.get(index).copied().ok_or(MoEError::RouteOutOfRange),
            Self::Sparse(experts) => match experts.get(index) {
                Some(Some(expert)) => Ok(*expert),
                Some(None) => Err(MoEError::MissingRoutedExpert { index }),
                None => Err(MoEError::RouteOutOfRange),
            },
        }
    }
}

impl<'a> MoEReference<'a> {
    /// Returns the validated one-token hidden width.
    #[must_use]
    pub const fn hidden_width(&self) -> usize {
        self.config.hidden_width
    }

    /// Validates geometry, buffer lengths and the logical expert-work budget.
    /// Numerical gate validation occurs in [`Self::forward_token`].
    ///
    /// # Errors
    ///
    /// * [`MoEError::NoRoutedExperts`] and [`MoEError::TopKExceedsExperts`]
    ///   when the routed table cannot supply `top_k` experts.
    /// * [`MoEError::Length`] when the gate or bias does not match the expert
    ///   count and hidden width, and [`MoEError::ExpertGeometry`] when an
    ///   expert's widths differ from the configuration.
    /// * [`MoEError::ShapeOverflow`], [`MoEError::WorkOverflow`] and
    ///   [`MoEError::WorkTooLarge`] when the logical work does not fit the
    ///   bounded reference.
    pub fn new(
        config: MoEConfig,
        gate_bf16: &'a [u16],
        bias: &'a [f32],
        routed: &'a [Fp4ExpertWeights<'a>],
        shared: Fp8ExpertWeights<'a>,
    ) -> Result<Self, MoEError> {
        Self::new_with_routed(
            config,
            gate_bf16,
            bias,
            RoutedExperts::Dense(routed),
            shared,
        )
    }

    /// Validates a sparse logical routed-expert table.
    ///
    /// Missing entries retain their original gate IDs and are rejected only
    /// when routing selects them, before any expert arithmetic begins.
    ///
    /// # Errors
    ///
    /// * [`MoEError::NoRoutedExperts`] and [`MoEError::TopKExceedsExperts`]
    ///   when the routed table cannot supply `top_k` experts.
    /// * [`MoEError::Length`] when the gate or bias does not match the expert
    ///   count and hidden width, and [`MoEError::ExpertGeometry`] when an
    ///   expert's widths differ from the configuration.
    /// * [`MoEError::ShapeOverflow`], [`MoEError::WorkOverflow`] and
    ///   [`MoEError::WorkTooLarge`] when the logical work does not fit the
    ///   bounded reference.
    pub fn new_sparse(
        config: MoEConfig,
        gate_bf16: &'a [u16],
        bias: &'a [f32],
        routed: &'a [Option<Fp4ExpertWeights<'a>>],
        shared: Fp8ExpertWeights<'a>,
    ) -> Result<Self, MoEError> {
        Self::new_with_routed(
            config,
            gate_bf16,
            bias,
            RoutedExperts::Sparse(routed),
            shared,
        )
    }

    fn new_with_routed(
        config: MoEConfig,
        gate_bf16: &'a [u16],
        bias: &'a [f32],
        routed: RoutedExperts<'a>,
        shared: Fp8ExpertWeights<'a>,
    ) -> Result<Self, MoEError> {
        let expert_count = routed.len();
        if expert_count == 0 {
            return Err(MoEError::NoRoutedExperts);
        }
        if config.top_k > expert_count {
            return Err(MoEError::TopKExceedsExperts {
                top_k: config.top_k,
                experts: expert_count,
            });
        }
        let expected_gate = checked_product("gate", expert_count, config.hidden_width)?;
        if gate_bf16.len() != expected_gate {
            return Err(MoEError::Length {
                field: "gate_bf16",
                actual: gate_bf16.len(),
                expected: expected_gate,
            });
        }
        if bias.len() != expert_count {
            return Err(MoEError::Length {
                field: "bias",
                actual: bias.len(),
                expected: expert_count,
            });
        }
        if !routed.geometry_matches(config) {
            return Err(MoEError::ExpertGeometry);
        }
        if shared.hidden_width != config.hidden_width
            || shared.intermediate_width != config.intermediate_width
        {
            return Err(MoEError::ExpertGeometry);
        }
        let expert_work = checked_product(
            "expert work",
            config.hidden_width,
            config.intermediate_width,
        )?;
        let selected = config.top_k.checked_add(1).ok_or(MoEError::WorkOverflow)?;
        let work = expert_work
            .checked_mul(3)
            .and_then(|value| value.checked_mul(selected))
            .ok_or(MoEError::WorkOverflow)?;
        if work > MAX_MOE_WORK {
            return Err(MoEError::WorkTooLarge {
                elements: work,
                max: MAX_MOE_WORK,
            });
        }
        Ok(Self {
            config,
            gate_bf16,
            bias,
            routed,
            shared,
        })
    }

    /// Executes one V4.1 text `MoE` token in the source's expert-ID order.
    ///
    /// # Errors
    ///
    /// * [`MoEError::Length`] when `input_bf16` is not one hidden-width token.
    /// * [`MoEError::RouteOutOfRange`] and [`MoEError::MissingRoutedExpert`]
    ///   when routing selects an expert the table cannot supply; this is
    ///   checked before any expert runs.
    /// * [`MoEError::NonFinite`] and [`MoEError::Bf16Overflow`] when a stage
    ///   leaves the finite FP32 or BF16 range.
    /// * [`MoEError::Activation`], [`MoEError::Fp4`] and [`MoEError::Fp8`] when
    ///   a quantization or projection leaf rejects its input or overflows.
    /// * [`MoEError::Allocation`] when a temporary buffer cannot be reserved.
    pub fn forward_token(&self, input_bf16: &[u16]) -> Result<MoEDiagnostic, MoEError> {
        let routes = self.route_token(input_bf16)?;
        for route in &routes {
            let _ = self.routed.selected(route.expert_index())?;
        }
        self.run_routes(input_bf16, routes, |index, weight| {
            self.routed.selected(index)?.forward_token(
                input_bf16,
                self.config.swiglu_limit,
                Some(weight),
            )
        })
    }

    /// Executes one token with routed experts supplied after routing selects them.
    ///
    /// The construction-time routed table is not consulted; a model too large
    /// to hold every expert can build this reference over an empty sparse table
    /// and fetch only the selected experts. Supplied experts must match the
    /// configured geometry. No state changes, so a source failure leaves
    /// nothing partially applied.
    ///
    /// # Errors
    ///
    /// * [`MoEError::Length`] when `input_bf16` is not one hidden-width token.
    /// * [`MoEError::ExpertUnavailable`] when `source` cannot supply a selected
    ///   expert, and [`MoEError::ExpertGeometry`] when a supplied expert's
    ///   widths differ from the configuration.
    /// * [`MoEError::NonFinite`] and [`MoEError::Bf16Overflow`] when a stage
    ///   leaves the finite FP32 or BF16 range.
    /// * [`MoEError::Activation`], [`MoEError::Fp4`] and [`MoEError::Fp8`] when
    ///   a quantization or projection leaf rejects its input or overflows.
    /// * [`MoEError::Allocation`] when a temporary buffer cannot be reserved.
    pub fn forward_token_with(
        &self,
        input_bf16: &[u16],
        source: &dyn RoutedExpertSource,
    ) -> Result<MoEDiagnostic, MoEError> {
        let routes = self.route_token(input_bf16)?;
        self.run_routes(input_bf16, routes, |index, weight| {
            source.with_expert(index, &mut |expert| {
                if expert.hidden_width != self.config.hidden_width
                    || expert.intermediate_width != self.config.intermediate_width
                {
                    return Err(MoEError::ExpertGeometry);
                }
                expert.forward_token(input_bf16, self.config.swiglu_limit, Some(weight))
            })
        })
    }

    fn route_token(&self, input_bf16: &[u16]) -> Result<Vec<ExpertRoute>, MoEError> {
        if input_bf16.len() != self.config.hidden_width {
            return Err(MoEError::Length {
                field: "input_bf16",
                actual: input_bf16.len(),
                expected: self.config.hidden_width,
            });
        }
        Ok(flash_bf16_gate_routes(
            input_bf16,
            self.gate_bf16,
            self.routed.len(),
            self.config.hidden_width,
            self.bias,
            self.config.top_k,
            self.config.gate_temperature,
            self.config.normalize_top_k,
            self.config.route_scale,
        )?)
    }

    fn run_routes(
        &self,
        input_bf16: &[u16],
        routes: Vec<ExpertRoute>,
        mut run_expert: impl FnMut(usize, f32) -> Result<Vec<u16>, MoEError>,
    ) -> Result<MoEDiagnostic, MoEError> {
        let mut accumulator = allocate_f32("accumulator", self.config.hidden_width)?;
        let mut selected_outputs = Vec::new();
        selected_outputs
            .try_reserve_exact(routes.len())
            .map_err(|_| MoEError::Allocation {
                field: "selected outputs",
            })?;
        for route in &routes {
            let output = run_expert(route.expert_index(), route.weight())?;
            accumulate_bf16(&mut accumulator, &output, "routed accumulation")?;
            selected_outputs.push(output);
        }
        let shared_output = project_fp8_expert(input_bf16, &self.shared, self.config.swiglu_limit)?;
        accumulate_bf16(&mut accumulator, &shared_output, "shared accumulation")?;
        let output_bf16 = narrow_row(&accumulator, "final MoE output")?;
        Ok(MoEDiagnostic {
            routes,
            selected_outputs,
            shared_output,
            accumulator,
            output_bf16,
        })
    }
}

/// Bounded source-order observations from one token execution.
#[derive(Clone, Debug, PartialEq)]
pub struct MoEDiagnostic {
    routes: Vec<ExpertRoute>,
    selected_outputs: Vec<Vec<u16>>,
    shared_output: Vec<u16>,
    accumulator: Vec<f32>,
    output_bf16: Vec<u16>,
}

impl MoEDiagnostic {
    /// Returns selected routes in the pinned ascending expert-ID execution order.
    #[must_use]
    pub fn routes(&self) -> &[ExpertRoute] {
        &self.routes
    }

    /// Returns routed W2 BF16 outputs aligned with [`Self::routes`].
    #[must_use]
    pub fn selected_outputs_bf16(&self) -> &[Vec<u16>] {
        &self.selected_outputs
    }

    /// Returns the unweighted shared-expert W2 BF16 output.
    #[must_use]
    pub fn shared_output_bf16(&self) -> &[u16] {
        &self.shared_output
    }

    /// Returns the FP32 routed-plus-shared sum before the final BF16 narrowing.
    #[must_use]
    pub fn accumulator_f32(&self) -> &[f32] {
        &self.accumulator
    }

    /// Returns the final one-token BF16 output.
    #[must_use]
    pub fn output_bf16(&self) -> &[u16] {
        &self.output_bf16
    }
}

/// A malformed bounded `MoE` reference input or non-finite intermediate.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum MoEError {
    /// A routed-expert source could not supply the selected expert.
    #[error("routed expert {index} is unavailable: {reason}")]
    ExpertUnavailable {
        /// Original gate expert ID.
        index: usize,
        /// Source-specific reason, such as a missing or corrupt payload.
        reason: String,
    },
    /// A vector width was zero, not group-aligned, or beyond the bounded reference limit.
    #[error("{field} width {width} must be nonzero, group-aligned, and at most {max}")]
    InvalidWidth {
        /// Named width role.
        field: &'static str,
        /// Supplied width.
        width: usize,
        /// Maximum accepted width.
        max: usize,
    },
    /// The `SwiGLU` clamp must be finite and nonnegative; zero disables it.
    #[error("SwiGLU limit must be finite and nonnegative")]
    InvalidSwiGluLimit,
    /// `MoE` requires at least one selected routed expert.
    #[error("MoE Top-K must be nonzero")]
    InvalidTopK,
    /// Gate temperature must be finite and positive.
    #[error("MoE gate temperature must be finite and positive")]
    InvalidGateTemperature,
    /// Route scale must be finite and positive.
    #[error("MoE route scale must be finite and positive")]
    InvalidRouteScale,
    /// A direct routed-expert weight must be finite and nonnegative.
    #[error("FP4 expert route weight must be finite and nonnegative")]
    InvalidRouteWeight,
    /// The reference needs at least one routed expert.
    #[error("MoE requires at least one routed expert")]
    NoRoutedExperts,
    /// The requested selected count exceeds the provided routed experts.
    #[error("MoE Top-K {top_k} exceeds routed expert count {experts}")]
    TopKExceedsExperts {
        /// Requested selected experts.
        top_k: usize,
        /// Available routed experts.
        experts: usize,
    },
    /// An encoded buffer length differs from its exact row-major role.
    #[error("{field} length is {actual}, expected {expected}")]
    Length {
        /// Input role.
        field: &'static str,
        /// Actual elements.
        actual: usize,
        /// Required elements.
        expected: usize,
    },
    /// A supplied expert geometry differs from the reference configuration.
    #[error("expert geometry differs from the MoE configuration")]
    ExpertGeometry,
    /// Checked shape arithmetic overflowed.
    #[error("MoE {field} shape arithmetic overflowed")]
    ShapeOverflow {
        /// Shape role.
        field: &'static str,
    },
    /// Checked work arithmetic overflowed.
    #[error("MoE work arithmetic overflowed")]
    WorkOverflow,
    /// The bounded scalar reference would perform too much work.
    #[error("MoE work {elements} exceeds maximum {max}")]
    WorkTooLarge {
        /// Requested scalar work.
        elements: usize,
        /// Maximum scalar work.
        max: usize,
    },
    /// A fallible temporary allocation failed.
    #[error("could not allocate MoE {field}")]
    Allocation {
        /// Temporary buffer role.
        field: &'static str,
    },
    /// A selected route refers to an absent routed expert.
    #[error("selected route refers to an absent routed expert")]
    RouteOutOfRange,
    /// The sparse logical routed-expert table has no payload for a selected ID.
    #[error("selected routed expert {index} has no loaded FP4 payload")]
    MissingRoutedExpert {
        /// Logical gate/expert ID.
        index: usize,
    },
    /// A finite FP32 stage cannot be represented as finite BF16.
    #[error("{stage} cannot remain finite BF16 at element {element}")]
    Bf16Overflow {
        /// Stage name.
        stage: &'static str,
        /// Flat element index.
        element: usize,
    },
    /// A scalar stage overflowed or became non-finite.
    #[error("{stage} became non-finite at element {element}")]
    NonFinite {
        /// Stage name.
        stage: &'static str,
        /// Flat element index.
        element: usize,
    },
    /// Existing activation preparation rejected the BF16 row.
    #[error(transparent)]
    Activation(#[from] ActivationQuantError),
    /// Existing FP4 projection rejected its encoded buffers or arithmetic.
    #[error(transparent)]
    Fp4(#[from] Fp4LinearError),
    /// Existing FP8 projection rejected its encoded buffers or arithmetic.
    #[error(transparent)]
    Fp8(#[from] Fp8LinearError),
    /// Existing BF16 gate projection or routing rejected the token.
    #[error(transparent)]
    Routing(#[from] FlashGateProjectionError),
    /// Inside a device scope, a Metal FP8 linear or FP4 expert failed.
    #[cfg(feature = "metal")]
    #[error("Metal expert projection failed: {0}")]
    Device(crate::precision::Fp8MetalError),
}

impl From<Fp8ForwardError> for MoEError {
    fn from(error: Fp8ForwardError) -> Self {
        match error {
            Fp8ForwardError::Scalar(error) => Self::Fp8(error),
            #[cfg(feature = "metal")]
            Fp8ForwardError::Device(error) => Self::Device(error),
        }
    }
}

fn validate_width(field: &'static str, width: usize) -> Result<(), MoEError> {
    if width == 0 || !width.is_multiple_of(GROUP_WIDTH) || width > MAX_MOE_ELEMENTS {
        return Err(MoEError::InvalidWidth {
            field,
            width,
            max: MAX_MOE_ELEMENTS,
        });
    }
    Ok(())
}

fn validate_fp4_matrix(
    name: &'static str,
    codes: &[u8],
    scales: &[u8],
    rows: usize,
    reduction: usize,
) -> Result<(), MoEError> {
    let code_length = checked_product(name, rows, reduction / 2)?;
    let scale_length = checked_product(name, rows, reduction / GROUP_WIDTH)?;
    if codes.len() != code_length {
        return Err(MoEError::Length {
            field: name,
            actual: codes.len(),
            expected: code_length,
        });
    }
    if scales.len() != scale_length {
        return Err(MoEError::Length {
            field: "FP4 scales",
            actual: scales.len(),
            expected: scale_length,
        });
    }
    Ok(())
}

fn validate_fp8_matrix(
    name: &'static str,
    codes: &[u8],
    scales: &[u8],
    rows: usize,
    reduction: usize,
) -> Result<(), MoEError> {
    let code_length = checked_product(name, rows, reduction)?;
    let scale_rows = rows.div_ceil(GROUP_WIDTH);
    let scale_length = checked_product(name, scale_rows, reduction / GROUP_WIDTH)?;
    if codes.len() != code_length {
        return Err(MoEError::Length {
            field: name,
            actual: codes.len(),
            expected: code_length,
        });
    }
    if scales.len() != scale_length {
        return Err(MoEError::Length {
            field: "FP8 scales",
            actual: scales.len(),
            expected: scale_length,
        });
    }
    Ok(())
}

fn checked_product(field: &'static str, left: usize, right: usize) -> Result<usize, MoEError> {
    left.checked_mul(right)
        .ok_or(MoEError::ShapeOverflow { field })
}

fn validate_expert_work(hidden_width: usize, intermediate_width: usize) -> Result<(), MoEError> {
    let projection = checked_product("expert work", hidden_width, intermediate_width)?;
    let work = projection.checked_mul(3).ok_or(MoEError::WorkOverflow)?;
    if work > MAX_MOE_WORK {
        return Err(MoEError::WorkTooLarge {
            elements: work,
            max: MAX_MOE_WORK,
        });
    }
    Ok(())
}

fn allocate_u8(field: &'static str, length: usize) -> Result<Vec<u8>, MoEError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| MoEError::Allocation { field })?;
    result.resize(length, 0);
    Ok(result)
}

fn allocate_f32(field: &'static str, length: usize) -> Result<Vec<f32>, MoEError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(length)
        .map_err(|_| MoEError::Allocation { field })?;
    result.resize(length, 0.0);
    Ok(result)
}

fn fp4_projection(
    input: &[u16],
    output_width: usize,
    codes: &[u8],
    scales: &[u8],
) -> Result<Vec<u16>, MoEError> {
    let mut activation_codes = allocate_u8("FP4 activation codes", input.len())?;
    let mut activation_scales = allocate_u8("FP4 activation scales", input.len() / GROUP_WIDTH)?;
    quantize_bf16_activations_e4m3fn(
        input,
        1,
        input.len(),
        ActivationGroup::Elements32,
        &mut activation_codes,
        &mut activation_scales,
    )?;
    let projected = fp4_linear_runtime_f32_owned(
        &activation_codes,
        &activation_scales,
        codes,
        scales,
        1,
        input.len(),
        output_width,
        ActivationGroup::Elements32,
    )?;
    narrow_row(&projected, "FP4 linear output")
}

fn fp8_projection(
    input: &[u16],
    output_width: usize,
    codes: &[u8],
    scales: &[u8],
) -> Result<Vec<u16>, MoEError> {
    let mut activation_codes = allocate_u8("FP8 activation codes", input.len())?;
    let mut activation_scales = allocate_u8("FP8 activation scales", input.len() / GROUP_WIDTH)?;
    quantize_bf16_activations_e4m3fn(
        input,
        1,
        input.len(),
        ActivationGroup::Elements32,
        &mut activation_codes,
        &mut activation_scales,
    )?;
    let mut projected = allocate_f32("FP8 projection", output_width)?;
    fp8_linear_f32(
        &activation_codes,
        &activation_scales,
        codes,
        scales,
        1,
        input.len(),
        output_width,
        ActivationGroup::Elements32,
        &mut projected,
    )?;
    narrow_row(&projected, "FP8 linear output")
}

/// The routed expert: on the device inside a `DeviceLinears`
/// scope, scalar otherwise or when the device cannot run it.
fn project_fp4_expert(
    input: &[u16],
    expert: &Fp4ExpertWeights<'_>,
    swiglu_limit: f32,
    route_weight: Option<f32>,
) -> Result<Vec<u16>, MoEError> {
    #[cfg(feature = "metal")]
    if let Some(device) = crate::precision::active_device() {
        let mut codes = allocate_u8("FP4 expert activation codes", input.len())?;
        let mut scales = allocate_u8("FP4 expert activation scales", input.len() / GROUP_WIDTH)?;
        quantize_bf16_activations_e4m3fn(
            input,
            1,
            expert.hidden_width,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )?;
        match crate::precision::metal_fp4_expert(
            (&codes, &scales),
            expert.hidden_width,
            expert.intermediate_width,
            (expert.w1_codes, expert.w1_scales),
            (expert.w2_codes, expert.w2_scales),
            (expert.w3_codes, expert.w3_scales),
            swiglu_limit,
            route_weight,
        )
        .map_err(MoEError::Device)?
        {
            Some(output) => {
                device.count_fp4_expert();
                return Ok(output);
            }
            None => device.count_fallback(),
        }
    }
    project_fp4_expert_scalar(input, expert, swiglu_limit, route_weight)
}

/// The scalar routed expert: the reference the device path is checked against.
pub(crate) fn project_fp4_expert_scalar(
    input: &[u16],
    expert: &Fp4ExpertWeights<'_>,
    swiglu_limit: f32,
    route_weight: Option<f32>,
) -> Result<Vec<u16>, MoEError> {
    let gate = fp4_projection(
        input,
        expert.intermediate_width,
        expert.w1_codes,
        expert.w1_scales,
    )?;
    let up = fp4_projection(
        input,
        expert.intermediate_width,
        expert.w3_codes,
        expert.w3_scales,
    )?;
    let hidden = swiglu_hidden(&gate, &up, swiglu_limit, route_weight)?;
    fp4_projection(
        &hidden,
        expert.hidden_width,
        expert.w2_codes,
        expert.w2_scales,
    )
}

fn project_fp8_expert(
    input: &[u16],
    expert: &Fp8ExpertWeights<'_>,
    swiglu_limit: f32,
) -> Result<Vec<u16>, MoEError> {
    let gate = fp8_projection(
        input,
        expert.intermediate_width,
        expert.w1_codes,
        expert.w1_scales,
    )?;
    let up = fp8_projection(
        input,
        expert.intermediate_width,
        expert.w3_codes,
        expert.w3_scales,
    )?;
    let hidden = swiglu_hidden(&gate, &up, swiglu_limit, None)?;
    fp8_projection(
        &hidden,
        expert.hidden_width,
        expert.w2_codes,
        expert.w2_scales,
    )
}

fn swiglu_hidden(
    gate_bf16: &[u16],
    up_bf16: &[u16],
    swiglu_limit: f32,
    route_weight: Option<f32>,
) -> Result<Vec<u16>, MoEError> {
    if gate_bf16.len() != up_bf16.len() {
        return Err(MoEError::ExpertGeometry);
    }
    let mut hidden = allocate_f32("SwiGLU hidden", gate_bf16.len())?;
    for (index, ((destination, &gate_bits), &up_bits)) in
        hidden.iter_mut().zip(gate_bf16).zip(up_bf16).enumerate()
    {
        let mut gate = bf16_to_f32(gate_bits);
        let mut up = bf16_to_f32(up_bits);
        if !gate.is_finite() || !up.is_finite() {
            return Err(MoEError::NonFinite {
                stage: "SwiGLU input",
                element: index,
            });
        }
        if swiglu_limit > 0.0 {
            gate = gate.min(swiglu_limit);
            up = up.clamp(-swiglu_limit, swiglu_limit);
        }
        let mut value = (gate / (1.0 + (-gate).exp())) * up;
        if let Some(weight) = route_weight {
            value *= weight;
        }
        if !value.is_finite() {
            return Err(MoEError::NonFinite {
                stage: "route-weighted SwiGLU",
                element: index,
            });
        }
        *destination = value;
    }
    narrow_row(&hidden, "SwiGLU hidden")
}

fn accumulate_bf16(
    accumulator: &mut [f32],
    input: &[u16],
    stage: &'static str,
) -> Result<(), MoEError> {
    if accumulator.len() != input.len() {
        return Err(MoEError::ExpertGeometry);
    }
    for (index, (destination, &bits)) in accumulator.iter_mut().zip(input).enumerate() {
        *destination += bf16_to_f32(bits);
        if !destination.is_finite() {
            return Err(MoEError::NonFinite {
                stage,
                element: index,
            });
        }
    }
    Ok(())
}

fn narrow_row(input: &[f32], stage: &'static str) -> Result<Vec<u16>, MoEError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(input.len())
        .map_err(|_| MoEError::Allocation { field: stage })?;
    for (index, &value) in input.iter().enumerate() {
        if !value.is_finite() {
            return Err(MoEError::Bf16Overflow {
                stage,
                element: index,
            });
        }
        let bits = f32_to_bf16_rne(value);
        if !bf16_to_f32(bits).is_finite() {
            return Err(MoEError::Bf16Overflow {
                stage,
                element: index,
            });
        }
        result.push(bits);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::{
        Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEError, MoEReference, RoutedExpertSource,
        project_fp4_expert, swiglu_hidden,
    };

    const WIDTH: usize = 32;

    struct ExpertBuffers {
        w1: Vec<u8>,
        w1_scales: Vec<u8>,
        w2: Vec<u8>,
        w2_scales: Vec<u8>,
        w3: Vec<u8>,
        w3_scales: Vec<u8>,
    }

    fn fp4_buffers() -> ExpertBuffers {
        fp4_buffers_with_codes(0x11, 0x11, 0x11)
    }

    fn fp4_buffers_with_codes(w1_code: u8, w2_code: u8, w3_code: u8) -> ExpertBuffers {
        ExpertBuffers {
            w1: vec![w1_code; WIDTH * WIDTH / 2],
            w1_scales: vec![127; WIDTH],
            w2: vec![w2_code; WIDTH * WIDTH / 2],
            w2_scales: vec![127; WIDTH],
            w3: vec![w3_code; WIDTH * WIDTH / 2],
            w3_scales: vec![127; WIDTH],
        }
    }

    fn fp8_buffers() -> ExpertBuffers {
        ExpertBuffers {
            w1: vec![0x30; WIDTH * WIDTH],
            w1_scales: vec![127; 1],
            w2: vec![0x30; WIDTH * WIDTH],
            w2_scales: vec![127; 1],
            w3: vec![0x30; WIDTH * WIDTH],
            w3_scales: vec![127; 1],
        }
    }

    #[test]
    fn route_weight_precedes_bf16_narrowing_and_w2() {
        let buffers = fp4_buffers();
        let expert = Fp4ExpertWeights::new(
            WIDTH,
            WIDTH,
            &buffers.w1,
            &buffers.w1_scales,
            &buffers.w2,
            &buffers.w2_scales,
            &buffers.w3,
            &buffers.w3_scales,
        )
        .expect("complete packed FP4 expert");
        let input = [0x3f80; WIDTH];
        let first = project_fp4_expert(&input, &expert, 4.0, Some(0.3))
            .expect("finite source-order expert");
        let second = project_fp4_expert(&input, &expert, 4.0, Some(0.1995))
            .expect("finite source-order expert");
        assert_eq!(first, vec![0x4290; WIDTH]); // BF16 72.
        assert_eq!(second, vec![0x4250; WIDTH]); // BF16 52.
    }

    #[test]
    fn fp4_expert_leaf_matches_existing_composition_and_rejects_preflight_errors() {
        let buffers = fp4_buffers();
        let expert = Fp4ExpertWeights::new(
            WIDTH,
            WIDTH,
            &buffers.w1,
            &buffers.w1_scales,
            &buffers.w2,
            &buffers.w2_scales,
            &buffers.w3,
            &buffers.w3_scales,
        )
        .expect("complete packed FP4 expert");
        let input = [0x3f80; WIDTH];
        assert_eq!(
            expert
                .forward_token(&input, 4.0, Some(0.3))
                .expect("checked FP4 expert leaf"),
            project_fp4_expert(&input, &expert, 4.0, Some(0.3))
                .expect("existing FP4 expert composition"),
        );
        assert!(matches!(
            expert.forward_token(&input[..WIDTH - 1], 4.0, None),
            Err(MoEError::Length {
                field: "input_bf16",
                ..
            })
        ));
        let mut nonfinite = input;
        nonfinite[3] = 0x7f80;
        assert!(matches!(
            expert.forward_token(&nonfinite, 4.0, None),
            Err(MoEError::NonFinite {
                stage: "FP4 expert input",
                element: 3,
            })
        ));
        assert!(matches!(
            expert.forward_token(&input, f32::NAN, None),
            Err(MoEError::InvalidSwiGluLimit)
        ));
        assert!(matches!(
            expert.forward_token(&input, 4.0, Some(-0.1)),
            Err(MoEError::InvalidRouteWeight)
        ));
    }

    #[test]
    fn fp4_expert_leaf_rejects_work_before_encoded_projection_buffers() {
        const HIDDEN: usize = 5_120;
        const INTERMEDIATE: usize = 17_504;
        let expert = Fp4ExpertWeights {
            w1_codes: &[],
            w1_scales: &[],
            w2_codes: &[],
            w2_scales: &[],
            w3_codes: &[],
            w3_scales: &[],
            hidden_width: HIDDEN,
            intermediate_width: INTERMEDIATE,
        };
        let input = vec![0x3f80; HIDDEN];
        assert!(matches!(
            expert.forward_token(&input, 0.0, None),
            Err(MoEError::WorkTooLarge {
                elements,
                max: super::MAX_MOE_WORK,
            }) if elements == 3 * HIDDEN * INTERMEDIATE
        ));
    }

    fn fp4_expert(buffers: &ExpertBuffers) -> Fp4ExpertWeights<'_> {
        Fp4ExpertWeights::new(
            WIDTH,
            WIDTH,
            &buffers.w1,
            &buffers.w1_scales,
            &buffers.w2,
            &buffers.w2_scales,
            &buffers.w3,
            &buffers.w3_scales,
        )
        .expect("complete packed FP4 expert")
    }

    fn fp8_expert(buffers: &ExpertBuffers) -> Fp8ExpertWeights<'_> {
        Fp8ExpertWeights::new(
            WIDTH,
            WIDTH,
            &buffers.w1,
            &buffers.w1_scales,
            &buffers.w2,
            &buffers.w2_scales,
            &buffers.w3,
            &buffers.w3_scales,
        )
        .expect("complete packed FP8 expert")
    }

    #[test]
    fn sparse_experts_preserve_dense_ids_and_reject_only_selected_holes() {
        let routed_buffers = fp4_buffers();
        let expert = fp4_expert(&routed_buffers);
        let shared_buffers = fp8_buffers();
        let shared = fp8_expert(&shared_buffers);
        let config =
            MoEConfig::new(WIDTH, WIDTH, 4.0, 1, 1.0, true, 1.0).expect("one-route configuration");
        let gate = vec![0; 2 * WIDTH];
        let bias = [1.0, 0.0];
        let dense = [expert, expert];
        let sparse = [Some(expert), Some(expert)];
        let input = [0x3f80; WIDTH];
        assert_eq!(
            MoEReference::new(config, &gate, &bias, &dense, shared)
                .expect("dense reference")
                .forward_token(&input)
                .expect("dense token"),
            MoEReference::new_sparse(config, &gate, &bias, &sparse, shared)
                .expect("sparse reference")
                .forward_token(&input)
                .expect("sparse token"),
            "sparse storage retains the full logical gate table",
        );
        let unselected_hole = [Some(expert), None];
        assert!(
            MoEReference::new_sparse(config, &gate, &bias, &unselected_hole, shared)
                .expect("unselected sparse hole is valid")
                .forward_token(&input)
                .is_ok()
        );
        let selected_hole = [None, Some(expert)];
        assert!(matches!(
            MoEReference::new_sparse(config, &gate, &bias, &selected_hole, shared)
                .expect("selected-hole table has valid geometry")
                .forward_token(&input),
            Err(MoEError::MissingRoutedExpert { index: 0 })
        ));
    }

    #[test]
    fn sparse_missing_payload_follows_the_current_route_and_validates_present_geometry() {
        let routed_buffers = fp4_buffers();
        let expert = fp4_expert(&routed_buffers);
        let shared_buffers = fp8_buffers();
        let shared = fp8_expert(&shared_buffers);
        let config =
            MoEConfig::new(WIDTH, WIDTH, 4.0, 1, 1.0, true, 1.0).expect("one-route configuration");
        let mut gate = vec![0; 2 * WIDTH];
        gate[0] = 0x3f80;
        gate[WIDTH] = 0xbf80;
        let bias = [0.0, 0.0];
        let sparse = [Some(expert), None];
        let reference =
            MoEReference::new_sparse(config, &gate, &bias, &sparse, shared).expect("sparse table");
        let mut positive = [0; WIDTH];
        positive[0] = 0x3f80;
        assert!(reference.forward_token(&positive).is_ok());
        let mut negative = positive;
        negative[0] = 0xbf80;
        assert!(matches!(
            reference.forward_token(&negative),
            Err(MoEError::MissingRoutedExpert { index: 1 })
        ));
        let malformed = Fp4ExpertWeights {
            hidden_width: WIDTH * 2,
            ..expert
        };
        assert!(matches!(
            MoEReference::new_sparse(config, &gate, &bias, &[Some(malformed), None], shared),
            Err(MoEError::ExpertGeometry)
        ));
    }

    /// Lends experts from a borrowed table, like a cache would from fetched bytes.
    struct TableSource<'a>(&'a [Option<Fp4ExpertWeights<'a>>]);

    impl RoutedExpertSource for TableSource<'_> {
        fn with_expert(
            &self,
            index: usize,
            run: &mut dyn FnMut(Fp4ExpertWeights<'_>) -> Result<Vec<u16>, MoEError>,
        ) -> Result<Vec<u16>, MoEError> {
            match self.0.get(index) {
                Some(Some(expert)) => run(*expert),
                _ => Err(MoEError::ExpertUnavailable {
                    index,
                    reason: String::from("not in table"),
                }),
            }
        }
    }

    #[test]
    fn supplied_experts_reproduce_the_table_path_without_consulting_the_table() {
        let a = fp4_buffers_with_codes(0x11, 0x22, 0x13);
        let b = fp4_buffers_with_codes(0x23, 0x11, 0x21);
        let table = [Some(fp4_expert(&a)), Some(fp4_expert(&b))];
        let shared_buffers = fp8_buffers();
        let shared = fp8_expert(&shared_buffers);
        let config = MoEConfig::new(WIDTH, WIDTH, 4.0, 2, 1.0, true, 1.5).expect("two routes");
        let mut gate = vec![0; 2 * WIDTH];
        gate[0] = 0x3f80;
        gate[WIDTH + 1] = 0x3f00;
        let bias = [0.0, 0.0];
        let with_table =
            MoEReference::new_sparse(config, &gate, &bias, &table, shared).expect("table");
        let empty = [None, None];
        let without_table =
            MoEReference::new_sparse(config, &gate, &bias, &empty, shared).expect("empty table");
        for first in [0x3f80_u16, 0xbf80, 0x4040] {
            let mut input = [0x3e00_u16; WIDTH];
            input[0] = first;
            let expected = with_table.forward_token(&input).expect("table path");
            let supplied = without_table
                .forward_token_with(&input, &TableSource(&table))
                .expect("supplied path");
            assert_eq!(supplied, expected);
        }
        // The empty table alone cannot run the same token.
        assert!(matches!(
            without_table.forward_token(&[0x3e00; WIDTH]),
            Err(MoEError::MissingRoutedExpert { .. })
        ));
    }

    #[test]
    fn supplied_experts_fail_closed_on_absence_and_geometry() {
        let buffers = fp4_buffers();
        let expert = fp4_expert(&buffers);
        let shared_buffers = fp8_buffers();
        let shared = fp8_expert(&shared_buffers);
        let config = MoEConfig::new(WIDTH, WIDTH, 4.0, 1, 1.0, true, 1.0).expect("one route");
        let mut gate = vec![0; 2 * WIDTH];
        gate[0] = 0x3f80;
        let bias = [0.0, 0.0];
        let empty = [None, None];
        let reference =
            MoEReference::new_sparse(config, &gate, &bias, &empty, shared).expect("empty table");
        let mut input = [0; WIDTH];
        input[0] = 0x3f80;
        assert!(matches!(
            reference.forward_token_with(&input, &TableSource(&[None, None])),
            Err(MoEError::ExpertUnavailable { index: 0, .. })
        ));
        let wide = Fp4ExpertWeights {
            hidden_width: WIDTH * 2,
            ..expert
        };
        assert!(matches!(
            reference.forward_token_with(&input, &TableSource(&[Some(wide), None])),
            Err(MoEError::ExpertGeometry)
        ));
    }

    #[test]
    fn sparse_selected_id_survives_preceding_holes() {
        let first_buffers = fp4_buffers_with_codes(0x11, 0x11, 0x11);
        let selected_buffers = fp4_buffers_with_codes(0x22, 0x22, 0x22);
        let first = fp4_expert(&first_buffers);
        let selected = fp4_expert(&selected_buffers);
        let shared_buffers = fp8_buffers();
        let shared = fp8_expert(&shared_buffers);
        let config = MoEConfig::new(WIDTH, WIDTH, 4.0, 1, 1.0, true, 1.0).unwrap();
        let gate = vec![0; 2 * WIDTH];
        let bias = [0.0, 1.0];
        let input = [0x3f80; WIDTH];
        assert_ne!(
            first.forward_token(&input, 4.0, Some(1.0)).unwrap(),
            selected.forward_token(&input, 4.0, Some(1.0)).unwrap(),
            "distinct expert payloads must produce distinguishable outputs",
        );
        let dense = [first, selected];
        let sparse = [None, Some(selected)];
        let expected = MoEReference::new(config, &gate, &bias, &dense, shared)
            .unwrap()
            .forward_token(&input)
            .unwrap();
        let actual = MoEReference::new_sparse(config, &gate, &bias, &sparse, shared)
            .unwrap()
            .forward_token(&input)
            .unwrap();
        assert_eq!(actual.routes()[0].expert_index(), 1);
        assert_eq!(actual, expected);
    }

    #[test]
    fn sparse_missing_later_payload_preempts_earlier_expert_arithmetic() {
        let mut invalid_buffers = fp4_buffers();
        invalid_buffers.w1_scales[0] = 0xff;
        let invalid = fp4_expert(&invalid_buffers);
        let shared_buffers = fp8_buffers();
        let shared = fp8_expert(&shared_buffers);
        let config = MoEConfig::new(WIDTH, WIDTH, 4.0, 2, 1.0, true, 1.0).unwrap();
        let gate = vec![0; 2 * WIDTH];
        let bias = [1.0, 0.0];
        let input = [0x3f80; WIDTH];
        assert!(invalid.forward_token(&input, 4.0, Some(0.5)).is_err());
        let sparse = [Some(invalid), None];
        assert!(matches!(
            MoEReference::new_sparse(config, &gate, &bias, &sparse, shared)
                .unwrap()
                .forward_token(&input),
            Err(MoEError::MissingRoutedExpert { index: 1 })
        ));
    }

    #[test]
    fn shared_expert_is_added_once_without_route_weight() {
        let routed_buffers = fp4_buffers();
        let routed = [Fp4ExpertWeights::new(
            WIDTH,
            WIDTH,
            &routed_buffers.w1,
            &routed_buffers.w1_scales,
            &routed_buffers.w2,
            &routed_buffers.w2_scales,
            &routed_buffers.w3,
            &routed_buffers.w3_scales,
        )
        .expect("routed expert")];
        let shared_buffers = fp8_buffers();
        let shared = Fp8ExpertWeights::new(
            WIDTH,
            WIDTH,
            &shared_buffers.w1,
            &shared_buffers.w1_scales,
            &shared_buffers.w2,
            &shared_buffers.w2_scales,
            &shared_buffers.w3,
            &shared_buffers.w3_scales,
        )
        .expect("shared expert");
        let config =
            MoEConfig::new(WIDTH, WIDTH, 4.0, 1, 1.0, true, 0.3).expect("finite configuration");
        let gate = vec![0x0000; WIDTH];
        let reference = MoEReference::new(config, &gate, &[0.0], &routed, shared)
            .expect("complete one-expert reference");
        let output = reference
            .forward_token(&[0x3f80; WIDTH])
            .expect("finite one-token MoE");
        assert!(matches!(
            reference.forward_token(&[0x3f80; WIDTH - 1]),
            Err(MoEError::Length {
                field: "input_bf16",
                ..
            })
        ));
        assert_eq!(output.routes().len(), 1);
        assert_ne!(
            output.shared_output_bf16(),
            output.selected_outputs_bf16()[0]
        );
        for ((&sum, &selected), &shared) in output
            .accumulator_f32()
            .iter()
            .zip(output.selected_outputs_bf16()[0].iter())
            .zip(output.shared_output_bf16())
        {
            assert_eq!(
                sum.to_bits(),
                (super::bf16_to_f32(selected) + super::bf16_to_f32(shared)).to_bits()
            );
        }
    }

    #[test]
    fn rejects_malformed_weights_input_configuration_and_overflow() {
        assert!(matches!(
            MoEConfig::new(31, WIDTH, 0.0, 1, 1.0, false, 1.0),
            Err(MoEError::InvalidWidth { .. })
        ));
        assert!(matches!(
            Fp4ExpertWeights::new(WIDTH, WIDTH, &[], &[], &[], &[], &[], &[]),
            Err(MoEError::Length { .. })
        ));
        assert!(matches!(
            swiglu_hidden(&[0x7f7f], &[0x7f7f], 0.0, None),
            Err(MoEError::NonFinite { .. })
        ));
    }

    #[test]
    fn bf16_narrowing_accepts_neighbours_that_round_back_to_finite_maximum() {
        for (next, midpoint, expected) in [
            (0x7f7f_0001, 0x7f7f_8000, 0x7f7f),
            (0xff7f_0001, 0xff7f_8000, 0xff7f),
        ] {
            assert_eq!(
                super::narrow_row(&[f32::from_bits(next)], "boundary")
                    .expect("neighbour rounds to finite BF16"),
                vec![expected]
            );
            assert!(matches!(
                super::narrow_row(&[f32::from_bits(midpoint)], "boundary"),
                Err(MoEError::Bf16Overflow { .. })
            ));
        }
    }

    #[test]
    fn actual_swiglu_path_keeps_gate_lower_unclamped_and_bounds_up_both_sides() {
        let negative_gate =
            swiglu_hidden(&[0xc180], &[0x4180], 4.0, None).expect("finite clamped SwiGLU");
        let gate_unclamped = -16.0 / (1.0 + 16.0_f32.exp()) * 4.0;
        let gate_symmetric = -4.0 / (1.0 + 4.0_f32.exp()) * 4.0;
        assert_eq!(
            negative_gate,
            super::narrow_row(&[gate_unclamped], "expected").expect("finite expected")
        );
        assert_ne!(
            negative_gate,
            super::narrow_row(&[gate_symmetric], "wrong").expect("finite wrong")
        );

        let negative_up =
            swiglu_hidden(&[0x4080], &[0xc180], 4.0, None).expect("finite clamped SwiGLU");
        let up_bounded = (4.0 / (1.0 + (-4.0_f32).exp())) * -4.0;
        let up_unbounded = (4.0 / (1.0 + (-4.0_f32).exp())) * -16.0;
        assert_eq!(
            negative_up,
            super::narrow_row(&[up_bounded], "expected").expect("finite expected")
        );
        assert_ne!(
            negative_up,
            super::narrow_row(&[up_unbounded], "wrong").expect("finite wrong")
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]
        /// The up branch is odd through its symmetric clamp. This deliberately
        /// keeps the gate fixed, because the source clamps that branch only above.
        #[test]
        fn swiglu_up_branch_is_odd(
            gate_quarters in -32_i16..=32,
            up_quarters in -32_i16..=32,
            limit_halves in 0_u8..=8,
            route_tenths in 1_u8..=10,
        ) {
            let gate = f32::from(gate_quarters) / 4.0;
            let up = f32::from(up_quarters) / 4.0;
            let limit = f32::from(limit_halves) / 2.0;
            let weight = f32::from(route_tenths) / 10.0;
            let gate_bits = super::f32_to_bf16_rne(gate);
            let up_bits = super::f32_to_bf16_rne(up);
            let positive = swiglu_hidden(&[gate_bits], &[up_bits], limit, Some(weight))
                .expect("bounded finite property input");
            let negative = swiglu_hidden(&[gate_bits], &[up_bits ^ 0x8000], limit, Some(weight))
                .expect("bounded finite sign-flipped property input");
            prop_assert_eq!(negative[0] & 0x7fff, positive[0] & 0x7fff);
            if positive[0] & 0x7fff != 0 {
                prop_assert_eq!(negative[0], positive[0] ^ 0x8000);
            }
        }

        /// Renaming two selected routed experts together with their gate rows
        /// and biases cannot change the final sum. The source-order accumulator
        /// is intentionally not generalized to three terms, where FP32
        /// reassociation is an invalid invariant.
        #[test]
        fn two_expert_renaming_preserves_complete_moe(
            input_quarters in -8_i16..=8,
            gate0_quarters in -8_i16..=8,
            gate1_quarters in -8_i16..=8,
            bias0_quarters in -8_i16..=8,
            bias1_quarters in -8_i16..=8,
            expert0_codes in (0_u8..16, 0_u8..16, 0_u8..16),
            expert1_codes in (0_u8..16, 0_u8..16, 0_u8..16),
        ) {
            prop_assume!(input_quarters != 0);
            prop_assume!(expert0_codes != expert1_codes);
            let input_bits = super::f32_to_bf16_rne(f32::from(input_quarters) / 4.0);
            let input = vec![input_bits; WIDTH];
            let expert0_buffers = fp4_buffers_with_codes(
                expert0_codes.0,
                expert0_codes.1,
                expert0_codes.2,
            );
            let expert1_buffers = fp4_buffers_with_codes(
                expert1_codes.0,
                expert1_codes.1,
                expert1_codes.2,
            );
            let expert0 = Fp4ExpertWeights::new(
                WIDTH, WIDTH,
                &expert0_buffers.w1, &expert0_buffers.w1_scales,
                &expert0_buffers.w2, &expert0_buffers.w2_scales,
                &expert0_buffers.w3, &expert0_buffers.w3_scales,
            ).expect("generated packed expert zero");
            let expert1 = Fp4ExpertWeights::new(
                WIDTH, WIDTH,
                &expert1_buffers.w1, &expert1_buffers.w1_scales,
                &expert1_buffers.w2, &expert1_buffers.w2_scales,
                &expert1_buffers.w3, &expert1_buffers.w3_scales,
            ).expect("generated packed expert one");
            let gate0 = super::f32_to_bf16_rne(f32::from(gate0_quarters) / 4.0);
            let gate1 = super::f32_to_bf16_rne(f32::from(gate1_quarters) / 4.0);
            let mut gate_rows = vec![0_u16; 2 * WIDTH];
            gate_rows[0] = gate0;
            gate_rows[WIDTH] = gate1;
            let bias0 = f32::from(bias0_quarters) / 4.0;
            let bias1 = f32::from(bias1_quarters) / 4.0;
            let config = MoEConfig::new(WIDTH, WIDTH, 4.0, 2, 1.0, true, 1.0)
                .expect("fixed finite two-expert configuration");
            let shared_buffers = fp8_buffers();
            let shared = Fp8ExpertWeights::new(
                WIDTH, WIDTH,
                &shared_buffers.w1, &shared_buffers.w1_scales,
                &shared_buffers.w2, &shared_buffers.w2_scales,
                &shared_buffers.w3, &shared_buffers.w3_scales,
            ).expect("fixed shared expert");
            let original_experts = [expert0, expert1];
            let original = MoEReference::new(
                config, &gate_rows, &[bias0, bias1], &original_experts, shared,
            ).expect("original reference").forward_token(&input)
                .expect("original finite execution");
            let mut renamed_gate_rows = vec![0_u16; 2 * WIDTH];
            renamed_gate_rows[0] = gate1;
            renamed_gate_rows[WIDTH] = gate0;
            let renamed_experts = [expert1, expert0];
            let renamed = MoEReference::new(
                config, &renamed_gate_rows, &[bias1, bias0], &renamed_experts, shared,
            ).expect("renamed reference").forward_token(&input)
                .expect("renamed finite execution");
            prop_assert_eq!(original.output_bf16(), renamed.output_bf16());
            prop_assert_eq!(original.accumulator_f32(), renamed.accumulator_f32());
            prop_assert_eq!(original.routes().len(), 2);
            prop_assert_eq!(renamed.routes().len(), 2);
            prop_assert_eq!(original.routes()[0].weight().to_bits(), renamed.routes()[1].weight().to_bits());
            prop_assert_eq!(original.routes()[1].weight().to_bits(), renamed.routes()[0].weight().to_bits());
            prop_assert_eq!(&original.selected_outputs_bf16()[0], &renamed.selected_outputs_bf16()[1]);
            prop_assert_eq!(&original.selected_outputs_bf16()[1], &renamed.selected_outputs_bf16()[0]);
        }
    }
}
