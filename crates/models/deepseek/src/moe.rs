//! Bounded one-token reference composition for the V4.1 text `MoE`.
//!
//! This preserves the pinned routed and shared expert order over encoded
//! FP4/FP8 weights. It is a scalar CPU reference for reduced-graph comparison,
//! not a scheduler, checkpoint loader, production serving path, or hardware
//! reduction-parity claim.

use thiserror::Error;

use crate::{
    precision::{
        ActivationGroup, ActivationQuantError, Fp4LinearError, Fp8LinearError, bf16_to_f32,
        f32_to_bf16_rne, fp4_linear_runtime_f32, fp8_linear_runtime_f32,
        quantize_bf16_activations_e4m3fn,
    },
    routing::{ExpertRoute, FlashGateProjectionError, flash_bf16_gate_routes},
};

const GROUP_WIDTH: usize = 32;
/// Largest accepted per-token vector in this bounded reference.
pub const MAX_MOE_ELEMENTS: usize = 1 << 22;
/// Largest logical projection-term count across selected and shared experts.
///
/// Scalar linear leaves perform validation and write passes, so this is not an
/// exact instruction count. Gate projection and activation preparation add work.
pub const MAX_MOE_WORK: usize = 1 << 28;

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
    routed: &'a [Fp4ExpertWeights<'a>],
    shared: Fp8ExpertWeights<'a>,
}

impl<'a> MoEReference<'a> {
    /// Returns the validated one-token hidden width.
    #[must_use]
    pub const fn hidden_width(&self) -> usize {
        self.config.hidden_width
    }

    /// Validates geometry, buffer lengths and the logical expert-work budget.
    /// Numerical gate validation occurs in [`Self::forward_token`].
    pub fn new(
        config: MoEConfig,
        gate_bf16: &'a [u16],
        bias: &'a [f32],
        routed: &'a [Fp4ExpertWeights<'a>],
        shared: Fp8ExpertWeights<'a>,
    ) -> Result<Self, MoEError> {
        if routed.is_empty() {
            return Err(MoEError::NoRoutedExperts);
        }
        if config.top_k > routed.len() {
            return Err(MoEError::TopKExceedsExperts {
                top_k: config.top_k,
                experts: routed.len(),
            });
        }
        let expected_gate = checked_product("gate", routed.len(), config.hidden_width)?;
        if gate_bf16.len() != expected_gate {
            return Err(MoEError::Length {
                field: "gate_bf16",
                actual: gate_bf16.len(),
                expected: expected_gate,
            });
        }
        if bias.len() != routed.len() {
            return Err(MoEError::Length {
                field: "bias",
                actual: bias.len(),
                expected: routed.len(),
            });
        }
        if routed.iter().any(|expert| {
            expert.hidden_width != config.hidden_width
                || expert.intermediate_width != config.intermediate_width
        }) {
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
    pub fn forward_token(&self, input_bf16: &[u16]) -> Result<MoEDiagnostic, MoEError> {
        if input_bf16.len() != self.config.hidden_width {
            return Err(MoEError::Length {
                field: "input_bf16",
                actual: input_bf16.len(),
                expected: self.config.hidden_width,
            });
        }
        let routes = flash_bf16_gate_routes(
            input_bf16,
            self.gate_bf16,
            self.routed.len(),
            self.config.hidden_width,
            self.bias,
            self.config.top_k,
            self.config.gate_temperature,
            self.config.normalize_top_k,
            self.config.route_scale,
        )?;
        let mut accumulator = allocate_f32("accumulator", self.config.hidden_width)?;
        let mut selected_outputs = Vec::new();
        selected_outputs
            .try_reserve_exact(routes.len())
            .map_err(|_| MoEError::Allocation {
                field: "selected outputs",
            })?;
        for route in &routes {
            let expert = self
                .routed
                .get(route.expert_index())
                .ok_or(MoEError::RouteOutOfRange)?;
            let output = project_fp4_expert(
                input_bf16,
                expert,
                self.config.swiglu_limit,
                Some(route.weight()),
            )?;
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
    let mut projected = allocate_f32("FP4 projection", output_width)?;
    fp4_linear_runtime_f32(
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
    fp8_linear_runtime_f32(
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

fn project_fp4_expert(
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
        Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEError, MoEReference, project_fp4_expert,
        swiglu_hidden,
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
