//! MLX graphs for [`route`] and [`routed_experts`]. Nothing here evaluates.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use mlx_rs::{
    Array, Dtype, StreamOrDevice,
    ops::{
        self,
        indexing::{Ellipsis, IndexOp},
    },
};

use super::{Activation, MoeError, Router, SORT_THRESHOLD};

/// The experts each token is sent to and their weights, both `[.., top_k]`.
///
/// `chosen` indexes axis 0 of whatever expert stack [`routed_experts`] is
/// given, so a caller may remap it onto a compacted stack. `weights` are f32.
pub struct Routing {
    /// Expert indices, `u32`, largest weight first.
    pub chosen: Array,
    /// Mixing weights, f32.
    pub weights: Array,
}

/// A stack of matrices in MLX affine quantization: `bits`-wide codes packed
/// into `u32` words along the input axis, with per-group `scales` and
/// `biases` (the dequantization offsets, not an output bias).
pub struct AffineMatrix {
    /// `[E, out, in * bits / 32]`, `u32`.
    pub codes: Array,
    /// `[E, out, in / group_size]`.
    pub scales: Array,
    /// `[E, out, in / group_size]`.
    pub biases: Array,
    /// Inputs per scale.
    pub group_size: i32,
    /// Bits per code.
    pub bits: i32,
}

/// One stacked expert projection, `[E, out, in]` as stored.
#[derive(Clone, Copy)]
pub enum Matrix<'a> {
    /// A float stack.
    Dense(&'a Array),
    /// An MLX affine-quantized stack.
    Affine(&'a AffineMatrix),
}

impl Matrix<'_> {
    /// `(experts, out, in)` of the logical float stack.
    fn shape(self) -> Result<(i32, i32, i32), MoeError> {
        let (shape, inputs) = match self {
            Self::Dense(weight) => (weight.shape(), weight.shape().get(2).copied()),
            Self::Affine(weight) => (
                weight.codes.shape(),
                weight
                    .scales
                    .shape()
                    .get(2)
                    .map(|groups| groups * weight.group_size),
            ),
        };
        match (shape, inputs) {
            ([experts, out, _], Some(inputs)) => Ok((*experts, *out, inputs)),
            _ => Err(MoeError::Shape(format!(
                "expert stack has shape {shape:?}, not [experts, out, in]"
            ))),
        }
    }

    /// `input` times the transpose of the stacked matrix each of `indices`
    /// selects.
    fn gather(self, input: &Array, indices: &Array, sorted: bool) -> Result<Array, MoeError> {
        let gpu = StreamOrDevice::gpu();
        match self {
            Self::Dense(weight) => Ok(ops::gather_mm_device(
                input,
                weight.swap_axes_device(-1, -2, &gpu)?,
                None,
                indices,
                sorted,
                &gpu,
            )?),
            Self::Affine(weight) => Ok(ops::gather_qmm_device(
                input,
                &weight.codes,
                &weight.scales,
                &weight.biases,
                None,
                indices,
                true,
                weight.group_size,
                weight.bits,
                sorted,
                &gpu,
            )?),
        }
    }
}

/// One layer's routed experts.
pub enum Experts<'a> {
    /// Hugging Face's layout: `gate_up` `[E, 2I, H]` (gate rows, then up
    /// rows) and `down` `[E, H, I]`.
    Fused {
        /// Gate and up projections, stacked.
        gate_up: Matrix<'a>,
        /// Down projection.
        down: Matrix<'a>,
    },
    /// Separate stacks: `gate` and `up` `[E, I, H]`, `down` `[E, H, I]`.
    Split {
        /// Gate projection.
        gate: Matrix<'a>,
        /// Up projection.
        up: Matrix<'a>,
        /// Down projection.
        down: Matrix<'a>,
    },
}

impl Experts<'_> {
    /// `(experts, hidden, intermediate)`, checked across the stacks.
    fn geometry(&self) -> Result<(i32, i32, i32), MoeError> {
        let (gate, gate_rows, down) = match self {
            Self::Fused { gate_up, down } => {
                let shape = gate_up.shape()?;
                (shape, shape.1 / 2, *down)
            }
            Self::Split { gate, up, down } => {
                let shape = gate.shape()?;
                if up.shape()? != shape {
                    return Err(MoeError::Shape(format!(
                        "up stack {:?} differs from gate stack {shape:?}",
                        up.shape()?
                    )));
                }
                (shape, shape.1, *down)
            }
        };
        let (experts, _, hidden) = gate;
        let down = down.shape()?;
        let fused_even = !matches!(self, Self::Fused { .. }) || gate.1 % 2 == 0;
        if !fused_even || down != (experts, hidden, gate_rows) {
            return Err(MoeError::Shape(format!(
                "down stack {down:?} does not invert gate stack {gate:?}"
            )));
        }
        Ok((experts, hidden, gate_rows))
    }
}

/// Chooses each token's `top_k` experts from router logits `[.., E]`.
///
/// Softmax over all experts in f32, then the `top_k` largest, then (if the
/// router says so) division by their sum, then (if the router says so)
/// multiplication by `per_expert_scale[chosen]`. Ties at the top-k boundary
/// keep the lower expert index. Nothing is evaluated.
///
/// # Errors
///
/// [`MoeError::TopK`] when `top_k` exceeds the expert count; a missing,
/// unexpected or misshapen per-expert scale; or an MLX failure.
pub fn route(
    logits: &Array,
    router: &Router,
    per_expert_scale: Option<&Array>,
) -> Result<Routing, MoeError> {
    let gpu = StreamOrDevice::gpu();
    let experts = logits.shape().last().copied().unwrap_or(0);
    let top_k = i32::try_from(router.top_k()).unwrap_or(i32::MAX);
    if top_k > experts {
        return Err(MoeError::TopK {
            top_k: router.top_k(),
            experts: usize::try_from(experts).ok(),
        });
    }
    let scale = match (router.per_expert_scale(), per_expert_scale) {
        (true, Some(scale)) if scale.shape() == [experts] => Some(scale),
        (true, Some(scale)) => {
            return Err(MoeError::Shape(format!(
                "per-expert scale has shape {:?}, not [{experts}]",
                scale.shape()
            )));
        }
        (true, None) => return Err(MoeError::MissingPerExpertScale),
        (false, Some(_)) => {
            return Err(MoeError::Shape(
                "router takes no per-expert scale".to_owned(),
            ));
        }
        (false, None) => None,
    };
    let probabilities = ops::softmax_axis_device(
        logits.as_dtype_device(Dtype::Float32, &gpu)?,
        -1,
        true,
        &gpu,
    )?;
    // MLX's argsort is stable, so sorting the negated probabilities puts the
    // largest first and, among exactly equal ones, the lower index first:
    // the expert torch.topk keeps on CPU. argpartition would keep the higher.
    let order = ops::argsort_axis_device(probabilities.negative_device(&gpu)?, -1, &gpu)?;
    let chosen = order.index((Ellipsis, ..top_k));
    let mut weights = probabilities.take_along_axis_device(&chosen, -1, &gpu)?;
    if router.renormalize() {
        weights = weights.divide_device(weights.sum_axis_device(-1, true, &gpu)?, &gpu)?;
    }
    if let Some(scale) = scale {
        let scale = scale.as_dtype_device(Dtype::Float32, &gpu)?;
        weights = weights.multiply_device(scale.take_axis_device(&chosen, 0, &gpu)?, &gpu)?;
    }
    Ok(Routing { chosen, weights })
}

/// The weighted sum of each token's chosen experts over `input` `[n, H]`.
///
/// Each chosen expert applies `down(act(gate(x)) * up(x))`; the outputs are
/// multiplied by the routing weights and summed. With `SORT_THRESHOLD` or
/// more selections the rows are sorted by expert first. The result has the
/// input's dtype.
///
/// # Errors
///
/// [`MoeError::Shape`] when the stacks, input and routing do not compose;
/// or an MLX failure.
pub fn routed_experts(
    input: &Array,
    experts: &Experts<'_>,
    routing: &Routing,
    activation: Activation,
) -> Result<Array, MoeError> {
    let sorted = routing.chosen.size() >= SORT_THRESHOLD;
    run_experts(input, experts, routing, activation, sorted)
}

pub(super) fn run_experts(
    input: &Array,
    experts: &Experts<'_>,
    routing: &Routing,
    activation: Activation,
    sorted: bool,
) -> Result<Array, MoeError> {
    let gpu = StreamOrDevice::gpu();
    let (_, hidden, intermediate) = experts.geometry()?;
    let (tokens, top_k) = match (input.shape(), routing.chosen.shape()) {
        ([tokens, width], [chosen_tokens, top_k])
            if *width == hidden
                && chosen_tokens == tokens
                && routing.weights.shape() == routing.chosen.shape() =>
        {
            (*tokens, *top_k)
        }
        (input_shape, chosen_shape) => {
            return Err(MoeError::Shape(format!(
                "input {input_shape:?}, chosen {chosen_shape:?} and weights {:?} \
                 do not match [n, {hidden}], [n, k], [n, k]",
                routing.weights.shape()
            )));
        }
    };

    // Each (token, slot) is one row `[1, H]`: `[n, 1, 1, H]` broadcast over k.
    let rows = input.expand_dims_axes_device(&[-2, -3], &gpu)?;
    let (rows, chosen, restore) = if sorted {
        // Row `i` of the flattened selections belongs to token `i / top_k`.
        let flat = routing.chosen.reshape_device(&[-1], &gpu)?;
        let order = ops::argsort_axis_device(&flat, 0, &gpu)?;
        let restore = ops::argsort_axis_device(&order, 0, &gpu)?;
        let rows = rows
            .reshape_device(&[-1, 1, hidden], &gpu)?
            .take_axis_device(
                order.floor_divide_device(Array::from_int(top_k), &gpu)?,
                0,
                &gpu,
            )?;
        (rows, flat.take_axis_device(&order, 0, &gpu)?, Some(restore))
    } else {
        (rows, routing.chosen.clone(), None)
    };

    let (gate, up) = match experts {
        Experts::Fused { gate_up, .. } => {
            let projected = gate_up.gather(&rows, &chosen, sorted)?;
            let mut halves = ops::split_sections_device(&projected, &[intermediate], -1, &gpu)?;
            let up = halves
                .pop()
                .ok_or_else(|| MoeError::Shape("empty split".to_owned()))?;
            let gate = halves
                .pop()
                .ok_or_else(|| MoeError::Shape("empty split".to_owned()))?;
            (gate, up)
        }
        Experts::Split { gate, up, .. } => (
            gate.gather(&rows, &chosen, sorted)?,
            up.gather(&rows, &chosen, sorted)?,
        ),
    };
    let hidden_states = activate(&gate, activation)?.multiply_device(&up, &gpu)?;
    let down = match experts {
        Experts::Fused { down, .. } | Experts::Split { down, .. } => *down,
    };
    let mut outputs = down.gather(&hidden_states, &chosen, sorted)?;
    if let Some(restore) = restore {
        outputs = outputs
            .take_axis_device(&restore, 0, &gpu)?
            .reshape_device(&[tokens, top_k, 1, hidden], &gpu)?;
    }
    let weights = routing
        .weights
        .as_dtype_device(outputs.dtype(), &gpu)?
        .expand_dims_device(-1, &gpu)?;
    Ok(outputs
        .squeeze_axes_device(&[-2], &gpu)?
        .multiply_device(&weights, &gpu)?
        .sum_axis_device(-2, false, &gpu)?
        .as_dtype_device(input.dtype(), &gpu)?)
}

fn activate(x: &Array, activation: Activation) -> Result<Array, MoeError> {
    let gpu = StreamOrDevice::gpu();
    match activation {
        Activation::Silu => Ok(ops::sigmoid_device(x, &gpu)?.multiply_device(x, &gpu)?),
        Activation::GeluTanh => {
            // 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))), as gemma's
            // dense MLP computes it.
            let dtype = x.dtype();
            let scalar = |value: f32| Array::from_f32(value).as_dtype_device(dtype, &gpu);
            let cubic = x
                .multiply_device(x, &gpu)?
                .multiply_device(x, &gpu)?
                .multiply_device(scalar(0.044_715)?, &gpu)?;
            let inner = x
                .add_device(&cubic, &gpu)?
                .multiply_device(scalar((2.0 / std::f32::consts::PI).sqrt())?, &gpu)?;
            let gate = ops::tanh_device(&inner, &gpu)?.add_device(scalar(1.0)?, &gpu)?;
            Ok(x.multiply_device(scalar(0.5)?, &gpu)?
                .multiply_device(&gate, &gpu)?)
        }
    }
}

#[cfg(test)]
mod tests;
