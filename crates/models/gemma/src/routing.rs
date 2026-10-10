//! Gemma 4 `MoE` geometry and router.
//!
//! The router follows `Gemma4TextRouter` in Transformers commit
//! `a906d3c4b65095f2308b6a6a193e934d03b8eb5d`: unweighted RMS normalization of
//! the raw residual, learned input scaling, `H^-0.5`, a projection to expert
//! logits, then `layers::moe::route` (softmax in f32, top-k, renormalization,
//! learned per-expert scaling).

use thiserror::Error;

/// Validated Gemma expert and router dimensions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gemma4MoeGeometry {
    hidden: usize,
    experts: usize,
    top_k: usize,
    intermediate: usize,
}

impl Gemma4MoeGeometry {
    /// Checks nonzero widths, `1 <= top_k <= experts`, and tensor products.
    ///
    /// # Errors
    ///
    /// Returns an error for zero dimensions, an invalid selected expert count,
    /// or a router/expert tensor element count that overflows `usize`.
    pub fn new(
        hidden: usize,
        experts: usize,
        top_k: usize,
        intermediate: usize,
    ) -> Result<Self, Gemma4RouterError> {
        if hidden == 0 || experts == 0 || intermediate == 0 {
            return Err(Gemma4RouterError::ZeroDimension);
        }
        if top_k == 0 || top_k > experts {
            return Err(Gemma4RouterError::InvalidTopK);
        }
        experts
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(intermediate))
            .and_then(|n| n.checked_mul(2))
            .ok_or(Gemma4RouterError::ShapeOverflow)?;
        Ok(Self {
            hidden,
            experts,
            top_k,
            intermediate,
        })
    }

    /// Width of each residual row and router scale.
    #[must_use]
    pub fn hidden(self) -> usize {
        self.hidden
    }
    /// Number of routed experts.
    #[must_use]
    pub fn experts(self) -> usize {
        self.experts
    }
    /// Number of experts selected for each row.
    #[must_use]
    pub fn top_k(self) -> usize {
        self.top_k
    }
    /// Width of each expert's gate and up projection.
    #[must_use]
    pub fn intermediate(self) -> usize {
        self.intermediate
    }
}

/// An invalid Gemma expert geometry.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum Gemma4RouterError {
    /// At least one tensor dimension is zero.
    #[error("router dimensions must be positive")]
    ZeroDimension,
    /// The selected expert count is outside `1..=experts`.
    #[error("top-k must be between one and the expert count")]
    InvalidTopK,
    /// A required tensor element count overflows usize.
    #[error("router or expert tensor shape overflows")]
    ShapeOverflow,
}

/// Routes each row of the raw residual `[.., H]` to its experts.
///
/// `scale` `[H]`, `projection` `[E, H]` and `per_expert_scale` `[E]` are the
/// layer's `router.*` tensors. The input transform runs in the residual's
/// dtype, as the source does; [`layers::moe::route`] then works in f32.
/// Nothing is evaluated.
///
/// # Errors
///
/// [`layers::moe::MoeError`] when `top_k` exceeds the expert count, the
/// tensors do not compose, or MLX fails.
#[cfg(feature = "metal")]
#[allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]
pub fn route(
    residual: &mlx_rs::Array,
    scale: &mlx_rs::Array,
    projection: &mlx_rs::Array,
    per_expert_scale: &mlx_rs::Array,
    top_k: usize,
    epsilon: f32,
) -> Result<layers::moe::Routing, layers::moe::MoeError> {
    use mlx_rs::{Array, StreamOrDevice, fast};

    let gpu = StreamOrDevice::gpu();
    let dtype = residual.dtype();
    let hidden = residual.shape().last().copied().unwrap_or(0);
    // The source multiplies by the Python double `H ** -0.5`: torch converts
    // it to f32 once, computes the product in f32 and rounds that to the
    // residual's dtype, so the root stays f32 here (MLX promotes) and only
    // the product is cast.
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the root of a positive width is well inside f32's range"
    )]
    let root = Array::from_f32(f64::from(hidden).powf(-0.5) as f32);
    let input = mlx_rs::with_stream(gpu.as_ref(), || fast::rms_norm(residual, None, epsilon))?
        .multiply_device(scale, &gpu)?
        .multiply_device(&root, &gpu)?
        .as_dtype_device(dtype, &gpu)?;
    let logits = input.matmul_device(projection.transpose_device(&gpu)?, &gpu)?;
    let router = layers::moe::Router::softmax(top_k, true, true)?;
    layers::moe::route(&logits, &router, Some(per_expert_scale))
}
