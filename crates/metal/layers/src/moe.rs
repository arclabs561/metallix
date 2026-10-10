//! Mixture-of-experts routing and routed experts.
//!
//! Two separate steps, so a caller can inspect or remap the routing between
//! them (an expert pager reads `chosen` back to the host there) without
//! paying for an evaluation it does not use:
//!
//! 1. [`route`] turns router logits `[.., E]`, which the model computes with
//!    its own input transform, into the chosen experts and their weights.
//! 2. [`routed_experts`] runs each chosen expert's gated MLP on the expert
//!    input and returns the weighted sum.
//!
//! Numerics follow transformers: softmax over all experts in f32, top-k,
//! renormalization in f32, then any per-expert scale. On an exact tie at the
//! top-k boundary the lower expert index is kept, as `torch.topk` does on CPU
//! (recorded by `scripts/moe-block-reference.py`). mlx-lm's order and its
//! compiled activation, whose rounding differs, are not implemented here.

use thiserror::Error;

#[cfg(feature = "metal")]
mod graph;
#[cfg(feature = "metal")]
pub use graph::{AffineMatrix, Experts, Matrix, Routing, route, routed_experts};

/// Selections (tokens times `top_k`) at which [`routed_experts`] sorts rows
/// by expert, so each expert's weights are read once per call. mlx-lm uses
/// the same threshold; MLX's sorted gather kernels round differently from
/// the unsorted ones, so changing it is a measured change, not a tuning knob.
pub const SORT_THRESHOLD: usize = 64;

/// How a router turns logits into expert weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Router {
    top_k: usize,
    renormalize: bool,
    per_expert_scale: bool,
}

impl Router {
    /// A softmax router choosing `top_k` experts.
    ///
    /// `renormalize` divides the chosen probabilities by their sum.
    /// `per_expert_scale` multiplies each weight, after renormalization, by
    /// a stored `[E]` scale (Gemma 4), which [`route`] then requires.
    ///
    /// # Errors
    ///
    /// [`MoeError::TopK`] when `top_k` is zero.
    pub fn softmax(
        top_k: usize,
        renormalize: bool,
        per_expert_scale: bool,
    ) -> Result<Self, MoeError> {
        if top_k == 0 {
            return Err(MoeError::TopK {
                top_k,
                experts: None,
            });
        }
        Ok(Self {
            top_k,
            renormalize,
            per_expert_scale,
        })
    }

    /// Experts chosen per token.
    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    /// Whether the chosen probabilities are divided by their sum.
    #[must_use]
    pub const fn renormalize(&self) -> bool {
        self.renormalize
    }

    /// Whether weights are multiplied by a per-expert scale.
    #[must_use]
    pub const fn per_expert_scale(&self) -> bool {
        self.per_expert_scale
    }
}

/// The gate activation of each expert: `down(act(gate(x)) * up(x))`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Activation {
    /// `x * sigmoid(x)` (Qwen, Kolibri).
    Silu,
    /// Tanh-approximated GELU, transformers' `gelu_pytorch_tanh` (Gemma 4).
    GeluTanh,
}

/// A routing or expert layout this block refuses.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MoeError {
    /// `top_k` is zero or larger than the expert count.
    #[error("top_k {top_k} is outside 1..=experts ({experts:?})")]
    TopK {
        /// Requested experts per token.
        top_k: usize,
        /// Experts available, when known.
        experts: Option<usize>,
    },
    /// Tensors whose shapes do not compose.
    #[error("{0}")]
    Shape(String),
    /// The router needs a tensor the caller did not pass.
    #[error("router needs a per-expert scale")]
    MissingPerExpertScale,
    /// An MLX operation failed.
    #[cfg(feature = "metal")]
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
}

#[cfg(test)]
mod tests {
    use super::{MoeError, Router};

    #[test]
    fn a_router_needs_at_least_one_expert_per_token() {
        assert!(matches!(
            Router::softmax(0, true, false),
            Err(MoeError::TopK { top_k: 0, .. })
        ));
        let router = Router::softmax(3, true, true).expect("valid router");
        assert_eq!(
            (
                router.top_k(),
                router.renormalize(),
                router.per_expert_scale()
            ),
            (3, true, true)
        );
    }
}
