//! Gemma 4 `MoE` router geometry and a scalar float32 reference boundary.
//!
//! This does not enable `MoE` checkpoint admission or expert execution. It follows
//! `Gemma4TextRouter` in Transformers commit
//! `a906d3c4b65095f2308b6a6a193e934d03b8eb5d`: unweighted RMS normalization,
//! learned input scaling, softmax, selected-weight renormalization, then learned
//! per-expert scaling. Scalar reductions are not backend reduction-parity claims.

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

/// One selected expert and its weight after per-expert scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4ExpertRoute {
    expert: usize,
    weight: f32,
}

impl Gemma4ExpertRoute {
    /// Expert index in the checkpoint tensors.
    #[must_use]
    pub fn expert(self) -> usize {
        self.expert
    }
    /// Renormalized top-k probability times the learned expert scale.
    #[must_use]
    pub fn weight(self) -> f32 {
        self.weight
    }
}

/// Borrowed, shape-checked Gemma router tensors for one residual row at a time.
pub struct Gemma4Router<'a> {
    geometry: Gemma4MoeGeometry,
    scale: &'a [f32],
    projection: &'a [f32],
    expert_scale: &'a [f32],
    epsilon: f32,
}

impl<'a> Gemma4Router<'a> {
    /// Borrows `scale[H]`, row-major `projection[E,H]`, and `expert_scale[E]`.
    ///
    /// Learned scales may be negative. Epsilon must be finite and positive;
    /// all tensor values must be finite.
    ///
    /// # Errors
    ///
    /// Returns an error for tensor lengths inconsistent with the geometry,
    /// nonfinite tensor values, or nonpositive/nonfinite RMS epsilon.
    pub fn new(
        geometry: Gemma4MoeGeometry,
        scale: &'a [f32],
        projection: &'a [f32],
        expert_scale: &'a [f32],
        epsilon: f32,
    ) -> Result<Self, Gemma4RouterError> {
        if scale.len() != geometry.hidden
            || projection.len() != geometry.experts * geometry.hidden
            || expert_scale.len() != geometry.experts
        {
            return Err(Gemma4RouterError::ShapeMismatch);
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(Gemma4RouterError::InvalidEpsilon);
        }
        if !scale
            .iter()
            .chain(projection)
            .chain(expert_scale)
            .all(|v| v.is_finite())
        {
            return Err(Gemma4RouterError::NonFinite);
        }
        Ok(Self {
            geometry,
            scale,
            projection,
            expert_scale,
            epsilon,
        })
    }

    /// Routes one pre-MLP residual row, without applying expert computation.
    ///
    /// Ties use ascending expert index in this scalar reference. `PyTorch` does
    /// not promise stable tie indices, so exact source index parity requires
    /// distinct scores at the selection boundary. Final weights need not sum
    /// to one: per-expert scales are applied after selected normalization.
    ///
    /// # Errors
    ///
    /// Returns an error for a residual row with the wrong width or nonfinite
    /// values, or if normalization, projection or route weights become nonfinite.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "tensor widths are converted to source float32 arithmetic"
    )]
    pub fn route(&self, residual: &[f32]) -> Result<Vec<Gemma4ExpertRoute>, Gemma4RouterError> {
        let g = self.geometry;
        if residual.len() != g.hidden {
            return Err(Gemma4RouterError::ShapeMismatch);
        }
        if !residual.iter().all(|v| v.is_finite()) {
            return Err(Gemma4RouterError::NonFinite);
        }
        let variance = residual.iter().map(|v| v * v).sum::<f32>() / g.hidden as f32;
        let denominator = variance + self.epsilon;
        if !denominator.is_finite() {
            return Err(Gemma4RouterError::NonFinite);
        }
        let inverse = denominator.powf(-0.5);
        let root = (g.hidden as f64).powf(-0.5) as f32;
        let input: Vec<f32> = residual
            .iter()
            .zip(self.scale)
            .map(|(x, scale)| ((x * inverse) * scale) * root)
            .collect();
        let scores: Vec<f32> = self
            .projection
            .chunks_exact(g.hidden)
            .map(|row| row.iter().zip(&input).map(|(w, x)| w * x).sum())
            .collect();
        if !variance.is_finite() || !scores.iter().all(|v| v.is_finite()) {
            return Err(Gemma4RouterError::NonFinite);
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
        let denominator: f32 = probabilities.iter().sum();
        for probability in &mut probabilities {
            *probability /= denominator;
        }
        let mut selected: Vec<usize> = (0..g.experts).collect();
        selected.sort_by(|&a, &b| {
            probabilities[b]
                .total_cmp(&probabilities[a])
                .then(a.cmp(&b))
        });
        selected.truncate(g.top_k);
        let selected_sum: f32 = selected.iter().map(|&i| probabilities[i]).sum();
        let routes: Vec<_> = selected
            .into_iter()
            .map(|expert| Gemma4ExpertRoute {
                expert,
                weight: (probabilities[expert] / selected_sum) * self.expert_scale[expert],
            })
            .collect();
        if !routes.iter().all(|r| r.weight.is_finite()) {
            return Err(Gemma4RouterError::NonFinite);
        }
        Ok(routes)
    }
}

/// An invalid Gemma router contract or nonfinite intermediate.
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
    /// A tensor or residual row has the wrong element count.
    #[error("router tensor shape mismatch")]
    ShapeMismatch,
    /// RMS epsilon is nonfinite or nonpositive.
    #[error("router epsilon must be finite and positive")]
    InvalidEpsilon,
    /// Input, parameter or intermediate arithmetic is nonfinite.
    #[error("nonfinite router value")]
    NonFinite,
}
