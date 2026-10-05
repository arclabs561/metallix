//! Token picks computed on the GPU from logit rows, so the host reads back
//! token IDs instead of whole vocabulary rows.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use mlx_rs::{Array, StreamOrDevice, ops};

use super::Qwen3ForwardError;

/// Greedy picks for one or more logit rows that may still be computing on
/// the GPU. Decode makes one row; a speculative verify can make several.
pub struct Qwen3TokenPicks {
    /// `[rows]` uint32 token IDs.
    pub(super) tokens: Array,
    finite: Array,
    /// The executor weights these picks may be appended to.
    pub(super) binding: usize,
}

impl Qwen3TokenPicks {
    /// Queues the per-row argmax of `[rows, vocab]` logits and a check that
    /// every logit is finite.
    pub(crate) fn start(rows: &Array, binding: usize) -> Result<Self, Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        let tokens = ops::indexing::argmax_axis_device(rows, -1, false, &stream)?;
        let finite = rows.is_finite_device(&stream)?.all_device(false, &stream)?;
        mlx_rs::transforms::async_eval([&tokens, &finite])?;
        Ok(Self {
            tokens,
            finite,
            binding,
        })
    }

    /// Waits for the picks and reads back one token ID per row. Ties go to
    /// the lowest ID, as in a host argmax that keeps the first maximum.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3ForwardError::NonFiniteLogits`] when any logit was NaN
    /// or infinite, which the host argmax also refuses.
    pub fn wait(&self) -> Result<Vec<i32>, Qwen3ForwardError> {
        mlx_rs::transforms::eval([&self.tokens, &self.finite])?;
        if !self.finite.item::<bool>() {
            return Err(Qwen3ForwardError::NonFiniteLogits);
        }
        // MLX returns argmax indices as uint32.
        self.tokens
            .as_slice::<u32>()
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| Qwen3ForwardError::ShapeOverflow))
            .collect()
    }

    /// [`Self::wait`] for a single-row pick, such as one decode step.
    pub fn wait_one(&self) -> Result<i32, Qwen3ForwardError> {
        match self.wait()?.as_slice() {
            [token] => Ok(*token),
            _ => Err(Qwen3ForwardError::CacheInconsistent),
        }
    }
}
