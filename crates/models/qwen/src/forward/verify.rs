//! Logits at every appended position.
//!
//! Speculative decoding scores a drafted continuation in one append and then
//! rolls back with [`Qwen3ForwardExecutor::truncate_cached_tokens`].
//! Teacher-forced scoring uses the same rows without a rollback.

use std::hash::BuildHasher;

use mlx_rs::{Array, StreamOrDevice};

use super::{
    LogitRows, Qwen3ForwardError, Qwen3ForwardExecutor, Qwen3TokenPicks, as_i32, validate_input_ids,
};

/// Row-major `[positions, vocab_size]` f32 logits from one append.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3PositionLogits {
    values: Vec<f32>,
    vocab_size: usize,
}

impl Qwen3PositionLogits {
    /// Number of scored positions.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.values.len() / self.vocab_size
    }

    /// Vocabulary width of each row.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Next-token logits after appended position `index`.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.vocab_size)?;
        self.values.get(start..start.checked_add(self.vocab_size)?)
    }

    /// The row-major values, for callers that wrap them in their own type.
    #[must_use]
    pub fn into_values(self) -> Vec<f32> {
        self.values
    }
}

impl<S: BuildHasher> Qwen3ForwardExecutor<'_, S> {
    /// Starts a sequence and returns logits after every prompt position, for
    /// teacher-forced scoring. Row `i` is the next-token distribution after
    /// `input_ids[..=i]`.
    ///
    /// # Errors
    ///
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn prefill_all_logits(
        &mut self,
        input_ids: &[i32],
    ) -> Result<Qwen3PositionLogits, Qwen3ForwardError> {
        self.reset();
        self.append_rows(input_ids)
    }

    /// Appends a chunk to a prefilled sequence and returns logits after every
    /// chunk position. Row `i` agrees with
    /// [`Self::extend_last_logits`] of `input_ids[..=i]` up to kernel
    /// reduction order. Each row costs one vocabulary-wide readback.
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn extend_all_logits(
        &mut self,
        input_ids: &[i32],
    ) -> Result<Qwen3PositionLogits, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        self.append_rows(input_ids)
    }

    /// Appends a chunk to a prefilled sequence and starts the greedy pick
    /// after every chunk position on the GPU: pick `i` is the argmax of the
    /// row [`Self::extend_all_logits`] would return at `i`, with the same
    /// lowest-ID tie rule as decode. Only the `input_ids.len()` token IDs
    /// are read back, not the vocabulary rows, which is what a greedy
    /// speculative verify needs.
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    /// * [`ShapeOverflow`](crate::forward::Qwen3ForwardError::ShapeOverflow)
    ///   when a pick does not fit a token ID.
    pub fn extend_greedy(
        &mut self,
        input_ids: &[i32],
    ) -> Result<Qwen3TokenPicks, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        validate_input_ids(
            self.config,
            input_ids,
            self.cached_tokens,
            self.maximum_context_tokens,
        )?;
        let seq_len =
            i32::try_from(input_ids.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        let pending = self
            .append_ids(
                &Array::from_slice(input_ids, &[seq_len]),
                seq_len,
                LogitRows::All,
            )
            .and_then(|logits| {
                let rows = logits.reshape_device(
                    &[seq_len, as_i32(self.config.vocab_size)?],
                    StreamOrDevice::gpu(),
                )?;
                Qwen3TokenPicks::start(&rows, self.weights_address())
            });
        if pending.is_err() {
            // A partly built append must not leave some layers ahead.
            self.reset();
        }
        pending
    }

    fn append_rows(&mut self, input_ids: &[i32]) -> Result<Qwen3PositionLogits, Qwen3ForwardError> {
        let values = self.append(input_ids, LogitRows::All)?;
        Ok(Qwen3PositionLogits {
            values,
            vocab_size: self.config.vocab_size,
        })
    }
}

/// Reads `[1, seq_len, vocab]` logits back as row-major f32 in one
/// evaluation.
pub(super) fn read_all_logits(
    logits: &Array,
    seq_len: i32,
    vocab_size: usize,
) -> Result<Vec<f32>, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let total = seq_len
        .checked_mul(as_i32(vocab_size)?)
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    let rows = logits
        .reshape_device(&[total], &stream)?
        .as_type_device::<f32>(&stream)?;
    rows.eval()?;
    Ok(rows.as_slice::<f32>().to_vec())
}
