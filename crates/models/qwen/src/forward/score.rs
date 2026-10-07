//! Teacher-forced scoring: the log probability of caller-given tokens after
//! each of their prefixes, with the output head applied a bounded number of
//! rows at a time, so device memory does not grow with the sequence the way
//! a `[positions, vocab]` logit block would (about 2.4 GB of f32 at 4,096
//! positions of a 151,936-word vocabulary).

use std::hash::BuildHasher;

use mlx_rs::{Array, StreamOrDevice, ops, ops::indexing::IndexOp};

use super::{
    LogitRows, Qwen3ForwardError, Qwen3ForwardExecutor, as_i32, project, validate_input_ids,
};

/// Rows the output head scores at once: 512 rows of a 151,936-word
/// vocabulary are about 300 MB of f32.
pub const SCORE_CHUNK_ROWS: usize = 512;

/// One scored token.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ScoredToken {
    /// Natural-log probability of the token after its prefix.
    pub logprob: f32,
    /// The `top` most likely tokens after the same prefix, most likely
    /// first, as `(token ID, log probability)`.
    pub top: Vec<(i32, f32)>,
}

/// The scored tokens, and the `top` most likely tokens after the whole
/// sequence, which a caller reads as greedy next-token candidates.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3Scores {
    /// One row for each caller-selected token, in input order.
    pub tokens: Vec<Qwen3ScoredToken>,
    /// Ranked next-token IDs and natural-log probabilities after the full input.
    pub next: Vec<(i32, f32)>,
}

impl<S: BuildHasher> Qwen3ForwardExecutor<'_, S> {
    /// Starts a sequence over `ids` and scores `ids[from..]`: entry `j` is
    /// token `ids[from + j]` after `ids[..from + j]`, so `from` is at least
    /// 1; `from == ids.len()` scores nothing and ranks only the next token.
    /// `chunk_rows` bounds how many rows the output head computes at once;
    /// it changes memory, not the values beyond reduction order.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid token IDs, context or scoring ranges, zero
    /// chunk size, excessive candidate count, or device execution failure.
    pub fn score(
        &mut self,
        ids: &[i32],
        from: usize,
        top: usize,
        chunk_rows: usize,
    ) -> Result<Qwen3Scores, Qwen3ForwardError> {
        self.reset();
        let scored = self.score_inner(ids, None, from, top, chunk_rows);
        // The cache holds the whole sequence either way; scoring keeps none
        // of it, so a later turn never builds on it by accident.
        self.reset();
        scored
    }

    /// [`Self::score`] with `ids[..split]` and `ids[split..]` appended as
    /// two prefills, as a cached prompt prefix would be. The values differ
    /// from one prefill only by reduction order, which bounds that noise.
    ///
    /// # Errors
    ///
    /// Returns the errors described by [`Self::score`], or an invalid-range
    /// error when `split` does not divide the input into two nonempty slices.
    pub fn score_split(
        &mut self,
        ids: &[i32],
        split: usize,
        from: usize,
        top: usize,
        chunk_rows: usize,
    ) -> Result<Qwen3Scores, Qwen3ForwardError> {
        if split == 0 || split >= ids.len() {
            return Err(Qwen3ForwardError::InvalidScoreRange {
                from: split,
                tokens: ids.len(),
            });
        }
        self.reset();
        let scored = self.score_inner(ids, Some(split), from, top, chunk_rows);
        self.reset();
        scored
    }

    fn score_inner(
        &mut self,
        ids: &[i32],
        split: Option<usize>,
        from: usize,
        top: usize,
        chunk_rows: usize,
    ) -> Result<Qwen3Scores, Qwen3ForwardError> {
        if from == 0 || from > ids.len() {
            return Err(Qwen3ForwardError::InvalidScoreRange {
                from,
                tokens: ids.len(),
            });
        }
        if chunk_rows == 0 || top >= self.config.vocab_size {
            return Err(Qwen3ForwardError::ShapeOverflow);
        }
        validate_input_ids(self.config, ids, 0, self.maximum_context_tokens)?;
        let stream = StreamOrDevice::gpu();
        let mut parts = Vec::new();
        let bounds = match split {
            Some(split) => vec![0, split, ids.len()],
            None => vec![0, ids.len()],
        };
        for window in bounds.windows(2) {
            let part = &ids[window[0]..window[1]];
            let len = as_i32(part.len())?;
            parts.push(self.append_ids(
                &Array::from_slice(part, &[len]),
                len,
                LogitRows::Hidden,
            )?);
        }
        let hidden = if parts.len() == 1 {
            parts.remove(0)
        } else {
            ops::concatenate_axis_device(&parts, 1, &stream)?
        };
        let vocab = as_i32(self.config.vocab_size)?;
        let wanted = as_i32(top)?;
        let mut scored = Vec::with_capacity(ids.len() - from);
        let mut next = Vec::new();
        // Row `r` of the hidden states predicts token `r + 1`; the last row
        // predicts past the sequence and gathers a placeholder target.
        for start in (from - 1..ids.len()).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(ids.len());
            let rows = as_i32(end - start)?;
            let chunk = hidden.index((.., as_i32(start)?..as_i32(end)?, ..));
            let logits = project(
                self.config,
                self.weights,
                &chunk,
                self.config.output_projection(),
            )?
            .reshape_device(&[rows, vocab], &stream)?
            .as_type_device::<f32>(&stream)?;
            let normalizer = logits.logsumexp_axis_device(-1, true, &stream)?;
            let targets: Vec<i32> = (start + 1..=end)
                .map(|position| ids.get(position).copied().unwrap_or(0))
                .collect();
            let targets = Array::from_slice(&targets, &[rows, 1]);
            let target_logprobs =
                ops::indexing::take_along_axis_device(&logits, &targets, -1, &stream)?
                    .subtract_device(&normalizer, &stream)?;
            let (top_ids, top_logprobs) = if top > 0 {
                // The first `top` of an ascending partition of the negated
                // row are the row's `top` largest entries.
                let top_ids = ops::argpartition_axis_device(
                    logits.negative_device(&stream)?,
                    wanted - 1,
                    -1,
                    &stream,
                )?
                .index((.., 0..wanted))
                .contiguous()?;
                let top_logprobs =
                    ops::indexing::take_along_axis_device(&logits, &top_ids, -1, &stream)?
                        .subtract_device(&normalizer, &stream)?;
                (Some(top_ids), Some(top_logprobs))
            } else {
                (None, None)
            };
            target_logprobs.eval()?;
            if let (Some(ids), Some(values)) = (&top_ids, &top_logprobs) {
                ids.eval()?;
                values.eval()?;
            }
            let targets = target_logprobs.as_slice::<f32>();
            for (row, &logprob) in targets.iter().enumerate() {
                let mut best = Vec::new();
                if let (Some(ids), Some(values)) = (&top_ids, &top_logprobs) {
                    let width = top;
                    // MLX partition indices are unsigned; token IDs cross the
                    // public boundary only after a checked conversion.
                    let ids = &ids.as_slice::<u32>()[row * width..(row + 1) * width];
                    let values = &values.as_slice::<f32>()[row * width..(row + 1) * width];
                    best = checked_token_scores(ids, values)?;
                    // Descending probability, ties by the lower ID.
                    best.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                }
                if start + row + 1 == ids.len() {
                    next = best;
                } else {
                    scored.push(Qwen3ScoredToken { logprob, top: best });
                }
            }
        }
        Ok(Qwen3Scores {
            tokens: scored,
            next,
        })
    }
}

/// Converts MLX's unsigned partition indices at the host token-ID boundary.
fn checked_token_scores(ids: &[u32], values: &[f32]) -> Result<Vec<(i32, f32)>, Qwen3ForwardError> {
    ids.iter()
        .zip(values)
        .map(|(&id, &value)| {
            let id = i32::try_from(id).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
            Ok((id, value))
        })
        .collect::<Result<Vec<_>, Qwen3ForwardError>>()
}
