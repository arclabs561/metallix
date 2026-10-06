//! Prompt-lookup speculative decoding for a resident Qwen chat turn.
//!
//! The resident executor's K/V is a per-position prefix, so it can score a
//! drafted chunk and drop rejected positions; that is the capability the
//! engine's speculation step needs. A recurrent-state adapter would need a
//! checkpoint scheme before it could join.

use chat_format::ChatFormat;
use engine::speculative::{
    DraftLength, GreedySpeculativeTarget, Pick, PositionLogits, SpeculativeTarget, StepOutcome,
    VerifyCost, greedy_speculative_step, speculative_step,
};
use qwen::{
    forward::{Qwen3ForwardError, Qwen3ForwardExecutor},
    metal::Qwen3WeightPrecision,
};

use super::turn::{Accepted, TurnLoop, TurnStep};

type Executor<'w> = Qwen3ForwardExecutor<'w, std::collections::hash_map::RandomState>;

/// Query rows MLX attention scores with its vector kernel. MLX v0.32.2 uses
/// `sdpa_vector` only while the query has at most 8 positions and the full
/// kernel above that
/// (<https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp#L812>);
/// on Qwen3-0.6B a 9-row verify costs ~1.7 decode steps against ~1.2 at 8.
const MLX_VECTOR_ATTENTION_ROWS: usize = 8;

/// Longest draft one verify scores: the draft plus the last emitted token
/// stay within the vector attention kernel.
pub(super) const MAX_DRAFT_TOKENS: usize = MLX_VECTOR_ATTENTION_ROWS - 1;
const _: () = assert!(MAX_DRAFT_TOKENS < MLX_VECTOR_ATTENTION_ROWS);

/// Verify cost for weights at `precision`, in units of one pipelined decode
/// token, from the ignored `speculative_checkpoint` probe in the qwen crate on
/// Qwen3-0.6B. 16-bit weights: one scored row costs ~1.2 pipelined tokens
/// (the pipeline overlap is lost) and each further row ~0.05. Float32 reads
/// twice the weight bytes per step, so a verify's extra rows are relatively
/// cheaper; it was measured against an unpipelined f32 decode at ~0.6 fixed
/// and ~0.035 per row.
///
/// A quantized target needs its own measured row before it speculates: MLX
/// runs a 4-bit matmul as a matrix-vector product while the rows stay below
/// `get_qmv_batch_limit`
/// (<https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp#L85>),
/// and that cost grows with every row, so on 4-bit 8B shapes each extra
/// verified row measured ~0.2 of a decode step rather than ~0.05.
pub(super) fn verify_cost(precision: Qwen3WeightPrecision) -> VerifyCost {
    let (fixed, per_token) = match precision {
        Qwen3WeightPrecision::BFloat16 | Qwen3WeightPrecision::Float16 => (0.2, 0.055),
        Qwen3WeightPrecision::Float32 => (0.6, 0.035),
    };
    VerifyCost::new(fixed, per_token).expect("constant verify cost terms are valid")
}

/// The draft length controller for one turn on weights at `precision`.
pub(super) fn draft_length(precision: Qwen3WeightPrecision) -> DraftLength {
    DraftLength::new(MAX_DRAFT_TOKENS, verify_cost(precision))
        .expect("constant draft range is valid")
}

/// The resident executor as a speculation target.
struct Target<'e, 'w>(&'e mut Executor<'w>);

impl SpeculativeTarget for Target<'_, '_> {
    type Error = Qwen3ForwardError;

    fn cached_tokens(&self) -> usize {
        self.0.cached_tokens()
    }

    fn verify(&mut self, tokens: &[i32]) -> Result<PositionLogits, Qwen3ForwardError> {
        let rows = self.0.extend_all_logits(tokens)?;
        let vocabulary = rows.vocab_size();
        // The executor returns exactly one vocabulary row per token.
        PositionLogits::new(rows.into_values(), vocabulary)
            .map_err(|_| Qwen3ForwardError::CacheInconsistent)
    }

    fn truncate(&mut self, tokens: usize) -> Result<(), Qwen3ForwardError> {
        self.0.truncate_cached_tokens(tokens)
    }
}

impl GreedySpeculativeTarget for Target<'_, '_> {
    fn verify_greedy(&mut self, tokens: &[i32]) -> Result<Vec<i32>, Qwen3ForwardError> {
        self.0.extend_greedy(tokens)?.wait()
    }
}

/// Scores `last_token` plus `draft` in one forward pass and accepts tokens
/// from the scored rows through the turn's own rules until one leaves the
/// draft or ends the turn, then rolls the K/V back to the accepted tokens.
///
/// Every accepted token goes through `turn`: [`TurnLoop::pick`] (after
/// suppression, with logprob receipts, consuming the sampler's draws in
/// order) or, for a plain greedy turn, [`TurnLoop::accept_gpu_greedy`] on the
/// GPU's argmax. So a sampled turn keeps its distribution and a seeded turn
/// its random stream, and stop and length rules apply to verified tokens as
/// to decoded ones.
pub(super) fn verify(
    format: &ChatFormat,
    executor: &mut Executor<'_>,
    turn: &mut TurnLoop,
    last_token: i32,
    draft: &[i32],
    gpu_greedy: bool,
) -> Result<(StepOutcome, Vec<Accepted>), String> {
    let mut accepted = Vec::with_capacity(draft.len() + 1);
    let mut target = Target(executor);
    let outcome = if gpu_greedy {
        let mut failure = None;
        let mut is_stop = |token: i32| match turn.accept_gpu_greedy(format, token) {
            Ok(token) => {
                accepted.push(token);
                matches!(token.step, TurnStep::Stop(_))
            }
            Err(error) => {
                failure = Some(error);
                true
            }
        };
        let outcome = greedy_speculative_step(&mut target, last_token, draft, &mut is_stop)
            .map_err(|error| error.to_string())?;
        if let Some(error) = failure {
            return Err(error);
        }
        outcome
    } else {
        let mut pick = |row: &[f32]| -> Result<Pick, String> {
            let mut row = row.to_vec();
            let token = turn.pick(format, &mut row)?;
            accepted.push(token);
            Ok(Pick {
                token: token.token,
                stop: matches!(token.step, TurnStep::Stop(_)),
            })
        };
        speculative_step(&mut target, last_token, draft, &mut pick)
            .map_err(|error| error.to_string())?
    };
    Ok((outcome, accepted))
}

#[cfg(test)]
mod tests {
    use qwen::metal::Qwen3WeightPrecision;

    use super::{MAX_DRAFT_TOKENS, draft_length};

    #[test]
    fn full_acceptance_drafts_the_cap_at_every_precision() {
        for precision in [
            Qwen3WeightPrecision::BFloat16,
            Qwen3WeightPrecision::Float16,
            Qwen3WeightPrecision::Float32,
        ] {
            let mut length = draft_length(precision);
            for _ in 0..50 {
                length.observe(MAX_DRAFT_TOKENS, MAX_DRAFT_TOKENS);
            }
            assert_eq!(length.next(), MAX_DRAFT_TOKENS, "{precision:?}");
        }
    }
}
