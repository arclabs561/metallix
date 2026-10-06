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
use qwen::forward::{Qwen3ForwardError, Qwen3ForwardExecutor};

use super::turn::{Accepted, TurnLoop, TurnStep};

type Executor<'w> = Qwen3ForwardExecutor<'w, std::collections::hash_map::RandomState>;

/// Longest draft one verify scores.
pub(super) const MAX_DRAFT_TOKENS: usize = 8;

/// Verify cost in units of one pipelined decode token. Measured for
/// Qwen3-0.6B at BF16 on an M-series Mac with the ignored
/// `speculative_checkpoint` probe in the qwen crate: one scored row costs
/// ~1.2 pipelined tokens (the pipeline overlap is lost) and each further row
/// ~0.05. Larger models are more bandwidth bound, so rows are relatively
/// cheaper there; these terms err toward drafting less.
pub(super) fn draft_length() -> DraftLength {
    let cost = VerifyCost::new(0.2, 0.055).expect("constant verify cost terms are valid");
    DraftLength::new(MAX_DRAFT_TOKENS, cost).expect("constant draft range is valid")
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
