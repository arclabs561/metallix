//! Prompt-lookup speculative decoding for a resident Qwen chat turn.
//!
//! The resident executor's K/V is a per-position prefix, so it can score a
//! drafted chunk and drop rejected positions; that is the capability the
//! engine's speculation step needs. A recurrent-state adapter would need a
//! checkpoint scheme before it could join.

#[cfg(test)]
use std::sync::atomic::Ordering;

use chat_format::ChatFormat;
use engine::speculative::{
    DraftLength, Pick, PositionLogits, SpeculativeTarget, StepOutcome, VerifyCost, accept_verified,
    speculative_step,
};
use qwen::{
    forward::{Qwen3ForwardError, Qwen3ForwardExecutor, Qwen3PickRule},
    metal::Qwen3FloatPrecision,
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
pub(super) fn verify_cost(precision: Qwen3FloatPrecision) -> VerifyCost {
    let (fixed, per_token) = match precision {
        Qwen3FloatPrecision::BFloat16 | Qwen3FloatPrecision::Float16 => (0.2, 0.055),
        Qwen3FloatPrecision::Float32 => (0.6, 0.035),
    };
    VerifyCost::new(fixed, per_token).expect("constant verify cost terms are valid")
}

/// The draft length controller for one turn on weights at `precision`.
pub(super) fn draft_length(precision: Qwen3FloatPrecision) -> DraftLength {
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

/// What one verify did, beyond the engine's outcome.
pub(super) struct Verified {
    pub(super) outcome: StepOutcome,
    /// The accepted tokens, in order.
    pub(super) tokens: Vec<Accepted>,
    /// Rows settled from whole vocabulary rows read back to the host: every
    /// settled row on the host-row path, and on the GPU path only the rows
    /// whose candidates could not settle them.
    pub(super) full_rows: usize,
}

/// Scores `last_token` plus `draft` in one forward pass and accepts tokens
/// from the scored rows through the turn's own rules until one leaves the
/// draft or ends the turn, then rolls the K/V back to the accepted tokens.
///
/// With `gpu_rule` (from [`TurnLoop::gpu_rule_rows`]) the GPU picks every
/// row and the host settles each kept row from its candidates with
/// [`TurnLoop::settle_row`], reading a whole row only when those cannot
/// decide it. Without it every kept row is read back and picked with
/// [`TurnLoop::pick`] after suppression. Either way each kept token takes
/// the sampler's next draw, in order, so a sampled turn keeps its
/// distribution and a seeded turn its random stream, and stop and length
/// rules apply to verified tokens as to decoded ones.
pub(super) fn verify(
    format: &ChatFormat,
    executor: &mut Executor<'_>,
    turn: &mut TurnLoop,
    last_token: i32,
    draft: &[i32],
    gpu_rule: Option<&Qwen3PickRule>,
) -> Result<Verified, String> {
    let mut tokens = Vec::with_capacity(draft.len() + 1);
    let mut full_rows = 0;
    let mut kept = |token: Accepted| {
        tokens.push(token);
        Pick {
            token: token.token,
            stop: matches!(token.step, TurnStep::Stop(_)),
        }
    };
    #[cfg(test)]
    let gpu_rule = gpu_rule.filter(|_| !test_hooks::HOST_ROW_VERIFY.load(Ordering::Relaxed));
    let outcome = if let Some(rule) = gpu_rule {
        #[cfg(test)]
        test_hooks::GPU_VERIFIES.fetch_add(1, Ordering::Relaxed);
        let base = executor.cached_tokens();
        let mut input = Vec::with_capacity(draft.len() + 1);
        input.push(last_token);
        input.extend_from_slice(draft);
        let picks = executor
            .extend_picks(&input, rule)
            .map_err(|error| error.to_string())?;
        let (gpu_tokens, candidates) = picks
            .wait_with_candidates()
            .map_err(|error| error.to_string())?;
        if gpu_tokens.len() != input.len() {
            return Err(format!(
                "verify picked {} rows for {} tokens",
                gpu_tokens.len(),
                input.len()
            ));
        }
        let mut settle = |row: usize| -> Result<Pick, String> {
            let row_candidates = candidates.as_ref().and_then(|rows| rows.get(row));
            let (token, read_whole) =
                turn.settle_row(format, &picks, row, gpu_tokens[row], row_candidates)?;
            full_rows += usize::from(read_whole);
            Ok(kept(token))
        };
        accept_verified(&mut Target(executor), base, draft, &mut settle)
    } else {
        let mut pick = |row: &[f32]| -> Result<Pick, String> {
            let mut row = row.to_vec();
            full_rows += 1;
            Ok(kept(turn.pick(format, &mut row)?))
        };
        speculative_step(&mut Target(executor), last_token, draft, &mut pick)
    }
    .map_err(|error| error.to_string())?;
    Ok(Verified {
        outcome,
        tokens,
        full_rows,
    })
}

/// Switches for checkpoint tests that compare the two verify paths.
#[cfg(test)]
pub(super) mod test_hooks {
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    /// Forces every verify onto whole host rows.
    pub(crate) static HOST_ROW_VERIFY: AtomicBool = AtomicBool::new(false);
    /// Verifies that took the GPU-pick path.
    pub(crate) static GPU_VERIFIES: AtomicUsize = AtomicUsize::new(0);
}

#[cfg(test)]
mod tests {
    use qwen::metal::Qwen3FloatPrecision;

    use super::{MAX_DRAFT_TOKENS, draft_length};

    #[test]
    fn full_acceptance_drafts_the_cap_at_every_precision() {
        for precision in [
            Qwen3FloatPrecision::BFloat16,
            Qwen3FloatPrecision::Float16,
            Qwen3FloatPrecision::Float32,
        ] {
            let mut length = draft_length(precision);
            for _ in 0..50 {
                length.observe(MAX_DRAFT_TOKENS, MAX_DRAFT_TOKENS);
            }
            assert_eq!(length.next(), MAX_DRAFT_TOKENS, "{precision:?}");
        }
    }
}
