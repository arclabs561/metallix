//! Speculative decoding independent of a model graph.
//!
//! A drafter proposes `k` tokens; the target model scores the last emitted
//! token plus the draft in one forward pass, returning logits at every
//! position; the caller's ordinary token picker then runs on those rows in
//! order. With a deterministic draft `d` (prompt lookup, or any drafter that
//! does not sample) the draft distribution is a point mass, and the
//! speculative-sampling acceptance rule of Leviathan et al. (arXiv
//! 2211.17192, Algorithm 1) and Chen et al. (arXiv 2302.01318, Algorithm 2)
//! reduces to: draw `x` from the target row as usual, keep going while
//! `x == d`, stop at the first mismatch and emit `x`. Accepting `d` has
//! probability `p(d)`, and on rejection `x` follows `p` with `d` removed and
//! renormalized, which is exactly `norm(max(0, p - delta_d))`. The picker
//! therefore keeps its temperature, truncation, grammar mask and random
//! stream, and consumes one uniform per emitted token, as it would without
//! speculation.
//!
//! Drafters that sample from their own distribution `q` (draft models, MTP
//! heads) need the general rule in [`accept_or_resample`].

use thiserror::Error;

/// A model whose cached sequence can be scored in one chunk and rolled back.
///
/// This is the capability an adapter declares to take part in speculative
/// decoding. Only caches whose state is a per-position prefix can implement
/// [`Self::truncate`] cheaply: attention K/V qualifies, while a recurrent
/// state (linear-attention or SSM layers) cannot drop its last positions and
/// should not implement this trait without a checkpoint scheme.
pub trait SpeculativeTarget {
    /// Adapter error.
    type Error;

    /// Number of positions in the cache.
    fn cached_tokens(&self) -> usize;

    /// Appends `tokens` to the cache and returns logits for every appended
    /// position: row `i` is the next-token distribution after `tokens[..=i]`.
    fn verify(&mut self, tokens: &[i32]) -> Result<PositionLogits, Self::Error>;

    /// Drops cached positions at and after `tokens`.
    fn truncate(&mut self, tokens: usize) -> Result<(), Self::Error>;
}

/// Row-major `[positions, vocabulary]` logits from one verification pass.
#[derive(Clone, Debug, PartialEq)]
pub struct PositionLogits {
    values: Vec<f32>,
    vocabulary: usize,
}

impl PositionLogits {
    /// Wraps row-major logits.
    ///
    /// # Errors
    ///
    /// Returns [`SpeculationError::LogitShape`] when `vocabulary` is zero or
    /// does not divide the value count, or there are no rows.
    pub fn new(values: Vec<f32>, vocabulary: usize) -> Result<Self, SpeculationError> {
        if vocabulary == 0 || values.is_empty() || !values.len().is_multiple_of(vocabulary) {
            return Err(SpeculationError::LogitShape);
        }
        Ok(Self { values, vocabulary })
    }

    /// Number of scored positions.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.values.len() / self.vocabulary
    }

    /// Vocabulary width of every row.
    #[must_use]
    pub const fn vocabulary(&self) -> usize {
        self.vocabulary
    }

    /// Logits after position `index`, if it was scored.
    #[must_use]
    pub fn row(&self, index: usize) -> Option<&[f32]> {
        let start = index.checked_mul(self.vocabulary)?;
        self.values.get(start..start.checked_add(self.vocabulary)?)
    }
}

/// One token chosen by the caller's picker.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Pick {
    /// The selected token.
    pub token: i32,
    /// Whether generation ends with this token (EOS, completed grammar).
    pub stop: bool,
}

/// Result of one draft-and-verify step.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StepOutcome {
    /// Tokens to emit, in order; never empty. The last one is not in the
    /// cache yet: it is the `last_token` of the next step.
    pub emitted: Vec<i32>,
    /// Draft tokens that were scored.
    pub drafted: usize,
    /// Leading draft tokens the picker reproduced.
    pub accepted: usize,
    /// Whether the picker ended generation.
    pub stopped: bool,
}

/// Failure of a speculative step.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SpeculationError<T = std::convert::Infallible, P = std::convert::Infallible> {
    /// The target model failed; its cache state is the adapter's contract.
    #[error("speculative target failed: {0}")]
    Target(T),
    /// The token picker failed.
    #[error("token picker failed: {0}")]
    Pick(P),
    /// Verification returned a different number of rows than tokens sent.
    #[error("verification returned {actual} logit rows for {expected} tokens")]
    RowCount {
        /// Tokens sent.
        expected: usize,
        /// Rows returned.
        actual: usize,
    },
    /// Logits are not a nonempty whole number of vocabulary rows.
    #[error("logits must be a nonempty whole number of vocabulary rows")]
    LogitShape,
    /// A probability vector is malformed.
    #[error("probabilities must be finite, nonnegative, equal length and sum to a positive mass")]
    Probabilities,
    /// A token is outside the vocabulary.
    #[error("token {0} is outside the vocabulary")]
    TokenOutOfRange(i32),
    /// A uniform variate is not in [0, 1).
    #[error("uniform variate must be finite and in [0, 1)")]
    InvalidUniform,
}

/// Scores `last_token` plus `draft` in one target pass, then picks tokens
/// row by row until the picker diverges from the draft, stops, or runs past
/// the draft (the bonus token).
///
/// `last_token` is the most recent emitted token, not yet cached. On return
/// the cache holds every emitted token except the last one, so plain decode
/// and speculation can alternate. An empty draft is an ordinary decode step.
///
/// # Errors
///
/// Returns the target's or picker's error, or [`SpeculationError::RowCount`]
/// when the adapter breaks the one-row-per-token contract.
pub fn speculative_step<T, P>(
    target: &mut T,
    last_token: i32,
    draft: &[i32],
    pick: &mut dyn FnMut(&[f32]) -> Result<Pick, P>,
) -> Result<StepOutcome, SpeculationError<T::Error, P>>
where
    T: SpeculativeTarget + ?Sized,
{
    let base = target.cached_tokens();
    let mut input = Vec::with_capacity(draft.len() + 1);
    input.push(last_token);
    input.extend_from_slice(draft);
    let logits = target.verify(&input).map_err(SpeculationError::Target)?;
    if logits.positions() != input.len() {
        return Err(SpeculationError::RowCount {
            expected: input.len(),
            actual: logits.positions(),
        });
    }

    accept_and_roll_back(target, base, draft, |index| {
        let row = logits.row(index).ok_or(SpeculationError::LogitShape)?;
        pick(row).map_err(SpeculationError::Pick)
    })
}

/// A target that can also pick the greedy token after every scored position
/// on the device, so a greedy verify reads back token IDs instead of
/// vocabulary rows.
pub trait GreedySpeculativeTarget: SpeculativeTarget {
    /// Appends `tokens` and returns the greedy (lowest-ID tie) token after
    /// every appended position: entry `i` follows `tokens[..=i]`.
    fn verify_greedy(&mut self, tokens: &[i32]) -> Result<Vec<i32>, Self::Error>;
}

/// [`speculative_step`] for greedy decoding on a target that picks on the
/// device. `is_stop` marks tokens that end generation. The emitted tokens
/// are the ones plain greedy decoding would produce, given the same rows.
///
/// # Errors
///
/// Returns the target's error, or [`SpeculationError::RowCount`] when the
/// adapter returns a different number of picks than tokens sent.
pub fn greedy_speculative_step<T>(
    target: &mut T,
    last_token: i32,
    draft: &[i32],
    is_stop: &mut dyn FnMut(i32) -> bool,
) -> Result<StepOutcome, SpeculationError<T::Error>>
where
    T: GreedySpeculativeTarget + ?Sized,
{
    let base = target.cached_tokens();
    let mut input = Vec::with_capacity(draft.len() + 1);
    input.push(last_token);
    input.extend_from_slice(draft);
    let picks = target
        .verify_greedy(&input)
        .map_err(SpeculationError::Target)?;
    if picks.len() != input.len() {
        return Err(SpeculationError::RowCount {
            expected: input.len(),
            actual: picks.len(),
        });
    }
    accept_and_roll_back(target, base, draft, |index| {
        let token = picks[index];
        Ok(Pick {
            token,
            stop: is_stop(token),
        })
    })
}

/// Picks row by row until a pick diverges from the draft, stops, or is the
/// bonus row, then drops cache positions conditioned on rejected drafts.
fn accept_and_roll_back<T, P>(
    target: &mut T,
    base: usize,
    draft: &[i32],
    mut pick_at: impl FnMut(usize) -> Result<Pick, SpeculationError<T::Error, P>>,
) -> Result<StepOutcome, SpeculationError<T::Error, P>>
where
    T: SpeculativeTarget + ?Sized,
{
    let mut emitted = Vec::with_capacity(draft.len() + 1);
    let mut accepted = 0;
    let mut stopped = false;
    for index in 0..=draft.len() {
        let picked = pick_at(index)?;
        emitted.push(picked.token);
        let matches_draft = draft.get(index) == Some(&picked.token);
        if matches_draft {
            accepted += 1;
        }
        if picked.stop {
            stopped = true;
            break;
        }
        if !matches_draft {
            break;
        }
    }
    // Rows past the first divergence were conditioned on rejected draft
    // tokens; their K/V must not survive into the next step.
    target
        .truncate(base + emitted.len())
        .map_err(SpeculationError::Target)?;
    Ok(StepOutcome {
        emitted,
        drafted: draft.len(),
        accepted,
        stopped,
    })
}

/// Outcome of the general speculative-sampling rule for one position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Verdict {
    /// The draft token is kept; verification continues at the next position.
    Accepted,
    /// The draft token is rejected and this token is emitted instead.
    Resampled(usize),
}

/// Leviathan et al. Algorithm 1 for a draft token `drafted` sampled from
/// `draft`: accept with probability `min(1, target/draft)`, otherwise sample
/// from `norm(max(0, target - draft))`. `target` and `draft` are probability
/// vectors after the same temperature and truncation (Sec. 2.2 of the paper).
/// For a deterministic draft, prefer [`speculative_step`]'s sample-and-compare,
/// which needs one uniform per token instead of two.
///
/// # Errors
///
/// Returns an error for mismatched or invalid probabilities, an out-of-range
/// token, or a uniform outside [0, 1).
pub fn accept_or_resample(
    target: &[f64],
    draft: &[f64],
    drafted: usize,
    accept_uniform: f64,
    resample_uniform: f64,
) -> Result<Verdict, SpeculationError> {
    let valid = |values: &[f64]| {
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && values.iter().sum::<f64>() > 0.0
    };
    if target.len() != draft.len() || !valid(target) || !valid(draft) {
        return Err(SpeculationError::Probabilities);
    }
    for uniform in [accept_uniform, resample_uniform] {
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err(SpeculationError::InvalidUniform);
        }
    }
    let target_total: f64 = target.iter().sum();
    let draft_total: f64 = draft.iter().sum();
    let (Some(&p), Some(&q)) = (target.get(drafted), draft.get(drafted)) else {
        return Err(SpeculationError::TokenOutOfRange(
            i32::try_from(drafted).unwrap_or(i32::MAX),
        ));
    };
    let (p, q) = (p / target_total, q / draft_total);
    if q <= 0.0 {
        // The drafter could not have produced this token.
        return Err(SpeculationError::Probabilities);
    }
    if accept_uniform < p / q {
        return Ok(Verdict::Accepted);
    }
    let residual =
        |index: usize| (target[index] / target_total - draft[index] / draft_total).max(0.0);
    let mass: f64 = (0..target.len()).map(residual).sum();
    if mass <= 0.0 {
        // p == q up to rounding, so rejection had probability ~0; the target
        // row itself is the correct fallback.
        return categorical(target, resample_uniform * target_total).map(Verdict::Resampled);
    }
    let threshold = resample_uniform * mass;
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for index in 0..target.len() {
        let weight = residual(index);
        if weight > 0.0 {
            last_positive = Some(index);
        }
        cumulative += weight;
        if threshold < cumulative {
            return Ok(Verdict::Resampled(index));
        }
    }
    last_positive
        .map(Verdict::Resampled)
        .ok_or(SpeculationError::Probabilities)
}

fn categorical(weights: &[f64], threshold: f64) -> Result<usize, SpeculationError> {
    let mut cumulative = 0.0;
    let mut last_positive = None;
    for (index, &weight) in weights.iter().enumerate() {
        if weight > 0.0 {
            last_positive = Some(index);
        }
        cumulative += weight;
        if threshold < cumulative {
            return Ok(index);
        }
    }
    last_positive.ok_or(SpeculationError::Probabilities)
}

/// Prompt-lookup drafting: match the newest n-gram of the sequence against
/// earlier text and propose what followed it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromptLookup {
    min_ngram: usize,
    max_ngram: usize,
}

impl PromptLookup {
    /// Creates a drafter that tries suffixes from `max_ngram` down to
    /// `min_ngram` tokens.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::NgramRange`] unless `1 <= min <= max`.
    pub const fn new(min_ngram: usize, max_ngram: usize) -> Result<Self, ConfigError> {
        if min_ngram == 0 || min_ngram > max_ngram {
            return Err(ConfigError::NgramRange);
        }
        Ok(Self {
            min_ngram,
            max_ngram,
        })
    }

    /// Proposes up to `limit` tokens that followed the most recent earlier
    /// occurrence of the longest matching suffix of `history`. Returns an
    /// empty slice when nothing matches.
    ///
    /// shortcut: linear backward scan, O(`len * max_ngram`) per call; upgrade
    /// to a rolling-hash index when contexts beyond ~64K tokens show it in a
    /// profile.
    #[must_use]
    pub fn propose<'h>(&self, history: &'h [i32], limit: usize) -> &'h [i32] {
        if limit == 0 {
            return &[];
        }
        let len = history.len();
        for n in (self.min_ngram..=self.max_ngram.min(len.saturating_sub(1))).rev() {
            let suffix = &history[len - n..];
            // The newest occurrence is usually the most relevant (an edit
            // being replayed); start right before the suffix itself.
            for start in (0..len - n).rev() {
                if history[start..start + n] == *suffix {
                    let from = start + n;
                    let to = from.saturating_add(limit).min(len);
                    return &history[from..to];
                }
            }
        }
        &[]
    }
}

impl Default for PromptLookup {
    fn default() -> Self {
        Self {
            min_ngram: 2,
            max_ngram: 4,
        }
    }
}

/// Invalid speculation configuration.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// The n-gram range is empty or starts at zero.
    #[error("prompt lookup needs 1 <= min_ngram <= max_ngram")]
    NgramRange,
    /// The maximum draft length is zero.
    #[error("the maximum draft length must be positive")]
    DraftRange,
    /// A verification cost term is negative or non-finite.
    #[error("verification cost terms must be finite and nonnegative")]
    VerifyCost,
}

/// Cost of one verification pass relative to one ordinary decode step.
///
/// Scoring `k` draft tokens plus the last emitted token costs
/// `1 + fixed + per_token * k` decode steps. On Apple Silicon the shape is
/// not linear from zero: moving from a one-token decode to any multi-token
/// chunk changes the kernels used and the readback size, then each further
/// row adds little. Measure both terms on the target machine and model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerifyCost {
    fixed: f64,
    per_token: f64,
}

impl VerifyCost {
    /// Creates a cost model.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::VerifyCost`] for a negative or non-finite term.
    pub fn new(fixed: f64, per_token: f64) -> Result<Self, ConfigError> {
        if [fixed, per_token]
            .iter()
            .any(|term| !term.is_finite() || *term < 0.0)
        {
            return Err(ConfigError::VerifyCost);
        }
        Ok(Self { fixed, per_token })
    }

    /// Relative cost of verifying `drafted` tokens; zero is a plain decode.
    #[must_use]
    pub fn relative(self, drafted: usize) -> f64 {
        if drafted == 0 {
            return 1.0;
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "draft lengths are small integers"
        )]
        let drafted = drafted as f64;
        1.0 + self.per_token.mul_add(drafted, self.fixed)
    }
}

/// Chooses the draft length from the observed acceptance rate.
///
/// Acceptance is modeled as Leviathan et al.'s i.i.d. per-token rate `a`,
/// estimated from exponentially decayed counts of accepted tokens and
/// first-rejections (each step's draft stops at its first rejection, so
/// `accepted / drafted` would be biased low). The draft length maximizes the
/// expected emitted tokens per step, `(1 - a^(k+1)) / (1 - a)` (their Eq. 1),
/// per unit of [`VerifyCost`], which plays the role of their cost term when
/// the drafter itself is free. Zero, a plain decode step, wins when
/// acceptance is too low to pay for verification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftLength {
    max: usize,
    cost: VerifyCost,
    accepted: f64,
    rejected: f64,
}

/// Weight kept by older observations at each update.
const ACCEPTANCE_DECAY: f64 = 0.9;
/// Optimistic prior (a = 0.75), so the first matches are tried at a useful
/// length; a few rejections pull it down quickly.
const PRIOR_ACCEPTED: f64 = 3.0;
const PRIOR_REJECTED: f64 = 1.0;

impl DraftLength {
    /// Creates a controller choosing `0..=max` draft tokens.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::DraftRange`] when `max` is zero.
    pub const fn new(max: usize, cost: VerifyCost) -> Result<Self, ConfigError> {
        if max == 0 {
            return Err(ConfigError::DraftRange);
        }
        Ok(Self {
            max,
            cost,
            accepted: PRIOR_ACCEPTED,
            rejected: PRIOR_REJECTED,
        })
    }

    /// Estimated per-token acceptance rate.
    #[must_use]
    pub fn acceptance_rate(&self) -> f64 {
        self.accepted / (self.accepted + self.rejected)
    }

    /// Draft length to request next; zero means decode without drafting.
    #[must_use]
    pub fn next(&self) -> usize {
        let rate = self.acceptance_rate().min(1.0 - 1e-9);
        let mut best = (0, 1.0 / self.cost.relative(0));
        for k in 1..=self.max {
            let exponent = i32::try_from(k + 1).unwrap_or(i32::MAX);
            let tokens = (1.0 - rate.powi(exponent)) / (1.0 - rate);
            let value = tokens / self.cost.relative(k);
            if value > best.1 {
                best = (k, value);
            }
        }
        best.0
    }

    /// Records one verified draft.
    pub fn observe(&mut self, drafted: usize, accepted: usize) {
        if drafted == 0 {
            return;
        }
        #[allow(
            clippy::cast_precision_loss,
            reason = "per-step draft counts are small integers"
        )]
        let accepted_now = accepted.min(drafted) as f64;
        let rejected_now = if accepted < drafted { 1.0 } else { 0.0 };
        self.accepted = self.accepted.mul_add(ACCEPTANCE_DECAY, accepted_now);
        self.rejected = self.rejected.mul_add(ACCEPTANCE_DECAY, rejected_now);
    }

    /// Records a step decoded without a draft. The estimate relaxes toward
    /// the prior, so a stretch of low acceptance does not switch drafting
    /// off for the rest of the request.
    pub fn idle(&mut self) {
        let keep = ACCEPTANCE_DECAY;
        self.accepted = self.accepted.mul_add(keep, PRIOR_ACCEPTED * (1.0 - keep));
        self.rejected = self.rejected.mul_add(keep, PRIOR_REJECTED * (1.0 - keep));
    }
}

/// A request's speculation preference.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SpeculationRequest {
    /// Speculate only while this is the sole request in the engine.
    #[default]
    Automatic,
    /// Speculate even under load.
    Enabled,
    /// Never speculate.
    Disabled,
}

impl SpeculationRequest {
    /// Whether to speculate on the next step, given the other requests that
    /// are running or waiting besides this one. Under concurrency the spare
    /// compute that verification uses is already spent on other sequences,
    /// and published Mac measurements show speculation halving aggregate
    /// throughput at four concurrent requests.
    #[must_use]
    pub const fn allows(self, other_active: usize, queued: usize) -> bool {
        match self {
            Self::Automatic => other_active == 0 && queued == 0,
            Self::Enabled => true,
            Self::Disabled => false,
        }
    }
}

/// Per-request speculation counters for receipts and telemetry.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpeculationStats {
    /// Target passes that scored a nonempty draft.
    pub verify_steps: usize,
    /// Draft tokens scored.
    pub drafted_tokens: usize,
    /// Draft tokens the picker reproduced.
    pub accepted_tokens: usize,
}

impl SpeculationStats {
    /// Adds one step's outcome.
    pub const fn record(&mut self, outcome: &StepOutcome) {
        if outcome.drafted > 0 {
            self.verify_steps += 1;
            self.drafted_tokens += outcome.drafted;
            self.accepted_tokens += outcome.accepted;
        }
    }

    /// Accepted over drafted tokens, if anything was drafted.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "token counts are far below 2^52"
    )]
    pub fn acceptance_rate(&self) -> Option<f64> {
        (self.drafted_tokens > 0).then(|| self.accepted_tokens as f64 / self.drafted_tokens as f64)
    }
}

#[cfg(test)]
mod tests;
