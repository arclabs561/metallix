use std::convert::Infallible;

use super::{
    ConfigError, DraftLength, GreedySpeculativeTarget, Pick, PositionLogits, PromptLookup,
    SpeculationError, SpeculationRequest, SpeculationStats, SpeculativeTarget, StepOutcome,
    Verdict, VerifyCost, accept_or_resample, greedy_speculative_step, speculative_step,
};
use crate::sampling::sample_categorical;

const VOCAB: usize = 3;
const PROMPT: [i32; 3] = [0, 2, 1];

/// A tiny autoregressive "model" whose next-token logits depend on the last
/// two cached tokens and the sequence length. Because rows depend on the
/// cache contents, a step that fails to truncate rejected draft positions
/// produces different logits on the following step.
struct ToyModel {
    cache: Vec<i32>,
    verify_calls: usize,
}

impl ToyModel {
    fn prefilled(prompt: &[i32]) -> (Self, Vec<f32>) {
        let model = Self {
            cache: prompt.to_vec(),
            verify_calls: 0,
        };
        let logits = toy_logits(prompt);
        (model, logits)
    }
}

fn toy_logits(context: &[i32]) -> Vec<f32> {
    let last = context.last().copied().unwrap_or(0);
    let before = context
        .len()
        .checked_sub(2)
        .map_or(0, |index| context[index]);
    let length = i32::try_from(context.len()).expect("small context");
    (0..VOCAB)
        .map(|token| {
            let token = i32::try_from(token).expect("small vocabulary");
            let mix = u8::try_from((last * 7 + before * 3 + token * 5 + length) % 11)
                .expect("nonnegative residue");
            f32::from(mix) * 0.37 - if token == last { 0.8 } else { 0.0 }
        })
        .collect()
}

impl SpeculativeTarget for ToyModel {
    type Error = Infallible;

    fn cached_tokens(&self) -> usize {
        self.cache.len()
    }

    fn verify(&mut self, tokens: &[i32]) -> Result<PositionLogits, Infallible> {
        self.verify_calls += 1;
        let mut values = Vec::with_capacity(tokens.len() * VOCAB);
        for &token in tokens {
            self.cache.push(token);
            values.extend(toy_logits(&self.cache));
        }
        Ok(PositionLogits::new(values, VOCAB).expect("toy rows"))
    }

    fn truncate(&mut self, tokens: usize) -> Result<(), Infallible> {
        self.cache.truncate(tokens);
        Ok(())
    }
}

impl GreedySpeculativeTarget for ToyModel {
    fn verify_greedy(&mut self, tokens: &[i32]) -> Result<Vec<i32>, Infallible> {
        let rows = self.verify(tokens)?;
        Ok((0..rows.positions())
            .map(|index| greedy(rows.row(index).expect("row")))
            .collect())
    }
}

#[test]
fn device_greedy_step_reproduces_plain_greedy_and_stops() {
    let expected = decode_plain(40, &mut greedy);
    let mut rng = Rng(5);
    let lookup = PromptLookup::new(1, 3).expect("valid range");
    for random in [false, true] {
        let (mut model, logits) = ToyModel::prefilled(&PROMPT);
        let mut output = vec![greedy(&logits)];
        while output.len() < 40 {
            let mut history = PROMPT.to_vec();
            history.extend(&output);
            let limit = (40 - output.len() - 1).min(3);
            let draft = if random {
                random_draft(&mut rng)(&history, limit)
            } else {
                lookup.propose(&history, limit).to_vec()
            };
            let last = *output.last().expect("first token");
            let outcome = greedy_speculative_step(&mut model, last, &draft, &mut |_| false)
                .expect("toy model");
            output.extend(outcome.emitted);
            assert_eq!(model.cache.len(), PROMPT.len() + output.len() - 1);
        }
        assert_eq!(output, expected, "random={random}");
    }

    // A stop pick ends the step even when it matches the draft.
    let (mut model, logits) = ToyModel::prefilled(&PROMPT);
    let first = greedy(&logits);
    let draft = expected[1..4].to_vec();
    let stop = expected[2];
    let outcome = greedy_speculative_step(&mut model, first, &draft, &mut |token| token == stop)
        .expect("toy model");
    let cut = expected[1..]
        .iter()
        .position(|&token| token == stop)
        .expect("stop")
        + 1;
    assert_eq!(outcome.emitted, expected[1..=cut]);
    assert!(outcome.stopped);
}

/// `SplitMix64`, enough for reproducible test uniforms.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[allow(clippy::cast_precision_loss, reason = "53-bit mantissa draw")]
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn token(&mut self) -> i32 {
        let draw = usize::try_from(self.next_u64() % 1024).expect("bounded draw");
        i32::try_from(draw % VOCAB).expect("small vocabulary")
    }
}

fn greedy(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate() {
        if value > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).expect("small vocabulary")
}

fn sample(logits: &[f32], temperature: f64, uniform: f64) -> i32 {
    let legal = vec![true; logits.len()];
    let sampled = sample_categorical(logits, &legal, temperature, uniform).expect("valid toy row");
    i32::try_from(sampled.token_id).expect("small vocabulary")
}

/// Plain autoregressive decoding: one model call per token.
fn decode_plain(length: usize, pick: &mut dyn FnMut(&[f32]) -> i32) -> Vec<i32> {
    let (mut model, mut logits) = ToyModel::prefilled(&PROMPT);
    let mut output = Vec::new();
    while output.len() < length {
        let token = pick(&logits);
        output.push(token);
        if output.len() < length {
            logits = model
                .verify(&[token])
                .expect("toy model")
                .row(0)
                .expect("one row")
                .to_vec();
        }
    }
    output
}

/// Speculative decoding with drafts from `draft(history, limit)`.
fn decode_speculative(
    length: usize,
    pick: &mut dyn FnMut(&[f32]) -> i32,
    draft: &mut dyn FnMut(&[i32], usize) -> Vec<i32>,
) -> (Vec<i32>, SpeculationStats, usize) {
    let (mut model, logits) = ToyModel::prefilled(&PROMPT);
    let mut output = vec![pick(&logits)];
    let mut stats = SpeculationStats::default();
    while output.len() < length {
        let mut history = PROMPT.to_vec();
        history.extend(&output);
        // Never draft past the output budget: the bonus token fills the last slot.
        let proposal = draft(&history, length - output.len() - 1);
        let last = *output.last().expect("first token");
        let mut picker = |row: &[f32]| {
            Ok::<_, Infallible>(Pick {
                token: pick(row),
                stop: false,
            })
        };
        let outcome =
            speculative_step(&mut model, last, &proposal, &mut picker).expect("toy model");
        assert_eq!(
            model.cache.len(),
            PROMPT.len() + output.len() + outcome.emitted.len() - 1,
            "cache must hold every emitted token except the newest"
        );
        stats.record(&outcome);
        output.extend(&outcome.emitted);
    }
    assert_eq!(output.len(), length, "drafts never overrun the budget");
    (output, stats, model.verify_calls)
}

fn random_draft(rng: &mut Rng) -> impl FnMut(&[i32], usize) -> Vec<i32> + '_ {
    move |_, limit| {
        let k = usize::try_from(rng.next_u64() % 4)
            .expect("bounded draw")
            .min(limit);
        (0..k).map(|_| rng.token()).collect()
    }
}

#[test]
fn greedy_speculation_reproduces_plain_greedy_tokens() {
    let expected = decode_plain(40, &mut greedy);
    let lookup = PromptLookup::new(1, 3).expect("valid range");
    let (lookup_output, lookup_stats, calls) =
        decode_speculative(40, &mut greedy, &mut |history, limit| {
            lookup.propose(history, limit).to_vec()
        });
    assert_eq!(lookup_output, expected);
    assert!(
        lookup_stats.accepted_tokens > 0 && calls < 39,
        "the toy sequence repeats, so lookup should save model calls: {lookup_stats:?}, {calls} calls"
    );

    // Adversarial drafts exercise every rejection position.
    let mut rng = Rng(7);
    let (random_output, random_stats, _) =
        decode_speculative(40, &mut greedy, &mut random_draft(&mut rng));
    assert_eq!(random_output, expected);
    assert!(random_stats.drafted_tokens > random_stats.accepted_tokens);
}

#[test]
fn seeded_sampling_reproduces_plain_sampling_token_for_token() {
    // One uniform per emitted token in both modes, so a shared seed yields
    // the same sequence: the point-mass reduction of speculative sampling.
    for seed in 0..20 {
        let mut plain_rng = Rng(seed);
        let expected = decode_plain(24, &mut |row| sample(row, 0.9, plain_rng.uniform()));
        let mut spec_rng = Rng(seed);
        let mut draft_rng = Rng(seed ^ 0xABCD);
        let (output, _, _) = decode_speculative(
            24,
            &mut |row| sample(row, 0.9, spec_rng.uniform()),
            &mut random_draft(&mut draft_rng),
        );
        assert_eq!(output, expected, "seed {seed}");
    }
}

/// Exact probability of `sequence` as the first tokens after `PROMPT`.
fn exact_probability(sequence: &[i32], temperature: f64) -> f64 {
    let mut context = PROMPT.to_vec();
    let mut probability = 1.0;
    for &token in sequence {
        let logits = toy_logits(&context);
        let weights: Vec<f64> = logits
            .iter()
            .map(|&logit| (f64::from(logit) / temperature).exp())
            .collect();
        let index = usize::try_from(token).expect("nonnegative token");
        probability *= weights[index] / weights.iter().sum::<f64>();
        context.push(token);
    }
    probability
}

fn sequence_index(sequence: &[i32]) -> usize {
    sequence.iter().fold(0, |index, &token| {
        index * VOCAB + usize::try_from(token).expect("nonnegative token")
    })
}

fn all_sequences(length: usize) -> Vec<Vec<i32>> {
    let count = VOCAB.pow(u32::try_from(length).expect("small length"));
    (0..count)
        .map(|mut index| {
            let mut sequence = vec![0; length];
            for slot in sequence.iter_mut().rev() {
                *slot = i32::try_from(index % VOCAB).expect("small vocabulary");
                index /= VOCAB;
            }
            sequence
        })
        .collect()
}

/// Pearson chi-square of observed counts against the exact law.
fn chi_square(counts: &[usize], length: usize, temperature: f64, samples: usize) -> f64 {
    #[allow(clippy::cast_precision_loss, reason = "sample counts are small")]
    let samples = samples as f64;
    all_sequences(length)
        .iter()
        .map(|sequence| {
            let expected = exact_probability(sequence, temperature) * samples;
            #[allow(clippy::cast_precision_loss, reason = "sample counts are small")]
            let observed = counts[sequence_index(sequence)] as f64;
            (observed - expected).powi(2) / expected
        })
        .sum()
}

const LENGTH: usize = 4;
const SAMPLES: usize = 40_000;
// 81 cells, 80 degrees of freedom: mean 80, standard deviation ~12.6. The
// bound is ~6 standard deviations, so a correct sampler fails it with
// negligible probability, while a biased one (below) lands far beyond it.
const CHI_SQUARE_BOUND: f64 = 160.0;

#[test]
fn speculative_sampling_matches_the_exact_sequence_law() {
    let temperature = 0.8;
    let mut rng = Rng(11);
    let mut draft_rng = Rng(12);
    let mut counts = vec![0; VOCAB.pow(4)];
    for _ in 0..SAMPLES {
        let (output, _, _) = decode_speculative(
            LENGTH,
            &mut |row| sample(row, temperature, rng.uniform()),
            &mut random_draft(&mut draft_rng),
        );
        counts[sequence_index(&output)] += 1;
    }
    let statistic = chi_square(&counts, LENGTH, temperature, SAMPLES);
    assert!(statistic < CHI_SQUARE_BOUND, "chi-square {statistic}");
}

#[test]
fn chi_square_bound_rejects_a_biased_sampler() {
    // Calibrates the test above: accepting a draft token whenever it is
    // merely plausible (p > 0.2) instead of sampling biases the law.
    let temperature = 0.8;
    let mut rng = Rng(13);
    let mut counts = vec![0; VOCAB.pow(4)];
    for _ in 0..SAMPLES {
        let (mut model, mut logits) = ToyModel::prefilled(&PROMPT);
        let mut output = Vec::new();
        while output.len() < LENGTH {
            let guess = rng.token();
            let weights: Vec<f64> = logits
                .iter()
                .map(|&logit| (f64::from(logit) / temperature).exp())
                .collect();
            let index = usize::try_from(guess).expect("nonnegative token");
            let plausible = weights[index] / weights.iter().sum::<f64>() > 0.2;
            let token = if plausible {
                guess
            } else {
                sample(&logits, temperature, rng.uniform())
            };
            output.push(token);
            logits = model
                .verify(&[token])
                .expect("toy")
                .row(0)
                .expect("row")
                .to_vec();
        }
        counts[sequence_index(&output)] += 1;
    }
    let statistic = chi_square(&counts, LENGTH, temperature, SAMPLES);
    assert!(statistic > CHI_SQUARE_BOUND, "chi-square {statistic}");
}

fn probabilities(logits: &[f32], temperature: f64) -> Vec<f64> {
    let weights: Vec<f64> = logits
        .iter()
        .map(|&logit| (f64::from(logit) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|weight| weight / total).collect()
}

#[test]
fn stochastic_draft_rule_matches_the_exact_sequence_law() {
    // A draft "model" with a different, deliberately wrong distribution.
    let draft_probabilities = |context: &[i32]| {
        let last = usize::try_from(*context.last().expect("nonempty")).expect("token");
        let mut q = vec![0.15; VOCAB];
        q[(last + 1) % VOCAB] = 0.7;
        q
    };
    let temperature = 0.8;
    let mut rng = Rng(21);
    let mut counts = vec![0; VOCAB.pow(4)];
    let mut resampled = 0;
    for _ in 0..SAMPLES {
        let mut context = PROMPT.to_vec();
        while context.len() < PROMPT.len() + LENGTH {
            // Draft up to three tokens from q, then verify left to right.
            let mut drafts = Vec::new();
            let mut draft_context = context.clone();
            for _ in 0..3.min(PROMPT.len() + LENGTH - context.len()) {
                let q = draft_probabilities(&draft_context);
                let token = categorical_index(&q, rng.uniform());
                drafts.push((token, q));
                draft_context.push(i32::try_from(token).expect("token"));
            }
            let mut all_accepted = true;
            for (token, q) in drafts {
                let p = probabilities(&toy_logits(&context), temperature);
                match accept_or_resample(&p, &q, token, rng.uniform(), rng.uniform())
                    .expect("valid rule input")
                {
                    Verdict::Accepted => context.push(i32::try_from(token).expect("token")),
                    Verdict::Resampled(other) => {
                        resampled += 1;
                        context.push(i32::try_from(other).expect("token"));
                        all_accepted = false;
                        break;
                    }
                }
            }
            if all_accepted && context.len() < PROMPT.len() + LENGTH {
                let p = probabilities(&toy_logits(&context), temperature);
                let bonus = categorical_index(&p, rng.uniform());
                context.push(i32::try_from(bonus).expect("token"));
            }
        }
        counts[sequence_index(&context[PROMPT.len()..])] += 1;
    }
    assert!(
        resampled > SAMPLES / 4,
        "the wrong draft must be rejected often"
    );
    let statistic = chi_square(&counts, LENGTH, temperature, SAMPLES);
    assert!(statistic < CHI_SQUARE_BOUND, "chi-square {statistic}");
}

fn categorical_index(probabilities: &[f64], uniform: f64) -> usize {
    let mut cumulative = 0.0;
    for (index, &probability) in probabilities.iter().enumerate() {
        cumulative += probability;
        if uniform < cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

#[test]
fn rule_accepts_when_target_dominates_and_rejects_impossible_tokens() {
    let target = [0.5, 0.5, 0.0];
    let draft = [0.25, 0.25, 0.5];
    // p/q = 2 for token 0: always accepted.
    assert_eq!(
        accept_or_resample(&target, &draft, 0, 0.999, 0.0).expect("valid"),
        Verdict::Accepted
    );
    // Token 2 has no target mass: always replaced by a token with p > q.
    for uniform in [0.0, 0.3, 0.6, 0.999] {
        let verdict = accept_or_resample(&target, &draft, 2, 0.0, uniform).expect("valid");
        assert!(matches!(verdict, Verdict::Resampled(0 | 1)), "{verdict:?}");
    }
    assert!(matches!(
        accept_or_resample(&target, &draft, 3, 0.0, 0.0),
        Err(SpeculationError::TokenOutOfRange(3))
    ));
    assert!(matches!(
        accept_or_resample(&target, &draft[..2], 0, 0.0, 0.0),
        Err(SpeculationError::Probabilities)
    ));
    assert!(matches!(
        accept_or_resample(&target, &draft, 0, 1.0, 0.0),
        Err(SpeculationError::InvalidUniform)
    ));
}

#[test]
fn step_stops_at_a_stop_token_and_rolls_back_rejected_positions() {
    let (mut model, _) = ToyModel::prefilled(&PROMPT);
    let mut picks = [1, 2, 0].into_iter();
    let mut picker = |_: &[f32]| {
        let token = picks.next().expect("three rows");
        Ok::<_, Infallible>(Pick {
            token,
            stop: token == 2,
        })
    };
    let outcome = speculative_step(&mut model, 0, &[1, 2, 2], &mut picker).expect("toy");
    assert_eq!(
        outcome,
        StepOutcome {
            emitted: vec![1, 2],
            drafted: 3,
            accepted: 2,
            stopped: true,
        }
    );
    // Cached: prompt, last token 0, accepted draft 1; not the stop token or
    // the unverified third draft.
    assert_eq!(model.cache, [0, 2, 1, 0, 1]);
}

#[test]
fn step_rejects_a_target_that_returns_the_wrong_row_count() {
    struct Short;
    impl SpeculativeTarget for Short {
        type Error = Infallible;
        fn cached_tokens(&self) -> usize {
            1
        }
        fn verify(&mut self, _: &[i32]) -> Result<PositionLogits, Infallible> {
            Ok(PositionLogits::new(vec![0.0; VOCAB], VOCAB).expect("one row"))
        }
        fn truncate(&mut self, _: usize) -> Result<(), Infallible> {
            Ok(())
        }
    }
    let mut picker = |_: &[f32]| {
        Ok::<_, Infallible>(Pick {
            token: 0,
            stop: false,
        })
    };
    assert!(matches!(
        speculative_step(&mut Short, 0, &[1, 2], &mut picker),
        Err(SpeculationError::RowCount {
            expected: 3,
            actual: 1
        })
    ));
}

#[test]
fn position_logits_require_whole_rows() {
    assert!(PositionLogits::new(vec![0.0; 6], 3).is_ok());
    assert!(PositionLogits::new(vec![0.0; 5], 3).is_err());
    assert!(PositionLogits::new(Vec::new(), 3).is_err());
    assert!(PositionLogits::new(vec![0.0; 3], 0).is_err());
    let logits = PositionLogits::new(vec![1.0, 2.0, 3.0, 4.0], 2).expect("two rows");
    assert_eq!(logits.row(1), Some(&[3.0, 4.0][..]));
    assert_eq!(logits.row(2), None);
}

#[test]
fn prompt_lookup_proposes_what_followed_the_newest_longest_match() {
    let lookup = PromptLookup::new(1, 3).expect("valid");
    // Suffix [7, 8, 9] occurs at 0 (followed by 1) and at 5 (followed by 4):
    // the newer occurrence wins.
    let history = [7, 8, 9, 1, 2, 7, 8, 9, 4, 5, 6, 7, 8, 9];
    assert_eq!(lookup.propose(&history, 3), &[4, 5, 6]);
    assert_eq!(lookup.propose(&history, 1), &[4]);
    assert_eq!(lookup.propose(&history, 0), &[] as &[i32]);
    // Falls back to a shorter suffix: only [9] matches.
    assert_eq!(lookup.propose(&[9, 3, 5, 9], 2), &[3, 5]);
    // A match whose continuation is cut by the end of history is shorter.
    assert_eq!(lookup.propose(&[1, 2, 1, 2], 5), &[1, 2]);
    assert_eq!(lookup.propose(&[1, 2, 3], 4), &[] as &[i32]);
    assert_eq!(lookup.propose(&[], 4), &[] as &[i32]);
    assert_eq!(PromptLookup::new(0, 2), Err(ConfigError::NgramRange));
    assert_eq!(PromptLookup::new(3, 2), Err(ConfigError::NgramRange));
}

#[test]
fn draft_length_follows_acceptance_and_verify_cost() {
    // Shape measured for Qwen3-0.6B on one M-series Mac: any chunk costs
    // ~0.7 decode steps extra, each further row ~0.03.
    let measured = VerifyCost::new(0.7, 0.03).expect("valid cost");
    let mut controller = DraftLength::new(8, measured).expect("valid");
    for _ in 0..30 {
        controller.observe(8, 8);
    }
    assert_eq!(controller.next(), 8, "full acceptance drafts the maximum");
    for _ in 0..100 {
        controller.observe(4, 0);
    }
    assert!(controller.acceptance_rate() < 0.05);
    assert_eq!(controller.next(), 0, "constant rejection stops drafting");
    for _ in 0..40 {
        controller.idle();
    }
    assert!(
        controller.next() > 0,
        "idle steps relax toward the prior and drafting resumes"
    );

    // At a = 2/3 a steep per-row cost makes drafting a loss, a flat one
    // makes the longest draft best.
    let mut costly = DraftLength::new(8, VerifyCost::new(0.7, 0.5).expect("valid")).expect("valid");
    let mut cheap = DraftLength::new(8, VerifyCost::new(0.7, 0.01).expect("valid")).expect("valid");
    for _ in 0..30 {
        costly.observe(4, 2);
        cheap.observe(4, 2);
    }
    assert_eq!(costly.next(), 0);
    assert_eq!(cheap.next(), 8);

    // Discarding a queued step makes a short draft a loss: at a = 2/3 one
    // extra decode step of overhead outweighs one drafted token.
    assert_eq!(cheap.next_with_overhead(0.0), cheap.next());
    assert!(cheap.next_with_overhead(1.0) > 0);
    let mut short = DraftLength::new(1, VerifyCost::new(0.2, 0.05).expect("valid")).expect("valid");
    for _ in 0..30 {
        short.observe(1, 0);
        short.observe(1, 1);
        short.observe(1, 1);
    }
    assert_eq!(short.next(), 1);
    assert_eq!(short.next_with_overhead(1.0), 0);

    // A free verifier always drafts the maximum, even at low acceptance.
    let mut free = DraftLength::new(4, VerifyCost::new(0.0, 0.0).expect("valid")).expect("valid");
    free.observe(4, 0);
    assert_eq!(free.next(), 4);

    assert!((measured.relative(0) - 1.0).abs() < 1e-12);
    assert!((measured.relative(2) - 1.76).abs() < 1e-12);
    assert_eq!(DraftLength::new(0, measured), Err(ConfigError::DraftRange));
    assert_eq!(VerifyCost::new(-0.1, 0.0), Err(ConfigError::VerifyCost));
    assert_eq!(VerifyCost::new(0.1, f64::NAN), Err(ConfigError::VerifyCost));
}

#[test]
fn automatic_speculation_runs_only_alone() {
    assert!(SpeculationRequest::Automatic.allows(0, 0));
    assert!(!SpeculationRequest::Automatic.allows(1, 0));
    assert!(!SpeculationRequest::Automatic.allows(0, 1));
    assert!(SpeculationRequest::Enabled.allows(3, 5));
    assert!(!SpeculationRequest::Disabled.allows(0, 0));
}

#[test]
fn stats_count_only_steps_that_drafted() {
    let mut stats = SpeculationStats::default();
    assert_eq!(stats.acceptance_rate(), None);
    stats.record(&StepOutcome {
        emitted: vec![1],
        drafted: 0,
        accepted: 0,
        stopped: false,
    });
    stats.record(&StepOutcome {
        emitted: vec![1, 2, 3],
        drafted: 4,
        accepted: 2,
        stopped: false,
    });
    assert_eq!(stats.verify_steps, 1);
    assert_eq!(stats.acceptance_rate(), Some(0.5));
}
