//! All-position logits and cache rollback for speculative decoding.
//!
//! Oracles: an all-position row must match the last-position logits of the
//! same prefix computed by the existing append path, and a truncated cache
//! must continue exactly like a fresh prefill of the kept tokens. The greedy
//! test runs the engine's speculative step against this executor and
//! compares with plain greedy decoding.

use engine::speculative::{
    GreedySpeculativeTarget, Pick, PositionLogits, PromptLookup, SpeculativeTarget,
    greedy_speculative_step, speculative_step,
};

use super::prefix_extend::{sensitive_weights, two_layer_long_config};
use super::*;

const PROMPT: [i32; 6] = [1, 5, 2, 7, 3, 6];

fn resident<'a>(
    config: &'a Qwen3ForwardConfig,
    weights: &'a HashMap<String, Array>,
) -> Qwen3ForwardExecutor<'a, std::collections::hash_map::RandomState> {
    let plan = config
        .resident_chat_plan(600, u64::MAX, crate::forward::Qwen3FloatPrecision::Float32)
        .expect("tiny resident plan");
    Qwen3ForwardExecutor::new_for_resident_chat(config, weights, plan)
}

#[test]
fn every_chunk_row_matches_the_last_logits_of_its_prefix() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let chunk = [4, 2, 7, 0, 5];
    for diagnostic in [false, true] {
        let new = || {
            if diagnostic {
                Qwen3ForwardExecutor::new(&config, &weights)
            } else {
                resident(&config, &weights)
            }
        };
        let mut verifier = new();
        verifier.prefill_last_logits(&PROMPT).expect("prefill");
        let rows = verifier.extend_all_logits(&chunk).expect("all rows");
        assert_eq!(rows.positions(), chunk.len());
        assert_eq!(rows.vocab_size(), config.vocab_size);
        assert_eq!(verifier.cached_tokens(), PROMPT.len() + chunk.len());
        for end in 1..=chunk.len() {
            let mut oracle = new();
            oracle.prefill_last_logits(&PROMPT).expect("prefill");
            let expected = oracle.extend_last_logits(&chunk[..end]).expect("prefix");
            assert_logits_match(expected, rows.row(end - 1).expect("row").to_vec());
        }
        assert!(rows.row(chunk.len()).is_none());
    }
}

#[test]
fn prefill_rows_score_every_prompt_position() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let mut scorer = resident(&config, &weights);
    let rows = scorer.prefill_all_logits(&PROMPT).expect("scoring prefill");
    assert_eq!(rows.positions(), PROMPT.len());
    for end in 1..=PROMPT.len() {
        let expected = resident(&config, &weights)
            .prefill_last_logits(&PROMPT[..end])
            .expect("prefix prefill");
        assert_logits_match(expected, rows.row(end - 1).expect("row").to_vec());
    }
}

#[test]
fn truncated_cache_continues_like_a_fresh_prefill() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let long: Vec<i32> = (0..140).map(|index| (index * 5 + 3) % 8).collect();
    // (prefilled, kept): inside the 128 tier, and back across the 512 -> 128
    // tier boundary.
    for (prefilled, kept) in [(10, 7), (140, 120), (140, 139)] {
        for diagnostic in [false, true] {
            let new = || {
                if diagnostic {
                    Qwen3ForwardExecutor::new(&config, &weights)
                } else {
                    resident(&config, &weights)
                }
            };
            let mut rolled = new();
            rolled
                .prefill_last_logits(&long[..prefilled])
                .expect("prefill");
            rolled.truncate_cached_tokens(kept).expect("rollback");
            assert_eq!(rolled.cached_tokens(), kept);
            if !diagnostic {
                // Rollback returns storage to the tier forks check.
                rolled.fork_prefilled().expect("fork after rollback");
            }
            let mut fresh = new();
            fresh.prefill_last_logits(&long[..kept]).expect("fresh");
            // A different continuation than the dropped tokens, so stale
            // rows would show.
            for token in [6, 1, 4] {
                assert_logits_match(
                    fresh.decode_last_logits(token).expect("fresh decode"),
                    rolled.decode_last_logits(token).expect("rolled decode"),
                );
            }
        }
    }
}

#[test]
fn truncation_keeps_between_one_token_and_the_whole_cache() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let mut executor = resident(&config, &weights);
    executor.prefill_last_logits(&PROMPT).expect("prefill");
    assert!(matches!(
        executor.truncate_cached_tokens(PROMPT.len() + 1),
        Err(Qwen3ForwardError::TruncateOutOfRange {
            requested: 7,
            cached: 6
        })
    ));
    assert!(matches!(
        executor.truncate_cached_tokens(0),
        Err(Qwen3ForwardError::TruncateOutOfRange {
            requested: 0,
            cached: 6
        })
    ));
    executor
        .truncate_cached_tokens(PROMPT.len())
        .expect("no-op rollback");
    assert_eq!(executor.cached_tokens(), PROMPT.len());
    assert!(matches!(
        resident(&config, &weights).extend_all_logits(&[1]),
        Err(Qwen3ForwardError::DecodeWithoutPrefill)
    ));
}

struct Target<'e, 'a>(&'e mut Qwen3ForwardExecutor<'a, std::collections::hash_map::RandomState>);

impl SpeculativeTarget for Target<'_, '_> {
    type Error = Qwen3ForwardError;

    fn cached_tokens(&self) -> usize {
        self.0.cached_tokens()
    }

    fn verify(&mut self, tokens: &[i32]) -> Result<PositionLogits, Qwen3ForwardError> {
        let rows = self.0.extend_all_logits(tokens)?;
        let vocab = rows.vocab_size();
        Ok(PositionLogits::new(rows.into_values(), vocab).expect("whole rows"))
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

#[test]
fn gpu_greedy_verify_matches_host_argmax_of_every_row() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let chunk = [4, 2, 7, 0, 5];
    let mut host = resident(&config, &weights);
    host.prefill_last_logits(&PROMPT).expect("prefill");
    let rows = host.extend_all_logits(&chunk).expect("rows");
    let mut device = resident(&config, &weights);
    device.prefill_last_logits(&PROMPT).expect("prefill");
    let picks = device
        .extend_greedy(&chunk)
        .expect("picks")
        .wait()
        .expect("wait");
    let expected: Vec<i32> = (0..chunk.len())
        .map(|index| argmax(rows.row(index).expect("row")))
        .collect();
    assert_eq!(picks, expected);
    assert_eq!(device.cached_tokens(), PROMPT.len() + chunk.len());
    assert!(matches!(
        resident(&config, &weights).extend_greedy(&[1]),
        Err(Qwen3ForwardError::DecodeWithoutPrefill)
    ));
}

#[test]
fn gpu_greedy_speculation_reproduces_plain_greedy_decoding() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let length = 40;
    let expected = decode_plain(&config, &weights, length, |row, _| argmax(row));
    let mut executor = resident(&config, &weights);
    let first = executor.prefill_last_logits(&PROMPT).expect("prefill");
    let mut output = vec![argmax(&first)];
    let (mut drafted, mut accepted) = (0, 0);
    while output.len() < length {
        let limit = (length - output.len() - 1).min(4);
        // Alternate a likely draft (the last token repeated, which this
        // fixture mostly emits) with guesses that miss.
        let draft: Vec<i32> = (0..limit)
            .map(|offset| {
                if output.len() % 3 == 0 {
                    i32::try_from((output.len() + offset * 5) % 8).expect("small")
                } else {
                    *output.last().expect("first token")
                }
            })
            .collect();
        let last = *output.last().expect("first token");
        let outcome =
            greedy_speculative_step(&mut Target(&mut executor), last, &draft, &mut |_| false)
                .expect("speculative step");
        drafted += outcome.drafted;
        accepted += outcome.accepted;
        output.extend(outcome.emitted);
        assert_eq!(executor.cached_tokens(), PROMPT.len() + output.len() - 1);
    }
    assert_eq!(output, expected);
    assert!(accepted > 0 && drafted > accepted, "{accepted}/{drafted}");
}

fn argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate() {
        if value > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).expect("small vocabulary")
}

/// Gumbel-max sampling with a fixed noise stream: emitted token `n` is
/// `argmax(logits + g_n)`, an exact softmax sample whose randomness depends
/// only on `n`. Plain and speculative decoding call it with the same `n`
/// for the same output position, so they must agree token for token.
fn gumbel_pick(logits: &[f32], emitted: usize) -> i32 {
    let noisy = logits
        .iter()
        .enumerate()
        .map(|(token, &logit)| {
            let mut state = (emitted * 8191 + token * 131_071 + 17) as u64;
            state = state.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            state ^= state >> 29;
            state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            state ^= state >> 32;
            let uniform = (f64::from(u32::try_from(state >> 40).expect("24 bits")) + 0.5)
                / f64::from(1_u32 << 24);
            #[allow(
                clippy::cast_possible_truncation,
                reason = "noise needs f32 range only"
            )]
            let gumbel = (-(-uniform.ln()).ln()) as f32;
            logit + gumbel
        })
        .collect::<Vec<_>>();
    argmax(&noisy)
}

type Picker = fn(&[f32], usize) -> i32;

fn decode_plain(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array>,
    length: usize,
    pick: Picker,
) -> Vec<i32> {
    let mut plain = resident(config, weights);
    let mut logits = plain.prefill_last_logits(&PROMPT).expect("prefill");
    let mut output = Vec::new();
    while output.len() < length {
        let token = pick(&logits, output.len());
        output.push(token);
        logits = plain.decode_last_logits(token).expect("decode");
    }
    output
}

#[test]
fn speculation_reproduces_plain_decoding_on_the_executor() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = two_layer_long_config();
    let weights = sensitive_weights();
    let length = 48;
    let lookup = PromptLookup::new(1, 3).expect("valid range");
    let pickers: [(&str, Picker); 2] = [("greedy", |row, _| argmax(row)), ("gumbel", gumbel_pick)];
    for (name, picker) in pickers {
        let expected = decode_plain(&config, &weights, length, picker);
        // Prompt lookup, plus guesses that vary per step, cover acceptance
        // and rejection at every draft position.
        for wrong_guess in [false, true] {
            let mut executor = resident(&config, &weights);
            let first = executor.prefill_last_logits(&PROMPT).expect("prefill");
            let mut output = vec![picker(&first, 0)];
            let (mut drafted, mut accepted) = (0, 0);
            while output.len() < length {
                let mut history = PROMPT.to_vec();
                history.extend(&output);
                let limit = (length - output.len() - 1).min(4);
                let draft = if wrong_guess {
                    (0..limit)
                        .map(|offset| {
                            i32::try_from((output.len() * 3 + offset * 5) % 8).expect("small")
                        })
                        .collect()
                } else {
                    lookup.propose(&history, limit).to_vec()
                };
                let mut emitted = output.len();
                let mut pick = |row: &[f32]| {
                    let token = picker(row, emitted);
                    emitted += 1;
                    Ok::<_, std::convert::Infallible>(Pick { token, stop: false })
                };
                let last = *output.last().expect("first token");
                let outcome = speculative_step(&mut Target(&mut executor), last, &draft, &mut pick)
                    .expect("speculative step");
                drafted += outcome.drafted;
                accepted += outcome.accepted;
                output.extend(outcome.emitted);
                assert_eq!(executor.cached_tokens(), PROMPT.len() + output.len() - 1);
            }
            println!(
                "picker={name} wrong_guess={wrong_guess} drafted={drafted} accepted={accepted} tokens={output:?}"
            );
            assert_eq!(output, expected, "picker={name} wrong_guess={wrong_guess}");
            assert!(accepted > 0, "picker={name} wrong_guess={wrong_guess}");
            if wrong_guess {
                assert!(drafted > accepted, "rejections were exercised");
            }
        }
        if name == "gumbel" {
            let mut distinct = expected.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert!(
                distinct.len() >= 4,
                "sampled sequence is varied: {expected:?}"
            );
        }
    }
}
