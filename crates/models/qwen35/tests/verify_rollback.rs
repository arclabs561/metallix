//! Opt-in checks of speculative verify and rollback against the plain decode
//! loop on a real checkpoint.
//!
//! Set `METALLIX_QWEN35_MODEL` to a Qwen3.5 checkpoint directory. Token
//! streams come from the prompts in
//! `fixtures/qwen3.5-0.8b/hybrid-reference.json`.
//!
//! A verify computes its rows with matrix products and the chunked delta rule,
//! while decode uses matrix-vector products and the token-by-token recurrence,
//! so the two disagree in low bits. Rows are compared under two rules: the
//! greedy token must match wherever the decode row's top two logits are more
//! than [`NEAR_TIE`] apart, and the largest |logit difference| must stay under
//! a bound measured by `verify_rows_vs_decode_baseline`.

#![cfg(feature = "metal")]

use std::{env, path::PathBuf};

use mlx_rs::Array;
use qwen35::forward::{Qwen35Error, Qwen35Executor, Qwen35Precision, Qwen35Taps, Qwen35Weights};
use serde_json::Value;

/// Top-1 minus top-2 logit gap at or below which a greedy flip between
/// verify and decode is accepted as rounding.
const NEAR_TIE: f32 = 0.5;

/// Draft length of every verify.
const DRAFT: usize = 8;

/// Largest accepted |logit difference| between a verify or rolled-back row
/// and the decode loop, per precision. On Qwen3.5-0.8B,
/// `verify_rows_vs_decode_baseline` measured at most 0.25 at BF16 (one BF16
/// step for logits between 32 and 64) and 4.3e-5 at f32 over 232 rows; a
/// rollback that keeps a wrong row differs by 15 or more.
fn bound(precision: Qwen35Precision) -> f32 {
    match precision {
        Qwen35Precision::Float32 => 1e-3,
        Qwen35Precision::Checkpoint => 0.5,
    }
}

fn model() -> Option<PathBuf> {
    let model = env::var_os("METALLIX_QWEN35_MODEL").map(PathBuf::from);
    if model.is_none() {
        eprintln!("skipping: METALLIX_QWEN35_MODEL is not set");
    }
    model
}

/// The longest fixture prompt.
fn prompt() -> Vec<i32> {
    let reference: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/qwen3.5-0.8b/hybrid-reference.json"
    ))
    .expect("reference JSON");
    reference["cases"][3]["steps"][0]["input_ids"]
        .as_array()
        .expect("id array")
        .iter()
        .map(|id| i32::try_from(id.as_i64().expect("integer id")).expect("i32 id"))
        .collect()
}

/// `length` tokens cycling the prompt, so any position has a fixed token.
fn stream(length: usize) -> Vec<i32> {
    prompt().into_iter().cycle().take(length).collect()
}

/// A wrong draft token for position `position`: a different token from the
/// same stream, so a rollback that keeps rejected tokens changes the state.
fn rejected(stream: &[i32], position: usize) -> i32 {
    let token = stream[(position + 61) % stream.len()];
    if token == stream[position] {
        token + 1
    } else {
        token
    }
}

fn first_argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (index, &logit) in row.iter().enumerate() {
        if logit > row[best] {
            best = index;
        }
    }
    best
}

fn top_gap(row: &[f32]) -> f32 {
    let best = first_argmax(row);
    let second = row
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != best)
        .map(|(_, &logit)| logit)
        .fold(f32::NEG_INFINITY, f32::max);
    row[best] - second
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// Running comparison of rows against the decode loop's rows.
#[derive(Clone, Default)]
struct Tally {
    rows: usize,
    max_diff: f32,
    flips: usize,
    /// Largest decode-row top-two gap among greedy flips.
    max_flip_gap: f32,
}

impl Tally {
    fn add(&mut self, row: &[f32], decode: &[f32]) {
        self.rows += 1;
        self.max_diff = self.max_diff.max(max_abs_diff(row, decode));
        if first_argmax(row) != first_argmax(decode) {
            self.flips += 1;
            self.max_flip_gap = self.max_flip_gap.max(top_gap(decode));
        }
    }

    fn check(&self, what: &str, precision: Qwen35Precision) {
        eprintln!(
            "{what} at {precision:?}: {} rows, max |dlogit| {:.3e}, {} greedy flips (largest gap {:.3})",
            self.rows, self.max_diff, self.flips, self.max_flip_gap
        );
        assert!(
            self.max_flip_gap <= NEAR_TIE,
            "{what}: greedy flip with a top-two gap of {}",
            self.max_flip_gap
        );
        assert!(
            self.max_diff <= bound(precision),
            "{what}: max |dlogit| {} over {}",
            self.max_diff,
            bound(precision)
        );
    }
}

fn verify_rows(executor: &mut Qwen35Executor<'_>, tokens: &[i32]) -> Vec<Vec<f32>> {
    let logits = executor
        .verify_logits(tokens, Qwen35Taps::None)
        .expect("verify")
        .rows;
    let vocab = logits.len() / tokens.len();
    assert_eq!(vocab * tokens.len(), logits.len());
    logits.chunks_exact(vocab).map(<[f32]>::to_vec).collect()
}

/// Verify rows from a shared state against decoding the same tokens one at a
/// time, at several positions. Prints the distribution the bounds above are
/// set from; it asserts nothing.
#[test]
#[ignore = "measurement; run with --ignored"]
fn verify_rows_vs_decode_baseline() {
    let Some(model) = model() else { return };
    for precision in [Qwen35Precision::Checkpoint, Qwen35Precision::Float32] {
        let weights = Qwen35Weights::load(&model, precision).expect("load");
        let tokens = stream(1_024);
        let mut decode = weights.executor();
        decode.prefill_last_logits(&tokens[..100]).expect("prefill");
        let mut by_row = vec![Tally::default(); DRAFT];
        let mut start = 100;
        // Each round compares DRAFT rows, then decodes 23 more tokens.
        while start + DRAFT + 23 <= tokens.len() {
            let draft = &tokens[start..start + DRAFT];
            let snapshot = decode.snapshot().expect("snapshot");
            let mut verify = weights.executor_from(&snapshot).expect("restore");
            let rows = verify_rows(&mut verify, draft);
            for (index, (row, &token)) in rows.iter().zip(draft).enumerate() {
                let expected = decode.decode_last_logits(token).expect("decode");
                by_row[index].add(row, &expected);
            }
            start += DRAFT + 23;
            for &token in &tokens[start - 23..start] {
                decode.decode_last_logits(token).expect("advance");
            }
        }
        for (index, tally) in by_row.iter().enumerate() {
            eprintln!(
                "{precision:?} row {index}: {} rows, max |dlogit| {:.3e}, {} flips (largest gap {:.3})",
                tally.rows, tally.max_diff, tally.flips, tally.max_flip_gap
            );
        }
    }
}

/// Keeping `keep` of 8 verified tokens, for every `keep`, leaves the state the
/// decode loop reaches by consuming those `keep` tokens: the kept rows match
/// it, and so do the next four decode steps after the rollback.
#[test]
fn partial_accept_matches_the_decode_loop() {
    let Some(model) = model() else { return };
    let tokens = stream(256);
    let start = 133;
    for precision in [Qwen35Precision::Checkpoint, Qwen35Precision::Float32] {
        let weights = Qwen35Weights::load(&model, precision).expect("load");
        let mut prefix = weights.executor();
        prefix
            .prefill_last_logits(&tokens[..start])
            .expect("prefill");
        let prefix = prefix.snapshot().expect("snapshot");
        let mut kept_rows = Tally::default();
        let mut after = Tally::default();
        for keep in 0..=DRAFT {
            let mut draft = tokens[start..start + keep].to_vec();
            draft.extend((start + keep..start + DRAFT).map(|p| rejected(&tokens, p)));

            let mut speculative = weights.executor_from(&prefix).expect("restore");
            let rows = verify_rows(&mut speculative, &draft);
            speculative.truncate_to(start + keep).expect("truncate");
            assert_eq!(speculative.tokens(), start + keep);

            let mut decode = weights.executor_from(&prefix).expect("restore");
            for (row, &token) in rows.iter().zip(&tokens[start..start + keep]) {
                kept_rows.add(row, &decode.decode_last_logits(token).expect("decode"));
            }
            for &token in &tokens[start + keep..start + keep + 4] {
                after.add(
                    &speculative.decode_last_logits(token).expect("resumed"),
                    &decode.decode_last_logits(token).expect("decode"),
                );
            }
        }
        kept_rows.check("kept verify rows", precision);
        after.check("decode after rollback", precision);
    }
}

/// The greedy verify picks the first maximum of each verify row.
#[test]
fn greedy_verify_picks_each_row_argmax() {
    let Some(model) = model() else { return };
    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    let tokens = stream(160);
    let mut prefix = weights.executor();
    prefix.prefill_last_logits(&tokens[..150]).expect("prefill");
    let prefix = prefix.snapshot().expect("snapshot");
    let rows = verify_rows(
        &mut weights.executor_from(&prefix).expect("restore"),
        &tokens[150..158],
    );
    let picks = weights
        .executor_from(&prefix)
        .expect("restore")
        .verify_greedy(&tokens[150..158], Qwen35Taps::None)
        .expect("greedy verify")
        .rows;
    let expected: Vec<i32> = rows
        .iter()
        .map(|row| i32::try_from(first_argmax(row)).expect("token id"))
        .collect();
    assert_eq!(picks, expected);
}

/// Truncation needs a pending verify and a length inside it; a refused
/// truncation leaves the verify pending, and an ordinary extend commits it.
#[test]
fn truncate_is_bounded_by_the_pending_verify() {
    let Some(model) = model() else { return };
    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    let tokens = stream(64);
    let mut executor = weights.executor();
    assert!(matches!(
        executor.truncate_to(0),
        Err(Qwen35Error::NoPendingVerify)
    ));
    executor
        .prefill_last_logits(&tokens[..40])
        .expect("prefill");
    let maximum = executor.max_verify_tokens();
    assert!(maximum >= DRAFT, "max_verify_tokens {maximum}");
    assert!(matches!(
        executor.verify_logits(&stream(maximum + 1), Qwen35Taps::None),
        Err(Qwen35Error::VerifyLength { requested, maximum: m }) if requested == maximum + 1 && m == maximum
    ));
    executor
        .verify_logits(&tokens[40..44], Qwen35Taps::None)
        .expect("verify");
    for requested in [39, 45] {
        assert!(matches!(
            executor.truncate_to(requested),
            Err(Qwen35Error::TruncateRange {
                minimum: 40,
                maximum: 44,
                ..
            })
        ));
    }
    executor.truncate_to(42).expect("truncate");
    assert_eq!(executor.tokens(), 42);
    assert!(matches!(
        executor.truncate_to(42),
        Err(Qwen35Error::NoPendingVerify)
    ));
    executor
        .verify_logits(&tokens[42..46], Qwen35Taps::None)
        .expect("verify");
    executor.decode_last_logits(tokens[46]).expect("commit");
    assert_eq!(executor.tokens(), 47);
    assert!(matches!(
        executor.truncate_to(44),
        Err(Qwen35Error::NoPendingVerify)
    ));
}

/// A speculative loop over 16K tokens tracks the decode loop at every commit.
/// Each round verifies the next stream tokens it keeps plus rejected tokens,
/// with the kept count cycling 0..=8, and rolls back; every kept row is
/// compared with the decode loop's row at the same position, and every 64th
/// round both loops' states decode the next token from a snapshot. Release
/// build and a heavy slot: the decode loop alone is 16K steps.
#[test]
#[ignore = "long; run with --ignored in release"]
fn long_speculative_run_tracks_the_decode_loop() {
    let Some(model) = model() else { return };
    let precision = Qwen35Precision::Float32;
    let weights = Qwen35Weights::load(&model, precision).expect("load");
    let total = 16_500;
    let tokens = stream(total + DRAFT + 1);
    let mut speculative = weights.executor();
    let mut decode = weights.executor();
    speculative
        .prefill_last_logits(&tokens[..300])
        .expect("prefill");
    decode.prefill_last_logits(&tokens[..300]).expect("prefill");
    let mut rows_tally = Tally::default();
    let mut probes = Tally::default();
    let mut position = 300;
    let mut round = 0_usize;
    // Kept counts follow a fixed permutation so every value recurs often.
    let keeps = [5, 0, 8, 3, 7, 1, 6, 2, 4];
    while position < total {
        let keep = keeps[round % keeps.len()];
        let mut draft = tokens[position..position + keep].to_vec();
        draft.extend((position + keep..position + DRAFT).map(|p| rejected(&tokens, p)));
        let rows = verify_rows(&mut speculative, &draft);
        speculative.truncate_to(position + keep).expect("truncate");
        for (row, &token) in rows.iter().zip(&tokens[position..position + keep]) {
            rows_tally.add(row, &decode.decode_last_logits(token).expect("decode"));
        }
        position += keep;
        round += 1;
        if round.is_multiple_of(64) {
            let next = tokens[position];
            let probe = |executor: &Qwen35Executor<'_>| {
                weights
                    .executor_from(&executor.snapshot().expect("snapshot"))
                    .expect("restore")
                    .decode_last_logits(next)
                    .expect("probe")
            };
            probes.add(&probe(&speculative), &probe(&decode));
        }
    }
    eprintln!("{round} rounds to {position} tokens");
    rows_tally.check("kept verify rows", precision);
    probes.check("decode from committed state", precision);
}

/// An array's values as f32 on the host.
fn host(array: &Array) -> Vec<f32> {
    let array = array.as_type::<f32>().expect("f32");
    array.eval().expect("eval");
    array.as_slice::<f32>().to_vec()
}

/// One sink call: the chunk's first position, its row count and its values.
type TapChunk = (usize, i32, Vec<f32>);

/// Extends `executor` by `tokens` in pieces of `piece`, collecting what the
/// tap sink receives.
fn tapped_extend(
    executor: &mut Qwen35Executor<'_>,
    tokens: &[i32],
    piece: usize,
    taps: Qwen35Taps<'_>,
) -> Vec<TapChunk> {
    let mut chunks = Vec::new();
    for part in tokens.chunks(piece) {
        executor
            .extend_last_logits_tapped(part, taps, |rows, first| {
                chunks.push((first, rows.shape()[0], host(rows)));
                Ok(())
            })
            .expect("tapped extend");
    }
    chunks
}

/// Prefill taps reach the sink once per evaluation chunk, in order, with
/// each chunk's first position, and do not depend on how the prompt is split
/// into calls.
#[test]
fn prefill_taps_stream_one_chunk_at_a_time() {
    let Some(model) = model() else { return };
    let precision = Qwen35Precision::Float32;
    let weights = Qwen35Weights::load(&model, precision).expect("load");
    let hidden = weights.config().hidden_size();
    let tokens = stream(300);

    let mut whole = weights.executor();
    let chunks = tapped_extend(&mut whole, &tokens, tokens.len(), Qwen35Taps::FinalNormed);
    let shape: Vec<(usize, i32)> = chunks.iter().map(|c| (c.0, c.1)).collect();
    assert_eq!(shape, [(0, 128), (128, 128), (256, 44)]);
    for (_, rows, values) in &chunks {
        assert_eq!(values.len(), usize::try_from(*rows).expect("rows") * hidden);
    }

    let mut split = weights.executor();
    let pieces = tapped_extend(&mut split, &tokens, 100, Qwen35Taps::FinalNormed);
    let firsts: Vec<usize> = pieces.iter().map(|c| c.0).collect();
    assert_eq!(firsts, [0, 100, 200]);
    let joined = |chunks: &[TapChunk]| -> Vec<f32> {
        chunks.iter().flat_map(|c| c.2.iter().copied()).collect()
    };
    let diff = max_abs_diff(&joined(&chunks), &joined(&pieces));
    eprintln!("final-normed taps, one call vs three: max |diff| {diff:.3e}");
    assert!(diff <= bound(precision), "max |diff| {diff}");
}

/// A verify's taps are the rows an ordinary extend of the same tokens hands
/// to its sink (the same forward, so bit-identical), and a layer list
/// concatenates each layer's rows in list order.
#[test]
fn verify_taps_match_extend_taps() {
    let Some(model) = model() else { return };
    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    let hidden = weights.config().hidden_size();
    let last = weights.config().layers().len() - 1;
    let tokens = stream(160);
    let mut prefix = weights.executor();
    prefix.prefill_last_logits(&tokens[..150]).expect("prefill");
    let prefix = prefix.snapshot().expect("snapshot");
    let draft = &tokens[150..158];

    for taps in [Qwen35Taps::FinalNormed, Qwen35Taps::Layers(&[last, 0])] {
        let verified = weights
            .executor_from(&prefix)
            .expect("restore")
            .verify_greedy(draft, taps)
            .expect("verify");
        let verified = host(verified.taps.as_ref().expect("taps"));
        let extended = tapped_extend(
            &mut weights.executor_from(&prefix).expect("restore"),
            draft,
            draft.len(),
            taps,
        );
        assert_eq!(extended.len(), 1);
        assert_eq!(extended[0].0, 150);
        assert_eq!(extended[0].2, verified, "{taps:?}");
    }

    let both = weights
        .executor_from(&prefix)
        .expect("restore")
        .verify_logits(draft, Qwen35Taps::Layers(&[last, 0]))
        .expect("verify");
    let both = host(both.taps.as_ref().expect("taps"));
    assert_eq!(both.len(), draft.len() * 2 * hidden);
    for layer_slot in [0, 1] {
        let layer = [last, 0][layer_slot];
        let single = weights
            .executor_from(&prefix)
            .expect("restore")
            .verify_logits(draft, Qwen35Taps::Layers(&[layer]))
            .expect("verify");
        let single = host(single.taps.as_ref().expect("taps"));
        let columns: Vec<f32> = both
            .chunks_exact(2 * hidden)
            .flat_map(|row| row[layer_slot * hidden..(layer_slot + 1) * hidden].to_vec())
            .collect();
        assert_eq!(columns, single, "layer {layer}");
    }
}

/// An empty layer list or a layer past the last is refused before anything
/// runs, so no verify is left pending.
#[test]
fn bad_tap_lists_are_refused() {
    let Some(model) = model() else { return };
    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    let layers = weights.config().layers().len();
    let tokens = stream(48);
    let mut executor = weights.executor();
    executor
        .prefill_last_logits(&tokens[..40])
        .expect("prefill");
    assert!(matches!(
        executor.verify_greedy(&tokens[40..44], Qwen35Taps::Layers(&[])),
        Err(Qwen35Error::TapLayer { layer: None, layers: l }) if l == layers
    ));
    assert!(matches!(
        executor.verify_greedy(&tokens[40..44], Qwen35Taps::Layers(&[0, layers])),
        Err(Qwen35Error::TapLayer { layer: Some(l), .. }) if l == layers
    ));
    assert!(matches!(
        executor.truncate_to(40),
        Err(Qwen35Error::NoPendingVerify)
    ));
    assert!(matches!(
        executor.extend_last_logits_tapped(
            &tokens[40..44],
            Qwen35Taps::Layers(&[layers]),
            |_, _| Ok(())
        ),
        Err(Qwen35Error::TapLayer { .. })
    ));
    assert_eq!(executor.tokens(), 40);
}
