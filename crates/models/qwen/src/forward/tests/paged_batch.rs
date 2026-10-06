//! Batched paged decode against single-sequence references.
//!
//! A batched step runs matrix-matrix kernels and a masked, padded SDPA, so
//! its logits match a single decode only to rounding. The gate:
//!
//! - one row: bit-identical to the contiguous executor;
//! - several rows: every logit within `tolerance` of the row's reference,
//!   and a greedy token may differ only where the reference's top-two gap is
//!   below `2 * tolerance`. That bound is implied by the first (two logits
//!   each off by at most `tolerance` can swap order only if they were within
//!   `2 * tolerance`), so it is asserted as a consistency check and the
//!   divergence rate is printed.
//!
//! References are teacher-forced: each reference decodes the token the
//! batch chose, so one divergence does not cascade into the next steps.

use std::{collections::hash_map::RandomState, time::Instant};

use engine::blocks::{BlockTokens, PoolConfig, SequenceId};

use super::paged_kv::{MIB, fixed_tokens, millis, paged_config, paged_weights, pool, prompt};
use super::*;
use crate::forward::{BatchDecoded, BatchReadback, PagedQwen3Session, Qwen3WeightPrecision};

/// Largest batched-vs-single logit difference accepted on the fixture.
/// Declared after the first measurement, whose largest difference over
/// B = 2..16 was 7.0e-6 (printed as `max_abs_diff`).
const FIXTURE_TOLERANCE: f32 = 1e-4;

/// The lowest index of the largest logit, matching MLX's `argmax`.
fn first_argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).expect("vocabulary index fits i32")
}

/// Top-1 minus top-2 logit.
fn top_gap(logits: &[f32]) -> f32 {
    let mut sorted = logits.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    sorted[0] - sorted[1]
}

fn logits(decoded: BatchDecoded) -> Vec<Vec<f32>> {
    match decoded {
        BatchDecoded::Logits(rows) => rows,
        BatchDecoded::Greedy(_) => panic!("asked for logits"),
    }
}

#[derive(Debug, Default)]
struct Agreement {
    row_steps: usize,
    divergences: usize,
    max_abs_diff: f32,
}

impl Agreement {
    /// Records one row-step; panics when the D2 rule is broken.
    fn check(&mut self, batched: &[f32], reference: &[f32], tolerance: f32, context: &str) {
        assert_eq!(batched.len(), reference.len(), "{context}: logit count");
        let diff = batched
            .iter()
            .zip(reference)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            diff <= tolerance,
            "{context}: max |batched - reference| {diff} exceeds {tolerance}"
        );
        self.max_abs_diff = self.max_abs_diff.max(diff);
        self.row_steps += 1;
        if first_argmax(batched) != first_argmax(reference) {
            let gap = top_gap(reference);
            assert!(
                gap < 2.0 * tolerance,
                "{context}: greedy token diverged with reference top-two gap {gap}"
            );
            self.divergences += 1;
        }
    }
}

/// Prefills one row per prompt on both paths; rows `0` and `1` are a fork
/// pair when `fork` is set (row 1 shares row 0's blocks).
fn start_rows<'a>(
    paged: &mut PagedQwen3Session<'a, RandomState>,
    config: &'a Qwen3ForwardConfig,
    weights: &'a HashMap<String, Array>,
    prompts: &[Vec<i32>],
    fork: bool,
) -> (
    Vec<SequenceId>,
    Vec<Qwen3ForwardExecutor<'a, RandomState>>,
    Vec<Vec<f32>>,
) {
    let mut seqs: Vec<SequenceId> = Vec::new();
    let mut references: Vec<Qwen3ForwardExecutor<'a, RandomState>> = Vec::new();
    let mut last: Vec<Vec<f32>> = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let seq = SequenceId(index as u64 + 1);
        if fork && index == 1 {
            paged.fork(seqs[0], seq).expect("fork");
            references.push(references[0].fork_prefilled().expect("reference fork"));
            last.push(last[0].clone());
        } else {
            last.push(paged.prefill_last_logits(seq, prompt).expect("prefill"));
            let mut reference = Qwen3ForwardExecutor::new_for_resident_chat(
                config,
                weights,
                config
                    .resident_chat_plan(
                        2048,
                        u64::MAX,
                        crate::forward::Qwen3WeightPrecision::Float32,
                    )
                    .expect("plan"),
            );
            reference
                .prefill_last_logits(prompt)
                .expect("reference prefill");
            references.push(reference);
        }
        seqs.push(seq);
    }
    (seqs, references, last)
}

/// Teacher-forced batched greedy decode for `steps` steps.
#[allow(clippy::too_many_arguments, reason = "test driver")]
fn run_batch<'a>(
    paged: &mut PagedQwen3Session<'a, RandomState>,
    seqs: &[SequenceId],
    references: &mut [Qwen3ForwardExecutor<'a, RandomState>],
    mut last: Vec<Vec<f32>>,
    steps: usize,
    tolerance: f32,
    fork: bool,
    agreement: &mut Agreement,
) {
    for step in 0..steps {
        let mut tokens = last.iter().map(|row| first_argmax(row)).collect::<Vec<_>>();
        if fork && step == 0 {
            // Different first tokens make the fork pair copy its shared tail.
            tokens[1] = (tokens[0] + 1) % i32::try_from(last[1].len()).expect("vocab");
        }
        let rows = seqs
            .iter()
            .copied()
            .zip(tokens.iter().copied())
            .collect::<Vec<_>>();
        let batched = logits(
            paged
                .decode_batch(&rows, BatchReadback::Logits)
                .expect("batched decode"),
        );
        for (row, (reference, &token)) in references.iter_mut().zip(&tokens).enumerate() {
            let expected = reference
                .decode_last_logits(token)
                .expect("reference decode");
            agreement.check(
                &batched[row],
                &expected,
                tolerance,
                &format!("B={} row {row} step {step}", seqs.len()),
            );
        }
        last = batched;
    }
}

fn mixed_prompts(rows: usize, seed: usize) -> Vec<Vec<i32>> {
    (0..rows)
        .map(|row| prompt(1 + (row * 13 + seed * 7) % 47, row + seed))
        .collect()
}

#[test]
fn one_row_batch_is_bit_identical_to_single_decode() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut paged = PagedQwen3Session::new(&config, &weights, pool(1)).expect("paged");
    let mut contiguous = Qwen3ForwardExecutor::new(&config, &weights);
    let seq = SequenceId(1);
    let prompt = prompt(21, 8);
    let mut last = paged.prefill_last_logits(seq, &prompt).expect("prefill");
    contiguous.prefill_last_logits(&prompt).expect("prefill");
    for step in 0..12 {
        let token = first_argmax(&last);
        let batched = logits(
            paged
                .decode_batch(&[(seq, token)], BatchReadback::Logits)
                .expect("batch"),
        )
        .remove(0);
        let expected = contiguous.decode_last_logits(token).expect("decode");
        let identical = batched
            .iter()
            .zip(&expected)
            .all(|(left, right)| left.to_bits() == right.to_bits());
        assert!(identical, "step {step}: one-row batch differs from decode");
        last = batched;
    }
}

#[test]
fn batched_decode_agrees_with_single_sequences() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    for (batch, fork) in [(2, false), (3, true), (5, false), (8, true), (16, false)] {
        let mut paged = PagedQwen3Session::new(&config, &weights, pool(3)).expect("paged");
        let prompts = mixed_prompts(batch, batch);
        let (seqs, mut references, last) =
            start_rows(&mut paged, &config, &weights, &prompts, fork);
        let mut agreement = Agreement::default();
        run_batch(
            &mut paged,
            &seqs,
            &mut references,
            last,
            16,
            FIXTURE_TOLERANCE,
            fork,
            &mut agreement,
        );
        println!(
            "paged_batch fixture B={batch} fork={fork} row_steps={} divergences={} \
             max_abs_diff={:e} tolerance={FIXTURE_TOLERANCE:e}",
            agreement.row_steps, agreement.divergences, agreement.max_abs_diff,
        );
        if fork {
            let parent = paged.blocks().block_table(seqs[0]).expect("live");
            let child = paged.blocks().block_table(seqs[1]).expect("live");
            // Blocks full at the fork stay shared; the partial tail was copied.
            let full = prompts[0].len() / 16;
            assert_eq!(parent[..full], child[..full], "full blocks stay shared");
            assert_ne!(parent[full], child[full], "the shared tail was copied");
        }
    }
}

#[test]
fn equal_length_rows_need_no_mask_and_agree() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut paged = PagedQwen3Session::new(&config, &weights, pool(1)).expect("paged");
    let prompts = vec![prompt(16, 1), prompt(16, 2), prompt(16, 3)];
    let (seqs, mut references, last) = start_rows(&mut paged, &config, &weights, &prompts, false);
    let mut agreement = Agreement::default();
    run_batch(
        &mut paged,
        &seqs,
        &mut references,
        last,
        8,
        FIXTURE_TOLERANCE,
        false,
        &mut agreement,
    );
}

#[test]
fn greedy_readback_matches_the_logits_argmax() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let prompts = mixed_prompts(6, 3);
    let mut by_logits = PagedQwen3Session::new(&config, &weights, pool(1)).expect("paged");
    let mut by_greedy = PagedQwen3Session::new(&config, &weights, pool(1)).expect("paged");
    let mut tokens = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let seq = SequenceId(index as u64 + 1);
        let last = by_logits.prefill_last_logits(seq, prompt).expect("prefill");
        by_greedy.prefill_last_logits(seq, prompt).expect("prefill");
        tokens.push((seq, first_argmax(&last)));
    }
    for step in 0..8 {
        let rows = logits(
            by_logits
                .decode_batch(&tokens, BatchReadback::Logits)
                .expect("logits"),
        );
        let BatchDecoded::Greedy(greedy) = by_greedy
            .decode_batch(&tokens, BatchReadback::Greedy)
            .expect("greedy")
        else {
            panic!("asked for greedy tokens");
        };
        let expected = rows.iter().map(|row| first_argmax(row)).collect::<Vec<_>>();
        assert_eq!(greedy, expected, "step {step}");
        for (row, token) in tokens.iter_mut().zip(greedy) {
            row.1 = token;
        }
    }
}

#[test]
fn batch_refusals_change_nothing() {
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut paged = PagedQwen3Session::new(&config, &weights, pool(1)).expect("paged");
    let (a, b) = (SequenceId(1), SequenceId(2));
    // a: 15 full blocks and 15 tokens; b: 16 full blocks. b's next token
    // needs block 33 of a 32-block pool.
    let first = paged
        .prefill_last_logits(a, &prompt(16 * 16 - 1, 1))
        .expect("a");
    paged
        .prefill_last_logits(b, &prompt(16 * 16, 2))
        .expect("b");
    assert_eq!(paged.blocks().free_blocks(), 0);
    let refused = paged.decode_batch(&[(a, 1), (b, 2)], BatchReadback::Greedy);
    assert!(
        matches!(refused, Err(Qwen3ForwardError::KvBlocks(_))),
        "{refused:?}"
    );
    assert_eq!(paged.cached_tokens(a).expect("live"), 16 * 16 - 1);
    assert_eq!(paged.cached_tokens(b).expect("live"), 16 * 16);
    let repeated = paged.decode_batch(&[(a, 1), (a, 2)], BatchReadback::Greedy);
    assert!(
        matches!(repeated, Err(Qwen3ForwardError::RepeatedBatchSequence(1))),
        "{repeated:?}"
    );
    // a alone still fits its last free slot.
    paged
        .decode_batch(&[(a, first_argmax(&first))], BatchReadback::Greedy)
        .expect("a's last slot");
    assert_eq!(paged.cached_tokens(a).expect("live"), 16 * 16);
}

/// Largest batched-vs-single logit difference accepted on Qwen3-0.6B F32.
/// Declared before the first run; that run's largest difference over
/// B = 2..16 was 1.26e-4 (printed as `max_abs_diff`).
const QWEN_TOLERANCE: f32 = 2e-3;

/// Qwen3-0.6B at F32: the D2 gate at B = 2, 4, 8, 16, then aggregate greedy
/// decode throughput at B = 1, 2, 4, 8, 16.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn paged_batch_qwen3_06b_agrees_and_reports_throughput() {
    use crate::metal::Qwen3MlxWeights;

    let model = std::env::var_os("METALLIX_QWEN_MODEL")
        .map(std::path::PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored test");
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut checkpoint = Qwen3MlxWeights::load(model).expect("checkpoint load");
    checkpoint.prepare_float32().expect("float32 weights");
    let layout = checkpoint
        .resident_chat_executor(4096, u64::MAX)
        .expect("resident plan");
    let (config, weights) = (layout.config, layout.weights);
    let pool = || {
        PagedQwen3Session::pool_for_budget(config, weights, 1024 * MIB, BlockTokens::DEFAULT)
            .expect("pool")
            .with_prefix_caching(false)
    };

    for (batch, fork) in [(2, false), (4, true), (8, false), (16, true)] {
        let mut paged = PagedQwen3Session::new(config, weights, pool()).expect("paged");
        let prompts = (0..batch)
            .map(|row| fixed_tokens(40 + row * 23, 100 + row as u64))
            .collect::<Vec<_>>();
        let (seqs, mut references, last) = start_rows(&mut paged, config, weights, &prompts, fork);
        let mut agreement = Agreement::default();
        run_batch(
            &mut paged,
            &seqs,
            &mut references,
            last,
            32,
            QWEN_TOLERANCE,
            fork,
            &mut agreement,
        );
        println!(
            "paged_batch qwen3_06b B={batch} fork={fork} row_steps={} divergences={} \
             max_abs_diff={:e} tolerance={QWEN_TOLERANCE:e}",
            agreement.row_steps, agreement.divergences, agreement.max_abs_diff,
        );
    }

    for batch in [1_usize, 2, 4, 8, 16] {
        throughput(config, weights, pool(), batch);
    }
}

fn throughput(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array>,
    pool: PoolConfig,
    batch: usize,
) {
    const STEPS: usize = 48;
    let mut paged = PagedQwen3Session::new(config, weights, pool).expect("paged");
    let mut rows = Vec::new();
    for row in 0..batch {
        let seq = SequenceId(row as u64 + 1);
        let last = paged
            .prefill_last_logits(seq, &fixed_tokens(128, 7 + row as u64))
            .expect("prefill");
        rows.push((seq, first_argmax(&last)));
    }
    let mut step = |rows: &mut Vec<(SequenceId, i32)>| {
        let BatchDecoded::Greedy(tokens) = paged
            .decode_batch(rows, BatchReadback::Greedy)
            .expect("decode")
        else {
            panic!("asked for greedy tokens");
        };
        for (row, token) in rows.iter_mut().zip(tokens) {
            row.1 = token;
        }
    };
    for _ in 0..4 {
        step(&mut rows);
    }
    let started = Instant::now();
    for _ in 0..STEPS {
        step(&mut rows);
    }
    let elapsed = started.elapsed();
    let tokens = f64::from(u32::try_from(STEPS * batch).expect("small"));
    println!(
        "paged_batch qwen3_06b throughput B={batch} context=128..{} steps={STEPS} \
         step_ms={:.3} aggregate_tok_s={:.1}",
        128 + 4 + STEPS,
        millis(elapsed) / f64::from(u32::try_from(STEPS).expect("small")),
        tokens / elapsed.as_secs_f64(),
    );
}

/// Qwen3-0.6B decode-step profile, BF16 then F32: wall step
/// time, host graph-build time and evaluate-plus-readback time at B = 1..16
/// and contexts 128 and 1024. Measurement only; nothing is asserted.
#[test]
#[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
fn paged_batch_qwen3_06b_profile() {
    use crate::{forward::paged::profile, metal::Qwen3MlxWeights};

    let model = std::env::var_os("METALLIX_QWEN_MODEL")
        .map(std::path::PathBuf::from)
        .expect("METALLIX_QWEN_MODEL is required for this ignored test");
    let _gpu = GPU_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut checkpoint = Qwen3MlxWeights::load(model).expect("checkpoint load");
    for (dtype, precision) in [
        ("bf16", Qwen3WeightPrecision::BFloat16),
        ("f32", Qwen3WeightPrecision::Float32),
    ] {
        checkpoint
            .prepare_precision(precision)
            .expect("weights at the profiled precision");
        let layout = checkpoint
            .resident_chat_executor(4096, u64::MAX)
            .expect("resident plan");
        let (config, weights) = (layout.config, layout.weights);
        for context in [128_usize, 1024] {
            for batch in [1_usize, 2, 4, 8, 16] {
                let pool = PagedQwen3Session::pool_for_budget(
                    config,
                    weights,
                    4096 * MIB,
                    BlockTokens::DEFAULT,
                )
                .expect("pool")
                .with_prefix_caching(false);
                let mut paged = PagedQwen3Session::new(config, weights, pool).expect("paged");
                let mut rows = Vec::new();
                for row in 0..batch {
                    let seq = SequenceId(row as u64 + 1);
                    let last = paged
                        .prefill_last_logits(seq, &fixed_tokens(context, 7 + row as u64))
                        .expect("prefill");
                    rows.push((seq, first_argmax(&last)));
                }
                let mut step = |rows: &mut Vec<(SequenceId, i32)>| {
                    let BatchDecoded::Greedy(tokens) = paged
                        .decode_batch(rows, BatchReadback::Greedy)
                        .expect("decode")
                    else {
                        panic!("asked for greedy tokens");
                    };
                    for (row, token) in rows.iter_mut().zip(tokens) {
                        row.1 = token;
                    }
                };
                for _ in 0..4 {
                    step(&mut rows);
                }
                profile::take();
                let started = Instant::now();
                for _ in 0..32 {
                    step(&mut rows);
                }
                let wall = millis(started.elapsed()) / 32.0;
                let steps = profile::take();
                let median = |pick: fn(&(std::time::Duration, std::time::Duration)) -> f64| {
                    let mut values = steps.iter().map(pick).collect::<Vec<_>>();
                    values.sort_by(f64::total_cmp);
                    values[values.len() / 2]
                };
                println!(
                    "paged_profile dtype={dtype} context={context} B={batch} step_ms={wall:.3} \
                     build_ms={:.3} eval_ms={:.3} aggregate_tok_s={:.1}",
                    median(|step| millis(step.0)),
                    median(|step| millis(step.1)),
                    1e3 * f64::from(u32::try_from(batch).expect("small")) / wall,
                );
            }
        }
    }
}

/// Queued (pipelined) greedy steps pick the same tokens as synchronous
/// greedy batches of the same composition. Step k + 1 is queued with every
/// row's input still on the device before step k is read back. A row that
/// leaves after step k stays allocated until its queued step k + 1 settles;
/// a row that joins brings a host token. The synchronous run decodes the leaving
/// row in step k + 1 too and frees it after, so both runs batch the same rows
/// in the same order at every step.
#[test]
fn queued_steps_match_synchronous_greedy_steps() {
    use crate::forward::{QueuedDecode, StepInput};

    const LEAVE_AFTER: usize = 3;
    const JOIN_AT: usize = 5;
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = paged_config();
    let weights = paged_weights(&config);
    let prompts = mixed_prompts(5, 4);
    let mut sync = PagedQwen3Session::new(&config, &weights, pool(2)).expect("paged");
    let mut piped = PagedQwen3Session::new(&config, &weights, pool(2)).expect("paged");
    let prefill = |session: &mut PagedQwen3Session<'_, RandomState>, row: usize| {
        let seq = SequenceId(row as u64 + 1);
        let logits = session
            .prefill_last_logits(seq, &prompts[row])
            .expect("prefill");
        (seq, first_argmax(&logits))
    };
    let leaving = SequenceId(2);

    // Synchronous reference: every step's rows and picks.
    let mut rows = (0..4)
        .map(|row| prefill(&mut sync, row))
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for step in 0..10 {
        if step == JOIN_AT {
            rows.push(prefill(&mut sync, 4));
        }
        let BatchDecoded::Greedy(tokens) = sync
            .decode_batch(&rows, BatchReadback::Greedy)
            .expect("sync step")
        else {
            panic!("asked for greedy tokens");
        };
        expected.push(
            rows.iter()
                .map(|&(seq, _)| seq)
                .zip(tokens.clone())
                .collect::<Vec<_>>(),
        );
        for (row, token) in rows.iter_mut().zip(tokens) {
            row.1 = token;
        }
        if step == LEAVE_AFTER + 1 {
            sync.free(leaving).expect("live");
            rows.retain(|&(seq, _)| seq != leaving);
        }
    }

    // Pipelined: queue step k + 1, then read step k.
    let start = (0..4)
        .map(|row| prefill(&mut piped, row))
        .collect::<Vec<_>>();
    let mut pending: QueuedDecode = piped
        .queue_decode(
            &start
                .iter()
                .map(|&(seq, token)| (seq, StepInput::Host(token)))
                .collect::<Vec<_>>(),
            None,
        )
        .expect("queue step 0");
    let mut live = start.iter().map(|&(seq, _)| seq).collect::<Vec<_>>();
    for (step, expected) in expected.iter().enumerate() {
        let mut next = pending
            .rows()
            .iter()
            .enumerate()
            .filter(|(_, seq)| live.contains(seq))
            .map(|(index, &seq)| (seq, StepInput::Previous(index)))
            .collect::<Vec<_>>();
        if step + 1 == JOIN_AT {
            let (seq, token) = prefill(&mut piped, 4);
            next.push((seq, StepInput::Host(token)));
            live.push(seq);
        }
        let queued =
            (step + 1 < 10).then(|| piped.queue_decode(&next, Some(&pending)).expect("queue"));
        let tokens = piped.finish_decode(&pending).expect("finish");
        let actual = pending
            .rows()
            .iter()
            .copied()
            .zip(tokens)
            .collect::<Vec<_>>();
        assert_eq!(&actual, expected, "step {step}");
        if step == LEAVE_AFTER {
            // Stop scheduling it; the already queued next step still owns it.
            live.retain(|&seq| seq != leaving);
        }
        if step == LEAVE_AFTER + 1 {
            // Its last queued step has settled; its slots are reusable now.
            piped.free(leaving).expect("live");
        }
        match queued {
            Some(queued) => pending = queued,
            None => break,
        }
    }
    assert_eq!(
        piped.blocks().free_blocks(),
        sync.blocks().free_blocks(),
        "both runs hold the same blocks"
    );
}

/// A failed readback must not return rows to the allocator while the next
/// queued step still owns them. Cleanup belongs to the scheduling caller.
#[test]
fn failed_queued_readback_retains_rows_until_caller_retires_them() {
    use crate::forward::StepInput;
    let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
    let config = paged_config();
    let weights = paged_weights(&config);
    let mut session = PagedQwen3Session::new(&config, &weights, pool(2)).expect("paged");
    let seq = SequenceId(1);
    let logits = session
        .prefill_last_logits(seq, &[1, 2, 4])
        .expect("prefill");
    let mut first = session
        .queue_decode(&[(seq, StepInput::Host(first_argmax(&logits)))], None)
        .expect("first step");
    let next = session
        .queue_decode(&[(seq, StepInput::Previous(0))], Some(&first))
        .expect("next step");
    first.invalidate_commit_length();
    assert!(session.finish_decode(&first).is_err());
    let retained = session.blocks().num_tokens(seq).is_ok();
    // Always settle pending GPU work, including when the old behavior is red.
    let _ = session.finish_decode(&next);
    let _ = session.free(seq);
    assert!(
        retained,
        "failed readback freed rows owned by a queued next step"
    );
    assert_eq!(session.blocks().sequences(), 0);
    assert_eq!(
        session.blocks().free_blocks(),
        session.blocks().total_blocks()
    );
}
