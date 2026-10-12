//! Opt-in greedy agreement with mlx-lm on an MLX affine checkpoint, against
//! the `cases` of `scripts/qwen35-mlx-affine-reference.py reference`.
//!
//! Set `METALLIX_QWEN35_AFFINE_MODEL` to the checkpoint directory and
//! `METALLIX_QWEN35_AFFINE_REFERENCE` to the script's JSON for it. Each case
//! prefills the prompt and then decodes mlx-lm's own greedy tokens, so every
//! step is compared from identical inputs. The component test separately
//! requires quantized modules to match mlx-lm bit for bit, but the linear
//! attention follows Transformers' op order, so logits are not bitwise
//! equal: the argmax must agree wherever mlx-lm's top two logits are at
//! least `TIE_GAP` apart, and at every step the logit error on mlx-lm's top
//! 8 must stay within `FLOOR_FACTOR` times the reference's `aa` floor for that
//! step: the largest distance from mlx-lm of its own other configurations
//! (gated-delta kernel or ops, prompt prefilled whole or in chunks). A single
//! such configuration is not enough: rounding differences flip near-tied
//! expert choices, which amplifies them unevenly from step to step. Two
//! errors are bounded: the centered error `max |d - mean(d)|` (a common
//! shift of every logit changes no probability) and the error of the top-1
//! minus top-2 margin.

#![cfg(feature = "metal")]

use std::{env, fs, path::PathBuf};

use qwen35::forward::{Qwen35Precision, Qwen35Taps, Qwen35Weights};
use serde_json::Value;

/// Below this gap between mlx-lm's two largest logits, a different argmax is
/// counted as a near-tie rather than a failure.
const TIE_GAP: f32 = 0.5;

/// How far above mlx-lm's own spread across configurations the native
/// errors may go.
const FLOOR_FACTOR: f64 = 1.5;

/// Fewer distinct configurations than this would make the floor a sample too
/// small to bound the spread.
const MIN_ARMS: usize = 8;

/// The spacing of BF16 values at `x`'s magnitude: logits are BF16 on both
/// sides, so where every mlx-lm configuration agrees bit for bit (a floor of
/// 0) one rounding step at the step's largest logit is still admitted.
fn bf16_step(x: f64) -> f64 {
    // BF16 keeps 8 significant bits, so values in [2^e, 2^(e+1)) are
    // 2^(e-7) apart.
    assert!(x.is_finite(), "nonfinite reference logit");
    (x.abs().log2().floor() - 7.0).max(-133.0).exp2()
}

/// The bound on one error at one step, given that step's largest mlx-lm
/// logit, and which term set it.
fn bound(floor: f64, top_logit: f64) -> (f64, &'static str) {
    let scaled = FLOOR_FACTOR * floor;
    let step = bf16_step(top_logit);
    if scaled >= step {
        (scaled, "1.5 x floor")
    } else {
        (step, "BF16 step")
    }
}

/// Centered error, margin error and max |d| of `got` against `want` on the
/// same ids, `want` sorted descending.
fn errors(got: &[f64], want: &[f64]) -> (f64, f64, f64) {
    let d: Vec<f64> = got.iter().zip(want).map(|(g, w)| g - w).collect();
    #[allow(clippy::cast_precision_loss, reason = "8 ids")]
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    let centered = d.iter().map(|x| (x - mean).abs()).fold(0.0, f64::max);
    let margin = ((got[0] - got[1]) - (want[0] - want[1])).abs();
    let max_abs = d.iter().map(|x| x.abs()).fold(0.0, f64::max);
    (centered, margin, max_abs)
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("number array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn ids(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("id array")
        .iter()
        .map(|id| i32::try_from(id.as_i64().expect("integer id")).expect("i32 id"))
        .collect()
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("nonempty logits")
        .0
}

/// A step error over its bound: (case, step, error name, value, bound).
type Over = (usize, usize, &'static str, f64, f64);

/// Checks one step's centered and margin errors against their bounds,
/// pushing any over its bound to `over`. Returns how many are within bound
/// only because of the BF16-step minimum.
fn check_step(
    (case_index, step): (usize, usize),
    ours: (f64, f64, f64),
    floor: [f64; 3],
    top_logit: f64,
    over: &mut Vec<Over>,
) -> usize {
    let mut by_step_only = 0;
    for (i, (name, value)) in [("centered", ours.0), ("margin", ours.1)]
        .into_iter()
        .enumerate()
    {
        assert!(value.is_finite() && floor[i].is_finite() && floor[i] >= 0.0);
        let (limit, term) = bound(floor[i], top_logit);
        if value > FLOOR_FACTOR * floor[i] && value <= limit {
            by_step_only += 1;
        }
        if value > limit {
            eprintln!(
                "case {case_index} step {step}: {name} {value:.4} > {limit:.4} (bound set by {term})"
            );
            over.push((case_index, step, name, value, limit));
        }
    }
    by_step_only
}

/// Refuses a case whose floor comes from fewer than `MIN_ARMS` distinct
/// mlx-lm configurations.
fn check_arms(case_index: usize, case: &Value) {
    let arms = case["aa_arms"]
        .as_array()
        .expect("aa_arms: regenerate the reference")
        .iter()
        .filter(|arm| arm["duplicate"] == Value::Bool(false))
        .count();
    assert!(
        arms >= MIN_ARMS,
        "case {case_index}: {arms} distinct configurations in the floor, need {MIN_ARMS}"
    );
}

/// Cooperative materialization-boundary bound for the four-layer qualification.
/// MLX active plus cache and historical peak exclude other process/driver memory.
fn check_reference_memory(phase: &str, report: bool) {
    let active = mlx_rs::memory::active_memory().expect("active memory");
    let cached = mlx_rs::memory::cache_memory().expect("cached memory");
    let peak = mlx_rs::memory::peak_memory().expect("peak memory");
    let cap = 8_usize << 30;
    if report {
        eprintln!("MLX {phase}: active={active} cached={cached} peak={peak} cap={cap}");
    }
    assert!(
        peak <= cap && active.saturating_add(cached) <= cap,
        "MLX resource cap exceeded after {phase}: active={active} cached={cached} peak={peak} cap={cap}"
    );
}

fn load_reference_weights(model: &std::path::Path) -> Qwen35Weights {
    mlx_rs::memory::set_cache_limit(2 << 30).expect("allocator cache limit");
    mlx_rs::memory::reset_peak_memory().expect("reset peak");
    let weights = Qwen35Weights::load(model, Qwen35Precision::Checkpoint).expect("load");
    check_reference_memory("load", true);
    weights
}

#[test]
fn greedy_steps_agree_with_mlx_lm() {
    let (Some(model), Some(reference)) = (
        env::var_os("METALLIX_QWEN35_AFFINE_MODEL").map(PathBuf::from),
        env::var_os("METALLIX_QWEN35_AFFINE_REFERENCE"),
    ) else {
        eprintln!("skipping: METALLIX_QWEN35_AFFINE_MODEL or _REFERENCE is not set");
        return;
    };
    let reference: Value =
        serde_json::from_str(&fs::read_to_string(reference).expect("reference")).expect("JSON");
    assert_eq!(reference["mlx"], "0.32.3", "regenerate the reference");
    assert_eq!(reference["qualification"]["bar"], "v2");
    assert_eq!(reference["qualification"]["prompts"], "heldout");
    let cases = reference["cases"].as_array().expect("cases");
    assert_eq!(cases.len(), 4, "four held-out cases required");
    for (i, case) in cases.iter().enumerate() {
        check_arms(i, case);
        assert_eq!(case["steps"].as_array().expect("steps").len(), 32);
        assert_eq!(case["aa"].as_array().expect("floor").len(), 32);
    }
    let weights = load_reference_weights(&model);
    let (mut steps, mut ties, mut overlap) = (0_usize, 0_usize, 0_usize);
    // (centered, margin, max |d|): ours and mlx-lm's own floor.
    let (mut ours_max, mut floor_max) = ([0.0_f64; 3], [0.0_f64; 3]);
    let mut over: Vec<Over> = Vec::new();
    // Errors within their bound only because of the BF16-step minimum.
    let mut by_step_only = 0_usize;
    for (case_index, case) in reference["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .enumerate()
    {
        let floor = case["aa"]
            .as_array()
            .expect("aa floor: regenerate the reference");
        check_arms(case_index, case);
        let mut executor = weights.executor();
        let mut logits = executor
            .prefill_last_logits(&ids(&case["prompt_ids"]))
            .expect("prefill");
        check_reference_memory("prefill", true);
        for (step, expected) in case["steps"].as_array().expect("steps").iter().enumerate() {
            let top = ids(&expected["top8_ids"]);
            let want = floats(&expected["top8_logits"]);
            let got: Vec<f64> = top
                .iter()
                .map(|id| f64::from(logits[usize::try_from(*id).expect("id")]))
                .collect();
            let ours = errors(&got, &want);
            let aa = &floor[step];
            let aa = [&aa["centered"], &aa["margin"], &aa["max_abs"]]
                .map(|value| value.as_f64().expect("floor value"));
            for (i, value) in [ours.0, ours.1, ours.2].into_iter().enumerate() {
                ours_max[i] = ours_max[i].max(value);
                floor_max[i] = floor_max[i].max(aa[i]);
            }
            eprintln!(
                "case {case_index} step {step}: centered {:.4} (floor {:.4}), margin {:.4} (floor {:.4}), max |d| {:.4} (floor {:.4})",
                ours.0, aa[0], ours.1, aa[1], ours.2, aa[2]
            );
            by_step_only += check_step((case_index, step), ours, aa, want[0], &mut over);
            let mut our_top: Vec<usize> = (0..logits.len()).collect();
            our_top.select_nth_unstable_by(7, |a, b| logits[*b].total_cmp(&logits[*a]));
            overlap += top
                .iter()
                .filter(|id| our_top[..8].contains(&usize::try_from(**id).expect("id")))
                .count();
            let chosen = usize::try_from(top[0]).expect("id");
            if argmax(&logits) != chosen {
                let gap = expected["top2_gap"].as_f64().expect("gap");
                #[allow(clippy::cast_possible_truncation, reason = "f32 gap")]
                let gap = gap as f32;
                eprintln!(
                    "step {step}: argmax {} vs mlx-lm {chosen}, mlx-lm top1-top2 margin {gap:.4}",
                    argmax(&logits)
                );
                assert!(gap < TIE_GAP, "step {step}: margin {gap} is not a near-tie");
                ties += 1;
            }
            steps += 1;
            logits = executor.decode_last_logits(top[0]).expect("decode");
            check_reference_memory("decode", false);
        }
    }
    eprintln!(
        "steps {steps}, near-tie argmax differences {ties}, top-8 overlap {overlap}/{}; on mlx-lm's top 8, max centered error {:.4} (floor {:.4}), max margin error {:.4} (floor {:.4}), max |dlogit| {:.4} (floor {:.4})",
        steps * 8,
        ours_max[0],
        floor_max[0],
        ours_max[1],
        floor_max[1],
        ours_max[2],
        floor_max[2],
    );
    check_reference_memory("heldout complete", true);
    eprintln!("errors within bound only by the BF16-step minimum: {by_step_only}");
    assert!(steps > 0);
    assert!(
        over.is_empty(),
        "{} step errors over their per-step bound (case, step, error, value, bound): {over:?}",
        over.len()
    );
}

#[test]
fn per_step_bound_counts_floor_only_passes_and_rejects_excess() {
    let mut over = Vec::new();
    // At top1=8 BF16 spacing is 1/16, rather than the retired flat 1/4.
    assert_eq!(bf16_step(8.0), 0.0625);
    assert_eq!(
        check_step((0, 0), (0.0625, 0.0, 0.0), [0.0; 3], 8.0, &mut over),
        1
    );
    assert!(over.is_empty());
    assert_eq!(
        check_step((0, 1), (0.125, 0.0, 0.0), [0.0; 3], 8.0, &mut over),
        0
    );
    assert_eq!(over.len(), 1);
    assert_eq!(over[0].1, 1);
    // A different step's larger A/A floor must not mask that failure.
    assert_eq!(
        check_step((0, 2), (0.125, 0.0, 0.0), [0.125; 3], 8.0, &mut over),
        0
    );
    assert_eq!(over.len(), 1);
}

fn snapshot_fixture() -> Option<(Qwen35Weights, Vec<i32>)> {
    let (Some(model), Some(reference)) = (
        env::var_os("METALLIX_QWEN35_AFFINE_MODEL"),
        env::var_os("METALLIX_QWEN35_AFFINE_REFERENCE"),
    ) else {
        eprintln!("skipping: METALLIX_QWEN35_AFFINE_MODEL or _REFERENCE is not set");
        return None;
    };
    let reference: Value =
        serde_json::from_str(&fs::read_to_string(reference).expect("reference")).expect("JSON");
    let prompt = ids(&reference["cases"][0]["prompt_ids"]);
    assert!(prompt.len() >= 8, "nontrivial prefix and tail");
    let weights = load_reference_weights(&PathBuf::from(model));
    Some((weights, prompt))
}

/// Exercise real affine recurrent/KV consumers without comparing different
/// projection batch schedules. Every equality covers the complete vocabulary.
#[test]
#[ignore = "set METALLIX_QWEN35_AFFINE_MODEL and METALLIX_QWEN35_AFFINE_REFERENCE; run with --ignored"]
fn affine_snapshot_and_partial_verify_preserve_accepted_prefix() {
    let Some((weights, prompt)) = snapshot_fixture() else {
        return;
    };
    let boundary = prompt.len() - 4;
    let mut original = weights.executor();
    original
        .prefill_last_logits(&prompt[..boundary])
        .expect("prefix");
    check_reference_memory("snapshot prefix", false);
    let snapshot = original.snapshot().expect("snapshot");
    assert_eq!(snapshot.tokens(), boundary);
    let state_bytes = snapshot.state_bytes();
    assert!(state_bytes > 0);
    let mut restored = weights.executor_from(&snapshot).expect("restore");
    let expected = original
        .extend_last_logits(&prompt[boundary..])
        .expect("original tail");
    let resumed = restored
        .extend_last_logits(&prompt[boundary..])
        .expect("restored tail");
    assert!(!expected.is_empty());
    assert!(expected.iter().all(|x| x.is_finite()));
    assert_eq!(resumed, expected, "full-vocabulary restored continuation");
    check_reference_memory("snapshot continuation", false);
    original
        .decode_last_logits(1)
        .expect("advance rejected branch");
    assert_eq!(original.tokens(), prompt.len() + 1);
    assert_eq!(snapshot.tokens(), boundary);
    assert_eq!(snapshot.state_bytes(), state_bytes);
    let replayed = weights
        .executor_from(&snapshot)
        .expect("retained snapshot")
        .extend_last_logits(&prompt[boundary..])
        .expect("replay tail");
    assert_eq!(
        replayed, expected,
        "origin advancement must not mutate retained snapshot"
    );
    check_reference_memory("snapshot replay", false);
    drop(original);
    drop(restored);

    // Independent branches have identical batch lengths and accepted tokens,
    // but distinct rejected tails. Neither branch is an oracle with a different
    // GEMM schedule. Their causal accepted rows and restored continuations must
    // be independent of the rejected suffix.
    let draft_a = [prompt[0], prompt[1], 1, 1];
    let draft_b = [prompt[0], prompt[1], 2, 2];
    let mut a = weights.executor_from(&snapshot).expect("branch a");
    let mut b = weights.executor_from(&snapshot).expect("branch b");
    let rows_a = a
        .verify_logits(&draft_a, Qwen35Taps::None)
        .expect("verify a")
        .rows;
    check_reference_memory("verify a", false);
    let rows_b = b
        .verify_logits(&draft_b, Qwen35Taps::None)
        .expect("verify b")
        .rows;
    check_reference_memory("verify b", false);
    let vocab = expected.len();
    assert_eq!(rows_a.len(), 4 * vocab);
    assert_eq!(rows_b.len(), 4 * vocab);
    assert!(rows_a.iter().chain(&rows_b).all(|x| x.is_finite()));
    assert_eq!(
        &rows_a[..2 * vocab],
        &rows_b[..2 * vocab],
        "causal accepted rows"
    );
    assert_eq!(a.tokens(), boundary + 4);
    assert_eq!(b.tokens(), boundary + 4);
    a.truncate_to(boundary + 2).expect("accept two from a");
    b.truncate_to(boundary + 2).expect("accept two from b");
    assert_eq!(a.tokens(), boundary + 2);
    assert_eq!(b.tokens(), boundary + 2);
    assert_eq!(
        a.snapshot().expect("accepted snapshot a").tokens(),
        boundary + 2
    );
    assert_eq!(
        b.snapshot().expect("accepted snapshot b").tokens(),
        boundary + 2
    );
    for token in [prompt[2], prompt[3]] {
        let logits_a = a.decode_last_logits(token).expect("common decode a");
        let logits_b = b.decode_last_logits(token).expect("common decode b");
        assert_eq!(logits_a.len(), vocab);
        assert!(logits_a.iter().chain(&logits_b).all(|x| x.is_finite()));
        assert_eq!(
            logits_a, logits_b,
            "full-vocabulary partial-acceptance continuation"
        );
        check_reference_memory("rollback common decode", false);
    }
    assert_eq!(a.tokens(), boundary + 4);
    assert_eq!(b.tokens(), boundary + 4);
    assert_eq!(snapshot.tokens(), boundary);
    eprintln!(
        "affine snapshot/rollback: vocabulary={vocab}, prefix={boundary}, accepted=2, common decode=2, exact full-vocabulary comparisons"
    );
    check_reference_memory("snapshot rollback complete", true);
}
