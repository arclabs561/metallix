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

use qwen35::forward::{Qwen35Precision, Qwen35Weights};
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
    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
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
