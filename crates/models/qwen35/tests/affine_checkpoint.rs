//! Opt-in greedy agreement with mlx-lm on an MLX affine checkpoint, against
//! the `cases` of `scripts/qwen35-mlx-affine-reference.py reference`.
//!
//! Set `METALLIX_QWEN35_AFFINE_MODEL` to the checkpoint directory and
//! `METALLIX_QWEN35_AFFINE_REFERENCE` to the script's JSON for it. Each case
//! prefills the prompt and then decodes mlx-lm's own greedy tokens, so every
//! step is compared from identical inputs. The quantized modules match
//! mlx-lm bit for bit (see the crate's component test), but the linear
//! attention follows Transformers' op order, so logits are not bitwise
//! equal: the argmax must agree wherever mlx-lm's top two logits are at
//! least `TIE_GAP` apart, and the logit error on mlx-lm's top 8 must stay
//! within `FLOOR_FACTOR` times the reference's `aa` floor: how far mlx-lm
//! moves from itself when its gated-delta recurrence runs as ops instead of
//! its Metal kernel. Two errors are bounded: the centered error
//! `max |d - mean(d)|` (a common shift of every logit changes no
//! probability) and the error of the top-1 minus top-2 margin.

#![cfg(feature = "metal")]

use std::{env, fs, path::PathBuf};

use qwen35::forward::{Qwen35Precision, Qwen35Weights};
use serde_json::Value;

/// Below this gap between mlx-lm's two largest logits, a different argmax is
/// counted as a near-tie rather than a failure.
const TIE_GAP: f32 = 0.5;

/// How far above mlx-lm's own kernel-versus-ops difference the native
/// errors may go.
const FLOOR_FACTOR: f64 = 1.5;

/// The bound never drops below one BF16 step at the logit scale: logits up
/// to 64 in magnitude are spaced 0.25 apart in BF16, so a floor of 0 (both
/// mlx-lm paths agreeing bit for bit) still admits one rounding step.
const MIN_BOUND: f64 = 0.25;

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
    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    let (mut steps, mut ties, mut overlap) = (0_usize, 0_usize, 0_usize);
    // (centered, margin, max |d|): ours and mlx-lm's own floor.
    let (mut ours_max, mut floor_max) = ([0.0_f64; 3], [0.0_f64; 3]);
    for (case_index, case) in reference["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .enumerate()
    {
        let floor = case["aa"]
            .as_array()
            .expect("aa floor: regenerate the reference");
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
    assert!(steps > 0);
    for (i, name) in ["centered", "margin"].into_iter().enumerate() {
        let scaled = FLOOR_FACTOR * floor_max[i];
        let (bound, term) = if scaled >= MIN_BOUND {
            (scaled, "1.5 x floor")
        } else {
            (MIN_BOUND, "BF16 step")
        };
        eprintln!(
            "{name} error {:.4} <= {bound:.4}? bound set by {term}",
            ours_max[i]
        );
        assert!(
            ours_max[i] <= bound,
            "{name} error {} exceeds {bound} ({term}; floor {})",
            ours_max[i],
            floor_max[i]
        );
    }
}
