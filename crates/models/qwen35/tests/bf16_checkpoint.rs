//! Opt-in agreement at checkpoint precision (BF16) against the CPU bfloat16
//! Transformers reference in `fixtures/qwen3.5-9b/bf16-reference.json`,
//! written by `scripts/qwen35-hybrid-reference.py --dtype bfloat16`.
//!
//! Qwen3.5-9B has twice as many `GatedDeltaNet` value heads as key heads, so
//! each query and key head serves two consecutive value heads. Qwen3.5-0.8B
//! has equal counts, so `hybrid_checkpoint.rs` never runs that repeat; this
//! test does. Both sides compute in BF16 with different kernels and summation
//! orders, so it checks the ranking, not logit values.
//!
//! Set `METALLIX_QWEN35_BF16_MODEL` to the Qwen/Qwen3.5-9B checkpoint
//! directory; the test refuses any other configuration. Each case prefills a
//! prompt and then decodes the reference's own greedy tokens one at a time.
//! `METALLIX_QWEN35_BF16_REFERENCE` overrides the fixture path.

#![cfg(feature = "metal")]

use std::{env, fs, path::PathBuf};

use qwen35::forward::{Qwen35Precision, Qwen35Weights};
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// Mean overlap of the native and reference top-5 sets, over all steps.
/// Declared before the first run against the committed fixture. The
/// 2026-10-05 scratch comparison on three other prompts measured 4.81; with
/// the key heads tiled instead of repeated consecutively it measured 1.89.
const MIN_MEAN_TOP5_OVERLAP: f64 = 4.5;

#[derive(Deserialize)]
struct Manifest {
    reference: Provenance,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Provenance {
    config_sha256: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    input_ids: Vec<i32>,
    argmax: usize,
    top8: Top8,
}

#[derive(Deserialize)]
struct Top8 {
    ids: Vec<usize>,
    logits: Vec<f32>,
}

/// Spacing of BF16 values at `value`'s magnitude: BF16 keeps 7 explicit
/// mantissa bits, so one unit in the last place is `2^(exponent - 7)`.
fn bf16_ulp(value: f32) -> f32 {
    let exponent = i32::try_from((value.abs().to_bits() >> 23) & 0xff).expect("8-bit exponent");
    2.0_f32.powi(exponent - 127 - 7)
}

/// Indices of the `k` largest values, largest first, lower index first on
/// equal values. A tie across the top-5 boundary can cost one overlap; the
/// mean bound absorbs that.
fn top_k(values: &[f32], k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|a, b| values[*b].total_cmp(&values[*a]).then(a.cmp(b)));
    indices.truncate(k);
    indices
}

#[test]
fn bf16_ranking_matches_the_source_with_repeated_key_heads() {
    let Some(model) = env::var_os("METALLIX_QWEN35_BF16_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN35_BF16_MODEL is not set");
        return;
    };
    let manifest: Manifest =
        serde_json::from_str(&match env::var_os("METALLIX_QWEN35_BF16_REFERENCE") {
            Some(path) => fs::read_to_string(path).expect("reference JSON"),
            None => include_str!("../../../../fixtures/qwen3.5-9b/bf16-reference.json").to_owned(),
        })
        .expect("reference JSON");
    let config = fs::read(model.join("config.json")).expect("config.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(&config)),
        manifest.reference.config_sha256,
        "the checkpoint is not the one the reference was captured from"
    );

    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    let (mut steps, mut overlap_sum, mut ties) = (0_u32, 0_u32, 0_u32);
    for case in &manifest.cases {
        let mut executor = weights.executor();
        for (index, step) in case.steps.iter().enumerate() {
            let logits = if index == 0 {
                executor.prefill_last_logits(&step.input_ids)
            } else {
                assert_eq!(step.input_ids.len(), 1, "decode steps feed one token");
                executor.decode_last_logits(step.input_ids[0])
            }
            .expect("forward");
            assert_eq!(logits.len(), weights.config().vocab_size());

            let native = top_k(&logits, 5);
            let reference = &step.top8.ids[..5];
            let overlap = native.iter().filter(|id| reference.contains(id)).count();
            let margin = step.top8.logits[0] - step.top8.logits[1];
            let tie = margin <= bf16_ulp(step.top8.logits[0]);
            eprintln!(
                "{} step {index}: argmax native {} reference {}, top-5 overlap {overlap}/5, reference margin {margin}",
                case.name, native[0], step.argmax
            );
            if native[0] != step.argmax {
                assert!(
                    tie,
                    "{} step {index}: argmax {} differs from the reference's {} by more than a BF16 tie (margin {margin})",
                    case.name, native[0], step.argmax
                );
                ties += 1;
            }
            steps += 1;
            overlap_sum += u32::try_from(overlap).expect("at most 5");
        }
    }
    let mean = f64::from(overlap_sum) / f64::from(steps);
    eprintln!(
        "{steps} steps: argmax differs at {ties} reference ties, mean top-5 overlap {mean:.2}/5"
    );
    assert!(
        mean >= MIN_MEAN_TOP5_OVERLAP,
        "mean top-5 overlap {mean:.2} is below {MIN_MEAN_TOP5_OVERLAP}"
    );
}
