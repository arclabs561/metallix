//! Opt-in parity against the CPU f32 Transformers reference in
//! `fixtures/qwen3.5-0.8b/hybrid-reference.json`, written by
//! `scripts/qwen35-hybrid-reference.py`.
//!
//! Set `METALLIX_QWEN35_MODEL` to the checkpoint directory. Each case prefills
//! a prompt and then decodes the reference's own greedy tokens one at a time,
//! so the cached recurrent, convolution and K/V state is compared at every
//! step. The committed fixture holds each step's top-8 logits; set
//! `METALLIX_QWEN35_LOGITS_DIR` to the script's `--logits-dir` output to also
//! compare the full vocabulary. `METALLIX_QWEN35_REFERENCE` overrides the
//! fixture path, for another checkpoint of the same family.

#![cfg(feature = "metal")]

use std::{env, fs, path::PathBuf, time::Instant};

use qwen35::forward::{Qwen35Precision, Qwen35Weights};
use serde_json::Value;

/// Largest accepted |logit difference| against the f32 reference. Qwen3.5-0.8B
/// measured 4.9e-5 over the full vocabulary (2026-10-05); this leaves room for
/// kernel and summation-order changes. Dropping the carried convolution window
/// or the recurrent decay changes an argmax.
const MAX_ABS_LOGIT_DIFF: f32 = 1e-3;

struct Case {
    model: PathBuf,
    reference: Value,
    logits_dir: Option<PathBuf>,
}

fn case() -> Option<Case> {
    let Some(model) = env::var_os("METALLIX_QWEN35_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN35_MODEL is not set");
        return None;
    };
    let reference = match env::var_os("METALLIX_QWEN35_REFERENCE") {
        Some(path) => fs::read_to_string(path).expect("reference JSON"),
        None => include_str!("../../../../fixtures/qwen3.5-0.8b/hybrid-reference.json").to_owned(),
    };
    Some(Case {
        model,
        reference: serde_json::from_str(&reference).expect("reference JSON"),
        logits_dir: env::var_os("METALLIX_QWEN35_LOGITS_DIR").map(PathBuf::from),
    })
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

fn read_f32le(path: &PathBuf) -> Vec<f32> {
    fs::read(path)
        .expect("logits sidecar")
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four bytes")))
        .collect()
}

#[test]
fn prefill_and_cached_decode_match_the_source_reference() {
    let Some(case) = case() else { return };
    let weights = Qwen35Weights::load(&case.model, Qwen35Precision::Float32).expect("load");
    let mut worst_top8 = 0.0_f32;
    let mut worst_full = None::<f32>;
    let mut steps = 0;
    for reference in case.reference["cases"].as_array().expect("cases") {
        let name = reference["name"].as_str().expect("case name");
        let mut executor = weights.executor();
        for (index, step) in reference["steps"]
            .as_array()
            .expect("steps")
            .iter()
            .enumerate()
        {
            let input = ids(&step["input_ids"]);
            let logits = if index == 0 {
                executor.prefill_last_logits(&input)
            } else {
                assert_eq!(input.len(), 1, "{name}: decode steps feed one token");
                executor.decode_last_logits(input[0])
            }
            .expect("forward");
            assert_eq!(logits.len(), weights.config().vocab_size());

            let expected_argmax =
                usize::try_from(step["argmax"].as_u64().expect("argmax")).expect("usize");
            assert_eq!(
                argmax(&logits),
                expected_argmax,
                "{name} step {index}: argmax"
            );
            let top_ids = ids(&step["top8"]["ids"]);
            for (id, expected) in top_ids
                .iter()
                .zip(step["top8"]["logits"].as_array().expect("top8"))
            {
                let id = usize::try_from(*id).expect("usize id");
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "the reference logits are f32"
                )]
                let expected = expected.as_f64().expect("logit") as f32;
                worst_top8 = worst_top8.max((logits[id] - expected).abs());
            }
            if let Some(dir) = &case.logits_dir {
                let file = step["logits_f32le"]["file"].as_str().expect("sidecar name");
                let oracle = read_f32le(&dir.join(file));
                assert_eq!(
                    oracle.len(),
                    logits.len(),
                    "{name} step {index}: vocabulary"
                );
                let diff = logits
                    .iter()
                    .zip(&oracle)
                    .map(|(native, oracle)| (native - oracle).abs())
                    .fold(0.0_f32, f32::max);
                worst_full = Some(worst_full.unwrap_or(0.0).max(diff));
            }
            steps += 1;
        }
    }
    eprintln!(
        "{steps} steps: argmax agrees on all; max |dlogit| top-8 {worst_top8:.3e}, full vocabulary {}",
        worst_full.map_or_else(|| "not compared".to_owned(), |diff| format!("{diff:.3e}"))
    );
    assert!(
        worst_top8 <= MAX_ABS_LOGIT_DIFF,
        "top-8 logit diff {worst_top8}"
    );
    if let Some(diff) = worst_full {
        assert!(
            diff <= MAX_ABS_LOGIT_DIFF,
            "full-vocabulary logit diff {diff}"
        );
    }
}

/// A prompt consumed in one call, across the prefill chunk boundary, matches
/// the same prompt consumed as a shorter prefill followed by an extension:
/// the carried state, not the call boundaries, decides the logits.
#[test]
fn chunked_extension_matches_a_single_prefill() {
    let Some(case) = case() else { return };
    let weights = Qwen35Weights::load(&case.model, Qwen35Precision::Float32).expect("load");
    let prompt = ids(&case.reference["cases"][3]["steps"][0]["input_ids"]);
    assert!(prompt.len() > qwen35::forward::PREFILL_CHUNK_TOKENS);
    let whole = weights
        .executor()
        .prefill_last_logits(&prompt)
        .expect("whole");
    let mut split = weights.executor();
    split.prefill_last_logits(&prompt[..37]).expect("head");
    split.extend_last_logits(&prompt[37..90]).expect("middle");
    let pieces = split.extend_last_logits(&prompt[90..]).expect("tail");
    let diff = whole
        .iter()
        .zip(&pieces)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("split vs whole prefill: max |dlogit| {diff:.3e}");
    assert!(diff <= 1e-3, "split prefill diverged by {diff}");
}

/// Greedy decode throughput at checkpoint precision. Opt-in and ignored by
/// default because it measures rather than checks.
#[test]
#[ignore = "measurement; run with --ignored"]
fn decode_throughput() {
    let Some(case) = case() else { return };
    let weights = Qwen35Weights::load(&case.model, Qwen35Precision::Checkpoint).expect("load");
    let prompt = ids(&case.reference["cases"][2]["steps"][0]["input_ids"]);
    let mut executor = weights.executor();
    let started = Instant::now();
    let mut logits = executor.prefill_last_logits(&prompt).expect("prefill");
    let prefill = started.elapsed();
    let steps = 128;
    let started = Instant::now();
    for _ in 0..steps {
        let next = i32::try_from(argmax(&logits)).expect("token id");
        logits = executor.decode_last_logits(next).expect("decode");
    }
    let decode = started.elapsed();
    eprintln!(
        "checkpoint precision, {} weight bytes: prefill {} tokens in {:.3}s, decode {steps} tokens at {:.1} tok/s",
        weights.logical_weight_bytes(),
        prompt.len(),
        prefill.as_secs_f64(),
        f64::from(steps) / decode.as_secs_f64()
    );
}
