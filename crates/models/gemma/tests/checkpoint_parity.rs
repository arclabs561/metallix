//! Opt-in parity against the CPU float32 Transformers reference in
//! `fixtures/gemma-4-12b/reference.json`, written by
//! `scripts/gemma4-reference.py`.
//!
//! Set `METALLIX_GEMMA4_MODEL` to the checkpoint directory and run with
//! `--ignored`. The committed fixture holds each step's top-8 logits and the
//! source's own bfloat16 distance from its float32 output; set
//! `METALLIX_GEMMA4_LOGITS_DIR` to the float32 capture's `--output` directory
//! to also compare the full vocabulary. `METALLIX_GEMMA4_REFERENCE` overrides
//! the fixture path. `METALLIX_GEMMA4_PRECISION=bf16` checks the bf16 path; the
//! default is f32, which matches the oracle's arithmetic.
#![cfg(feature = "metal")]

use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use gemma::metal::{Gemma4MlxWeights, Gemma4Precision};
use serde::Deserialize;

const CONTEXT_TOKENS: usize = 4096;

#[derive(Deserialize)]
struct Manifest {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    input_ids: Vec<i32>,
    logits_f32le: Sidecar,
    argmax: usize,
    top8: Top8,
    source_bfloat16: SourceBfloat16,
}

#[derive(Deserialize)]
struct Sidecar {
    file: String,
    element_count: usize,
}

#[derive(Deserialize)]
struct Top8 {
    ids: Vec<usize>,
    logits: Vec<f32>,
}

/// The source's own bfloat16 output at this step, against its float32 output.
#[derive(Deserialize)]
struct SourceBfloat16 {
    max_abs: f32,
}

struct Paths {
    model: PathBuf,
    manifest: Manifest,
    logits_dir: Option<PathBuf>,
    precision: Gemma4Precision,
}

fn paths() -> Option<Paths> {
    let Some(model) = env::var_os("METALLIX_GEMMA4_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_GEMMA4_MODEL is not set");
        return None;
    };
    let manifest = match env::var_os("METALLIX_GEMMA4_REFERENCE") {
        Some(path) => fs::read_to_string(path).expect("reference JSON"),
        None => include_str!("../../../../fixtures/gemma-4-12b/reference.json").to_owned(),
    };
    let precision = match env::var("METALLIX_GEMMA4_PRECISION").as_deref() {
        Ok("bf16") => Gemma4Precision::BFloat16,
        Ok("f32") | Err(_) => Gemma4Precision::Float32,
        Ok(other) => panic!("METALLIX_GEMMA4_PRECISION must be f32 or bf16, got {other}"),
    };
    Some(Paths {
        model,
        manifest: serde_json::from_str(&manifest).expect("reference JSON"),
        logits_dir: env::var_os("METALLIX_GEMMA4_LOGITS_DIR").map(PathBuf::from),
        precision,
    })
}

fn read_logits(logits_dir: &Path, sidecar: &Sidecar) -> Vec<f32> {
    let bytes = fs::read(logits_dir.join(&sidecar.file)).expect("reference logits");
    assert_eq!(bytes.len(), sidecar.element_count * 4, "{}", sidecar.file);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4 bytes")))
        .collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        // The first maximum, as torch.argmax picks; bf16 logits often tie.
        .fold(
            0,
            |best, (index, value)| {
                if *value > values[best] { index } else { best }
            },
        )
}

struct Comparison {
    /// Largest |difference| over the full vocabulary when the sidecar is
    /// available, else over the reference's top 8.
    max_abs: f32,
    same_argmax: bool,
    /// Gap between the reference's two largest logits.
    reference_margin: f32,
}

fn compare(actual: &[f32], step: &Step, full: Option<&[f32]>) -> Comparison {
    let max_abs = match full {
        Some(expected) => {
            assert_eq!(actual.len(), expected.len());
            actual
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max)
        }
        None => step
            .top8
            .ids
            .iter()
            .zip(&step.top8.logits)
            .map(|(&id, expected)| (actual[id] - expected).abs())
            .fold(0.0_f32, f32::max),
    };
    Comparison {
        max_abs,
        same_argmax: argmax(actual) == step.argmax,
        reference_margin: step.top8.logits[0] - step.top8.logits[1],
    }
}

/// Per-step bounds on soft-capped logits in [-30, 30] against the float32
/// source, 12B, 21 steps plus one split prefill (2026-10-05):
///
/// - f32 drifts by kernel reduction order only: worst 7.5e-4 over the full
///   vocabulary, every argmax equal. The bound keeps 10x headroom and every
///   argmax must match.
/// - bf16 rounds weights, activations and K/V. The yardstick is the source's
///   own bf16 output fed the same tokens: its full-vocabulary distance from
///   float32 per step (`d`, 0.23 to 2.78) and its argmax flips, all three at
///   a top-2 margin below `d`. Metallix bf16 must stay within `2 d` (measured
///   worst ratio 1.57, on a chat step where `d` is 0.28) and match the argmax
///   wherever the float32 margin exceeds `d`, so it may only flip where the
///   source itself can. It measured one flip, at a 0.34 margin where the
///   source also flips. The factor 2 is set from these same 22 comparisons;
///   no held-out prompt informed it.
fn passes(precision: Gemma4Precision, step: &Step, comparison: &Comparison) -> bool {
    let (max_abs, argmax_margin) = match precision {
        Gemma4Precision::Float32 => (0.01, 0.0),
        Gemma4Precision::BFloat16 => {
            let noise = step.source_bfloat16.max_abs;
            (2.0 * noise, noise)
        }
    };
    comparison.max_abs <= max_abs
        && (comparison.same_argmax || comparison.reference_margin <= argmax_margin)
}

#[test]
#[ignore = "requires METALLIX_GEMMA4_MODEL (google/gemma-4-12B-it@707f0a3) on Apple-Silicon Metal"]
fn prefill_and_cached_decode_match_the_source_reference() {
    let Some(paths) = paths() else { return };
    let manifest = &paths.manifest;
    let full_logits = |step: &Step| {
        paths
            .logits_dir
            .as_deref()
            .map(|dir| read_logits(dir, &step.logits_f32le))
    };
    let started = Instant::now();
    let weights = Gemma4MlxWeights::load(&paths.model, paths.precision).expect("weights");
    eprintln!(
        "loaded {:?} weights ({} logical bytes) in {:.1}s",
        paths.precision,
        weights.logical_weight_bytes(),
        started.elapsed().as_secs_f64()
    );
    let mut failures = Vec::new();
    let mut worst_overall = 0.0_f32;
    for case in &manifest.cases {
        let mut executor = weights.executor(CONTEXT_TOKENS).expect("executor");
        for (index, step) in case.steps.iter().enumerate() {
            let expected = full_logits(step);
            if let Some(expected) = &expected {
                assert_eq!(
                    argmax(expected),
                    step.argmax,
                    "{}: manifest argmax",
                    case.name
                );
            }
            let started = Instant::now();
            // Decode feeds the reference's greedy tokens, so both sides see
            // the same history even if an argmax disagrees.
            let actual = if index == 0 {
                executor.prefill_last_logits(&step.input_ids)
            } else {
                executor.extend_last_logits(&step.input_ids)
            }
            .expect("forward");
            let comparison = compare(&actual, step, expected.as_deref());
            worst_overall = worst_overall.max(comparison.max_abs);
            eprintln!(
                "{} step {index} ({} tokens at {}): max_abs={:.5} argmax_match={} margin={:.3} {:.1} ms",
                case.name,
                step.input_ids.len(),
                executor.cached_tokens() - step.input_ids.len(),
                comparison.max_abs,
                comparison.same_argmax,
                comparison.reference_margin,
                started.elapsed().as_secs_f64() * 1000.0,
            );
            if !passes(paths.precision, step, &comparison) {
                failures.push(format!(
                    "{} step {index}: max_abs={} argmax_match={} margin={}",
                    case.name,
                    comparison.max_abs,
                    comparison.same_argmax,
                    comparison.reference_margin
                ));
            }
        }
        eprintln!("{} retained K/V: {} bytes", case.name, executor.kv_bytes());
    }

    // A prompt split across an extend crosses the window inside the cache.
    let long = manifest
        .cases
        .iter()
        .find(|case| case.name == "long_window")
        .expect("long_window case");
    let ids = &long.steps[0].input_ids;
    let split = ids.len() - 300;
    let mut executor = weights.executor(CONTEXT_TOKENS).expect("executor");
    executor.prefill_last_logits(&ids[..split]).expect("prefix");
    let actual = executor.extend_last_logits(&ids[split..]).expect("extend");
    let comparison = compare(
        &actual,
        &long.steps[0],
        full_logits(&long.steps[0]).as_deref(),
    );
    eprintln!(
        "long_window split at {split}: max_abs={:.5} argmax_match={}",
        comparison.max_abs, comparison.same_argmax
    );
    if !passes(paths.precision, &long.steps[0], &comparison) {
        failures.push(format!(
            "long_window split: max_abs={} argmax_match={}",
            comparison.max_abs, comparison.same_argmax
        ));
    }

    eprintln!(
        "worst max_abs over all steps: {worst_overall:.5} ({})",
        if paths.logits_dir.is_some() {
            "full vocabulary"
        } else {
            "reference top 8"
        }
    );
    assert!(
        failures.is_empty(),
        "parity failures:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore = "timing; requires METALLIX_GEMMA4_MODEL on Apple-Silicon Metal"]
fn greedy_decode_throughput() {
    let Some(paths) = paths() else { return };
    let prompt = &paths
        .manifest
        .cases
        .iter()
        .find(|case| case.name == "chat_turn")
        .expect("chat_turn case")
        .steps[0]
        .input_ids;
    let weights = Gemma4MlxWeights::load(&paths.model, paths.precision).expect("weights");
    let mut executor = weights.executor(CONTEXT_TOKENS).expect("executor");
    let started = Instant::now();
    let mut logits = executor.prefill_last_logits(prompt).expect("prefill");
    let prefill = started.elapsed();
    // Two warm-up steps, then a timed run.
    for _ in 0..2 {
        let token = i32::try_from(argmax(&logits)).expect("token");
        logits = executor.decode_last_logits(token).expect("decode");
    }
    let steps = 64;
    let started = Instant::now();
    for _ in 0..steps {
        let token = i32::try_from(argmax(&logits)).expect("token");
        logits = executor.decode_last_logits(token).expect("decode");
    }
    let decode = started.elapsed();
    eprintln!(
        "{:?}: prefill {} tokens {:.1} ms; decode {steps} tokens {:.2} tok/s ({:.1} ms/token, includes logit readback)",
        paths.precision,
        prompt.len(),
        prefill.as_secs_f64() * 1000.0,
        f64::from(steps) / decode.as_secs_f64(),
        decode.as_secs_f64() * 1000.0 / f64::from(steps),
    );
}
