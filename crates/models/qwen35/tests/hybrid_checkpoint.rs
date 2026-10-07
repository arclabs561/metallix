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

/// Index of the first maximum, as MLX's argmax and a host pick that keeps
/// the first maximum both break ties.
fn first_argmax(logits: &[f32]) -> i32 {
    let mut best = 0;
    for (index, &logit) in logits.iter().enumerate() {
        if logit > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).expect("token id")
}

/// The GPU greedy pick returns the host argmax of the same step's full
/// logits row at every step of a cached decode, so switching the decode
/// loop to picks cannot change greedy output.
#[test]
fn greedy_pick_matches_the_host_argmax() {
    let Some(case) = case() else { return };
    let weights = Qwen35Weights::load(&case.model, Qwen35Precision::Checkpoint).expect("load");
    let prompt = ids(&case.reference["cases"][2]["steps"][0]["input_ids"]);
    let mut host = weights.executor();
    let mut gpu = weights.executor();
    let mut logits = host.prefill_last_logits(&prompt).expect("host prefill");
    let mut picked = gpu.extend_greedy(&prompt).expect("gpu prefill");
    for step in 0..24 {
        let expected = first_argmax(&logits);
        assert_eq!(picked, expected, "step {step}: GPU pick vs host argmax");
        logits = host.decode_last_logits(expected).expect("host decode");
        picked = gpu.decode_greedy(expected).expect("gpu decode");
    }
}

/// Full-attention K/V grows in tiers (128, 512, then doubling). A sequence
/// fed in pieces whose ends cross each tier, including single-token decode
/// steps over a boundary, matches one prefill of the whole sequence.
#[test]
fn appends_across_kv_tiers_match_a_single_prefill() {
    let Some(case) = case() else { return };
    let weights = Qwen35Weights::load(&case.model, Qwen35Precision::Float32).expect("load");
    let seed = ids(&case.reference["cases"][3]["steps"][0]["input_ids"]);
    let sequence: Vec<i32> = seed.iter().copied().cycle().take(1_100).collect();
    let whole = weights
        .executor()
        .prefill_last_logits(&sequence)
        .expect("whole");
    let mut split = weights.executor();
    split.prefill_last_logits(&sequence[..126]).expect("head");
    for &token in &sequence[126..130] {
        split.decode_last_logits(token).expect("decode over 128");
    }
    split
        .extend_last_logits(&sequence[130..511])
        .expect("to 511");
    split
        .extend_last_logits(&sequence[511..514])
        .expect("over 512");
    split
        .extend_last_logits(&sequence[514..1_023])
        .expect("to 1023");
    for &token in &sequence[1_023..1_099] {
        split.decode_last_logits(token).expect("decode over 1024");
    }
    let pieces = split
        .extend_last_logits(&sequence[1_099..])
        .expect("last token");
    let diff = whole
        .iter()
        .zip(&pieces)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    eprintln!("tiered appends vs whole prefill: max |dlogit| {diff:.3e}");
    assert!(diff <= 1e-3, "tiered appends diverged by {diff}");
}

/// A snapshot taken mid-prompt and restored into a fresh executor, then
/// extended with the rest of the prompt, gives the one-shot prefill's logits;
/// the executor the snapshot came from keeps going unaffected. Boundaries sit
/// inside and past the first prefill chunk.
#[test]
fn restored_snapshot_matches_a_single_prefill() {
    let Some(case) = case() else { return };
    let weights = Qwen35Weights::load(&case.model, Qwen35Precision::Float32).expect("load");
    let prompt = ids(&case.reference["cases"][3]["steps"][0]["input_ids"]);
    assert!(prompt.len() > 130);
    let whole = weights
        .executor()
        .prefill_last_logits(&prompt)
        .expect("whole");
    let max_diff = |logits: &[f32]| {
        whole
            .iter()
            .zip(logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    };
    for boundary in [37, 130] {
        let mut original = weights.executor();
        original
            .prefill_last_logits(&prompt[..boundary])
            .expect("head");
        let snapshot = original.snapshot().expect("snapshot");
        assert_eq!(snapshot.tokens(), boundary);
        let mut restored = weights.executor_from(&snapshot).expect("restore");
        let resumed = restored
            .extend_last_logits(&prompt[boundary..])
            .expect("restored tail");
        let continued = original
            .extend_last_logits(&prompt[boundary..])
            .expect("original tail");
        // Roll back to the same complete hybrid state after a different branch
        // has advanced. Recurrent state must be restored, not truncated like KV.
        restored
            .decode_last_logits(1)
            .expect("advance a rejected branch");
        let replayed = weights
            .executor_from(&snapshot)
            .expect("restore after branch")
            .extend_last_logits(&prompt[boundary..])
            .expect("replay tail");
        assert_eq!(
            replayed, resumed,
            "stash and replay must recover the same logits"
        );
        let (resumed_diff, continued_diff) = (max_diff(&resumed), max_diff(&continued));
        eprintln!(
            "boundary {boundary} ({} snapshot bytes): restored max |dlogit| {resumed_diff:.3e}, original {continued_diff:.3e}",
            snapshot.state_bytes()
        );
        assert_eq!(argmax(&resumed), argmax(&whole), "boundary {boundary}");
        assert!(
            resumed_diff <= 1e-3,
            "restored at {boundary}: {resumed_diff}"
        );
        assert!(
            continued_diff <= 1e-3,
            "original at {boundary}: {continued_diff}"
        );
    }
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
