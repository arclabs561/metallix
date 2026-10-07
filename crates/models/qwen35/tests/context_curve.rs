//! Opt-in measurement: decode milliseconds per token at several context
//! lengths, through the public executor API only, so the same file measures
//! any build. Ignored by default because it measures rather than checks.
//!
//! `METALLIX_QWEN35_MODEL` names the checkpoint; `METALLIX_QWEN35_CONTEXTS`
//! is a comma-separated list of prompt lengths (default `1024,16384`).
//! Prints one JSON line per context.
#![cfg(feature = "metal")]

use std::{env, path::PathBuf, time::Instant};

use qwen35::forward::{Qwen35Precision, Qwen35Weights};

#[test]
#[ignore = "measurement; run with --ignored"]
fn decode_ms_at_context() {
    let Some(model) = env::var_os("METALLIX_QWEN35_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN35_MODEL is not set");
        return;
    };
    let contexts: Vec<usize> = env::var("METALLIX_QWEN35_CONTEXTS")
        .unwrap_or_else(|_| "1024,16384".to_owned())
        .split(',')
        .map(|value| value.trim().parse().expect("context length"))
        .collect();
    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    for context in contexts {
        // Deterministic ordinary-text-range IDs; decode cost does not depend
        // on their values.
        let prompt: Vec<i32> = (0..context)
            .map(|index| i32::try_from(1_000 + (index * 7_919) % 50_000).expect("id"))
            .collect();
        let mut executor = weights.executor();
        let started = Instant::now();
        executor.prefill_last_logits(&prompt).expect("prefill");
        let prefill_s = started.elapsed().as_secs_f64();
        let mut step_ms = Vec::new();
        for step in 0..40 {
            let token = 2_000 + step;
            let started = Instant::now();
            executor.decode_last_logits(token).expect("decode");
            step_ms.push(started.elapsed().as_secs_f64() * 1e3);
        }
        let mut timed = step_ms[8..].to_vec();
        timed.sort_by(f64::total_cmp);
        let median = timed[timed.len() / 2];
        println!(
            "{{\"context\": {context}, \"prefill_s\": {prefill_s:.2}, \"decode_ms_median\": {median:.3}, \"decode_ms\": {step_ms:?}}}"
        );
    }
}
