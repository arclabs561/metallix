//! Opt-in measurement for the GPU greedy pick: per decode step, the full
//! logits row read back and argmaxed on the host versus one token ID picked
//! on the GPU, interleaved step by step in one process at one context
//! length (`METALLIX_QWEN35_CONTEXT`, default 4096). Prints paired ratios.
#![cfg(feature = "metal")]

use std::{env, path::PathBuf, time::Instant};

use qwen35::forward::{Qwen35Precision, Qwen35Weights};

#[test]
#[ignore = "measurement; run with --ignored"]
fn host_row_versus_gpu_pick() {
    let Some(model) = env::var_os("METALLIX_QWEN35_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN35_MODEL is not set");
        return;
    };
    let context: usize = env::var("METALLIX_QWEN35_CONTEXT")
        .map_or(4_096, |value| value.parse().expect("context length"));
    let weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    let prompt: Vec<i32> = (0..context)
        .map(|index| i32::try_from(1_000 + (index * 7_919) % 50_000).expect("id"))
        .collect();
    let mut host = weights.executor();
    let mut gpu = weights.executor();
    host.prefill_last_logits(&prompt).expect("host prefill");
    gpu.prefill_last_logits(&prompt).expect("gpu prefill");
    let mut ratios = Vec::new();
    for step in 0..48 {
        let token = 2_000 + step;
        let started = Instant::now();
        let logits = host.decode_last_logits(token).expect("host decode");
        let (host_id, _) = logits.iter().copied().enumerate().fold(
            (0, f32::NEG_INFINITY),
            |best, (index, value)| {
                if value > best.1 { (index, value) } else { best }
            },
        );
        let host_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        let gpu_id = gpu.decode_greedy(token).expect("gpu decode");
        let gpu_ms = started.elapsed().as_secs_f64() * 1e3;
        assert!(logits.iter().all(|value| value.is_finite()));
        assert_eq!(i32::try_from(host_id).expect("argmax id"), gpu_id);
        if step >= 8 {
            ratios.push(host_ms / gpu_ms);
        }
    }
    ratios.sort_by(f64::total_cmp);
    println!(
        "{{\"context\": {context}, \"host_over_gpu_median\": {:.3}, \"p10\": {:.3}, \"p90\": {:.3}}}",
        ratios[ratios.len() / 2],
        ratios[ratios.len() / 10],
        ratios[ratios.len() * 9 / 10]
    );
}
