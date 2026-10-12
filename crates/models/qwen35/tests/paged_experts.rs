//! Opt-in: routed experts read on demand give the same logits, bit for bit,
//! as the fully resident model on a real MLX affine `qwen3_5_moe`
//! checkpoint, at budgets from one layer's experts to all of them.
//!
//! Set `METALLIX_QWEN35_PAGED_MODEL` to the checkpoint directory (the 4-layer
//! truncation of Qwen3.6-35B-A3B 4-bit from
//! `scripts/qwen35-mlx-affine-reference.py truncate` keeps this to a few GB).
//! `METALLIX_QWEN35_PAGED_STEPS` sets the greedy steps per prompt (default
//! 128). `METALLIX_QWEN35_PAGED_PROMPTS` (comma-separated indices) and
//! `METALLIX_QWEN35_PAGED_BUDGETS` (`layer`, `half`, `all`) narrow the run.
//! `METALLIX_QWEN35_PAGED_GPU_GIB` sets the MLX byte cap (default 8 GiB for
//! the four-layer checkpoint). Both paths use a 2 GiB allocator-cache limit.
//! Checks after loading, prefill and each decode enforce the observed MLX peak
//! and active-plus-cache bytes. These are cooperative evaluation-boundary checks,
//! not a hard allocation limit or a bound on other host/process memory.
//! Pager reads set `F_NOCACHE`, but pages already in the OS file cache (from
//! the resident load) may still be served from it.

#![cfg(feature = "metal")]

use std::{env, path::PathBuf, time::Instant};

use checkpoint::{ReadCache, safetensors::SafetensorsCheckpoint};
use qwen35::{
    Qwen35Config, Qwen35Mlp,
    experts::Qwen35ExpertLocator,
    forward::{Qwen35Paging, Qwen35Precision, Qwen35Weights},
};

fn argmax(logits: &[f32]) -> i32 {
    let index = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("nonempty logits")
        .0;
    i32::try_from(index).expect("token id")
}

/// Prefill `prompt`, then decode the greedy tokens `steps` times; every
/// step's logits.
fn run(weights: &Qwen35Weights, prompt: &[i32], steps: usize, cap: usize) -> Vec<Vec<f32>> {
    let mut executor = weights.executor();
    let mut out = vec![executor.prefill_last_logits(prompt).expect("prefill")];
    check_memory("prefill", cap, true);
    for _ in 0..steps {
        let token = argmax(out.last().expect("logits"));
        out.push(executor.decode_last_logits(token).expect("decode"));
        check_memory("decode", cap, false);
    }
    check_memory("prompt complete", cap, true);
    out
}

/// MLX counters include device allocations, unlike process RSS. The peak is
/// reset once per model run; cache bytes are separately included in live use.
fn check_memory(phase: &str, cap: usize, report: bool) {
    let active = mlx_rs::memory::active_memory().expect("active memory");
    let cached = mlx_rs::memory::cache_memory().expect("cached memory");
    let peak = mlx_rs::memory::peak_memory().expect("peak memory");
    if report {
        eprintln!("MLX {phase}: active={active} cached={cached} peak={peak} cap={cap}");
    }
    assert!(
        peak <= cap && active.saturating_add(cached) <= cap,
        "MLX resource cap exceeded after {phase}: active={active} cached={cached} peak={peak} cap={cap}"
    );
}

/// Deterministic token ids below `vocab`.
fn ids(count: usize, seed: u64, vocab: u64) -> Vec<i32> {
    (0..count as u64)
        .map(|i| {
            let mixed = (i + 1).wrapping_mul(2_654_435_761).wrapping_add(seed) % vocab;
            i32::try_from(mixed).expect("id")
        })
        .collect()
}

fn configure_memory_policy() -> usize {
    let gib: usize = env::var("METALLIX_QWEN35_PAGED_GPU_GIB")
        .unwrap_or_else(|_| "8".to_owned())
        .parse()
        .expect("GPU cap must be an integer GiB count");
    assert!(gib > 0, "GPU cap must be positive");
    let cap = gib.checked_mul(1 << 30).expect("GPU cap overflow");
    mlx_rs::memory::set_cache_limit(2 << 30).expect("allocator cache limit");
    cap
}

#[test]
fn paged_experts_match_resident_logits() {
    let Some(model) = env::var_os("METALLIX_QWEN35_PAGED_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN35_PAGED_MODEL is not set");
        return;
    };
    let cap = configure_memory_policy();
    let steps = env::var("METALLIX_QWEN35_PAGED_STEPS")
        .ok()
        .and_then(|steps| steps.parse().ok())
        .unwrap_or(128);
    let config =
        Qwen35Config::parse(&std::fs::read_to_string(model.join("config.json")).expect("config"))
            .expect("config");
    let Qwen35Mlp::Experts { experts, .. } = config.mlp() else {
        panic!("not a mixture-of-experts checkpoint");
    };
    let layers = config.layers().len();
    let source = SafetensorsCheckpoint::open(&model, ReadCache::Use).expect("table");
    let locator = Qwen35ExpertLocator::mlx_affine(&source, layers, experts).expect("locator");
    let layer_bytes = locator.expert_bytes(0) * experts as u64;
    let all = layer_bytes * layers as u64;
    let vocab = config.vocab_size() as u64;

    // Short prompts (unsorted gathers at decode), and one long prompt whose
    // prefill routes to nearly every expert of every layer.
    let prompts = [
        ids(12, 1, vocab),
        ids(40, 2, vocab),
        ids(7, 3, vocab),
        ids(1651, 4, vocab),
    ];
    let chosen: Vec<usize> = env::var("METALLIX_QWEN35_PAGED_PROMPTS").map_or_else(
        |_| (0..prompts.len()).collect(),
        |list| {
            list.split(',')
                .map(|i| i.trim().parse().expect("prompt index"))
                .collect()
        },
    );
    let resident_run = || {
        mlx_rs::memory::reset_peak_memory().expect("reset peak");
        let started = Instant::now();
        let resident = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("resident");
        check_memory("resident load", cap, true);
        let logits: Vec<_> = chosen
            .iter()
            .map(|&index| run(&resident, &prompts[index], steps, cap))
            .collect();
        eprintln!(
            "resident: prompts {chosen:?} x {steps} steps in {:.1?}",
            started.elapsed()
        );
        logits
    };
    let expected = resident_run();
    // A/A: a second load and run of the resident model. The bar stays
    // bitwise; this records whether the reference itself repeats.
    let repeat = differences(&chosen, &resident_run(), &expected);
    eprintln!("resident A/A: (prompt, step, max |d|) {repeat:?}");

    let budgets: Vec<u64> = env::var("METALLIX_QWEN35_PAGED_BUDGETS")
        .unwrap_or_else(|_| "layer,half,all".to_owned())
        .split(',')
        .map(|name| match name.trim() {
            "layer" => layer_bytes,
            "half" => all / 2,
            "all" => all,
            other => panic!("unknown budget {other:?}"),
        })
        .collect();
    let mut failures = Vec::new();
    for budget in budgets {
        mlx_rs::memory::reset_peak_memory().expect("reset peak");
        let started = Instant::now();
        let paged = Qwen35Weights::load_paged(
            &model,
            Qwen35Paging {
                budget_bytes: budget,
                readers: 8,
                cache: ReadCache::Bypass,
            },
        )
        .expect("paged load");
        check_memory("paged load", cap, true);
        let actual: Vec<_> = chosen
            .iter()
            .map(|&index| run(&paged, &prompts[index], steps, cap))
            .collect();
        let differing = differences(&chosen, &actual, &expected);
        let stats = paged.pager_stats().expect("stats");
        eprintln!(
            "budget {budget} ({}% of {all} expert bytes): {:.1?}, {stats:?}",
            budget * 100 / all,
            started.elapsed()
        );
        eprintln!("budget {budget}: (prompt, step, max |d|) {differing:?}");
        assert!(stats.resident_bytes <= budget);
        if !differing.is_empty() {
            failures.push((budget, differing.len()));
        }
    }
    assert!(
        failures.is_empty(),
        "differing steps per budget {failures:?}; resident A/A {repeat:?}"
    );
}

/// `(prompt, step, max |d|)` for every step whose logits differ.
fn differences(
    prompts: &[usize],
    actual: &[Vec<Vec<f32>>],
    expected: &[Vec<Vec<f32>>],
) -> Vec<(usize, usize, f32)> {
    let mut out = Vec::new();
    for ((&prompt, actual), expected) in prompts.iter().zip(actual).zip(expected) {
        for (step, (a, e)) in actual.iter().zip(expected).enumerate() {
            if a != e {
                let worst = a
                    .iter()
                    .zip(e)
                    .map(|(a, e)| (a - e).abs())
                    .fold(0.0_f32, f32::max);
                out.push((prompt, step, worst));
            }
        }
    }
    out
}
