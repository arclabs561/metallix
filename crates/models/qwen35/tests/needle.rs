//! Opt-in long-context needle runner: greedy continuations of prompts whose
//! token IDs a reference engine wrote (`niah_ref.py --prompts-out`), so both
//! engines see identical input. Scoring against the reference happens
//! outside; this only generates.
//!
//! `METALLIX_QWEN35_MODEL` names the checkpoint, `METALLIX_QWEN35_NEEDLE`
//! the prompts JSON (a list of objects with `ids`, `length`, `depth`, `seed`
//! and `value`), and the optional `METALLIX_QWEN35_ROPE_SCALING=yarn:4`
//! applies the qualified override. Prints one JSON line per prompt, with
//! MLX active, cache and peak bytes after each prefill. The optional
//! `METALLIX_QWEN35_NEEDLE_CACHE_GIB` caps MLX's buffer cache, so a capped
//! and an uncapped run tell allocator retention from live growth.
#![cfg(feature = "metal")]

use std::{env, fs, path::PathBuf, time::Instant};

use qwen35::{
    RopeScalingRequest,
    forward::{Qwen35Precision, Qwen35Weights},
};
use serde_json::Value;

const NEW_TOKENS: usize = 24;

#[test]
#[ignore = "long-context generation; run with --ignored"]
fn greedy_needle_continuations() {
    let (Some(model), Some(prompts)) = (
        env::var_os("METALLIX_QWEN35_MODEL").map(PathBuf::from),
        env::var_os("METALLIX_QWEN35_NEEDLE").map(PathBuf::from),
    ) else {
        eprintln!("skipping: METALLIX_QWEN35_MODEL or METALLIX_QWEN35_NEEDLE is not set");
        return;
    };
    let mut weights = Qwen35Weights::load(&model, Qwen35Precision::Checkpoint).expect("load");
    if let Ok(scaling) = env::var("METALLIX_QWEN35_ROPE_SCALING") {
        let factor = scaling
            .strip_prefix("yarn:")
            .and_then(|factor| factor.parse().ok())
            .expect("METALLIX_QWEN35_ROPE_SCALING must be yarn:<factor>");
        weights = weights
            .with_rope_scaling(RopeScalingRequest::Yarn { factor })
            .expect("qualified scaling");
    }
    if let Ok(gib) = env::var("METALLIX_QWEN35_NEEDLE_CACHE_GIB") {
        let gib: usize = gib
            .parse()
            .expect("METALLIX_QWEN35_NEEDLE_CACHE_GIB must be an integer");
        mlx_rs::memory::set_cache_limit(gib << 30).expect("cache limit");
    }
    let prompts: Vec<Value> =
        serde_json::from_str(&fs::read_to_string(prompts).expect("prompts file")).expect("JSON");
    for prompt in prompts {
        let ids: Vec<i32> = prompt["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|id| i32::try_from(id.as_i64().expect("integer id")).expect("i32 id"))
            .collect();
        let mut executor = weights.executor();
        mlx_rs::memory::reset_peak_memory().expect("reset peak");
        let started = Instant::now();
        let mut token = executor.extend_greedy(&ids).expect("prefill");
        let prefill_s = started.elapsed().as_secs_f64();
        let memory = serde_json::json!({
            "active": mlx_rs::memory::active_memory().expect("active"),
            "cache": mlx_rs::memory::cache_memory().expect("cache"),
            "peak": mlx_rs::memory::peak_memory().expect("peak"),
        });
        let mut output = vec![token];
        let started = Instant::now();
        while output.len() < NEW_TOKENS {
            token = executor.decode_greedy(token).expect("decode");
            output.push(token);
        }
        let steps = f64::from(u32::try_from(NEW_TOKENS - 1).expect("small"));
        let decode_ms = started.elapsed().as_secs_f64() * 1e3 / steps;
        println!(
            "{}",
            serde_json::json!({
                "tokens": ids.len(),
                "length": prompt["length"],
                "depth": prompt["depth"],
                "seed": prompt["seed"],
                "value": prompt["value"],
                "output_ids": output,
                "prefill_s": prefill_s,
                "prefill_mlx_bytes": memory,
                "decode_ms_per_token": decode_ms,
            })
        );
    }
}
