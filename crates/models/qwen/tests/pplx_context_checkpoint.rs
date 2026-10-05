//! Opt-in: pplx-embed-context-v1-0.6b on Metal against the CPU float32 source
//! oracle in `fixtures/pplx-embed-context-v1-0.6b/context-reference.json`,
//! written by `scripts/pplx_context_reference.py` from the pinned revision's
//! own `encode` path (chunks joined with `<|endoftext|>`, bidirectional
//! attention, mean pooling per chunk span, int8 tanh or binary quantization).
//!
//! Tolerances, declared before the first run (also in the fixture's
//! `tolerance_policy`); the checkpoint is float32, used as stored:
//! - token IDs from `tokenizer.json` equal the oracle's;
//! - per chunk, cosine(native pooled, oracle pooled) >= 1 - 1e-5, and an
//!   all-zero oracle chunk (an empty span) is exactly zero natively;
//! - int8 codes differ from the oracle's by at most 1;
//! - binary signs equal the oracle's except where |oracle pooled| <= 1e-4.
#![cfg(feature = "metal")]

use std::{env, path::PathBuf, time::Instant};

use qwen::{
    embedding::{join_context_chunks, quantize_binary, quantize_int8_tanh},
    metal::Qwen3MlxWeights,
};
use serde_json::Value;
use tokenizers::Tokenizer;

const POOLED_COSINE_MIN: f64 = 1.0 - 1e-5;
const INT8_MAX_CODE_DIFF: i32 = 1;
const BINARY_NEAR_ZERO: f32 = 1e-4;

fn dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}

fn f32le_hex(value: &Value) -> Vec<f32> {
    let bytes: Vec<u8> = value
        .as_str()
        .expect("hex")
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn codes(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("codes")
        .iter()
        .map(|code| i32::try_from(code.as_i64().expect("code")).expect("small"))
        .collect()
}

fn setup() -> Option<(Qwen3MlxWeights, Tokenizer, Value)> {
    let Some(model) = env::var_os("METALLIX_PPLX_CONTEXT_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_PPLX_CONTEXT_MODEL is not set");
        return None;
    };
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/pplx-embed-context-v1-0.6b/context-reference.json"
    ))
    .expect("fixture JSON");
    assert_eq!(
        fixture["revision"],
        "b42df969d4d78d1840769e45c68c4e4ce763768b"
    );
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    let weights = Qwen3MlxWeights::load(&model).expect("load checkpoint");
    Some((weights, tokenizer, fixture))
}

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<i32> {
    tokenizer
        .encode(text, true)
        .expect("encode")
        .get_ids()
        .iter()
        .map(|&id| i32::try_from(id).expect("token ID"))
        .collect()
}

#[test]
#[ignore = "requires METALLIX_PPLX_CONTEXT_MODEL pointing to pplx-embed-context-v1-0.6b@b42df96 on Apple-Silicon Metal"]
fn context_chunks_match_the_source_oracle() {
    let Some((weights, tokenizer, fixture)) = setup() else {
        return;
    };
    let (mut worst_gap, mut int8_equal, mut int8_total) = (0.0_f64, 0_usize, 0_usize);
    for document in fixture["documents"].as_array().expect("documents") {
        let name = document["name"].as_str().expect("name");
        let chunks: Vec<&str> = document["chunks"]
            .as_array()
            .expect("chunks")
            .iter()
            .map(|chunk| chunk.as_str().expect("chunk"))
            .collect();
        let ids = encode(&tokenizer, &join_context_chunks(&chunks).expect("join"));
        assert_eq!(ids, codes(&document["input_ids"]), "{name}: token IDs");

        let native = weights.embed_context_chunks(&ids).expect("embed chunks");
        assert_eq!(native.len(), chunks.len(), "{name}: chunk count");
        for (index, pooled) in native.iter().enumerate() {
            let oracle = f32le_hex(&document["pooled_f32le"][index]);
            if oracle.iter().all(|&value| value == 0.0) {
                assert!(pooled.iter().all(|&value| value == 0.0), "{name}[{index}]");
            } else {
                let gap = 1.0
                    - dot(pooled, &oracle) / (dot(pooled, pooled) * dot(&oracle, &oracle)).sqrt();
                worst_gap = worst_gap.max(gap);
                assert!(
                    gap <= 1.0 - POOLED_COSINE_MIN,
                    "{name}[{index}]: 1 - cosine {gap}"
                );
            }

            let int8 = quantize_int8_tanh(pooled);
            for (native_code, oracle_code) in int8.iter().zip(codes(&document["int8"][index])) {
                let diff = (i32::from(*native_code) - oracle_code).abs();
                assert!(
                    diff <= INT8_MAX_CODE_DIFF,
                    "{name}[{index}]: int8 diff {diff}"
                );
                int8_equal += usize::from(diff == 0);
                int8_total += 1;
            }
            let binary = quantize_binary(pooled);
            for ((native_sign, oracle_sign), value) in binary
                .iter()
                .zip(codes(&document["binary"][index]))
                .zip(&oracle)
            {
                assert!(
                    i32::from(*native_sign) == oracle_sign || value.abs() <= BINARY_NEAR_ZERO,
                    "{name}[{index}]: binary sign at {value}"
                );
            }
        }
    }
    eprintln!("worst 1 - cosine {worst_gap:.3e}; int8 exact {int8_equal}/{int8_total}");
}

/// Timing only: one forward plus pooling per call, at sequence lengths built
/// by repeating the longest fixture document's tokens. Asserts nothing about
/// time.
#[test]
#[ignore = "timing; requires METALLIX_PPLX_CONTEXT_MODEL pointing to pplx-embed-context-v1-0.6b@b42df96"]
fn context_latency_by_length() {
    let Some((weights, _, fixture)) = setup() else {
        return;
    };
    let source = fixture["documents"]
        .as_array()
        .expect("documents")
        .iter()
        .map(|document| codes(&document["input_ids"]))
        .max_by_key(Vec::len)
        .expect("a document");
    for tokens in [8, 64, 256, 512] {
        let ids: Vec<i32> = source.iter().copied().cycle().take(tokens).collect();
        let started = Instant::now();
        weights.embed_context_chunks(&ids).expect("embed chunks");
        let first = started.elapsed();
        let mut warm: Vec<_> = (0..10)
            .map(|_| {
                let started = Instant::now();
                weights.embed_context_chunks(&ids).expect("embed chunks");
                started.elapsed()
            })
            .collect();
        warm.sort();
        let ms = |duration: std::time::Duration| duration.as_secs_f64() * 1e3;
        eprintln!(
            "f32 {tokens:>3} tokens: first {:>7.1} ms, warm median {:>6.1} ms, min {:>6.1} ms",
            ms(first),
            ms(warm[5]),
            ms(warm[0])
        );
    }
}
