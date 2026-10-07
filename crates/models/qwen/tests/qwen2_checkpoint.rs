//! Opt-in: Qwen2.5-0.5B-Instruct (`Qwen2ForCausalLM`) on Metal against the CPU
//! float32 transformers oracle in
//! `fixtures/qwen2.5-0.5b-instruct/logit-reference.json`, written by
//! `scripts/qwen2-reference.py` at the pinned revision.
//!
//! Set `METALLIX_QWEN2_MODEL` to the checkpoint directory and
//! `METALLIX_QWEN2_REFERENCE` to the script's `--output-dir`, which holds
//! the full-vocabulary f32 logits. Each logit file is bound to the fixture by
//! its exact top-8 IDs and values.
//!
//! Tolerances, declared before the first run and equal to the `MiniCPM5` and
//! Qwen3 decoder logit parity checks:
//! - token IDs from `tokenizer.json` (no added special tokens) for each
//!   rendered prompt equal the oracle's;
//! - with weights promoted to float32, every logit of the cached prefill and
//!   of each teacher-forced cached decode step is within
//!   `5e-4 + 1e-4 * |reference|` of the oracle, and every argmax agrees.
#![cfg(feature = "metal")]

use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use qwen::{DecoderFamily, forward::DEFAULT_RESIDENT_CHAT_KV_BUDGET_BYTES, metal::Qwen3MlxWeights};
use serde_json::Value;
use tokenizers::Tokenizer;

const REVISION: &str = "7ae557604adf67be50417f59c2c2f167def9a775";
const VOCABULARY: usize = 151_936;
const ABSOLUTE_TOLERANCE: f64 = 5e-4;
const RELATIVE_TOLERANCE: f64 = 1e-4;
const CONTEXT_TOKENS: usize = 512;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../../fixtures/qwen2.5-0.5b-instruct/logit-reference.json"
    ))
    .expect("fixture JSON")
}

fn paths() -> Option<(PathBuf, PathBuf)> {
    let model = env::var_os("METALLIX_QWEN2_MODEL").map(PathBuf::from);
    let reference = env::var_os("METALLIX_QWEN2_REFERENCE").map(PathBuf::from);
    let paths = model.zip(reference);
    if paths.is_none() {
        eprintln!("skipping: METALLIX_QWEN2_MODEL and METALLIX_QWEN2_REFERENCE are not both set");
    }
    paths
}

fn ids(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .expect("ID array")
        .iter()
        .map(|id| i32::try_from(id.as_i64().expect("integer ID")).expect("ID fits i32"))
        .collect()
}

/// Reads one reference logit file and checks it is the one the fixture names.
fn reference_logits(directory: &Path, step: &Value) -> Vec<f32> {
    let name = step["logits_file"].as_str().expect("file name");
    let bytes = fs::read(directory.join(name)).expect("reference logits file");
    assert_eq!(bytes.len(), VOCABULARY * 4, "{name} length");
    let logits: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
        .collect();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    for (rank, (id, value)) in step["top8_ids"]
        .as_array()
        .expect("top-8 IDs")
        .iter()
        .zip(step["top8_logits"].as_array().expect("top-8 logits"))
        .enumerate()
    {
        let id = usize::try_from(id.as_u64().expect("ID")).expect("ID fits usize");
        // The capture wrote f32 values; their JSON doubles convert exactly.
        #[allow(clippy::cast_possible_truncation, reason = "exact f32 round trip")]
        let value = value.as_f64().expect("logit") as f32;
        assert_eq!((order[rank], logits[id]), (id, value), "{name} rank {rank}");
    }
    logits
}

fn argmax(values: &[f32]) -> usize {
    (0..values.len())
        .max_by(|&a, &b| values[a].total_cmp(&values[b]))
        .expect("nonempty logits")
}

struct Comparison {
    max_absolute_error: f64,
    mismatches: usize,
}

fn compare(actual: &[f32], expected: &[f32]) -> Comparison {
    assert_eq!(actual.len(), expected.len());
    let mut comparison = Comparison {
        max_absolute_error: 0.0,
        mismatches: 0,
    };
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert!(actual.is_finite(), "non-finite native logit");
        let difference = (f64::from(actual) - f64::from(expected)).abs();
        comparison.max_absolute_error = comparison.max_absolute_error.max(difference);
        if difference > ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * f64::from(expected).abs() {
            comparison.mismatches += 1;
        }
    }
    comparison
}

#[test]
fn tokenizer_reproduces_rendered_prompt_ids() {
    let Some((model, _)) = paths() else { return };
    let fixture = fixture();
    assert_eq!(fixture["revision"], REVISION);
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    for case in fixture["cases"].as_array().expect("cases") {
        let Some(rendered) = case["rendered"].as_str() else {
            continue;
        };
        let encoded = tokenizer.encode(rendered, false).expect("encode");
        let encoded: Vec<i32> = encoded
            .get_ids()
            .iter()
            .map(|&id| i32::try_from(id).expect("ID fits i32"))
            .collect();
        assert_eq!(encoded, ids(&case["input_ids"]), "{}", case["name"]);
    }
}

#[test]
fn cached_prefill_and_decode_match_the_float32_oracle() {
    let Some((model, reference)) = paths() else {
        return;
    };
    let fixture = fixture();
    assert_eq!(fixture["revision"], REVISION);

    let started = Instant::now();
    let mut weights = Qwen3MlxWeights::load(&model).expect("checkpoint loads");
    weights.prepare_float32().expect("float32 weights");
    eprintln!(
        "load_and_float32_ms={:.0}",
        started.elapsed().as_secs_f64() * 1e3
    );
    assert_eq!(
        weights.inspection().contract().family(),
        DecoderFamily::Qwen2
    );

    let mut worst = 0.0_f64;
    let mut steps = 0;
    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("case name");
        let records = case["steps"].as_array().expect("steps");
        let mut executor = weights
            .resident_chat_executor(CONTEXT_TOKENS, DEFAULT_RESIDENT_CHAT_KV_BUDGET_BYTES)
            .expect("resident executor");
        let mut case_worst = 0.0_f64;
        for (index, step) in records.iter().enumerate() {
            // Teacher forcing: feed the oracle's token so a near-tie cannot
            // move later steps onto a different sequence.
            let native = if index == 0 {
                executor
                    .prefill_last_logits(&ids(&case["input_ids"]))
                    .expect("prefill")
            } else {
                let fed = i32::try_from(step["fed_token"].as_i64().expect("fed token"))
                    .expect("ID fits i32");
                executor.decode_last_logits(fed).expect("decode")
            };
            let expected = reference_logits(&reference, step);
            let comparison = compare(&native, &expected);
            assert_eq!(
                argmax(&native),
                usize::try_from(step["argmax"].as_u64().expect("argmax")).expect("usize"),
                "{name} step {index} argmax"
            );
            assert_eq!(
                comparison.mismatches, 0,
                "{name} step {index}: {} logits outside tolerance, max abs error {}",
                comparison.mismatches, comparison.max_absolute_error
            );
            case_worst = case_worst.max(comparison.max_absolute_error);
            steps += 1;
        }
        eprintln!(
            "{name}: steps={} max_abs_error={case_worst:.3e}",
            records.len()
        );
        worst = worst.max(case_worst);
    }
    eprintln!("all: steps={steps} argmax_agreement={steps}/{steps} max_abs_error={worst:.3e}");

    // Batch-1 greedy decode rate on the serving path, after a chat prefill.
    let chat = &fixture["cases"][2];
    let mut executor = weights
        .resident_chat_executor(CONTEXT_TOKENS, DEFAULT_RESIDENT_CHAT_KV_BUDGET_BYTES)
        .expect("resident executor");
    let started = Instant::now();
    let mut logits = executor
        .prefill_last_logits(&ids(&chat["input_ids"]))
        .expect("prefill");
    let prefill = started.elapsed();
    let decode_tokens: u32 = 64;
    let started = Instant::now();
    for _ in 0..decode_tokens {
        let next = i32::try_from(argmax(&logits)).expect("ID fits i32");
        logits = executor.decode_last_logits(next).expect("decode");
    }
    let decode = started.elapsed();
    eprintln!(
        "float32 prefill_tokens={} prefill_ms={:.1} decode_tokens={decode_tokens} decode_tok_s={:.2}",
        executor.cached_tokens() - usize::try_from(decode_tokens).expect("small count"),
        prefill.as_secs_f64() * 1e3,
        f64::from(decode_tokens) / decode.as_secs_f64()
    );
}
