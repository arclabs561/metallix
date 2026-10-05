//! Opt-in: Qwen3-Embedding-0.6B on Metal against the CPU float32 source oracle
//! in `fixtures/qwen3-embedding-0.6b/embedding-reference.json`, written by
//! `scripts/qwen3_embedding_reference.py` from the model card's recipe. The
//! fixture holds the card's four retrieval inputs plus 64 stress inputs
//! (lengths up to 500 tokens, many scripts, code, numbers, near-duplicates,
//! queries with and without instructions) and 21 scored pairs.
//!
//! Float32 tolerances, declared before the first run:
//! - token IDs from `tokenizer.json` with special tokens equal the oracle's;
//! - per input, cosine(native, oracle) >= 1 - 1e-5, with weights promoted to
//!   float32, and likewise for a 256-dimension truncation;
//! - pair scores and the card's 2 x 2 scores within 1e-4 of the oracle's, and
//!   the 2 x 2 within 1e-3 of the card's published scores.
//!
//! BF16 tolerances (checkpoint weights as stored), set from the stress set
//! measured with mlx-rs 0.32.0 (MLX 0.32.2): worst 1 - cosine 2.9e-4 on `ko`,
//! worst score delta 3.3e-3, BF16 runs bit-identical run to run. The bounds
//! leave about 1.4x and 1.5x headroom:
//! - per input, cosine(native, oracle) >= 1 - 4e-4;
//! - pair scores and the card's 2 x 2 scores within 5e-3 of the oracle's.
//!
//! The HF BF16 source's worst 1 - cosine is 4.7e-4. Under MLX 0.25.1 the
//! native BF16 worst was 8.0e-4 (on `hi`, score delta 5.2e-3) because MLX
//! before 0.29.3 computes BF16 sigmoid imprecisely, which `SiLU` uses in every
//! MLP; the bounds were then 1e-3 and 1e-2.
#![cfg(feature = "metal")]

use std::{collections::HashMap, env, path::PathBuf};

use qwen::{embedding::QWEN3_EMBEDDING_EOS_ID, metal::Qwen3MlxWeights};
use serde_json::Value;
use tokenizers::Tokenizer;

const F32_COSINE_MIN: f64 = 1.0 - 1e-5;
const F32_SCORE_VS_ORACLE: f64 = 1e-4;
const F32_SCORE_VS_CARD: f64 = 1e-3;
const BF16_COSINE_MIN: f64 = 1.0 - 4e-4;
const BF16_SCORE_VS_ORACLE: f64 = 5e-3;

fn dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    dot(left, right) / (dot(left, left) * dot(right, right)).sqrt()
}

fn f32le_hex(value: &Value) -> Vec<f32> {
    let hex = value.as_str().expect("hex embedding").as_bytes();
    let bytes: Vec<u8> = hex
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

struct Case {
    model: PathBuf,
    fixture: Value,
    tokenizer: Tokenizer,
}

fn case() -> Option<Case> {
    let Some(model) = env::var_os("METALLIX_QWEN_EMBEDDING_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN_EMBEDDING_MODEL is not set");
        return None;
    };
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/qwen3-embedding-0.6b/embedding-reference.json"
    ))
    .expect("fixture JSON");
    assert_eq!(
        fixture["revision"],
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    );
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    Some(Case {
        model,
        fixture,
        tokenizer,
    })
}

/// Embeds every input, checking token IDs; returns name -> (native, oracle).
fn embed_all(
    case: &Case,
    weights: &Qwen3MlxWeights,
    dimensions: Option<usize>,
) -> HashMap<String, (Vec<f32>, Vec<f32>)> {
    let mut embeddings = HashMap::new();
    for input in case.fixture["inputs"].as_array().expect("inputs") {
        let name = input["name"].as_str().expect("name");
        let ids: Vec<i32> = case
            .tokenizer
            .encode(input["text"].as_str().expect("text"), true)
            .expect("encode")
            .get_ids()
            .iter()
            .map(|&id| i32::try_from(id).expect("token ID"))
            .collect();
        let expected_ids: Vec<i32> = input["input_ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|id| i32::try_from(id.as_i64().expect("id")).expect("token ID"))
            .collect();
        assert_eq!(ids, expected_ids, "{name}: token IDs");
        assert_eq!(ids.last(), Some(&QWEN3_EMBEDDING_EOS_ID));
        let native = weights.embed(&ids, dimensions).expect("embed");
        let mut oracle = f32le_hex(&input["embedding_f32le"]);
        oracle.truncate(native.len());
        embeddings.insert(name.to_owned(), (native, oracle));
    }
    embeddings
}

/// The worst `1 - cosine` over inputs, after printing the five worst.
fn worst_vector_gap(label: &str, embeddings: &HashMap<String, (Vec<f32>, Vec<f32>)>) -> f64 {
    let mut gaps: Vec<(f64, &str)> = embeddings
        .iter()
        .map(|(name, (native, oracle))| (1.0 - cosine(native, oracle), name.as_str()))
        .collect();
    gaps.sort_by(|a, b| b.0.total_cmp(&a.0));
    eprintln!("{label}: worst 1 - cosine {:?}", &gaps[..5]);
    gaps[0].0
}

/// Every pair and card score as (label, native, oracle, card if any).
fn scores(
    case: &Case,
    embeddings: &HashMap<String, (Vec<f32>, Vec<f32>)>,
) -> Vec<(String, f64, f64, Option<f64>)> {
    let native = |name: &str| &embeddings[name].0;
    let mut scores = Vec::new();
    for pair in case.fixture["pair_scores"].as_array().expect("pairs") {
        let (left, right) = (
            pair["left"].as_str().unwrap(),
            pair["right"].as_str().unwrap(),
        );
        scores.push((
            format!("{left}~{right}"),
            dot(native(left), native(right)),
            pair["score"].as_f64().unwrap(),
            None,
        ));
    }
    for (q, query) in ["card_query_0", "card_query_1"].into_iter().enumerate() {
        for (d, document) in ["card_document_0", "card_document_1"]
            .into_iter()
            .enumerate()
        {
            scores.push((
                format!("{query}~{document}"),
                dot(native(query), native(document)),
                case.fixture["scores"][q][d].as_f64().unwrap(),
                case.fixture["card_scores"][q][d].as_f64(),
            ));
        }
    }
    scores
}

fn worst_score_delta(label: &str, scores: &[(String, f64, f64, Option<f64>)]) -> f64 {
    let (name, native, oracle, _) = scores
        .iter()
        .max_by(|a, b| (a.1 - a.2).abs().total_cmp(&(b.1 - b.2).abs()))
        .expect("scores");
    eprintln!(
        "{label}: worst score delta {:.3e} at {name} ({native:.6} vs {oracle:.6})",
        (native - oracle).abs()
    );
    (native - oracle).abs()
}

#[test]
#[ignore = "requires METALLIX_QWEN_EMBEDDING_MODEL pointing to Qwen3-Embedding-0.6B@97b0c61 on Apple-Silicon Metal"]
fn float32_embeddings_match_the_source_oracle_and_the_model_card() {
    let Some(case) = case() else { return };
    let mut weights = Qwen3MlxWeights::load(&case.model).expect("load checkpoint");
    weights.prepare_float32().expect("float32 weights");

    let full = embed_all(&case, &weights, None);
    assert!(worst_vector_gap("f32", &full) <= 1.0 - F32_COSINE_MIN);
    let truncated = embed_all(&case, &weights, Some(256));
    assert!(worst_vector_gap("f32 256-dim", &truncated) <= 1.0 - F32_COSINE_MIN);

    let scores = scores(&case, &full);
    assert!(worst_score_delta("f32", &scores) <= F32_SCORE_VS_ORACLE);
    for (name, native, _, card) in &scores {
        if let Some(card) = card {
            assert!((native - card).abs() <= F32_SCORE_VS_CARD, "{name} vs card");
        }
    }
}

#[test]
#[ignore = "requires METALLIX_QWEN_EMBEDDING_MODEL pointing to Qwen3-Embedding-0.6B@97b0c61 on Apple-Silicon Metal"]
fn bf16_embeddings_stay_within_the_serving_tolerance() {
    let Some(case) = case() else { return };
    let weights = Qwen3MlxWeights::load(&case.model).expect("load checkpoint");

    let full = embed_all(&case, &weights, None);
    assert!(worst_vector_gap("bf16", &full) <= 1.0 - BF16_COSINE_MIN);
    let scores = scores(&case, &full);
    assert!(worst_score_delta("bf16", &scores) <= BF16_SCORE_VS_ORACLE);
}

/// Timing only: prints load time and, per mode and input length, the first
/// call at that length and the median and minimum of ten warm calls. Inputs are
/// token prefixes of the fixture's longest English passage ending in the
/// appended end-of-text, so lengths are exact. Asserts nothing about time.
#[test]
#[ignore = "timing; requires METALLIX_QWEN_EMBEDDING_MODEL pointing to Qwen3-Embedding-0.6B@97b0c61"]
fn embedding_latency_by_length_and_precision() {
    use std::time::{Duration, Instant};

    let Some(case) = case() else { return };
    let passage: Vec<i32> = case.fixture["inputs"]
        .as_array()
        .expect("inputs")
        .iter()
        .find(|input| input["name"] == "long_en_500")
        .expect("long_en_500")["input_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|id| i32::try_from(id.as_i64().expect("id")).expect("token ID"))
        .filter(|&id| id != QWEN3_EMBEDDING_EOS_ID)
        .collect();
    let input = |tokens: usize| -> Vec<i32> {
        let mut ids: Vec<i32> = passage.iter().copied().cycle().take(tokens - 1).collect();
        ids.push(QWEN3_EMBEDDING_EOS_ID);
        ids
    };
    let ms = |duration: Duration| duration.as_secs_f64() * 1e3;

    let started = Instant::now();
    let mut weights = Qwen3MlxWeights::load(&case.model).expect("load checkpoint");
    eprintln!("load (BF16): {:.1} ms", ms(started.elapsed()));
    for mode in ["bf16", "f32"] {
        if mode == "f32" {
            let started = Instant::now();
            weights.prepare_float32().expect("float32 weights");
            eprintln!("prepare_float32: {:.1} ms", ms(started.elapsed()));
        }
        for tokens in [8, 64, 256, 512] {
            let ids = input(tokens);
            let started = Instant::now();
            weights.embed(&ids, None).expect("embed");
            let first = started.elapsed();
            let mut warm: Vec<Duration> = (0..10)
                .map(|_| {
                    let started = Instant::now();
                    weights.embed(&ids, None).expect("embed");
                    started.elapsed()
                })
                .collect();
            warm.sort();
            eprintln!(
                "{mode} {tokens:>3} tokens: first {:>7.1} ms, warm median {:>6.1} ms, min {:>6.1} ms",
                ms(first),
                ms(warm[5]),
                ms(warm[0])
            );
        }
    }
}
