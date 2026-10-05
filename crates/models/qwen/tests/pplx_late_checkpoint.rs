//! Opt-in: pplx-embed-v1-late-0.6b on Metal against the CPU float32 source
//! oracle in `fixtures/pplx-embed-v1-late-0.6b/late-reference.json`, written
//! by `scripts/pplx_late_reference.py` from the card's `MultiVectorEncoder`.
//!
//! Tolerances, declared before the first native run (also in the fixture's
//! `tolerance_policy`); the checkpoint is float32, used as stored:
//! - encoder input IDs (queries with expansion) equal the oracle's, and so do
//!   the scored positions;
//! - per scored token, cosine(native, oracle) >= 1 - 1e-5;
//! - `MaxSim` scores within 1e-3 of the oracle's, and the card's three within
//!   1e-2 of its printed values.
#![cfg(feature = "metal")]

use std::{collections::HashMap, env, path::PathBuf, time::Instant};

use qwen::late::{
    LateEmbedding, LateTask, PPLX_LATE_EXPANSION_ID, PPLX_LATE_SKIPLIST_IDS, PplxLateEncoder,
    maxsim,
};
use serde_json::Value;
use tokenizers::Tokenizer;

const TOKEN_COSINE_MIN: f64 = 1.0 - 1e-5;
const MAXSIM_VS_ORACLE: f64 = 1e-3;
const MAXSIM_VS_CARD: f64 = 1e-2;

fn integers(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .expect("integers")
        .iter()
        .map(|v| v.as_i64().expect("integer"))
        .collect()
}

/// Rows of 128 little-endian f32 values from a hex string.
fn rows(value: &Value) -> Vec<Vec<f32>> {
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
        .collect::<Vec<_>>()
        .chunks_exact(128)
        .map(<[f32]>::to_vec)
        .collect()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum::<f64>()
    };
    dot(left, right) / (dot(left, left) * dot(right, right)).sqrt()
}

fn task(input: &Value) -> LateTask {
    match input["task"].as_str().expect("task") {
        "query" => LateTask::Query,
        _ => LateTask::Document,
    }
}

struct Setup {
    encoder: PplxLateEncoder,
    tokenizer: Tokenizer,
    fixture: Value,
}

fn setup() -> Option<Setup> {
    let Some(model) = env::var_os("METALLIX_PPLX_LATE_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_PPLX_LATE_MODEL is not set");
        return None;
    };
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/pplx-embed-v1-late-0.6b/late-reference.json"
    ))
    .expect("fixture JSON");
    assert_eq!(
        fixture["revision"],
        "4d28cf627d225552cfc29fb7df6cb0705ea0f1b3"
    );
    let mut tokenizer = Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    // tokenizer.json truncates to 511 and pads batches; the model's own lengths
    // (32 for queries, 512 for documents) are applied by `late_input_ids`.
    tokenizer.with_truncation(None).expect("disable truncation");
    tokenizer.with_padding(None);
    let encoder = PplxLateEncoder::load(&model).expect("load checkpoint");
    Some(Setup {
        encoder,
        tokenizer,
        fixture,
    })
}

fn encode(setup: &Setup, task: LateTask, text: &str) -> LateEmbedding {
    let ids: Vec<i32> = setup
        .tokenizer
        .encode(task.prompt(text), true)
        .expect("encode")
        .get_ids()
        .iter()
        .map(|&id| i32::try_from(id).expect("token ID"))
        .collect();
    setup.encoder.encode(task, &ids).expect("late encode")
}

#[test]
#[ignore = "requires METALLIX_PPLX_LATE_MODEL pointing to pplx-embed-v1-late-0.6b@4d28cf6 on Apple-Silicon Metal"]
fn late_embeddings_match_the_source_oracle_and_the_model_card() {
    let Some(setup) = setup() else { return };
    let reference = &setup.fixture["reference"];
    assert_eq!(
        reference["mask_token_id"].as_i64(),
        Some(i64::from(PPLX_LATE_EXPANSION_ID))
    );
    assert_eq!(
        integers(&reference["skiplist_token_ids"]),
        PPLX_LATE_SKIPLIST_IDS.map(i64::from)
    );

    let mut native = HashMap::new();
    let mut worst = 0.0_f64;
    for input in setup.fixture["inputs"].as_array().expect("inputs") {
        let name = input["name"].as_str().expect("name");
        let embedding = encode(&setup, task(input), input["text"].as_str().expect("text"));
        assert_eq!(
            embedding
                .input_ids
                .iter()
                .map(|&id| i64::from(id))
                .collect::<Vec<_>>(),
            integers(&input["input_ids"]),
            "{name}: input IDs"
        );
        assert_eq!(
            embedding
                .positions
                .iter()
                .map(|&p| i64::try_from(p).unwrap())
                .collect::<Vec<_>>(),
            integers(&input["scored_positions"]),
            "{name}: scored positions"
        );
        let oracle = rows(&input["embeddings_f32le"]);
        assert_eq!(embedding.vectors.len(), oracle.len(), "{name}: token count");
        for (position, (left, right)) in embedding.vectors.iter().zip(&oracle).enumerate() {
            let gap = 1.0 - cosine(left, right);
            worst = worst.max(gap);
            assert!(
                gap <= 1.0 - TOKEN_COSINE_MIN,
                "{name}[{position}]: 1 - cosine {gap}"
            );
        }
        native.insert(name.to_owned(), embedding.vectors);
    }

    let mut worst_score = 0.0_f64;
    for pair in setup.fixture["pair_scores"].as_array().expect("pairs") {
        let (query, document) = (
            pair["query"].as_str().unwrap(),
            pair["document"].as_str().unwrap(),
        );
        let score = maxsim(&native[query], &native[document]);
        let delta = (score - pair["score"].as_f64().unwrap()).abs();
        worst_score = worst_score.max(delta);
        assert!(delta <= MAXSIM_VS_ORACLE, "{query}~{document}: {delta}");
    }
    for (index, card) in setup.fixture["card_scores"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let score = maxsim(
            &native["card_query"],
            &native[&format!("card_document_{index}")],
        );
        let delta = (score - card.as_f64().unwrap()).abs();
        assert!(delta <= MAXSIM_VS_CARD, "card {index}: {score} vs {card}");
        eprintln!("card {index}: native {score:.4}, card {card}");
    }
    eprintln!("worst token 1 - cosine {worst:.3e}; worst MaxSim delta {worst_score:.3e}");
}

/// Timing only: query encoding, document encoding at several lengths (prefixes
/// of the fixture's long document), and `MaxSim` scoring. Asserts nothing about
/// time.
#[test]
#[ignore = "timing; requires METALLIX_PPLX_LATE_MODEL pointing to pplx-embed-v1-late-0.6b@4d28cf6"]
fn late_latency_by_length() {
    let Some(setup) = setup() else { return };
    let inputs = setup.fixture["inputs"].as_array().expect("inputs");
    let text = |name: &str| {
        inputs
            .iter()
            .find(|input| input["name"] == name)
            .expect("input")["text"]
            .as_str()
            .expect("text")
            .to_owned()
    };
    let long_ids: Vec<i32> = integers(
        &inputs
            .iter()
            .find(|input| input["name"] == "long_document")
            .expect("long document")["input_ids"],
    )
    .into_iter()
    .map(|id| i32::try_from(id).unwrap())
    .collect();
    let time = |label: &str, run: &dyn Fn()| {
        let started = Instant::now();
        run();
        let first = started.elapsed();
        let mut warm: Vec<_> = (0..10)
            .map(|_| {
                let started = Instant::now();
                run();
                started.elapsed()
            })
            .collect();
        warm.sort();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        eprintln!(
            "{label}: first {:>7.1} ms, warm median {:>6.1} ms, min {:>6.1} ms",
            ms(first),
            ms(warm[5]),
            ms(warm[0])
        );
    };
    let query = text("card_query");
    time("query (32 ids)", &|| {
        encode(&setup, LateTask::Query, &query);
    });
    for tokens in [64, 256, 512] {
        let ids = &long_ids[..tokens];
        time(&format!("document ({tokens} ids)"), &|| {
            setup
                .encoder
                .encode(LateTask::Document, ids)
                .expect("encode");
        });
    }
    let q = encode(&setup, LateTask::Query, &query).vectors;
    let d = setup
        .encoder
        .encode(LateTask::Document, &long_ids)
        .expect("encode")
        .vectors;
    time(&format!("MaxSim (32 x {} vectors)", d.len()), &|| {
        std::hint::black_box(maxsim(&q, &d));
    });
}
