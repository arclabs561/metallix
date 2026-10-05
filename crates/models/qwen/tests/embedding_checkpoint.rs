//! Opt-in: Qwen3-Embedding-0.6B on Metal against the CPU float32 source oracle
//! in `fixtures/qwen3-embedding-0.6b/embedding-reference.json`, written by
//! `scripts/qwen3_embedding_reference.py` from the model card's recipe.
//!
//! Tolerances, declared before the first run (they are also recorded in the
//! fixture's `tolerance_policy`):
//! - token IDs from `tokenizer.json` with special tokens equal the oracle's;
//! - per input, cosine(native, oracle) >= 1 - 1e-5, with weights promoted to
//!   float32, and likewise for a 256-dimension truncation;
//! - the 2 x 2 query-document scores within 1e-4 of the oracle's and within
//!   1e-3 of the card's published scores.
#![cfg(feature = "metal")]

use std::{env, path::PathBuf};

use qwen::{embedding::QWEN3_EMBEDDING_EOS_ID, metal::Qwen3MlxWeights};
use serde_json::Value;
use tokenizers::Tokenizer;

const COSINE_MIN: f64 = 1.0 - 1e-5;
const SCORE_VS_ORACLE: f64 = 1e-4;
const SCORE_VS_CARD: f64 = 1e-3;

fn dot(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(&a, &b)| f64::from(a) * f64::from(b))
        .sum()
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    dot(left, right) / (dot(left, left) * dot(right, right)).sqrt()
}

fn floats(value: &Value) -> Vec<f32> {
    #[allow(clippy::cast_possible_truncation, reason = "fixture values are f32")]
    value
        .as_array()
        .expect("float list")
        .iter()
        .map(|v| v.as_f64().expect("float") as f32)
        .collect()
}

#[test]
#[ignore = "requires METALLIX_QWEN_EMBEDDING_MODEL pointing to Qwen3-Embedding-0.6B@97b0c61 on Apple-Silicon Metal"]
fn embeddings_match_the_source_oracle_and_the_model_card() {
    let Some(model) = env::var_os("METALLIX_QWEN_EMBEDDING_MODEL").map(PathBuf::from) else {
        eprintln!("skipping: METALLIX_QWEN_EMBEDDING_MODEL is not set");
        return;
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
    let mut weights = Qwen3MlxWeights::load(&model).expect("load checkpoint");
    weights.prepare_float32().expect("float32 weights");

    let mut native = std::collections::HashMap::new();
    for input in fixture["inputs"].as_array().expect("inputs") {
        let name = input["name"].as_str().expect("name");
        let ids: Vec<i32> = tokenizer
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

        let oracle = floats(&input["embedding"]);
        let embedding = weights.embed(&ids, None).expect("embed");
        let full = cosine(&embedding, &oracle);
        let truncated = cosine(
            &weights.embed(&ids, Some(256)).expect("truncated embed"),
            &oracle[..256],
        );
        eprintln!("{name}: cosine {full:.9}, 256-dim cosine {truncated:.9}");
        assert!(full >= COSINE_MIN, "{name}: cosine {full}");
        assert!(
            truncated >= COSINE_MIN,
            "{name}: 256-dim cosine {truncated}"
        );
        native.insert(name.to_owned(), embedding);
    }

    for (query, row) in ["card_query_0", "card_query_1"].into_iter().enumerate() {
        for (document, column) in ["card_document_0", "card_document_1"]
            .into_iter()
            .enumerate()
        {
            let score = dot(&native[row], &native[column]);
            let oracle = fixture["scores"][query][document].as_f64().expect("score");
            let card = fixture["card_scores"][query][document]
                .as_f64()
                .expect("card score");
            eprintln!(
                "score[{query}][{document}]: native {score:.7}, oracle {oracle:.7}, card {card:.7}"
            );
            assert!((score - oracle).abs() <= SCORE_VS_ORACLE, "vs oracle");
            assert!((score - card).abs() <= SCORE_VS_CARD, "vs card");
        }
    }
}
