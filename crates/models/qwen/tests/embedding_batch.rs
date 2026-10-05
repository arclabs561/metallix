//! Opt-in: batched Qwen3-Embedding-0.6B on Metal against the CPU float32 source
//! oracle in `fixtures/qwen3-embedding-0.6b/embedding-reference.json`.
//!
//! One right-padded forward pass per batch must keep every vector within the
//! fixture's float32 policy, cosine(native, oracle) >= 1 - 1e-5, and agree with
//! embedding each input alone. It also reports the latency of a 16-input batch
//! against 16 single passes.
#![cfg(feature = "metal")]

use std::{env, path::PathBuf, time::Instant};

use qwen::{
    embedding::{EMBEDDING_BATCH_PADDING_RATIO, padding_groups},
    metal::Qwen3MlxWeights,
};
use serde_json::Value;
use tokenizers::Tokenizer;

const COSINE_MIN: f64 = 1.0 - 1e-5;
const BATCH: usize = 16;

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    let dot = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum::<f64>()
    };
    dot(left, right) / (dot(left, left) * dot(right, right)).sqrt()
}

fn f32le_hex(value: &Value) -> Vec<f32> {
    let hex = value.as_str().expect("hex embedding").as_bytes();
    hex.chunks_exact(8)
        .map(|word| {
            let bytes: Vec<u8> = word
                .chunks_exact(2)
                .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
                .collect();
            f32::from_le_bytes(bytes.try_into().unwrap())
        })
        .collect()
}

#[test]
#[ignore = "requires METALLIX_QWEN_EMBEDDING_MODEL pointing to Qwen3-Embedding-0.6B@97b0c61 on Apple-Silicon Metal"]
fn batched_embeddings_match_the_oracle_and_single_passes() {
    let model = PathBuf::from(env::var_os("METALLIX_QWEN_EMBEDDING_MODEL").expect("model path"));
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/qwen3-embedding-0.6b/embedding-reference.json"
    ))
    .expect("fixture JSON");
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json")).expect("tokenizer");
    let mut weights = Qwen3MlxWeights::load(&model).expect("load checkpoint");
    weights.prepare_float32().expect("float32 weights");

    let inputs = fixture["inputs"].as_array().expect("inputs");
    let ids: Vec<Vec<i32>> = inputs
        .iter()
        .map(|input| {
            tokenizer
                .encode(input["text"].as_str().expect("text"), true)
                .expect("encode")
                .get_ids()
                .iter()
                .map(|&id| i32::try_from(id).expect("token ID"))
                .collect()
        })
        .collect();

    let mut worst_oracle: f64 = 1.0;
    let mut worst_single: f64 = 1.0;
    for (chunk_index, chunk) in ids.chunks(BATCH).enumerate() {
        let sequences: Vec<&[i32]> = chunk.iter().map(Vec::as_slice).collect();
        let batched = weights
            .embed_batch(&sequences, None)
            .expect("batched embed");
        for (offset, (sequence, vector)) in sequences.iter().zip(&batched).enumerate() {
            let input = &inputs[chunk_index * BATCH + offset];
            let name = input["name"].as_str().unwrap();
            let oracle = f32le_hex(&input["embedding_f32le"]);
            let single = weights.embed(sequence, None).expect("single embed");
            let to_oracle = cosine(vector, &oracle);
            let to_single = cosine(vector, &single);
            worst_oracle = worst_oracle.min(to_oracle);
            worst_single = worst_single.min(to_single);
            assert!(
                to_oracle >= COSINE_MIN,
                "{name}: cosine to oracle {to_oracle}"
            );
            assert!(
                to_single >= COSINE_MIN,
                "{name}: cosine to single pass {to_single}"
            );
        }
    }
    eprintln!("worst cosine: to oracle {worst_oracle}, to single pass {worst_single}");

    // Latency: alternate batched and single runs on three 16-input mixes.
    let longest_input = ids.iter().max_by_key(|ids| ids.len()).unwrap();
    for (label, sequences) in [
        (
            "fixture inputs 0-15",
            ids[..BATCH].iter().map(Vec::as_slice).collect::<Vec<_>>(),
        ),
        ("16 copies of input 0", vec![ids[0].as_slice(); BATCH]),
        (
            "16 copies of the longest input",
            vec![longest_input.as_slice(); BATCH],
        ),
    ] {
        let lengths: Vec<usize> = sequences.iter().map(|ids| ids.len()).collect();
        let tokens: usize = lengths.iter().sum();
        let groups = padding_groups(&lengths, EMBEDDING_BATCH_PADDING_RATIO);
        let padded: usize = groups
            .iter()
            .map(|group| group.len() * group.iter().map(|&i| lengths[i]).max().unwrap())
            .sum();
        let mut batched_ms = Vec::new();
        let mut single_ms = Vec::new();
        for _ in 0..5 {
            let started = Instant::now();
            weights
                .embed_batch(&sequences, None)
                .expect("batched embed");
            batched_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            let started = Instant::now();
            for sequence in &sequences {
                weights.embed(sequence, None).expect("single embed");
            }
            single_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        batched_ms.sort_by(f64::total_cmp);
        single_ms.sort_by(f64::total_cmp);
        eprintln!(
            "{label} ({tokens} tokens, {} passes padded to {padded}): batched median {:.1} ms, singles median {:.1} ms",
            groups.len(),
            batched_ms[2],
            single_ms[2]
        );
    }
}
