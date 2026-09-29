//! Test-owned decoding of source operands into the runtime Engram session.
use deepseek::{
    engram::{
        EngramHashLayout,
        embedding::{EngramEmbeddingLayout, engram_embedding_bf16_reference},
    },
    reduced::{EngramSession, EngramSessionConfig, EngramSessionWeights},
};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn count(value: &Value) -> usize {
    usize::try_from(value.as_u64().expect("fixture dimension")).unwrap()
}

fn bytes(value: &Value, dtype: &str) -> Vec<u8> {
    assert_eq!(value["dtype"].as_str(), Some(dtype));
    let elements: usize = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(count)
        .product();
    assert_eq!(count(&value["numel"]), elements);
    let bytes_per_element = match dtype {
        "torch.int64" => 8,
        "torch.bfloat16" => 2,
        "torch.float8_e4m3fn" | "torch.float8_e8m0fnu" => 1,
        _ => panic!("unsupported fixture dtype"),
    };
    let encoded = value["storage_hex"].as_str().unwrap();
    assert_eq!(encoded.len(), elements * bytes_per_element * 2);
    let raw: Vec<_> = encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|word| u8::from_str_radix(std::str::from_utf8(word).unwrap(), 16).unwrap())
        .collect();
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        value["storage_sha256"].as_str().unwrap()
    );
    raw
}

fn i64s(value: &Value) -> Vec<i64> {
    bytes(value, "torch.int64")
        .chunks_exact(8)
        .map(|word| i64::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn bf16(value: &Value) -> Vec<u16> {
    bytes(value, "torch.bfloat16")
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

/// Decodes immutable source operands so callers can construct fresh request-local sessions.
pub(crate) fn definition(root: &Value, layer: u64) -> (EngramSessionConfig, EngramSessionWeights) {
    let model = &root["model"];
    let state = &root["engram"]["hash_state"];
    let layout = &root["engram"]["layout"];
    let layers = layout["layer_ids"].as_array().unwrap();
    let hash_layer = layers
        .iter()
        .position(|id| id.as_u64() == Some(layer))
        .unwrap();
    let hash_layout = EngramHashLayout::new(
        count(&layout["max_ngram_size"]),
        count(&layout["n_heads"]),
        layers.len(),
        state["pad_id"].as_i64().unwrap(),
        i64s(&state["primes"]),
        i64s(&state["offsets"]),
        i64s(&state["multipliers"]),
    )
    .unwrap();
    let capacity = root["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| count(&case["start_pos"]) + count(&case["input_ids"]["shape"][1]))
        .max()
        .unwrap();
    let config = EngramSessionConfig::new(
        hash_layout,
        i64s(&state["token_map"]),
        hash_layer,
        capacity,
        count(&model["copies"]),
        count(&model["dim"]),
        count(&layout["num_embeddings"][hash_layer]),
        count(&model["embedding_dim"]),
        serde_json::from_value(model["norm_eps"].clone()).unwrap(),
        serde_json::from_value(model["gate_clamp"].clone()).unwrap(),
    )
    .unwrap();
    let parameters = &root["encoded_parameters"];
    let tensor = |name: &str| &parameters[format!("layers.{layer}.engram.{name}")];
    let weights = EngramSessionWeights::new(
        bytes(tensor("embed.weight"), "torch.float8_e4m3fn"),
        bytes(tensor("embed.scale"), "torch.float8_e8m0fnu"),
        bytes(tensor("wkv.weight"), "torch.float8_e4m3fn"),
        bytes(tensor("wkv.scale"), "torch.float8_e8m0fnu"),
        bf16(tensor("q_weight")),
        bf16(tensor("k_weight")),
    );
    (config, weights)
}

pub(crate) fn session(root: &Value, layer: u64) -> EngramSession {
    let (config, weights) = definition(root, layer);
    EngramSession::new(config, weights).unwrap()
}

pub(crate) fn assert_output(case: &Value, output: &deepseek::reduced::EngramStepOutput) {
    assert_eq!(output.hash_ids(), i64s(&case["captured_hash_ids"]));
    for (actual, expected) in [
        (output.embedding(), "embedding"),
        (output.wkv(), "wkv_output"),
        (output.key(), "key"),
        (output.value(), "value"),
        (output.output(), "output"),
        (output.output(), "block_entry"),
    ] {
        assert_eq!(actual, bf16(&case[expected]), "runtime Engram {expected}");
    }
}

// Retain the source integration's masked-hash negative control outside runtime.
pub(crate) fn assert_masked_embedding(
    root: &Value,
    layer: u64,
    output: &deepseek::reduced::EngramStepOutput,
) {
    let layout = &root["engram"]["layout"];
    let selected = layout["layer_ids"]
        .as_array()
        .unwrap()
        .iter()
        .position(|id| id.as_u64() == Some(layer))
        .unwrap();
    let parameters = &root["encoded_parameters"];
    let codes = bytes(
        &parameters[format!("layers.{layer}.engram.embed.weight")],
        "torch.float8_e4m3fn",
    );
    let scales = bytes(
        &parameters[format!("layers.{layer}.engram.embed.scale")],
        "torch.float8_e8m0fnu",
    );
    let embedding = EngramEmbeddingLayout::new(
        count(&layout["num_embeddings"][selected]),
        count(&root["model"]["embedding_dim"]),
        32,
    )
    .unwrap();
    let mut wrong_ids = output.hash_ids().to_vec();
    wrong_ids[0] = -1;
    let mut changed = vec![0; output.embedding().len()];
    engram_embedding_bf16_reference(&wrong_ids, &codes, &scales, embedding, &mut changed).unwrap();
    assert_ne!(
        changed,
        output.embedding(),
        "masked hash must change native embedding"
    );
}
