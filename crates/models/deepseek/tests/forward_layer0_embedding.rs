//! Source-grounded layer-zero token embedding boundary.
//!
//! This keeps the first transformer input explicit while the production model
//! graph is still being built. The fixture is a bounded synthetic source run;
//! it does not claim checkpoint or full-model parity.

use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE_SHA256: &str = "2e1638167e83b9165f7adb84e70cbaadf297d3e6d3654c75232f757ed3d1f3d6";
const WIDTH: usize = 128;

fn fixture() -> Value {
    let raw = include_str!("../../../../fixtures/deepseek-v41/layer0-embedding-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    serde_json::from_str(raw).expect("layer-zero embedding fixture JSON")
}

fn bytes(value: &Value, field: &str) -> Vec<u8> {
    let hex = value[field]
        .as_str()
        .unwrap_or_else(|| panic!("{field} storage_hex must be a string"));
    assert_eq!(hex.len() % 2, 0, "{field} storage hex has odd length");
    (0..hex.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&hex[index..index + 2], 16).expect("hex byte"))
        .collect()
}

fn u64_storage(value: &Value, field: &str) -> Vec<u64> {
    let raw = bytes(value, field);
    assert_eq!(raw.len() % 8, 0, "{field} integer storage alignment");
    raw.chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bf16_storage(value: &Value, field: &str) -> Vec<u16> {
    let raw = bytes(value, field);
    assert_eq!(raw.len() % 2, 0, "{field} BF16 storage alignment");
    raw.chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

fn source_lookup(root: &Value, ids: &[u64]) -> Result<Vec<u16>, &'static str> {
    let parameter = &root["embedding_parameter"];
    let shape = parameter["shape"]
        .as_array()
        .ok_or("missing parameter shape")?;
    if shape.len() != 2 || shape[1].as_u64() != Some(WIDTH as u64) {
        return Err("embedding parameter shape");
    }
    let rows = usize::try_from(shape[0].as_u64().ok_or("embedding row count")?)
        .map_err(|_| "embedding row count overflow")?;
    let table = bf16_storage(parameter, "storage_hex");
    if table.len() != rows * WIDTH {
        return Err("embedding parameter storage length");
    }
    let mut output = Vec::with_capacity(ids.len() * WIDTH);
    for &id in ids {
        let row = usize::try_from(id).map_err(|_| "token id conversion")?;
        if row >= rows {
            return Err("token id outside embedding table");
        }
        output.extend_from_slice(&table[row * WIDTH..(row + 1) * WIDTH]);
    }
    Ok(output)
}

#[test]
fn source_token_embedding_matches_each_layer_zero_stream() {
    let root = fixture();
    assert_eq!(root["schema_version"].as_u64(), Some(1));
    assert_eq!(root["cases"].as_array().unwrap().len(), 3);
    for case in root["cases"].as_array().unwrap() {
        let ids = u64_storage(&case["input_ids"], "storage_hex");
        let shape = case["embedding"]["shape"].as_array().unwrap();
        assert_eq!(shape[0].as_u64(), Some(1));
        assert_eq!(shape[1].as_u64(), Some(ids.len() as u64));
        assert_eq!(shape[2].as_u64(), Some(WIDTH as u64));
        assert_eq!(
            source_lookup(&root, &ids).unwrap(),
            bf16_storage(&case["embedding"], "storage_hex")
        );
    }
}

#[test]
fn layer_zero_rejects_unknown_token_and_changed_token_changes_row() {
    let root = fixture();
    let first = &root["cases"][0];
    let ids = u64_storage(&first["input_ids"], "storage_hex");
    let mut changed = ids.clone();
    changed[0] = (changed[0] + 1) % 8;
    assert_ne!(source_lookup(&root, &ids), source_lookup(&root, &changed));
    assert_eq!(
        source_lookup(&root, &[8]),
        Err("token id outside embedding table")
    );
}
