use std::{
    fs,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::*;
use crate::{V41ArtifactInspection, V41ArtifactInspectionError};

static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Artifact(PathBuf);

impl Drop for Artifact {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("small-artifact.json")).unwrap()
}

// Serialize independent fixture metadata into real, small local shards. The
// production inspector reads their headers and file lengths, never the payload.
fn artifact(fixture: &Value, missing_index: Option<&str>) -> Artifact {
    let root = std::env::temp_dir().join(format!(
        "metallix-native-admission-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&root).unwrap();
    let artifact = Artifact(root);
    fs::write(
        artifact.0.join("config.json"),
        fixture["config"].to_string(),
    )
    .unwrap();
    fs::write(artifact.0.join("tokenizer.json"), "{}").unwrap();
    fs::write(
        artifact.0.join("tokenizer_config.json"),
        r#"{"chat_template":"{{ messages }}"}"#,
    )
    .unwrap();
    let mut header = serde_json::Map::new();
    let mut assignments = serde_json::Map::new();
    let mut offset = 0_u64;
    for (name, tensor) in fixture["tensors"].as_object().unwrap() {
        let width = match tensor["dtype"].as_str().unwrap() {
            "F32" => 4,
            "BF16" => 2,
            _ => 1,
        };
        let bytes = tensor["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_u64().unwrap())
            .product::<u64>()
            * width;
        let mut record = tensor.clone();
        record["data_offsets"] = json!([offset, offset + bytes]);
        header.insert(name.clone(), record);
        if missing_index != Some(name) {
            assignments.insert(name.clone(), json!("tiny.safetensors"));
        }
        offset += bytes;
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut shard = fs::File::create(artifact.0.join("tiny.safetensors")).unwrap();
    shard
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    shard.write_all(&header).unwrap();
    shard.set_len(8 + header.len() as u64 + offset).unwrap();
    fs::write(
        artifact.0.join("model.safetensors.index.json"),
        json!({
            "metadata":{"total_size":offset}, "weight_map":assignments
        })
        .to_string(),
    )
    .unwrap();
    artifact
}

fn rejected(mutated: &Value, expected: &str) {
    let root = artifact(mutated, None);
    let error = V41ArtifactInspection::inspect(&root.0).unwrap_err();
    assert!(
        matches!(error, V41ArtifactInspectionError::NativeTensorAdmission(_)),
        "{error}"
    );
    assert!(error.to_string().contains(expected), "{error}");
}

#[test]
fn native_inspector_accepts_independent_text_draft_and_vision_fixture() {
    let input = fixture();
    let root = artifact(&input, None);
    let report = V41ArtifactInspection::inspect(&root.0).unwrap();
    assert_eq!(report.tensor_count(), 239);
    assert_eq!(report.shard_count(), 1);
    assert_eq!(report.index_kind(), crate::V41ArtifactIndexKind::V41);
    let table = V41TensorTable::from_config(&input["config"].to_string()).unwrap();
    assert_eq!(table.routed_expert_bytes, 1632);
    assert_eq!(
        table.role("vision.test.weight"),
        Some(V41TensorRole::Vision)
    );
    assert_eq!(table.role("mtp.1.norm.weight"), Some(V41TensorRole::Draft));
}

#[test]
fn rejects_wrong_expert_shape_and_scale_dtype() {
    for (name, field, value) in [
        ("layers.0.ffn.experts.0.w2.weight", "shape", json!([32, 15])),
        ("layers.0.ffn.experts.0.w1.scale", "dtype", json!("F8_E4M3")),
    ] {
        let mut input = fixture();
        input["tensors"][name][field] = value;
        rejected(&input, name);
    }
}

#[test]
fn rejects_expert_absent_from_index_even_when_header_has_it() {
    let root = artifact(&fixture(), Some("layers.0.ffn.experts.1.w2.weight"));
    assert!(matches!(
        V41ArtifactInspection::inspect(&root.0),
        Err(V41ArtifactInspectionError::InvalidShardHeader(_))
    ));
}

#[test]
fn rejects_gate_on_ratio_one_source_and_swapped_engram_rows() {
    let mut input = fixture();
    input["tensors"]["layers.2.attn.compressor.wgate.weight"] =
        json!({"dtype":"BF16","shape":[32,32]});
    rejected(&input, "layers.2.attn.compressor.wgate.weight");
    let mut input = fixture();
    input["config"]["text_config"]["engram_num_embeddings"] = json!([11, 7]);
    rejected(&input, "layers.0.engram.embed");
}

#[test]
fn rejects_missing_and_wrong_draft_tensors() {
    let mut input = fixture();
    input["tensors"]
        .as_object_mut()
        .unwrap()
        .remove("mtp.1.confidence_head.proj.weight");
    rejected(&input, "mtp.1.confidence_head.proj.weight");
    let mut input = fixture();
    input["tensors"]["mtp.0.main_proj.weight"]["shape"] = json!([32, 32]);
    rejected(&input, "mtp.0.main_proj.weight");
}

#[test]
fn rejects_unclassified_names_and_wrong_index_total() {
    let mut input = fixture();
    input["tensors"]["unknown.weight"] = json!({"dtype":"BF16","shape":[1]});
    rejected(&input, "unknown.weight");
    let root = artifact(&fixture(), None);
    let path = root.0.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    index["metadata"]["total_size"] = json!(1);
    fs::write(path, index.to_string()).unwrap();
    let error = V41ArtifactInspection::inspect(&root.0).unwrap_err();
    assert!(matches!(
        error,
        V41ArtifactInspectionError::NativeTensorAdmission(_)
    ));
    assert!(error.to_string().contains("index declares 1"), "{error}");
}

#[test]
fn bounds_config_expansion_and_tensor_byte_arithmetic() {
    let mut input = fixture();
    input["config"]["text_config"]["n_routed_experts"] = json!(65535);
    assert!(matches!(
        V41TensorTable::from_config(&input["config"].to_string()),
        Err(V41TensorTableError::Capacity)
    ));
    let mut input = fixture();
    input["config"]["text_config"]["engram_num_embeddings"] = json!([u64::MAX, 11]);
    assert!(matches!(
        V41TensorTable::from_config(&input["config"].to_string()),
        Err(V41TensorTableError::Capacity)
    ));
}

#[test]
#[ignore = "set METALLIX_V41_ADMISSION_METADATA to metadata-only config/index/headers receipt directory"]
fn admits_saved_native_headers_without_weight_payloads() {
    let Some(root) = std::env::var_os("METALLIX_V41_ADMISSION_METADATA") else {
        eprintln!("skipping: METALLIX_V41_ADMISSION_METADATA is not set");
        return;
    };
    let root = PathBuf::from(root);
    let config = fs::read_to_string(root.join("config.json")).unwrap();
    let index_bytes = fs::read(root.join("model.safetensors.index.json")).unwrap();
    let index = V41SafetensorsIndex::parse(std::str::from_utf8(&index_bytes).unwrap()).unwrap();
    let receipts: Value =
        serde_json::from_slice(&fs::read(root.join("headers.json")).unwrap()).unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(&index_bytes)),
        receipts["index_sha256"].as_str().unwrap()
    );
    assert_eq!(
        receipts["index_sha256"].as_str().unwrap(),
        "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
    );
    let mut headers = Vec::new();
    for shard in index.shard_paths() {
        let bytes = fs::read(root.join("headers").join(format!("{shard}.header.bin"))).unwrap();
        let file_bytes = receipts["shards"][shard]["file_bytes"].as_u64().unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            receipts["shards"][shard]["header_sha256"].as_str().unwrap()
        );
        let header = if bytes.first() == Some(&b'{') {
            V41SafetensorsHeader::parse(&bytes, file_bytes)
        } else {
            V41SafetensorsHeader::parse_prefixed_header(&bytes, file_bytes)
        }
        .unwrap();
        headers.push((shard.as_str(), header));
    }
    let table = V41TensorTable::from_config(&config).unwrap();
    let admitted = table
        .admit(&index, headers.iter().map(|(name, header)| (*name, header)))
        .unwrap();
    assert_eq!(index.tensor_count(), 96_085);
    assert_eq!(index.shard_paths().len(), 48);
    assert_eq!(admitted.total_bytes, 510_286_023_000);
    assert_eq!(table.routed_expert_bytes, 18_800_640);
    assert!(admitted.draft_tensors > 0);
    assert!(admitted.vision_tensors > 0);
}
