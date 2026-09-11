use std::{
    fs,
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use serde_json::json;

use super::{
    Qwen3CheckpointError, Qwen3CheckpointInspection, RawCheckpointLayout, read_header,
    required_dense_tensors,
};
use crate::Qwen3TextContract;

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "metallix-qwen-checkpoint-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("create fixture directory");
        Self { path }
    }

    fn write_config(&self) {
        fs::write(
                self.path.join("config.json"),
                r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"max_position_embeddings":16,"vocab_size":8,"intermediate_size":6,"tie_word_embeddings":true}"#,
            )
            .expect("write config");
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.path).expect("remove fixture directory");
    }
}

fn expected_tensors() -> Vec<super::ExpectedTensor> {
    let contract = Qwen3TextContract::parse(
            r#"{"model_type":"qwen3","num_hidden_layers":1,"hidden_size":4,"vocab_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,"max_position_embeddings":16}"#,
        )
        .expect("valid contract");
    required_dense_tensors(
        &contract,
        &RawCheckpointLayout {
            vocab_size: 8,
            intermediate_size: 6,
            tie_word_embeddings: true,
        },
    )
    .expect("valid layout")
}

fn write_safetensors(path: &Path, tensors: &[super::ExpectedTensor]) {
    let payload_bytes = tensors
        .iter()
        .map(|tensor| tensor.shape.iter().product::<u64>() * 2)
        .sum::<u64>();
    write_safetensors_with_payload(
        path,
        tensors,
        &vec![0; usize::try_from(payload_bytes).expect("small payload")],
    );
}

fn write_safetensors_with_payload(path: &Path, tensors: &[super::ExpectedTensor], payload: &[u8]) {
    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), json!({"format":"pt"}));
    for tensor in tensors {
        let element_count = tensor.shape.iter().product::<u64>();
        let byte_length = element_count * 2;
        header.insert(
                tensor.name.clone(),
                json!({"dtype":"BF16","shape": tensor.shape,"data_offsets":[offset, offset + byte_length]}),
            );
        offset += byte_length;
    }
    let header = serde_json::to_vec(&header).expect("serialize header");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    assert_eq!(
        payload.len(),
        usize::try_from(offset).expect("small payload")
    );
    bytes.extend_from_slice(payload);
    fs::write(path, bytes).expect("write safetensors fixture");
}

fn write_safetensors_with_one_dtype(
    path: &Path,
    tensors: &[super::ExpectedTensor],
    overridden_name: &str,
    overridden_dtype: &str,
) {
    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), json!({"format":"pt"}));
    for tensor in tensors {
        let dtype = if tensor.name == overridden_name {
            overridden_dtype
        } else {
            "BF16"
        };
        let bytes_per_element = if dtype == "F32" { 4 } else { 2 };
        let byte_length = tensor.shape.iter().product::<u64>() * bytes_per_element;
        header.insert(
                tensor.name.clone(),
                json!({"dtype":dtype, "shape": tensor.shape, "data_offsets":[offset, offset + byte_length]}),
            );
        offset += byte_length;
    }
    let header = serde_json::to_vec(&header).expect("serialize header");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.resize(
        bytes.len() + usize::try_from(offset).expect("small payload"),
        0,
    );
    fs::write(path, bytes).expect("write typed safetensors fixture");
}

fn payload_range(tensors: &[super::ExpectedTensor], name: &str) -> std::ops::Range<usize> {
    let mut offset = 0_usize;
    for tensor in tensors {
        let len = usize::try_from(tensor.shape.iter().product::<u64>() * 2).expect("small payload");
        if tensor.name == name {
            return offset..offset + len;
        }
        offset += len;
    }
    panic!("fixture tensor must exist: {name}");
}

fn write_raw_shard(path: &Path, header: &serde_json::Value, payload_bytes: usize) {
    let header = serde_json::to_vec(header).expect("serialize header");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.resize(bytes.len() + payload_bytes, 0);
    fs::write(path, bytes).expect("write raw shard");
}

#[test]
fn inspects_qwen3_headers_without_reading_tensor_payloads() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    write_safetensors(&fixture.path.join("model-00001.safetensors"), &tensors);

    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");
    assert_eq!(inspection.contract().total_layers(), 1);
    assert_eq!(inspection.tensor_count(), tensors.len());
    assert!(inspection.tensor_bytes() > tensors.len() as u64);
    assert_eq!(inspection.shards().len(), 1);
    assert!(inspection.shards()[0].ends_with("model-00001.safetensors"));
}

#[test]
fn reads_only_one_validated_tensor_with_exact_bytes_and_header_facts() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    let total = tensors
        .iter()
        .map(|tensor| tensor.shape.iter().product::<u64>() * 2)
        .sum::<u64>();
    let payload = (0..usize::try_from(total).expect("small payload"))
        .map(|index| u8::try_from(index % 251).expect("bounded byte"))
        .collect::<Vec<_>>();
    let path = fixture.path.join("model.safetensors");
    write_safetensors_with_payload(&path, &tensors, &payload);

    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");
    let name = "model.layers.0.input_layernorm.weight";
    let tensor = inspection
        .read_tensor(name, 8)
        .expect("bounded tensor read");
    assert_eq!(tensor.dtype(), "BF16");
    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.bytes(), &payload[payload_range(&tensors, name)]);
}

#[test]
fn reads_middle_last_and_full_bf16_matrix_rows_at_their_exact_offsets() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    let total = tensors
        .iter()
        .map(|tensor| tensor.shape.iter().product::<u64>() * 2)
        .sum::<u64>();
    let payload = (0..usize::try_from(total).expect("small payload"))
        .map(|index| u8::try_from(index % 251).expect("bounded byte"))
        .collect::<Vec<_>>();
    let path = fixture.path.join("model.safetensors");
    write_safetensors_with_payload(&path, &tensors, &payload);
    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");
    let name = "model.embed_tokens.weight";
    let tensor_range = payload_range(&tensors, name);
    let matrix = &payload[tensor_range];

    for (rows, expected_shape, expected_bytes) in [
        (0..8, vec![8, 4], &matrix[0..64]),
        (2..5, vec![3, 4], &matrix[16..40]),
        (7..8, vec![1, 4], &matrix[56..64]),
    ] {
        let selected = inspection
            .read_bf16_rows(
                name,
                rows,
                u64::try_from(expected_bytes.len()).expect("small"),
            )
            .expect("selected rows must read exactly");
        assert_eq!(selected.dtype(), "BF16");
        assert_eq!(selected.shape(), expected_shape);
        assert_eq!(selected.bytes(), expected_bytes);
    }
}

#[test]
fn rows_reject_one_byte_under_budget_before_opening_the_shard() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    let path = fixture.path.join("model.safetensors");
    write_safetensors(&path, &tensors);
    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");
    fs::remove_file(path).expect("make an accidental open fail");

    let error = inspection
        .read_bf16_rows("model.embed_tokens.weight", 2..5, 23)
        .expect_err("three rows of four BF16 values require 24 bytes");
    assert!(matches!(
        error,
        Qwen3CheckpointError::TensorExceedsReadBudget {
            tensor_bytes: 24,
            max_bytes: 23,
            ..
        }
    ));
}

#[test]
fn rows_reject_unknown_rank_one_non_bf16_and_invalid_ranges() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    let path = fixture.path.join("model.safetensors");
    write_safetensors(&path, &tensors);
    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");

    let unknown = inspection
        .read_bf16_rows("not.a.validated.tensor", 0..1, 8)
        .expect_err("unknown tensor must not open a shard");
    assert!(matches!(unknown, Qwen3CheckpointError::UnknownTensor(_)));
    let rank = inspection
        .read_bf16_rows("model.layers.0.input_layernorm.weight", 0..1, 8)
        .expect_err("rank-one scale is not a matrix");
    assert!(matches!(
        rank,
        Qwen3CheckpointError::TensorRowsRequireMatrix { .. }
    ));
    for rows in [0..0, 8..9, std::ops::Range { start: 9, end: 8 }] {
        let range = inspection
            .read_bf16_rows("model.embed_tokens.weight", rows, 64)
            .expect_err("row range must be nonempty and in bounds");
        assert!(matches!(
            range,
            Qwen3CheckpointError::InvalidTensorRowRange { .. }
        ));
    }

    let typed_path = fixture.path.join("typed.safetensors");
    write_safetensors_with_one_dtype(&typed_path, &tensors, "model.embed_tokens.weight", "F32");
    fs::remove_file(path).expect("remove duplicate tensor names from fixture");
    let typed = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid F32 layout");
    let dtype = typed
        .read_bf16_rows("model.embed_tokens.weight", 0..1, 32)
        .expect_err("F32 rows are not BF16 rows");
    assert!(matches!(
        dtype,
        Qwen3CheckpointError::TensorRowsRequireBf16 { .. }
    ));
}

#[test]
fn rejects_a_selected_tensor_before_allocating_beyond_the_budget() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    write_safetensors(&fixture.path.join("model.safetensors"), &tensors);

    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");
    fs::remove_file(fixture.path.join("model.safetensors"))
        .expect("make a failed open observable if the budget is checked too late");
    let error = inspection
        .read_tensor("model.layers.0.input_layernorm.weight", 7)
        .expect_err("the 8-byte tensor must not allocate under a 7-byte budget");
    assert!(matches!(
        error,
        Qwen3CheckpointError::TensorExceedsReadBudget {
            tensor_bytes: 8,
            max_bytes: 7,
            ..
        }
    ));
}

#[test]
fn rejects_an_unknown_tensor_before_opening_any_shard() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    let path = fixture.path.join("model.safetensors");
    write_safetensors(&path, &tensors);
    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");
    fs::remove_file(path).expect("make an accidental open fail");

    let error = inspection
        .read_tensor("not.a.validated.tensor", 0)
        .expect_err("unknown name must be rejected from validated metadata alone");
    assert!(
        matches!(error, Qwen3CheckpointError::UnknownTensor(name) if name == "not.a.validated.tensor")
    );
}

#[test]
fn rejects_a_truncated_shard_after_its_header_was_validated() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    let path = fixture.path.join("model.safetensors");
    write_safetensors(&path, &tensors);
    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");

    fs::write(&path, [0_u8; 3]).expect("truncate fixture shard");
    let error = inspection
        .read_tensor("model.layers.0.input_layernorm.weight", 8)
        .expect_err("changed shard must not be read through a validated range");
    assert!(matches!(
        error,
        Qwen3CheckpointError::ShardMetadataDrift { .. }
    ));
}

#[test]
fn rejects_same_length_shard_mutation_when_the_timestamp_drifts() {
    let fixture = Fixture::new();
    fixture.write_config();
    let tensors = expected_tensors();
    let path = fixture.path.join("model.safetensors");
    write_safetensors(&path, &tensors);
    let inspection = Qwen3CheckpointInspection::inspect(&fixture.path).expect("valid checkpoint");
    let before = fs::metadata(&path)
        .expect("fixture metadata")
        .modified()
        .expect("fixture modification time");

    let mut shard = fs::File::options()
        .write(true)
        .open(&path)
        .expect("open fixture for same-length write");
    shard
        .seek(SeekFrom::End(-1))
        .expect("seek to the final payload byte");
    shard.write_all(&[0]).expect("overwrite one payload byte");
    let changed = before
        .checked_add(Duration::from_secs(2))
        .expect("fixture timestamp has headroom");
    shard
        .set_times(fs::FileTimes::new().set_modified(changed))
        .expect("set distinct fixture modification time");

    let error = inspection
        .read_tensor("model.layers.0.input_layernorm.weight", 8)
        .expect_err("same-size shard with a new timestamp must be rejected");
    assert!(matches!(
        error,
        Qwen3CheckpointError::ShardMetadataDrift { .. }
    ));
}

#[test]
fn rejects_missing_required_dense_tensor() {
    let fixture = Fixture::new();
    fixture.write_config();
    let mut tensors = expected_tensors();
    tensors.pop();
    write_safetensors(&fixture.path.join("model.safetensors"), &tensors);

    let error = Qwen3CheckpointInspection::inspect(&fixture.path)
        .expect_err("incomplete decoder layout must be rejected");
    assert!(
        matches!(error, Qwen3CheckpointError::MissingRequiredTensor(name) if name == "model.norm.weight")
    );
}

#[test]
fn rejects_a_tensor_range_outside_the_payload() {
    let fixture = Fixture::new();
    fixture.write_config();
    let header = serde_json::to_vec(&json!({
        "model.embed_tokens.weight": {"dtype":"BF16", "shape":[8, 4], "data_offsets": [0, 999]}
    }))
    .expect("serialize header");
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    fs::write(fixture.path.join("model.safetensors"), bytes).expect("write malformed shard");

    let error = Qwen3CheckpointInspection::inspect(&fixture.path)
        .expect_err("out-of-range tensor must be rejected");
    assert!(matches!(
        error,
        Qwen3CheckpointError::TensorOutsidePayload { .. }
    ));
}

#[test]
fn rejects_byte_ranges_that_do_not_match_bf16_shape() {
    let fixture = Fixture::new();
    let path = fixture.path.join("bad.safetensors");
    write_raw_shard(
        &path,
        &json!({"weight": {"dtype":"BF16", "shape":[2], "data_offsets":[0, 2]}}),
        2,
    );

    let error = read_header(&path).expect_err("two BF16 values need four bytes");
    assert!(matches!(
        error,
        Qwen3CheckpointError::TensorByteLengthMismatch { .. }
    ));
}

#[test]
fn rejects_overlapping_tensor_ranges() {
    let fixture = Fixture::new();
    let path = fixture.path.join("overlap.safetensors");
    write_raw_shard(
        &path,
        &json!({
            "first": {"dtype":"BF16", "shape":[1], "data_offsets":[0, 2]},
            "second": {"dtype":"BF16", "shape":[1], "data_offsets":[1, 3]}
        }),
        3,
    );

    let error = read_header(&path).expect_err("ranges must not overlap");
    assert!(matches!(
        error,
        Qwen3CheckpointError::OverlappingTensorRanges(_)
    ));
}

#[test]
fn rejects_holes_and_trailing_payload_bytes() {
    let fixture = Fixture::new();
    let path = fixture.path.join("hole.safetensors");
    write_raw_shard(
        &path,
        &json!({"weight": {"dtype":"BF16", "shape":[1], "data_offsets":[1, 3]}}),
        3,
    );

    let error = read_header(&path).expect_err("payload must be exactly covered");
    assert!(matches!(
        error,
        Qwen3CheckpointError::NonContiguousPayload(_)
    ));
}

#[test]
fn rejects_duplicate_top_level_header_keys_including_escaped_names() {
    // Raw JSON is essential: serializing a map would erase the duplicate.
    // The safetensors format explicitly disallows duplicate keys.
    // Provenance: docs/research/README.md#checkpoint-header-key-uniqueness.
    for header in [
        r#"{"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        r#"{"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"\u0077eight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        r#"{"__metadata__":{},"__metadata__":{},"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
    ] {
        let fixture = Fixture::new();
        let path = fixture.path.join("duplicate.safetensors");
        let mut bytes = u64::try_from(header.len())
            .expect("small header")
            .to_le_bytes()
            .to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0, 0]);
        fs::write(&path, bytes).expect("write duplicate-key header");

        let error = read_header(&path).expect_err("duplicate keys must not be collapsed");
        assert!(matches!(error, Qwen3CheckpointError::HeaderJson { .. }));
    }
}

#[test]
fn rejects_duplicate_nested_tensor_and_metadata_fields() {
    for header in [
        r#"{"weight":{"dtype":"F32","dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        r#"{"weight":{"dtype":"BF16","shape":[2],"shape":[1],"data_offsets":[0,2]}}"#,
        r#"{"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,1],"data_offsets":[0,2]}}"#,
        r#"{"__metadata__":{"format":"bad","\u0066ormat":"pt"},"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#,
        r#"{"weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2],"extra":[{"x":1,"x":2}]}}"#,
    ] {
        let fixture = Fixture::new();
        let path = fixture.path.join("duplicate-nested.safetensors");
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0, 0]);
        fs::write(&path, bytes).expect("write ambiguous raw header");
        assert!(
            matches!(
                read_header(&path),
                Err(Qwen3CheckpointError::HeaderJson { .. })
            ),
            "must reject {header}"
        );
    }
}

#[test]
fn unique_header_preserves_valid_json_value_types() {
    let json = r#"{"array":[true,false,null,-7,18446744073709551615,1.25,"escaped\ntext",{"key":"value"}],"empty":{}}"#;
    let actual = serde_json::from_str::<super::UniqueHeader>(json)
        .expect("valid unique JSON")
        .0;
    let expected: std::collections::BTreeMap<String, serde_json::Value> =
        serde_json::from_str(json).expect("independent JSON value parser");
    assert_eq!(actual, expected);
}

#[test]
fn unique_header_keeps_the_json_recursion_limit() {
    let json = format!("{{\"x\":{}0{}}}", "[".repeat(150), "]".repeat(150));
    assert!(serde_json::from_str::<super::UniqueHeader>(&json).is_err());
}

#[test]
fn rejects_non_object_metadata_without_treating_it_as_a_tensor() {
    let fixture = Fixture::new();
    let path = fixture.path.join("metadata.safetensors");
    write_raw_shard(&path, &json!({"__metadata__": "not an object"}), 0);

    let error = read_header(&path).expect_err("metadata must be an object");
    assert!(matches!(error, Qwen3CheckpointError::InvalidMetadata(_)));
}
