//! Regression tests for malformed checkpoint headers.

use checkpoint::gguf::GgufFile;

fn temp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "metallix-checkpoint-regressions-{}-{name}",
        std::process::id()
    ))
}

fn string(text: &str) -> Vec<u8> {
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    bytes
}

/// A GGUF file with no tensors and one U64 metadata key.
fn gguf_with_u64_key(key: &str, value: u64) -> std::path::PathBuf {
    let mut out = b"GGUF".to_vec();
    out.extend(3_u32.to_le_bytes());
    out.extend(0_u64.to_le_bytes()); // tensors
    out.extend(1_u64.to_le_bytes()); // keys
    out.extend(string(key));
    out.extend(10_u32.to_le_bytes()); // U64
    out.extend(value.to_le_bytes());
    let path = temp_path(&format!("{key}.gguf"));
    std::fs::write(&path, out).unwrap();
    path
}

/// `general.base_model.count` comes from the file; a huge value must not
/// size an allocation before any `general.base_model.N.*` key is read.
#[test]
fn base_model_count_is_bounded() {
    let path = gguf_with_u64_key("general.base_model.count", u64::MAX);
    let file = GgufFile::open(&path).expect("header parses");
    let result = std::panic::catch_unwind(|| file.metadata().base_models().len());
    std::fs::remove_file(&path).ok();
    assert!(
        result.is_ok(),
        "base_models() panicked on a huge declared count"
    );
}

fn safetensors(name: &str, header: &str, payload: usize) -> std::path::PathBuf {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend(std::iter::repeat_n(0_u8, payload));
    let path = temp_path(&format!("{name}.safetensors"));
    std::fs::write(&path, out).unwrap();
    path
}

/// The safetensors format requires that the byte buffer be entirely indexed
/// with no holes. Here four unindexed bytes precede the only tensor.
#[test]
fn safetensors_holes_in_the_byte_buffer_are_rejected() {
    let header = r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[4,8]}}"#;
    let path = safetensors("hole", header, 8);
    let result = checkpoint::safetensors::SafetensorsFile::open(&path, checkpoint::ReadCache::Use);
    std::fs::remove_file(&path).ok();
    assert!(result.is_err(), "file with an unindexed hole accepted");
}

/// Bytes after the last tensor are a hole too: the reference loader requires
/// the payloads to end exactly at the end of the file.
#[test]
fn safetensors_trailing_unindexed_bytes_are_rejected() {
    let header = r#"{"w":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
    let exact = safetensors("exact", header, 4);
    let trailing = safetensors("trailing", header, 8);
    let open =
        |path| checkpoint::safetensors::SafetensorsFile::open(path, checkpoint::ReadCache::Use);
    let (exact_result, trailing_result) = (open(&exact), open(&trailing));
    std::fs::remove_file(&exact).ok();
    std::fs::remove_file(&trailing).ok();
    assert!(exact_result.is_ok(), "{exact_result:?}");
    assert!(
        matches!(
            trailing_result,
            Err(checkpoint::CheckpointError::Unindexed(_, 4, 8))
        ),
        "{trailing_result:?}"
    );
}
