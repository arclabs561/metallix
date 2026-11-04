//! Regression tests for malformed late-interaction projection files.

use qwen::late::LateProjection;

fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "metallix-qwen-regressions-{}-{name}",
        std::process::id()
    ));
    std::fs::write(&path, bytes).unwrap();
    path
}

/// A header length near `u64::MAX` is a malformed layout, reported as an
/// error rather than an arithmetic overflow.
#[test]
fn huge_header_length_is_an_error_not_a_panic() {
    let mut bytes = u64::MAX.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    let path = write("late-huge-header.safetensors", &bytes);
    let result = std::panic::catch_unwind(|| LateProjection::read(&path).is_err());
    std::fs::remove_file(&path).ok();
    assert_eq!(result.ok(), Some(true), "LateProjection::read panicked");
}

/// A shape whose byte size does not fit in `usize` is an error, not an
/// overflow or a match against an empty payload.
#[test]
fn overflowing_shape_is_an_error_not_a_panic_or_acceptance() {
    let header = br#"{"linear.weight":{"dtype":"F32","shape":[4611686018427387904,1024],"data_offsets":[0,0]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    let path = write("late-overflow-shape.safetensors", &bytes);
    let result = std::panic::catch_unwind(|| LateProjection::read(&path).is_err());
    std::fs::remove_file(&path).ok();
    assert_eq!(
        result.ok(),
        Some(true),
        "inconsistent shape panicked or was accepted"
    );
}
