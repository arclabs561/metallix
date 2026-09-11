//! Numerical comparison for locally generated reference logits.

use std::{
    fs,
    io::{BufReader, Read},
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Deserialize)]
struct ReferenceManifest {
    input_ids: Vec<i32>,
    reference: ReferenceProvenance,
    last_token_logits_f32le: LogitArtifact,
}

#[derive(Debug, Deserialize)]
struct ReferenceProvenance {
    config_sha256: String,
    weights_sha256: String,
}

#[derive(Debug, Deserialize)]
struct LogitArtifact {
    element_count: usize,
    byte_count: usize,
    sha256: String,
}

/// Elementwise agreement, including finite-value and full-vocabulary checks.
#[derive(Debug, Serialize)]
pub(crate) struct LogitComparison {
    compared: usize,
    max_absolute_error: f64,
    root_mean_squared_error: f64,
    mismatches: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    passed: bool,
}

impl LogitComparison {
    pub(crate) const fn passed(&self) -> bool {
        self.passed
    }
}

pub(crate) fn read_reference(
    path: &Path,
    manifest_path: &Path,
    model_dir: &Path,
    input_ids: &[i32],
    vocab_size: usize,
) -> Result<Vec<f32>, String> {
    let expected_bytes = vocab_size
        .checked_mul(4)
        .ok_or("vocabulary size overflow")?;
    let metadata = fs::metadata(path).map_err(|error| error.to_string())?;
    if metadata.len() != u64::try_from(expected_bytes).map_err(|error| error.to_string())? {
        return Err(format!(
            "reference must contain exactly {vocab_size} little-endian f32 logits"
        ));
    }

    let manifest = read_manifest(manifest_path)?;
    if manifest.input_ids != input_ids {
        return Err("reference manifest input IDs do not match this forward".into());
    }
    if manifest.last_token_logits_f32le.element_count != vocab_size
        || manifest.last_token_logits_f32le.byte_count != expected_bytes
    {
        return Err("reference manifest logit shape does not match this vocabulary".into());
    }

    validate_hash(
        &model_dir.join("config.json"),
        &manifest.reference.config_sha256,
        "reference manifest config",
    )?;
    validate_hash(
        &model_dir.join("model.safetensors"),
        &manifest.reference.weights_sha256,
        "reference manifest weights",
    )?;
    validate_hash(
        path,
        &manifest.last_token_logits_f32le.sha256,
        "reference logits",
    )?;

    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() != expected_bytes {
        return Err("reference changed size while reading".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

fn read_manifest(path: &Path) -> Result<ReferenceManifest, String> {
    let metadata = fs::metadata(path).map_err(|error| error.to_string())?;
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "reference manifest exceeds {MAX_MANIFEST_BYTES} byte limit"
        ));
    }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    if bytes.len() > usize::try_from(MAX_MANIFEST_BYTES).map_err(|error| error.to_string())? {
        return Err("reference manifest changed size while reading".into());
    }
    serde_json::from_slice(&bytes).map_err(|error| format!("invalid reference manifest: {error}"))
}

fn validate_hash(path: &Path, expected: &str, label: &str) -> Result<(), String> {
    let actual = sha256_file(path)?;
    if actual != expected {
        return Err(format!("{label} SHA-256 does not match"));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn compare_logits(actual: &[f32], expected: &[f32]) -> Result<LogitComparison, String> {
    compare_vectors(actual, expected, 0.0005, 0.0001)
}

pub(crate) fn compare_index_scores(
    actual: &[f32],
    expected: &[f32],
) -> Result<LogitComparison, String> {
    compare_vectors(actual, expected, 0.0001, 0.00001)
}

fn compare_vectors(
    actual: &[f32],
    expected: &[f32],
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> Result<LogitComparison, String> {
    if actual.is_empty() || actual.len() != expected.len() {
        return Err("actual and reference must have the same nonempty length".into());
    }
    let mut max_absolute_error = 0.0_f64;
    let mut squared_error = 0.0;
    let mut mismatches = 0;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err("non-finite value in actual or reference logits".into());
        }
        let difference = (f64::from(actual) - f64::from(expected)).abs();
        max_absolute_error = max_absolute_error.max(difference);
        squared_error += difference * difference;
        if difference > absolute_tolerance + relative_tolerance * f64::from(expected).abs() {
            mismatches += 1;
        }
    }
    let count = f64::from(u32::try_from(actual.len()).map_err(|error| error.to_string())?);
    Ok(LogitComparison {
        compared: actual.len(),
        max_absolute_error,
        root_mean_squared_error: (squared_error / count).sqrt(),
        mismatches,
        absolute_tolerance,
        relative_tolerance,
        passed: mismatches == 0,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        sync::atomic::{AtomicU64, Ordering},
    };

    use serde_json::json;

    use super::{compare_index_scores, compare_logits, read_reference, sha256_file};

    #[test]
    fn index_score_tolerance_is_stricter_than_decoder_logit_tolerance() {
        assert!(compare_logits(&[0.0002], &[0.0]).unwrap().passed());
        assert!(!compare_index_scores(&[0.0002], &[0.0]).unwrap().passed());
    }

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("metallix-parity-{}-{sequence}", std::process::id()));
        fs::create_dir(&path).expect("create unique temporary directory");
        path
    }

    fn write_manifest(
        path: &Path,
        model_dir: &Path,
        reference: &Path,
        input_ids: &[i32],
        vocab_size: usize,
    ) {
        let manifest = json!({
            "input_ids": input_ids,
            "reference": {
                "config_sha256": sha256_file(&model_dir.join("config.json")).unwrap(),
                "weights_sha256": sha256_file(&model_dir.join("model.safetensors")).unwrap(),
            },
            "last_token_logits_f32le": {
                "element_count": vocab_size,
                "byte_count": vocab_size * 4,
                "sha256": sha256_file(reference).unwrap(),
            },
        });
        fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    #[test]
    fn comparison_catches_errors_outside_the_top_logits() {
        let reference = [12.0, 10.0, -1.0, -20.0];
        assert!(compare_logits(&reference, &reference).unwrap().passed());
        let result = compare_logits(&[12.0, 10.0, -1.0, -19.0], &reference).unwrap();
        assert!(!result.passed());
        assert_eq!(result.mismatches, 1);
    }

    #[test]
    fn comparison_rejects_nonfinite_empty_and_partial_vectors() {
        assert!(compare_logits(&[], &[]).is_err());
        assert!(compare_logits(&[1.0], &[1.0, 2.0]).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(compare_logits(&[value], &[0.0]).is_err());
            assert!(compare_logits(&[0.0], &[value]).is_err());
        }
    }

    #[test]
    fn reference_manifest_binds_model_input_and_logit_payload() {
        let root = temporary_directory();
        let model = root.join("model");
        fs::create_dir(&model).unwrap();
        fs::write(model.join("config.json"), b"config").unwrap();
        fs::write(model.join("model.safetensors"), b"weights").unwrap();
        let logits = root.join("logits.f32");
        fs::write(
            &logits,
            [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat(),
        )
        .unwrap();
        let manifest = root.join("reference.json");
        write_manifest(&manifest, &model, &logits, &[7, 8], 2);

        assert_eq!(
            read_reference(&logits, &manifest, &model, &[7, 8], 2).unwrap(),
            vec![1.0, 2.0]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reference_manifest_rejects_wrong_input_or_hash() {
        let root = temporary_directory();
        let model = root.join("model");
        fs::create_dir(&model).unwrap();
        fs::write(model.join("config.json"), b"config").unwrap();
        fs::write(model.join("model.safetensors"), b"weights").unwrap();
        let logits = root.join("logits.f32");
        fs::write(
            &logits,
            [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat(),
        )
        .unwrap();
        let manifest = root.join("reference.json");
        write_manifest(&manifest, &model, &logits, &[7, 8], 2);

        assert!(read_reference(&logits, &manifest, &model, &[8, 7], 2).is_err());
        fs::write(model.join("config.json"), b"changed config").unwrap();
        assert!(read_reference(&logits, &manifest, &model, &[7, 8], 2).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
