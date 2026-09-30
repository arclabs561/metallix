//! Scalar reduced-DeepSeek CLI agreement with its bounded library artifact.

use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

use deepseek::reduced::{MAX_REDUCED_ARTIFACT_BYTES, ReducedArtifact};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct TempArtifact {
    path: PathBuf,
}

impl TempArtifact {
    fn from_bytes(bytes: &[u8]) -> Self {
        let (path, mut file) = exclusive_file();
        file.write_all(bytes)
            .unwrap_or_else(|error| panic!("write temporary artifact: {error}"));
        Self { path }
    }

    fn sparse(length: u64) -> Self {
        let (path, file) = exclusive_file();
        file.set_len(length)
            .unwrap_or_else(|error| panic!("size temporary artifact: {error}"));
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn exclusive_file() -> (PathBuf, File) {
    for _ in 0..64 {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "metallix-deepseek-reduced-cli-{}-{sequence}.json",
            std::process::id()
        ));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return (path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => panic!("create exclusive temporary artifact: {error}"),
        }
    }
    panic!("could not allocate an exclusive temporary artifact name");
}

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn export_artifact() -> Vec<u8> {
    let root = repository_root();
    let output = Command::new("python3")
        .arg(root.join("scripts/export_v41_reduced_artifact.py"))
        .arg("--source")
        .arg(root.join("fixtures/deepseek-v41/reduced-runner-reference.json"))
        .args(["--output", "-"])
        .output()
        .expect("repository Python exporter launches");
    assert!(
        output.status.success(),
        "exporter failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "exporter wrote diagnostics");
    output.stdout
}

fn run_cli(artifact: &Path, prefill_tokens: usize) -> Output {
    run_cli_with_score_execution(artifact, prefill_tokens, None)
}

fn run_cli_with_score_execution(
    artifact: &Path,
    prefill_tokens: usize,
    score_execution: Option<&str>,
) -> Output {
    run_cli_with_execution(artifact, prefill_tokens, score_execution, None)
}

fn run_cli_with_execution(
    artifact: &Path,
    prefill_tokens: usize,
    score_execution: Option<&str>,
    key_rotary_execution: Option<&str>,
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mx"));
    command
        .args(["run-deepseek-reduced", "--artifact"])
        .arg(artifact)
        .args(["--input-ids", "0,1,2,3,4,5,6", "--prefill-tokens"])
        .arg(prefill_tokens.to_string());
    if let Some(score_execution) = score_execution {
        command.args(["--score-execution", score_execution]);
    }
    if let Some(key_rotary_execution) = key_rotary_execution {
        command.args(["--key-rotary-execution", key_rotary_execution]);
    }
    command.output().expect("reduced CLI launches")
}

fn expected_output(artifact: &ReducedArtifact, bytes: &[u8], prefill_tokens: usize) -> Value {
    let ids = [0, 1, 2, 3, 4, 5, 6];
    let calls = artifact
        .run(&ids, prefill_tokens)
        .expect("artifact request succeeds")
        .iter()
        .enumerate()
        .map(|(call, output)| {
            let positions = if call == 0 { prefill_tokens } else { 1 };
            let start = if call == 0 {
                0
            } else {
                prefill_tokens + call - 1
            };
            let logits_fp32_bits = output
                .heads()
                .last()
                .expect("nonempty reduced call")
                .logits()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            json!({
                "start": start,
                "positions": positions,
                "logits_fp32_bits": logits_fp32_bits,
            })
        })
        .collect::<Vec<_>>();
    json!({
        "schema_version": 1,
        "operation": "deepseek-reduced-request",
        "backend": "scalar",
        "artifact_sha256": format!("{:x}", Sha256::digest(bytes)),
        "calls": calls,
    })
}

#[test]
fn cli_matches_library_artifact_for_both_fixed_partitions() {
    let bytes = export_artifact();
    let artifact_file = TempArtifact::from_bytes(&bytes);
    assert_eq!(fs::read(artifact_file.path()).unwrap(), bytes);
    let artifact = ReducedArtifact::parse(&bytes).expect("exported artifact parses");

    for prefill_tokens in [5, 4] {
        let output = run_cli(artifact_file.path(), prefill_tokens);
        assert!(
            output.status.success(),
            "CLI failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let actual: Value = serde_json::from_slice(&output.stdout).expect("CLI emits JSON");
        assert_eq!(actual, expected_output(&artifact, &bytes, prefill_tokens));
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_cli_matches_scalar_library_output_for_both_fixed_partitions() {
    let bytes = export_artifact();
    let artifact_file = TempArtifact::from_bytes(&bytes);
    let artifact = ReducedArtifact::parse(&bytes).expect("exported artifact parses");

    for prefill_tokens in [5, 4] {
        let output =
            run_cli_with_score_execution(artifact_file.path(), prefill_tokens, Some("metal-bf16"));
        assert!(
            output.status.success(),
            "CLI failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.stderr.is_empty());
        let actual: Value = serde_json::from_slice(&output.stdout).expect("CLI emits JSON");
        let mut expected = expected_output(&artifact, &bytes, prefill_tokens);
        expected["backend"] = json!("mixed-cpu-metal");
        expected["score_execution"] = json!("metal-bf16");
        assert_eq!(actual, expected);
    }
    // The all-prefill source case has an ambiguous L4 selection tie. Selecting
    // Metal must preserve that rejection and must not emit a success receipt.
    let rejected = run_cli_with_score_execution(artifact_file.path(), 7, Some("metal-bf16"));
    assert!(!rejected.status.success());
    assert!(rejected.stdout.is_empty());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("execution was rejected"));
}

#[cfg(feature = "metal")]
#[test]
fn metal_key_rotation_alone_and_with_scores_matches_scalar_cli() {
    let bytes = export_artifact();
    let file = TempArtifact::from_bytes(&bytes);
    let artifact = ReducedArtifact::parse(&bytes).unwrap();
    for score in ["scalar", "metal-bf16"] {
        for prefill in [4, 5] {
            let output =
                run_cli_with_execution(file.path(), prefill, Some(score), Some("metal-fp32"));
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.stderr.is_empty());
            let mut expected = expected_output(&artifact, &bytes, prefill);
            expected["backend"] = json!("mixed-cpu-metal");
            expected["score_execution"] = json!(score);
            expected["key_rotary_execution"] = json!("metal-fp32");
            assert_eq!(
                serde_json::from_slice::<Value>(&output.stdout).unwrap(),
                expected
            );
        }
        let rejected = run_cli_with_execution(file.path(), 7, Some(score), Some("metal-fp32"));
        assert!(!rejected.status.success());
        assert!(rejected.stdout.is_empty());
    }
}

#[test]
fn unavailable_key_rotation_rejects_before_artifact_read() {
    let value = if cfg!(feature = "metal") {
        "unknown"
    } else {
        "metal-fp32"
    };
    let output = run_cli_with_execution(Path::new("missing-artifact"), 5, None, Some(value));
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
}

#[test]
fn invalid_score_execution_rejects_before_artifact_read_or_stdout() {
    let output = run_cli_with_score_execution(
        Path::new("missing-artifact"),
        5,
        Some("not-an-execution-mode"),
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
}

#[cfg(not(feature = "metal"))]
#[test]
fn nonmetal_cli_rejects_metal_score_execution_before_artifact_read_or_stdout() {
    let output = run_cli_with_score_execution(Path::new("missing-artifact"), 5, Some("metal-bf16"));
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
}

#[test]
fn invalid_prefill_rejects_before_artifact_read_or_stdout() {
    let owned = TempArtifact::from_bytes(b"unused");
    let missing = owned.path.clone();
    drop(owned);
    assert!(!missing.exists());
    let output = run_cli(&missing, 1);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("prefill token count"));
}

#[test]
fn malformed_or_unknown_artifacts_reject_without_stdout() {
    let malformed = TempArtifact::from_bytes(b"{");
    let malformed_output = run_cli(malformed.path(), 5);
    assert!(!malformed_output.status.success());
    assert!(malformed_output.stdout.is_empty());

    let mut unknown: Value = serde_json::from_slice(&export_artifact()).expect("exported JSON");
    unknown["unexpected"] = json!(true);
    let unknown = TempArtifact::from_bytes(&serde_json::to_vec(&unknown).unwrap());
    let unknown_output = run_cli(unknown.path(), 5);
    assert!(!unknown_output.status.success());
    assert!(unknown_output.stdout.is_empty());
}

#[test]
fn oversized_sparse_artifact_rejects_without_stdout() {
    let length = u64::try_from(MAX_REDUCED_ARTIFACT_BYTES)
        .expect("artifact limit fits u64")
        .checked_add(1)
        .expect("oversized test length fits u64");
    let oversized = TempArtifact::sparse(length);
    let output = run_cli(oversized.path(), 5);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("64 MiB"));
}
