//! Scalar reduced-DeepSeek CLI agreement with its bounded library artifact.

use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

use deepseek::reduced::{MAX_REDUCED_ARTIFACT_BYTES, ReducedArtifact};
#[cfg(feature = "metal")]
use deepseek::{
    indexer::{
        key::{IndexKeyPreparationExecution, IndexKeyRotaryExecution},
        query::IndexScoreExecution,
    },
    reduced::FinalHeadExecution,
};
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
    run_cli_with_all_execution(
        artifact,
        prefill_tokens,
        score_execution,
        key_rotary_execution,
        None,
    )
}

fn run_cli_with_all_execution(
    artifact: &Path,
    prefill_tokens: usize,
    score_execution: Option<&str>,
    key_rotary_execution: Option<&str>,
    head_execution: Option<&str>,
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
    if let Some(head_execution) = head_execution {
        command.args(["--head-execution", head_execution]);
    }
    command.output().expect("reduced CLI launches")
}

fn run_cli_with_key_preparation_execution(
    artifact: &Path,
    prefill_tokens: usize,
    score_execution: Option<&str>,
    key_preparation_execution: &str,
    head_execution: Option<&str>,
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mx"));
    command
        .args(["run-deepseek-reduced", "--artifact"])
        .arg(artifact)
        .args(["--input-ids", "0,1,2,3,4,5,6", "--prefill-tokens"])
        .arg(prefill_tokens.to_string())
        .args(["--key-preparation-execution", key_preparation_execution]);
    if let Some(score_execution) = score_execution {
        command.args(["--score-execution", score_execution]);
    }
    if let Some(head_execution) = head_execution {
        command.args(["--head-execution", head_execution]);
    }
    command.output().expect("reduced CLI launches")
}

fn expected_output(artifact: &ReducedArtifact, bytes: &[u8], prefill_tokens: usize) -> Value {
    let ids = [0, 1, 2, 3, 4, 5, 6];
    let outputs = artifact
        .run(&ids, prefill_tokens)
        .expect("scalar artifact request succeeds");
    expected_output_for_calls(bytes, prefill_tokens, &outputs)
}

fn expected_output_for_calls(
    bytes: &[u8],
    prefill_tokens: usize,
    outputs: &[deepseek::reduced::RequestStepOutput],
) -> Value {
    let calls = outputs
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

#[cfg(feature = "metal")]
fn expected_output_with_head_execution(
    artifact: &ReducedArtifact,
    bytes: &[u8],
    prefill_tokens: usize,
    score_execution: IndexScoreExecution,
    key_rotary_execution: IndexKeyRotaryExecution,
) -> Value {
    let ids = [0, 1, 2, 3, 4, 5, 6];
    let outputs = artifact
        .run_with_head_execution(
            &ids,
            prefill_tokens,
            score_execution,
            key_rotary_execution,
            FinalHeadExecution::MetalFp32,
        )
        .expect("selected artifact request succeeds");
    expected_output_for_calls(bytes, prefill_tokens, &outputs)
}

#[cfg(feature = "metal")]
fn expected_output_with_key_preparation_execution(
    artifact: &ReducedArtifact,
    bytes: &[u8],
    prefill_tokens: usize,
    score_execution: IndexScoreExecution,
    key_preparation_execution: IndexKeyPreparationExecution,
    head_execution: FinalHeadExecution,
) -> Value {
    let ids = [0, 1, 2, 3, 4, 5, 6];
    let outputs = artifact
        .run_with_key_preparation_execution(
            &ids,
            prefill_tokens,
            score_execution,
            key_preparation_execution,
            head_execution,
        )
        .expect("selected artifact request succeeds");
    expected_output_for_calls(bytes, prefill_tokens, &outputs)
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

        let explicit_scalar = run_cli_with_key_preparation_execution(
            artifact_file.path(),
            prefill_tokens,
            None,
            "scalar",
            None,
        );
        assert!(
            explicit_scalar.status.success(),
            "CLI failed: {}",
            String::from_utf8_lossy(&explicit_scalar.stderr)
        );
        assert!(explicit_scalar.stderr.is_empty());
        let actual: Value =
            serde_json::from_slice(&explicit_scalar.stdout).expect("CLI emits JSON");
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

#[cfg(feature = "metal")]
#[test]
fn metal_key_preparation_receipt_and_execution_match_selected_library() {
    let bytes = export_artifact();
    let file = TempArtifact::from_bytes(&bytes);
    let artifact = ReducedArtifact::parse(&bytes).unwrap();
    for score in [IndexScoreExecution::Scalar, IndexScoreExecution::MetalBf16] {
        let score_arg = if score == IndexScoreExecution::Scalar {
            "scalar"
        } else {
            "metal-bf16"
        };
        for prefill in [4, 5] {
            let output = run_cli_with_key_preparation_execution(
                file.path(),
                prefill,
                Some(score_arg),
                "metal-prefp4",
                None,
            );
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.stderr.is_empty());
            let mut expected = expected_output_with_key_preparation_execution(
                &artifact,
                &bytes,
                prefill,
                score,
                IndexKeyPreparationExecution::MetalPreFp4,
                FinalHeadExecution::Scalar,
            );
            expected["backend"] = json!("mixed-cpu-metal");
            expected["key_preparation_execution"] = json!("metal-prefp4");
            if score == IndexScoreExecution::MetalBf16 {
                expected["score_execution"] = json!(score_arg);
            }
            assert_eq!(
                serde_json::from_slice::<Value>(&output.stdout).unwrap(),
                expected
            );
        }
        let rejected = run_cli_with_key_preparation_execution(
            file.path(),
            7,
            Some(score_arg),
            "metal-prefp4",
            None,
        );
        assert!(!rejected.status.success());
        assert!(rejected.stdout.is_empty());
    }
}

#[cfg(feature = "metal")]
#[test]
fn metal_final_head_cli_matches_selected_library_execution_for_both_fixed_partitions() {
    let bytes = export_artifact();
    let file = TempArtifact::from_bytes(&bytes);
    let artifact = ReducedArtifact::parse(&bytes).unwrap();
    for (score, key) in [
        ("scalar", "scalar"),
        ("metal-bf16", "scalar"),
        ("scalar", "metal-fp32"),
        ("metal-bf16", "metal-fp32"),
    ] {
        let score_execution = match score {
            "scalar" => IndexScoreExecution::Scalar,
            "metal-bf16" => IndexScoreExecution::MetalBf16,
            _ => unreachable!("listed score execution"),
        };
        let key_rotary_execution = match key {
            "scalar" => IndexKeyRotaryExecution::Scalar,
            "metal-fp32" => IndexKeyRotaryExecution::MetalFp32,
            _ => unreachable!("listed key rotation execution"),
        };
        for prefill in [4, 5] {
            let output = run_cli_with_all_execution(
                file.path(),
                prefill,
                Some(score),
                Some(key),
                Some("metal-fp32"),
            );
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(output.stderr.is_empty());
            let mut expected = expected_output_with_head_execution(
                &artifact,
                &bytes,
                prefill,
                score_execution,
                key_rotary_execution,
            );
            expected["backend"] = json!("mixed-cpu-metal");
            if key == "metal-fp32" {
                expected["score_execution"] = json!(score);
                expected["key_rotary_execution"] = json!("metal-fp32");
            } else if score == "metal-bf16" {
                expected["score_execution"] = json!("metal-bf16");
            }
            expected["head_execution"] = json!("metal-fp32");
            assert_eq!(
                serde_json::from_slice::<Value>(&output.stdout).unwrap(),
                expected
            );
        }
        let rejected =
            run_cli_with_all_execution(file.path(), 7, Some(score), Some(key), Some("metal-fp32"));
        assert!(!rejected.status.success());
        assert!(rejected.stdout.is_empty());
        assert!(String::from_utf8_lossy(&rejected.stderr).contains("execution was rejected"));
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
fn unavailable_key_preparation_rejects_before_artifact_read() {
    let value = if cfg!(feature = "metal") {
        "unknown"
    } else {
        "metal-prefp4"
    };
    let output =
        run_cli_with_key_preparation_execution(Path::new("missing-artifact"), 5, None, value, None);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("invalid value"));
}

#[cfg(feature = "metal")]
#[test]
fn conflicting_key_execution_flags_reject_before_artifact_read() {
    let mut command = Command::new(env!("CARGO_BIN_EXE_mx"));
    let output = command
        .args(["run-deepseek-reduced", "--artifact", "missing-artifact"])
        .args(["--input-ids", "0,1", "--prefill-tokens", "2"])
        .args(["--key-rotary-execution", "scalar"])
        .args(["--key-preparation-execution", "metal-prefp4"])
        .output()
        .expect("reduced CLI launches");
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(String::from_utf8_lossy(&output.stderr).contains("conflicts"));
}

#[test]
fn unavailable_final_head_rejects_before_artifact_read() {
    let value = if cfg!(feature = "metal") {
        "unknown"
    } else {
        "metal-fp32"
    };
    let output =
        run_cli_with_all_execution(Path::new("missing-artifact"), 5, None, None, Some(value));
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
