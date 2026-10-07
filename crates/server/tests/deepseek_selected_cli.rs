//! Black-box refusals before checkpoint payload reads; no model fixture required.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicU64, Ordering},
};

const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    dir: PathBuf,
    capture_sha: String,
}
impl Fixture {
    fn new() -> Self {
        let mut selected = None;
        for _ in 0..64 {
            let dir = std::env::temp_dir().join(format!(
                "metallix-selected-cli-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&dir) {
                Ok(()) => {
                    selected = Some(dir);
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("create fixture: {error}"),
            }
        }
        let dir = selected.expect("unique temporary directory");
        let bytes = vec![0_u8; 5120 * 2];
        fs::write(dir.join("input.bin"), &bytes).expect("write input");
        fs::write(dir.join("expected.bin"), &bytes).expect("write expected");
        fs::write(dir.join("routes.json"), serde_json::to_vec(&json!({"revision":REVISION,"runs":[{"routes":[{"layer":0,"ids":[[0,1,2,3,4,5]]}]}]})).expect("serialize")).expect("write routes");
        Self {
            dir,
            capture_sha: format!("{:x}", Sha256::digest(&bytes)),
        }
    }
    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_mx"));
        command
            .arg("run-deepseek-selected")
            .arg("--index")
            .arg(self.dir.join("missing-index.json"))
            .arg("--headers-dir")
            .arg(&self.dir)
            .arg("--weights-dir")
            .arg(self.dir.join("missing-weights"))
            .args([
                "--revision",
                REVISION,
                "--tokens",
                "1",
                "--input-sha256",
                &self.capture_sha,
                "--expected-sha256",
                &self.capture_sha,
            ])
            .arg("--input-bf16")
            .arg(self.dir.join("input.bin"))
            .arg("--expected-bf16")
            .arg(self.dir.join("expected.bin"))
            .arg("--routes")
            .arg(self.dir.join("routes.json"));
        command
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
fn assert_refused(output: &Output, reason: &str) {
    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stdout.is_empty(),
        "refusal must not manufacture a comparison receipt"
    );
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(reason),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mutated_expected_capture_is_rejected_before_missing_checkpoint_metadata() {
    let fixture = Fixture::new();
    let mut changed = vec![0_u8; 5120 * 2];
    changed[0] = 1;
    fs::write(fixture.dir.join("expected.bin"), changed).expect("mutate oracle");
    let output = fixture.command().output().expect("mx launches");
    assert_refused(&output, "input/expected SHA256 mismatch");
}

#[test]
fn captures_over_the_row_bound_are_refused_before_reading_routes() {
    let fixture = Fixture::new();
    fs::File::create(fixture.dir.join("input.bin"))
        .expect("input")
        .set_len(5120 * 2 + 1)
        .expect("sparse size");
    let output = fixture.command().output().expect("mx launches");
    assert_refused(&output, "byte bound");
}

#[test]
fn duplicate_or_absent_source_routes_cannot_be_a_successful_comparison() {
    let fixture = Fixture::new();
    for routes in [json!([]), json!([{"layer":0,"ids":[[0,0,2,3,4,5]]}])] {
        fs::write(
            fixture.dir.join("routes.json"),
            serde_json::to_vec(&json!({"revision":REVISION,"runs":[{"routes":routes}]}))
                .expect("serialize"),
        )
        .expect("routes");
        let output = fixture.command().output().expect("mx launches");
        assert_refused(&output, "route");
    }
}

#[test]
fn aggregate_header_budget_is_refused_before_index_or_payload_io() {
    let fixture = Fixture::new();
    fs::write(fixture.dir.join("headers.json"), serde_json::to_vec(&json!({"revision":REVISION,"index_sha256":"unused","shards":{"model.safetensors":{"header_bytes":33*1024*1024,"header_sha256":"unused","file_bytes":33*1024*1024}}})).expect("serialize")).expect("manifest");
    let output = fixture.command().output().expect("mx launches");
    assert_refused(&output, "aggregate headers exceed 32 MiB");
}

#[test]
fn explicit_zero_budget_is_rejected_before_any_checkpoint_read() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .args(["--payload-budget-mib", "0"])
        .output()
        .expect("mx launches");
    assert_refused(&output, "budgets must be in 1..=512 MiB");
}

#[test]
fn missing_expected_output_cannot_be_reported_as_zero_error() {
    let fixture = Fixture::new();
    fs::rename(
        fixture.dir.join("expected.bin"),
        fixture.dir.join("unused.bin"),
    )
    .expect("hide expected output");
    let output = fixture.command().output().expect("mx launches");
    assert_refused(&output, "deepseek selected refused");
}
