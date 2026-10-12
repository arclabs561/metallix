//! Raw-token generation admission is enforced before any checkpoint is opened.

use std::process::{Command, Output};

fn invoke(input_ids: &str, max_tokens: &str, context_tokens: &str, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mx"))
        .args([
            "generate",
            "deepseek",
            "--weights",
            "/no-deepseek-checkpoint-needed",
            "--input-ids",
            input_ids,
            "--max-tokens",
            max_tokens,
            "--context-tokens",
            context_tokens,
            "--json",
        ])
        .args(extra)
        .output()
        .expect("mx launches")
}

fn assert_refusal(output: &Output, code: i32, message: &str) {
    assert_eq!(output.status.code(), Some(code));
    assert!(
        output.stdout.is_empty(),
        "failed generation has no success receipt"
    );
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains(message), "unexpected failure: {error}");
}

#[test]
fn prompt_and_output_budget_fail_before_missing_checkpoint() {
    assert_refusal(&invoke("1,2", "2", "2", &[]), 1, "exceeds context-tokens");
    assert_refusal(&invoke("1", "1", "3", &[]), 1, "at least two input IDs");
}

#[test]
fn invalid_limits_ids_and_backend_are_cli_errors() {
    for output in [
        invoke("1,2", "0", "4", &[]),
        invoke("1,2", "257", "1024", &[]),
        invoke("1,2", "1", "32769", &[]),
        invoke("1,2", "1", "0", &[]),
        invoke("1,2", "1", "4", &["--cache-mib", "0"]),
        invoke("1,2", "1", "4", &["--resident-mib", "0"]),
        invoke("4294967296,2", "1", "4", &[]),
        invoke("1,2", "1", "4", &["--backend", "metal"]),
    ] {
        assert_refusal(&output, 2, "invalid value");
    }
}

#[test]
fn admitted_request_reaches_local_metadata_without_fetching() {
    assert_refusal(
        &invoke("1,2", "1", "2", &[]),
        1,
        "local store metadata admission failed",
    );
}

#[test]
fn both_binary_names_expose_scalar_generation() {
    for binary in [env!("CARGO_BIN_EXE_mx"), env!("CARGO_BIN_EXE_metallix")] {
        let output = Command::new(binary)
            .args(["generate", "deepseek", "--help"])
            .output()
            .expect("generation help launches");
        assert!(output.status.success());
        let help = String::from_utf8(output.stdout).expect("UTF-8 help");
        assert!(help.contains("--context-tokens"));
        assert!(help.contains("scalar"));
        assert!(help.contains("--eos-token-id"));
    }
}
