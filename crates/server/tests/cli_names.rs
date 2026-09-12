use std::process::Command;

#[test]
fn both_names_share_commands_and_report_the_invoked_name() {
    for (binary, name) in [
        (env!("CARGO_BIN_EXE_metallix"), "metallix"),
        (env!("CARGO_BIN_EXE_mx"), "mx"),
    ] {
        let output = Command::new(binary).arg("--help").output().unwrap();
        assert!(output.status.success());
        assert!(output.stderr.is_empty());
        let help = String::from_utf8(output.stdout).unwrap();
        assert!(help.contains(&format!("Usage: {name} <COMMAND>")));
        assert!(help.contains("inspect-v41"));
        assert!(help.contains("inspect-qwen-checkpoint"));

        let invalid = Command::new(binary).arg("not-a-command").output().unwrap();
        assert_eq!(invalid.status.code(), Some(2));
        assert!(invalid.stdout.is_empty());
        assert!(
            String::from_utf8(invalid.stderr)
                .unwrap()
                .contains("not-a-command")
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn prompt_and_raw_ids_conflict_before_model_loading() {
    for binary in [env!("CARGO_BIN_EXE_metallix"), env!("CARGO_BIN_EXE_mx")] {
        let output = Command::new(binary)
            .args([
                "gen",
                "--model",
                "/no-model-needed",
                "--prompt",
                "Hello",
                "--input-ids",
                "1,2,3",
            ])
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains("cannot be used with"));
        assert!(error.contains("--prompt"));
        assert!(error.contains("--input-ids"));
    }
}

#[cfg(all(feature = "metal", feature = "structured-output"))]
#[test]
fn inline_and_file_schema_conflict_before_model_loading() {
    let output = Command::new(env!("CARGO_BIN_EXE_mx"))
        .args([
            "gen",
            "--model",
            "/no-model-needed",
            "--json-schema",
            "schema.json",
            "--json-schema-inline",
            r#"{"type":"string"}"#,
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(error.contains("cannot be used with"));
    assert!(error.contains("--json-schema-inline"));
}
