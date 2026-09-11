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
