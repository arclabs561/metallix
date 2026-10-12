//! The `metallix` binary, identical to `mx`: both call [`metallix_cli::run`].

fn main() -> std::process::ExitCode {
    metallix_cli::run()
}
