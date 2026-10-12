//! The `mx` binary. Every subcommand, `mx serve` included, runs through
//! [`metallix_cli::run`].

fn main() -> std::process::ExitCode {
    metallix_cli::run()
}
