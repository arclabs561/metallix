//! The `mx` binary. Every subcommand, `mx serve` included, runs through
//! [`server::run`].

fn main() -> std::process::ExitCode {
    server::run()
}
