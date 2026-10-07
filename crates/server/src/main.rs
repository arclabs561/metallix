//! The `metallix` binary, identical to `mx`: both call [`server::run`].

fn main() -> std::process::ExitCode {
    server::run()
}
