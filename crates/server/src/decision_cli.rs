//! Bounded arguments for direct option scoring, separate from generation.

use std::{path::PathBuf, process::ExitCode};

use clap::Args;

use crate::{chat_generation::ResidentChatLimits, qwen_decisions};

#[derive(Debug, Args)]
pub(crate) struct DecisionArgs {
    /// Existing local Qwen3 checkpoint directory.
    #[arg(long)]
    model: PathBuf,
    /// JSON state and named choice, noul, or score questions (at most 1 MiB).
    #[arg(long)]
    request: PathBuf,
    /// Positive option-softmax temperature; does not establish calibration.
    #[arg(long, default_value = "1", value_parser = super::parse_temperature)]
    temperature: f64,
    /// Maximum rendered tokens per question; overflow fails without truncation.
    #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(1..=16384))]
    context_tokens: u32,
    /// Logical K/V admission budget in MiB, not a physical memory limit.
    #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=8192))]
    kv_budget_mib: u32,
}

impl DecisionArgs {
    pub(crate) fn run(self) -> ExitCode {
        qwen_decisions::decide(
            &self.model,
            &self.request,
            self.temperature,
            ResidentChatLimits::from_mib(self.context_tokens as usize, self.kv_budget_mib),
        )
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::{Cli, Command};

    #[test]
    fn direct_decisions_have_explicit_input_and_no_generation_arguments() {
        let parsed = Cli::try_parse_from([
            "mx",
            "decide",
            "--model",
            "local",
            "--request",
            "request.json",
        ])
        .expect("decision command");
        let Command::Decide(args) = parsed.command else {
            panic!("wrong command")
        };
        assert_eq!(args.temperature.to_bits(), 1.0_f64.to_bits());
        assert_eq!(args.context_tokens, 2048);
        for extra in [
            vec!["--temperature", "NaN"],
            vec!["--temperature", "0"],
            vec!["--context-tokens", "0"],
            vec!["--max-tokens", "1"],
            vec!["--sample"],
        ] {
            let mut args = vec![
                "mx",
                "decide",
                "--model",
                "local",
                "--request",
                "request.json",
            ];
            args.extend(extra);
            assert!(Cli::try_parse_from(args).is_err());
        }
    }
}
