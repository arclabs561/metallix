use std::{fs, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use deepseek::V41TextContract;

#[derive(Debug, Parser)]
#[command(about = "Apple-Silicon model-serving runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate a DeepSeek-V4.1 configuration without loading weights.
    InspectV41 {
        /// Path to the upstream model configuration.
        #[arg(long)]
        config: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::InspectV41 { config } => inspect_v41(&config),
    }
}

fn inspect_v41(config: &PathBuf) -> ExitCode {
    let json = match fs::read_to_string(config) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("could not read {}: {error}", config.display());
            return ExitCode::FAILURE;
        }
    };
    let contract = match V41TextContract::parse(&json) {
        Ok(contract) => contract,
        Err(error) => {
            eprintln!(
                "{} is not a supported V4.1 configuration: {error}",
                config.display()
            );
            return ExitCode::FAILURE;
        }
    };

    println!("V4.1 text execution contract");
    println!("layers: {} CED", contract.total_layers());
    println!(
        "routing: {} experts, top-{}",
        contract.local_experts(),
        contract.experts_per_token()
    );
    println!(
        "engram: n-grams through {}",
        contract.engram_max_ngram_size()
    );
    println!("required backend: CED, sparse MoE, Engram, paged weights, tiered cache");
    ExitCode::SUCCESS
}
