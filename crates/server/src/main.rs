use std::{fs, path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};
use deepseek::{V41TextContract, manifest::V41SafetensorsIndex};
use qwen::Qwen3TextContract;

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
    /// Validate a Qwen3 configuration without loading weights.
    InspectQwen {
        /// Path to the upstream model configuration.
        #[arg(long)]
        config: PathBuf,
    },
    /// Validate a DeepSeek-V4.1 safetensors index without downloading weights.
    InspectV41Index {
        /// Path to the upstream safetensors index.
        #[arg(long)]
        index: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::InspectV41 { config } => inspect_v41(&config),
        Command::InspectQwen { config } => inspect_qwen(&config),
        Command::InspectV41Index { index } => inspect_v41_index(&index),
    }
}

fn inspect_v41_index(index: &PathBuf) -> ExitCode {
    let json = match fs::read_to_string(index) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("could not read {}: {error}", index.display());
            return ExitCode::FAILURE;
        }
    };
    let index = match V41SafetensorsIndex::parse(&json) {
        Ok(index) => index,
        Err(error) => {
            eprintln!(
                "{} is not a valid V4.1 safetensors index: {error}",
                index.display()
            );
            return ExitCode::FAILURE;
        }
    };
    println!("V4.1 safetensors index");
    println!("tensors: {}", index.tensor_count());
    println!("shards: {}", index.shard_paths().len());
    println!("declared total bytes: {}", index.total_bytes());
    println!("next gate: resolve individual shard sizes before download planning");
    ExitCode::SUCCESS
}

fn inspect_qwen(config: &PathBuf) -> ExitCode {
    let json = match fs::read_to_string(config) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("could not read {}: {error}", config.display());
            return ExitCode::FAILURE;
        }
    };
    let contract = match Qwen3TextContract::parse(&json) {
        Ok(contract) => contract,
        Err(error) => {
            eprintln!(
                "{} is not a supported Qwen3 configuration: {error}",
                config.display()
            );
            return ExitCode::FAILURE;
        }
    };

    println!("Qwen3 text execution contract");
    println!("layers: {} transformer", contract.total_layers());
    println!("hidden size: {}", contract.hidden_size());
    println!("attention heads: {}", contract.attention_heads());
    println!("maximum positions: {}", contract.max_position_embeddings());
    println!("required backend: dense attention, paged KV, continuous batching");
    ExitCode::SUCCESS
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
    println!("layers: {} transformer", contract.total_layers());
    println!(
        "routing: {} experts, top-{}",
        contract.local_experts(),
        contract.experts_per_token()
    );
    println!(
        "engram: n-grams through {}",
        contract.engram_max_ngram_size()
    );
    let quantization = contract.quantization();
    println!(
        "checkpoint quantization: dynamic FP8, FP4 experts, {}x{} blocks",
        quantization.weight_block_rows(),
        quantization.weight_block_columns()
    );
    println!("required backend: CED, sparse MoE, Engram, paged weights, tiered cache");
    ExitCode::SUCCESS
}
