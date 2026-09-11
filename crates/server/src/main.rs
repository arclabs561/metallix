use std::{fs, path::PathBuf, process::ExitCode};

#[cfg(feature = "metal")]
mod parity;
#[cfg(feature = "metal")]
mod qwen_forward;
#[cfg(feature = "metal")]
mod v41_indexer;

use clap::{Parser, Subcommand};
use deepseek::{V41TextContract, manifest::V41SafetensorsIndex};
use qwen::{
    Qwen3TextContract, checkpoint::Qwen3CheckpointInspection, preflight::Qwen3ExecutionPreflight,
};

#[derive(Debug, Parser)]
#[command(about = "Apple-Silicon model-serving runtime")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compare V4.1 FP32 index-score arithmetic on Metal with a pinned CPU fixture.
    #[cfg(feature = "metal")]
    CheckV41IndexerMetal {
        #[arg(
            long,
            default_value = "fixtures/deepseek-v41/index-score-reference.json"
        )]
        fixture: PathBuf,
        /// Repeats per case after one excluded warmup; not model throughput.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=100))]
        repeats: u32,
    },
    /// Generate greedy raw token IDs with per-sequence KV reuse on Metal.
    #[cfg(feature = "metal")]
    GenerateQwenMetal {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
        input_ids: Vec<i32>,
        #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..=256))]
        max_tokens: u32,
        /// Compare each cached result with a full forward outside timed regions.
        #[arg(long)]
        verify_cache: bool,
    },
    /// Run and time the complete uncached Qwen3 decoder on raw token IDs.
    #[cfg(feature = "metal")]
    ForwardQwenMetal {
        #[arg(long)]
        model: PathBuf,
        /// Comma-separated raw token IDs; excludes chat-template/tokenizer effects.
        #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
        input_ids: Vec<i32>,
        /// Full final-token reference logits, little-endian float32.
        #[arg(long, requires = "reference_manifest")]
        reference: Option<PathBuf>,
        /// CPU capture JSON binding token IDs, checkpoint hashes, and reference bytes.
        #[arg(long, requires = "reference")]
        reference_manifest: Option<PathBuf>,
        /// Number of measured repeats after one excluded warmup.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=100))]
        repeats: u32,
    },
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
    /// Validate a local Qwen3 checkpoint's safetensors headers without loading payloads.
    InspectQwenCheckpoint {
        /// Directory containing config.json and safetensors shard files.
        #[arg(long)]
        model: PathBuf,
    },
    /// Prove the optional Qwen MLX substrate can execute one graph on Metal.
    #[cfg(feature = "metal")]
    SmokeQwenMetal,
    /// Load a validated Qwen3 checkpoint as MLX arrays and evaluate its embedding on Metal.
    #[cfg(feature = "metal")]
    LoadQwenMetal {
        /// Directory containing config.json and safetensors shard files.
        #[arg(long)]
        model: PathBuf,
    },
    /// Execute the fixed [1, 2, 3] Qwen3 embedding lookup on Metal.
    #[cfg(feature = "metal")]
    EmbedQwenMetal {
        /// Directory containing config.json and safetensors shard files.
        #[arg(long)]
        model: PathBuf,
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
        #[cfg(feature = "metal")]
        Command::CheckV41IndexerMetal { fixture, repeats } => v41_indexer::run(&fixture, repeats),
        #[cfg(feature = "metal")]
        Command::GenerateQwenMetal {
            model,
            input_ids,
            max_tokens,
            verify_cache,
        } => qwen_forward::generate(&model, &input_ids, max_tokens, verify_cache),
        #[cfg(feature = "metal")]
        Command::ForwardQwenMetal {
            model,
            input_ids,
            reference,
            reference_manifest,
            repeats,
        } => qwen_forward::run(
            &model,
            &input_ids,
            reference.as_deref().zip(reference_manifest.as_deref()),
            repeats,
        ),
        Command::InspectV41 { config } => inspect_v41(&config),
        Command::InspectQwen { config } => inspect_qwen(&config),
        Command::InspectQwenCheckpoint { model } => inspect_qwen_checkpoint(&model),
        Command::InspectV41Index { index } => inspect_v41_index(&index),
        #[cfg(feature = "metal")]
        Command::SmokeQwenMetal => smoke_qwen_metal(),
        #[cfg(feature = "metal")]
        Command::LoadQwenMetal { model } => load_qwen_metal(&model),
        #[cfg(feature = "metal")]
        Command::EmbedQwenMetal { model } => embed_qwen_metal(&model),
    }
}

#[cfg(feature = "metal")]
fn embed_qwen_metal(model: &PathBuf) -> ExitCode {
    match qwen::metal::Qwen3MlxWeights::load(model)
        .and_then(|weights| weights.embed_token_ids(&[1, 2, 3]))
    {
        Ok(shape) => {
            println!("Qwen MLX Metal embedding lookup qualified");
            println!("input IDs: 3 fixed raw tokens");
            println!("embedding output shape: {shape:?}");
            println!("scope: fixed embedding lookup; no decoder logits");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Qwen MLX Metal embedding lookup failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn load_qwen_metal(model: &PathBuf) -> ExitCode {
    match qwen::metal::Qwen3MlxWeights::load(model) {
        Ok(weights) => {
            println!("Qwen MLX Metal checkpoint load qualified");
            println!("tensors loaded: {}", weights.tensor_count());
            println!("embedding shape: {:?}", weights.embedding_shape());
            println!(
                "declared checkpoint bytes: {}",
                weights.inspection().tensor_bytes()
            );
            println!("scope: tensors loaded; decoder not evaluated");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Qwen MLX Metal checkpoint load failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn smoke_qwen_metal() -> ExitCode {
    match qwen::metal::run_metal_smoke() {
        Ok(smoke) => {
            println!("Qwen MLX Metal substrate qualified");
            println!("1x1 GPU matrix product: {}", smoke.product());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("Qwen MLX Metal substrate failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn inspect_qwen_checkpoint(model: &PathBuf) -> ExitCode {
    match Qwen3CheckpointInspection::inspect(model) {
        Ok(checkpoint) => {
            println!("Qwen3 checkpoint contract");
            println!("tensors: {}", checkpoint.tensor_count());
            println!("shards: {}", checkpoint.shards().len());
            println!("declared tensor bytes: {}", checkpoint.tensor_bytes());
            println!("scope: checkpoint metadata and byte ranges; no tensor evaluation");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!(
                "{} is not a valid Qwen3 checkpoint: {error}",
                model.display()
            );
            ExitCode::FAILURE
        }
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
    let preflight = match Qwen3ExecutionPreflight::from_contract(&contract) {
        Ok(preflight) => preflight,
        Err(error) => {
            eprintln!(
                "{} cannot form a Qwen3 execution plan: {error}",
                config.display()
            );
            return ExitCode::FAILURE;
        }
    };

    println!("Qwen3 text execution contract");
    println!("layers: {} transformer", contract.total_layers());
    println!("hidden size: {}", contract.hidden_size());
    println!("attention heads: {}", contract.attention_heads());
    println!("key/value heads: {}", preflight.key_value_heads());
    println!("head dimension: {}", preflight.head_dim());
    println!(
        "BF16 KV bytes per token: {}",
        preflight.kv_bytes_per_token_bf16()
    );
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
