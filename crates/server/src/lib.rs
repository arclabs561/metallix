use std::{fs, path::PathBuf, process::ExitCode};

#[cfg(feature = "metal")]
mod generation_preview;
#[cfg(feature = "metal")]
mod parity;
#[cfg(all(feature = "metal", feature = "structured-output"))]
mod qwen_constraints;
#[cfg(feature = "metal")]
mod qwen_forward;
#[cfg(feature = "metal")]
mod v41_indexer;
#[cfg(feature = "metal")]
mod v41_rotary;

use clap::{Parser, Subcommand};
use deepseek::{V41TextContract, manifest::V41SafetensorsIndex};
use qwen::{
    Qwen3TextContract, checkpoint::Qwen3CheckpointInspection, preflight::Qwen3ExecutionPreflight,
};

#[derive(Debug, Parser)]
#[command(
    about = "Inspect model files and run experimental Metal inference",
    after_help = "\
Scope:
  No HTTP serving is implemented.
  Inspect commands read model configuration or checkpoint headers; they do not load weights.
  Metal commands require an Apple-Silicon build with --features metal.
  Qwen forward and generate use a local --model directory, raw token IDs, and one sequence."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compare V4.1 FP32 rotary tails on Metal with pinned upstream fixtures.
    #[cfg(feature = "metal")]
    CheckV41RotaryMetal {
        #[arg(long, default_value = "fixtures/deepseek-v41/rotary-reference.json")]
        fixture: PathBuf,
        /// Repeats after one excluded warmup; diagnostic round-trip timing only.
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=100))]
        repeats: u32,
    },
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
    /// Generate greedy Qwen3 raw token IDs with per-sequence KV reuse on Metal.
    #[cfg(feature = "metal")]
    #[command(visible_alias = "gen")]
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
        /// Emit phase timing and logical-memory diagnostics on stderr.
        #[arg(short, long, visible_alias = "debug")]
        verbose: bool,
        /// Include selected-token natural-log probabilities in the JSON report.
        #[arg(long)]
        logprobs: bool,
        /// Show generated content and diagnostics on stderr; color only on a terminal.
        #[arg(long)]
        preview: bool,
        /// Constrain generated JSON using a local schema (32 KiB maximum).
        #[cfg(feature = "structured-output")]
        #[arg(long)]
        json_schema: Option<PathBuf>,
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
        /// Also validate dimensions and mode relationships needed for execution planning.
        #[arg(long)]
        execution_shape: bool,
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
    /// Compare one bounded BF16 tensor read with the resident MLX loader.
    #[cfg(feature = "metal")]
    CheckQwenTensorMetal {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "model.layers.0.input_layernorm.weight")]
        tensor: String,
        /// Bound only the selected raw payload; the comparison loads the resident checkpoint.
        #[arg(long, default_value_t = 1_048_576, value_parser = clap::value_parser!(u64).range(1..=67_108_864))]
        max_bytes: u64,
    },
    /// Compare contiguous BF16 matrix rows with the resident MLX loader.
    #[cfg(feature = "metal")]
    CheckQwenRowsMetal {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "model.embed_tokens.weight")]
        tensor: String,
        #[arg(long, default_value_t = 0)]
        start_row: usize,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=4096))]
        rows: u32,
        /// Bound only the selected raw rows; resident comparison is outside this budget.
        #[arg(long, default_value_t = 1_048_576, value_parser = clap::value_parser!(u64).range(1..=67_108_864))]
        max_bytes: u64,
    },
    /// Compare selected Qwen embedding rows with the resident MLX loader.
    #[cfg(feature = "metal")]
    CheckQwenEmbeddingMetal {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
        input_ids: Vec<i32>,
        /// Bound only the selected raw embedding payload; resident comparison is outside this budget.
        #[arg(long, default_value_t = 1_048_576, value_parser = clap::value_parser!(u64).range(1..=67_108_864))]
        max_bytes: u64,
    },
    /// Compare tiled Qwen projection output with the resident MLX loader.
    #[cfg(feature = "metal")]
    CheckQwenProjectionMetal {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 1_024, value_parser = clap::value_parser!(u32).range(1..=4_096))]
        tile_rows: u32,
        /// Bound one tile's raw payload only; not aggregate or process peak memory.
        #[arg(long, default_value_t = 8_388_608, value_parser = clap::value_parser!(u64).range(1..=67_108_864))]
        max_bytes: u64,
    },
    /// Compare a complete uncached streamed Qwen forward with resident weights.
    #[cfg(feature = "metal")]
    CheckQwenStreamMetal {
        #[arg(long)]
        model: PathBuf,
        /// Raw token IDs; at most 32 tokens in this qualification path.
        #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
        input_ids: Vec<i32>,
        #[arg(long, default_value_t = 1_024, value_parser = clap::value_parser!(u32).range(1..=4_096))]
        tile_rows: u32,
        /// Logical weights and loading staging; excludes scratch and resident reference.
        #[arg(long, default_value_t = 134_217_728, value_parser = clap::value_parser!(u64).range(1..=1_073_741_824))]
        max_weight_bytes: u64,
        /// Run only the streamed candidate for process-memory measurement; emits logits but no parity result.
        #[arg(long)]
        candidate_only: bool,
    },
    /// Check streamed KV appends against resident cached and full forwards.
    #[cfg(feature = "metal")]
    CheckQwenStreamCacheMetal {
        #[arg(long)]
        model: PathBuf,
        /// Prefill raw token IDs; prompt plus appends must fit 32 tokens.
        #[arg(long, value_delimiter = ',', default_value = "1,2,3")]
        input_ids: Vec<i32>,
        /// Known token IDs to append one at a time; these are not generated tokens.
        #[arg(long, value_delimiter = ',', default_value = "4,5,6")]
        decode_ids: Vec<i32>,
        #[arg(long, default_value_t = 1_024, value_parser = clap::value_parser!(u32).range(1..=4_096))]
        tile_rows: u32,
        /// Logical weights/loading staging only; excludes KV, scratch and oracles.
        #[arg(long, default_value_t = 134_217_728, value_parser = clap::value_parser!(u64).range(1..=1_073_741_824))]
        max_weight_bytes: u64,
        /// Logical retained KV only; excludes transient copies and process overhead.
        #[arg(long, default_value_t = 67_108_864, value_parser = clap::value_parser!(u64).range(1..=1_073_741_824))]
        max_kv_bytes: u64,
    },
    /// Evaluate one selected-weight Qwen block and compare with resident weights.
    #[cfg(feature = "metal")]
    CheckQwenLayerMetal {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=32))]
        tokens: u32,
        /// Repeat this same synthetic-input layer diagnostic in one process.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=64))]
        repeats: u32,
        /// Logical weights plus read/conversion staging; excludes scratch and reference.
        #[arg(long, default_value_t = 268_435_456, value_parser = clap::value_parser!(u64).range(1..=1_073_741_824))]
        max_weight_bytes: u64,
        /// Skip the resident comparison for candidate-only process-memory measurement.
        #[arg(long)]
        candidate_only: bool,
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

/// Runs the shared CLI, preserving the invoked executable name in help output.
#[must_use]
#[allow(
    clippy::too_many_lines,
    reason = "exhaustive CLI dispatch; handlers stay separate"
)]
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "metal")]
        Command::CheckV41RotaryMetal { fixture, repeats } => v41_rotary::run(&fixture, repeats),
        #[cfg(feature = "metal")]
        Command::CheckV41IndexerMetal { fixture, repeats } => v41_indexer::run(&fixture, repeats),
        #[cfg(feature = "metal")]
        Command::GenerateQwenMetal {
            model,
            input_ids,
            max_tokens,
            verify_cache,
            verbose,
            logprobs,
            preview,
            #[cfg(feature = "structured-output")]
            json_schema,
        } => qwen_forward::generate(
            &model,
            &input_ids,
            max_tokens,
            verify_cache,
            qwen_forward::GenerationDiagnostics {
                verbose,
                logprobs,
                preview,
            },
            #[cfg(feature = "structured-output")]
            json_schema.as_deref(),
        ),
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
        Command::InspectV41 {
            config,
            execution_shape,
        } => inspect_v41(&config, execution_shape),
        Command::InspectQwen { config } => inspect_qwen(&config),
        Command::InspectQwenCheckpoint { model } => inspect_qwen_checkpoint(&model),
        Command::InspectV41Index { index } => inspect_v41_index(&index),
        #[cfg(feature = "metal")]
        Command::SmokeQwenMetal => smoke_qwen_metal(),
        #[cfg(feature = "metal")]
        Command::LoadQwenMetal { model } => load_qwen_metal(&model),
        #[cfg(feature = "metal")]
        Command::CheckQwenTensorMetal {
            model,
            tensor,
            max_bytes,
        } => check_qwen_tensor_metal(&model, &tensor, max_bytes),
        #[cfg(feature = "metal")]
        Command::CheckQwenRowsMetal {
            model,
            tensor,
            start_row,
            rows,
            max_bytes,
        } => check_qwen_rows_metal(&model, &tensor, start_row, rows, max_bytes),
        #[cfg(feature = "metal")]
        Command::CheckQwenEmbeddingMetal {
            model,
            input_ids,
            max_bytes,
        } => check_qwen_embedding_metal(&model, &input_ids, max_bytes),
        #[cfg(feature = "metal")]
        Command::CheckQwenProjectionMetal {
            model,
            tile_rows,
            max_bytes,
        } => check_qwen_projection_metal(&model, tile_rows, max_bytes),
        #[cfg(feature = "metal")]
        Command::CheckQwenStreamMetal {
            model,
            input_ids,
            tile_rows,
            max_weight_bytes,
            candidate_only,
        } => check_qwen_stream_metal(
            &model,
            &input_ids,
            tile_rows,
            max_weight_bytes,
            candidate_only,
        ),
        #[cfg(feature = "metal")]
        Command::CheckQwenStreamCacheMetal {
            model,
            input_ids,
            decode_ids,
            tile_rows,
            max_weight_bytes,
            max_kv_bytes,
        } => check_qwen_stream_cache_metal(
            &model,
            &input_ids,
            &decode_ids,
            tile_rows,
            max_weight_bytes,
            max_kv_bytes,
        ),
        #[cfg(feature = "metal")]
        Command::EmbedQwenMetal { model } => embed_qwen_metal(&model),
        #[cfg(feature = "metal")]
        Command::CheckQwenLayerMetal {
            model,
            layer,
            tokens,
            repeats,
            max_weight_bytes,
            candidate_only,
        } => check_qwen_layer_metal(
            &model,
            layer,
            tokens as usize,
            repeats,
            max_weight_bytes,
            candidate_only,
        ),
    }
}

#[cfg(feature = "metal")]
#[derive(serde::Serialize)]
struct LayerCheckCycles<T> {
    schema_version: u32,
    operation: &'static str,
    repeats: usize,
    runs: Vec<T>,
    scope: &'static str,
}

#[cfg(feature = "metal")]
fn layer_check_cycles<T>(runs: Vec<T>) -> LayerCheckCycles<T> {
    let repeats = runs.len();
    LayerCheckCycles {
        schema_version: 1,
        operation: "qwen3_selected_layer_cycles",
        repeats,
        runs,
        scope: "repeated same-layer synthetic-input diagnostic, not sequential model layers or model inference; every cycle reinspects config and headers and creates fresh synthetic input, so this measures a whole diagnostic lifecycle rather than an inner hot loop",
    }
}

#[cfg(feature = "metal")]
fn check_qwen_layer_metal(
    model: &std::path::Path,
    layer: usize,
    tokens: usize,
    repeats: u32,
    max_weight_bytes: u64,
    candidate_only: bool,
) -> ExitCode {
    let mode = if candidate_only {
        qwen::metal::LayerCheckMode::CandidateOnly
    } else {
        qwen::metal::LayerCheckMode::CompareResident
    };
    let mut reports = Vec::new();
    for _ in 0..repeats {
        match qwen::metal::qualify_layer(model, layer, tokens, max_weight_bytes, mode) {
            Ok(report) => reports.push(report),
            Err(error) => {
                eprintln!("Qwen layer check failed: {error}");
                return ExitCode::FAILURE;
            }
        }
    }
    let json = if repeats == 1 {
        serde_json::to_string_pretty(&reports[0])
    } else {
        serde_json::to_string_pretty(&layer_check_cycles(reports))
    };
    match json {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("could not serialize layer check: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn check_qwen_tensor_metal(model: &std::path::Path, tensor: &str, max_bytes: u64) -> ExitCode {
    match qwen::metal::qualify_tensor_range(model, tensor, max_bytes) {
        Ok(result) => match serde_json::to_string_pretty(&result) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not serialize tensor comparison: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Qwen tensor comparison failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn check_qwen_rows_metal(
    model: &std::path::Path,
    tensor: &str,
    start_row: usize,
    rows: u32,
    max_bytes: u64,
) -> ExitCode {
    let Some(row_range) = checked_row_range(start_row, rows) else {
        eprintln!("Qwen row check failed: start row plus row count overflows");
        return ExitCode::FAILURE;
    };
    match qwen::metal::qualify_tensor_rows(model, tensor, row_range, max_bytes) {
        Ok(result) => match serde_json::to_string_pretty(&result) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not serialize row comparison: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Qwen row check failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn checked_row_range(start_row: usize, rows: u32) -> Option<std::ops::Range<usize>> {
    let rows = usize::try_from(rows).ok()?;
    start_row
        .checked_add(rows)
        .map(|end_row| start_row..end_row)
}

#[cfg(feature = "metal")]
fn check_qwen_embedding_metal(
    model: &std::path::Path,
    input_ids: &[i32],
    max_bytes: u64,
) -> ExitCode {
    match qwen::metal::qualify_embedding(model, input_ids, max_bytes) {
        Ok(result) => match serde_json::to_string_pretty(&result) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not serialize embedding comparison: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Qwen embedding check failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn check_qwen_projection_metal(
    model: &std::path::Path,
    tile_rows: u32,
    max_bytes: u64,
) -> ExitCode {
    let Ok(tile_rows) = usize::try_from(tile_rows) else {
        eprintln!("Qwen projection check failed: tile row count does not fit usize");
        return ExitCode::FAILURE;
    };
    match qwen::metal::qualify_projection(model, tile_rows, max_bytes) {
        Ok(result) => match serde_json::to_string_pretty(&result) {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("could not serialize projection comparison: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("Qwen projection check failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn check_qwen_stream_metal(
    model: &std::path::Path,
    input_ids: &[i32],
    tile_rows: u32,
    max_weight_bytes: u64,
    candidate_only: bool,
) -> ExitCode {
    let Ok(tile_rows) = usize::try_from(tile_rows) else {
        eprintln!("Qwen stream check failed: tile row count does not fit usize");
        return ExitCode::FAILURE;
    };
    if candidate_only {
        match qwen::metal::run_streamed_forward_candidate(
            model,
            input_ids,
            max_weight_bytes,
            tile_rows,
        ) {
            Ok(report) => print_stream_report(report),
            Err(error) => {
                eprintln!("Qwen stream check failed: {error}");
                ExitCode::FAILURE
            }
        }
    } else {
        match qwen::metal::qualify_streamed_forward(model, input_ids, max_weight_bytes, tile_rows) {
            Ok(report) => print_stream_report(report),
            Err(error) => {
                eprintln!("Qwen stream check failed: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

#[cfg(feature = "metal")]
fn print_stream_report(report: impl serde::Serialize) -> ExitCode {
    match serde_json::to_string_pretty(&report) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("could not serialize stream report: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(feature = "metal")]
fn check_qwen_stream_cache_metal(
    model: &std::path::Path,
    input_ids: &[i32],
    decode_ids: &[i32],
    tile_rows: u32,
    max_weight_bytes: u64,
    max_kv_bytes: u64,
) -> ExitCode {
    let Ok(tile_rows) = usize::try_from(tile_rows) else {
        eprintln!("Qwen streamed cache check failed: tile rows do not fit usize");
        return ExitCode::FAILURE;
    };
    match qwen::metal::qualify_streamed_cached_forward(
        model,
        input_ids,
        decode_ids,
        max_weight_bytes,
        max_kv_bytes,
        tile_rows,
    ) {
        Ok(report) => print_stream_report(report),
        Err(error) => {
            eprintln!("Qwen streamed cache check failed: {error}");
            ExitCode::FAILURE
        }
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

fn inspect_v41(config: &PathBuf, execution_shape: bool) -> ExitCode {
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

    // The stronger execution gate is opt-in; metadata inspection remains useful
    // for configuration documents that are not yet executable load plans.
    if execution_shape {
        match deepseek::V41ExecutionShape::parse(&json) {
            Ok(shape) => {
                println!(
                    "execution dimensions: hidden {}, vocabulary {}, head width {}",
                    shape.hidden_size(),
                    shape.vocab_size(),
                    shape.head_dim()
                );
                println!(
                    "attention heads: {} query, {} KV; output groups: {}",
                    shape.attention_heads(),
                    shape.key_value_heads(),
                    shape.output_groups()
                );
                println!(
                    "CSA2 schedule: {} entries, {} KV sources, {} index sources",
                    shape.csa2().compress_ratios().len(),
                    shape.csa2().kv_source_layers().len(),
                    shape.csa2().index_source_layers().len()
                );
            }
            Err(error) => {
                eprintln!("V4.1 execution-shape qualification failed: {error}");
                return ExitCode::FAILURE;
            }
        }
        println!("scope: configuration dimensions only; no tensor loading or inference");
    }

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

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::Cli;

    #[test]
    fn execution_shape_inspection_is_explicitly_opt_in() {
        for (args, expected) in [
            (
                vec!["metallix", "inspect-v41", "--config", "config.json"],
                false,
            ),
            (
                vec![
                    "metallix",
                    "inspect-v41",
                    "--config",
                    "config.json",
                    "--execution-shape",
                ],
                true,
            ),
        ] {
            let cli = Cli::try_parse_from(args).expect("valid inspection command");
            assert!(
                matches!(cli.command, super::Command::InspectV41 { execution_shape, .. } if execution_shape == expected)
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qwen_generation_alias_preserves_the_existing_raw_id_diagnostic() {
        let default = Cli::try_parse_from(["mx", "gen", "--model", "model"])
            .expect("generation alias with defaults");
        assert!(matches!(
            default.command,
            super::Command::GenerateQwenMetal {
                input_ids,
                max_tokens: 32,
                verify_cache: false,
                verbose: false,
                logprobs: false,
                preview: false,
                ..
            } if input_ids == [1, 2, 3]
        ));

        let scores = Cli::try_parse_from(["mx", "gen", "--model", "model", "--logprobs"])
            .expect("opt-in log probabilities");
        assert!(matches!(
            scores.command,
            super::Command::GenerateQwenMetal { logprobs: true, .. }
        ));
        for flag in ["--verbose", "--debug", "-v"] {
            let cli = Cli::try_parse_from(["mx", "gen", "--model", "model", flag])
                .expect("generation diagnostic verbosity spelling");
            assert!(matches!(
                cli.command,
                super::Command::GenerateQwenMetal { verbose: true, .. }
            ));
        }
    }

    #[cfg(all(feature = "metal", feature = "structured-output"))]
    #[test]
    fn generation_accepts_an_explicit_local_json_schema() {
        let cli = Cli::try_parse_from([
            "mx",
            "gen",
            "--model",
            "model",
            "--json-schema",
            "schema.json",
            "--debug",
            "--preview",
        ])
        .expect("constrained generation arguments");
        assert!(matches!(cli.command,
            super::Command::GenerateQwenMetal { json_schema: Some(path), verbose: true, preview: true, .. }
            if path == std::path::Path::new("schema.json")
        ));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn tensor_check_has_a_finite_explicit_payload_budget() {
        let cli = Cli::try_parse_from(["metallix", "check-qwen-tensor-metal", "--model", "model"])
            .expect("default tensor diagnostic");
        assert!(matches!(
            cli.command,
            super::Command::CheckQwenTensorMetal {
                max_bytes: 1_048_576,
                ..
            }
        ));
        for limit in ["0", "67108865"] {
            assert!(
                Cli::try_parse_from([
                    "metallix",
                    "check-qwen-tensor-metal",
                    "--model",
                    "model",
                    "--max-bytes",
                    limit,
                ])
                .is_err()
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn row_check_bounds_its_selected_contiguous_range() {
        let cli = Cli::try_parse_from(["metallix", "check-qwen-rows-metal", "--model", "model"])
            .expect("default row diagnostic");
        assert!(matches!(
            cli.command,
            super::Command::CheckQwenRowsMetal {
                tensor,
                start_row: 0,
                rows: 3,
                max_bytes: 1_048_576,
                ..
            } if tensor == "model.embed_tokens.weight"
        ));
        for (flag, value) in [
            ("--rows", "0"),
            ("--rows", "4097"),
            ("--max-bytes", "0"),
            ("--max-bytes", "67108865"),
        ] {
            assert!(
                Cli::try_parse_from([
                    "metallix",
                    "check-qwen-rows-metal",
                    "--model",
                    "model",
                    flag,
                    value,
                ])
                .is_err()
            );
        }
        assert_eq!(super::checked_row_range(usize::MAX, 1), None);
        assert_eq!(super::checked_row_range(4, 3), Some(4..7));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn embedding_check_requires_ids_and_bounds_the_selected_payload() {
        let cli =
            Cli::try_parse_from(["metallix", "check-qwen-embedding-metal", "--model", "model"])
                .expect("default embedding diagnostic");
        assert!(matches!(
            cli.command,
            super::Command::CheckQwenEmbeddingMetal {
                input_ids,
                max_bytes: 1_048_576,
                ..
            } if input_ids == [1, 2, 3]
        ));
        assert!(
            Cli::try_parse_from([
                "metallix",
                "check-qwen-embedding-metal",
                "--model",
                "model",
                "--input-ids",
            ])
            .is_err()
        );
        for limit in ["0", "67108865"] {
            assert!(
                Cli::try_parse_from([
                    "metallix",
                    "check-qwen-embedding-metal",
                    "--model",
                    "model",
                    "--max-bytes",
                    limit,
                ])
                .is_err()
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn stream_cache_check_keeps_append_ids_and_budgets_distinct() {
        let cli = Cli::try_parse_from([
            "mx",
            "check-qwen-stream-cache-metal",
            "--model",
            "model",
            "--input-ids",
            "9707,11",
            "--decode-ids",
            "1879,151935",
            "--max-weight-bytes",
            "81798144",
            "--max-kv-bytes",
            "1048576",
        ])
        .expect("known appends and independent budgets");
        assert!(
            matches!(cli.command, super::Command::CheckQwenStreamCacheMetal {
            input_ids, decode_ids, max_weight_bytes: 81_798_144,
            max_kv_bytes: 1_048_576, tile_rows: 1024, ..
        } if input_ids == [9707, 11] && decode_ids == [1879, 151_935])
        );
        assert!(Cli::try_parse_from(["mx", "check-qwen-stream-cache-metal"]).is_err());
        for (flag, value) in [
            ("--tile-rows", "0"),
            ("--tile-rows", "4097"),
            ("--max-weight-bytes", "0"),
            ("--max-kv-bytes", "0"),
            ("--max-kv-bytes", "1073741825"),
        ] {
            assert!(
                Cli::try_parse_from([
                    "mx",
                    "check-qwen-stream-cache-metal",
                    "--model",
                    "model",
                    flag,
                    value,
                ])
                .is_err()
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn stream_check_requires_a_model_and_bounds_its_working_weights() {
        let cli = Cli::try_parse_from(["mx", "check-qwen-stream-metal", "--model", "model"])
            .expect("default streamed diagnostic");
        assert!(matches!(cli.command, super::Command::CheckQwenStreamMetal {
            input_ids, tile_rows: 1024, max_weight_bytes: 134_217_728, candidate_only: false, ..
        } if input_ids == [1, 2, 3]));
        let candidate = Cli::try_parse_from([
            "mx",
            "check-qwen-stream-metal",
            "--model",
            "model",
            "--candidate-only",
        ])
        .expect("candidate-only streamed diagnostic");
        assert!(matches!(
            candidate.command,
            super::Command::CheckQwenStreamMetal {
                candidate_only: true,
                ..
            }
        ));
        assert!(Cli::try_parse_from(["mx", "check-qwen-stream-metal"]).is_err());
        for (flag, value) in [
            ("--tile-rows", "0"),
            ("--tile-rows", "4097"),
            ("--max-weight-bytes", "0"),
            ("--max-weight-bytes", "1073741825"),
        ] {
            assert!(
                Cli::try_parse_from([
                    "mx",
                    "check-qwen-stream-metal",
                    "--model",
                    "model",
                    flag,
                    value,
                ])
                .is_err()
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn projection_check_requires_a_model_and_bounds_each_tile() {
        let cli = Cli::try_parse_from([
            "metallix",
            "check-qwen-projection-metal",
            "--model",
            "model",
        ])
        .expect("default projection diagnostic");
        assert!(matches!(
            cli.command,
            super::Command::CheckQwenProjectionMetal {
                tile_rows: 1_024,
                max_bytes: 8_388_608,
                ..
            }
        ));
        assert!(Cli::try_parse_from(["metallix", "check-qwen-projection-metal"]).is_err());
        for (flag, value) in [
            ("--tile-rows", "0"),
            ("--tile-rows", "4097"),
            ("--max-bytes", "0"),
            ("--max-bytes", "67108865"),
        ] {
            assert!(
                Cli::try_parse_from([
                    "metallix",
                    "check-qwen-projection-metal",
                    "--model",
                    "model",
                    flag,
                    value,
                ])
                .is_err()
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn layer_check_bounds_inputs_and_keeps_comparison_on_by_default() {
        let cli = Cli::try_parse_from(["metallix", "check-qwen-layer-metal", "--model", "model"])
            .unwrap();
        assert!(matches!(
            cli.command,
            super::Command::CheckQwenLayerMetal {
                tokens: 3,
                repeats: 1,
                max_weight_bytes: 268_435_456,
                candidate_only: false,
                ..
            }
        ));
        for (flag, value) in [
            ("--tokens", "0"),
            ("--tokens", "33"),
            ("--repeats", "0"),
            ("--repeats", "65"),
            ("--max-weight-bytes", "0"),
            ("--max-weight-bytes", "1073741825"),
        ] {
            assert!(
                Cli::try_parse_from([
                    "metallix",
                    "check-qwen-layer-metal",
                    "--model",
                    "model",
                    flag,
                    value
                ])
                .is_err()
            );
        }
        let cli = Cli::try_parse_from([
            "metallix",
            "check-qwen-layer-metal",
            "--model",
            "model",
            "--candidate-only",
            "--repeats",
            "64",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            super::Command::CheckQwenLayerMetal {
                candidate_only: true,
                repeats: 64,
                ..
            }
        ));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn repeated_layer_checks_serialize_a_typed_cycle_envelope() {
        let output = super::layer_check_cycles(vec![
            serde_json::json!({"layer": 0}),
            serde_json::json!({"layer": 0}),
        ]);
        let json = serde_json::to_value(output).expect("cycle envelope is serializable");

        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["operation"], "qwen3_selected_layer_cycles");
        assert_eq!(json["repeats"], 2);
        assert_eq!(
            json["runs"],
            serde_json::json!([{"layer": 0}, {"layer": 0}])
        );
        let repeats = usize::try_from(json["repeats"].as_u64().expect("JSON repeat count"))
            .expect("repeat count fits usize");
        assert_eq!(json["runs"].as_array().map(Vec::len), Some(repeats));
        assert!(
            json["scope"]
                .as_str()
                .is_some_and(|scope| scope.contains("whole diagnostic lifecycle"))
        );
    }

    fn root_help() -> String {
        Cli::command().render_long_help().to_string()
    }

    #[test]
    fn root_help_explains_the_current_scope() {
        let help = root_help();

        assert!(help.contains("Inspect model files and run experimental Metal inference"));
        assert!(help.contains("No HTTP serving is implemented."));
        assert!(help.contains("Inspect commands read model configuration or checkpoint headers"));
        assert!(help.contains("--features metal"));
        assert!(help.contains("raw token IDs, and one sequence"));
        assert!(help.contains("inspect-v41"));
        assert!(help.contains("inspect-qwen"));
    }

    #[cfg(not(feature = "metal"))]
    #[test]
    fn default_help_hides_metal_commands() {
        let help = root_help();

        assert!(!help.contains("generate-qwen-metal"));
        assert!(!help.contains("forward-qwen-metal"));
        assert!(!help.contains("check-v41-indexer-metal"));
        assert!(!help.contains("check-qwen-stream-cache-metal"));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_help_lists_the_existing_metal_commands() {
        let help = root_help();

        assert!(help.contains("generate-qwen-metal"));
        assert!(help.contains("forward-qwen-metal"));
        assert!(help.contains("check-v41-indexer-metal"));
        assert!(help.contains("check-qwen-stream-cache-metal"));
    }
}
