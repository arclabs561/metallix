use std::{fs, io::Read, path::PathBuf, process::ExitCode};

#[cfg(feature = "metal")]
use std::time::Duration;

#[cfg(feature = "metal")]
use crate::chat_generation::ResidentChatLimits;

#[cfg(feature = "metal")]
mod agent_receipt;
#[cfg(feature = "metal")]
mod chat_cli;
#[cfg(feature = "metal")]
mod chat_generation;
#[cfg(feature = "metal")]
mod chat_tools;
#[cfg(feature = "metal")]
mod http_transport;
#[cfg(feature = "metal")]
mod responses;

#[cfg(feature = "metal")]
mod generation_preview;
#[cfg(feature = "metal")]
mod parity;
#[cfg(all(feature = "metal", feature = "structured-output"))]
mod qwen_constraints;
#[cfg(feature = "metal")]
mod qwen_forward;
#[cfg(feature = "metal")]
mod qwen_tokenizer;
#[cfg(feature = "metal")]
mod v41_indexer;
#[cfg(feature = "metal")]
mod v41_rotary;

use clap::{Parser, Subcommand};
#[cfg(feature = "metal")]
use deepseek::checkpoint::mlx::{read_affine_rows_from_shard, read_bf16_tensor_from_shard};
use deepseek::{
    V41TextContract,
    checkpoint::mlx::{
        collapse_hc_hidden, mix_hc_coefficients, read_affine_row_from_shard,
        read_f32_tensor_from_shard,
    },
    manifest::{MlxSafetensorsIndex, V41SafetensorsIndex},
};
use qwen::{
    Qwen3TextContract, checkpoint::Qwen3CheckpointInspection, preflight::Qwen3ExecutionPreflight,
};

#[derive(Debug, Parser)]
#[command(
    about = "Inspect model files and run experimental Metal inference",
    after_help = "\
Scope:
  Experimental native chat, read-only agent, and loopback Responses serving require Metal.
  Inspect commands read model configuration or checkpoint headers; they do not load weights.
  Metal commands require an Apple-Silicon build with --features metal.
  Qwen forward uses raw token IDs; generate also accepts a local-tokenizer plain-text prompt for one sequence."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[cfg(feature = "metal")]
const fn resident_chat_limits(context_tokens: u32, kv_budget_mib: u32) -> ResidentChatLimits {
    ResidentChatLimits::from_mib(context_tokens as usize, kv_budget_mib)
}

/// Which Qwen weight residency contract a generation run uses.
///
/// This belongs to the Qwen diagnostic command rather than the engine: the
/// streamed executor has adapter-specific layout and cache semantics.
#[cfg(feature = "metal")]
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum GenerationMemoryMode {
    /// Load the complete checkpoint once and retain its MLX arrays.
    Resident,
    /// Read one layer at a time under explicit weight and KV budgets.
    Streamed,
}

#[cfg(feature = "metal")]
const fn generation_max_tokens(mode: GenerationMemoryMode, requested: Option<u32>) -> u32 {
    match requested {
        Some(tokens) => tokens,
        None => match mode {
            GenerationMemoryMode::Resident => 32,
            // Keep the bare streamed command inside the qualified 32-token
            // total-context envelope for its default three-token prompt.
            GenerationMemoryMode::Streamed => 4,
        },
    }
}

#[cfg(feature = "metal")]
enum GenerationInput {
    RawIds(Vec<i32>),
    Prompt(String),
}

/// Keeps the existing raw-ID diagnostic default while making text input an
/// explicit, tokenizer-backed request.
#[cfg(feature = "metal")]
fn generation_input(
    input_ids: Option<Vec<i32>>,
    prompt: Option<String>,
) -> Result<GenerationInput, String> {
    match (input_ids, prompt) {
        (Some(input_ids), None) => Ok(GenerationInput::RawIds(input_ids)),
        (None, Some(prompt)) => Ok(GenerationInput::Prompt(prompt)),
        (None, None) => Ok(GenerationInput::RawIds(vec![1, 2, 3])),
        (Some(_), Some(_)) => Err(String::from(
            "--input-ids and --prompt cannot be used together",
        )),
    }
}

#[cfg(feature = "metal")]
fn parse_temperature(value: &str) -> Result<f64, String> {
    let temperature = value
        .parse::<f64>()
        .map_err(|_| String::from("temperature must be a finite number greater than zero"))?;
    if temperature.is_finite() && temperature > 0.0 {
        Ok(temperature)
    } else {
        Err(String::from(
            "temperature must be a finite number greater than zero",
        ))
    }
}

#[cfg(feature = "metal")]
fn sampling_configuration(
    sample: bool,
    temperature: Option<f64>,
    seed: Option<u64>,
) -> Result<Option<qwen_forward::SamplingConfiguration>, String> {
    match (sample, temperature, seed) {
        (false, None, None) => Ok(None),
        (true, Some(temperature), Some(seed)) if temperature.is_finite() && temperature > 0.0 => {
            Ok(Some(qwen_forward::SamplingConfiguration {
                seed,
                temperature,
            }))
        }
        (true, _, _) => Err(String::from(
            "--sample requires a finite positive --temperature and --seed",
        )),
        (false, _, _) => Err(String::from("--temperature and --seed require --sample")),
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Chat using the checkpoint template and one resident Qwen model.
    #[cfg(feature = "metal")]
    Chat {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        prompt: Option<String>,
        #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..=256))]
        max_tokens: u32,
        /// Total prompt plus output budget for this resident session.
        #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(1..=16384))]
        context_tokens: u32,
        /// Logical resident K/V admission budget in MiB; not an MLX allocation limit.
        #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=8192))]
        kv_budget_mib: u32,
        /// Emit a structured receipt for a single prompt.
        #[arg(long, requires = "prompt")]
        json: bool,
    },
    /// Run a bounded agent with workspace file-read and search tools.
    #[cfg(feature = "metal")]
    Agent {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        workspace: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..=256))]
        max_tokens: u32,
        #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=16))]
        max_turns: u32,
        /// Total prompt plus output budget for each agent turn.
        #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(1..=16384))]
        context_tokens: u32,
        /// Logical resident K/V admission budget in MiB; not an MLX allocation limit.
        #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=8192))]
        kv_budget_mib: u32,
        /// Emit a structured execution receipt. Task success is assessed by the caller.
        #[arg(long)]
        json: bool,
    },
    /// Serve the native Qwen control model through a local Responses endpoint.
    #[cfg(feature = "metal")]
    Serve {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value = "metallix-qwen3")]
        model_id: String,
        #[arg(long, default_value = "127.0.0.1:8321")]
        listen: std::net::SocketAddr,
        /// Total prompt plus output budget for each request.
        #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(1..=16384))]
        context_tokens: u32,
        /// Logical resident K/V admission budget in MiB; not an MLX allocation limit.
        #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=8192))]
        kv_budget_mib: u32,
        /// Cooperative generation budget per request, in milliseconds.
        #[arg(long, default_value_t = 60_000, value_parser = clap::value_parser!(u32).range(1..=120_000))]
        generation_timeout_ms: u32,
    },
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
    /// Generate one Qwen3 sequence with per-sequence KV reuse on Metal.
    #[cfg(feature = "metal")]
    #[command(visible_alias = "gen")]
    GenerateQwenMetal {
        #[arg(long)]
        model: PathBuf,
        /// Comma-separated raw token IDs; defaults to 1,2,3 when --prompt is absent.
        #[arg(long, value_delimiter = ',', conflicts_with = "prompt")]
        input_ids: Option<Vec<i32>>,
        /// Plain-text prompt (at most 1 MiB), encoded by local tokenizer.json without a chat template or special tokens.
        #[arg(long, conflicts_with = "input_ids")]
        prompt: Option<String>,
        /// Generated-token limit; default is 32 resident or 4 streamed. Streamed prompt plus limit must fit 32.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=256))]
        max_tokens: Option<u32>,
        /// Keep the complete checkpoint resident (default) or stream one layer at a time.
        #[arg(long, value_enum, default_value_t = GenerationMemoryMode::Resident)]
        memory_mode: GenerationMemoryMode,
        /// Streamed mode only: maximum planned layer weights plus loading/conversion staging.
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=1_073_741_824))]
        max_weight_bytes: Option<u64>,
        /// Streamed mode only: maximum planned detached KV bytes at the promised context length.
        #[arg(long, value_parser = clap::value_parser!(u64).range(1..=1_073_741_824))]
        max_kv_bytes: Option<u64>,
        /// Streamed mode only: projection rows loaded per tile.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=4_096))]
        tile_rows: Option<u32>,
        /// Compare each cached result with a full forward outside timed regions.
        #[arg(long)]
        verify_cache: bool,
        /// Emit phase timing and logical-memory diagnostics on stderr; streamed mode also adds phase profiles to JSON.
        #[arg(short, long, visible_alias = "debug")]
        verbose: bool,
        /// Include selected-token natural-log probabilities in the JSON report; does not change greedy selection.
        #[arg(long)]
        logprobs: bool,
        /// Enable explicit reproducible categorical sampling; requires --temperature and --seed.
        #[arg(long, requires_all = ["temperature", "seed"])]
        sample: bool,
        /// Positive finite categorical-sampling temperature; requires --sample.
        #[arg(long, requires = "sample", value_parser = parse_temperature)]
        temperature: Option<f64>,
        /// Deterministic categorical-sampling seed; requires --sample.
        #[arg(long, requires = "sample")]
        seed: Option<u64>,
        /// Render a bounded stderr summary; decoded text for --prompt or schema output, otherwise raw IDs.
        #[arg(long)]
        preview: bool,
        /// Constrain generated JSON using a local schema (32 KiB maximum).
        #[cfg(feature = "structured-output")]
        #[arg(long)]
        json_schema: Option<PathBuf>,
        /// Constrain generated JSON using inline JSON schema text (32 KiB maximum).
        #[cfg(feature = "structured-output")]
        #[arg(long, conflicts_with = "json_schema")]
        json_schema_inline: Option<String>,
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
        /// Omit resident controls for process-memory measurement; emits per-step logits, not parity.
        #[arg(long)]
        candidate_only: bool,
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
    /// Validate one real DeepSeek/MLX safetensors shard header without reading payloads.
    InspectV41Shard {
        /// Path to a safetensors shard.
        shard: PathBuf,
    },
    /// Decode one bounded MLX `DeepSeek` embedding row from a real shard.
    InspectV41EmbeddingRow {
        /// Path to the embedding shard.
        shard: PathBuf,
        /// Embedding row/token ID.
        #[arg(long, default_value_t = 0)]
        row: usize,
        /// Decode every row for the bounded layer-zero `wq_a` matrix.
        #[arg(long)]
        all_rows: bool,
        /// Row family to decode: `embedding`, `layer0-wq-a`, `layer0-q-chain`, or `layer0-kv-row`.
        #[arg(long, default_value = "embedding")]
        kind: String,
        /// Optional embedding shard to apply to a full layer-zero projection.
        #[arg(long)]
        input_shard: Option<PathBuf>,
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
        Command::Chat {
            model,
            prompt,
            max_tokens,
            json,
            context_tokens,
            kv_budget_mib,
        } => chat_cli::chat(
            &model,
            prompt,
            max_tokens,
            json,
            resident_chat_limits(context_tokens, kv_budget_mib),
        ),
        #[cfg(feature = "metal")]
        Command::Agent {
            model,
            workspace,
            prompt,
            max_tokens,
            max_turns,
            context_tokens,
            kv_budget_mib,
            json,
        } => chat_cli::agent(
            &model,
            &workspace,
            prompt,
            max_tokens,
            max_turns,
            resident_chat_limits(context_tokens, kv_budget_mib),
            json,
        ),
        #[cfg(feature = "metal")]
        Command::Serve {
            model,
            model_id,
            listen,
            context_tokens,
            kv_budget_mib,
            generation_timeout_ms,
        } => responses::serve(
            &model,
            &model_id,
            listen,
            resident_chat_limits(context_tokens, kv_budget_mib),
            Duration::from_millis(u64::from(generation_timeout_ms)),
        ),
        #[cfg(feature = "metal")]
        Command::CheckV41RotaryMetal { fixture, repeats } => v41_rotary::run(&fixture, repeats),
        #[cfg(feature = "metal")]
        Command::CheckV41IndexerMetal { fixture, repeats } => v41_indexer::run(&fixture, repeats),
        #[cfg(feature = "metal")]
        Command::GenerateQwenMetal {
            model,
            input_ids,
            prompt,
            max_tokens,
            memory_mode,
            max_weight_bytes,
            max_kv_bytes,
            tile_rows,
            verify_cache,
            verbose,
            logprobs,
            sample,
            temperature,
            seed,
            preview,
            #[cfg(feature = "structured-output")]
            json_schema,
            #[cfg(feature = "structured-output")]
            json_schema_inline,
        } => {
            let max_tokens = generation_max_tokens(memory_mode, max_tokens);
            let (input_ids, tokenizer) = match generation_input(input_ids, prompt) {
                Ok(GenerationInput::RawIds(input_ids)) => (input_ids, None),
                Ok(GenerationInput::Prompt(prompt)) => {
                    let tokenizer = match qwen_tokenizer::QwenTokenizer::load(&model) {
                        Ok(tokenizer) => tokenizer,
                        Err(error) => {
                            eprintln!("Qwen prompt failed: {error}");
                            return ExitCode::FAILURE;
                        }
                    };
                    let input_ids = match tokenizer.encode_prompt(&prompt) {
                        Ok(input_ids) => input_ids,
                        Err(error) => {
                            eprintln!("Qwen prompt failed: {error}");
                            return ExitCode::FAILURE;
                        }
                    };
                    (input_ids, Some(tokenizer))
                }
                Err(error) => {
                    eprintln!("Qwen prompt failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let sampling = match sampling_configuration(sample, temperature, seed) {
                Ok(sampling) => sampling,
                Err(error) => {
                    eprintln!("Qwen generation failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            qwen_forward::generate(
                &model,
                &input_ids,
                max_tokens,
                verify_cache,
                qwen_forward::GenerationMemoryConfig {
                    mode: match memory_mode {
                        GenerationMemoryMode::Resident => {
                            qwen_forward::GenerationMemoryMode::Resident
                        }
                        GenerationMemoryMode::Streamed => {
                            qwen_forward::GenerationMemoryMode::Streamed
                        }
                    },
                    max_weight_bytes,
                    max_kv_bytes,
                    // Metal generation is Apple-Silicon only; the CLI parser has
                    // already bounded this `u32` to 1..=4096.
                    tile_rows: tile_rows.map(|rows| rows as usize),
                },
                qwen_forward::GenerationDiagnostics {
                    tokenizer: tokenizer.as_ref(),
                    verbose,
                    logprobs,
                    preview,
                    sampling,
                },
                #[cfg(feature = "structured-output")]
                match (json_schema.as_deref(), json_schema_inline.as_deref()) {
                    (Some(path), None) => Some(qwen_constraints::SchemaSource::File(path)),
                    (None, Some(schema)) => Some(qwen_constraints::SchemaSource::Inline(schema)),
                    (None, None) => None,
                    (Some(_), Some(_)) => {
                        eprintln!(
                            "Qwen generation failed: --json-schema and --json-schema-inline cannot be used together"
                        );
                        return ExitCode::FAILURE;
                    }
                },
            )
        }
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
        Command::InspectV41Shard { shard } => inspect_v41_shard(&shard),
        Command::InspectV41EmbeddingRow {
            shard,
            row,
            kind,
            all_rows,
            input_shard,
        } => inspect_v41_embedding_row(&shard, row, &kind, all_rows, input_shard.as_ref()),
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
            candidate_only,
        } => check_qwen_stream_cache_metal(
            &model,
            &input_ids,
            &decode_ids,
            tile_rows,
            max_weight_bytes,
            max_kv_bytes,
            candidate_only,
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
    candidate_only: bool,
) -> ExitCode {
    let Ok(tile_rows) = usize::try_from(tile_rows) else {
        eprintln!("Qwen streamed cache check failed: tile rows do not fit usize");
        return ExitCode::FAILURE;
    };
    if candidate_only {
        return match qwen::metal::run_streamed_cached_candidate(
            model,
            input_ids,
            decode_ids,
            max_weight_bytes,
            max_kv_bytes,
            tile_rows,
        ) {
            Ok(report) => print_stream_report(report),
            Err(error) => {
                eprintln!("Qwen streamed cache candidate failed: {error}");
                ExitCode::FAILURE
            }
        };
    }
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
    match V41SafetensorsIndex::parse(&json) {
        Ok(index) => {
            println!("V4.1 safetensors index");
            println!("tensors: {}", index.tensor_count());
            println!("shards: {}", index.shard_paths().len());
            println!("declared total bytes: {}", index.total_bytes());
            println!("next gate: resolve individual shard sizes before download planning");
            ExitCode::SUCCESS
        }
        Err(strict_error) => match MlxSafetensorsIndex::parse(&json) {
            Ok(index) => {
                println!("MLX safetensors index");
                println!("tensors: {}", index.tensor_count());
                println!("shards: {}", index.shard_paths().len());
                println!("scope: tensor placement only; no native DeepSeek execution");
                ExitCode::SUCCESS
            }
            Err(mlx_error) => {
                eprintln!(
                    "{} is not a supported V4.1 or MLX safetensors index: strict={strict_error}; mlx={mlx_error}",
                    index.display()
                );
                ExitCode::FAILURE
            }
        },
    }
}

fn inspect_v41_shard(shard: &PathBuf) -> ExitCode {
    let file_bytes = match fs::metadata(shard) {
        Ok(metadata) => metadata.len(),
        Err(error) => {
            eprintln!("could not stat {}: {error}", shard.display());
            return ExitCode::FAILURE;
        }
    };
    let mut file = match fs::File::open(shard) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("could not open {}: {error}", shard.display());
            return ExitCode::FAILURE;
        }
    };
    let mut prefix = [0_u8; 8];
    if let Err(error) = file.read_exact(&mut prefix) {
        eprintln!("could not read safetensors prefix: {error}");
        return ExitCode::FAILURE;
    }
    let header_bytes = u64::from_le_bytes(prefix);
    let header_len = match usize::try_from(header_bytes) {
        Ok(length) if length <= 100 * 1024 * 1024 => length,
        _ => {
            eprintln!("safetensors header exceeds the bounded 100 MiB inspection limit");
            return ExitCode::FAILURE;
        }
    };
    let mut prefixed_header = Vec::with_capacity(8 + header_len);
    prefixed_header.extend_from_slice(&prefix);
    prefixed_header.resize(8 + header_len, 0);
    if let Err(error) = file.read_exact(&mut prefixed_header[8..]) {
        eprintln!("could not read safetensors header: {error}");
        return ExitCode::FAILURE;
    }
    match deepseek::V41SafetensorsHeader::parse_prefixed_header(&prefixed_header, file_bytes) {
        Ok(header) => {
            println!("DeepSeek safetensors shard header");
            println!("file bytes: {file_bytes}");
            println!("tensors: {}", header.tensors().len());
            println!(
                "embedding present: {}",
                header.tensor("model.embed_tokens").is_some()
            );
            println!("scope: header validation only; tensor payloads were not read");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!(
                "{} is not a supported safetensors shard: {error}",
                shard.display()
            );
            ExitCode::FAILURE
        }
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "bounded CLI artifact inspection keeps its I/O phases explicit"
)]
fn inspect_v41_embedding_row(
    shard: &PathBuf,
    row: usize,
    kind: &str,
    all_rows: bool,
    input_shard: Option<&PathBuf>,
) -> ExitCode {
    let file_bytes = match fs::metadata(shard) {
        Ok(metadata) => metadata.len(),
        Err(error) => {
            eprintln!("could not stat {}: {error}", shard.display());
            return ExitCode::FAILURE;
        }
    };
    let mut file = match fs::File::open(shard) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("could not open {}: {error}", shard.display());
            return ExitCode::FAILURE;
        }
    };
    let mut prefix = [0_u8; 8];
    if file.read_exact(&mut prefix).is_err() {
        eprintln!("could not read safetensors prefix");
        return ExitCode::FAILURE;
    }
    let header_bytes = u64::from_le_bytes(prefix);
    let Ok(header_len) = usize::try_from(header_bytes) else {
        eprintln!("safetensors header length overflows usize");
        return ExitCode::FAILURE;
    };
    if header_len > 100 * 1024 * 1024 {
        eprintln!("safetensors header exceeds the bounded 100 MiB inspection limit");
        return ExitCode::FAILURE;
    }
    let mut prefixed_header = vec![0_u8; 8 + header_len];
    prefixed_header[..8].copy_from_slice(&prefix);
    if file.read_exact(&mut prefixed_header[8..]).is_err() {
        eprintln!("could not read safetensors header");
        return ExitCode::FAILURE;
    }
    let header =
        match deepseek::V41SafetensorsHeader::parse_prefixed_header(&prefixed_header, file_bytes) {
            Ok(header) => header,
            Err(error) => {
                eprintln!("unsupported safetensors shard: {error}");
                return ExitCode::FAILURE;
            }
        };
    #[cfg(feature = "metal")]
    if kind == "layer0-kv-row" {
        const KV_RANK: usize = 512;
        const HIDDEN: usize = 4096;
        let row = row.min(KV_RANK.saturating_sub(1));
        let values = match read_affine_row_from_shard(
            shard,
            &header,
            "model.layers.0.attn.wkv.weight",
            "model.layers.0.attn.wkv.scales",
            "model.layers.0.attn.wkv.biases",
            row,
            HIDDEN,
            6,
            128,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("wkv decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let kv_norm = match read_bf16_tensor_from_shard(
            shard,
            &header,
            "model.layers.0.attn.kv_norm.weight",
            KV_RANK,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("kv_norm decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let checksum = values.iter().fold(0_u64, |hash, value| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(u64::from(value.to_bits()))
        });
        let norm_checksum = kv_norm.iter().fold(0_u64, |hash, value| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(u64::from(value.to_bits()))
        });
        println!("DeepSeek native KV projection row");
        println!("row: {row}");
        println!("wkv_width: {}", values.len());
        println!("wkv_checksum: {checksum:016x}");
        println!("kv_norm_width: {}", kv_norm.len());
        println!("kv_norm_checksum: {norm_checksum:016x}");
        println!("quantization: bits=6 group=128");
        let metal =
            match deepseek::checkpoint::mlx::apply_affine_matrix_mlx(&values, 1, HIDDEN, &values) {
                Ok(output) => {
                    let value = output.as_slice::<f32>()[0];
                    println!("metal_eval: passed");
                    println!("self_dot_checksum: {:016x}", u64::from(value.to_bits()));
                    true
                }
                Err(error) => {
                    eprintln!("wkv Metal projection failed: {error}");
                    false
                }
            };
        println!("scope: layer-zero wkv row and learned kv_norm; self-dot device smoke={metal}");
        if !metal {
            return ExitCode::FAILURE;
        }
        return ExitCode::SUCCESS;
    }
    #[cfg(feature = "metal")]
    if kind == "layer0-q-chain" {
        const HEADS: usize = 64;
        const HEAD_DIMENSION: usize = 512;
        const Q_RANK: usize = 1024;
        let Some(input_shard) = input_shard else {
            eprintln!("layer0-q-chain requires --input-shard");
            return ExitCode::FAILURE;
        };
        let input_bytes = match fs::metadata(input_shard) {
            Ok(metadata) => metadata.len(),
            Err(error) => {
                eprintln!("could not stat {}: {error}", input_shard.display());
                return ExitCode::FAILURE;
            }
        };
        let mut input_file = match fs::File::open(input_shard) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("could not open {}: {error}", input_shard.display());
                return ExitCode::FAILURE;
            }
        };
        let mut prefix = [0_u8; 8];
        if input_file.read_exact(&mut prefix).is_err() {
            eprintln!("could not read input safetensors prefix");
            return ExitCode::FAILURE;
        }
        let header_len = usize::try_from(u64::from_le_bytes(prefix)).unwrap_or(usize::MAX);
        if header_len > 100 * 1024 * 1024 {
            eprintln!("input safetensors header exceeds the bounded limit");
            return ExitCode::FAILURE;
        }
        let mut input_header_bytes = vec![0_u8; 8 + header_len];
        input_header_bytes[..8].copy_from_slice(&prefix);
        if input_file.read_exact(&mut input_header_bytes[8..]).is_err() {
            eprintln!("could not read input safetensors header");
            return ExitCode::FAILURE;
        }
        let input_header = match deepseek::V41SafetensorsHeader::parse_prefixed_header(
            &input_header_bytes,
            input_bytes,
        ) {
            Ok(header) => header,
            Err(error) => {
                eprintln!("unsupported input safetensors shard: {error}");
                return ExitCode::FAILURE;
            }
        };
        let embedding = match read_affine_row_from_shard(
            input_shard,
            &input_header,
            "model.embed_tokens.weight",
            "model.embed_tokens.scales",
            "model.embed_tokens.biases",
            0,
            4096,
            8,
            64,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("embedding decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let mut wq_a = Vec::with_capacity(1024 * 4096);
        for row in 0..1024 {
            match read_affine_row_from_shard(
                shard,
                &header,
                "model.layers.0.attn.wq_a.weight",
                "model.layers.0.attn.wq_a.scales",
                "model.layers.0.attn.wq_a.biases",
                row,
                4096,
                6,
                128,
            ) {
                Ok(values) => wq_a.extend(values),
                Err(error) => {
                    eprintln!("wq_a decode failed: {error}");
                    return ExitCode::FAILURE;
                }
            }
        }
        let q_a =
            match deepseek::checkpoint::mlx::apply_affine_matrix_mlx(&wq_a, 1024, 4096, &embedding)
            {
                Ok(output) => output,
                Err(error) => {
                    eprintln!("wq_a Metal projection failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
        let q_a_values = q_a.as_slice::<f32>();
        let norm =
            (q_a_values.iter().map(|value| value * value).sum::<f32>() / 1024.0 + 1e-6).sqrt();
        let q_norm = match read_bf16_tensor_from_shard(
            shard,
            &header,
            "model.layers.0.attn.q_norm.weight",
            1024,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("q_norm decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let normalized = q_a_values
            .iter()
            .zip(q_norm)
            .map(|(value, weight)| value / norm * weight)
            .collect::<Vec<_>>();
        let mut q_b_outputs = Vec::with_capacity(HEADS * HEAD_DIMENSION);
        for head in 0..HEADS {
            let wq_b = match read_affine_rows_from_shard(
                shard,
                &header,
                "model.layers.0.attn.wq_b.weight",
                "model.layers.0.attn.wq_b.scales",
                "model.layers.0.attn.wq_b.biases",
                head * HEAD_DIMENSION,
                HEAD_DIMENSION,
                Q_RANK,
                6,
                128,
            ) {
                Ok(values) => values,
                Err(error) => {
                    eprintln!("wq_b head {head} decode failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let q_b = match deepseek::checkpoint::mlx::apply_affine_matrix_mlx(
                &wq_b,
                HEAD_DIMENSION,
                Q_RANK,
                &normalized,
            ) {
                Ok(output) => output,
                Err(error) => {
                    eprintln!("wq_b head {head} Metal projection failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            q_b_outputs.extend_from_slice(q_b.as_slice::<f32>());
        }
        let checksum = q_b_outputs.iter().fold(0_u64, |hash, value| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(u64::from(value.to_bits()))
        });
        let mut q_b_norm = Vec::with_capacity(q_b_outputs.len());
        for head in q_b_outputs.chunks_exact(HEAD_DIMENSION) {
            let norm =
                (head.iter().map(|value| value * value).sum::<f32>() / 512.0_f32 + 1e-6).sqrt();
            q_b_norm.extend(head.iter().map(|value| *value / norm));
        }
        let normalized_checksum = q_b_norm.iter().fold(0_u64, |hash, value| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(u64::from(value.to_bits()))
        });
        println!("DeepSeek native Q chain");
        println!("q_a_width: {}", q_a_values.len());
        println!("q_b_width: {}", q_b_outputs.len());
        println!("q_b_checksum: {checksum:016x}");
        println!("q_b_norm_checksum: {normalized_checksum:016x}");
        println!("metal_eval: passed");
        println!("q_b_heads: {HEADS}");
        println!("scope: token-0 embedding through wq_a/q_norm/all wq_b heads");
        return ExitCode::SUCCESS;
    }
    if kind == "layer0-hc-mix" {
        let Some(input_shard) = input_shard else {
            eprintln!("layer0-hc-mix requires --input-shard");
            return ExitCode::FAILURE;
        };
        let input_bytes = match fs::metadata(input_shard) {
            Ok(metadata) => metadata.len(),
            Err(error) => {
                eprintln!("could not stat {}: {error}", input_shard.display());
                return ExitCode::FAILURE;
            }
        };
        let mut input_file = match fs::File::open(input_shard) {
            Ok(file) => file,
            Err(error) => {
                eprintln!("could not open {}: {error}", input_shard.display());
                return ExitCode::FAILURE;
            }
        };
        let mut input_prefix = [0_u8; 8];
        if input_file.read_exact(&mut input_prefix).is_err() {
            eprintln!("could not read input safetensors prefix");
            return ExitCode::FAILURE;
        }
        let input_header_len =
            usize::try_from(u64::from_le_bytes(input_prefix)).unwrap_or(usize::MAX);
        if input_header_len > 100 * 1024 * 1024 {
            eprintln!("input safetensors header exceeds the bounded limit");
            return ExitCode::FAILURE;
        }
        let mut input_header_bytes = vec![0_u8; 8 + input_header_len];
        input_header_bytes[..8].copy_from_slice(&input_prefix);
        if input_file.read_exact(&mut input_header_bytes[8..]).is_err() {
            eprintln!("could not read input safetensors header");
            return ExitCode::FAILURE;
        }
        let input_header = match deepseek::V41SafetensorsHeader::parse_prefixed_header(
            &input_header_bytes,
            input_bytes,
        ) {
            Ok(header) => header,
            Err(error) => {
                eprintln!("unsupported input safetensors shard: {error}");
                return ExitCode::FAILURE;
            }
        };
        let hidden = match read_affine_row_from_shard(
            input_shard,
            &input_header,
            "model.embed_tokens.weight",
            "model.embed_tokens.scales",
            "model.embed_tokens.biases",
            0,
            4096,
            8,
            64,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("input embedding decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let fn_matrix = match read_f32_tensor_from_shard(
            shard,
            &header,
            "model.layers.0.attn_hc.fn",
            24,
            16_384,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("hyper-connection fn decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let base = match read_f32_tensor_from_shard(
            shard,
            &header,
            "model.layers.0.attn_hc.base",
            1,
            24,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("hyper-connection base decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let scale = match read_f32_tensor_from_shard(
            shard,
            &header,
            "model.layers.0.attn_hc.scale",
            1,
            3,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("hyper-connection scale decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let scale: [f32; 3] = scale.try_into().expect("three HC scales");
        let coefficients =
            match mix_hc_coefficients(&fn_matrix, &base, &scale, &hidden, 4, 1e-6, 20) {
                Ok(coefficients) => coefficients,
                Err(error) => {
                    eprintln!("hyper-connection coefficient mix failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
        let checksum = coefficients
            .pre()
            .iter()
            .chain(coefficients.post())
            .chain(coefficients.comb().iter())
            .fold(0_u64, |hash, value| {
                hash.wrapping_mul(1_099_511_628_211)
                    .wrapping_add(u64::from(value.to_bits()))
            });
        let collapsed = match collapse_hc_hidden(&hidden, &coefficients) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("hyper-connection collapse failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let collapsed_checksum = collapsed.iter().fold(0_u64, |hash, value| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(u64::from(value.to_bits()))
        });
        println!("DeepSeek layer-zero HC mix");
        println!("copies: {}", coefficients.copies());
        println!("coefficients_checksum: {checksum:016x}");
        println!("collapsed_hidden_width: {}", collapsed.len());
        println!("collapsed_hidden_checksum: {collapsed_checksum:016x}");
        println!("scope: real shard parameters and token-0 embedding; no block execution");
        return ExitCode::SUCCESS;
    }
    if kind == "layer0-hc-fn" {
        let values = match read_f32_tensor_from_shard(
            shard,
            &header,
            "model.layers.0.attn_hc.fn",
            24,
            16_384,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("hyper-connection tensor decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        let checksum = values.iter().fold(0_u64, |hash, value| {
            hash.wrapping_mul(1_099_511_628_211)
                .wrapping_add(u64::from(value.to_bits()))
        });
        #[cfg(feature = "metal")]
        if let Err(error) = deepseek::checkpoint::mlx::decode_affine_row_mlx(&values) {
            eprintln!("MLX hyper-connection evaluation failed: {error}");
            return ExitCode::FAILURE;
        }
        println!("DeepSeek MLX hyper-connection tensor");
        println!("kind: {kind}");
        println!("rows: 24");
        println!("width: 16384");
        println!("fp32_checksum: {checksum:016x}");
        #[cfg(feature = "metal")]
        println!("metal_eval: passed");
        println!("scope: hyper-connection parameter decode; no model execution");
        return ExitCode::SUCCESS;
    }
    let (weight, scales, biases, width, group_size) = match kind {
        "embedding" => (
            "model.embed_tokens.weight",
            "model.embed_tokens.scales",
            "model.embed_tokens.biases",
            4096,
            64,
        ),
        "layer0-wq-a" => (
            "model.layers.0.attn.wq_a.weight",
            "model.layers.0.attn.wq_a.scales",
            "model.layers.0.attn.wq_a.biases",
            4096,
            128,
        ),
        "layer0-wq-b-head0" => (
            "model.layers.0.attn.wq_b.weight",
            "model.layers.0.attn.wq_b.scales",
            "model.layers.0.attn.wq_b.biases",
            1024,
            128,
        ),
        _ => {
            eprintln!("unknown row kind {kind:?}; expected embedding or layer0-wq-a");
            return ExitCode::FAILURE;
        }
    };
    let row_count = if all_rows {
        if !matches!(kind, "layer0-wq-a" | "layer0-wq-b-head0") {
            eprintln!("--all-rows is only supported for layer0-wq-a or layer0-wq-b-head0");
            return ExitCode::FAILURE;
        }
        if kind == "layer0-wq-b-head0" {
            512
        } else {
            1024
        }
    } else {
        1
    };
    let mut matrix = Vec::new();
    for current_row in 0..row_count {
        let decoded = match read_affine_row_from_shard(
            shard,
            &header,
            weight,
            scales,
            biases,
            if all_rows { current_row } else { row },
            width,
            if matches!(kind, "layer0-wq-a" | "layer0-wq-b-head0") {
                6
            } else {
                8
            },
            group_size,
        ) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("affine row decode failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        matrix.extend(decoded);
        if !all_rows {
            break;
        }
    }
    let checksum = matrix.iter().fold(0_u64, |hash, value| {
        hash.wrapping_mul(1_099_511_628_211)
            .wrapping_add(u64::from(value.to_bits()))
    });
    #[cfg(feature = "metal")]
    if let Err(error) = deepseek::checkpoint::mlx::decode_affine_row_mlx(&matrix) {
        eprintln!("MLX affine evaluation failed: {error}");
        return ExitCode::FAILURE;
    }
    println!("DeepSeek MLX affine tensor");
    println!("kind: {kind}");
    println!("rows: {row_count}");
    println!("width: {width}");
    println!("fp32_checksum: {checksum:016x}");
    #[cfg(feature = "metal")]
    println!("metal_eval: passed");
    #[cfg(feature = "metal")]
    if let Some(input_shard) = input_shard {
        if kind != "layer0-wq-a" || !all_rows {
            eprintln!("--input-shard requires --kind layer0-wq-a --all-rows");
            return ExitCode::FAILURE;
        }
        #[cfg(feature = "metal")]
        {
            let input_bytes = match fs::metadata(input_shard) {
                Ok(metadata) => metadata.len(),
                Err(error) => {
                    eprintln!("could not stat {}: {error}", input_shard.display());
                    return ExitCode::FAILURE;
                }
            };
            let mut input_file = match fs::File::open(input_shard) {
                Ok(file) => file,
                Err(error) => {
                    eprintln!("could not open {}: {error}", input_shard.display());
                    return ExitCode::FAILURE;
                }
            };
            let mut input_prefix = [0_u8; 8];
            if input_file.read_exact(&mut input_prefix).is_err() {
                eprintln!("could not read input safetensors prefix");
                return ExitCode::FAILURE;
            }
            let input_header_len =
                usize::try_from(u64::from_le_bytes(input_prefix)).unwrap_or(usize::MAX);
            if input_header_len > 100 * 1024 * 1024 {
                eprintln!("input safetensors header exceeds the bounded limit");
                return ExitCode::FAILURE;
            }
            let mut input_header_bytes = vec![0_u8; 8 + input_header_len];
            input_header_bytes[..8].copy_from_slice(&input_prefix);
            if input_file.read_exact(&mut input_header_bytes[8..]).is_err() {
                eprintln!("could not read input safetensors header");
                return ExitCode::FAILURE;
            }
            let input_header = match deepseek::V41SafetensorsHeader::parse_prefixed_header(
                &input_header_bytes,
                input_bytes,
            ) {
                Ok(header) => header,
                Err(error) => {
                    eprintln!("unsupported input safetensors shard: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let input = match read_affine_row_from_shard(
                input_shard,
                &input_header,
                "model.embed_tokens.weight",
                "model.embed_tokens.scales",
                "model.embed_tokens.biases",
                0,
                4096,
                8,
                64,
            ) {
                Ok(values) => values,
                Err(error) => {
                    eprintln!("input embedding decode failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let projected = match deepseek::checkpoint::mlx::apply_affine_matrix_mlx(
                &matrix, 1024, 4096, &input,
            ) {
                Ok(output) => output,
                Err(error) => {
                    eprintln!("native embedding projection failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let projected_checksum =
                projected
                    .as_slice::<f32>()
                    .iter()
                    .fold(0_u64, |hash, value| {
                        hash.wrapping_mul(1_099_511_628_211)
                            .wrapping_add(u64::from(value.to_bits()))
                    });
            println!("projected_width: 1024");
            println!("projected_checksum: {projected_checksum:016x}");
            println!("projection_metal_eval: passed");
        }
        #[cfg(not(feature = "metal"))]
        {
            eprintln!("--input-shard requires a Metal build");
            return ExitCode::FAILURE;
        }
    }
    println!("scope: bounded affine tensor decoded; no model execution");
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
    fn qwen_generation_defaults_and_streamed_limits_are_explicit() {
        let default = Cli::try_parse_from(["mx", "gen", "--model", "model"])
            .expect("generation alias with defaults");
        assert!(matches!(
            default.command,
            super::Command::GenerateQwenMetal {
                input_ids: None,
                prompt: None,
                max_tokens: None,
                memory_mode: super::GenerationMemoryMode::Resident,
                max_weight_bytes: None,
                max_kv_bytes: None,
                tile_rows: None,
                verify_cache: false,
                verbose: false,
                logprobs: false,
                sample: false,
                temperature: None,
                seed: None,
                preview: false,
                ..
            }
        ));
        assert!(matches!(
            super::generation_input(None, None),
            Ok(super::GenerationInput::RawIds(input_ids)) if input_ids == [1, 2, 3]
        ));

        let prompt =
            Cli::try_parse_from(["mx", "gen", "--model", "model", "--prompt", "plain prompt"])
                .expect("plain prompt generation");
        assert!(matches!(
            prompt.command,
            super::Command::GenerateQwenMetal {
                input_ids: None,
                prompt: Some(prompt),
                ..
            } if prompt == "plain prompt"
        ));
        assert!(
            Cli::try_parse_from([
                "mx",
                "gen",
                "--model",
                "model",
                "--input-ids",
                "1,2,3",
                "--prompt",
                "plain prompt",
            ])
            .is_err()
        );

        let streamed = Cli::try_parse_from([
            "mx",
            "gen",
            "--model",
            "model",
            "--memory-mode",
            "streamed",
            "--max-weight-bytes",
            "81798144",
            "--max-kv-bytes",
            "1376256",
            "--tile-rows",
            "1024",
        ])
        .expect("explicit streamed generation limits");
        assert!(matches!(
            streamed.command,
            super::Command::GenerateQwenMetal {
                memory_mode: super::GenerationMemoryMode::Streamed,
                max_weight_bytes: Some(81_798_144),
                max_kv_bytes: Some(1_376_256),
                tile_rows: Some(1_024),
                ..
            }
        ));
        assert_eq!(
            super::generation_max_tokens(super::GenerationMemoryMode::Resident, None),
            32
        );
        assert_eq!(
            super::generation_max_tokens(super::GenerationMemoryMode::Streamed, None),
            4
        );
        assert_eq!(
            super::generation_max_tokens(super::GenerationMemoryMode::Streamed, Some(29)),
            29
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn serve_generation_timeout_is_bounded_and_defaults_to_one_minute() {
        let default =
            Cli::try_parse_from(["mx", "serve", "--model", "model"]).expect("serve defaults");
        assert!(matches!(
            default.command,
            super::Command::Serve {
                generation_timeout_ms: 60_000,
                ..
            }
        ));
        for value in ["0", "120001"] {
            assert!(
                Cli::try_parse_from([
                    "mx",
                    "serve",
                    "--model",
                    "model",
                    "--generation-timeout-ms",
                    value,
                ])
                .is_err()
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qwen_sampling_and_diagnostic_flags_have_explicit_contracts() {
        let scores = Cli::try_parse_from(["mx", "gen", "--model", "model", "--logprobs"])
            .expect("opt-in log probabilities");
        assert!(matches!(
            scores.command,
            super::Command::GenerateQwenMetal { logprobs: true, .. }
        ));
        let sampled = Cli::try_parse_from([
            "mx",
            "gen",
            "--model",
            "model",
            "--sample",
            "--temperature",
            "0.7",
            "--seed",
            "9",
        ])
        .expect("complete sampled policy");
        assert!(matches!(
            sampled.command,
            super::Command::GenerateQwenMetal {
                sample: true,
                temperature: Some(temperature),
                seed: Some(9),
                ..
            } if (temperature - 0.7).abs() < f64::EPSILON
        ));
        for arguments in [
            vec!["mx", "gen", "--model", "model", "--temperature", "0.7"],
            vec!["mx", "gen", "--model", "model", "--seed", "9"],
            vec!["mx", "gen", "--model", "model", "--sample", "--seed", "9"],
            vec![
                "mx",
                "gen",
                "--model",
                "model",
                "--sample",
                "--temperature",
                "0.7",
            ],
            vec![
                "mx",
                "gen",
                "--model",
                "model",
                "--sample",
                "--temperature",
                "0",
                "--seed",
                "9",
            ],
        ] {
            assert!(
                Cli::try_parse_from(arguments).is_err(),
                "incomplete or invalid sampled-policy arguments must fail"
            );
        }
        for flag in ["--verbose", "--debug", "-v"] {
            let cli = Cli::try_parse_from(["mx", "gen", "--model", "model", flag])
                .expect("generation diagnostic verbosity spelling");
            assert!(matches!(
                cli.command,
                super::Command::GenerateQwenMetal { verbose: true, .. }
            ));
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn qwen_generation_help_describes_the_diagnostic_scope() {
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("generate-qwen-metal")
            .expect("generation subcommand")
            .render_long_help()
            .to_string();
        let normalized_help = help.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(normalized_help.contains("streamed mode also adds phase profiles to JSON"));
        assert!(normalized_help.contains("does not change greedy selection"));
        assert!(normalized_help.contains("without a chat template or special tokens"));
        assert!(
            normalized_help
                .contains("decoded text for --prompt or schema output, otherwise raw IDs")
        );
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
            super::Command::GenerateQwenMetal { json_schema: Some(path), json_schema_inline: None, verbose: true, preview: true, .. }
            if path == std::path::Path::new("schema.json")
        ));

        let inline = Cli::try_parse_from([
            "mx",
            "gen",
            "--model",
            "model",
            "--json-schema-inline",
            r#"{"type":"object"}"#,
        ])
        .expect("inline constrained generation arguments");
        assert!(matches!(inline.command,
            super::Command::GenerateQwenMetal { json_schema: None, json_schema_inline: Some(schema), .. }
            if schema == r#"{"type":"object"}"#
        ));
        assert!(
            Cli::try_parse_from([
                "mx",
                "gen",
                "--model",
                "model",
                "--json-schema",
                "schema.json",
                "--json-schema-inline",
                r#"{"type":"object"}"#,
            ])
            .is_err()
        );
    }

    #[cfg(all(feature = "metal", feature = "structured-output"))]
    #[test]
    fn sampled_generation_accepts_a_json_schema_before_runtime() {
        let cli = Cli::try_parse_from([
            "mx",
            "gen",
            "--model",
            "model",
            "--sample",
            "--temperature",
            "1",
            "--seed",
            "3",
            "--json-schema",
            "schema.json",
        ])
        .expect("sampled constrained generation arguments");
        assert!(matches!(
            cli.command,
            super::Command::GenerateQwenMetal {
                sample: true,
                json_schema: Some(path),
                ..
            } if path == std::path::Path::new("schema.json")
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
            max_kv_bytes: 1_048_576, tile_rows: 1024, candidate_only: false, ..
        } if input_ids == [9707, 11] && decode_ids == [1879, 151_935])
        );
        assert!(Cli::try_parse_from(["mx", "check-qwen-stream-cache-metal"]).is_err());
        let candidate = Cli::try_parse_from([
            "mx",
            "check-qwen-stream-cache-metal",
            "--model",
            "model",
            "--candidate-only",
        ])
        .expect("explicit candidate-only cached stream");
        assert!(matches!(
            candidate.command,
            super::Command::CheckQwenStreamCacheMetal {
                candidate_only: true,
                ..
            }
        ));
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
        let help = root_help().split_whitespace().collect::<Vec<_>>().join(" ");

        assert!(help.contains("Inspect model files and run experimental Metal inference"));
        assert!(help.contains("loopback Responses serving require Metal"));
        assert!(help.contains("Inspect commands read model configuration or checkpoint headers"));
        assert!(help.contains("--features metal"));
        assert!(help.contains("Qwen forward uses raw token IDs"));
        assert!(help.contains("plain-text prompt for one sequence"));
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
