//! Command-line definitions for `mx` and `metallix`.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[cfg(feature = "metal")]
use crate::{decision_cli, julia_decisions};
use crate::{deepseek_reduced_cli, deepseek_selected_cli};

/// Invalid explicit context input; model-specific admission happens at load.
#[cfg(feature = "metal")]
#[derive(Debug)]
pub(crate) enum ExplicitContextError {
    Integer(std::num::ParseIntError),
    Empty,
    Shape { requested: u32 },
}

#[cfg(feature = "metal")]
impl std::fmt::Display for ExplicitContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Integer(error) => write!(f, "invalid explicit context: {error}"),
            Self::Empty => f.write_str("explicit context must contain at least one token"),
            Self::Shape { requested } => write!(
                f,
                "explicit context {requested} exceeds the MLX i32 shape limit"
            ),
        }
    }
}

#[cfg(feature = "metal")]
impl std::error::Error for ExplicitContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Integer(error) => Some(error),
            Self::Empty | Self::Shape { .. } => None,
        }
    }
}

/// Parse only the explicit token count, leaving model and memory caps to adapters.
#[cfg(feature = "metal")]
pub(crate) fn parse_context_tokens(value: &str) -> Result<u32, ExplicitContextError> {
    let tokens = value.parse().map_err(ExplicitContextError::Integer)?;
    if tokens == 0 {
        return Err(ExplicitContextError::Empty);
    }
    if i32::try_from(tokens).is_err() {
        return Err(ExplicitContextError::Shape { requested: tokens });
    }
    Ok(tokens)
}

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
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
    /// Write a Chrome/Perfetto JSON timeline of spans (build with --features timeline).
    ///
    /// `mx serve` children write `<stem>.<model-id>.<extension>` beside it.
    #[arg(long, global = true, value_name = "FILE")]
    pub(crate) trace_out: Option<PathBuf>,
    /// Record a Metal capture (`.gputrace`) of this command; needs `MTL_CAPTURE_ENABLED=1`.
    ///
    /// For `mx serve`, each child captures its first request to
    /// `<stem>.<model-id>.gputrace` beside this path.
    #[cfg(feature = "metal")]
    #[arg(long, global = true, value_name = "PATH")]
    pub(crate) gpu_capture: Option<PathBuf>,
}

/// Which Qwen weight residency contract a generation run uses.
///
/// This belongs to the Qwen diagnostic command rather than the engine: the
/// streamed executor has adapter-specific layout and cache semantics.
#[cfg(feature = "metal")]
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub(crate) enum GenerationMemoryMode {
    /// Load the complete checkpoint once and retain its MLX arrays.
    Resident,
    /// Read one layer at a time under explicit weight and KV budgets.
    Streamed,
}

#[cfg(feature = "metal")]
pub(crate) fn parse_temperature(value: &str) -> Result<f64, String> {
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

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Run a fixed five-block reduced `DeepSeek` model from a synthetic artifact.
    RunDeepseekReduced(deepseek_reduced_cli::ReducedArgs),
    /// Compare local pinned V4.1 layer-zero `MoE` weights against source captures.
    RunDeepseekSelected(deepseek_selected_cli::SelectedArgs),
    /// Score typed decision options directly with Qwen3, without generating text.
    #[cfg(feature = "metal")]
    Decide(decision_cli::DecisionArgs),
    /// Score typed decision options with a local Julia-1 checkpoint on the CPU.
    #[cfg(feature = "metal")]
    DecideJulia(julia_decisions::JuliaDecisionArgs),
    /// Acquire supported model artifacts through the Hugging Face CLI.
    Fetch {
        #[command(subcommand)]
        command: FetchCommand,
    },
    /// Inspect local model and checkpoint artifacts without loading weights.
    Inspect {
        #[command(subcommand)]
        command: InspectCommand,
    },
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
        /// Validated against the model and K/V budget before loading weights.
        #[arg(long, default_value_t = 2048, value_parser = parse_context_tokens)]
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
        /// Validated against the model and K/V budget before loading weights.
        #[arg(long, default_value_t = 2048, value_parser = parse_context_tokens)]
        context_tokens: u32,
        /// Logical resident K/V admission budget in MiB; not an MLX allocation limit.
        #[arg(long, default_value_t = 512, value_parser = clap::value_parser!(u32).range(1..=8192))]
        kv_budget_mib: u32,
        /// Emit a structured execution receipt. Task success is assessed by the caller.
        #[arg(long)]
        json: bool,
    },
    /// Serve registered models for generation, decisions, embeddings and reranking.
    #[cfg(feature = "metal")]
    Serve {
        /// Qwen checkpoint served as `--model-id`; shorthand for a one-entry registry.
        #[arg(long, required_unless_present_any = ["registry", "worker_entry"])]
        model: Option<PathBuf>,
        #[arg(long, default_value = "metallix-qwen3")]
        model_id: String,
        /// JSON manifest `{"models": [{"id", "kind", "path", "residency", "memory_mib"}]}`.
        ///
        /// `kind` is `qwen`, `julia`, `qwen_embedding`, `pplx_context` or `pplx_late`.
        /// `residency` is `resident` (default; started with the server) or `on_demand`
        /// (started on first request). `memory_mib` is the measured process footprint,
        /// required for every entry under `--memory-budget-mib`.
        #[arg(long)]
        registry: Option<PathBuf>,
        /// Total `memory_mib` that running models may declare; on-demand models
        /// are stopped, least recently used first, to stay within it.
        #[arg(long)]
        memory_budget_mib: Option<u64>,
        /// Internal: serve one registry entry (JSON) in this process as a child.
        #[arg(long, hide = true, conflicts_with_all = ["model", "registry"])]
        worker_entry: Option<String>,
        #[arg(long, default_value = "127.0.0.1:8321")]
        listen: std::net::SocketAddr,
        /// Total prompt plus output budget for each request.
        /// Validated against the model and K/V budget before loading weights.
        #[arg(long, default_value_t = 2048, value_parser = parse_context_tokens)]
        context_tokens: u32,
        /// Paged K/V pool per generating model, in MiB, allocated when the
        /// model loads. Defaults to the smaller of 4096 and a quarter of
        /// physical memory.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..=32768))]
        kv_budget_mib: Option<u32>,
        /// Per-model budget in MiB for prompt-prefix K/V reused across requests
        /// (shared system and tool preambles, earlier turns); separate from
        /// `--kv-budget-mib`. 0 disables reuse.
        #[arg(long, default_value_t = crate::chat_generation::DEFAULT_PREFIX_CACHE_MIB, value_parser = clap::value_parser!(u32).range(0..=65_536))]
        prefix_cache_mib: u32,
        /// Cooperative generation budget per request, in milliseconds.
        #[arg(long, default_value_t = 60_000, value_parser = clap::value_parser!(u32).range(1..=120_000))]
        generation_timeout_ms: u32,
        /// Requests that may wait, in arrival order, while a model is busy; 0
        /// refuses any overlap. A registry entry's `queue_depth` overrides it.
        #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u32).range(0..=1024))]
        queue_depth: u32,
        /// Longest a queued request waits for its model, in milliseconds. A
        /// registry entry's `queue_wait_ms` overrides it.
        #[arg(long, default_value_t = 60_000, value_parser = clap::value_parser!(u32).range(1..=600_000))]
        queue_wait_ms: u32,
        /// Sequences a generating model decodes together in one batched
        /// step; the paged K/V pool is `--kv-budget-mib`. The default, 1,
        /// serves one request at a time without the batching engine, which
        /// is faster for a single stream until its per-step cost is closed.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=256))]
        max_num_seqs: u32,
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
    #[command(
        visible_alias = "gen",
        after_help = "Shorthand: invoke this command as `mx gen`."
    )]
    GenerateQwenMetal {
        /// Local Qwen3 checkpoint directory containing config.json and weights.
        #[arg(long, value_name = "MODEL_DIR")]
        model: PathBuf,
        /// Comma-separated raw token IDs; defaults to 1,2,3 when --prompt is absent.
        #[arg(long, value_delimiter = ',', conflicts_with = "prompt")]
        input_ids: Option<Vec<i32>>,
        /// Plain-text prompt (at most 1 MiB), encoded by local tokenizer.json without a chat template or special tokens.
        #[arg(long, conflicts_with = "input_ids")]
        prompt: Option<String>,
        /// Render --prompt as one non-thinking user message using the checkpoint's local chat template.
        #[arg(long, requires = "prompt")]
        chat_template: bool,
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
        /// Generate resident schema-constrained `{"intervals":[{"start":number,"end":number}]}` candidates and accept only non-overlap.
        #[cfg(feature = "structured-output")]
        #[arg(long)]
        verify_schedule: bool,
        /// Exact integer-tick durations and time window (local JSON, at most 64 KiB).
        #[cfg(feature = "structured-output")]
        #[arg(long, requires = "verify_schedule")]
        schedule_requirements: Option<PathBuf>,
        /// Maximum complete candidates considered by --verify-schedule.
        #[cfg(feature = "structured-output")]
        #[arg(long, default_value_t = 4, requires = "verify_schedule", value_parser = clap::value_parser!(u32).range(1..=16))]
        max_attempts: u32,
        /// Cooperative wall-clock ceiling for the full verifier request, including fork and verification work.
        #[cfg(feature = "structured-output")]
        #[arg(long, default_value_t = 60_000, requires = "verify_schedule", value_parser = clap::value_parser!(u64).range(1..=600_000))]
        max_candidate_ms: u64,
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
    /// Decode one bounded MLX `DeepSeek` tensor or layer-zero Q/KV activation.
    InspectV41EmbeddingRow {
        /// Primary shard: embeddings for ordinary modes, or resident layer-zero Q/KV weights for `layer0-resident`.
        shard: PathBuf,
        /// Embedding row/token ID.
        #[arg(long, default_value_t = 0)]
        row: usize,
        /// Decode every row for the bounded layer-zero `wq_a` matrix.
        #[arg(long)]
        all_rows: bool,
        /// Tensor or activation family: `embedding`, `layer0-wq-a`, `layer0-wq-b-head0`, `layer0-q-chain`, `layer0-kv-row`, `layer0-hc-fn`, `layer0-hc-mix`, or `layer0-resident`.
        #[arg(long, default_value = "embedding")]
        kind: String,
        /// Optional embedding shard for layer-zero activation modes; with `layer0-resident`, the positional shard remains the resident Q/KV shard.
        #[arg(long)]
        input_shard: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum FetchCommand {
    /// Fetch and validate a `DeepSeek` V4.1 artifact directory.
    Deepseek {
        /// Destination directory for ordinary Hugging Face-compatible files.
        directory: PathBuf,
        /// Opt out of checkpoint weight shards and fetch metadata only.
        #[arg(long)]
        metadata_only: bool,
        /// Show the resolved plan without downloading files.
        #[arg(long)]
        dry_run: bool,
        /// Confirm the large weight download.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum InspectCommand {
    /// Inspect a `DeepSeek` V4.1 configuration, artifact, index, shard, or tensor.
    Deepseek {
        #[command(subcommand)]
        command: DeepseekInspectCommand,
    },
    /// Inspect a Qwen3 configuration or local checkpoint.
    Qwen {
        #[command(subcommand)]
        command: QwenInspectCommand,
    },
    /// Report whether an adapter loads each `config.json` (or diffusers `model_index.json`).
    ///
    /// Each file goes through the configuration gates the loaders run before
    /// reading weights. Exits 2 when a file cannot be read or is not JSON.
    #[cfg(feature = "metal")]
    Supports {
        /// Hugging Face `config.json` or diffusers `model_index.json` files.
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        /// Print one JSON object per line: `path`, `supported`, `adapter`, `model_type`, `architectures`, `reason`.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum DeepseekInspectCommand {
    /// Validate config.json and the V4.1 execution contract.
    Config {
        /// Path to the `DeepSeek` V4.1 config.json file.
        path: PathBuf,
        /// Also validate dimensions and mode relationships needed for execution planning.
        #[arg(long)]
        execution_shape: bool,
    },
    /// Validate config, tokenizer/template, index, and shard headers without loading payloads.
    Artifact {
        /// Directory containing the complete local `DeepSeek` artifact.
        model: PathBuf,
    },
    /// Validate a `DeepSeek` safetensors index without loading weights.
    Index {
        /// Path to the `DeepSeek` safetensors index JSON.
        path: PathBuf,
    },
    /// Validate one `DeepSeek` safetensors shard header without reading payloads.
    Shard {
        /// Path to a `DeepSeek` safetensors shard.
        path: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum QwenInspectCommand {
    /// Validate a Qwen3 configuration without loading weights.
    Config {
        /// Path to the Qwen3 config.json file.
        path: PathBuf,
    },
    /// Validate a local Qwen3 checkpoint's safetensors headers.
    Checkpoint {
        /// Directory containing config.json and safetensors shard files.
        model: PathBuf,
    },
}
