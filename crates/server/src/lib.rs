//! The `mx` command-line program and its HTTP server, `mx serve`.
//!
//! Both binaries, `mx` and the identical `metallix`, call [`run`], which parses
//! the command line and runs one subcommand. The library target exists so the
//! two binaries and the integration tests share one implementation. Its only
//! public items are [`run`] and [`range_fetch`], which the `DeepSeek` V4.1
//! integration tests use; neither is a stable API for other crates.
//!
//! Model architectures live in their own crates. This crate loads them,
//! prepares prompts, schedules requests and serves the HTTP protocols.
//!
//! # Layout
//!
//! Every module is private; these are the ones to start from.
//!
//! * `cli` defines the subcommands and their arguments.
//! * `serving` is `mx serve`. `serve_proxy` runs one child `mx serve` process
//!   per registered model (`serve_registry`), `admission_queue` orders each
//!   model's requests, and `generation_routes` parses each protocol
//!   (`responses`, `chat_completions`, `anthropic_messages`) before any
//!   generation starts.
//!   The opt-in `engine_loop` owns paged decoder state and batches active
//!   sequences while protocol writer threads handle their output.
//! * `chat_generation` and `chat_decoder` run resident chat generation;
//!   `qwen_prefix_cache` and `qwen_speculation` add prefix reuse and
//!   speculative decoding to it.
//! * `inspect`, `supports` and `commands` implement the read-only and
//!   diagnostic subcommands.
//! * `telemetry`, `trace_context` and `gpu` provide spans, request IDs and
//!   MLX memory figures for diagnostics.
//!
//! # Features
//!
//! * `metal` builds everything that runs a model on the GPU, including
//!   `serve`, `gen` and `chat`. Without it only `inspect`, `fetch` and the
//!   reduced `DeepSeek` runner are available.
//! * `structured-output` enables JSON Schema constrained generation.
//! * `timeline` writes a Chrome/Perfetto JSON timeline to `--trace-out`.

use std::process::ExitCode;

#[cfg(feature = "metal")]
use std::time::Duration;

#[cfg(feature = "metal")]
use crate::chat_generation::{ResidentChatLimits, render_generation_prompt};

#[cfg(feature = "metal")]
mod admission_queue;
#[cfg(feature = "metal")]
mod agent_receipt;
#[cfg(feature = "metal")]
mod anthropic_messages;
#[cfg(feature = "metal")]
mod chat_cli;
#[cfg(feature = "metal")]
mod chat_completions;
#[cfg(feature = "metal")]
mod chat_generation;
#[cfg(feature = "metal")]
mod chat_tools;
mod cli;
mod commands;
#[cfg(feature = "metal")]
mod completions;
#[cfg(feature = "metal")]
mod decision_cli;
mod deepseek_reduced_cli;
mod deepseek_selected_cli;
#[cfg(feature = "metal")]
mod engine_loop;
#[cfg(feature = "metal")]
mod generation_routes;
#[cfg(feature = "metal")]
mod http_transport;
mod inspect;
#[cfg(feature = "metal")]
mod julia_decisions;
mod model_registry;
#[cfg(feature = "metal")]
mod multipart;
#[cfg(feature = "metal")]
mod pplx_context_embeddings;
#[cfg(feature = "metal")]
mod pplx_late_embeddings;
#[cfg(feature = "metal")]
mod qwen_asr;
#[cfg(feature = "metal")]
mod qwen_decisions;
#[cfg(feature = "metal")]
mod qwen_embeddings;
pub mod range_fetch;
#[cfg(feature = "metal")]
mod responses;
#[cfg(feature = "metal")]
mod scoring;
#[cfg(feature = "metal")]
mod serve_proxy;
#[cfg(feature = "metal")]
mod serve_registry;
#[cfg(feature = "metal")]
mod serving;
#[cfg(feature = "metal")]
mod sse;
#[cfg(feature = "metal")]
mod supports;
#[cfg(feature = "metal")]
mod transcriptions;

#[cfg(feature = "metal")]
mod generation_preview;
#[cfg(feature = "metal")]
mod gpu;
#[cfg(feature = "metal")]
mod parity;
#[cfg(all(feature = "metal", feature = "structured-output"))]
mod qwen_constraints;
#[cfg(feature = "metal")]
mod qwen_forward;
#[cfg(all(test, feature = "metal"))]
mod qwen_particle_tests;
#[cfg(all(feature = "metal", feature = "structured-output"))]
mod schedule_requirements;
mod telemetry;
#[cfg(feature = "metal")]
mod trace_context;
#[cfg(feature = "metal")]
mod v41_indexer;
#[cfg(feature = "metal")]
mod v41_rotary;

use clap::Parser;
use cli::{Cli, Command, DeepseekInspectCommand, FetchCommand, InspectCommand, QwenInspectCommand};
#[cfg(feature = "metal")]
use cli::{GenerationMemoryMode, parse_temperature};
#[cfg(test)]
use commands::format_bytes;
#[cfg(feature = "metal")]
use commands::{
    check_qwen_embedding_metal, check_qwen_layer_metal, check_qwen_projection_metal,
    check_qwen_rows_metal, check_qwen_stream_cache_metal, check_qwen_stream_metal,
    check_qwen_tensor_metal, embed_qwen_metal, load_qwen_metal, smoke_qwen_metal,
};
#[cfg(all(test, feature = "metal"))]
use commands::{checked_row_range, layer_check_cycles};
use commands::{fetch_deepseek, free_space_bytes};
use inspect::{
    inspect_qwen, inspect_qwen_checkpoint, inspect_v41, inspect_v41_artifact,
    inspect_v41_embedding_row, inspect_v41_index, inspect_v41_shard,
};

/// `mx serve`'s paged K/V pool when `--kv-budget-mib` is omitted: 4096 MiB,
/// or a quarter of physical memory when that is smaller. Each generating
/// child allocates its whole pool at load, so a small Mac must not be handed
/// a fixed 4 GiB reservation. If physical memory cannot be read, 512 MiB.
#[cfg(feature = "metal")]
fn default_kv_budget_mib() -> u32 {
    const MIB: u64 = 1 << 20;
    let physical = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| text.trim().parse::<u64>().ok());
    if let Some(bytes) = physical {
        u32::try_from((bytes / 4 / MIB).min(4096))
            .unwrap_or(4096)
            .max(1)
    } else {
        tracing::warn!("could not read physical memory; --kv-budget-mib defaults to 512");
        512
    }
}

#[cfg(feature = "metal")]
const fn resident_chat_limits(context_tokens: u32, kv_budget_mib: u32) -> ResidentChatLimits {
    ResidentChatLimits::from_mib(context_tokens as usize, kv_budget_mib)
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

/// Runs the shared CLI, preserving the invoked executable name in help output.
#[must_use]
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    if let Err(error) = telemetry::init(cli.trace_out.as_deref()) {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    // Before any command touches MLX: every mx process, serve children
    // included, shares this cap.
    #[cfg(feature = "metal")]
    if let Err(error) = gpu::cap_cache() {
        eprintln!("{error}");
        return ExitCode::FAILURE;
    }
    let code = with_capture(cli);
    telemetry::finish();
    code
}

/// Wraps a one-shot command in `--gpu-capture`; `serve` instead hands the path
/// to each child, which captures its first request.
#[cfg(feature = "metal")]
fn with_capture(cli: Cli) -> ExitCode {
    let Some(path) = cli.gpu_capture.clone() else {
        return dispatch(cli);
    };
    if matches!(cli.command, Command::Serve { .. }) {
        return dispatch(cli);
    }
    let capture = match gpu::Capture::start(&path) {
        Ok(capture) => capture,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    let code = dispatch(cli);
    drop(capture);
    code
}

#[cfg(not(feature = "metal"))]
fn with_capture(cli: Cli) -> ExitCode {
    dispatch(cli)
}

#[allow(
    clippy::too_many_lines,
    reason = "exhaustive CLI dispatch; handlers stay separate"
)]
fn dispatch(cli: Cli) -> ExitCode {
    match cli.command {
        Command::RunDeepseekReduced(args) => args.run(),
        Command::RunDeepseekSelected(args) => args.run(),
        #[cfg(feature = "metal")]
        Command::Decide(args) => args.run(),
        #[cfg(feature = "metal")]
        Command::DecideJulia(args) => args.run(),
        Command::Fetch {
            command:
                FetchCommand::Deepseek {
                    directory,
                    metadata_only,
                    dry_run,
                    yes,
                },
        } => fetch_deepseek(&directory, metadata_only, dry_run, yes),
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
            registry,
            memory_budget_mib,
            worker_entry,
            listen,
            context_tokens,
            kv_budget_mib,
            prefix_cache_mib,
            generation_timeout_ms,
            queue_depth,
            queue_wait_ms,
            max_num_seqs,
        } => {
            let kv_budget_mib = kv_budget_mib.unwrap_or_else(default_kv_budget_mib);
            if let Some(entry) = worker_entry {
                return match serde_json::from_str(&entry) {
                    Ok(entry) => serving::serve_child(
                        entry,
                        listen,
                        resident_chat_limits(context_tokens, kv_budget_mib)
                            .with_prefix_cache_mib(prefix_cache_mib),
                        engine_loop::EngineLimits {
                            max_num_seqs: max_num_seqs as usize,
                        },
                        Duration::from_millis(u64::from(generation_timeout_ms)),
                        cli.gpu_capture.as_deref(),
                    ),
                    Err(error) => {
                        tracing::error!("mx serve: invalid worker entry: {error}");
                        ExitCode::FAILURE
                    }
                };
            }
            match serve_registry::entries(registry.as_deref(), model.as_deref(), &model_id) {
                Ok(models) => serve_proxy::serve(
                    &models,
                    listen,
                    serve_proxy::ChildSettings {
                        context_tokens,
                        kv_budget_mib,
                        prefix_cache_mib,
                        max_num_seqs,
                        generation_timeout_ms,
                        trace_out: cli.trace_out,
                        gpu_capture: cli.gpu_capture,
                    },
                    memory_budget_mib,
                    admission_queue::QueueSettings {
                        depth: queue_depth as usize,
                        wait: Duration::from_millis(u64::from(queue_wait_ms)),
                        max_running: max_num_seqs as usize,
                    },
                ),
                Err(error) => {
                    tracing::error!("mx serve: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        #[cfg(feature = "metal")]
        Command::CheckV41RotaryMetal { fixture, repeats } => v41_rotary::run(&fixture, repeats),
        #[cfg(feature = "metal")]
        Command::CheckV41IndexerMetal { fixture, repeats } => v41_indexer::run(&fixture, repeats),
        #[cfg(feature = "metal")]
        Command::GenerateQwenMetal {
            model,
            input_ids,
            prompt,
            chat_template,
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
            verify_schedule,
            #[cfg(feature = "structured-output")]
            schedule_requirements,
            #[cfg(feature = "structured-output")]
            max_attempts,
            #[cfg(feature = "structured-output")]
            max_candidate_ms,
            #[cfg(feature = "structured-output")]
            json_schema,
            #[cfg(feature = "structured-output")]
            json_schema_inline,
        } => {
            #[cfg(feature = "structured-output")]
            let requirements = match schedule_requirements
                .as_deref()
                .map(schedule_requirements::ScheduleRequirements::load)
                .transpose()
            {
                Ok(requirements) => requirements,
                Err(error) => {
                    eprintln!("Qwen schedule requirements failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let max_tokens = generation_max_tokens(memory_mode, max_tokens);
            let (input_ids, tokenizer, input_format) = match generation_input(input_ids, prompt) {
                Ok(GenerationInput::RawIds(input_ids)) => (
                    input_ids,
                    None,
                    qwen_forward::GenerationInputFormat {
                        kind: "raw_token_ids",
                        chat_template_sha256: None,
                    },
                ),
                Ok(GenerationInput::Prompt(prompt)) => {
                    let rendered = if chat_template {
                        match render_generation_prompt(&model, &prompt) {
                            Ok(rendered) => Some(rendered),
                            Err(error) => {
                                eprintln!("Qwen prompt failed: {error}");
                                return ExitCode::FAILURE;
                            }
                        }
                    } else {
                        None
                    };
                    let tokenizer = match chat_format::QwenTokenizer::load(&model) {
                        Ok(tokenizer) => tokenizer,
                        Err(error) => {
                            eprintln!("Qwen prompt failed: {error}");
                            return ExitCode::FAILURE;
                        }
                    };
                    let input = rendered
                        .as_ref()
                        .map_or(&prompt, |rendered| &rendered.rendered);
                    let input_ids = match tokenizer.encode_prompt(input) {
                        Ok(input_ids) => input_ids,
                        Err(error) => {
                            eprintln!("Qwen prompt failed: {error}");
                            return ExitCode::FAILURE;
                        }
                    };
                    (
                        input_ids,
                        Some(tokenizer),
                        qwen_forward::GenerationInputFormat {
                            kind: if rendered.is_some() {
                                "qwen_chat_template_user_message"
                            } else {
                                "plain_text_prompt"
                            },
                            chat_template_sha256: rendered
                                .as_ref()
                                .map(|rendered| rendered.template_sha256.clone()),
                        },
                    )
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
                #[cfg(feature = "structured-output")]
                verify_schedule
                    .then_some(qwen_forward::ScheduleVerificationConfig {
                        max_attempts,
                        max_elapsed_ms: max_candidate_ms,
                        requirements,
                    })
                    .as_ref(),
                qwen_forward::GenerationDiagnostics {
                    tokenizer: tokenizer.as_ref(),
                    verbose,
                    logprobs,
                    preview,
                    sampling,
                    input_format,
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
        Command::Inspect { command } => match command {
            InspectCommand::Deepseek {
                command:
                    DeepseekInspectCommand::Config {
                        path,
                        execution_shape,
                    },
            } => inspect_v41(&path, execution_shape),
            InspectCommand::Deepseek {
                command: DeepseekInspectCommand::Artifact { model },
            } => inspect_v41_artifact(&model),
            InspectCommand::Deepseek {
                command: DeepseekInspectCommand::Index { path },
            } => inspect_v41_index(&path),
            InspectCommand::Deepseek {
                command: DeepseekInspectCommand::Shard { path },
            } => inspect_v41_shard(&path),
            InspectCommand::Qwen {
                command: QwenInspectCommand::Config { path },
            } => inspect_qwen(&path),
            InspectCommand::Qwen {
                command: QwenInspectCommand::Checkpoint { model },
            } => inspect_qwen_checkpoint(&model),
            #[cfg(feature = "metal")]
            InspectCommand::Supports { paths, json } => supports::inspect_supports(&paths, json),
        },
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

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::{Cli, Command, DeepseekInspectCommand, InspectCommand};
    use std::path::Path;

    #[cfg(feature = "metal")]
    #[test]
    fn explicit_context_reaches_resident_limits_for_each_consumer() {
        for (command, extra) in [
            ("chat", vec![]),
            ("agent", vec!["--workspace", ".", "--prompt", "hello"]),
            ("serve", vec![]),
        ] {
            for requested in ["16385", "262144"] {
                let mut args = vec![
                    "mx",
                    command,
                    "--model",
                    "local",
                    "--context-tokens",
                    requested,
                ];
                args.extend_from_slice(&extra);
                let cli = Cli::try_parse_from(args)
                    .expect("explicit context parses before model admission");
                let (Command::Chat {
                    context_tokens: tokens,
                    ..
                }
                | Command::Agent {
                    context_tokens: tokens,
                    ..
                }
                | Command::Serve {
                    context_tokens: tokens,
                    ..
                }) = cli.command
                else {
                    panic!("unexpected consumer");
                };
                let limits = super::resident_chat_limits(tokens, 512);
                assert_eq!(limits.context_tokens(), requested.parse::<usize>().unwrap());
                assert_eq!(limits.kv_budget_bytes(), 512 * 1024 * 1024);
            }
            for invalid in ["0", "auto", "2147483648", "4294967296"] {
                let mut args = vec![
                    "mx",
                    command,
                    "--model",
                    "local",
                    "--context-tokens",
                    invalid,
                ];
                args.extend_from_slice(&extra);
                assert!(Cli::try_parse_from(args).is_err(), "{command}: {invalid}");
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn explicit_context_errors_distinguish_empty_from_invalid_integer() {
        use crate::cli::{ExplicitContextError, parse_context_tokens};
        assert!(matches!(
            parse_context_tokens("0"),
            Err(ExplicitContextError::Empty)
        ));
        assert!(matches!(
            parse_context_tokens("2147483648"),
            Err(ExplicitContextError::Shape {
                requested: 2_147_483_648
            })
        ));
        assert_eq!(parse_context_tokens("2147483647").unwrap(), 2_147_483_647);
        for invalid in ["auto", "-1", "4294967296"] {
            assert!(matches!(
                parse_context_tokens(invalid),
                Err(ExplicitContextError::Integer(_))
            ));
        }
    }

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
    fn serve_defaults_to_one_sequence_at_a_time() {
        let cli = Cli::try_parse_from(["mx", "serve", "--model", "model"]).expect("serve defaults");
        let super::Command::Serve { max_num_seqs, .. } = cli.command else {
            panic!("parsed as serve");
        };
        assert_eq!(max_num_seqs, 1, "batching stays opt-in");
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
    fn generation_template_flag_requires_a_prompt() {
        assert!(Cli::try_parse_from(["mx", "gen", "--model", "model", "--chat-template"]).is_err());
        let templated = Cli::try_parse_from([
            "mx",
            "gen",
            "--model",
            "model",
            "--prompt",
            "plain prompt",
            "--chat-template",
        ])
        .expect("templated prompt generation");
        assert!(matches!(
            templated.command,
            super::Command::GenerateQwenMetal {
                prompt: Some(prompt),
                chat_template: true,
                ..
            } if prompt == "plain prompt"
        ));
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
                .contains("Local Qwen3 checkpoint directory containing config.json and weights")
        );
        assert!(normalized_help.contains("Shorthand: invoke this command as `mx gen`."));
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

    #[cfg(all(feature = "metal", feature = "structured-output"))]
    #[test]
    fn schedule_verification_has_explicit_bounded_cli_controls() {
        let cli = Cli::try_parse_from([
            "mx",
            "gen",
            "--model",
            "model",
            "--verify-schedule",
            "--sample",
            "--temperature",
            "0.8",
            "--seed",
            "9",
            "--json-schema-inline",
            r#"{"type":"object"}"#,
            "--max-attempts",
            "3",
            "--max-candidate-ms",
            "1200",
            "--schedule-requirements",
            "requirements.json",
        ])
        .expect("bounded schedule verifier arguments");
        assert!(matches!(
            cli.command,
            super::Command::GenerateQwenMetal {
                verify_schedule: true,
                max_attempts: 3,
                max_candidate_ms: 1200,
                ..
            }
        ));
        assert!(
            Cli::try_parse_from(["mx", "gen", "--model", "model", "--max-attempts", "3"]).is_err()
        );
        assert!(
            Cli::try_parse_from([
                "mx",
                "gen",
                "--model",
                "model",
                "--schedule-requirements",
                "requirements.json"
            ])
            .is_err()
        );
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

    #[test]
    fn grouped_deepseek_artifact_inspection_uses_a_positional_path() {
        let cli = Cli::try_parse_from(["mx", "inspect", "deepseek", "artifact", "model"])
            .expect("grouped artifact inspection");
        assert!(matches!(
            cli.command,
            Command::Inspect {
                command: InspectCommand::Deepseek {
                    command: DeepseekInspectCommand::Artifact { model }
                }
            } if model == Path::new("model")
        ));
    }

    #[test]
    fn byte_messages_are_human_readable() {
        assert_eq!(super::format_bytes(0), "0 B");
        assert_eq!(super::format_bytes(1024), "1.0 KiB");
        assert_eq!(super::format_bytes(144_509_558_784), "134.6 GiB");
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
