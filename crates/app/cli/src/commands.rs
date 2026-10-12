//! Fetch and Metal diagnostic command implementations.

use std::{
    fs,
    path::Path,
    process::{Command as ProcessCommand, ExitCode},
};

#[cfg(feature = "metal")]
use std::path::PathBuf;

use crate::model_registry;

#[allow(
    clippy::too_many_lines,
    reason = "fetch orchestration keeps preflight, metadata, template, and artifact gates visible"
)]
pub(crate) fn fetch_deepseek(
    directory: &Path,
    metadata_only: bool,
    dry_run: bool,
    yes: bool,
) -> ExitCode {
    let weights = !metadata_only;
    if weights && !dry_run && !yes {
        eprintln!(
            "weights are included by default; use --yes to download them or --metadata-only to opt out (try --dry-run first)"
        );
        return ExitCode::FAILURE;
    }
    let source = match model_registry::deepseek() {
        Ok(source) => source,
        Err(error) => {
            eprintln!("cannot load model registry: {error}");
            return ExitCode::FAILURE;
        }
    };
    if weights && !dry_run {
        let parent = directory.parent().unwrap_or_else(|| Path::new("."));
        match free_space_bytes(parent) {
            Some(available) if available < source.weight_bytes => {
                eprintln!(
                    "not enough free space for DeepSeek weights: need {}, have {}; use --metadata-only or choose another volume",
                    format_bytes(source.weight_bytes),
                    format_bytes(available)
                );
                return ExitCode::FAILURE;
            }
            None => {
                eprintln!("could not determine free space for {}", parent.display());
                return ExitCode::FAILURE;
            }
            Some(_) => {}
        }
    }
    let mut model = ProcessCommand::new("hf");
    model
        .arg("download")
        .arg(&source.model_repo)
        .arg("--revision")
        .arg(&source.model_revision)
        .arg("--local-dir")
        .arg(directory);
    for pattern in &source.metadata_files {
        model.arg("--include").arg(pattern);
    }
    if weights {
        model.arg("--include").arg(&source.weight_pattern);
    }
    if dry_run {
        model.arg("--dry-run");
        // Planning does not benefit from parallel workers and the HF CLI can
        // otherwise race its local cache-marker initialization, emitting
        // misleading lock-wait noise before the plan is printed.
        model.arg("--max-workers").arg("1");
    }
    if !run_hf(&mut model) {
        return ExitCode::FAILURE;
    }

    // The official V4.1 tokenizer metadata does not embed a template. The
    // registry pins the separately published compatible template explicitly.
    let mut template = ProcessCommand::new("hf");
    template
        .arg("download")
        .arg(&source.template_repo)
        .arg("chat_template.jinja")
        .arg("--revision")
        .arg(&source.template_revision)
        .arg("--local-dir")
        .arg(directory);
    if dry_run {
        template.arg("--dry-run");
        template.arg("--max-workers").arg("1");
    }
    if !run_hf(&mut template) {
        return ExitCode::FAILURE;
    }

    if dry_run {
        println!("DeepSeek fetch plan ready; no files downloaded");
        println!("weights_included: {weights}");
        return ExitCode::SUCCESS;
    }

    if !weights {
        if metadata_ready(directory) {
            println!("DeepSeek metadata ready");
            println!("directory: {}", directory.display());
            println!("weights_downloaded: false");
            println!(
                "next: run `mx fetch deepseek {} --yes`",
                directory.display()
            );
            return ExitCode::SUCCESS;
        }
        eprintln!("metadata download completed but required files are missing");
        return ExitCode::FAILURE;
    }
    match deepseek::V41ArtifactInspection::inspect(directory) {
        Ok(inspection) => {
            println!("DeepSeek artifact ready");
            println!("directory: {}", directory.display());
            println!("index: {:?}", inspection.index_kind());
            println!("shards: {}", inspection.shard_count());
            println!("weights_downloaded: {weights}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("download completed but artifact validation failed: {error}");
            ExitCode::FAILURE
        }
    }
}

pub(crate) fn metadata_ready(directory: &Path) -> bool {
    [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    .iter()
    .all(|name| directory.join(name).is_file())
        && fs::read_dir(directory)
            .ok()
            .into_iter()
            .flatten()
            .flatten()
            .any(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .is_some_and(|name| name.ends_with(".safetensors.index.json"))
            })
}

pub(crate) fn free_space_bytes(path: &Path) -> Option<u64> {
    let output = ProcessCommand::new("df")
        .args(["-Pk", path.to_str()?])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().last()?;
    let available_kib = line.split_whitespace().nth(3)?.parse::<u64>().ok()?;
    available_kib.checked_mul(1024)
}

#[allow(clippy::cast_precision_loss, reason = "human-readable display only")]
pub(crate) fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

pub(crate) fn run_hf(command: &mut ProcessCommand) -> bool {
    match command.status() {
        Ok(status) if status.success() => true,
        Ok(status) => {
            eprintln!("Hugging Face download failed with status {status}");
            false
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Hugging Face CLI `hf` was not found; install it before fetching artifacts");
            false
        }
        Err(error) => {
            eprintln!("could not start Hugging Face CLI: {error}");
            false
        }
    }
}

#[cfg(feature = "metal")]
#[derive(serde::Serialize)]
pub(crate) struct LayerCheckCycles<T> {
    schema_version: u32,
    operation: &'static str,
    repeats: usize,
    runs: Vec<T>,
    scope: &'static str,
}

#[cfg(feature = "metal")]
pub(crate) fn layer_check_cycles<T>(runs: Vec<T>) -> LayerCheckCycles<T> {
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
pub(crate) fn check_qwen_layer_metal(
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
pub(crate) fn check_qwen_tensor_metal(
    model: &std::path::Path,
    tensor: &str,
    max_bytes: u64,
) -> ExitCode {
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
pub(crate) fn check_qwen_rows_metal(
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
pub(crate) fn checked_row_range(start_row: usize, rows: u32) -> Option<std::ops::Range<usize>> {
    let rows = usize::try_from(rows).ok()?;
    start_row
        .checked_add(rows)
        .map(|end_row| start_row..end_row)
}

#[cfg(feature = "metal")]
pub(crate) fn check_qwen_embedding_metal(
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
pub(crate) fn check_qwen_projection_metal(
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
pub(crate) fn check_qwen_stream_metal(
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
pub(crate) fn print_stream_report(report: impl serde::Serialize) -> ExitCode {
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
pub(crate) fn check_qwen_stream_cache_metal(
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
pub(crate) fn embed_qwen_metal(model: &PathBuf) -> ExitCode {
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
pub(crate) fn load_qwen_metal(model: &PathBuf) -> ExitCode {
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
pub(crate) fn smoke_qwen_metal() -> ExitCode {
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
