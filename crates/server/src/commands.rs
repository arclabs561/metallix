//! Inspection, fetch and diagnostic command implementations.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command as ProcessCommand, ExitCode},
};

#[cfg(feature = "metal")]
use std::num::NonZeroUsize;

use crate::model_registry;

#[cfg(feature = "metal")]
use deepseek::checkpoint::mlx::{read_affine_rows_from_shard, read_bf16_tensor_from_shard};
#[cfg(feature = "metal")]
use deepseek::{RotaryDirection, RotaryFrequencyParameters, RotaryTailLayout, rotate_tail};
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

pub(crate) fn inspect_qwen_checkpoint(model: &PathBuf) -> ExitCode {
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

pub(crate) fn inspect_v41_index(index: &PathBuf) -> ExitCode {
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

pub(crate) fn inspect_v41_artifact(model: &Path) -> ExitCode {
    match deepseek::V41ArtifactInspection::inspect(model) {
        Ok(inspection) => {
            let text = inspection.text_contract();
            println!("DeepSeek-V4.1 local artifact");
            println!("index_kind: {:?}", inspection.index_kind());
            println!("tensors: {}", inspection.tensor_count());
            println!("shards: {}", inspection.shard_count());
            println!("layers: {}", text.total_layers());
            println!("routed_experts: {}", text.local_experts());
            println!("scope: metadata, tokenizer/template, and shard headers; no payload load");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!(
                "{} is not a valid bounded DeepSeek-V4.1 artifact: {error}",
                model.display()
            );
            ExitCode::FAILURE
        }
    }
}

pub(crate) fn inspect_v41_shard(shard: &PathBuf) -> ExitCode {
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
    clippy::cast_precision_loss,
    reason = "bounded CLI artifact inspection keeps its I/O phases explicit and uses fixed model ranks"
)]
pub(crate) fn inspect_v41_embedding_row(
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
    if kind == "layer0-resident" {
        let resident = match deepseek::checkpoint::mlx::LayerZeroQkvResident::load(shard, &header) {
            Ok(resident) => resident,
            Err(error) => {
                eprintln!("resident layer-zero load failed: {error}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(error) = resident.validate() {
            eprintln!("resident layer-zero validation failed: {error}");
            return ExitCode::FAILURE;
        }
        let checksum = |values: &[f32]| {
            values.iter().fold(0_u64, |hash, value| {
                hash.wrapping_mul(1_099_511_628_211)
                    .wrapping_add(u64::from(value.to_bits()))
            })
        };
        let bytes = (resident.wq_a.len()
            + resident.attn_norm.len()
            + resident.hc_fn.len()
            + resident.hc_base.len()
            + resident.hc_scale.len()
            + resident.q_norm.len()
            + resident.wkv.len()
            + resident.kv_norm.len())
            * std::mem::size_of::<f32>();
        println!("DeepSeek resident layer-zero Q/KV");
        println!(
            "wq_a: {}x{}",
            deepseek::checkpoint::mlx::LayerZeroQkvResident::WQ_A_ROWS,
            deepseek::checkpoint::mlx::LayerZeroQkvResident::HIDDEN_WIDTH
        );
        println!(
            "wkv: {}x{}",
            deepseek::checkpoint::mlx::LayerZeroQkvResident::WKV_ROWS,
            deepseek::checkpoint::mlx::LayerZeroQkvResident::HIDDEN_WIDTH
        );
        println!("resident_bytes: {bytes}");
        println!("wq_a_checksum: {:016x}", checksum(&resident.wq_a));
        println!("attn_norm_checksum: {:016x}", checksum(&resident.attn_norm));
        println!("hc_fn_checksum: {:016x}", checksum(&resident.hc_fn));
        println!("hc_base_checksum: {:016x}", checksum(&resident.hc_base));
        println!("hc_scale_checksum: {:016x}", checksum(&resident.hc_scale));
        println!("wkv_checksum: {:016x}", checksum(&resident.wkv));
        println!("q_norm_checksum: {:016x}", checksum(&resident.q_norm));
        println!("kv_norm_checksum: {:016x}", checksum(&resident.kv_norm));
        if let Some(input_shard) = input_shard {
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
            let embedding = match read_affine_row_from_shard(
                input_shard,
                &input_header,
                "model.embed_tokens.weight",
                "model.embed_tokens.scales",
                "model.embed_tokens.biases",
                row,
                deepseek::checkpoint::mlx::LayerZeroQkvResident::HIDDEN_WIDTH,
                8,
                64,
            ) {
                Ok(values) => values,
                Err(error) => {
                    eprintln!("embedding decode failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let prepared = match resident.prepare_attention_hidden(&embedding, 1e-6, 1e-20, 4) {
                Ok(values) => values,
                Err(error) => {
                    eprintln!("resident hidden preparation failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            let (q, kv) = match resident.project_qkv(&prepared, 1e-20) {
                Ok(values) => values,
                Err(error) => {
                    eprintln!("resident Q/KV projection failed: {error}");
                    return ExitCode::FAILURE;
                }
            };
            println!("token_row: {row}");
            println!("prepared_hidden_checksum: {:016x}", checksum(&prepared));
            println!("q_activation_checksum: {:016x}", checksum(&q));
            println!("kv_activation_checksum: {:016x}", checksum(&kv));
            println!("activation_scope: real embedding row through HC, attn_norm, Q/KV");
        }
        println!("scope: resident layer-zero weights; optional real embedding activation");
        return ExitCode::SUCCESS;
    }
    #[cfg(feature = "metal")]
    if kind == "layer0-kv-row" {
        const KV_RANK: usize = 512;
        const HIDDEN: usize = 4096;
        let row = row.min(KV_RANK.saturating_sub(1));
        let matrix = match read_affine_rows_from_shard(
            shard,
            &header,
            "model.layers.0.attn.wkv.weight",
            "model.layers.0.attn.wkv.scales",
            "model.layers.0.attn.wkv.biases",
            0,
            KV_RANK,
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
        let values = &matrix[row * HIDDEN..(row + 1) * HIDDEN];
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
        let metal = match deepseek::checkpoint::mlx::apply_affine_matrix_mlx(
            &matrix, KV_RANK, HIDDEN, values,
        ) {
            Ok(output) => {
                let projected = output.as_slice::<f32>();
                let rms = (projected.iter().map(|value| value * value).sum::<f32>()
                    / KV_RANK as f32
                    + 1e-6)
                    .sqrt();
                let normalized = projected
                    .iter()
                    .zip(kv_norm.iter())
                    .map(|(value, weight)| value / rms * weight)
                    .collect::<Vec<_>>();
                let projected_checksum = projected.iter().fold(0_u64, |hash, value| {
                    hash.wrapping_mul(1_099_511_628_211)
                        .wrapping_add(u64::from(value.to_bits()))
                });
                let normalized_checksum = normalized.iter().fold(0_u64, |hash, value| {
                    hash.wrapping_mul(1_099_511_628_211)
                        .wrapping_add(u64::from(value.to_bits()))
                });
                let mut rotary_tail = normalized[KV_RANK - 64..].to_vec();
                let rotary_layout = RotaryTailLayout::new(
                    NonZeroUsize::new(1).expect("nonzero batch"),
                    NonZeroUsize::new(1).expect("nonzero position"),
                    NonZeroUsize::new(1).expect("nonzero head"),
                    NonZeroUsize::new(32).expect("nonzero rotary pairs"),
                )
                .expect("valid KV rotary layout");
                let rotary_parameters = RotaryFrequencyParameters::new(
                    NonZeroUsize::new(64).expect("nonzero rotary width"),
                    65_536,
                    10_000.0,
                    16.0,
                    32.0,
                    1.0,
                )
                .expect("valid DeepSeek rotary parameters");
                let frequencies = rotary_parameters
                    .frequencies(1, NonZeroUsize::new(1).expect("nonzero position"))
                    .expect("bounded KV rotary frequencies");
                rotate_tail(
                    &mut rotary_tail,
                    rotary_layout,
                    &frequencies,
                    RotaryDirection::Forward,
                )
                .expect("KV rotary tail application");
                let rotary_checksum = rotary_tail.iter().fold(0_u64, |hash, value| {
                    hash.wrapping_mul(1_099_511_628_211)
                        .wrapping_add(u64::from(value.to_bits()))
                });
                println!("metal_eval: passed");
                println!("projected_width: {}", projected.len());
                println!("projected_checksum: {projected_checksum:016x}");
                println!("kv_norm_output_checksum: {normalized_checksum:016x}");
                println!("kv_rotary_tail_checksum: {rotary_checksum:016x}");
                true
            }
            Err(error) => {
                eprintln!("wkv Metal projection failed: {error}");
                false
            }
        };
        println!(
            "scope: full layer-zero wkv projection and learned kv_norm; bounded row self-input"
        );
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

pub(crate) fn inspect_qwen(config: &PathBuf) -> ExitCode {
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

pub(crate) fn inspect_v41(config: &PathBuf, execution_shape: bool) -> ExitCode {
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
