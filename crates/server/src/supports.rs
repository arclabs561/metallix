//! `mx inspect supports`: which adapter, if any, accepts a checkpoint's
//! `config.json`.
//!
//! Each adapter is asked through the configuration gate its loader runs
//! before reading weights, so this answer moves with the loaders instead of
//! a separate list. It reads no weights, tokenizer or template, and does not
//! apply the context and K/V admission limits `mx serve` checks after the
//! configuration. A gate that ignores a declared quantization would pass a
//! config whose weights its loader cannot run, so such configs are refused
//! here until that gate reads the field.

use std::{
    fmt, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::ExitCode,
};

use serde::{Serialize, Serializer};
use serde_json::Value;

use crate::serve_registry::ModelKind;

/// A model adapter whose configuration gate recognized a document.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Adapter {
    /// Causal `qwen3` through the Qwen3 decoder.
    Qwen3,
    /// `llama` (`MiniCPM5`) through the Qwen3 decoder without Q/K norms.
    Llama,
    /// Bidirectional `bidirectional_pplx_qwen3` (pplx-embed).
    PplxQwen3,
    /// Julia-1 decisions on the CPU.
    Julia,
    /// `qwen3_5` hybrid linear/full attention.
    Qwen35,
    /// Gemma 4 dense text.
    Gemma4,
    /// `DeepSeek` V4.1.
    DeepseekV41,
}

impl Adapter {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Llama => "llama",
            Self::PplxQwen3 => "pplx_qwen3",
            Self::Julia => "julia",
            Self::Qwen35 => "qwen35",
            Self::Gemma4 => "gemma4",
            Self::DeepseekV41 => "deepseek_v41",
        }
    }

    /// The `mx serve` registry kinds whose loaders take this adapter's
    /// checkpoints. Empty while no command loads the adapter.
    pub(crate) const fn served_as(self) -> &'static [ModelKind] {
        match self {
            Self::Qwen3 => &[ModelKind::Qwen, ModelKind::QwenEmbedding],
            Self::Llama => &[ModelKind::Qwen],
            Self::PplxQwen3 => &[ModelKind::PplxContext, ModelKind::PplxLate],
            Self::Julia => &[ModelKind::Julia],
            Self::Gemma4 => &[ModelKind::Gemma4],
            Self::Qwen35 | Self::DeepseekV41 => &[],
        }
    }

    /// Whether this adapter's configuration gate validates a declared
    /// `quantization` or `quantization_config` itself. A gate that does not
    /// would accept the config and leave the loader to fail on the weights,
    /// so such a config is refused here instead.
    const fn gate_reads_quantization(self) -> bool {
        match self {
            // V41TextContract requires the official fp8/fp4 block layout.
            Self::DeepseekV41 => true,
            // Qwen3ForwardConfig ignores the field today; once it parses a
            // typed quantization, return true for its three adapters.
            Self::Qwen3
            | Self::Llama
            | Self::PplxQwen3
            | Self::Julia
            | Self::Qwen35
            | Self::Gemma4 => false,
        }
    }

    fn qwen(family: qwen::DecoderFamily, attention: qwen::Qwen3Attention) -> Self {
        match (family, attention) {
            (qwen::DecoderFamily::Llama, _) => Self::Llama,
            (qwen::DecoderFamily::Qwen3, qwen::Qwen3Attention::Bidirectional) => Self::PplxQwen3,
            (qwen::DecoderFamily::Qwen3, qwen::Qwen3Attention::Causal) => Self::Qwen3,
        }
    }
}

impl Serialize for Adapter {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

/// What the adapters made of one parsed configuration document.
#[derive(Debug, PartialEq)]
pub(crate) enum Verdict {
    /// The adapter's configuration gate accepted the document.
    Accepted(Adapter),
    /// The adapter owns this architecture but refused this variant.
    Rejected { adapter: Adapter, reason: String },
    /// The adapter's gate accepted a quantized config without reading its
    /// quantization, so its loader cannot run the weights.
    QuantizationNotRead {
        adapter: Adapter,
        quantization: DeclaredQuantization,
    },
    /// A diffusers `model_index.json`, which no adapter loads.
    DiffusersPipeline { class_name: String },
    /// No adapter recognizes the architecture.
    Unrecognized { model_type: Option<String> },
}

/// One adapter's answer.
enum Claim {
    Accepted(Adapter),
    Rejected(Adapter, String),
    /// Another architecture, or a document this adapter's schema cannot read.
    NotMine,
}

impl Verdict {
    /// Asks every adapter, in a fixed order, through its loader's gate. Every
    /// gate checks the model type before anything else, so at most one claims
    /// a document.
    pub(crate) fn of(json: &str, document: &Value) -> Self {
        let claims = [
            julia_claim(document),
            qwen_claim(json, document),
            qwen35_claim(json),
            gemma4_claim(json),
            deepseek_v41_claim(json),
        ];
        for claim in claims {
            match claim {
                Claim::Accepted(adapter) => {
                    return match DeclaredQuantization::of(document) {
                        Some(quantization) if !adapter.gate_reads_quantization() => {
                            Self::QuantizationNotRead {
                                adapter,
                                quantization,
                            }
                        }
                        _ => Self::Accepted(adapter),
                    };
                }
                Claim::Rejected(adapter, reason) => return Self::Rejected { adapter, reason },
                Claim::NotMine => {}
            }
        }
        match document.get("_class_name").and_then(Value::as_str) {
            Some(class_name) => Self::DiffusersPipeline {
                class_name: class_name.to_owned(),
            },
            None => Self::Unrecognized {
                model_type: model_type(document),
            },
        }
    }

    pub(crate) const fn adapter(&self) -> Option<Adapter> {
        match self {
            Self::Accepted(adapter)
            | Self::Rejected { adapter, .. }
            | Self::QuantizationNotRead { adapter, .. } => Some(*adapter),
            Self::DiffusersPipeline { .. } | Self::Unrecognized { .. } => None,
        }
    }

    /// Whether an `mx serve` registry kind loads this checkpoint.
    pub(crate) fn supported(&self) -> bool {
        matches!(self, Self::Accepted(adapter) if !adapter.served_as().is_empty())
    }

    pub(crate) fn reason(&self) -> String {
        match self {
            Self::Accepted(adapter) if adapter.served_as().is_empty() => format!(
                "config accepted by the {} adapter{}; no mx serve kind loads it yet",
                adapter.name(),
                note(*adapter)
            ),
            Self::Accepted(adapter) => {
                let kinds: Vec<String> =
                    adapter.served_as().iter().copied().map(kind_name).collect();
                format!(
                    "loads as mx serve kind {}{}",
                    kinds.join(" or "),
                    note(*adapter)
                )
            }
            Self::Rejected { reason, .. } => reason.clone(),
            Self::QuantizationNotRead {
                adapter,
                quantization,
            } => format!(
                "config declares {quantization}, and the {} loader does not read quantized weights",
                adapter.name()
            ),
            Self::DiffusersPipeline { class_name } => {
                format!("diffusers pipeline {class_name}; no adapter loads diffusers pipelines")
            }
            Self::Unrecognized {
                model_type: Some(model_type),
            } => format!("no adapter accepts model_type {model_type:?}"),
            Self::Unrecognized { model_type: None } => {
                String::from("no adapter recognizes this config; it names no model_type")
            }
        }
    }
}

/// A `quantization` or `quantization_config` object in a config. MLX writes
/// `{"group_size": 64, "bits": 4}` (sometimes with `"mode": "affine"`);
/// Hugging Face writes `quant_method`.
#[derive(Debug, PartialEq)]
pub(crate) struct DeclaredQuantization {
    field: &'static str,
    method: Option<String>,
    bits: Option<u64>,
    group_size: Option<u64>,
}

impl DeclaredQuantization {
    fn of(document: &Value) -> Option<Self> {
        ["quantization_config", "quantization"]
            .into_iter()
            .find_map(|field| {
                let declared = document.get(field).filter(|value| !value.is_null())?;
                let text = |key: &str| declared.get(key).and_then(Value::as_str).map(str::to_owned);
                Some(Self {
                    field,
                    method: text("quant_method")
                        .or_else(|| text("method"))
                        .or_else(|| text("mode")),
                    bits: declared.get("bits").and_then(Value::as_u64),
                    group_size: declared.get("group_size").and_then(Value::as_u64),
                })
            })
    }
}

impl fmt::Display for DeclaredQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (", self.field)?;
        match &self.method {
            Some(method) => write!(f, "method {method:?}")?,
            None => write!(f, "no method stated")?,
        }
        if let Some(bits) = self.bits {
            write!(f, ", {bits}-bit")?;
        }
        if let Some(group_size) = self.group_size {
            write!(f, ", group size {group_size}")?;
        }
        write!(f, ")")
    }
}

fn note(adapter: Adapter) -> &'static str {
    if adapter == Adapter::Gemma4 {
        " (text tower only; vision encoder not loaded)"
    } else {
        ""
    }
}

fn kind_name(kind: ModelKind) -> String {
    match serde_json::to_value(kind) {
        Ok(Value::String(name)) => name,
        _ => format!("{kind:?}"),
    }
}

fn model_type(document: &Value) -> Option<String> {
    document
        .get("model_type")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

fn julia_claim(document: &Value) -> Claim {
    if document["architecture"] != julia::checkpoint::ARCHITECTURE {
        return Claim::NotMine;
    }
    match julia::JuliaCheckpoint::check_config("config.json", document) {
        Ok(()) => Claim::Accepted(Adapter::Julia),
        Err(error) => Claim::Rejected(Adapter::Julia, error.to_string()),
    }
}

/// `ChatSession`, the embedders and the pplx encoders all load through
/// `Qwen3MlxWeights`, which parses [`qwen::forward::Qwen3ForwardConfig`].
fn qwen_claim(json: &str, document: &Value) -> Claim {
    use qwen::forward::{Qwen3ForwardConfig, Qwen3ForwardError};
    match Qwen3ForwardConfig::parse(json) {
        Ok(config) => Claim::Accepted(Adapter::qwen(config.family(), config.attention())),
        Err(Qwen3ForwardError::UnsupportedModelType(_) | Qwen3ForwardError::Json(_)) => {
            Claim::NotMine
        }
        Err(error) => {
            let model_type = model_type(document).unwrap_or_default();
            let bidirectional = document["use_bidirectional_attention"] == true;
            match (
                qwen::DecoderFamily::from_model_type(&model_type),
                qwen::Qwen3Attention::from_config(&model_type, bidirectional),
            ) {
                (Some(family), Some(attention)) => {
                    Claim::Rejected(Adapter::qwen(family, attention), error.to_string())
                }
                _ => Claim::NotMine,
            }
        }
    }
}

fn qwen35_claim(json: &str) -> Claim {
    use qwen35::{Qwen35Config, Qwen35ConfigError};
    match Qwen35Config::parse(json) {
        Ok(_) => Claim::Accepted(Adapter::Qwen35),
        Err(Qwen35ConfigError::UnexpectedModelType(_) | Qwen35ConfigError::Json(_)) => {
            Claim::NotMine
        }
        Err(error) => Claim::Rejected(Adapter::Qwen35, error.to_string()),
    }
}

fn gemma4_claim(json: &str) -> Claim {
    use gemma::{Gemma4ConfigError, Gemma4TextConfig};
    match Gemma4TextConfig::parse(json) {
        Ok(_) => Claim::Accepted(Adapter::Gemma4),
        Err(Gemma4ConfigError::UnexpectedModelType(_) | Gemma4ConfigError::Json(_)) => {
            Claim::NotMine
        }
        Err(error) => Claim::Rejected(Adapter::Gemma4, error.to_string()),
    }
}

fn deepseek_v41_claim(json: &str) -> Claim {
    use deepseek::{V41ConfigError, V41TextContract};
    match V41TextContract::parse(json) {
        Ok(_) => Claim::Accepted(Adapter::DeepseekV41),
        Err(V41ConfigError::UnexpectedModelType(_) | V41ConfigError::Json(_)) => Claim::NotMine,
        Err(error) => Claim::Rejected(Adapter::DeepseekV41, error.to_string()),
    }
}

/// One output line, in the order of the `--json` contract.
#[derive(Debug, Serialize)]
struct Report {
    path: String,
    supported: bool,
    adapter: Option<Adapter>,
    model_type: Option<String>,
    architectures: Vec<String>,
    reason: String,
}

impl Report {
    /// Reads and classifies one file; `Err` carries the report for a file
    /// that could not be read or was not JSON.
    fn of(path: &Path) -> Result<Self, Self> {
        let failed = |reason: String| Self {
            path: path.display().to_string(),
            supported: false,
            adapter: None,
            model_type: None,
            architectures: Vec::new(),
            reason,
        };
        let json =
            fs::read_to_string(path).map_err(|error| failed(format!("could not read: {error}")))?;
        let document: Value = serde_json::from_str(&json)
            .map_err(|error| failed(format!("invalid JSON: {error}")))?;
        let verdict = Verdict::of(&json, &document);
        Ok(Self {
            path: path.display().to_string(),
            supported: verdict.supported(),
            adapter: verdict.adapter(),
            model_type: model_type(&document),
            architectures: document
                .get("architectures")
                .and_then(Value::as_array)
                .map(|names| {
                    names
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_default(),
            reason: verdict.reason(),
        })
    }

    fn human(&self) -> String {
        match (self.supported, self.adapter) {
            (true, Some(adapter)) => {
                format!(
                    "{}: supported ({}): {}",
                    self.path,
                    adapter.name(),
                    self.reason
                )
            }
            _ => format!("{}: unsupported: {}", self.path, self.reason),
        }
    }
}

/// Writes one line per path and returns the exit status: 0 when every file
/// was read and parsed as JSON, 2 otherwise.
fn write_reports(paths: &[PathBuf], json: bool, out: &mut impl Write) -> io::Result<u8> {
    let mut status = 0;
    for path in paths {
        let report = Report::of(path).unwrap_or_else(|report| {
            status = 2;
            report
        });
        if json {
            serde_json::to_writer(&mut *out, &report)?;
            writeln!(out)?;
        } else {
            writeln!(out, "{}", report.human())?;
        }
    }
    Ok(status)
}

pub(crate) fn inspect_supports(paths: &[PathBuf], json: bool) -> ExitCode {
    match write_reports(paths, json, &mut io::stdout().lock()) {
        Ok(status) => ExitCode::from(status),
        Err(error) => {
            eprintln!("could not write the report: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests;
