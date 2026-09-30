//! Bounded entry point for one reduced `DeepSeek` request artifact.

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Args;
use deepseek::{
    indexer::{key::IndexKeyRotaryExecution, query::IndexScoreExecution},
    reduced::FinalHeadExecution,
};
use serde_json::json;
use sha2::{Digest, Sha256};

const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_INPUT_IDS: usize = 16;

/// Bounded index-score execution used by the reduced request diagnostic.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum ScoreExecutionArg {
    /// Preserve the source-authoritative scalar BF16 stages.
    #[default]
    Scalar,
    /// Run only the bounded BF16 index scorer on Metal.
    #[cfg(feature = "metal")]
    MetalBf16,
}

impl From<ScoreExecutionArg> for IndexScoreExecution {
    fn from(value: ScoreExecutionArg) -> Self {
        match value {
            ScoreExecutionArg::Scalar => Self::Scalar,
            #[cfg(feature = "metal")]
            ScoreExecutionArg::MetalBf16 => Self::MetalBf16,
        }
    }
}

/// Key rotation remains independent of score execution for qualification.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum KeyRotaryExecutionArg {
    #[default]
    Scalar,
    #[cfg(feature = "metal")]
    MetalFp32,
}

impl From<KeyRotaryExecutionArg> for IndexKeyRotaryExecution {
    fn from(value: KeyRotaryExecutionArg) -> Self {
        match value {
            KeyRotaryExecutionArg::Scalar => Self::Scalar,
            #[cfg(feature = "metal")]
            KeyRotaryExecutionArg::MetalFp32 => Self::MetalFp32,
        }
    }
}

/// Final-head execution remains independent of indexed-layer diagnostics.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum HeadExecutionArg {
    #[default]
    Scalar,
    #[cfg(feature = "metal")]
    MetalFp32,
}

impl From<HeadExecutionArg> for FinalHeadExecution {
    fn from(value: HeadExecutionArg) -> Self {
        match value {
            HeadExecutionArg::Scalar => Self::Scalar,
            #[cfg(feature = "metal")]
            HeadExecutionArg::MetalFp32 => Self::MetalFp32,
        }
    }
}

/// Runs a bounded reduced request from one local artifact.
#[derive(Debug, Args)]
pub(crate) struct ReducedArgs {
    /// Local reduced-request artifact, limited to 64 MiB.
    #[arg(long)]
    artifact: PathBuf,
    /// Comma-delimited token IDs; the fixed reduced path accepts at most 16.
    #[arg(long, value_delimiter = ',', required = true)]
    input_ids: Vec<i64>,
    /// Number of initial IDs admitted as the prefill partition.
    #[arg(long)]
    prefill_tokens: usize,
    /// Index-score implementation; Metal only covers the bounded BF16 scorer.
    #[arg(long, value_enum, default_value_t = ScoreExecutionArg::Scalar)]
    score_execution: ScoreExecutionArg,
    /// Index-key rotation only; other key preparation remains scalar.
    #[arg(long, value_enum, default_value_t = KeyRotaryExecutionArg::Scalar)]
    key_rotary_execution: KeyRotaryExecutionArg,
    /// Final-head implementation; Metal covers only the bounded FP32 head.
    #[arg(long, value_enum, default_value_t = HeadExecutionArg::Scalar)]
    head_execution: HeadExecutionArg,
}

impl ReducedArgs {
    pub(crate) fn run(self) -> ExitCode {
        match run(&self) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("deepseek reduced request failed: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn run(args: &ReducedArgs) -> Result<(), String> {
    validate_args(args)?;
    let bytes = read_artifact(&args.artifact)?;
    let artifact_sha256 = format!("{:x}", Sha256::digest(&bytes));
    let artifact = deepseek::reduced::ReducedArtifact::parse(&bytes)
        .map_err(|_| "artifact is not a valid reduced request artifact".to_owned())?;
    let outputs = artifact
        .run_with_head_execution(
            &args.input_ids,
            args.prefill_tokens,
            args.score_execution.into(),
            args.key_rotary_execution.into(),
            args.head_execution.into(),
        )
        .map_err(|_| "reduced request execution was rejected".to_owned())?;
    let expected_calls = 1 + args.input_ids.len() - args.prefill_tokens;
    if outputs.len() != expected_calls {
        return Err("reduced request returned an invalid call count".to_owned());
    }
    let calls = outputs
        .iter()
        .enumerate()
        .map(|(call, output)| {
            let positions = if call == 0 { args.prefill_tokens } else { 1 };
            let start = if call == 0 {
                0
            } else {
                args.prefill_tokens + call - 1
            };
            let logits = output
                .heads()
                .last()
                .ok_or_else(|| "reduced request produced an empty call".to_owned())?
                .logits()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            Ok(json!({
                "start": start,
                "positions": positions,
                "logits_fp32_bits": logits,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let response = json!({
        "schema_version": 1,
        "operation": "deepseek-reduced-request",
        "backend": "scalar",
        "artifact_sha256": artifact_sha256,
        "calls": calls,
    });
    #[cfg(feature = "metal")]
    let response = execution_metadata(response, args);
    let rendered = serde_json::to_string(&response)
        .map_err(|_| "could not render reduced request response".to_owned())?;
    println!("{rendered}");
    Ok(())
}

#[cfg(feature = "metal")]
fn execution_metadata(mut response: serde_json::Value, args: &ReducedArgs) -> serde_json::Value {
    let score = matches!(args.score_execution, ScoreExecutionArg::MetalBf16);
    let key = matches!(args.key_rotary_execution, KeyRotaryExecutionArg::MetalFp32);
    let head = matches!(args.head_execution, HeadExecutionArg::MetalFp32);
    if score || key || head {
        response["backend"] = json!("mixed-cpu-metal");
    }
    if score || key {
        response["score_execution"] = json!(if score { "metal-bf16" } else { "scalar" });
    }
    if key {
        response["key_rotary_execution"] = json!("metal-fp32");
    }
    if head {
        response["head_execution"] = json!("metal-fp32");
    }
    response
}

fn validate_args(args: &ReducedArgs) -> Result<(), String> {
    if args.input_ids.iter().any(|&id| id < 0) {
        return Err("input token IDs must be nonnegative".to_owned());
    }
    if args.input_ids.len() > MAX_INPUT_IDS {
        return Err(format!("input accepts at most {MAX_INPUT_IDS} IDs"));
    }
    if !(2..=args.input_ids.len()).contains(&args.prefill_tokens) {
        return Err(
            "prefill token count must be at least two and no more than the input count".to_owned(),
        );
    }
    Ok(())
}

fn read_artifact(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = fs::metadata(path).map_err(|_| "artifact cannot be inspected".to_owned())?;
    if !metadata.file_type().is_file() {
        return Err("artifact must be one regular file".to_owned());
    }
    if metadata.len() > MAX_ARTIFACT_BYTES {
        return Err("artifact exceeds 64 MiB".to_owned());
    }
    let mut bytes = Vec::new();
    let file = File::open(path).map_err(|_| "artifact cannot be opened".to_owned())?;
    let opened = file
        .metadata()
        .map_err(|_| "opened artifact cannot be inspected".to_owned())?;
    if !opened.is_file() || opened.len() > MAX_ARTIFACT_BYTES {
        return Err("opened artifact must be a regular file within 64 MiB".to_owned());
    }
    file.take(MAX_ARTIFACT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "artifact cannot be read".to_owned())?;
    if u64::try_from(bytes.len()).map_err(|_| "artifact length overflow".to_owned())?
        > MAX_ARTIFACT_BYTES
    {
        return Err("artifact exceeds 64 MiB".to_owned());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::{
        HeadExecutionArg, KeyRotaryExecutionArg, ReducedArgs, ScoreExecutionArg, read_artifact,
        validate_args,
    };

    #[test]
    fn rejects_invalid_arguments_before_loading_an_artifact() {
        let args = ReducedArgs {
            artifact: std::path::PathBuf::from("missing"),
            input_ids: vec![0],
            prefill_tokens: 1,
            score_execution: ScoreExecutionArg::Scalar,
            key_rotary_execution: KeyRotaryExecutionArg::Scalar,
            head_execution: HeadExecutionArg::Scalar,
        };
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn rejects_nonregular_artifact() {
        assert!(read_artifact(std::path::Path::new(".")).is_err());
    }
}
