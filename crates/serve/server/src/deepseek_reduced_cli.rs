//! Bounded entry point for one reduced `DeepSeek` request artifact.

use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::ExitCode,
};

use clap::Args;
use deepseek::{
    indexer::{
        key::{IndexKeyPreparationExecution, IndexKeyRotaryExecution},
        query::IndexScoreExecution,
    },
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

/// Complete index-key preparation remains independent of score and head execution.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum KeyPreparationExecutionArg {
    Scalar,
    #[cfg(feature = "metal")]
    #[value(name = "metal-rotary-fp32")]
    MetalRotaryFp32,
    #[cfg(feature = "metal")]
    #[value(name = "metal-prefp4")]
    MetalPreFp4,
}

impl From<KeyPreparationExecutionArg> for IndexKeyPreparationExecution {
    fn from(value: KeyPreparationExecutionArg) -> Self {
        match value {
            KeyPreparationExecutionArg::Scalar => Self::Scalar,
            #[cfg(feature = "metal")]
            KeyPreparationExecutionArg::MetalRotaryFp32 => Self::MetalRotaryFp32,
            #[cfg(feature = "metal")]
            KeyPreparationExecutionArg::MetalPreFp4 => Self::MetalPreFp4,
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
    /// Number of initial IDs admitted as the replay prefill partition.
    #[arg(
        long,
        required_unless_present = "generation_max_new_tokens",
        conflicts_with = "generation_max_new_tokens"
    )]
    prefill_tokens: Option<usize>,
    /// Generate this many greedy IDs instead of replaying a fixed request partition.
    #[arg(long, conflicts_with = "prefill_tokens")]
    generation_max_new_tokens: Option<usize>,
    /// Optional EOS ID which stops greedy generation after it is selected.
    #[arg(long, requires = "generation_max_new_tokens")]
    generation_eos_token_id: Option<i64>,
    /// Index-score implementation; Metal only covers the bounded BF16 scorer.
    #[arg(long, value_enum, default_value_t = ScoreExecutionArg::Scalar)]
    score_execution: ScoreExecutionArg,
    /// Legacy index-key rotation diagnostic; conflicts with complete key preparation.
    #[arg(long, value_enum)]
    key_rotary_execution: Option<KeyRotaryExecutionArg>,
    /// Complete key preparation diagnostic; Metal runs through pre-FP4 staging.
    #[arg(long, value_enum)]
    key_preparation_execution: Option<KeyPreparationExecutionArg>,
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
    if let Some(max_new_tokens) = args.generation_max_new_tokens {
        let generation = artifact
            .generate_greedy(
                &args.input_ids,
                args.generation_eos_token_id,
                max_new_tokens,
            )
            .map_err(|_| "reduced generation was rejected".to_owned())?;
        let stop_reason = match generation.stop_reason() {
            deepseek::reduced::ReducedGenerationStop::Eos => "eos",
            deepseek::reduced::ReducedGenerationStop::MaxNewTokens => "max_new_tokens",
            _ => return Err("reduced generation returned an unknown stop reason".to_owned()),
        };
        let response = json!({
            "schema_version": 1,
            "operation": "deepseek-reduced-greedy-generation",
            "backend": "scalar",
            "artifact_sha256": artifact_sha256,
            "generated_ids": generation.generated_ids(),
            "stop_reason": stop_reason,
        });
        let rendered = serde_json::to_string(&response)
            .map_err(|_| "could not render reduced generation response".to_owned())?;
        println!("{rendered}");
        return Ok(());
    }

    let prefill_tokens = args
        .prefill_tokens
        .ok_or_else(|| "prefill tokens are required for reduced request replay".to_owned())?;
    let outputs = artifact
        .run_with_key_preparation_execution(
            &args.input_ids,
            prefill_tokens,
            args.score_execution.into(),
            key_preparation_execution(args),
            args.head_execution.into(),
        )
        .map_err(|_| "reduced request execution was rejected".to_owned())?;
    let expected_calls = 1 + args.input_ids.len() - prefill_tokens;
    if outputs.len() != expected_calls {
        return Err("reduced request returned an invalid call count".to_owned());
    }
    let calls = outputs
        .iter()
        .enumerate()
        .map(|(call, output)| {
            let positions = if call == 0 { prefill_tokens } else { 1 };
            let start = if call == 0 {
                0
            } else {
                prefill_tokens + call - 1
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
    let rotary = matches!(
        args.key_rotary_execution,
        Some(KeyRotaryExecutionArg::MetalFp32)
    );
    let preparation = match args.key_preparation_execution {
        Some(KeyPreparationExecutionArg::Scalar) | None => None,
        Some(KeyPreparationExecutionArg::MetalRotaryFp32) => Some("metal-rotary-fp32"),
        Some(KeyPreparationExecutionArg::MetalPreFp4) => Some("metal-prefp4"),
    };
    let head = matches!(args.head_execution, HeadExecutionArg::MetalFp32);
    if score || rotary || preparation.is_some() || head {
        response["backend"] = json!("mixed-cpu-metal");
    }
    if score || rotary {
        response["score_execution"] = json!(if score { "metal-bf16" } else { "scalar" });
    }
    if rotary {
        response["key_rotary_execution"] = json!("metal-fp32");
    }
    if let Some(preparation) = preparation {
        response["key_preparation_execution"] = json!(preparation);
    }
    if head {
        response["head_execution"] = json!("metal-fp32");
    }
    response
}

fn validate_args(args: &ReducedArgs) -> Result<(), String> {
    if args.key_rotary_execution.is_some() && args.key_preparation_execution.is_some() {
        return Err("key rotary execution conflicts with key preparation execution".to_owned());
    }
    if args.input_ids.iter().any(|&id| id < 0) {
        return Err("input token IDs must be nonnegative".to_owned());
    }
    if args.input_ids.len() > MAX_INPUT_IDS {
        return Err(format!("input accepts at most {MAX_INPUT_IDS} IDs"));
    }
    if let Some(max_new_tokens) = args.generation_max_new_tokens {
        if args.input_ids.len() < 2 {
            return Err("generation prompt requires at least two input IDs".to_owned());
        }
        if max_new_tokens == 0 {
            return Err("generation max new tokens must be nonzero".to_owned());
        }
        let occupied_positions = args
            .input_ids
            .len()
            .checked_add(max_new_tokens - 1)
            .ok_or_else(|| "generation position count overflow".to_owned())?;
        if occupied_positions > MAX_INPUT_IDS {
            return Err(format!(
                "generation prompt and max new tokens exceed {MAX_INPUT_IDS}-token capacity"
            ));
        }
        if args.generation_eos_token_id.is_some_and(|id| id < 0) {
            return Err("generation EOS token ID must be nonnegative".to_owned());
        }
        if args.key_rotary_execution.is_some()
            || args.key_preparation_execution.is_some()
            || !matches!(args.score_execution, ScoreExecutionArg::Scalar)
            || !matches!(args.head_execution, HeadExecutionArg::Scalar)
        {
            return Err("reduced generation does not support execution overrides".to_owned());
        }
        return Ok(());
    }
    let prefill_tokens = args
        .prefill_tokens
        .ok_or_else(|| "prefill tokens are required for reduced request replay".to_owned())?;
    if !(2..=args.input_ids.len()).contains(&prefill_tokens) {
        return Err(
            "prefill token count must be at least two and no more than the input count".to_owned(),
        );
    }
    Ok(())
}

fn key_preparation_execution(args: &ReducedArgs) -> IndexKeyPreparationExecution {
    args.key_preparation_execution
        .map(Into::into)
        .or_else(|| {
            args.key_rotary_execution.map(|execution| {
                IndexKeyPreparationExecution::from(IndexKeyRotaryExecution::from(execution))
            })
        })
        .unwrap_or(IndexKeyPreparationExecution::Scalar)
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
        HeadExecutionArg, KeyPreparationExecutionArg, KeyRotaryExecutionArg, ReducedArgs,
        ScoreExecutionArg, key_preparation_execution, read_artifact, validate_args,
    };

    #[test]
    fn rejects_invalid_arguments_before_loading_an_artifact() {
        let args = ReducedArgs {
            artifact: std::path::PathBuf::from("missing"),
            input_ids: vec![0],
            prefill_tokens: Some(1),
            generation_max_new_tokens: None,
            generation_eos_token_id: None,
            score_execution: ScoreExecutionArg::Scalar,
            key_rotary_execution: None,
            key_preparation_execution: None,
            head_execution: HeadExecutionArg::Scalar,
        };
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn legacy_rotary_maps_to_key_preparation() {
        let args = ReducedArgs {
            artifact: std::path::PathBuf::from("missing"),
            input_ids: vec![0, 1],
            prefill_tokens: Some(2),
            generation_max_new_tokens: None,
            generation_eos_token_id: None,
            score_execution: ScoreExecutionArg::Scalar,
            key_rotary_execution: Some(KeyRotaryExecutionArg::Scalar),
            key_preparation_execution: None,
            head_execution: HeadExecutionArg::Scalar,
        };
        assert_eq!(
            key_preparation_execution(&args),
            deepseek::indexer::key::IndexKeyPreparationExecution::Scalar
        );
    }

    #[test]
    fn explicit_scalar_key_preparation_preserves_scalar_selection() {
        let args = ReducedArgs {
            artifact: std::path::PathBuf::from("missing"),
            input_ids: vec![0, 1],
            prefill_tokens: Some(2),
            generation_max_new_tokens: None,
            generation_eos_token_id: None,
            score_execution: ScoreExecutionArg::Scalar,
            key_rotary_execution: None,
            key_preparation_execution: Some(KeyPreparationExecutionArg::Scalar),
            head_execution: HeadExecutionArg::Scalar,
        };
        assert!(validate_args(&args).is_ok());
        assert_eq!(
            key_preparation_execution(&args),
            deepseek::indexer::key::IndexKeyPreparationExecution::Scalar
        );
    }

    #[test]
    fn rejects_conflicting_key_execution_flags() {
        let args = ReducedArgs {
            artifact: std::path::PathBuf::from("missing"),
            input_ids: vec![0, 1],
            prefill_tokens: Some(2),
            generation_max_new_tokens: None,
            generation_eos_token_id: None,
            score_execution: ScoreExecutionArg::Scalar,
            key_rotary_execution: Some(KeyRotaryExecutionArg::Scalar),
            #[cfg(feature = "metal")]
            key_preparation_execution: Some(KeyPreparationExecutionArg::MetalPreFp4),
            #[cfg(not(feature = "metal"))]
            key_preparation_execution: Some(KeyPreparationExecutionArg::Scalar),
            head_execution: HeadExecutionArg::Scalar,
        };
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn rejects_nonregular_artifact() {
        assert!(read_artifact(std::path::Path::new(".")).is_err());
    }
}
