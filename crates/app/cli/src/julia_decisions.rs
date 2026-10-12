//! Typed decisions from a local Julia-1 checkpoint on the native CPU encoder and head.
//!
//! Accepts the same request shape as `decide`; rows, serialization and
//! readout follow the pinned source's `predict_typed` and strict `sequence`.

use std::{
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use clap::Args;
use julia::{
    EncoderInput, HeadInput, JuliaCheckpoint,
    typed::{self, SpecialTokens},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

use crate::qwen_decisions::read_request;

#[derive(Debug, Args)]
pub(crate) struct JuliaDecisionArgs {
    /// Local Julia-1 checkpoint directory (`model.safetensors`, `tokenizer/`, configs).
    #[arg(long)]
    model: PathBuf,
    /// JSON state and named choice, noul, or score questions (at most 1 MiB).
    #[arg(long)]
    request: PathBuf,
}

impl JuliaDecisionArgs {
    pub(crate) fn run(self) -> ExitCode {
        let outcome = read_request(&self.request).and_then(|request_bytes| {
            // Reject malformed requests before the checkpoint load.
            typed::parse_typed_request(&request_bytes).map_err(|error| error.to_string())?;
            JuliaDecider::load(&self.model)?
                .decide(&request_bytes, &self.model.display().to_string())
        });
        match outcome {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("mx decide-julia: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

/// A loaded Julia checkpoint and tokenizer, shared by the CLI and `mx serve`.
pub(crate) struct JuliaDecider {
    tokenizer: Tokenizer,
    tokenizer_sha256: String,
    special: SpecialTokens,
    checkpoint: JuliaCheckpoint,
    load_ms: u128,
}

impl JuliaDecider {
    pub(crate) fn load(model: &Path) -> Result<Self, String> {
        let tokenizer_dir = model.join("tokenizer");
        let tokenizer_bytes = std::fs::read(tokenizer_dir.join("tokenizer.json"))
            .map_err(|error| format!("tokenizer/tokenizer.json could not be read: {error}"))?;
        let tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| format!("tokenizer/tokenizer.json could not be parsed: {error}"))?;
        let special = special_tokens(&tokenizer, &tokenizer_dir)?;
        let started = Instant::now();
        let checkpoint = JuliaCheckpoint::load(model).map_err(|error| error.to_string())?;
        Ok(Self {
            tokenizer,
            tokenizer_sha256: format!("{:x}", Sha256::digest(&tokenizer_bytes)),
            special,
            checkpoint,
            load_ms: started.elapsed().as_millis(),
        })
    }

    /// The `mx decide-julia` receipt for `request_bytes`; `model` labels it.
    pub(crate) fn decide(&self, request_bytes: &[u8], model: &str) -> Result<Value, String> {
        let request =
            typed::parse_typed_request(request_bytes).map_err(|error| error.to_string())?;
        let encode = |text: &str| {
            self.tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|error| format!("text could not be tokenized: {error}"))
        };
        // Encode every question before running any, so a bad question fails fast.
        let serialized = request
            .rows
            .iter()
            .map(|row| typed::sequence(encode, &self.special, row, &request.state_text))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        let checkpoint = &self.checkpoint;
        let mut answers = serde_json::Map::new();
        let mut input_tokens = 0;
        for (row, serialized) in request.rows.iter().zip(serialized) {
            let ids: Vec<u64> = serialized.ids.iter().map(|&id| u64::from(id)).collect();
            let positions = ids.len();
            input_tokens += positions;
            let started = Instant::now();
            let fail = |error: &dyn std::fmt::Display| format!("{:?}: {error}", row.name);
            let encoder = checkpoint.encoder(&ids).map_err(|e| fail(&e))?;
            let hidden = encoder
                .forward(&EncoderInput {
                    input_ids: ids,
                    attention_mask: vec![true; positions],
                })
                .map_err(|e| fail(&e))?;
            let scores = checkpoint
                .head()
                .scores(&HeadInput {
                    hidden,
                    positions,
                    attention_mask: vec![true; positions],
                    marker_mask: vec![true; serialized.markers.len()],
                    marker_pos: serialized.markers.clone(),
                    qtype: serialized.qtype,
                })
                .map_err(|e| fail(&e))?;
            let mut answer = typed::typed_answer(row, &scores).map_err(|e| e.to_string())?;
            answer["input_ids"] = json!(serialized.ids);
            answer["markers"] = json!(serialized.markers);
            answer["forward_ms"] = json!(started.elapsed().as_millis());
            answers.insert(row.name.clone(), answer);
        }
        Ok(json!({
            "schema_version": 1,
            "operation": "julia_typed_decision",
            "model": model,
            "backend": "native CPU float32",
            "checkpoint_load_ms": self.load_ms,
            "encoding": {
                "source": "data.sequence",
                "max_length": typed::MAX_LENGTH,
                "head_length": typed::HEAD_LENGTH,
                "strict": true,
            },
            "calibration": {
                "status": "uncalibrated",
                "method": "option_softmax",
                "note": "Plain softmax over marker scores, as the source predict_typed computes; not calibrated confidence."
            },
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            "provenance": {
                "request_sha256": format!("{:x}", Sha256::digest(request_bytes)),
                "tokenizer_json_sha256": self.tokenizer_sha256,
                "model_identity_scope": "local directory, tensor names/dtypes/shapes and tokenizer bytes; checkpoint weights are not fingerprinted",
            },
        }))
    }
}

/// Resolves `mask_token`, `cls_token` and `sep_token` from `tokenizer_config.json`,
/// as the source constructs its tokenizer.
fn special_tokens(tokenizer: &Tokenizer, dir: &std::path::Path) -> Result<SpecialTokens, String> {
    let config: Value =
        serde_json::from_slice(&std::fs::read(dir.join("tokenizer_config.json")).map_err(
            |error| format!("tokenizer/tokenizer_config.json could not be read: {error}"),
        )?)
        .map_err(|error| format!("tokenizer/tokenizer_config.json could not be parsed: {error}"))?;
    let token = |field: &str| -> Result<(String, u32), String> {
        let text = config[field]
            .as_str()
            .ok_or_else(|| format!("tokenizer config must define {field}"))?;
        let id = tokenizer
            .token_to_id(text)
            .ok_or_else(|| format!("tokenizer has no ID for {field} {text:?}"))?;
        Ok((text.to_owned(), id))
    };
    let (mask_text, mask) = token("mask_token")?;
    Ok(SpecialTokens {
        mask_text,
        mask,
        cls: token("cls_token")?.1,
        sep: token("sep_token")?.1,
    })
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::{Cli, Command};

    #[test]
    fn julia_decisions_take_only_model_and_request() {
        let parsed = Cli::try_parse_from([
            "mx",
            "decide-julia",
            "--model",
            "local",
            "--request",
            "request.json",
        ])
        .expect("julia decision command");
        let Command::DecideJulia(args) = parsed.command else {
            panic!("wrong command")
        };
        assert_eq!(args.model.to_str(), Some("local"));
        for extra in [["--temperature", "1"], ["--context-tokens", "64"]] {
            let mut command_line = vec!["mx", "decide-julia", "--model", "m", "--request", "r"];
            command_line.extend(extra);
            assert!(Cli::try_parse_from(command_line).is_err());
        }
    }
}
