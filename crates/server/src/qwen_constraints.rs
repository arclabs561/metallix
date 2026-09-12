//! Bounded local inputs and diagnostics for constrained Qwen generation.

use std::{fs::File, io::Read, path::Path, time::Instant};

use engine::constraint::{ConstraintLimits, ConstraintStep, JsonConstraintSession};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MAX_SCHEMA_BYTES: usize = 32 * 1024;
const MAX_TOKENIZER_BYTES: usize = 64 * 1024 * 1024;

pub(crate) struct ConstraintRun {
    session: JsonConstraintSession,
    setup_ms: f64,
    sampling_ms: Vec<f64>,
    complete: bool,
    schema_json_sha256: String,
    tokenizer_json_sha256: String,
}

impl ConstraintRun {
    pub(crate) fn load(
        model: &Path,
        schema_path: &Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let schema = read_json(schema_path, MAX_SCHEMA_BYTES)?;
        let config = read_json(&model.join("config.json"), MAX_SCHEMA_BYTES)?;
        let eos = config["eos_token_id"]
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or("configuration requires a numeric EOS token ID")?;
        let vocab = config["vocab_size"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("configuration requires a vocabulary size")?;
        let tokenizer = read_json(&model.join("tokenizer.json"), MAX_TOKENIZER_BYTES)?;
        let schema_json_sha256 = format!("{:x}", Sha256::digest(serde_json::to_vec(&schema)?));
        let tokenizer_json_sha256 =
            format!("{:x}", Sha256::digest(serde_json::to_vec(&tokenizer)?));
        let session = JsonConstraintSession::new(
            &tokenizer,
            eos,
            vocab,
            schema,
            ConstraintLimits {
                max_schema_bytes: MAX_SCHEMA_BYTES,
                max_tokenizer_bytes: MAX_TOKENIZER_BYTES,
                max_output_bytes: 1024 * 1024,
            },
        )?;
        Ok(Self {
            session,
            setup_ms: started.elapsed().as_secs_f64() * 1000.0,
            sampling_ms: Vec::new(),
            complete: false,
            schema_json_sha256,
            tokenizer_json_sha256,
        })
    }

    pub(crate) fn sample(
        &mut self,
        logits: &[f32],
        logprobs: bool,
    ) -> Result<(i32, Option<Value>), Box<dyn std::error::Error>> {
        let started = Instant::now();
        let (step, scores) = if logprobs {
            let (step, scores) = self.session.select_argmax_with_logprobs(logits)?;
            (
                step,
                Some(json!({
                    "model_logprob": scores.model_logprob,
                    "constrained_logprob": scores.constrained_logprob,
                    "allowed_log_mass": scores.allowed_log_mass,
                })),
            )
        } else {
            (self.session.select_argmax(logits)?, None)
        };
        self.sampling_ms
            .push(started.elapsed().as_secs_f64() * 1000.0);
        let token_id = match step {
            ConstraintStep::Token { token_id } => token_id,
            ConstraintStep::Complete { token_id } => {
                self.complete = true;
                token_id
            }
        };
        Ok((i32::try_from(token_id)?, scores))
    }

    /// Samples one grammar-allowed token from caller-supplied uniform entropy.
    ///
    /// The caller retains responsibility for committing its entropy source only
    /// after this method succeeds. The model vocabulary is checked for the
    /// server's signed token-ID boundary before the grammar session can advance.
    pub(crate) fn sample_categorical(
        &mut self,
        logits: &[f32],
        temperature: f64,
        uniform: f64,
        logprobs: bool,
    ) -> Result<(i32, Option<Value>), Box<dyn std::error::Error>> {
        let maximum_token_id = logits
            .len()
            .checked_sub(1)
            .ok_or("sampled constrained generation requires nonempty logits")?;
        i32::try_from(maximum_token_id)
            .map_err(|_| "model vocabulary cannot be represented by server token IDs")?;

        let started = Instant::now();
        let (step, scores) =
            self.session
                .select_categorical_with_logprobs(logits, temperature, uniform)?;
        self.sampling_ms
            .push(started.elapsed().as_secs_f64() * 1000.0);
        let token_id = match step {
            ConstraintStep::Token { token_id } => token_id,
            ConstraintStep::Complete { token_id } => {
                self.complete = true;
                token_id
            }
        };
        let scores = logprobs.then(|| {
            json!({
                "model_logprob": scores.model_logprob,
                "constrained_logprob": scores.constrained_logprob,
                "allowed_log_mass": scores.allowed_log_mass,
                "sampling_logprob": scores.sampling_logprob,
            })
        });
        Ok((i32::try_from(token_id)?, scores))
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.complete
    }

    pub(crate) fn report(
        &self,
        verbose: bool,
        sampled: bool,
    ) -> Result<Value, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let output = if self.complete {
            Some(self.session.validate_complete()?)
        } else {
            None
        };
        let validation_ms = started.elapsed().as_secs_f64() * 1000.0;
        if verbose {
            eprintln!(
                "qwen generation diagnostic: phase=constraint setup_ms={:.3} sampling_total_ms={:.3} validation_ms={validation_ms:.3} complete={}",
                self.setup_ms,
                self.sampling_ms.iter().sum::<f64>(),
                self.complete,
            );
        }
        Ok(json!({
            "status": if self.complete { "validated" } else { "incomplete" },
            "output": output,
            "generated_text": String::from_utf8_lossy(self.session.decoded_bytes()),
            "setup_ms": self.setup_ms,
            "sampling_ms": self.sampling_ms,
            "validation_ms": validation_ms,
            "schema_json_sha256": self.schema_json_sha256,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "identity_scope": "SHA-256 of parsed JSON reserialized by serde_json, not original file bytes; binds constraint inputs, not checkpoint weights",
            "scope": if sampled {
                "LLGuidance masks; independent JSON Schema validation; caller-variate categorical sampling; no forced-token fast-forward; incomplete output is not success"
            } else {
                "LLGuidance masks; independent JSON Schema validation; greedy sampling; no forced-token fast-forward; incomplete output is not success"
            }
        }))
    }

    #[cfg(test)]
    pub(crate) fn decoded_bytes(&self) -> &[u8] {
        self.session.decoded_bytes()
    }

    #[cfg(test)]
    pub(crate) fn from_session(session: JsonConstraintSession) -> Self {
        Self {
            session,
            setup_ms: 0.0,
            sampling_ms: Vec::new(),
            complete: false,
            schema_json_sha256: String::new(),
            tokenizer_json_sha256: String::new(),
        }
    }
}

fn read_json(path: &Path, maximum: usize) -> Result<Value, Box<dyn std::error::Error>> {
    if !path.metadata()?.is_file() {
        return Err("constraint input must be a regular file".into());
    }
    let file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err("constraint input must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(u64::try_from(maximum)? + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > maximum {
        return Err("constraint input exceeds its byte limit".into());
    }
    serde_json::from_slice(&bytes).map_err(|_| "constraint input is not valid JSON".into())
}
