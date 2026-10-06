//! Bounded local inputs and diagnostics for constrained Qwen generation.

use std::{fs::File, io::Read, path::Path, time::Instant};

use engine::constraint::{
    ConstraintLimits, ConstraintStep, JsonConstraintCheckpoint, JsonConstraintSession,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MAX_SCHEMA_BYTES: usize = 32 * 1024;
const MAX_TOKENIZER_BYTES: usize = 64 * 1024 * 1024;

/// Explicit local schema input; both forms share compilation and validation.
#[derive(Clone, Copy)]
pub(crate) enum SchemaSource<'a> {
    File(&'a Path),
    Inline(&'a str),
}

impl SchemaSource<'_> {
    fn read(self) -> Result<Value, Box<dyn std::error::Error>> {
        match self {
            Self::File(path) => read_json(path, MAX_SCHEMA_BYTES),
            Self::Inline(text) => {
                if text.len() > MAX_SCHEMA_BYTES {
                    return Err("inline JSON schema exceeds its 32 KiB byte limit".into());
                }
                serde_json::from_str(text)
                    .map_err(|_| "inline JSON schema is not valid JSON".into())
            }
        }
    }
}

pub(crate) struct ConstraintRun {
    session: JsonConstraintSession,
    setup_ms: f64,
    sampling_ms: Vec<f64>,
    complete: bool,
    schema_json_sha256: String,
    tokenizer_json_sha256: String,
}

/// Transactional controller state for one constrained Qwen branch.
///
/// Model KV state and the sampler RNG live in their respective owners; this
/// checkpoint only covers grammar/output state and its diagnostic timing
/// cursor.
pub(crate) struct ConstraintCheckpoint {
    session: JsonConstraintCheckpoint,
    complete: bool,
    sampling_len: usize,
}

/// Result of the fixed, local schedule semantic check.
///
/// The schema still owns the surrounding JSON shape. This verifier deliberately
/// accepts only `{ "intervals": [{"start": number, "end": number}, ...] }`
/// so its meaning is stable and no executable verifier surface is needed.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct ScheduleVerification {
    pub(crate) interval_count: usize,
    pub(crate) rejection: Option<&'static str>,
}

impl ScheduleVerification {
    pub(crate) const fn accepted(&self) -> bool {
        self.rejection.is_none()
    }
}

pub(crate) fn verify_non_overlapping_schedule(value: &Value) -> ScheduleVerification {
    const MAX_INTERVALS: usize = 128;
    let Some(intervals) = value.get("intervals").and_then(Value::as_array) else {
        return ScheduleVerification {
            interval_count: 0,
            rejection: Some("missing_intervals"),
        };
    };
    if intervals.is_empty() {
        return ScheduleVerification {
            interval_count: 0,
            rejection: Some("empty_intervals"),
        };
    }
    if intervals.len() > MAX_INTERVALS {
        return ScheduleVerification {
            interval_count: intervals.len(),
            rejection: Some("too_many_intervals"),
        };
    }

    let mut bounds = Vec::with_capacity(intervals.len());
    for interval in intervals {
        let Some(start) = interval.get("start").and_then(Value::as_f64) else {
            return ScheduleVerification {
                interval_count: intervals.len(),
                rejection: Some("interval_start_not_finite_number"),
            };
        };
        let Some(end) = interval.get("end").and_then(Value::as_f64) else {
            return ScheduleVerification {
                interval_count: intervals.len(),
                rejection: Some("interval_end_not_finite_number"),
            };
        };
        if !start.is_finite() || !end.is_finite() {
            return ScheduleVerification {
                interval_count: intervals.len(),
                rejection: Some("interval_bound_not_finite"),
            };
        }
        if start >= end {
            return ScheduleVerification {
                interval_count: intervals.len(),
                rejection: Some("interval_not_positive_width"),
            };
        }
        bounds.push((start, end));
    }
    bounds.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.total_cmp(&right.1)));
    if bounds.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return ScheduleVerification {
            interval_count: intervals.len(),
            rejection: Some("overlapping_intervals"),
        };
    }
    ScheduleVerification {
        interval_count: intervals.len(),
        rejection: None,
    }
}

impl ConstraintRun {
    pub(crate) fn identity(&self) -> Value {
        json!({
            "schema_json_sha256": self.schema_json_sha256,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "scope": "reserialized parsed JSON; not checkpoint weights",
        })
    }

    pub(crate) fn checkpoint(&self) -> ConstraintCheckpoint {
        ConstraintCheckpoint {
            session: self.session.checkpoint(),
            complete: self.complete,
            sampling_len: self.sampling_ms.len(),
        }
    }

    pub(crate) fn restore(
        &mut self,
        checkpoint: ConstraintCheckpoint,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.session.restore(checkpoint.session)?;
        self.complete = checkpoint.complete;
        self.sampling_ms.truncate(checkpoint.sampling_len);
        Ok(())
    }

    pub(crate) fn load(
        model: &Path,
        schema_source: SchemaSource<'_>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let started = Instant::now();
        let schema = schema_source.read()?;
        let config = read_json(&model.join("config.json"), MAX_SCHEMA_BYTES)?;
        // The grammar ends output on one ID; any other stop the checkpoint
        // lists still ends the turn when the decode loop samples it.
        let eos = chat_format::StopTokens::load(model)?.end_turn().get();
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

    pub(crate) fn validated_output(&self) -> Result<Value, Box<dyn std::error::Error>> {
        if !self.complete {
            return Err("cannot verify an incomplete grammar output".into());
        }
        Ok(self.session.validate_complete()?)
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
        let verification = if self.complete {
            json!({
                "status": "passed",
                "kind": "independent_json_schema",
                "grammar": "tokenizer_aware_mask",
                "scope": "decoded JSON value; does not validate model quality or checkpoint identity",
            })
        } else {
            json!({
                "status": "not_run",
                "kind": "independent_json_schema",
                "reason": "grammar_incomplete",
            })
        };
        Ok(json!({
            "status": if self.complete { "validated" } else { "incomplete" },
            "verification": verification,
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

#[cfg(test)]
mod source_tests {
    use super::{MAX_SCHEMA_BYTES, SchemaSource, verify_non_overlapping_schedule};
    use serde_json::json;

    #[test]
    fn inline_schema_keeps_its_constraints() {
        let schema = SchemaSource::Inline(r#"{"type":"string","enum":["ready","waiting"]}"#)
            .read()
            .unwrap();
        assert_eq!(
            schema,
            json!({"type": "string", "enum": ["ready", "waiting"]})
        );
        assert_eq!(SchemaSource::Inline("false").read().unwrap(), json!(false));
    }

    #[test]
    fn file_and_inline_schema_have_identical_semantics() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/constraints/record.json");
        let file = SchemaSource::File(&path).read().unwrap();
        let inline =
            SchemaSource::Inline(include_str!("../../../fixtures/constraints/record.json"))
                .read()
                .unwrap();
        assert_eq!(file, inline);
        assert_eq!(file["additionalProperties"], json!(false));
        assert_eq!(file["required"], json!(["status", "count"]));
    }

    #[test]
    fn inline_schema_rejects_invalid_and_oversized_inputs_without_echoing_them() {
        let error = SchemaSource::Inline("private invalid text")
            .read()
            .unwrap_err()
            .to_string();
        assert_eq!(error, "inline JSON schema is not valid JSON");
        let oversized = " ".repeat(MAX_SCHEMA_BYTES + 1);
        assert!(
            SchemaSource::Inline(&oversized)
                .read()
                .unwrap_err()
                .to_string()
                .contains("32 KiB")
        );
        let exact = format!("{}true", " ".repeat(MAX_SCHEMA_BYTES - 4));
        assert_eq!(SchemaSource::Inline(&exact).read().unwrap(), json!(true));
    }

    #[test]
    fn schedule_verifier_enforces_nonempty_half_open_positive_intervals() {
        let accepted = verify_non_overlapping_schedule(&json!({
            "intervals": [{"start": 0, "end": 1}, {"start": 1, "end": 2.5}]
        }));
        assert!(accepted.accepted());
        assert_eq!(accepted.interval_count, 2);

        for (value, rejection) in [
            (json!({"intervals": []}), "empty_intervals"),
            (
                json!({"intervals": [{"start": 2, "end": 2}]}),
                "interval_not_positive_width",
            ),
            (
                json!({"intervals": [{"start": 0, "end": 2}, {"start": 1, "end": 3}]}),
                "overlapping_intervals",
            ),
            (
                json!({"intervals": [{"start": 0, "end": 4}, {"start": 1, "end": 2}]}),
                "overlapping_intervals",
            ),
            (
                json!({"intervals": [{"start": "zero", "end": 1}]}),
                "interval_start_not_finite_number",
            ),
        ] {
            let result = verify_non_overlapping_schedule(&value);
            assert_eq!(result.rejection, Some(rejection));
        }
    }
}
