//! `/v1/embeddings` for pplx-embed-v1-late checkpoints: one normalized
//! 128-value vector per scored token, for late-interaction (`MaxSim`) scoring.

use std::{path::Path, time::Instant};

use qwen::late::{LateTask, PPLX_LATE_DOCUMENT_LENGTH, PPLX_LATE_QUERY_LENGTH, PplxLateEncoder};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

const MAX_INPUTS: usize = 64;

/// Unknown fields are rejected, as for the other embedding kinds.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LateRequest {
    // Routing reads `model`; it is accepted here so the body parses whole.
    #[allow(dead_code, reason = "the front process routes on it")]
    model: String,
    input: Input,
    #[serde(default)]
    input_type: InputType,
    #[serde(default)]
    encoding_format: Option<String>,
    /// Accepted for client compatibility and ignored.
    #[serde(default)]
    #[allow(dead_code, reason = "accepted and ignored")]
    user: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Input {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum InputType {
    /// `[D] ` prefix, truncated to 512 IDs, punctuation not scored.
    #[default]
    Document,
    /// `[Q] ` prefix, exactly 32 IDs with mask-token expansion, all scored.
    Query,
}

impl InputType {
    const fn task(self) -> LateTask {
        match self {
            Self::Document => LateTask::Document,
            Self::Query => LateTask::Query,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Document => "document",
            Self::Query => "query",
        }
    }
}

/// The texts to encode and which side of the comparison they are.
#[derive(Debug, PartialEq, Eq)]
struct Prepared {
    texts: Vec<String>,
    input_type: InputType,
}

fn prepare(body: &[u8]) -> Result<Prepared, String> {
    let request: LateRequest = serde_json::from_slice(body).map_err(|error| {
        format!("late-interaction embedding input must be text or a list of texts: {error}")
    })?;
    if request
        .encoding_format
        .as_deref()
        .is_some_and(|format| format != "float")
    {
        return Err("only encoding_format \"float\" is supported".into());
    }
    let texts = match request.input {
        Input::One(text) => vec![text],
        Input::Many(texts) => texts,
    };
    if texts.is_empty() || texts.len() > MAX_INPUTS {
        return Err(format!("input needs 1 through {MAX_INPUTS} texts"));
    }
    if texts.iter().any(String::is_empty) {
        return Err("input texts must be nonempty".into());
    }
    Ok(Prepared {
        texts,
        input_type: request.input_type,
    })
}

/// A loaded pplx-embed-v1-late checkpoint (float32 weights) and its tokenizer.
pub(crate) struct PplxLateEmbedder {
    encoder: PplxLateEncoder,
    tokenizer: Tokenizer,
    tokenizer_sha256: String,
}

impl PplxLateEmbedder {
    pub(crate) fn load(model: &Path) -> Result<Self, String> {
        let tokenizer_bytes = std::fs::read(model.join("tokenizer.json"))
            .map_err(|error| format!("tokenizer.json could not be read: {error}"))?;
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| format!("tokenizer.json could not be parsed: {error}"))?;
        // tokenizer.json truncates to 511 and pads batches; the model's own
        // query and document lengths are applied by the encoder.
        tokenizer
            .with_truncation(None)
            .map_err(|_| String::from("tokenizer truncation could not be disabled"))?;
        tokenizer.with_padding(None);
        let encoder = PplxLateEncoder::load(model).map_err(|error| error.to_string())?;
        Ok(Self {
            encoder,
            tokenizer,
            tokenizer_sha256: format!("{:x}", Sha256::digest(&tokenizer_bytes)),
        })
    }

    /// The `{object: "list", data: [...]}` response with one entry per input,
    /// each holding one vector per scored token; `model` labels it.
    pub(crate) fn embed(&self, body: &[u8], model: &str) -> Result<Value, String> {
        let prepared = prepare(body)?;
        let task = prepared.input_type.task();
        let started = Instant::now();
        let mut data = Vec::with_capacity(prepared.texts.len());
        let mut prompt_tokens = 0;
        for (index, text) in prepared.texts.iter().enumerate() {
            let prompt = task.prompt(text);
            let prefix_chars = prompt.chars().count() - text.chars().count();
            let encoding = self
                .tokenizer
                .encode_char_offsets(prompt.as_str(), true)
                .map_err(|error| format!("input {index} could not be tokenized: {error}"))?;
            let ids = encoding
                .get_ids()
                .iter()
                .map(|&id| {
                    i32::try_from(id).map_err(|_| format!("input {index}: token ID overflows"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let embedded = self
                .encoder
                .encode(task, &ids)
                .map_err(|error| format!("input {index}: {error}"))?;
            prompt_tokens += embedded.input_ids.len();
            // Character offsets into the input text; null for the prefix token
            // and for query-expansion tokens, which have no text.
            let offsets: Vec<Value> = embedded
                .positions
                .iter()
                .map(|&position| match encoding.get_offsets().get(position) {
                    Some(&(start, end)) if start >= prefix_chars => {
                        json!([start - prefix_chars, end - prefix_chars])
                    }
                    _ => Value::Null,
                })
                .collect();
            data.push(json!({
                "object": "embedding",
                "index": index,
                "embedding": embedded.vectors,
                "positions": embedded.positions,
                "offsets": offsets,
                "truncated": ids.len() > embedded.input_ids.len(),
            }));
        }
        Ok(json!({
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            "metallix": {
                "pooling": "none",
                "vectors": "per_token",
                "similarity": "maxsim",
                "normalized": true,
                "dimensions": 128,
                "input_type": prepared.input_type.name(),
                "query_length": PPLX_LATE_QUERY_LENGTH,
                "document_length": PPLX_LATE_DOCUMENT_LENGTH,
                "precision": "float32",
                "embed_ms": started.elapsed().as_millis(),
                "tokenizer_json_sha256": self.tokenizer_sha256,
            },
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_choose_the_side_and_unsupported_shapes_are_rejected() {
        assert_eq!(
            prepare(br#"{"model": "l", "input": ["a", "b"]}"#).unwrap(),
            Prepared {
                texts: vec!["a".into(), "b".into()],
                input_type: InputType::Document,
            }
        );
        assert_eq!(
            prepare(br#"{"model": "l", "input": "q", "input_type": "query", "encoding_format": "float", "user": "u"}"#)
                .unwrap(),
            Prepared {
                texts: vec!["q".into()],
                input_type: InputType::Query,
            }
        );
        let many = format!(
            r#"{{"model": "l", "input": {}}}"#,
            serde_json::to_string(&vec!["x"; MAX_INPUTS + 1]).unwrap()
        );
        for invalid in [
            r#"{"model": "l", "input": [1, 2]}"#,
            r#"{"model": "l", "input": [["chunked"]]}"#,
            r#"{"model": "l", "input": []}"#,
            r#"{"model": "l", "input": ""}"#,
            r#"{"model": "l", "input": "x", "encoding_format": "int8"}"#,
            r#"{"model": "l", "input": "x", "input_type": "passage"}"#,
            r#"{"model": "l", "input": "x", "dimensions": 64}"#,
            r#"{"model": "l", "input": "x", "instruction": "Find code"}"#,
            many.as_str(),
        ] {
            assert!(prepare(invalid.as_bytes()).is_err(), "{invalid}");
        }
    }
}
