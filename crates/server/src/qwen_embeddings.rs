//! `/v1/embeddings` for Qwen3-Embedding checkpoints: the common
//! `{model, input, dimensions}` request plus `input_type` and `instruction`,
//! embedded with last-token pooling.

use std::{path::Path, time::Instant};

use qwen::{
    embedding::{QWEN3_EMBEDDING_WEB_SEARCH_TASK, qwen3_embedding_query},
    metal::Qwen3MlxWeights,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

const MAX_INPUTS: usize = 64;

/// Unknown fields are rejected, so a client asking for an output this server
/// does not produce (per-token vectors, contextual chunks) gets an error
/// instead of a pooled vector it did not ask for.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRequest {
    // Routing reads `model`; it is accepted here so the body parses whole.
    #[allow(dead_code, reason = "the front process routes on it")]
    model: String,
    input: Input,
    #[serde(default)]
    dimensions: Option<usize>,
    #[serde(default)]
    encoding_format: Option<String>,
    #[serde(default)]
    input_type: InputType,
    #[serde(default)]
    instruction: Option<String>,
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
    /// Embedded as written, as the model card does for passages.
    #[default]
    Document,
    /// Prefixed with a one-sentence task instruction, as the model card does.
    Query,
}

/// The texts to embed, already formatted for the model, and the requested width.
#[derive(Debug, PartialEq, Eq)]
struct Prepared {
    texts: Vec<String>,
    dimensions: Option<usize>,
    input_type: InputType,
    instruction: Option<String>,
}

fn prepare(body: &[u8]) -> Result<Prepared, String> {
    let request: EmbeddingRequest = serde_json::from_slice(body).map_err(|error| {
        format!("embedding request must be text input(s); token arrays are unsupported: {error}")
    })?;
    if request
        .encoding_format
        .as_deref()
        .is_some_and(|format| format != "float")
    {
        return Err("only encoding_format \"float\" is supported".into());
    }
    let inputs = match request.input {
        Input::One(text) => vec![text],
        Input::Many(texts) => texts,
    };
    if inputs.is_empty() || inputs.len() > MAX_INPUTS {
        return Err(format!("input needs 1 through {MAX_INPUTS} texts"));
    }
    if inputs.iter().any(String::is_empty) {
        return Err("input texts must be nonempty".into());
    }
    let (texts, instruction) = match (request.input_type, request.instruction) {
        (InputType::Document, Some(_)) => {
            return Err("instruction applies only to input_type \"query\"".into());
        }
        (InputType::Document, None) => (inputs, None),
        (InputType::Query, instruction) => {
            let task = instruction.unwrap_or_else(|| QWEN3_EMBEDDING_WEB_SEARCH_TASK.to_owned());
            if task.trim().is_empty() {
                return Err("instruction must be nonempty".into());
            }
            let texts = inputs
                .iter()
                .map(|query| qwen3_embedding_query(&task, query))
                .collect();
            (texts, Some(task))
        }
    };
    Ok(Prepared {
        texts,
        dimensions: request.dimensions,
        input_type: request.input_type,
        instruction,
    })
}

/// A loaded Qwen3-Embedding checkpoint with float32 weights and its tokenizer.
pub(crate) struct QwenEmbedder {
    weights: Qwen3MlxWeights,
    tokenizer: Tokenizer,
    tokenizer_sha256: String,
}

impl QwenEmbedder {
    pub(crate) fn load(model: &Path) -> Result<Self, String> {
        let tokenizer_bytes = std::fs::read(model.join("tokenizer.json"))
            .map_err(|error| format!("tokenizer.json could not be read: {error}"))?;
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| format!("tokenizer.json could not be parsed: {error}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|_| String::from("tokenizer truncation could not be disabled"))?;
        tokenizer.with_padding(None);
        let mut weights = Qwen3MlxWeights::load(model).map_err(|error| error.to_string())?;
        // The served path uses the float32 weights qualified against the source oracle.
        weights
            .prepare_float32()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            weights,
            tokenizer,
            tokenizer_sha256: format!("{:x}", Sha256::digest(&tokenizer_bytes)),
        })
    }

    /// The `{object: "list", data: [...]}` embedding response for `body`;
    /// `model` labels it.
    pub(crate) fn embed(&self, body: &[u8], model: &str) -> Result<Value, String> {
        let prepared = prepare(body)?;
        let started = Instant::now();
        let mut data = Vec::with_capacity(prepared.texts.len());
        let mut prompt_tokens = 0;
        let mut width = 0;
        for (index, text) in prepared.texts.iter().enumerate() {
            // With special tokens, so the tokenizer appends the pooled <|endoftext|>.
            let ids = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|error| format!("input {index} could not be tokenized: {error}"))?
                .get_ids()
                .iter()
                .map(|&id| {
                    i32::try_from(id).map_err(|_| format!("input {index}: token ID overflows"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            prompt_tokens += ids.len();
            let embedding = self
                .weights
                .embed(&ids, prepared.dimensions)
                .map_err(|error| format!("input {index}: {error}"))?;
            width = embedding.len();
            data.push(json!({"object": "embedding", "index": index, "embedding": embedding}));
        }
        Ok(json!({
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            "metallix": {
                "pooling": "last_token",
                "normalized": true,
                "dimensions": width,
                "input_type": match prepared.input_type {
                    InputType::Document => "document",
                    InputType::Query => "query",
                },
                "instruction": prepared.instruction,
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
    fn requests_follow_the_card_recipe_and_reject_unsupported_shapes() {
        let documents = prepare(br#"{"model": "e", "input": ["a b", "c"]}"#).unwrap();
        assert_eq!(documents.texts, ["a b", "c"]);
        assert_eq!(
            (
                documents.input_type,
                documents.instruction,
                documents.dimensions
            ),
            (InputType::Document, None, None)
        );

        let query = prepare(br#"{"model": "e", "input": "Explain gravity", "input_type": "query", "dimensions": 256}"#).unwrap();
        assert_eq!(
            query.texts,
            [
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:Explain gravity"
            ]
        );
        assert_eq!(query.dimensions, Some(256));
        let custom = prepare(
            br#"{"model": "e", "input": "x", "input_type": "query", "instruction": "Find code"}"#,
        )
        .unwrap();
        assert_eq!(custom.texts, ["Instruct: Find code\nQuery:x"]);

        let many = format!(
            r#"{{"model": "e", "input": {}}}"#,
            serde_json::to_string(&vec!["x"; 65]).unwrap()
        );
        for invalid in [
            r#"{"model": "e", "input": [1, 2]}"#,
            r#"{"model": "e", "input": []}"#,
            r#"{"model": "e", "input": ""}"#,
            r#"{"model": "e", "input": "x", "encoding_format": "base64"}"#,
            r#"{"model": "e", "input": "x", "instruction": "Find code"}"#,
            r#"{"model": "e", "input": "x", "input_type": "query", "instruction": " "}"#,
            r#"{"model": "e", "input": "x", "input_type": "passage"}"#,
            r#"{"model": "e", "input": "x", "output": "tokens"}"#,
            many.as_str(),
        ] {
            assert!(prepare(invalid.as_bytes()).is_err(), "{invalid}");
        }
        assert!(
            prepare(br#"{"model": "e", "input": "x", "encoding_format": "float", "user": "u"}"#)
                .is_ok()
        );
    }
}
