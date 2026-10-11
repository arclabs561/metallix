//! `/v1/embeddings` for pplx-embed-context checkpoints: each input is one
//! document given as a list of chunks, and the response holds one vector per
//! chunk, pooled with the whole document as context.

use std::{path::Path, time::Instant};

use qwen::{
    Qwen3Attention,
    embedding::{join_context_chunks, quantize_binary, quantize_int8_tanh},
    metal::Qwen3MlxWeights,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

const MAX_DOCUMENTS: usize = 64;
const MAX_CHUNKS: usize = 256;

/// Unknown fields are rejected, as for Qwen3-Embedding requests.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextRequest {
    // Routing reads `model`; it is accepted here so the body parses whole.
    #[allow(dead_code, reason = "the front process routes on it")]
    model: String,
    input: Vec<Vec<String>>,
    #[serde(default)]
    encoding_format: Option<String>,
    /// Accepted for client compatibility and ignored.
    #[serde(default)]
    #[allow(dead_code, reason = "accepted and ignored")]
    user: Option<String>,
}

/// How each pooled chunk vector is returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoding {
    /// The mean-pooled float vector, before the model's quantization.
    Float,
    /// The model's int8 output: `clamp(round(tanh(x) * 127), -128, 127)`.
    Int8,
    /// The model's binary output: `1` where `x >= 0`, else `-1`.
    Binary,
}

impl Encoding {
    const fn name(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Int8 => "int8",
            Self::Binary => "binary",
        }
    }
}

/// Each document's chunks joined into one sequence, and the output encoding.
#[derive(Debug, PartialEq, Eq)]
struct Prepared {
    documents: Vec<String>,
    chunk_counts: Vec<usize>,
    encoding: Encoding,
}

fn prepare(body: &[u8]) -> Result<Prepared, String> {
    let request: ContextRequest = serde_json::from_slice(body).map_err(|error| {
        format!("contextual embedding input must be a list of documents, each a list of chunk strings: {error}")
    })?;
    let encoding = match request.encoding_format.as_deref() {
        None | Some("float") => Encoding::Float,
        Some("int8") => Encoding::Int8,
        Some("binary") => Encoding::Binary,
        Some(other) => {
            return Err(format!(
                "encoding_format {other:?} is unsupported; use \"float\", \"int8\" or \"binary\""
            ));
        }
    };
    if request.input.is_empty() || request.input.len() > MAX_DOCUMENTS {
        return Err(format!("input needs 1 through {MAX_DOCUMENTS} documents"));
    }
    let mut documents = Vec::with_capacity(request.input.len());
    let mut chunk_counts = Vec::with_capacity(request.input.len());
    for (index, chunks) in request.input.iter().enumerate() {
        if chunks.is_empty() || chunks.len() > MAX_CHUNKS {
            return Err(format!(
                "document {index} needs 1 through {MAX_CHUNKS} chunks"
            ));
        }
        let chunks: Vec<&str> = chunks.iter().map(String::as_str).collect();
        documents.push(
            join_context_chunks(&chunks).map_err(|error| format!("document {index}: {error}"))?,
        );
        chunk_counts.push(chunks.len());
    }
    Ok(Prepared {
        documents,
        chunk_counts,
        encoding,
    })
}

/// A loaded pplx-embed-context checkpoint (float32 weights) and its tokenizer.
pub(crate) struct PplxContextEmbedder {
    weights: Qwen3MlxWeights,
    tokenizer: Tokenizer,
    tokenizer_sha256: String,
}

impl PplxContextEmbedder {
    pub(crate) fn load(model: &Path) -> Result<Self, String> {
        let tokenizer_bytes = std::fs::read(model.join("tokenizer.json"))
            .map_err(|error| format!("tokenizer.json could not be read: {error}"))?;
        let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)
            .map_err(|error| format!("tokenizer.json could not be parsed: {error}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|_| String::from("tokenizer truncation could not be disabled"))?;
        tokenizer.with_padding(None);
        // The published checkpoint is float32 and served as stored.
        let weights = Qwen3MlxWeights::load(model).map_err(|error| error.to_string())?;
        if weights.attention() != Qwen3Attention::Bidirectional {
            return Err(
                "pplx_context needs a bidirectional checkpoint (bidirectional_pplx_qwen3)".into(),
            );
        }
        Ok(Self {
            weights,
            tokenizer,
            tokenizer_sha256: format!("{:x}", Sha256::digest(&tokenizer_bytes)),
        })
    }

    /// The `{object: "list", data: [...]}` response with one entry per chunk,
    /// in document then chunk order; `model` labels it.
    pub(crate) fn embed(&self, body: &[u8], model: &str) -> Result<Value, String> {
        let prepared = prepare(body)?;
        let started = Instant::now();
        let mut data = Vec::new();
        let mut prompt_tokens = 0;
        let mut width = 0;
        for (document, (text, &chunks)) in prepared
            .documents
            .iter()
            .zip(&prepared.chunk_counts)
            .enumerate()
        {
            let ids = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|error| format!("document {document} could not be tokenized: {error}"))?
                .get_ids()
                .iter()
                .map(|&id| {
                    i32::try_from(id)
                        .map_err(|_| format!("document {document}: token ID overflows"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            prompt_tokens += ids.len();
            let pooled = self
                .weights
                .embed_context_chunks(&ids)
                .map_err(|error| format!("document {document}: {error}"))?;
            if pooled.len() != chunks {
                return Err(format!(
                    "document {document}: tokenizer produced {} chunks, expected {chunks}",
                    pooled.len()
                ));
            }
            for (chunk, vector) in pooled.iter().enumerate() {
                width = vector.len();
                let embedding = match prepared.encoding {
                    Encoding::Float => json!(vector),
                    Encoding::Int8 => json!(quantize_int8_tanh(vector)),
                    Encoding::Binary => json!(quantize_binary(vector)),
                };
                data.push(json!({
                    "object": "embedding",
                    "index": data.len(),
                    "document": document,
                    "chunk": chunk,
                    "embedding": embedding,
                }));
            }
        }
        Ok(json!({
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
            "metallix": {
                "pooling": "mean_per_chunk",
                "context": "document",
                "normalized": false,
                "encoding": prepared.encoding.name(),
                "dimensions": width,
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
    fn documents_join_their_chunks_and_unsupported_shapes_are_rejected() {
        let prepared = prepare(
            br#"{"model": "c", "input": [["a b", "c", ""], ["one"]], "encoding_format": "int8"}"#,
        )
        .unwrap();
        assert_eq!(
            prepared,
            Prepared {
                documents: vec!["a b<|endoftext|>c<|endoftext|>".into(), "one".into()],
                chunk_counts: vec![3, 1],
                encoding: Encoding::Int8,
            }
        );
        assert_eq!(
            prepare(br#"{"model": "c", "input": [["x"]]}"#)
                .unwrap()
                .encoding,
            Encoding::Float
        );
        assert_eq!(
            prepare(
                br#"{"model": "c", "input": [["x"]], "encoding_format": "binary", "user": "u"}"#
            )
            .unwrap()
            .encoding,
            Encoding::Binary
        );

        let many_documents = format!(
            r#"{{"model": "c", "input": {}}}"#,
            serde_json::to_string(&vec![vec!["x"]; MAX_DOCUMENTS + 1]).unwrap()
        );
        let many_chunks = format!(
            r#"{{"model": "c", "input": [{}]}}"#,
            serde_json::to_string(&vec!["x"; MAX_CHUNKS + 1]).unwrap()
        );
        for invalid in [
            r#"{"model": "c", "input": "one text"}"#,
            r#"{"model": "c", "input": ["flat", "texts"]}"#,
            r#"{"model": "c", "input": []}"#,
            r#"{"model": "c", "input": [[]]}"#,
            r#"{"model": "c", "input": [["a<|endoftext|>b"]]}"#,
            r#"{"model": "c", "input": [["x"]], "encoding_format": "base64"}"#,
            r#"{"model": "c", "input": [["x"]], "dimensions": 256}"#,
            r#"{"model": "c", "input": [["x"]], "input_type": "query"}"#,
            many_documents.as_str(),
            many_chunks.as_str(),
        ] {
            assert!(prepare(invalid.as_bytes()).is_err(), "{invalid}");
        }
    }
}
