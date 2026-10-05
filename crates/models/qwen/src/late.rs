//! pplx-embed-v1-late multi-vector (late-interaction) embeddings.
//!
//! Following the checkpoint's sentence-transformers pipeline (`Transformer`,
//! `Dense`, `MultiVectorMask`, `Normalize`): a query is `[Q] ` plus its text,
//! truncated or padded with the mask token to exactly 32 IDs, all attended and
//! all scored; a document is `[D] ` plus its text, truncated to 512 IDs, with
//! punctuation tokens dropped from scoring. Every scored position's final-norm
//! hidden state is projected from 1024 to 128 values without bias and
//! L2-normalized. Relevance is `MaxSim`: for each query vector, the best dot
//! product over document vectors, summed.

use std::path::Path;

use thiserror::Error;

/// Prompt prefix for queries; an added token (ID 151669) in the tokenizer.
pub const PPLX_LATE_QUERY_PREFIX: &str = "[Q] ";
/// Prompt prefix for documents; an added token (ID 151670) in the tokenizer.
pub const PPLX_LATE_DOCUMENT_PREFIX: &str = "[D] ";
const QUERY_PREFIX_ID: i32 = 151_669;
const DOCUMENT_PREFIX_ID: i32 = 151_670;
/// Every query is exactly this many IDs, prefix included.
pub const PPLX_LATE_QUERY_LENGTH: usize = 32;
/// Documents are truncated to this many IDs, prefix included.
pub const PPLX_LATE_DOCUMENT_LENGTH: usize = 512;
/// The tokenizer's mask token, which pads queries to their fixed length.
pub const PPLX_LATE_EXPANSION_ID: i32 = 151_642;
/// Token IDs of the 32 ASCII punctuation characters, dropped from documents.
pub const PPLX_LATE_SKIPLIST_IDS: [i32; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 25, 26, 27, 28, 29, 30, 31, 58, 59, 60, 61,
    62, 63, 90, 91, 92, 93,
];

/// Which side of a late-interaction comparison a text is encoded for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LateTask {
    /// Fixed-length, expanded with mask tokens, every position scored.
    Query,
    /// Truncated, punctuation dropped from scoring.
    Document,
}

impl LateTask {
    /// The text to tokenize for `text`: its prompt prefix followed by the text.
    ///
    /// Tokenize it with the tokenizer's own truncation and padding disabled:
    /// the pinned `tokenizer.json` truncates to 511 IDs and pads batches, while
    /// the model's lengths are applied by [`late_input_ids`].
    #[must_use]
    pub fn prompt(self, text: &str) -> String {
        let prefix = match self {
            Self::Query => PPLX_LATE_QUERY_PREFIX,
            Self::Document => PPLX_LATE_DOCUMENT_PREFIX,
        };
        format!("{prefix}{text}")
    }

    const fn prefix_id(self) -> i32 {
        match self {
            Self::Query => QUERY_PREFIX_ID,
            Self::Document => DOCUMENT_PREFIX_ID,
        }
    }
}

/// Builds the encoder input from the tokenized [`LateTask::prompt`].
///
/// A query is truncated or padded with [`PPLX_LATE_EXPANSION_ID`] to exactly
/// [`PPLX_LATE_QUERY_LENGTH`] IDs; a document is truncated to
/// [`PPLX_LATE_DOCUMENT_LENGTH`].
///
/// # Errors
///
/// Returns [`LateError::MissingPrefix`] unless the IDs start with the task's
/// prefix token, which catches a tokenizer without the `[Q] `/`[D] ` tokens.
pub fn late_input_ids(task: LateTask, tokenized: &[i32]) -> Result<Vec<i32>, LateError> {
    if tokenized.first() != Some(&task.prefix_id()) {
        return Err(LateError::MissingPrefix {
            expected: task.prefix_id(),
            found: tokenized.first().copied(),
        });
    }
    Ok(match task {
        LateTask::Query => {
            let mut ids: Vec<i32> = tokenized
                .iter()
                .copied()
                .take(PPLX_LATE_QUERY_LENGTH)
                .collect();
            ids.resize(PPLX_LATE_QUERY_LENGTH, PPLX_LATE_EXPANSION_ID);
            ids
        }
        LateTask::Document => tokenized
            .iter()
            .copied()
            .take(PPLX_LATE_DOCUMENT_LENGTH)
            .collect(),
    })
}

/// Positions of `input_ids` that are scored: all of a query; a document's
/// positions whose ID is not in [`PPLX_LATE_SKIPLIST_IDS`].
#[must_use]
pub fn late_scored_positions(task: LateTask, input_ids: &[i32]) -> Vec<usize> {
    input_ids
        .iter()
        .enumerate()
        .filter(|&(_, id)| task == LateTask::Query || !PPLX_LATE_SKIPLIST_IDS.contains(id))
        .map(|(position, _)| position)
        .collect()
}

/// The bias-free `Dense` head: `[outputs, inputs]` float32, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct LateProjection {
    weight: Vec<f32>,
    outputs: usize,
    inputs: usize,
}

impl LateProjection {
    /// Reads `1_Dense/model.safetensors`, which holds one float32 tensor,
    /// `linear.weight`, shaped `[outputs, inputs]`.
    ///
    /// # Errors
    ///
    /// Returns [`LateError::Dense`] for an unreadable file or any other layout.
    pub fn read(path: &Path) -> Result<Self, LateError> {
        let invalid = |reason: &str| LateError::Dense(format!("{}: {reason}", path.display()));
        let bytes = std::fs::read(path).map_err(|error| invalid(&error.to_string()))?;
        let header_len = bytes
            .get(..8)
            .and_then(|prefix| prefix.try_into().ok())
            .map(u64::from_le_bytes)
            .and_then(|len| usize::try_from(len).ok())
            .ok_or_else(|| invalid("truncated header length"))?;
        let header: serde_json::Value = bytes
            .get(8..8 + header_len)
            .and_then(|header| serde_json::from_slice(header).ok())
            .ok_or_else(|| invalid("unreadable header"))?;
        let tensor = &header["linear.weight"];
        let shape: Vec<usize> = tensor["shape"]
            .as_array()
            .map(|dims| {
                dims.iter()
                    .filter_map(|dim| dim.as_u64().and_then(|dim| usize::try_from(dim).ok()))
                    .collect()
            })
            .unwrap_or_default();
        let offsets: Vec<usize> = tensor["data_offsets"]
            .as_array()
            .map(|offsets| {
                offsets
                    .iter()
                    .filter_map(|offset| offset.as_u64().and_then(|o| usize::try_from(o).ok()))
                    .collect()
            })
            .unwrap_or_default();
        let (&[outputs, inputs], &[start, end]) = (shape.as_slice(), offsets.as_slice()) else {
            return Err(invalid("linear.weight needs a 2-D shape and two offsets"));
        };
        if tensor["dtype"] != "F32" || end.checked_sub(start) != Some(outputs * inputs * 4) {
            return Err(invalid(
                "linear.weight must be F32 with a matching byte range",
            ));
        }
        let payload = bytes
            .get(8 + header_len + start..8 + header_len + end)
            .ok_or_else(|| invalid("payload shorter than its header"))?;
        Ok(Self {
            weight: payload
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect(),
            outputs,
            inputs,
        })
    }

    /// Output width (128 for pplx-embed-v1-late).
    #[must_use]
    pub const fn outputs(&self) -> usize {
        self.outputs
    }

    /// Projects the `positions` rows of `hidden` (`[*, inputs]`) and
    /// L2-normalizes each projected vector.
    ///
    /// # Errors
    ///
    /// Returns [`LateError::Shape`] when `hidden` is not whole rows of width
    /// `inputs` or a position is out of range, and [`LateError::ZeroVector`]
    /// for a projection with zero norm.
    pub fn project_normalized(
        &self,
        hidden: &[f32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>, LateError> {
        if self.inputs == 0 || !hidden.len().is_multiple_of(self.inputs) {
            return Err(LateError::Shape);
        }
        let rows: Vec<&[f32]> = hidden.chunks_exact(self.inputs).collect();
        positions
            .iter()
            .map(|&position| {
                let row = rows.get(position).ok_or(LateError::Shape)?;
                let projected: Vec<f64> = self
                    .weight
                    .chunks_exact(self.inputs)
                    .map(|weights| {
                        weights
                            .iter()
                            .zip(*row)
                            .map(|(&w, &h)| f64::from(w) * f64::from(h))
                            .sum()
                    })
                    .collect();
                let norm = projected.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm == 0.0 {
                    return Err(LateError::ZeroVector { position });
                }
                #[allow(
                    clippy::cast_possible_truncation,
                    reason = "a unit-vector component fits f32"
                )]
                Ok(projected.iter().map(|v| (v / norm) as f32).collect())
            })
            .collect()
    }
}

/// `MaxSim` relevance: for each query vector the largest dot product with any
/// document vector, summed over query vectors.
#[must_use]
pub fn maxsim(query: &[Vec<f32>], document: &[Vec<f32>]) -> f64 {
    query
        .iter()
        .map(|q| {
            document
                .iter()
                .map(|d| {
                    q.iter()
                        .zip(d)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum()
                })
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .sum()
}

/// One encoded text: its encoder input IDs, the scored positions, and one
/// normalized vector per scored position.
#[derive(Clone, Debug, PartialEq)]
pub struct LateEmbedding {
    /// Encoder input, after prefixing, truncation and query expansion.
    pub input_ids: Vec<i32>,
    /// Scored positions within `input_ids`.
    pub positions: Vec<usize>,
    /// One L2-normalized vector per scored position.
    pub vectors: Vec<Vec<f32>>,
}

/// A loaded pplx-embed-v1-late checkpoint: the bidirectional Qwen3 decoder and
/// its `Dense` head.
#[cfg(feature = "metal")]
pub struct PplxLateEncoder {
    weights: crate::metal::Qwen3MlxWeights,
    projection: LateProjection,
}

#[cfg(feature = "metal")]
impl PplxLateEncoder {
    /// Loads the decoder from `model_dir` and the head from
    /// `model_dir/1_Dense/model.safetensors`.
    ///
    /// # Errors
    ///
    /// Returns [`LateError`] when either part fails to load, the decoder is not
    /// bidirectional, or the head's input width differs from the hidden size.
    pub fn load(model_dir: &Path) -> Result<Self, LateError> {
        let weights = crate::metal::Qwen3MlxWeights::load(model_dir)
            .map_err(|error| LateError::Load(error.to_string()))?;
        if weights.attention() != crate::Qwen3Attention::Bidirectional {
            return Err(LateError::Load(
                "pplx-embed-v1-late needs a bidirectional checkpoint".into(),
            ));
        }
        let projection = LateProjection::read(&model_dir.join("1_Dense/model.safetensors"))?;
        let hidden = weights.embedding_shape().get(1).copied().unwrap_or(0);
        if usize::try_from(hidden).ok() != Some(projection.inputs) {
            return Err(LateError::Shape);
        }
        Ok(Self {
            weights,
            projection,
        })
    }

    /// Encodes one tokenized [`LateTask::prompt`].
    ///
    /// # Errors
    ///
    /// Returns [`LateError`] for a missing prefix, a forward failure, or a
    /// zero-norm projection.
    pub fn encode(&self, task: LateTask, tokenized: &[i32]) -> Result<LateEmbedding, LateError> {
        let input_ids = late_input_ids(task, tokenized)?;
        let hidden = self
            .weights
            .hidden_states(&input_ids)
            .map_err(|error| LateError::Forward(error.to_string()))?;
        let positions = late_scored_positions(task, &input_ids);
        let vectors = self.projection.project_normalized(&hidden, &positions)?;
        Ok(LateEmbedding {
            input_ids,
            positions,
            vectors,
        })
    }
}

/// A late-interaction encoding could not be computed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LateError {
    /// The tokenized prompt does not start with the task's prefix token.
    #[error("tokenized prompt must start with prefix token {expected}, found {found:?}")]
    MissingPrefix {
        /// Prefix token ID for the task.
        expected: i32,
        /// First supplied ID, if any.
        found: Option<i32>,
    },
    /// The `Dense` head file is missing or malformed.
    #[error("Dense head: {0}")]
    Dense(String),
    /// Hidden states or positions do not fit the head's input width.
    #[error("hidden states do not match the Dense head's input width")]
    Shape,
    /// A projected vector has zero norm and cannot be normalized.
    #[error("projected vector at position {position} has zero norm")]
    ZeroVector {
        /// Offending position.
        position: usize,
    },
    /// The decoder checkpoint could not be loaded.
    #[error("{0}")]
    Load(String),
    /// The decoder forward pass failed.
    #[error("{0}")]
    Forward(String),
}

#[cfg(test)]
mod tests {
    use super::{
        LateError, LateProjection, LateTask, PPLX_LATE_EXPANSION_ID, late_input_ids,
        late_scored_positions, maxsim,
    };

    #[test]
    fn queries_are_padded_or_truncated_to_32_ids_and_documents_to_512() {
        assert_eq!(LateTask::Query.prompt("x"), "[Q] x");
        assert_eq!(LateTask::Document.prompt("x"), "[D] x");
        let short = late_input_ids(LateTask::Query, &[151_669, 7, 8]).unwrap();
        assert_eq!(short.len(), 32);
        assert_eq!(short[..3], [151_669, 7, 8]);
        assert!(short[3..].iter().all(|&id| id == PPLX_LATE_EXPANSION_ID));
        let long: Vec<i32> = std::iter::once(151_669).chain(100..140).collect();
        assert_eq!(late_input_ids(LateTask::Query, &long).unwrap(), long[..32]);

        let document: Vec<i32> = std::iter::once(151_670).chain(100..700).collect();
        assert_eq!(
            late_input_ids(LateTask::Document, &document).unwrap(),
            document[..512]
        );
        assert_eq!(
            late_input_ids(LateTask::Document, &[151_670, 5]).unwrap(),
            [151_670, 5]
        );
        assert!(matches!(
            late_input_ids(LateTask::Document, &[151_669, 5]),
            Err(LateError::MissingPrefix {
                expected: 151_670,
                found: Some(151_669)
            })
        ));
        assert!(matches!(
            late_input_ids(LateTask::Query, &[]),
            Err(LateError::MissingPrefix { found: None, .. })
        ));
    }

    #[test]
    fn only_document_punctuation_ids_are_dropped_from_scoring() {
        // 13 is ".", 30 is "?", 1773 is a non-ASCII full stop.
        let ids = [151_670, 500, 13, 30, 1773];
        assert_eq!(late_scored_positions(LateTask::Document, &ids), [0, 1, 4]);
        assert_eq!(
            late_scored_positions(LateTask::Query, &ids),
            [0, 1, 2, 3, 4]
        );
    }

    fn write_dense(path: &std::path::Path, header: &str, payload: &[f32]) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        for value in payload {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn dense_head_projects_and_normalizes_rows_and_rejects_bad_files() {
        let dir = std::env::temp_dir().join(format!("metallix-late-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("dense.safetensors");
        // [[1, 0, 0], [0, 2, 0]]: picks the first value and doubles the second.
        write_dense(
            &path,
            r#"{"linear.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
            &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
        );
        let projection = LateProjection::read(&path).unwrap();
        assert_eq!(projection.outputs(), 2);
        // Rows (3, 2, 9) and (0, 0, 1): the first projects to (3, 4) -> (0.6, 0.8).
        let hidden = [3.0, 2.0, 9.0, 0.0, 0.0, 1.0];
        assert_eq!(
            projection.project_normalized(&hidden, &[0]).unwrap(),
            [vec![0.6, 0.8]]
        );
        assert!(matches!(
            projection.project_normalized(&hidden, &[1]),
            Err(LateError::ZeroVector { position: 1 })
        ));
        assert!(matches!(
            projection.project_normalized(&hidden, &[2]),
            Err(LateError::Shape)
        ));
        assert!(matches!(
            projection.project_normalized(&hidden[..5], &[0]),
            Err(LateError::Shape)
        ));

        for header in [
            r#"{"linear.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,24]}}"#,
            r#"{"linear.weight":{"dtype":"F32","shape":[2,4],"data_offsets":[0,24]}}"#,
            r#"{"other":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#,
        ] {
            write_dense(&path, header, &[0.0; 6]);
            assert!(matches!(
                LateProjection::read(&path),
                Err(LateError::Dense(_))
            ));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn maxsim_sums_each_query_vectors_best_match() {
        let query = [vec![1.0, 0.0], vec![0.0, 1.0]];
        let document = [vec![0.6, 0.8], vec![1.0, 0.0]];
        // Best for (1, 0) is 1.0; best for (0, 1) is 0.8.
        assert!((maxsim(&query, &document) - 1.8).abs() < 1e-6);
    }
}
