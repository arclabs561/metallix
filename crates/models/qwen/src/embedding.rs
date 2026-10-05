//! Qwen3-Embedding text embeddings from the Qwen3 decoder.
//!
//! The pinned Qwen3-Embedding-0.6B pools the last position's final-norm hidden
//! state and L2-normalizes it. Its `tokenizer.json` post-processor appends
//! `<|endoftext|>` to every sequence, so that token is the pooled position:
//! callers encode with special tokens and pass the IDs unchanged. Queries
//! carry a one-sentence task instruction; documents are embedded as is.

use thiserror::Error;

/// `<|endoftext|>`, which the embedding tokenizer appends to every sequence.
pub const QWEN3_EMBEDDING_EOS_ID: i32 = 151_643;

/// Smallest output dimension the model card supports for truncation (MRL).
pub const QWEN3_EMBEDDING_MIN_DIMENSIONS: usize = 32;

/// Task instruction used by the model card's retrieval example.
pub const QWEN3_EMBEDDING_WEB_SEARCH_TASK: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Formats a query with its task instruction, exactly as the model card does.
///
/// There is no space after `Query:`. Documents take no instruction.
#[must_use]
pub fn qwen3_embedding_query(task: &str, query: &str) -> String {
    format!("Instruct: {task}\nQuery:{query}")
}

/// Checks that `input_ids` end with the appended `<|endoftext|>`.
///
/// # Errors
///
/// Returns [`Qwen3EmbeddingError::MissingEndOfText`] when the IDs are empty or
/// were encoded without the tokenizer's special-token template.
pub fn check_embedding_input(input_ids: &[i32]) -> Result<(), Qwen3EmbeddingError> {
    match input_ids.last() {
        Some(&QWEN3_EMBEDDING_EOS_ID) => Ok(()),
        last => Err(Qwen3EmbeddingError::MissingEndOfText {
            last: last.copied(),
        }),
    }
}

/// Truncates a pooled hidden state to `dimensions` and L2-normalizes it.
///
/// `None` keeps every dimension. Truncation happens before normalization, so a
/// truncated embedding is again a unit vector.
///
/// # Errors
///
/// Returns [`Qwen3EmbeddingError`] for an unsupported dimension, a nonfinite
/// value, or a zero vector.
pub fn normalize_embedding(
    hidden: &[f32],
    dimensions: Option<usize>,
) -> Result<Vec<f32>, Qwen3EmbeddingError> {
    let width = dimensions.unwrap_or(hidden.len());
    if !(QWEN3_EMBEDDING_MIN_DIMENSIONS..=hidden.len()).contains(&width) {
        return Err(Qwen3EmbeddingError::Dimensions {
            requested: width,
            minimum: QWEN3_EMBEDDING_MIN_DIMENSIONS,
            maximum: hidden.len(),
        });
    }
    let kept = &hidden[..width];
    if let Some(index) = kept.iter().position(|value| !value.is_finite()) {
        return Err(Qwen3EmbeddingError::NonFinite { index });
    }
    let norm = kept
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err(Qwen3EmbeddingError::ZeroNorm);
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "a unit-vector component fits f32"
    )]
    Ok(kept
        .iter()
        .map(|&value| (f64::from(value) / norm) as f32)
        .collect())
}

/// A Qwen3 embedding request could not be computed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3EmbeddingError {
    /// The IDs do not end with the tokenizer-appended `<|endoftext|>`.
    #[error("embedding input must end with <|endoftext|> (151643), found {last:?}")]
    MissingEndOfText {
        /// Last supplied ID, if any.
        last: Option<i32>,
    },
    /// The requested output dimension is outside the supported range.
    #[error("embedding dimensions {requested} outside {minimum}..={maximum}")]
    Dimensions {
        /// Requested dimension.
        requested: usize,
        /// Smallest supported dimension.
        minimum: usize,
        /// Hidden size.
        maximum: usize,
    },
    /// The pooled hidden state is nonfinite.
    #[error("pooled hidden state is nonfinite at {index}")]
    NonFinite {
        /// First nonfinite element.
        index: usize,
    },
    /// The pooled hidden state is all zeros.
    #[error("pooled hidden state has zero norm")]
    ZeroNorm,
    /// The decoder forward failed.
    #[cfg(feature = "metal")]
    #[error(transparent)]
    Forward(#[from] crate::forward::Qwen3ForwardError),
}

#[cfg(test)]
mod tests {
    use super::{
        QWEN3_EMBEDDING_EOS_ID, QWEN3_EMBEDDING_WEB_SEARCH_TASK, Qwen3EmbeddingError,
        check_embedding_input, normalize_embedding, qwen3_embedding_query,
    };

    #[test]
    fn query_matches_the_model_card_format() {
        assert_eq!(
            qwen3_embedding_query(
                QWEN3_EMBEDDING_WEB_SEARCH_TASK,
                "What is the capital of China?"
            ),
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?"
        );
    }

    #[test]
    fn input_must_end_with_the_appended_end_of_text() {
        assert!(check_embedding_input(&[9, QWEN3_EMBEDDING_EOS_ID]).is_ok());
        assert!(matches!(
            check_embedding_input(&[QWEN3_EMBEDDING_EOS_ID, 9]),
            Err(Qwen3EmbeddingError::MissingEndOfText { last: Some(9) })
        ));
        assert!(matches!(
            check_embedding_input(&[]),
            Err(Qwen3EmbeddingError::MissingEndOfText { last: None })
        ));
    }

    #[test]
    fn normalizes_full_and_truncated_vectors_to_unit_length() {
        // 3-4-0 in the first 32 components, then a large tail.
        let mut hidden = vec![0.0; 64];
        hidden[0] = 3.0;
        hidden[1] = 4.0;
        hidden[40] = 12.0;
        let full = normalize_embedding(&hidden, None).expect("full");
        assert_eq!(full.len(), 64);
        assert_eq!(
            (full[0], full[1], full[40]),
            (3.0 / 13.0, 4.0 / 13.0, 12.0 / 13.0)
        );
        // Truncation drops the tail before normalizing, so 3-4 becomes 0.6-0.8.
        let truncated = normalize_embedding(&hidden, Some(32)).expect("truncated");
        assert_eq!(truncated.len(), 32);
        assert_eq!((truncated[0], truncated[1]), (0.6, 0.8));
    }

    #[test]
    fn rejects_unsupported_dimensions_nonfinite_and_zero_vectors() {
        let hidden = vec![1.0; 64];
        for dimensions in [31, 65] {
            assert!(matches!(
                normalize_embedding(&hidden, Some(dimensions)),
                Err(Qwen3EmbeddingError::Dimensions { .. })
            ));
        }
        let mut bad = hidden.clone();
        bad[5] = f32::NAN;
        assert!(matches!(
            normalize_embedding(&bad, None),
            Err(Qwen3EmbeddingError::NonFinite { index: 5 })
        ));
        assert!(matches!(
            normalize_embedding(&[0.0; 32], None),
            Err(Qwen3EmbeddingError::ZeroNorm)
        ));
    }
}
