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

/// Separator between chunks of one pplx-embed-context document: `<|endoftext|>`.
pub const PPLX_CONTEXT_SEPARATOR: &str = "<|endoftext|>";

/// Token ID of [`PPLX_CONTEXT_SEPARATOR`] in the pplx-embed tokenizer.
pub const PPLX_CONTEXT_SEPARATOR_ID: i32 = 151_643;

/// Joins a document's chunks into the one sequence pplx-embed-context encodes.
///
/// No prefix or instruction is added. Encode the result with the model's own
/// `tokenizer.json`; its template appends nothing.
///
/// # Errors
///
/// Returns [`Qwen3EmbeddingError::SeparatorInChunk`] when a chunk contains the
/// separator text, which would split it into two chunks.
pub fn join_context_chunks(chunks: &[&str]) -> Result<String, Qwen3EmbeddingError> {
    if let Some(index) = chunks
        .iter()
        .position(|chunk| chunk.contains(PPLX_CONTEXT_SEPARATOR))
    {
        return Err(Qwen3EmbeddingError::SeparatorInChunk { index });
    }
    Ok(chunks.join(PPLX_CONTEXT_SEPARATOR))
}

/// Mean-pools each chunk span of final-norm hidden states `[positions, width]`.
///
/// Spans lie between separator tokens, which are excluded; the last span runs
/// from the last separator to the end. An empty span pools to zeros, as in the
/// source's `mean_pooling` with its clamped denominator.
///
/// # Errors
///
/// Returns [`Qwen3EmbeddingError::HiddenShape`] when `hidden` is not
/// `input_ids.len() * width` values.
pub fn mean_pool_context_chunks(
    hidden: &[f32],
    width: usize,
    input_ids: &[i32],
) -> Result<Vec<Vec<f32>>, Qwen3EmbeddingError> {
    if width == 0 || Some(hidden.len()) != input_ids.len().checked_mul(width) {
        return Err(Qwen3EmbeddingError::HiddenShape {
            values: hidden.len(),
            positions: input_ids.len(),
            width,
        });
    }
    let rows: Vec<&[f32]> = hidden.chunks_exact(width).collect();
    let mut chunks = Vec::new();
    let mut start = 0;
    let separators = input_ids
        .iter()
        .enumerate()
        .filter(|&(_, &id)| id == PPLX_CONTEXT_SEPARATOR_ID)
        .map(|(position, _)| position);
    for end in separators.chain([input_ids.len()]) {
        let span = &rows[start..end];
        let mut sum = vec![0.0_f64; width];
        for row in span {
            for (total, &value) in sum.iter_mut().zip(*row) {
                *total += f64::from(value);
            }
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            reason = "a mean of f32 values over a bounded span fits f32"
        )]
        chunks.push(
            sum.iter()
                .map(|&total| (total / span.len().max(1) as f64) as f32)
                .collect(),
        );
        start = end + 1;
    }
    Ok(chunks)
}

/// pplx-embed's int8 quantization: `clamp(round(tanh(x) * 127), -128, 127)`,
/// rounding half to even as `torch.round` does.
#[must_use]
pub fn quantize_int8_tanh(values: &[f32]) -> Vec<i8> {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the value is rounded and clamped to the i8 range first"
    )]
    values
        .iter()
        .map(|&value| {
            ((value.tanh() * 127.0)
                .round_ties_even()
                .clamp(-128.0, 127.0)) as i8
        })
        .collect()
}

/// pplx-embed's binary quantization: `1` where `x >= 0`, else `-1`.
#[must_use]
pub fn quantize_binary(values: &[f32]) -> Vec<i8> {
    values
        .iter()
        .map(|&value| if value >= 0.0 { 1 } else { -1 })
        .collect()
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
    /// A context chunk contains the separator text.
    #[error("context chunk {index} contains the <|endoftext|> separator")]
    SeparatorInChunk {
        /// Offending chunk.
        index: usize,
    },
    /// Hidden states do not have one `width` row per input ID.
    #[error("{values} hidden values do not form {positions} rows of width {width}")]
    HiddenShape {
        /// Supplied values.
        values: usize,
        /// Input positions.
        positions: usize,
        /// Row width.
        width: usize,
    },
    /// The checkpoint's attention layout does not suit this pooling.
    #[error("this embedding needs {required:?} attention")]
    AttentionMismatch {
        /// Layout the pooling was trained with.
        required: crate::Qwen3Attention,
    },
    /// The decoder forward failed.
    #[cfg(feature = "metal")]
    #[error(transparent)]
    Forward(#[from] crate::forward::Qwen3ForwardError),
}

#[cfg(test)]
mod tests {
    use super::{
        PPLX_CONTEXT_SEPARATOR_ID, QWEN3_EMBEDDING_EOS_ID, QWEN3_EMBEDDING_WEB_SEARCH_TASK,
        Qwen3EmbeddingError, check_embedding_input, join_context_chunks, mean_pool_context_chunks,
        normalize_embedding, quantize_binary, quantize_int8_tanh, qwen3_embedding_query,
    };

    #[test]
    fn joins_chunks_with_the_separator_and_refuses_one_inside_a_chunk() {
        assert_eq!(
            join_context_chunks(&["a b", "c", ""]).unwrap(),
            "a b<|endoftext|>c<|endoftext|>"
        );
        assert!(matches!(
            join_context_chunks(&["ok", "x<|endoftext|>y"]),
            Err(Qwen3EmbeddingError::SeparatorInChunk { index: 1 })
        ));
    }

    #[test]
    fn mean_pools_spans_between_separators_including_empty_ones() {
        let sep = PPLX_CONTEXT_SEPARATOR_ID;
        // Positions: [1, 3] sep [] sep [5] sep(last) -> trailing empty span.
        let ids = [7, 8, sep, sep, 9, sep];
        let hidden = [
            1.0, 10.0, 3.0, 30.0, 0.0, 0.0, 0.0, 0.0, 5.0, 50.0, 0.0, 0.0,
        ];
        let chunks = mean_pool_context_chunks(&hidden, 2, &ids).unwrap();
        assert_eq!(
            chunks,
            vec![
                vec![2.0, 20.0],
                vec![0.0, 0.0],
                vec![5.0, 50.0],
                vec![0.0, 0.0]
            ]
        );
        // No separator: one span over the whole sequence.
        assert_eq!(
            mean_pool_context_chunks(&[2.0, 4.0], 1, &[7, 8]).unwrap(),
            vec![vec![3.0]]
        );
        assert!(matches!(
            mean_pool_context_chunks(&[1.0; 3], 2, &[7, 8]),
            Err(Qwen3EmbeddingError::HiddenShape { .. })
        ));
    }

    #[test]
    fn quantizes_like_the_source_at_zero_saturation_and_sign() {
        // tanh(-0.3) * 127 = -36.997 -> -37; large magnitudes saturate at
        // +-127 because tanh never reaches 1, so -128 is never produced.
        assert_eq!(
            quantize_int8_tanh(&[0.0, 50.0, -50.0, -0.3]),
            [0, 127, -127, -37]
        );
        // `x >= 0` maps both zeros to 1, matching torch.where(x >= 0, 1, -1).
        assert_eq!(quantize_binary(&[0.0, -0.0, 1e-9, -1e-9]), [1, 1, 1, -1]);
    }

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
