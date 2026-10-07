//! Prefill from prompts whose placeholder tokens are replaced by encoder
//! output rows ([`media::EmbeddedPrompt`]).
//!
//! The ids stay the real placeholder ids, so the prompt still validates
//! against the vocabulary and the context limit; only the input embeddings
//! at span positions change. This decoder has plain 1-D `RoPE`, so it takes
//! audio spans, whose positions advance one per token (Qwen3-ASR gives every
//! token one position on all three `MRoPE` axes). Image spans need a family's
//! multi-axis positions and are refused.

use std::hash::BuildHasher;

use media::{EmbeddedPrompt, PromptChunk, SpanKind};
use mlx_rs::{Array, StreamOrDevice, ops, ops::indexing::IndexOp};

use super::{
    LogitRows, Qwen3ForwardError, Qwen3ForwardExecutor, as_i32, embed_rows, read_last_logits,
    validate_input_ids,
};

impl<S: BuildHasher> Qwen3ForwardExecutor<'_, S> {
    /// [`Self::prefill_last_logits`] with the prompt's span rows replacing
    /// the token embeddings at their positions.
    ///
    /// With no spans this is the same graph as the ids path, so the logits
    /// are bit-equal.
    ///
    /// # Errors
    ///
    /// * [`EmbeddedSpan`](crate::forward::Qwen3ForwardError::EmbeddedSpan) when
    ///   an embedded span does not fit the prompt.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn prefill_embedded_last_logits(
        &mut self,
        prompt: &EmbeddedPrompt<'_, Array>,
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        self.reset();
        self.extend_embedded_last_logits(&prompt.chunk(0..prompt.ids().len()))
    }

    /// Appends one chunk of an embedded prompt and returns its last logits.
    ///
    /// The chunk must start where the cache ends (`chunk.offset()` equals
    /// [`Self::cached_tokens`]), so a chunked prefill cannot skip or repeat
    /// positions. A span crossing the chunk's edges contributes only its
    /// rows inside the chunk.
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`EmbeddedSpan`](crate::forward::Qwen3ForwardError::EmbeddedSpan) when
    ///   an embedded span does not fit the chunk.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn extend_embedded_last_logits(
        &mut self,
        chunk: &PromptChunk<'_, Array>,
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        let result = self.append_embedded(chunk);
        if result.is_err() {
            // As in `append`: never keep a partly appended cache.
            self.reset();
        }
        result
    }

    fn append_embedded(
        &mut self,
        chunk: &PromptChunk<'_, Array>,
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        if chunk.offset() != self.cached_tokens {
            return Err(Qwen3ForwardError::EmbeddedSpan {
                position: chunk.offset(),
                reason: "the chunk must start where the cache ends",
            });
        }
        validate_input_ids(
            self.config,
            chunk.ids(),
            self.cached_tokens,
            self.maximum_context_tokens,
        )?;
        let stream = StreamOrDevice::gpu();
        let seq_len = as_i32(chunk.ids().len())?;
        let hidden = as_i32(self.config.hidden_size)?;
        let tokens = embed_rows(
            self.config,
            self.weights,
            &Array::from_slice(chunk.ids(), &[seq_len]),
        )?;
        let inputs = if chunk.spans().is_empty() {
            tokens
        } else {
            let mut pieces = Vec::with_capacity(2 * chunk.spans().len() + 1);
            let mut cursor = 0;
            for part in chunk.spans() {
                let span = part.span;
                let position = chunk.offset() + part.start;
                if !matches!(span.kind(), SpanKind::Audio) {
                    return Err(Qwen3ForwardError::EmbeddedSpan {
                        position,
                        reason: "image spans need a decoder with multimodal rope",
                    });
                }
                if span.rows().shape() != [as_i32(span.len())?, hidden] {
                    return Err(Qwen3ForwardError::EmbeddedSpan {
                        position,
                        reason: "span rows must be [len, hidden_size]",
                    });
                }
                if part.start > cursor {
                    pieces.push(
                        tokens.index_device((as_i32(cursor)?..as_i32(part.start)?, ..), &stream),
                    );
                }
                let first = as_i32(part.row_offset)?;
                let rows = span
                    .rows()
                    .index_device((first..first + as_i32(part.len)?, ..), &stream);
                // Encoder rows take the decoder's dtype, as the reference
                // casts features to the input embeddings' dtype.
                pieces.push(rows.as_dtype_device(tokens.dtype(), &stream)?);
                cursor = part.start + part.len;
            }
            if cursor < chunk.ids().len() {
                pieces.push(tokens.index_device((as_i32(cursor)?..seq_len, ..), &stream));
            }
            let pieces: Vec<&Array> = pieces.iter().collect();
            ops::concatenate_axis_device(&pieces, 0, &stream)?
        };
        let hidden_states = inputs.reshape_device(&[1, seq_len, hidden], &stream)?;
        let logits = self.append_hidden(hidden_states, seq_len, LogitRows::Last)?;
        read_last_logits(&logits, 1, self.config.vocab_size)
    }
}
