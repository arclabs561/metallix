//! klein's prompt encoder: a Qwen3 decoder read at three depths.
//!
//! `Flux2KleinPipeline._get_qwen3_prompt_embeds` right-pads the chat-rendered
//! prompt to 512 tokens, runs the Qwen3 causal LM with an attention mask, and
//! concatenates the residual streams after layers 9, 18 and 27 per position:
//! `[1, 512, 3 * hidden]`. Every padded row is kept, because the transformer
//! reads all 512 text tokens without a mask.

use std::path::Path;

use mlx_rs::{Array, ops};
use qwen::forward::{Qwen3FloatPrecision, Qwen3ForwardError, Qwen3WeightPrecision};
use qwen::metal::{Qwen3MetalLoadError, Qwen3MlxWeights};
use thiserror::Error;

use crate::transformer::Flux2Precision;

/// Hidden-state indices (1-based layer counts) the pipeline concatenates.
pub const TEXT_LAYERS: [usize; 3] = [9, 18, 27];
/// The pipeline's `max_sequence_length`.
pub const MAX_TEXT_TOKENS: usize = 512;
/// `<|endoftext|>`, the klein tokenizer's `pad_token`.
pub const PAD_TOKEN_ID: i32 = 151_643;

/// A Qwen text checkpoint or prompt embedding operation failed.
#[derive(Debug, Error)]
pub enum TextEncoderError {
    /// Loading or preparing the Qwen checkpoint failed.
    #[error(transparent)]
    Load(#[from] Qwen3MetalLoadError),
    /// Qwen hidden-state execution failed.
    #[error(transparent)]
    Forward(#[from] Qwen3ForwardError),
    /// No prompt token was supplied.
    #[error("a prompt needs at least one token")]
    Empty,
    /// Concatenating hidden-state arrays failed.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
}

/// The rendered prompt the pipeline tokenizes: Qwen3's chat template with one
/// user turn, `add_generation_prompt=True` and `enable_thinking=False`.
#[must_use]
pub fn klein_prompt_text(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// A loaded Qwen3 checkpoint used for klein prompt hidden states.
pub struct KleinTextEncoder {
    weights: Qwen3MlxWeights,
}

impl KleinTextEncoder {
    /// Loads the checkpoint's `text_encoder/` directory.
    ///
    /// # Errors
    ///
    /// Returns [`TextEncoderError::Load`] if the checkpoint cannot load or be
    /// converted to the requested precision.
    pub fn load(
        dir: impl AsRef<Path>,
        precision: Flux2Precision,
    ) -> Result<Self, TextEncoderError> {
        let mut weights = Qwen3MlxWeights::load(dir)?;
        weights.prepare_precision(match precision {
            Flux2Precision::F32 => Qwen3WeightPrecision::Dense(Qwen3FloatPrecision::Float32),
            Flux2Precision::Bf16 => Qwen3WeightPrecision::Dense(Qwen3FloatPrecision::BFloat16),
        })?;
        Ok(Self { weights })
    }

    /// Pads (or truncates) prompt token ids to [`MAX_TEXT_TOKENS`] as the
    /// tokenizer call does, returning the ids and the real length.
    ///
    /// # Errors
    ///
    /// Returns [`TextEncoderError::Empty`] for an empty token slice. Token IDs
    /// are preserved here; vocabulary validation belongs to forward execution.
    pub fn pad(ids: &[i32]) -> Result<(Vec<i32>, usize), TextEncoderError> {
        if ids.is_empty() {
            return Err(TextEncoderError::Empty);
        }
        let real_len = ids.len().min(MAX_TEXT_TOKENS);
        let mut padded = ids[..real_len].to_vec();
        padded.resize(MAX_TEXT_TOKENS, PAD_TOKEN_ID);
        Ok((padded, real_len))
    }

    /// The selected layers' residual streams for a padded prompt, in
    /// [`TEXT_LAYERS`] order, each `[1, 512, hidden]`.
    ///
    /// # Errors
    ///
    /// Returns [`TextEncoderError::Empty`] for no input or
    /// [`TextEncoderError::Forward`] for invalid token IDs, unavailable layer
    /// readouts, or execution failures.
    pub fn layer_states(&self, ids: &[i32]) -> Result<Vec<Array>, TextEncoderError> {
        let (padded, real_len) = Self::pad(ids)?;
        Ok(self
            .weights
            .forward_layer_states(&padded, real_len, &TEXT_LAYERS)?)
    }

    /// The prompt embedding `[1, 512, 3 * hidden]`.
    ///
    /// # Errors
    ///
    /// Propagates [`Self::layer_states`] errors, or [`TextEncoderError::Mlx`]
    /// when the selected hidden states cannot be concatenated.
    pub fn encode(&self, ids: &[i32]) -> Result<Array, TextEncoderError> {
        Ok(ops::concatenate(&self.layer_states(ids)?, -1)?)
    }
}

#[cfg(test)]
mod tests;
