//! Qwen3 forward paths for an uncached numerical oracle and cached resident chat.
//!
//! The uncached dense oracle remains bounded to 512 tokens so its full
//! causal-attention graph is safe for parity checks. The adapter-local cached
//! decoder has separate resident-context admission and retains Qwen3's Q/K
//! normalization and GQA tensors; any kernel fusion belongs behind an
//! equivalent numerical test.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::{collections::HashMap, hash::BuildHasher};

#[cfg(test)]
use std::{
    cell::RefCell,
    time::{Duration, Instant},
};

use mlx_rs::{
    Array, Dtype, StreamOrDevice, fast, ops,
    ops::indexing::{IndexMutOp, IndexOp},
};
use serde::Deserialize;
use thiserror::Error;

use crate::{DecoderFamily, Qwen3Attention};

mod embedded;
mod paged;
mod picks;
mod projection;
mod score;
mod snapshot;
mod verify;

pub(crate) use crate::quantization::is_quantizable;
pub use crate::quantization::{Qwen3AffineBits, Qwen3AffineGroupSize, Qwen3AffineQuantization};
pub use paged::{
    BatchDecoded, BatchReadback, PagedChunk, PagedPrefill, PagedQwen3Session, QueuedDecode,
    StepInput,
};
pub use picks::{Qwen3PickRule, Qwen3RowCandidates, Qwen3Selection, Qwen3TokenPicks};
pub use projection::Qwen3WeightPrecision;
pub(crate) use projection::{dense_weight, embed_rows, project};
pub use score::{Qwen3ScoredToken, Qwen3Scores, SCORE_CHUNK_ROWS};
pub use snapshot::Qwen3KvSnapshot;
pub use verify::Qwen3PositionLogits;

/// The largest prompt accepted by the uncached qualification forward path.
pub const MAX_DENSE_DEBUG_TOKENS: usize = 512;

/// Longest prompt piece built as one graph. A longer prefill or extension
/// runs as consecutive pieces, each evaluated before the next is built, so a
/// long prompt never becomes one command buffer or holds whole-prompt
/// activations (on Qwen3-0.6B the MLP intermediates alone are 1.6 GB at
/// 262K tokens). Each piece attends to the cached prefix, so the result is
/// one call's, up to kernel reduction order.
pub const PREFILL_CHUNK_TOKENS: usize = 2_048;

/// Default logical K/V budget for the resident-chat control path.
pub const DEFAULT_RESIDENT_CHAT_KV_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// A floating-point dtype for resident weights, and so for the K/V they
/// produce: the cached keys and values take the projection output's dtype.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Qwen3FloatPrecision {
    /// Serving precision: Qwen3 checkpoints ship BF16, and decode is bound
    /// by the weight bytes read per token.
    #[default]
    BFloat16,
    /// Half precision, for checkpoints stored that way.
    Float16,
    /// Reference precision for comparison with a CPU float32 oracle; twice
    /// the resident bytes of BF16.
    Float32,
}

impl Qwen3FloatPrecision {
    /// Bytes of one stored element.
    #[must_use]
    pub const fn bytes_per_element(self) -> u64 {
        match self {
            Self::BFloat16 | Self::Float16 => 2,
            Self::Float32 => 4,
        }
    }

    pub(crate) const fn dtype(self) -> Dtype {
        match self {
            Self::BFloat16 => Dtype::Bfloat16,
            Self::Float16 => Dtype::Float16,
            Self::Float32 => Dtype::Float32,
        }
    }

    const fn from_dtype(dtype: Dtype) -> Option<Self> {
        match dtype {
            Dtype::Bfloat16 => Some(Self::BFloat16),
            Dtype::Float16 => Some(Self::Float16),
            Dtype::Float32 => Some(Self::Float32),
            _ => None,
        }
    }
}

/// The dtype of the K/V a cached executor over `weights` retains: that of a
/// key projection of an embedded token, so MLX's promotion of the embedding
/// and key-projection dtypes; for packed weights, of their scales.
pub(crate) fn kv_precision<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
) -> Result<Qwen3FloatPrecision, Qwen3ForwardError> {
    let unpacked = if config.quantization().is_some() {
        "scales"
    } else {
        "weight"
    };
    let embedding = weight(weights, &format!("model.embed_tokens.{unpacked}"))?.dtype();
    let key = weight(
        weights,
        &format!("model.layers.0.self_attn.k_proj.{unpacked}"),
    )?
    .dtype();
    let promoted = if embedding == key {
        key
    } else {
        Dtype::Float32
    };
    for dtype in [embedding, key] {
        if Qwen3FloatPrecision::from_dtype(dtype).is_none() {
            return Err(Qwen3ForwardError::UnsupportedWeightDtype(format!(
                "{dtype:?}"
            )));
        }
    }
    Qwen3FloatPrecision::from_dtype(promoted)
        .ok_or_else(|| Qwen3ForwardError::UnsupportedWeightDtype(format!("{promoted:?}")))
}

/// A checked resident-chat context and its final logical K/V estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3ResidentChatPlan {
    maximum_context_tokens: usize,
    planned_kv_bytes: u64,
    kv_precision: Qwen3FloatPrecision,
}

/// An absolute, half-open token-position range selected for a residual intervention.
///
/// Positions are counted from the beginning of the prompt, so the same token is
/// selected whether it is processed during prefill or a later one-token decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3SteeringPositionRange {
    start_inclusive: usize,
    end_exclusive: usize,
}

impl Qwen3SteeringPositionRange {
    /// Creates a nonempty absolute token-position range.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`EmptyPositionRange`](crate::forward::Qwen3SteeringError::EmptyPositionRange)
    /// unless `start_inclusive < end_exclusive`.
    pub fn new(start_inclusive: usize, end_exclusive: usize) -> Result<Self, Qwen3SteeringError> {
        if start_inclusive >= end_exclusive {
            return Err(Qwen3SteeringError::EmptyPositionRange {
                start_inclusive,
                end_exclusive,
            });
        }
        Ok(Self {
            start_inclusive,
            end_exclusive,
        })
    }

    /// First selected absolute token position.
    #[must_use]
    pub const fn start_inclusive(self) -> usize {
        self.start_inclusive
    }

    /// First unselected absolute token position.
    #[must_use]
    pub const fn end_exclusive(self) -> usize {
        self.end_exclusive
    }

    const fn contains(self, position: usize) -> bool {
        position >= self.start_inclusive && position < self.end_exclusive
    }
}

/// Caller-supplied Qwen residual artifact before it has been bound to one
/// loaded Qwen configuration.
///
/// `model_identity` is retained as receipt provenance, but it is caller
/// asserted: this adapter cannot independently authenticate that identity.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ResidualSteeringArtifact {
    model_identity: String,
    declared_hidden_layers: usize,
    declared_hidden_size: usize,
    layer: usize,
    residual: Vec<f32>,
    coefficient: f32,
    positions: Qwen3SteeringPositionRange,
}

impl Qwen3ResidualSteeringArtifact {
    /// Creates an artifact whose model declaration is checked when bound.
    ///
    /// # Errors
    ///
    /// * [`MissingModelIdentity`](crate::forward::Qwen3SteeringError::MissingModelIdentity)
    ///   for an empty identity.
    /// * [`InvalidDeclaredDimensions`](crate::forward::Qwen3SteeringError::InvalidDeclaredDimensions),
    ///   [`LayerOutOfDeclaredRange`](crate::forward::Qwen3SteeringError::LayerOutOfDeclaredRange)
    ///   and
    ///   [`ResidualDimensionMismatch`](crate::forward::Qwen3SteeringError::ResidualDimensionMismatch)
    ///   when the dimensions, layer or residual length disagree.
    /// * [`NonFiniteResidual`](crate::forward::Qwen3SteeringError::NonFiniteResidual)
    ///   and
    ///   [`NonFiniteCoefficient`](crate::forward::Qwen3SteeringError::NonFiniteCoefficient)
    ///   for a NaN or infinite value.
    pub fn new(
        model_identity: impl Into<String>,
        declared_hidden_layers: usize,
        declared_hidden_size: usize,
        layer: usize,
        residual: Vec<f32>,
        coefficient: f32,
        positions: Qwen3SteeringPositionRange,
    ) -> Result<Self, Qwen3SteeringError> {
        let model_identity = model_identity.into();
        if model_identity.trim().is_empty() {
            return Err(Qwen3SteeringError::MissingModelIdentity);
        }
        if declared_hidden_layers == 0 || declared_hidden_size == 0 {
            return Err(Qwen3SteeringError::InvalidDeclaredDimensions {
                hidden_layers: declared_hidden_layers,
                hidden_size: declared_hidden_size,
            });
        }
        if !coefficient.is_finite() {
            return Err(Qwen3SteeringError::NonFiniteCoefficient(coefficient));
        }
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(Qwen3SteeringError::NonFiniteResidual);
        }
        if residual.len() != declared_hidden_size {
            return Err(Qwen3SteeringError::ResidualDimensionMismatch {
                actual: residual.len(),
                expected: declared_hidden_size,
            });
        }
        if layer >= declared_hidden_layers {
            return Err(Qwen3SteeringError::LayerOutOfDeclaredRange {
                layer,
                hidden_layers: declared_hidden_layers,
            });
        }
        Ok(Self {
            model_identity,
            declared_hidden_layers,
            declared_hidden_size,
            layer,
            residual,
            coefficient,
            positions,
        })
    }

    /// The caller-asserted model identity retained for request receipts.
    #[must_use]
    pub fn model_identity(&self) -> &str {
        &self.model_identity
    }
}

/// A Qwen-local residual intervention validated against a loaded configuration.
///
/// It is an execution mechanism only. No artifact or receipt from this type
/// establishes a behavioral benefit; that requires a separate held-out study.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ResidualSteering {
    artifact: Qwen3ResidualSteeringArtifact,
    bound_config: Qwen3ForwardConfig,
}

impl Qwen3ResidualSteering {
    /// Binds an artifact to this exact Qwen execution layout.
    ///
    /// # Errors
    ///
    /// * [`ConfigurationDimensionMismatch`](crate::forward::Qwen3SteeringError::ConfigurationDimensionMismatch)
    ///   and
    ///   [`LayerOutOfDeclaredRange`](crate::forward::Qwen3SteeringError::LayerOutOfDeclaredRange)
    ///   when the artifact was not made for `config`.
    /// * [`NonFiniteScaledResidual`](crate::forward::Qwen3SteeringError::NonFiniteScaledResidual)
    ///   when scaling the residual overflows.
    pub fn bind(
        artifact: Qwen3ResidualSteeringArtifact,
        config: &Qwen3ForwardConfig,
    ) -> Result<Self, Qwen3SteeringError> {
        if artifact.declared_hidden_layers != config.hidden_layers
            || artifact.declared_hidden_size != config.hidden_size
        {
            return Err(Qwen3SteeringError::ConfigurationDimensionMismatch {
                declared_hidden_layers: artifact.declared_hidden_layers,
                declared_hidden_size: artifact.declared_hidden_size,
                actual_hidden_layers: config.hidden_layers,
                actual_hidden_size: config.hidden_size,
            });
        }
        if artifact.layer >= config.hidden_layers {
            return Err(Qwen3SteeringError::LayerOutOfDeclaredRange {
                layer: artifact.layer,
                hidden_layers: config.hidden_layers,
            });
        }
        if artifact
            .residual
            .iter()
            .any(|value| !(value * artifact.coefficient).is_finite())
        {
            return Err(Qwen3SteeringError::NonFiniteScaledResidual);
        }
        Ok(Self {
            artifact,
            bound_config: config.clone(),
        })
    }

    /// Artifact provenance retained by this bound intervention.
    #[must_use]
    pub const fn artifact(&self) -> &Qwen3ResidualSteeringArtifact {
        &self.artifact
    }

    fn is_bound_to(&self, config: &Qwen3ForwardConfig) -> bool {
        self.bound_config == *config
    }
}

impl Qwen3ResidentChatPlan {
    /// Maximum prompt-plus-generated tokens admitted by this executor.
    #[must_use]
    pub const fn maximum_context_tokens(self) -> usize {
        self.maximum_context_tokens
    }

    /// Logical K/V bytes estimated for that maximum context at
    /// [`Self::kv_precision`].
    #[must_use]
    pub const fn planned_kv_bytes(self) -> u64 {
        self.planned_kv_bytes
    }

    /// The K/V element dtype the estimate assumed.
    #[must_use]
    pub const fn kv_precision(self) -> Qwen3FloatPrecision {
        self.kv_precision
    }
}

/// Dense Qwen3 configuration needed to build a reference forward graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3ForwardConfig {
    hidden_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    attention_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    attention: Qwen3Attention,
    family: DecoderFamily,
    tied_output_embedding: bool,
    /// How projection and embedding tensors are packed, if quantized.
    quantization: Option<Qwen3AffineQuantization>,
}

/// Checks nonzero dimensions and the grouped-query layout before tensor shapes
/// are derived from a checkpoint configuration.
fn validate_dimensions(raw: &RawForwardConfig) -> Result<(), Qwen3ForwardError> {
    let fields = [
        ("num_hidden_layers", raw.num_hidden_layers),
        ("hidden_size", raw.hidden_size),
        ("intermediate_size", raw.intermediate_size),
        ("vocab_size", raw.vocab_size),
        ("num_attention_heads", raw.num_attention_heads),
        ("num_key_value_heads", raw.num_key_value_heads),
        ("head_dim", raw.head_dim),
        ("max_position_embeddings", raw.max_position_embeddings),
    ];
    for (name, value) in fields {
        if value == 0 {
            return Err(Qwen3ForwardError::MissingDimension(name));
        }
    }
    if !raw.head_dim.is_multiple_of(2) {
        return Err(Qwen3ForwardError::OddHeadDimension(raw.head_dim));
    }
    if raw.head_dim > usize::from(u16::MAX) {
        return Err(Qwen3ForwardError::HeadDimensionTooLarge(raw.head_dim));
    }
    if !raw
        .num_attention_heads
        .is_multiple_of(raw.num_key_value_heads)
    {
        return Err(Qwen3ForwardError::InvalidGroupedQueryLayout {
            attention_heads: raw.num_attention_heads,
            key_value_heads: raw.num_key_value_heads,
        });
    }
    Ok(())
}

impl Qwen3ForwardConfig {
    /// Parses only the dense Qwen3, Llama or Qwen2 layout this qualification
    /// path implements; of the biases, only Qwen2's fixed Q/K/V projection
    /// bias. Sliding-window and scaled `RoPE` variants require their own
    /// reference vectors, so they are refused rather than silently ignored. An untied checkpoint projects logits
    /// through `lm_head.weight` instead of the token embedding.
    ///
    /// # Errors
    ///
    /// * [`Json`](crate::forward::Qwen3ForwardError::Json) for malformed JSON.
    /// * [`UnsupportedModelType`](crate::forward::Qwen3ForwardError::UnsupportedModelType),
    ///   [`UnsupportedBiasLayout`](crate::forward::Qwen3ForwardError::UnsupportedBiasLayout),
    ///   [`UnsupportedActivation`](crate::forward::Qwen3ForwardError::UnsupportedActivation),
    ///   [`UnsupportedRopeScaling`](crate::forward::Qwen3ForwardError::UnsupportedRopeScaling)
    ///   and
    ///   [`UnsupportedSlidingWindow`](crate::forward::Qwen3ForwardError::UnsupportedSlidingWindow)
    ///   for a variant this decoder does not implement.
    /// * [`MissingDimension`](crate::forward::Qwen3ForwardError::MissingDimension),
    ///   [`OddHeadDimension`](crate::forward::Qwen3ForwardError::OddHeadDimension),
    ///   [`HeadDimensionTooLarge`](crate::forward::Qwen3ForwardError::HeadDimensionTooLarge),
    ///   [`InvalidGroupedQueryLayout`](crate::forward::Qwen3ForwardError::InvalidGroupedQueryLayout),
    ///   [`InvalidRmsNormEpsilon`](crate::forward::Qwen3ForwardError::InvalidRmsNormEpsilon),
    ///   [`InvalidRopeTheta`](crate::forward::Qwen3ForwardError::InvalidRopeTheta),
    ///   [`MissingRopeTheta`](crate::forward::Qwen3ForwardError::MissingRopeTheta)
    ///   and
    ///   [`ConflictingRopeTheta`](crate::forward::Qwen3ForwardError::ConflictingRopeTheta)
    ///   for dimensions or controls the forward path cannot use.
    pub fn parse(json: &str) -> Result<Self, Qwen3ForwardError> {
        let mut raw: RawForwardConfig = serde_json::from_str(json)?;
        let (Some(family), Some(attention)) = (
            DecoderFamily::from_model_type(&raw.model_type),
            Qwen3Attention::from_config(&raw.model_type, raw.use_bidirectional_attention),
        ) else {
            return Err(Qwen3ForwardError::UnsupportedModelType(raw.model_type));
        };
        if raw.attention_bias || raw.mlp_bias {
            return Err(Qwen3ForwardError::UnsupportedBiasLayout);
        }
        if raw.hidden_act != "silu" {
            return Err(Qwen3ForwardError::UnsupportedActivation(raw.hidden_act));
        }
        if raw.rope_scaling.is_some() {
            return Err(Qwen3ForwardError::UnsupportedRopeScaling);
        }
        // transformers 5 writes `rope_parameters` in place of, or beside,
        // `rope_theta` and `rope_scaling` (pplx-embed has both). Only the
        // unscaled default form is read, and its theta must agree with a
        // stated `rope_theta`.
        if let Some(parameters) = raw.rope_parameters.take() {
            if parameters.rope_type != "default" || !parameters.other.is_empty() {
                return Err(Qwen3ForwardError::UnsupportedRopeScaling);
            }
            // Both bases are parsed from JSON numbers, so the same base is
            // bit-identical.
            match (raw.rope_theta, parameters.rope_theta) {
                (Some(rope_theta), Some(nested)) if rope_theta.to_bits() != nested.to_bits() => {
                    return Err(Qwen3ForwardError::ConflictingRopeTheta {
                        rope_theta,
                        rope_parameters: nested,
                    });
                }
                (None, nested) => raw.rope_theta = nested,
                (Some(_), _) => {}
            }
        }
        // Llama's documented default theta has changed across transformers
        // releases, so a Llama checkpoint must state it; so must Qwen2, whose
        // published configs all do.
        let rope_theta = match (family, raw.rope_theta) {
            (_, Some(theta)) => theta,
            (DecoderFamily::Qwen3, None) => default_rope_theta(),
            (DecoderFamily::Llama | DecoderFamily::Qwen2, None) => {
                return Err(Qwen3ForwardError::MissingRopeTheta);
            }
        };
        // As in transformers' Llama and Qwen2 attention, an omitted `head_dim`
        // is `hidden_size / num_attention_heads`.
        if family.derives_head_dim()
            && raw.head_dim == 0
            && raw.num_attention_heads != 0
            && raw.hidden_size.is_multiple_of(raw.num_attention_heads)
        {
            raw.head_dim = raw.hidden_size / raw.num_attention_heads;
        }
        // transformers' Qwen2Config and Qwen3Config keep `sliding_window`
        // only when `use_sliding_window` is set, so Qwen2.5's
        // `"sliding_window": 32768, "use_sliding_window": false` is full
        // attention. Llama has no such switch: any window is refused.
        let sliding_window = match family {
            DecoderFamily::Qwen3 | DecoderFamily::Qwen2 => raw.use_sliding_window,
            DecoderFamily::Llama => raw.use_sliding_window || raw.sliding_window.is_some(),
        };
        if sliding_window {
            return Err(Qwen3ForwardError::UnsupportedSlidingWindow);
        }

        validate_dimensions(&raw)?;
        if !raw.rms_norm_eps.is_finite() || raw.rms_norm_eps <= 0.0 {
            return Err(Qwen3ForwardError::InvalidRmsNormEpsilon(raw.rms_norm_eps));
        }
        if !rope_theta.is_finite() || rope_theta <= 0.0 {
            return Err(Qwen3ForwardError::InvalidRopeTheta(rope_theta));
        }

        Ok(Self {
            hidden_layers: raw.num_hidden_layers,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            vocab_size: raw.vocab_size,
            attention_heads: raw.num_attention_heads,
            key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta,
            attention,
            family,
            tied_output_embedding: raw.tie_word_embeddings,
            quantization: parse_quantization(
                raw.quantization.as_ref(),
                raw.quantization_config.as_ref(),
            )?,
        })
    }

    /// Returns the decoder family named by `model_type`.
    #[must_use]
    pub const fn family(&self) -> DecoderFamily {
        self.family
    }

    /// Whether logits reuse `model.embed_tokens.weight` rather than a
    /// separate `lm_head.weight`.
    #[must_use]
    pub const fn tied_output_embedding(&self) -> bool {
        self.tied_output_embedding
    }

    /// The projection logits are computed with: the token embedding when
    /// tied, else `lm_head`.
    fn output_projection(&self) -> &'static str {
        if self.tied_output_embedding {
            "model.embed_tokens"
        } else {
            "lm_head"
        }
    }

    /// How projection and embedding tensors are packed, if quantized: from
    /// `config.json`'s `quantization`, or as set when weights are quantized
    /// at load.
    #[must_use]
    pub const fn quantization(&self) -> Option<Qwen3AffineQuantization> {
        self.quantization
    }

    pub(crate) const fn set_quantization(&mut self, quantization: Option<Qwen3AffineQuantization>) {
        self.quantization = quantization;
    }

    /// Returns whether decoder layers attend causally or bidirectionally.
    #[must_use]
    pub const fn attention(&self) -> Qwen3Attention {
        self.attention
    }

    /// Returns the residual-stream width used by every decoder layer.
    #[must_use]
    pub(crate) const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Returns the number of decoder layers in the qualified dense layout.
    #[must_use]
    pub(crate) const fn hidden_layers(&self) -> usize {
        self.hidden_layers
    }

    /// Logical f32 bytes for all layer K/V arrays at `tokens` positions.
    ///
    /// The streamed checker stores detached f32 arrays, so this is deliberately
    /// not a checkpoint-byte estimate or a generic cache-layout abstraction.
    pub(crate) fn cached_kv_bytes(&self, tokens: usize) -> Result<u64, Qwen3ForwardError> {
        self.cached_kv_bytes_at(tokens, Qwen3FloatPrecision::Float32)
    }

    /// Logical bytes for all layer K/V arrays at `tokens` positions stored as
    /// `precision`.
    pub(crate) fn cached_kv_bytes_at(
        &self,
        tokens: usize,
        precision: Qwen3FloatPrecision,
    ) -> Result<u64, Qwen3ForwardError> {
        let values = self
            .hidden_layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.key_value_heads))
            .and_then(|value| value.checked_mul(tokens))
            .and_then(|value| value.checked_mul(self.head_dim))
            .ok_or(Qwen3ForwardError::ShapeOverflow)?;
        u64::try_from(values)
            .ok()
            .and_then(|value| value.checked_mul(precision.bytes_per_element()))
            .ok_or(Qwen3ForwardError::ShapeOverflow)
    }

    /// Effective context limit of the bounded dense qualification path.
    #[must_use]
    pub(crate) fn maximum_cached_tokens(&self) -> usize {
        MAX_DENSE_DEBUG_TOKENS.min(self.max_position_embeddings)
    }

    /// Largest context fitting the logical resident K/V budget at this precision.
    ///
    /// `None` means even one token does not fit. This excludes weights,
    /// scratch and allocator headroom; it does not change the requested context.
    ///
    /// # Errors
    ///
    /// Returns the same configuration or checked-size errors as
    /// [`Self::resident_chat_plan`].
    pub fn resident_chat_capacity(
        &self,
        maximum_kv_bytes: u64,
        kv_precision: Qwen3FloatPrecision,
    ) -> Result<Option<std::num::NonZeroUsize>, Qwen3ForwardError> {
        let one = self.resident_chat_plan(1, u64::MAX, kv_precision)?;
        let fitting = usize::try_from(maximum_kv_bytes / one.planned_kv_bytes())
            .unwrap_or(usize::MAX)
            .min(self.max_position_embeddings)
            .min(2_147_483_647);
        if fitting != 0 {
            self.resident_chat_plan(fitting, maximum_kv_bytes, kv_precision)?;
        }
        Ok(std::num::NonZeroUsize::new(fitting))
    }

    /// Validates a resident-chat context before checkpoint payloads are loaded.
    ///
    /// The K/V budget covers only the estimated retained cache arrays at
    /// `kv_precision`, which must be the dtype the executor's weights produce
    /// (executors built from [`crate::metal::Qwen3MlxWeights`] derive it from
    /// their tensors). It excludes model weights, activations, operator
    /// scratch, and allocator headroom; it does not preallocate or reserve MLX
    /// memory.
    ///
    /// # Errors
    ///
    /// * [`CachedBidirectional`](crate::forward::Qwen3ForwardError::CachedBidirectional)
    ///   for a non-causal model.
    /// * [`ResidentChatContextLimit`](crate::forward::Qwen3ForwardError::ResidentChatContextLimit)
    ///   and
    ///   [`ResidentChatKvBudget`](crate::forward::Qwen3ForwardError::ResidentChatKvBudget)
    ///   when the context or its KV cache does not fit the limits.
    pub fn resident_chat_plan(
        &self,
        maximum_context_tokens: usize,
        maximum_kv_bytes: u64,
        kv_precision: Qwen3FloatPrecision,
    ) -> Result<Qwen3ResidentChatPlan, Qwen3ForwardError> {
        if self.attention != Qwen3Attention::Causal {
            return Err(Qwen3ForwardError::CachedBidirectional);
        }
        // The model's declared positions are the only context ceiling; a
        // declared RoPE scaling is refused at parse time, so this never
        // extends past what the configuration states. Memory is the K/V
        // budget below.
        let maximum = self.max_position_embeddings;
        if maximum_context_tokens == 0 || maximum_context_tokens > maximum {
            return Err(Qwen3ForwardError::ResidentChatContextLimit {
                requested: maximum_context_tokens,
                maximum,
            });
        }
        let planned_kv_bytes = self.cached_kv_bytes_at(maximum_context_tokens, kv_precision)?;
        if planned_kv_bytes > maximum_kv_bytes {
            return Err(Qwen3ForwardError::ResidentChatKvBudget {
                required: planned_kv_bytes,
                maximum: maximum_kv_bytes,
            });
        }
        Ok(Qwen3ResidentChatPlan {
            maximum_context_tokens,
            planned_kv_bytes,
            kv_precision,
        })
    }
}

/// Runs a complete uncached Qwen3 forward pass and reads back the last-token
/// logits as f32 values.
///
/// `weights` is the direct safetensors name-to-array map. All operators are
/// constructed on the Metal GPU stream; only the final logit vector crosses
/// back to the host. The caller must keep the map resident for the duration of
/// the invocation.
///
/// # Errors
///
/// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId) and
///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong) when
///   the input is empty, names a token outside the vocabulary, or passes the
///   context limit.
/// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight) and
///   [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is absent or
///   MLX cannot build or evaluate the graph.
pub fn forward_last_logits<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
) -> Result<Vec<f32>, Qwen3ForwardError> {
    forward_last_logits_with_residual_steering(weights, config, input_ids, None)
}

/// Runs the bounded uncached Qwen forward with an optional validated residual
/// intervention.
///
/// # Errors
///
/// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId) and
///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong) when
///   the input is empty, names a token outside the vocabulary, or passes the
///   context limit.
/// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight) and
///   [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is absent or
///   MLX cannot build or evaluate the graph.
/// * [`Steering`](crate::forward::Qwen3ForwardError::Steering) when the
///   intervention does not apply to this input.
pub fn forward_last_logits_with_residual_steering<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
    steering: Option<&Qwen3ResidualSteering>,
) -> Result<Vec<f32>, Qwen3ForwardError> {
    let normalized = last_normalized_hidden(weights, config, input_ids, steering)?;
    let logits = project(config, weights, &normalized, config.output_projection())?;
    read_last_logits(&logits, 1, config.vocab_size)
}

/// Runs a complete uncached Qwen3 forward pass and reads back the last
/// position's final-norm hidden state as `hidden_size` f32 values.
///
/// This is the input of the output projection: the result of
/// [`forward_last_logits`] is this vector times the token-embedding matrix,
/// or `lm_head.weight` for an untied checkpoint.
/// It equals the last row of a source `Qwen3Model`'s `last_hidden_state`.
///
/// # Errors
///
/// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId) and
///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong) when
///   the input is empty, names a token outside the vocabulary, or passes the
///   context limit.
/// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight) and
///   [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is absent or
///   MLX cannot build or evaluate the graph.
pub fn forward_last_hidden<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
) -> Result<Vec<f32>, Qwen3ForwardError> {
    let normalized = last_normalized_hidden(weights, config, input_ids, None)?;
    // The readback helper reads any `[1, 1, width]` row, not only logits.
    read_last_logits(&normalized, 1, config.hidden_size)
}

/// Runs a complete uncached Qwen3 forward pass and reads back every
/// position's final-norm hidden state, flattened `[positions, hidden_size]`.
///
/// This is a source `Qwen3Model`'s `last_hidden_state` for one sequence.
///
/// # Errors
///
/// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId) and
///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong) when
///   the input is empty, names a token outside the vocabulary, or passes the
///   context limit.
/// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight) and
///   [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is absent or
///   MLX cannot build or evaluate the graph.
pub fn forward_hidden_states<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
) -> Result<Vec<f32>, Qwen3ForwardError> {
    let normalized = rms_norm(
        &decoder_states(weights, config, input_ids, None)?,
        weight(weights, "model.norm.weight")?,
        config.rms_norm_eps,
    )?
    .as_type_device::<f32>(StreamOrDevice::gpu())?;
    normalized.eval()?;
    Ok(normalized.as_slice::<f32>().to_vec())
}

/// Pre-norm residual streams after selected decoder layers, for one
/// right-padded causal sequence: `layers` are 1-based counts, so entry `k`
/// is a source `Qwen3Model`'s `hidden_states[k]` (the output of layer index
/// `k - 1`, before the final norm).
///
/// Positions at or after `real_len` are padding. No query attends them, and
/// padding queries attend the real prefix, as transformers' attention mask
/// does; their rows are returned too, because diffusion pipelines such as
/// `FLUX.2 [klein]` feed every padded row onward. Layers past the last
/// requested one are not run.
///
/// # Errors
///
/// Returns an error for noncausal attention, invalid token or layer selections,
/// inconsistent real-token length, missing weights, or backend operations.
pub fn forward_layer_states<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
    real_len: usize,
    layers: &[usize],
) -> Result<Vec<Array>, Qwen3ForwardError> {
    if config.attention != Qwen3Attention::Causal {
        return Err(Qwen3ForwardError::PaddedBidirectional);
    }
    if real_len == 0 || real_len > input_ids.len() {
        return Err(Qwen3ForwardError::RealLength {
            real_len,
            positions: input_ids.len(),
        });
    }
    let last = match layers.last() {
        Some(&last)
            if layers.windows(2).all(|pair| pair[0] < pair[1])
                && layers[0] >= 1
                && last <= config.hidden_layers =>
        {
            last
        }
        _ => {
            return Err(Qwen3ForwardError::LayerSelection {
                layers: layers.to_vec(),
                hidden_layers: config.hidden_layers,
            });
        }
    };
    validate_input_ids(config, input_ids, 0, config.maximum_cached_tokens())?;

    let stream = StreamOrDevice::gpu();
    let seq_len = as_i32(input_ids.len())?;
    let hidden = as_i32(config.hidden_size)?;
    let key_limit = Some(as_i32(real_len)?);
    let mut hidden_states = embed_rows(config, weights, &Array::from_slice(input_ids, &[seq_len]))?
        .reshape_device(&[1, seq_len, hidden], &stream)?;
    let mut selected = Vec::with_capacity(layers.len());
    for layer in 0..last {
        hidden_states =
            forward_layer_with_key_limit(config, weights, layer, &hidden_states, key_limit)?;
        if layers.contains(&(layer + 1)) {
            selected.push(hidden_states.clone());
        }
    }
    Ok(selected)
}

fn last_normalized_hidden<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
    steering: Option<&Qwen3ResidualSteering>,
) -> Result<Array, Qwen3ForwardError> {
    let hidden_states = decoder_states(weights, config, input_ids, steering)?;
    let seq_len = i32::try_from(input_ids.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    // Only the final position is requested; normalization and the output
    // projection are position-independent after the decoder layers.
    let last_hidden = hidden_states.take_axis_device(
        Array::from_slice(&[seq_len - 1], &[1]),
        1,
        StreamOrDevice::gpu(),
    )?;
    rms_norm(
        &last_hidden,
        weight(weights, "model.norm.weight")?,
        config.rms_norm_eps,
    )
}

/// Last-position final-norm hidden states for several sequences in one
/// forward pass, one `hidden_size` row per sequence, in input order.
///
/// Sequences are right-padded to the longest. Under causal attention a
/// position never attends to later positions, so padding after a sequence
/// cannot change its real positions, and each row is read at that sequence's
/// own last token. Bidirectional attention would need a padding mask, so it is
/// refused.
///
/// # Errors
///
/// * [`BatchedBidirectional`](crate::forward::Qwen3ForwardError::BatchedBidirectional)
///   for a non-causal model.
/// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId) and
///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong) when
///   the input is empty, names a token outside the vocabulary, or passes the
///   context limit.
/// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight) and
///   [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is absent or
///   MLX cannot build or evaluate the graph.
/// * [`ShapeOverflow`](crate::forward::Qwen3ForwardError::ShapeOverflow) when
///   the padded batch does not fit.
pub fn forward_last_hidden_batch<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    sequences: &[&[i32]],
) -> Result<Vec<Vec<f32>>, Qwen3ForwardError> {
    if config.attention != Qwen3Attention::Causal {
        return Err(Qwen3ForwardError::BatchedBidirectional);
    }
    let longest = sequences
        .iter()
        .map(|ids| ids.len())
        .max()
        .ok_or(Qwen3ForwardError::EmptyInput)?;
    let mut padded = Vec::with_capacity(sequences.len() * longest);
    let mut last_rows = Vec::with_capacity(sequences.len());
    for (index, ids) in sequences.iter().enumerate() {
        validate_input_ids(config, ids, 0, config.maximum_cached_tokens())?;
        padded.extend_from_slice(ids);
        // Any valid ID works as padding; repeating the last one keeps it in range.
        padded.extend(std::iter::repeat_n(ids[ids.len() - 1], longest - ids.len()));
        last_rows.push(as_i32(index * longest + ids.len() - 1)?);
    }
    let stream = StreamOrDevice::gpu();
    let batch = as_i32(sequences.len())?;
    let seq_len = as_i32(longest)?;
    let hidden = as_i32(config.hidden_size)?;
    let rows = batch
        .checked_mul(seq_len)
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    let mut hidden_states = embed_rows(config, weights, &Array::from_slice(&padded, &[rows]))?
        .reshape_device(&[batch, seq_len, hidden], &stream)?;
    for layer in 0..config.hidden_layers {
        hidden_states = forward_layer(config, weights, layer, &hidden_states)?;
    }
    let last = hidden_states
        .reshape_device(&[rows, hidden], &stream)?
        .take_axis_device(Array::from_slice(&last_rows, &[batch]), 0, &stream)?;
    let normalized = rms_norm(
        &last,
        weight(weights, "model.norm.weight")?,
        config.rms_norm_eps,
    )?
    .as_type_device::<f32>(&stream)?;
    normalized.eval()?;
    Ok(normalized
        .as_slice::<f32>()
        .chunks_exact(config.hidden_size)
        .map(<[f32]>::to_vec)
        .collect())
}

/// The residual stream after every decoder layer, `[1, positions, hidden]`.
fn decoder_states<S: BuildHasher>(
    weights: &HashMap<String, Array, S>,
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
    steering: Option<&Qwen3ResidualSteering>,
) -> Result<Array, Qwen3ForwardError> {
    if steering.is_some_and(|steering| !steering.is_bound_to(config)) {
        return Err(Qwen3SteeringError::BoundConfigurationMismatch.into());
    }
    validate_input_ids(config, input_ids, 0, config.maximum_cached_tokens())?;

    let stream = StreamOrDevice::gpu();
    let seq_len = i32::try_from(input_ids.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let hidden = as_i32(config.hidden_size)?;

    let ids = Array::from_slice(input_ids, &[seq_len]);
    let mut hidden_states =
        embed_rows(config, weights, &ids)?.reshape_device(&[1, seq_len, hidden], &stream)?;

    for layer in 0..config.hidden_layers {
        hidden_states = forward_layer(config, weights, layer, &hidden_states)?;
        hidden_states =
            apply_residual_steering(steering, layer, 0, input_ids.len(), &hidden_states)?;
    }
    Ok(hidden_states)
}

/// Executes one uncached dense Qwen3 decoder layer.
///
/// This is intentionally crate-private: selected-layer diagnostics borrow it
/// while retaining the same bounded, uncached attention contract as
/// [`forward_last_logits`]. Callers provide a residual stream with shape
/// `[batch, sequence, hidden_size]`; a batch shares one sequence length and one
/// attention mode, so only right-padded causal batches are exact.
pub(crate) fn forward_layer<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    layer: usize,
    hidden_states: &Array,
) -> Result<Array, Qwen3ForwardError> {
    forward_layer_with_key_limit(config, weights, layer, hidden_states, None)
}

/// [`forward_layer`] where, under causal attention, keys at or after
/// `key_limit` are masked for every query. Queries past the limit (right
/// padding) still attend the real prefix, as transformers' padding mask does.
fn forward_layer_with_key_limit<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    layer: usize,
    hidden_states: &Array,
    key_limit: Option<i32>,
) -> Result<Array, Qwen3ForwardError> {
    let seq_len = validate_layer_input(config, layer, hidden_states)?;
    let batch = hidden_states.shape()[0];
    let stream = StreamOrDevice::gpu();
    let hidden = as_i32(config.hidden_size)?;
    let intermediate = as_i32(config.intermediate_size)?;
    let base = format!("model.layers.{layer}");
    let attention_input = rms_norm(
        hidden_states,
        weight(weights, &format!("{base}.input_layernorm.weight"))?,
        config.rms_norm_eps,
    )?;
    let attention = attention(
        config,
        weights,
        &base,
        &attention_input,
        batch,
        seq_len,
        key_limit,
    )?;
    let residual = hidden_states.add_device(&attention, &stream)?;

    let mlp_input = rms_norm(
        &residual,
        weight(weights, &format!("{base}.post_attention_layernorm.weight"))?,
        config.rms_norm_eps,
    )?;
    let gate = project(
        config,
        weights,
        &mlp_input,
        &format!("{base}.mlp.gate_proj"),
    )?
    .reshape_device(&[batch, seq_len, intermediate], &stream)?;
    let up = project(config, weights, &mlp_input, &format!("{base}.mlp.up_proj"))?
        .reshape_device(&[batch, seq_len, intermediate], &stream)?;
    let activated = ops::sigmoid_device(&gate, &stream)?.multiply_device(&gate, &stream)?;
    let mlp = project(
        config,
        weights,
        &activated.multiply_device(&up, &stream)?,
        &format!("{base}.mlp.down_proj"),
    )?
    .reshape_device(&[batch, seq_len, hidden], &stream)?;
    residual.add_device(&mlp, &stream).map_err(Into::into)
}

fn apply_residual_steering(
    steering: Option<&Qwen3ResidualSteering>,
    layer: usize,
    absolute_position_start: usize,
    sequence_len: usize,
    hidden_states: &Array,
) -> Result<Array, Qwen3ForwardError> {
    let Some(steering) = steering else {
        return Ok(hidden_states.clone());
    };
    let artifact = &steering.artifact;
    if artifact.layer != layer || artifact.coefficient == 0.0 {
        return Ok(hidden_states.clone());
    }

    let mask = (0..sequence_len)
        .map(|offset| {
            absolute_position_start
                .checked_add(offset)
                .ok_or(Qwen3ForwardError::ShapeOverflow)
                .map(|position| {
                    if artifact.positions.contains(position) {
                        1.0
                    } else {
                        0.0
                    }
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if !mask.iter().any(|value| *value != 0.0) {
        return Ok(hidden_states.clone());
    }

    let hidden =
        i32::try_from(artifact.residual.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let sequence = i32::try_from(sequence_len).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let scaled_residual = artifact
        .residual
        .iter()
        .map(|value| value * artifact.coefficient)
        .collect::<Vec<_>>();
    let stream = StreamOrDevice::gpu();
    let residual = Array::from_slice(&scaled_residual, &[1, 1, hidden]);
    let mask = Array::from_slice(&mask, &[1, sequence, 1]);
    // The artifact is f32; keep the residual stream in the weights' precision
    // so later layers do not promote to f32 on BF16 weights.
    let selected_residual = residual
        .multiply_device(&mask, &stream)?
        .as_dtype_device(hidden_states.dtype(), &stream)?;
    hidden_states
        .add_device(&selected_residual, &stream)
        .map_err(Into::into)
}

/// Applies Qwen3's final RMS normalization to a selected hidden state.
///
/// The streamed qualification owns only the final norm vector at this point,
/// so it uses this narrow helper instead of retaining a decoder weight map.
pub(crate) fn final_rms_norm(
    config: &Qwen3ForwardConfig,
    hidden_state: &Array,
    scale: &Array,
) -> Result<Array, Qwen3ForwardError> {
    rms_norm(hidden_state, scale, config.rms_norm_eps)
}

/// A dense Qwen3 forward executor with KV state bound to one weights map.
///
/// The executor borrows its model tensors, while retaining its own per-layer
/// cache. Consequently there is no public cache object that could be applied
/// to an unrelated checkpoint.
pub struct Qwen3ForwardExecutor<'a, S: BuildHasher> {
    config: &'a Qwen3ForwardConfig,
    weights: &'a HashMap<String, Array, S>,
    cache: Vec<Option<Qwen3LayerKv>>,
    cached_tokens: usize,
    maximum_context_tokens: usize,
    resident_chat_plan: Option<Qwen3ResidentChatPlan>,
    resident_cache_capacity: Option<usize>,
    residual_steering: Option<Box<Qwen3ResidualSteering>>,
}

/// Adapter-local Qwen3 KV state.
///
/// This stays crate-visible: the bounded streamed checker needs to rebuild it
/// after each layer so no cache graph retains that layer's transient weights.
pub(crate) struct Qwen3LayerKv {
    pub(crate) keys: Array,
    pub(crate) values: Array,
}

impl<'a, S: BuildHasher> Qwen3ForwardExecutor<'a, S> {
    /// Creates an executor with an empty cache for this exact weights map.
    #[must_use]
    pub fn new(config: &'a Qwen3ForwardConfig, weights: &'a HashMap<String, Array, S>) -> Self {
        Self {
            config,
            weights,
            cache: (0..config.hidden_layers).map(|_| None).collect(),
            cached_tokens: 0,
            maximum_context_tokens: config.maximum_cached_tokens(),
            resident_chat_plan: None,
            resident_cache_capacity: None,
            residual_steering: None,
        }
    }

    /// Creates a separately bounded resident-chat executor from a preflighted
    /// plan. The normal constructor remains on the 512-token diagnostic cap.
    #[must_use]
    pub(crate) fn new_for_resident_chat(
        config: &'a Qwen3ForwardConfig,
        weights: &'a HashMap<String, Array, S>,
        plan: Qwen3ResidentChatPlan,
    ) -> Self {
        Self {
            config,
            weights,
            cache: (0..config.hidden_layers).map(|_| None).collect(),
            cached_tokens: 0,
            maximum_context_tokens: plan.maximum_context_tokens,
            resident_chat_plan: Some(plan),
            resident_cache_capacity: Some(plan.maximum_context_tokens),
            residual_steering: None,
        }
    }

    /// Starts a resident-chat executor over a weights map the caller loaded,
    /// such as the text decoder of a multimodal checkpoint, after validating
    /// the context and the logical K/V estimate at the weights' precision.
    ///
    /// The map uses this crate's canonical `model.`-prefixed names. Unlike
    /// [`crate::metal::Qwen3MlxWeights`], nothing has checked its tensor
    /// shapes; a missing tensor fails the first forward pass.
    ///
    /// # Errors
    ///
    /// * [`CachedBidirectional`](crate::forward::Qwen3ForwardError::CachedBidirectional),
    ///   [`ResidentChatContextLimit`](crate::forward::Qwen3ForwardError::ResidentChatContextLimit)
    ///   and
    ///   [`ResidentChatKvBudget`](crate::forward::Qwen3ForwardError::ResidentChatKvBudget)
    ///   as `Qwen3ForwardConfig::resident_chat_plan` describes.
    /// * [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when the cache cannot
    ///   be allocated.
    pub fn resident(
        config: &'a Qwen3ForwardConfig,
        weights: &'a HashMap<String, Array, S>,
        maximum_context_tokens: usize,
        maximum_kv_bytes: u64,
    ) -> Result<Self, Qwen3ForwardError> {
        let plan = config.resident_chat_plan(
            maximum_context_tokens,
            maximum_kv_bytes,
            kv_precision(config, weights)?,
        )?;
        Ok(Self::new_for_resident_chat(config, weights, plan))
    }

    /// Clears all resident KV arrays before a new sequence.
    pub fn reset(&mut self) {
        self.cache.iter_mut().for_each(|entry| *entry = None);
        self.cached_tokens = 0;
    }

    /// Installs or removes a residual intervention before this executor has
    /// materialized KV. Refusing a mid-sequence change prevents caches from
    /// mixing incompatible residual histories.
    ///
    /// # Errors
    ///
    /// * [`SteeringRequiresEmptyCache`](crate::forward::Qwen3SteeringError::SteeringRequiresEmptyCache)
    ///   when the executor already holds tokens.
    /// * [`BoundConfigurationMismatch`](crate::forward::Qwen3SteeringError::BoundConfigurationMismatch)
    ///   when `steering` was bound to another configuration.
    pub fn set_residual_steering(
        &mut self,
        steering: Option<Qwen3ResidualSteering>,
    ) -> Result<(), Qwen3SteeringError> {
        if self.cached_tokens != 0 {
            return Err(Qwen3SteeringError::SteeringRequiresEmptyCache);
        }
        if steering
            .as_ref()
            .is_some_and(|steering| !steering.is_bound_to(self.config))
        {
            return Err(Qwen3SteeringError::BoundConfigurationMismatch);
        }
        self.residual_steering = steering.map(Box::new);
        Ok(())
    }

    /// The installed residual intervention, if any.
    #[must_use]
    pub fn residual_steering(&self) -> Option<&Qwen3ResidualSteering> {
        self.residual_steering.as_deref()
    }

    /// Returns the number of tokens represented in every layer's cache.
    #[must_use]
    pub const fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    /// Maximum prompt-plus-generated tokens accepted by this executor.
    #[must_use]
    pub const fn maximum_context_tokens(&self) -> usize {
        self.maximum_context_tokens
    }

    /// The explicit resident-chat plan, when this was created for chat.
    #[must_use]
    pub const fn resident_chat_plan(&self) -> Option<Qwen3ResidentChatPlan> {
        self.resident_chat_plan
    }

    /// Estimated final logical K/V bytes for a resident-chat executor.
    ///
    /// This is absent on the 512-token diagnostic executor and does not
    /// represent an MLX allocation or reservation.
    #[must_use]
    pub const fn planned_kv_bytes(&self) -> Option<u64> {
        match self.resident_chat_plan {
            Some(plan) => Some(plan.planned_kv_bytes),
            None => None,
        }
    }

    /// Returns the byte total of resident K and V arrays, excluding allocator
    /// overhead. MLX reports this from the arrays' actual dtype and shape.
    #[must_use]
    pub fn kv_bytes(&self) -> usize {
        self.cache
            .iter()
            .flatten()
            .map(|entry| entry.keys.nbytes() + entry.values.nbytes())
            .sum()
    }

    /// Starts a sequence, fills its KV cache, and returns its final logits.
    ///
    /// # Errors
    ///
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn prefill_last_logits(
        &mut self,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        self.reset();
        self.append(input_ids, LogitRows::Last)
    }

    /// Appends exactly one token to the current sequence and returns its logits.
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn decode_last_logits(&mut self, input_id: i32) -> Result<Vec<f32>, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        self.append(&[input_id], LogitRows::Last)
    }

    /// Appends a chunk of one or more prompt tokens to a prefilled sequence
    /// and returns the chunk's final logits.
    ///
    /// This lets a caller prefill a shared token prefix once, fork it, and
    /// finish each branch's distinct suffix. A chunk attends causally to the
    /// cached prefix and to its own earlier positions. Logits agree with a
    /// single prefill of the concatenated tokens up to kernel reduction order,
    /// not bit-for-bit: MLX may tile the shorter matmuls differently.
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn extend_last_logits(&mut self, input_ids: &[i32]) -> Result<Vec<f32>, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        self.append(input_ids, LogitRows::Last)
    }

    /// Creates an independent decoder branch from this fully materialized KV
    /// snapshot. The branch borrows the same weights and owns independent
    /// cache handles; it is safe to discard when a controller or verifier
    /// rejects the candidate.
    ///
    /// The retained arrays are evaluated before their handles are cloned.
    /// Later decode appends build replacement K/V arrays through concatenation;
    /// they do not mutate the snapshot arrays owned by this executor.
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`CacheInconsistent`](crate::forward::Qwen3ForwardError::CacheInconsistent)
    ///   when the cache no longer matches the configuration.
    /// * [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when the cache cannot
    ///   be copied.
    pub fn fork_prefilled(&self) -> Result<Self, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        if self.cache.len() != self.config.hidden_layers || self.cache.iter().any(Option::is_none) {
            return Err(Qwen3ForwardError::CacheInconsistent);
        }
        // Resident storage holds the stepped tier for the current length, the
        // same capacity `stepped_cached_kv` allocated for the last append.
        let stored_tokens = match self.resident_cache_capacity {
            Some(maximum_capacity) => stepped_capacity(self.cached_tokens, maximum_capacity)?,
            None => self.cached_tokens,
        };
        let expected_shape = [
            1,
            as_i32(self.config.key_value_heads)?,
            as_i32(stored_tokens)?,
            as_i32(self.config.head_dim)?,
        ];
        for layer in self.cache.iter().flatten() {
            if layer.keys.shape() != expected_shape || layer.values.shape() != expected_shape {
                return Err(Qwen3ForwardError::CacheInconsistent);
            }
            layer.keys.eval()?;
            layer.values.eval()?;
        }
        Ok(Self {
            config: self.config,
            weights: self.weights,
            cache: self
                .cache
                .iter()
                .map(|entry| {
                    entry.as_ref().map(|layer| Qwen3LayerKv {
                        keys: layer.keys.clone(),
                        values: layer.values.clone(),
                    })
                })
                .collect(),
            cached_tokens: self.cached_tokens,
            maximum_context_tokens: self.maximum_context_tokens,
            resident_chat_plan: self.resident_chat_plan,
            resident_cache_capacity: self.resident_cache_capacity,
            residual_steering: self.residual_steering.clone(),
        })
    }

    fn append(
        &mut self,
        input_ids: &[i32],
        rows: LogitRows,
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        let result = self.append_inner(input_ids, rows);
        if result.is_err() {
            // A failed graph may have appended only some layers. Retaining
            // that partial state would make a later decode numerically wrong.
            self.reset();
        }
        result
    }

    fn append_inner(
        &mut self,
        input_ids: &[i32],
        rows: LogitRows,
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        validate_input_ids(
            self.config,
            input_ids,
            self.cached_tokens,
            self.maximum_context_tokens,
        )?;
        let mut input_ids = input_ids;
        let piece = prefill_chunk_tokens();
        if matches!(rows, LogitRows::Last) && input_ids.len() > piece {
            // Every piece but the last updates the cache only; its unread
            // logits graph is dropped unevaluated.
            let head = (input_ids.len() - 1) / piece * piece;
            for chunk in input_ids[..head].chunks(piece) {
                let len =
                    i32::try_from(chunk.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
                drop(self.append_ids(&Array::from_slice(chunk, &[len]), len, LogitRows::Last)?);
                let state: Vec<&Array> = self
                    .cache
                    .iter()
                    .flatten()
                    .flat_map(|layer| [&layer.keys, &layer.values])
                    .collect();
                mlx_rs::transforms::eval(state)?;
            }
            input_ids = &input_ids[head..];
        }
        let seq_len =
            i32::try_from(input_ids.len()).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        let logits = self.append_ids(&Array::from_slice(input_ids, &[seq_len]), seq_len, rows)?;
        match rows {
            LogitRows::Last => read_last_logits(&logits, 1, self.config.vocab_size),
            LogitRows::All => verify::read_all_logits(&logits, seq_len, self.config.vocab_size),
            LogitRows::Hidden => Err(Qwen3ForwardError::HiddenRowsAreNotLogits),
        }
    }

    /// Builds, without evaluating, the graph that appends `ids` (shape
    /// `[seq_len]`, already validated against the vocabulary and context) to
    /// every layer's cache, and returns the logits `rows` selects: the last
    /// position's `[1, 1, vocab]`, every position's `[1, seq_len, vocab]`,
    /// or every position's normalized `[1, seq_len, hidden]` state.
    fn append_ids(
        &mut self,
        ids: &Array,
        seq_len: i32,
        rows: LogitRows,
    ) -> Result<Array, Qwen3ForwardError> {
        // Keep rejection before graph construction on the ordinary ids path.
        if self.config.attention != Qwen3Attention::Causal {
            return Err(Qwen3ForwardError::CachedBidirectional);
        }
        let hidden = as_i32(self.config.hidden_size)?;
        let hidden_states = embed_rows(self.config, self.weights, ids)?
            .reshape_device(&[1, seq_len, hidden], StreamOrDevice::gpu())?;
        self.append_hidden(hidden_states, seq_len, rows)
    }

    /// Appends validated input embeddings `[1, seq_len, hidden]`.
    fn append_hidden(
        &mut self,
        mut hidden_states: Array,
        seq_len: i32,
        rows: LogitRows,
    ) -> Result<Array, Qwen3ForwardError> {
        // A cache of earlier positions cannot serve a layer whose earlier
        // positions also attend to later ones.
        if self.config.attention != Qwen3Attention::Causal {
            return Err(Qwen3ForwardError::CachedBidirectional);
        }
        let appended = usize::try_from(seq_len).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        let stream = StreamOrDevice::gpu();

        let rope_offset =
            i32::try_from(self.cached_tokens).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        for layer in 0..self.config.hidden_layers {
            hidden_states = forward_cached_layer_with_capacity(
                self.config,
                self.weights,
                layer,
                self.cache
                    .get_mut(layer)
                    .ok_or(Qwen3ForwardError::CacheInconsistent)?,
                &hidden_states,
                seq_len,
                rope_offset,
                self.resident_cache_capacity,
            )?;
            hidden_states = apply_residual_steering(
                self.residual_steering.as_deref(),
                layer,
                self.cached_tokens,
                appended,
                &hidden_states,
            )?;
        }

        self.cached_tokens += appended;
        let selected = match rows {
            LogitRows::Last => hidden_states.take_axis_device(
                Array::from_slice(&[seq_len - 1], &[1]),
                1,
                &stream,
            )?,
            LogitRows::All | LogitRows::Hidden => hidden_states,
        };
        let normalized = rms_norm(
            &selected,
            weight(self.weights, "model.norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        if matches!(rows, LogitRows::Hidden) {
            return Ok(normalized);
        }
        project(
            self.config,
            self.weights,
            &normalized,
            self.config.output_projection(),
        )
    }

    /// Appends one host-known token and starts computing the greedy next
    /// token on the GPU without waiting for it.
    ///
    /// Only the selected ID and a finiteness flag are read back, not the
    /// vocabulary row. Ties go to the lowest token ID, as in a host argmax
    /// that keeps the first maximum.
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn decode_greedy(&mut self, input_id: i32) -> Result<Qwen3TokenPicks, Qwen3ForwardError> {
        self.decode_picks(input_id, &Qwen3PickRule::GREEDY)
    }

    /// Appends `previous` while it may still be computing and starts the
    /// greedy token after it, so the host builds step `t + 1` while the GPU
    /// runs step `t`.
    ///
    /// The cache then holds `previous` even if the caller stops on it (an
    /// end-of-sequence token); [`Self::truncate_cached_tokens`] removes it.
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`CacheInconsistent`](crate::forward::Qwen3ForwardError::CacheInconsistent)
    ///   when the cache no longer matches the configuration.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    pub fn decode_greedy_after(
        &mut self,
        previous: &Qwen3TokenPicks,
    ) -> Result<Qwen3TokenPicks, Qwen3ForwardError> {
        self.decode_picks_after(previous, &Qwen3PickRule::GREEDY)
    }

    /// [`Self::decode_greedy`] under any [`Qwen3PickRule`].
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    /// * [`InvalidPickRule`](crate::forward::Qwen3ForwardError::InvalidPickRule)
    ///   for a rule outside its domain, and
    ///   [`NonFiniteLogits`](crate::forward::Qwen3ForwardError::NonFiniteLogits)
    ///   when the logits cannot be picked from.
    pub fn decode_picks(
        &mut self,
        input_id: i32,
        rule: &Qwen3PickRule,
    ) -> Result<Qwen3TokenPicks, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        validate_input_ids(
            self.config,
            &[input_id],
            self.cached_tokens,
            self.maximum_context_tokens,
        )?;
        self.append_picks(&Array::from_slice(&[input_id], &[1]), rule)
    }

    /// [`Self::decode_greedy_after`] under any [`Qwen3PickRule`].
    ///
    /// # Errors
    ///
    /// * [`DecodeWithoutPrefill`](crate::forward::Qwen3ForwardError::DecodeWithoutPrefill)
    ///   when the sequence has not been prefilled.
    /// * [`CacheInconsistent`](crate::forward::Qwen3ForwardError::CacheInconsistent)
    ///   when the cache no longer matches the configuration.
    /// * [`EmptyInput`](crate::forward::Qwen3ForwardError::EmptyInput),
    ///   [`InvalidTokenId`](crate::forward::Qwen3ForwardError::InvalidTokenId)
    ///   and
    ///   [`PromptTooLong`](crate::forward::Qwen3ForwardError::PromptTooLong)
    ///   when the input is empty, names a token outside the vocabulary, or
    ///   passes the context limit.
    /// * [`MissingWeight`](crate::forward::Qwen3ForwardError::MissingWeight)
    ///   and [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when a weight is
    ///   absent or MLX cannot build or evaluate the graph.
    /// * [`InvalidPickRule`](crate::forward::Qwen3ForwardError::InvalidPickRule)
    ///   for a rule outside its domain, and
    ///   [`NonFiniteLogits`](crate::forward::Qwen3ForwardError::NonFiniteLogits)
    ///   when the logits cannot be picked from.
    pub fn decode_picks_after(
        &mut self,
        previous: &Qwen3TokenPicks,
        rule: &Qwen3PickRule,
    ) -> Result<Qwen3TokenPicks, Qwen3ForwardError> {
        if self.cached_tokens == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        if previous.binding != self.weights_address() || previous.tokens.shape() != [1] {
            return Err(Qwen3ForwardError::CacheInconsistent);
        }
        let total = self
            .cached_tokens
            .checked_add(1)
            .ok_or(Qwen3ForwardError::ShapeOverflow)?;
        if total > self.maximum_context_tokens {
            return Err(Qwen3ForwardError::PromptTooLong {
                actual: total,
                maximum: self.maximum_context_tokens,
            });
        }
        // A pick over this checkpoint's logit rows is a valid token ID, so
        // the vocabulary check host IDs get is not needed here.
        self.append_picks(&previous.tokens, rule)
    }

    fn append_picks(
        &mut self,
        ids: &Array,
        rule: &Qwen3PickRule,
    ) -> Result<Qwen3TokenPicks, Qwen3ForwardError> {
        let pending = self.append_ids(ids, 1, LogitRows::Last).and_then(|logits| {
            let rows = logits
                .reshape_device(&[1, as_i32(self.config.vocab_size)?], StreamOrDevice::gpu())?;
            Qwen3TokenPicks::start_with(&rows, self.weights_address(), rule)
        });
        if pending.is_err() {
            // As in `append`: a partly built step must not leave some layers
            // one position ahead of the others.
            self.reset();
        }
        pending
    }

    /// Shortens the sequence to its first `tokens` cached positions, such as
    /// dropping a token appended by [`Self::decode_greedy_after`] that the
    /// caller then stopped on, or rejected draft positions.
    ///
    /// # Errors
    ///
    /// Returns
    /// [`TruncateOutOfRange`](crate::forward::Qwen3ForwardError::TruncateOutOfRange)
    /// when `tokens` is more than the cached length, and
    /// [`Mlx`](crate::forward::Qwen3ForwardError::Mlx) when the cache cannot be
    /// cut.
    pub fn truncate_cached_tokens(&mut self, tokens: usize) -> Result<(), Qwen3ForwardError> {
        if tokens == 0 || tokens > self.cached_tokens {
            return Err(Qwen3ForwardError::TruncateOutOfRange {
                requested: tokens,
                cached: self.cached_tokens,
            });
        }
        // Resident storage shrinks to the tier `stepped_cached_kv` chooses for
        // the shorter sequence, which forks check; unbounded caches hold
        // exactly the cached positions. Storage is never widened: compact
        // storage restored from a snapshot stays compact, and the next append
        // re-tiers it, as after `from_snapshot`.
        let stored = match self.resident_cache_capacity {
            Some(maximum_capacity) => stepped_capacity(tokens, maximum_capacity)?,
            None => tokens,
        };
        let stored = as_i32(stored)?;
        let stream = StreamOrDevice::gpu();
        for layer in self.cache.iter_mut().flatten() {
            if layer
                .keys
                .shape()
                .get(2)
                .is_some_and(|&length| length > stored)
            {
                layer.keys = layer.keys.index_device((.., .., 0..stored, ..), &stream);
                layer.values = layer.values.index_device((.., .., 0..stored, ..), &stream);
            }
        }
        self.cached_tokens = tokens;
        Ok(())
    }

    /// Identifies the borrowed weights map, so a pending token is never fed
    /// to an executor of another checkpoint load.
    fn weights_address(&self) -> usize {
        std::ptr::from_ref(self.weights).addr()
    }
}

/// Which appended positions an executor append projects to logits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LogitRows {
    /// Only the final position: prefill and ordinary decode.
    Last,
    /// Every appended position: speculative verification and scoring.
    All,
    /// Every appended position's normalized hidden state, before the output
    /// head: scoring applies the head itself, a bounded chunk at a time.
    Hidden,
}

/// Executes one cached Qwen3 decoder layer and updates its adapter-local KV.
///
/// The cache is intentionally not a cross-model engine interface. It carries
/// Qwen3's GQA layout and `RoPE` contract, while allowing the streamed checker
/// to detach the cache immediately after the layer evaluates.
pub(crate) fn forward_cached_layer<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    layer: usize,
    cache: &mut Option<Qwen3LayerKv>,
    hidden_states: &Array,
    seq_len: i32,
    rope_offset: i32,
) -> Result<Array, Qwen3ForwardError> {
    forward_cached_layer_with_capacity(
        config,
        weights,
        layer,
        cache,
        hidden_states,
        seq_len,
        rope_offset,
        None,
    )
}

#[allow(
    clippy::too_many_arguments,
    reason = "the adapter keeps cache capacity explicit at this narrow execution boundary"
)]
fn forward_cached_layer_with_capacity<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    layer: usize,
    cache: &mut Option<Qwen3LayerKv>,
    hidden_states: &Array,
    seq_len: i32,
    rope_offset: i32,
    resident_cache_capacity: Option<usize>,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let base = format!("model.layers.{layer}");
    let attention_input = rms_norm(
        hidden_states,
        weight(weights, &format!("{base}.input_layernorm.weight"))?,
        config.rms_norm_eps,
    )?;
    let attention = cached_attention(
        config,
        weights,
        cache,
        &base,
        &attention_input,
        seq_len,
        rope_offset,
        resident_cache_capacity,
    )?;
    let residual = hidden_states.add_device(&attention, &stream)?;
    mlp_residual(config, weights, &base, &residual, seq_len)
}

/// The post-attention half of a single-sequence decoder layer: RMS norm,
/// `SiLU`-gated MLP, and the residual add, on `[1, seq_len, hidden]`.
fn mlp_residual<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    base: &str,
    residual: &Array,
    seq_len: i32,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let hidden = as_i32(config.hidden_size)?;
    let intermediate = as_i32(config.intermediate_size)?;
    let mlp_input = rms_norm(
        residual,
        weight(weights, &format!("{base}.post_attention_layernorm.weight"))?,
        config.rms_norm_eps,
    )?;
    let gate = project(
        config,
        weights,
        &mlp_input,
        &format!("{base}.mlp.gate_proj"),
    )?
    .reshape_device(&[1, seq_len, intermediate], &stream)?;
    let up = project(config, weights, &mlp_input, &format!("{base}.mlp.up_proj"))?
        .reshape_device(&[1, seq_len, intermediate], &stream)?;
    let activated = ops::sigmoid_device(&gate, &stream)?.multiply_device(&gate, &stream)?;
    let mlp = project(
        config,
        weights,
        &activated.multiply_device(&up, &stream)?,
        &format!("{base}.mlp.down_proj"),
    )?
    .reshape_device(&[1, seq_len, hidden], &stream)?;
    residual.add_device(&mlp, &stream).map_err(Into::into)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the adapter keeps cache capacity explicit at this narrow execution boundary"
)]
fn cached_attention<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    cache: &mut Option<Qwen3LayerKv>,
    base: &str,
    input: &Array,
    seq_len: i32,
    rope_offset: i32,
    resident_cache_capacity: Option<usize>,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let attn = format!("{base}.self_attn");
    let (query, key, value) = rotated_qkv(
        config,
        weights,
        &attn,
        input,
        1,
        seq_len,
        RopePositions::Shared(rope_offset),
    )?;
    let (keys, values, attention_keys, attention_values, causal) = match resident_cache_capacity {
        Some(maximum_capacity) => {
            stepped_cached_kv(cache, &key, &value, rope_offset, maximum_capacity, &stream)?
        }
        None => {
            if let Some(previous) = cache.take() {
                (
                    ops::concatenate_axis_device(&[&previous.keys, &key], 2, &stream)?,
                    ops::concatenate_axis_device(&[&previous.values, &value], 2, &stream)?,
                    None,
                    None,
                    false,
                )
            } else {
                (key, value, None, None, true)
            }
        }
    };
    // A multi-token chunk after cached positions needs a causal mask aligned
    // to the last key: chunk query i sees the cached keys and its own earlier
    // chunk positions. MLX 0.25's fused kernel under-masked `Causal` when the
    // cached length was not a multiple of its key block, so this used an
    // explicit `[chunk, cached + chunk]` mask (512 MB per call at 262K
    // cached). The linked MLX (0.32.2) aligns `Causal` to the last key;
    // `fused_causal_mask_matches_the_explicit_chunk_mask` pins that. A single
    // decode token sees every key and keeps the unmasked kernel.
    let attention_keys = attention_keys.as_ref().unwrap_or(&keys);
    let attention_values = attention_values.as_ref().unwrap_or(&values);
    // Keep KV as dependencies of attention. The final-logits readback evaluates
    // the complete graph, including these retained arrays, in one submission
    // instead of blocking twice per layer. Reset drops all request-owned KV.
    let mask = (causal || seq_len > 1).then_some(fast::ScaledDotProductAttentionMask::Causal);
    let output = fast::scaled_dot_product_attention_device(
        &query,
        attention_keys,
        attention_values,
        attention_scale(config)?,
        mask,
        Option::<&Array>::None,
        &stream,
    )?;
    *cache = Some(Qwen3LayerKv { keys, values });
    attention_output(config, weights, &attn, &output, seq_len)
}

/// Where `RoPE` starts for the rows of a [`rotated_qkv`] call.
#[derive(Clone, Copy)]
enum RopePositions<'a> {
    /// Every row starts at this absolute position.
    Shared(i32),
    /// One int32 start per row, shape `[rows]`.
    PerRow(&'a Array),
}

/// Projects attention input to query, key and value, each `[rows, heads,
/// seq_len, head_dim]`, with Qwen3's Q/K norms and `RoPE` applied.
///
/// The input is `[1, rows * seq_len, hidden]` (one sequence when `rows` is
/// one, or one token per decode row when `seq_len` is one).
#[allow(
    clippy::too_many_arguments,
    reason = "the row layout and RoPE positions stay explicit at this boundary"
)]
fn rotated_qkv<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    attn: &str,
    input: &Array,
    rows: i32,
    seq_len: i32,
    positions: RopePositions<'_>,
) -> Result<(Array, Array, Array), Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let heads = as_i32(config.attention_heads)?;
    let kv_heads = as_i32(config.key_value_heads)?;
    let head_dim = as_i32(config.head_dim)?;
    let query = project(config, weights, input, &format!("{attn}.q_proj"))?
        .reshape_device(&[rows, seq_len, heads, head_dim], &stream)?;
    let key = project(config, weights, input, &format!("{attn}.k_proj"))?
        .reshape_device(&[rows, seq_len, kv_heads, head_dim], &stream)?;
    let value = project(config, weights, input, &format!("{attn}.v_proj"))?
        .reshape_device(&[rows, seq_len, kv_heads, head_dim], &stream)?;
    let rope = |projected: Array| -> Result<Array, Qwen3ForwardError> {
        Ok(match positions {
            RopePositions::Shared(offset) => fast::rope_device(
                &projected,
                head_dim,
                false,
                Some(config.rope_theta),
                1.0,
                offset,
                Option::<&Array>::None,
                &stream,
            )?,
            RopePositions::PerRow(offsets) => fast::rope_dynamic_device(
                &projected,
                head_dim,
                false,
                Some(config.rope_theta),
                1.0,
                offsets,
                Option::<&Array>::None,
                &stream,
            )?,
        })
    };
    let query = rope(
        qk_norm(config, weights, attn, "q_norm", query)?
            .transpose_axes_device(&[0, 2, 1, 3], &stream)?,
    )?;
    let key = rope(
        qk_norm(config, weights, attn, "k_norm", key)?
            .transpose_axes_device(&[0, 2, 1, 3], &stream)?,
    )?;
    let value = value.transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    Ok((query, key, value))
}

/// Merges `[rows, heads, len, head_dim]` attention output back to the packed
/// `[1, rows * len, heads * head_dim]` layout and applies the output
/// projection. `seq_len` is `rows * len`.
fn attention_output<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    attn: &str,
    output: &Array,
    seq_len: i32,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let heads = as_i32(config.attention_heads)?;
    let head_dim = as_i32(config.head_dim)?;
    let output = output
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?
        .reshape_device(
            &[
                1,
                seq_len,
                heads
                    .checked_mul(head_dim)
                    .ok_or(Qwen3ForwardError::ShapeOverflow)?,
            ],
            &stream,
        )?;
    project(config, weights, &output, &format!("{attn}.o_proj"))
}

/// `[chunk, cached + chunk]` boolean mask letting chunk position `i` (absolute
/// `cached + i`) attend to every key at or before it. Both executors now use
/// the fused `Causal` mask for later chunks; this stays as the oracle that
/// checks the fused mask's alignment.
#[cfg(test)]
fn chunk_causal_mask(
    cached_tokens: i32,
    chunk_tokens: i32,
    stream: &StreamOrDevice,
) -> Result<Array, Qwen3ForwardError> {
    let total = cached_tokens
        .checked_add(chunk_tokens)
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    let queries = Array::arange_device::<i32, i32>(cached_tokens, total, None, stream)?
        .reshape_device(&[chunk_tokens, 1], stream)?;
    let keys = Array::arange_device::<i32, i32>(0, total, None, stream)?
        .reshape_device(&[1, total], stream)?;
    queries.ge_device(&keys, stream).map_err(Into::into)
}

type SteppedKv = (Array, Array, Option<Array>, Option<Array>, bool);

fn stepped_cached_kv(
    cache: &mut Option<Qwen3LayerKv>,
    key: &Array,
    value: &Array,
    rope_offset: i32,
    maximum_capacity: usize,
    stream: &StreamOrDevice,
) -> Result<SteppedKv, Qwen3ForwardError> {
    let appended = *key
        .shape()
        .get(2)
        .ok_or(Qwen3ForwardError::CacheInconsistent)?;
    let next = rope_offset
        .checked_add(appended)
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    let next_usize = usize::try_from(next).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let capacity = stepped_capacity(next_usize, maximum_capacity)?;
    let capacity_i32 = as_i32(capacity)?;
    let key_shape = key.shape();
    let value_shape = value.shape();
    if key_shape.len() != 4 || value_shape != key_shape {
        return Err(Qwen3ForwardError::CacheInconsistent);
    }
    let expected_storage_shape = [key_shape[0], key_shape[1], capacity_i32, key_shape[3]];
    let (mut keys, mut values, causal) = match cache.take() {
        Some(previous)
            if previous.keys.shape() == expected_storage_shape
                && previous.values.shape() == expected_storage_shape =>
        {
            (previous.keys, previous.values, false)
        }
        Some(previous) => {
            let prior = previous
                .keys
                .shape()
                .get(2)
                .copied()
                .ok_or(Qwen3ForwardError::CacheInconsistent)?;
            if previous.keys.shape().len() != 4
                || previous.values.shape() != previous.keys.shape()
                || previous.keys.shape()[0] != key_shape[0]
                || previous.keys.shape()[1] != key_shape[1]
                || previous.keys.shape()[3] != key_shape[3]
                || prior < rope_offset
            {
                return Err(Qwen3ForwardError::CacheInconsistent);
            }
            let key_prefix = previous
                .keys
                .index_device((.., .., 0..rope_offset, ..), stream);
            let value_prefix = previous
                .values
                .index_device((.., .., 0..rope_offset, ..), stream);
            let mut keys = ops::zeros_dtype_device(&expected_storage_shape, key.dtype(), stream)?;
            let mut values =
                ops::zeros_dtype_device(&expected_storage_shape, value.dtype(), stream)?;
            keys.index_mut_device((.., .., 0..rope_offset, ..), &key_prefix, stream);
            values.index_mut_device((.., .., 0..rope_offset, ..), &value_prefix, stream);
            (keys, values, false)
        }
        None => (
            ops::zeros_dtype_device(&expected_storage_shape, key.dtype(), stream)?,
            ops::zeros_dtype_device(&expected_storage_shape, value.dtype(), stream)?,
            true,
        ),
    };
    keys.index_mut_device((.., .., rope_offset..next, ..), &key, stream);
    values.index_mut_device((.., .., rope_offset..next, ..), &value, stream);
    let attention_keys = keys.index_device((.., .., 0..next, ..), stream);
    let attention_values = values.index_device((.., .., 0..next, ..), stream);
    Ok((
        keys,
        values,
        Some(attention_keys),
        Some(attention_values),
        causal,
    ))
}

fn stepped_capacity(
    next_tokens: usize,
    maximum_capacity: usize,
) -> Result<usize, Qwen3ForwardError> {
    if next_tokens > maximum_capacity {
        return Err(Qwen3ForwardError::PromptTooLong {
            actual: next_tokens,
            maximum: maximum_capacity,
        });
    }
    // Tiers 128 and 512, then doubling, so a short sequence under a long
    // context limit never allocates K/V for the whole limit. Each doubling
    // copies the prefix once, which is amortized O(1) per appended token.
    let mut boundary = 128;
    while boundary < next_tokens {
        boundary = if boundary == 128 {
            512
        } else {
            boundary.saturating_mul(2)
        };
    }
    Ok(boundary.min(maximum_capacity))
}

fn validate_input_ids(
    config: &Qwen3ForwardConfig,
    input_ids: &[i32],
    existing_tokens: usize,
    maximum: usize,
) -> Result<(), Qwen3ForwardError> {
    if input_ids.is_empty() {
        return Err(Qwen3ForwardError::EmptyInput);
    }
    let total = existing_tokens
        .checked_add(input_ids.len())
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    if total > maximum {
        return Err(Qwen3ForwardError::PromptTooLong {
            actual: total,
            maximum,
        });
    }
    for &id in input_ids {
        if id < 0
            || usize::try_from(id)
                .ok()
                .is_none_or(|id| id >= config.vocab_size)
        {
            return Err(Qwen3ForwardError::InvalidTokenId {
                token_id: id,
                vocab_size: config.vocab_size,
            });
        }
    }
    Ok(())
}

fn validate_layer_input(
    config: &Qwen3ForwardConfig,
    layer: usize,
    hidden_states: &Array,
) -> Result<i32, Qwen3ForwardError> {
    if layer >= config.hidden_layers {
        return Err(Qwen3ForwardError::LayerOutOfRange {
            layer,
            hidden_layers: config.hidden_layers,
        });
    }

    let shape = hidden_states.shape();
    let expected_hidden = as_i32(config.hidden_size)?;
    let Some(&seq_len) = shape.get(1) else {
        return Err(Qwen3ForwardError::InvalidLayerInputShape {
            actual: shape.to_vec(),
            hidden_size: config.hidden_size,
        });
    };
    if shape.len() != 3 || shape[0] <= 0 || shape[2] != expected_hidden || seq_len <= 0 {
        return Err(Qwen3ForwardError::InvalidLayerInputShape {
            actual: shape.to_vec(),
            hidden_size: config.hidden_size,
        });
    }
    let sequence = usize::try_from(seq_len).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let maximum = MAX_DENSE_DEBUG_TOKENS.min(config.max_position_embeddings);
    if sequence > maximum {
        return Err(Qwen3ForwardError::PromptTooLong {
            actual: sequence,
            maximum,
        });
    }
    Ok(seq_len)
}

fn attention_scale(config: &Qwen3ForwardConfig) -> Result<f32, Qwen3ForwardError> {
    Ok(
        f32::from(u16::try_from(config.head_dim).map_err(|_| Qwen3ForwardError::ShapeOverflow)?)
            .powf(-0.5),
    )
}

fn read_last_logits(
    logits: &Array,
    seq_len: i32,
    vocab_size: usize,
) -> Result<Vec<f32>, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let last = logits
        .take_axis_device(Array::from_slice(&[seq_len - 1], &[1]), 1, &stream)?
        .reshape_device(&[as_i32(vocab_size)?], &stream)?
        .as_type_device::<f32>(&stream)?;
    #[cfg(test)]
    let evaluation_started = Instant::now();
    last.eval()?;
    #[cfg(test)]
    record_decode_profile_evaluation(evaluation_started.elapsed());

    #[cfg(test)]
    let readback_started = Instant::now();
    let output = last.as_slice::<f32>().to_vec();
    #[cfg(test)]
    record_decode_profile_readback(readback_started.elapsed());
    Ok(output)
}

fn attention<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    base: &str,
    input: &Array,
    batch: i32,
    seq_len: i32,
    key_limit: Option<i32>,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let heads = as_i32(config.attention_heads)?;
    let kv_heads = as_i32(config.key_value_heads)?;
    let head_dim = as_i32(config.head_dim)?;
    let attn = format!("{base}.self_attn");
    let query = project(config, weights, input, &format!("{attn}.q_proj"))?
        .reshape_device(&[batch, seq_len, heads, head_dim], &stream)?;
    let key = project(config, weights, input, &format!("{attn}.k_proj"))?
        .reshape_device(&[batch, seq_len, kv_heads, head_dim], &stream)?;
    let value = project(config, weights, input, &format!("{attn}.v_proj"))?
        .reshape_device(&[batch, seq_len, kv_heads, head_dim], &stream)?;

    // Qwen3 normalizes Q and K per head; both families then apply
    // nontraditional (rotate-half) RoPE.
    let query = qk_norm(config, weights, &attn, "q_norm", query)?
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    let key = qk_norm(config, weights, &attn, "k_norm", key)?
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    let value = value.transpose_axes_device(&[0, 2, 1, 3], &stream)?;
    let query = fast::rope_device(
        &query,
        head_dim,
        false,
        Some(config.rope_theta),
        1.0,
        0,
        Option::<&Array>::None,
        &stream,
    )?;
    let key = fast::rope_device(
        &key,
        head_dim,
        false,
        Some(config.rope_theta),
        1.0,
        0,
        Option::<&Array>::None,
        &stream,
    )?;
    let key_limit_mask = match (config.attention, key_limit) {
        (Qwen3Attention::Causal, Some(limit)) => {
            Some(causal_key_limit_mask(seq_len, limit, query.dtype())?)
        }
        _ => None,
    };
    let output = fast::scaled_dot_product_attention_device(
        &query,
        &key,
        &value,
        attention_scale(config)?,
        match (&key_limit_mask, config.attention) {
            (Some(mask), _) => Some(fast::ScaledDotProductAttentionMask::Array(mask)),
            (None, Qwen3Attention::Causal) => Some(fast::ScaledDotProductAttentionMask::Causal),
            // One unpadded sequence: every position attends to every position.
            (None, Qwen3Attention::Bidirectional) => None,
        },
        Option::<&Array>::None,
        &stream,
    )?
    .transpose_axes_device(&[0, 2, 1, 3], &stream)?
    .reshape_device(
        &[
            batch,
            seq_len,
            heads
                .checked_mul(head_dim)
                .ok_or(Qwen3ForwardError::ShapeOverflow)?,
        ],
        &stream,
    )?;
    project(config, weights, &output, &format!("{attn}.o_proj"))
}

fn causal_key_limit_mask(
    seq_len: i32,
    limit: i32,
    dtype: Dtype,
) -> Result<Array, Qwen3ForwardError> {
    let elements = seq_len
        .checked_mul(seq_len)
        .ok_or(Qwen3ForwardError::ShapeOverflow)?;
    let elements = usize::try_from(elements).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let mut mask = Vec::with_capacity(elements);
    for query in 0..seq_len {
        for key in 0..seq_len {
            let visible = key <= query && key < limit;
            mask.push(if visible { 0.0_f32 } else { f32::NEG_INFINITY });
        }
    }
    Ok(Array::from_slice(&mask, &[seq_len, seq_len])
        .as_dtype_device(dtype, StreamOrDevice::gpu())?)
}

/// Qwen3's per-head RMS norm on a `[batch, positions, heads, head_dim]`
/// query or key; Llama has none and passes the projection through.
fn qk_norm<S: BuildHasher>(
    config: &Qwen3ForwardConfig,
    weights: &HashMap<String, Array, S>,
    attn: &str,
    norm: &str,
    projected: Array,
) -> Result<Array, Qwen3ForwardError> {
    if !config.family.has_qk_norm() {
        return Ok(projected);
    }
    rms_norm(
        &projected,
        weight(weights, &format!("{attn}.{norm}.weight"))?,
        config.rms_norm_eps,
    )
}

fn rms_norm(input: &Array, scale: &Array, eps: f32) -> Result<Array, Qwen3ForwardError> {
    Ok(fast::rms_norm_device(
        input,
        scale,
        eps,
        StreamOrDevice::gpu(),
    )?)
}

/// Per-thread, opt-in timings for the local checkpoint decode-profile test.
///
/// These durations are host-side intervals around MLX API calls. In
/// particular, `evaluation` includes waiting for MLX work and is not GPU time.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(super) struct DecodeProfile {
    pub(super) transpose_node: Duration,
    pub(super) transpose_node_count: usize,
    pub(super) evaluation: Duration,
    pub(super) evaluation_count: usize,
    pub(super) readback: Duration,
    pub(super) readback_count: usize,
}

/// [`PREFILL_CHUNK_TOKENS`], or a test's override on this thread.
fn prefill_chunk_tokens() -> usize {
    #[cfg(test)]
    if let Some(tokens) = PREFILL_CHUNK_OVERRIDE.with(std::cell::Cell::get) {
        return tokens;
    }
    PREFILL_CHUNK_TOKENS
}

#[cfg(test)]
thread_local! {
    /// Piece size for tests that compare chunked and single-graph prefill.
    static PREFILL_CHUNK_OVERRIDE: std::cell::Cell<Option<usize>> =
        const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    static DECODE_PROFILE: RefCell<Option<DecodeProfile>> = const { RefCell::new(None) };
}

#[cfg(test)]
fn record_decode_profile_transpose_node(elapsed: Duration) {
    DECODE_PROFILE.with(|profile| {
        if let Some(profile) = profile.borrow_mut().as_mut() {
            profile.transpose_node += elapsed;
            profile.transpose_node_count += 1;
        }
    });
}

#[cfg(test)]
fn record_decode_profile_evaluation(elapsed: Duration) {
    DECODE_PROFILE.with(|profile| {
        if let Some(profile) = profile.borrow_mut().as_mut() {
            profile.evaluation += elapsed;
            profile.evaluation_count += 1;
        }
    });
}

#[cfg(test)]
fn record_decode_profile_readback(elapsed: Duration) {
    DECODE_PROFILE.with(|profile| {
        if let Some(profile) = profile.borrow_mut().as_mut() {
            profile.readback += elapsed;
            profile.readback_count += 1;
        }
    });
}

#[cfg(test)]
pub(super) struct DecodeProfileScope {
    active: bool,
}

#[cfg(test)]
impl DecodeProfileScope {
    pub(super) fn start() -> Self {
        DECODE_PROFILE.with(|profile| {
            assert!(
                profile.borrow().is_none(),
                "decode profile is already active"
            );
            *profile.borrow_mut() = Some(DecodeProfile::default());
        });
        Self { active: true }
    }

    pub(super) fn finish(mut self) -> DecodeProfile {
        let profile = DECODE_PROFILE.with(|profile| {
            profile
                .borrow_mut()
                .take()
                .expect("decode profile scope remains active")
        });
        self.active = false;
        profile
    }
}

#[cfg(test)]
impl Drop for DecodeProfileScope {
    fn drop(&mut self) {
        if self.active {
            DECODE_PROFILE.with(|profile| {
                profile.borrow_mut().take();
            });
        }
    }
}

fn weight<'a, S: BuildHasher>(
    weights: &'a HashMap<String, Array, S>,
    name: &str,
) -> Result<&'a Array, Qwen3ForwardError> {
    weights
        .get(name)
        .ok_or_else(|| Qwen3ForwardError::MissingWeight(name.to_owned()))
}

fn as_i32(value: usize) -> Result<i32, Qwen3ForwardError> {
    i32::try_from(value).map_err(|_| Qwen3ForwardError::ShapeOverflow)
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "the fields directly mirror upstream Qwen JSON gates"
)]
#[derive(Debug, Deserialize)]
struct RawForwardConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    use_bidirectional_attention: bool,
    #[serde(default)]
    num_hidden_layers: usize,
    #[serde(default)]
    hidden_size: usize,
    #[serde(default)]
    intermediate_size: usize,
    #[serde(default)]
    vocab_size: usize,
    #[serde(default)]
    num_attention_heads: usize,
    #[serde(default)]
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: usize,
    #[serde(default)]
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    rope_theta: Option<f32>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    hidden_act: String,
    #[serde(default)]
    tie_word_embeddings: bool,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<RawRopeParameters>,
    sliding_window: Option<usize>,
    #[serde(default)]
    use_sliding_window: bool,
    /// mlx-lm's `{"group_size": .., "bits": ..}`; per-module entries would
    /// mean mixed precision, which is refused.
    quantization: Option<serde_json::Map<String, serde_json::Value>>,
    /// transformers' quantization (fp8, GPTQ, AWQ, ...). mlx-lm writes a
    /// copy of `quantization` here; anything else is refused.
    quantization_config: Option<serde_json::Value>,
}

fn parse_quantization(
    quantization: Option<&serde_json::Map<String, serde_json::Value>>,
    quantization_config: Option<&serde_json::Value>,
) -> Result<Option<Qwen3AffineQuantization>, Qwen3ForwardError> {
    let parsed = quantization
        .map(Qwen3AffineQuantization::from_config)
        .transpose()
        .map_err(Qwen3ForwardError::UnsupportedQuantization)?;
    match quantization_config {
        None | Some(serde_json::Value::Null) => Ok(parsed),
        Some(serde_json::Value::Object(declared))
            if parsed.is_some()
                && Qwen3AffineQuantization::from_config(declared).ok() == parsed =>
        {
            Ok(parsed)
        }
        Some(declared) => Err(Qwen3ForwardError::UnsupportedQuantization(
            declared.to_string(),
        )),
    }
}

/// transformers 5 `rope_parameters`. Any key besides these two (`factor`,
/// `partial_rotary_factor`, ...) changes the rotation, so it is kept to be
/// refused.
#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    #[serde(default)]
    rope_type: String,
    rope_theta: Option<f32>,
    #[serde(flatten)]
    other: serde_json::Map<String, serde_json::Value>,
}

const fn default_eps() -> f32 {
    1e-6
}

const fn default_rope_theta() -> f32 {
    1_000_000.0
}

/// Errors while parsing or binding a Qwen-local residual intervention.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3SteeringError {
    /// The artifact has no caller-supplied model identity for its receipt.
    #[error("Qwen3 residual steering requires a nonempty model identity")]
    MissingModelIdentity,
    /// The artifact does not declare a usable Qwen residual layout.
    #[error(
        "Qwen3 residual steering declares invalid dimensions {hidden_layers} layers x {hidden_size} hidden"
    )]
    InvalidDeclaredDimensions {
        /// Declared decoder layer count.
        hidden_layers: usize,
        /// Declared residual width.
        hidden_size: usize,
    },
    /// The selected range does not contain a token position.
    #[error("Qwen3 residual steering range [{start_inclusive}, {end_exclusive}) is empty")]
    EmptyPositionRange {
        /// First selected position.
        start_inclusive: usize,
        /// First excluded position.
        end_exclusive: usize,
    },
    /// The artifact coefficient cannot be evaluated safely.
    #[error("Qwen3 residual steering coefficient is non-finite: {0}")]
    NonFiniteCoefficient(f32),
    /// At least one residual vector entry cannot be evaluated safely.
    #[error("Qwen3 residual steering vector contains a non-finite value")]
    NonFiniteResidual,
    /// Applying the finite coefficient would overflow a finite residual entry.
    #[error("Qwen3 residual steering coefficient overflows a residual value")]
    NonFiniteScaledResidual,
    /// The vector width differs from the artifact's declared residual width.
    #[error("Qwen3 residual steering vector has width {actual}, expected {expected}")]
    ResidualDimensionMismatch {
        /// Actual vector width.
        actual: usize,
        /// Declared residual width.
        expected: usize,
    },
    /// The selected decoder layer is absent from the declared layout.
    #[error("Qwen3 residual steering layer {layer} is outside {hidden_layers} declared layers")]
    LayerOutOfDeclaredRange {
        /// Selected zero-based layer.
        layer: usize,
        /// Declared layer count.
        hidden_layers: usize,
    },
    /// The artifact's declared Qwen layout differs from the loaded model.
    #[error(
        "Qwen3 residual steering declares {declared_hidden_layers} layers x {declared_hidden_size} hidden, loaded model has {actual_hidden_layers} layers x {actual_hidden_size} hidden"
    )]
    ConfigurationDimensionMismatch {
        /// Artifact's declared decoder layer count.
        declared_hidden_layers: usize,
        /// Artifact's declared residual width.
        declared_hidden_size: usize,
        /// Loaded decoder layer count.
        actual_hidden_layers: usize,
        /// Loaded residual width.
        actual_hidden_size: usize,
    },
    /// A steering object was bound to a different Qwen configuration.
    #[error("Qwen3 residual steering is bound to a different model configuration")]
    BoundConfigurationMismatch,
    /// Changing an intervention after KV materialization would mix histories.
    #[error("Qwen3 residual steering can change only while the KV cache is empty")]
    SteeringRequiresEmptyCache,
}

/// Errors from Qwen3 configuration qualification or its Metal forward graph.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3ForwardError {
    /// A residual intervention was not bound to this exact Qwen configuration.
    #[error(transparent)]
    Steering(#[from] Qwen3SteeringError),
    /// The configuration was not JSON.
    #[error("invalid Qwen3 forward configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// The KV-cache executor only supports causal attention.
    #[error("the Qwen3 KV-cache executor requires causal attention")]
    CachedBidirectional,
    /// Right-padded batches are exact only under causal attention.
    #[error("batched Qwen3 forward requires causal attention")]
    BatchedBidirectional,

    /// Padded layer readout requires causal attention.
    #[error("padded Qwen3 layer states require causal attention")]
    PaddedBidirectional,

    /// The real-token prefix is empty or exceeds the supplied positions.
    #[error("real length {real_len} must be in 1..={positions}")]
    RealLength {
        /// Number of real tokens requested.
        real_len: usize,
        /// Total supplied positions, including padding.
        positions: usize,
    },

    /// Layer selections must be nonempty, strictly increasing and in range.
    #[error("layers {layers:?} must be increasing and within 1..={hidden_layers}")]
    LayerSelection {
        /// Requested one-based layer indices.
        layers: Vec<usize>,
        /// Available decoder layers.
        hidden_layers: usize,
    },
    /// The configuration selected a non-Qwen3 architecture, or its attention
    /// flag disagrees with its model type.
    #[error(
        "expected model_type qwen3, or bidirectional_pplx_qwen3 with use_bidirectional_attention, got {0:?}"
    )]
    UnsupportedModelType(String),
    /// The forward path needs a positive model dimension.
    #[error("Qwen3 configuration has no usable {0}")]
    MissingDimension(&'static str),
    /// Qwen3's rotary head width must be even.
    #[error("Qwen3 head_dim must be even, got {0}")]
    OddHeadDimension(usize),
    /// The qualification path keeps its scale calculation exact in f32.
    #[error("Qwen3 head_dim {0} exceeds the qualification forward limit")]
    HeadDimensionTooLarge(usize),
    /// The query heads cannot be divided among key/value heads.
    #[error("Qwen3 has {attention_heads} query heads and {key_value_heads} key/value heads")]
    InvalidGroupedQueryLayout {
        /// Query heads.
        attention_heads: usize,
        /// Key/value heads.
        key_value_heads: usize,
    },
    /// The normalization epsilon cannot define a stable norm.
    #[error("invalid rms_norm_eps {0}")]
    InvalidRmsNormEpsilon(f32),
    /// The `RoPE` base cannot define a rotary frequency table.
    #[error("invalid rope_theta {0}")]
    InvalidRopeTheta(f32),
    /// The adapter has no reference vectors for projection bias.
    #[error("Qwen3 attention or MLP bias is unsupported by the qualification forward path")]
    UnsupportedBiasLayout,
    /// The forward path implements Qwen3's SiLU-gated MLP only.
    #[error("Qwen3 hidden_act {0:?} is unsupported; expected silu")]
    UnsupportedActivation(String),
    /// A diagnostic that projects logits through the token embedding was
    /// given an untied checkpoint.
    #[error(
        "this path projects logits through the token embedding; the checkpoint has a separate lm_head"
    )]
    UntiedOutputEmbedding,
    /// A Llama configuration omitted `rope_theta`.
    #[error("Llama configuration must state rope_theta")]
    MissingRopeTheta,
    /// `rope_theta` and `rope_parameters.rope_theta` name different bases.
    #[error("rope_theta {rope_theta} disagrees with rope_parameters.rope_theta {rope_parameters}")]
    ConflictingRopeTheta {
        /// Top-level `rope_theta`.
        rope_theta: f32,
        /// `rope_parameters.rope_theta`.
        rope_parameters: f32,
    },
    /// The adapter has no reference vectors for a RoPE-scaling variant.
    #[error("Qwen3 rope_scaling requires a dedicated qualification path")]
    UnsupportedRopeScaling,
    /// The adapter has no reference vectors for sliding-window attention.
    #[error("Qwen3 sliding_window requires a dedicated qualification path")]
    UnsupportedSlidingWindow,
    /// The raw input must contain at least one token.
    #[error("Qwen3 forward requires at least one token")]
    EmptyInput,
    /// A scoring range must leave at least one prefix token and one scored
    /// token.
    #[error("cannot score from token {from} of a {tokens}-token sequence")]
    InvalidScoreRange {
        /// First scored index.
        from: usize,
        /// Sequence length.
        tokens: usize,
    },
    /// Hidden-state rows were requested where logits are read back.
    #[error("hidden-state rows are not logits")]
    HiddenRowsAreNotLogits,
    /// The uncached reference attention graph is deliberately bounded.
    #[error("Qwen3 reference prompt has {actual} tokens, maximum is {maximum}")]
    PromptTooLong {
        /// Provided token count.
        actual: usize,
        /// Supported token count.
        maximum: usize,
    },
    /// A selected decoder layer is outside this model's configured range.
    #[error("Qwen3 layer {layer} is outside {hidden_layers} configured layers")]
    LayerOutOfRange {
        /// Requested zero-based layer index.
        layer: usize,
        /// Number of configured decoder layers.
        hidden_layers: usize,
    },
    /// A selected-layer diagnostic did not receive one residual stream.
    #[error(
        "Qwen3 layer input shape {actual:?} must be [batch, sequence, {hidden_size}] with positive batch and sequence"
    )]
    InvalidLayerInputShape {
        /// Actual MLX array shape.
        actual: Vec<i32>,
        /// Required residual-stream width.
        hidden_size: usize,
    },
    /// Decode needs a preceding prefill on this executor.
    #[error("Qwen3 decode requires a populated KV cache")]
    DecodeWithoutPrefill,
    /// Retained for API compatibility; chunked appends are now accepted by
    /// [`Qwen3ForwardExecutor::extend_last_logits`].
    #[error("Qwen3 cached append requires exactly one token")]
    CachedAppendRequiresOneToken,
    /// The executor's layer cache no longer matches its model contract.
    #[error("Qwen3 layer KV cache is inconsistent with its configuration")]
    CacheInconsistent,
    /// Only an unsteered resident-chat prefix of the cached tokens can be
    /// detached.
    #[error("Qwen3 K/V snapshot requires an unsteered resident prefix of the cached tokens")]
    KvSnapshotUnsupported,
    /// The snapshot came from another checkpoint load or resident plan.
    #[error("Qwen3 K/V snapshot does not belong to this checkpoint load and resident plan")]
    KvSnapshotMismatch,
    /// A rollback must keep between one token and the whole cache.
    #[error("cannot truncate a {cached}-token Qwen3 cache to {requested} tokens")]
    TruncateOutOfRange {
        /// Positions the caller asked to keep.
        requested: usize,
        /// Positions in the cache.
        cached: usize,
    },
    /// A token does not fit the checkpoint vocabulary.
    #[error("token ID {token_id} is outside vocabulary size {vocab_size}")]
    InvalidTokenId {
        /// Raw token ID.
        token_id: i32,
        /// Checkpoint vocabulary size.
        vocab_size: usize,
    },
    /// A resident-chat request exceeds its configured or qualified limit.
    #[error("Qwen3 resident chat context {requested} exceeds maximum {maximum}")]
    ResidentChatContextLimit {
        /// Requested prompt-plus-generated token capacity.
        requested: usize,
        /// The model's `max_position_embeddings`.
        maximum: usize,
    },
    /// A resident-chat K/V estimate exceeds the caller's logical budget.
    #[error("Qwen3 resident chat K/V requires {required} bytes, above budget {maximum}")]
    ResidentChatKvBudget {
        /// Logical final K/V bytes required.
        required: u64,
        /// Caller-provided logical K/V ceiling.
        maximum: u64,
    },
    /// A required checkpoint tensor was absent.
    #[error("Qwen3 checkpoint is missing {0}")]
    MissingWeight(String),
    /// A dimension cannot be represented by MLX's i32 shape API.
    #[error("Qwen3 shape exceeds MLX's i32 dimension limit")]
    ShapeOverflow,
    /// A batched decode named one sequence in two rows.
    #[error("Qwen3 batched decode names sequence {0} more than once")]
    RepeatedBatchSequence(u64),
    /// The paged KV pool could not be sized.
    #[error("Qwen3 paged KV pool: {0}")]
    KvPoolConfig(#[from] engine::blocks::BlockConfigError),
    /// The paged KV block manager refused an operation; nothing changed.
    #[error("Qwen3 paged KV: {0}")]
    KvBlocks(#[from] engine::blocks::BlockError),
    /// The paged KV pool was built for different weights or another dtype.
    #[error("Qwen3 paged KV pool dtype {pool:?} differs from activation dtype {activation:?}")]
    KvPoolDtype {
        /// The pool's element type.
        pool: mlx_rs::Dtype,
        /// The K/V projection's element type.
        activation: mlx_rs::Dtype,
    },
    /// A weight dtype is not one of [`Qwen3FloatPrecision`]'s.
    #[error("Qwen3 weight dtype {0} is not a supported floating-point precision")]
    UnsupportedWeightDtype(String),
    /// A GPU pick rule or its logits were malformed.
    #[error("invalid Qwen3 GPU pick rule: {0}")]
    InvalidPickRule(&'static str),
    /// A config's `quantization` is not one uniform affine layout this path
    /// implements.
    #[error("unsupported Qwen3 quantization {0}")]
    UnsupportedQuantization(String),
    /// A weight tensor's dtype does not match how the config says it is
    /// stored, such as packed words under a dense layout.
    #[error("Qwen3 weight layout mismatch: {0}")]
    WeightLayoutMismatch(String),
    /// The model produced a NaN or infinite logit, so no greedy token exists.
    #[error("Qwen3 produced non-finite vocabulary logits")]
    NonFiniteLogits,
    /// An embedded prompt chunk does not fit this decoder: an unsupported
    /// span kind, misshapen rows, or a chunk not starting where the cache
    /// ends.
    #[error("Qwen3 embedded prompt at position {position}: {reason}")]
    EmbeddedSpan {
        /// The prompt position involved.
        position: usize,
        /// What is wrong.
        reason: &'static str,
    },
    /// MLX could not construct, evaluate, or copy the Metal graph.
    #[error("MLX Qwen3 forward failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mlx_rs::Array;
    use proptest::prelude::*;

    use crate::{DecoderFamily, GPU_TEST_LOCK};

    const F32: super::Qwen3FloatPrecision = super::Qwen3FloatPrecision::Float32;

    use super::{
        Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor, Qwen3ResidualSteering,
        Qwen3ResidualSteeringArtifact, Qwen3SteeringError, Qwen3SteeringPositionRange,
        Qwen3TokenPicks, decoder_states, forward_hidden_states, forward_last_hidden,
        forward_last_hidden_batch, forward_last_logits, forward_last_logits_with_residual_steering,
        forward_layer, forward_layer_states, project, read_last_logits, rms_norm, stepped_capacity,
        weight,
    };

    const QWEN3_06B: &str = r#"{
      "model_type":"qwen3",
      "num_hidden_layers":28,
      "hidden_size":1024,
      "intermediate_size":3072,
      "vocab_size":151936,
      "num_attention_heads":16,
      "num_key_value_heads":8,
      "head_dim":128,
      "max_position_embeddings":40960,
      "rms_norm_eps":0.000001,
      "rope_theta":1000000,
      "hidden_act":"silu",
      "tie_word_embeddings":true,
      "attention_bias":false,
      "mlp_bias":false,
      "sliding_window":null,
      "use_sliding_window":false
    }"#;

    #[test]
    fn resident_capacity_matches_exact_plans_and_no_fit() {
        let config = Qwen3ForwardConfig::parse(QWEN3_06B).unwrap();
        for precision in [F32, super::Qwen3FloatPrecision::BFloat16] {
            for tokens in [1, 128, 129, 512, 513, 4096, 40_960] {
                let cost = config
                    .resident_chat_plan(tokens, u64::MAX, precision)
                    .unwrap()
                    .planned_kv_bytes();
                let fit = config
                    .resident_chat_capacity(cost, precision)
                    .unwrap()
                    .unwrap()
                    .get();
                assert_eq!(fit, tokens);
                assert!(config.resident_chat_plan(fit, cost, precision).is_ok());
                let under = config.resident_chat_capacity(cost - 1, precision).unwrap();
                assert_eq!(
                    under.map(std::num::NonZeroUsize::get),
                    (tokens > 1).then_some(tokens - 1)
                );
                assert!(matches!(
                    config.resident_chat_plan(tokens, cost - 1, precision),
                    Err(Qwen3ForwardError::ResidentChatKvBudget { .. })
                ));
            }
            assert_eq!(
                config
                    .resident_chat_capacity(u64::MAX, precision)
                    .unwrap()
                    .unwrap()
                    .get(),
                40_960
            );
            assert!(
                config
                    .resident_chat_capacity(0, precision)
                    .unwrap()
                    .is_none()
            );
        }
        assert_eq!(
            config
                .resident_chat_capacity(1024 * 1024, super::Qwen3FloatPrecision::BFloat16)
                .unwrap()
                .unwrap()
                .get(),
            9
        );
        let mut overflow = config.clone();
        overflow.hidden_layers = usize::MAX;
        assert!(matches!(
            overflow.resident_chat_capacity(u64::MAX, F32),
            Err(Qwen3ForwardError::ShapeOverflow)
        ));
    }

    fn qwen3_4b_kv_layout() -> Qwen3ForwardConfig {
        let layout = QWEN3_06B
            .replace("\"num_hidden_layers\":28", "\"num_hidden_layers\":36")
            .replace("\"hidden_size\":1024", "\"hidden_size\":2560")
            .replace("\"intermediate_size\":3072", "\"intermediate_size\":9728")
            .replace("\"num_attention_heads\":16", "\"num_attention_heads\":32");
        Qwen3ForwardConfig::parse(&layout).expect("Qwen3-4B K/V layout")
    }

    fn independent_kv_bytes(config: &Qwen3ForwardConfig, context_tokens: usize) -> u64 {
        u64::try_from(config.hidden_layers).expect("test layer count fits u64")
            * 2
            * u64::try_from(config.key_value_heads).expect("test K/V head count fits u64")
            * u64::try_from(context_tokens).expect("test context fits u64")
            * u64::try_from(config.head_dim).expect("test head dimension fits u64")
            * u64::try_from(size_of::<f32>()).expect("f32 byte width fits u64")
    }

    fn resident_production_layouts() -> [Qwen3ForwardConfig; 2] {
        [
            Qwen3ForwardConfig::parse(QWEN3_06B).expect("Qwen3-0.6B layout"),
            qwen3_4b_kv_layout(),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        /// Stepped resident storage always admits the requested prefix without
        /// exceeding the configured ceiling and only uses known growth points.
        #[test]
        fn stepped_capacity_preserves_prefix_bounds(
            maximum in 1_usize..=4096,
            next in 1_usize..=4096,
        ) {
            let result = stepped_capacity(next, maximum);
            if next > maximum {
                let rejected = matches!(
                    result,
                    Err(Qwen3ForwardError::PromptTooLong { actual, maximum: limit })
                        if actual == next && limit == maximum
                );
                prop_assert!(rejected);
            } else {
                let capacity = result.expect("admitted prefix has a capacity");
                prop_assert!(capacity >= next);
                prop_assert!(capacity <= maximum);
                prop_assert!(
                    capacity == maximum
                        || capacity == 128
                        || (capacity >= 512 && capacity.is_power_of_two())
                );
            }
        }

        /// Above 512 tokens storage stays within twice the sequence, however
        /// large the context limit: a short chat under a 262K limit must not
        /// allocate K/V for 262K positions.
        #[test]
        fn stepped_capacity_is_geometric_under_long_limits(
            maximum in 513_usize..=1_048_576,
            next in 1_usize..=1_048_576,
        ) {
            prop_assume!(next <= maximum);
            let capacity = stepped_capacity(next, maximum).expect("admitted");
            prop_assert!(capacity >= next);
            prop_assert!(capacity <= maximum);
            prop_assert!(capacity <= next.max(256) * 2);
        }

        /// Every admitted production-layout context uses exactly the logical
        /// f32 K/V formula, including the exact one-byte budget boundary.
        #[test]
        fn resident_kv_admission_matches_production_layouts(
            context_tokens in 1_usize..=16_384,
        ) {
            for config in resident_production_layouts() {
                let required = independent_kv_bytes(&config, context_tokens);
                let exact = config
                    .resident_chat_plan(context_tokens, required, F32)
                    .expect("exact K/V budget admits a valid context");
                prop_assert_eq!(exact.maximum_context_tokens(), context_tokens);
                prop_assert_eq!(exact.planned_kv_bytes(), required);
                prop_assert!(matches!(
                    config.resident_chat_plan(context_tokens, required - 1, F32),
                    Err(Qwen3ForwardError::ResidentChatKvBudget {
                        required: actual_required,
                        maximum,
                    }) if actual_required == required && maximum == required - 1
                ), "one byte below the exact K/V requirement must fail admission");
            }
        }

        /// One additional valid token must increase the retained K/V estimate
        /// for both the 0.6B and 36-layer 4B K/V layouts.
        #[test]
        fn resident_kv_estimate_is_strictly_monotone_for_production_layouts(
            context_tokens in 1_usize..16_384,
        ) {
            for config in resident_production_layouts() {
                let current = config
                    .resident_chat_plan(context_tokens, u64::MAX, F32)
                    .expect("valid production-layout context");
                let next = config
                    .resident_chat_plan(context_tokens + 1, u64::MAX, F32)
                    .expect("next valid production-layout context");
                prop_assert_eq!(
                    next.planned_kv_bytes() - current.planned_kv_bytes(),
                    independent_kv_bytes(&config, 1),
                );
                prop_assert!(next.planned_kv_bytes() > current.planned_kv_bytes());
            }
        }
    }

    #[test]
    fn resident_production_layouts_reject_the_token_after_the_model_positions() {
        for config in resident_production_layouts() {
            let positions = config.max_position_embeddings;
            assert!(
                config.resident_chat_plan(positions, u64::MAX, F32).is_ok(),
                "the model's whole declared context is admitted"
            );
            assert!(matches!(
                config.resident_chat_plan(positions + 1, u64::MAX, F32),
                Err(Qwen3ForwardError::ResidentChatContextLimit {
                    requested,
                    maximum,
                }) if requested == positions + 1 && maximum == positions
            ));
        }
    }

    #[test]
    fn accepts_qwen3_06b_expanded_query_width() {
        // Qwen3-0.6B deliberately has 16 * 128 = 2048 query features while
        // its residual stream is only 1024 wide; `o_proj` maps it back.
        Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
    }

    #[test]
    fn detached_kv_plan_uses_gqa_not_query_head_width() {
        let config = Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
        // 28 layers * (K + V) * 8 KV heads * 3 positions * 128 values * f32.
        assert_eq!(config.cached_kv_bytes(3).unwrap(), 688_128);
    }

    #[test]
    fn quantization_config_must_repeat_the_mlx_layout() {
        let base = QWEN3_06B.trim_end().trim_end_matches('}');
        let parse = |extra: &str| Qwen3ForwardConfig::parse(&format!("{base},{extra}}}"));
        let layout = r#"{"group_size":64,"bits":4}"#;
        assert_eq!(
            parse(&format!(
                r#""quantization":{layout},"quantization_config":{layout}"#
            ))
            .expect("mlx-lm writes both")
            .quantization(),
            Some(super::Qwen3AffineQuantization::FOUR_BIT_G64)
        );
        assert!(
            parse(r#""quantization_config":null"#)
                .expect("null is absent")
                .quantization()
                .is_none()
        );
        for refused in [
            r#""quantization_config":{"quant_method":"fp8","weight_block_size":[128,128]}"#
                .to_owned(),
            format!(
                r#""quantization":{layout},"quantization_config":{{"group_size":32,"bits":4}}"#
            ),
            r#""quantization":{"group_size":64,"bits":3}"#.to_owned(),
        ] {
            assert!(matches!(
                parse(&refused),
                Err(Qwen3ForwardError::UnsupportedQuantization(_))
            ));
        }
    }

    #[test]
    fn bfloat16_plan_reserves_half_the_float32_kv() {
        let config = Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
        let float32 = config
            .resident_chat_plan(4_096, u64::MAX, F32)
            .expect("float32 plan");
        let bfloat16 = config
            .resident_chat_plan(4_096, u64::MAX, super::Qwen3FloatPrecision::BFloat16)
            .expect("bfloat16 plan");
        assert_eq!(bfloat16.planned_kv_bytes() * 2, float32.planned_kv_bytes());
        assert_eq!(
            bfloat16.kv_precision(),
            super::Qwen3FloatPrecision::BFloat16
        );
        // A budget that holds BF16 K/V but not f32 admits only the BF16 plan.
        assert!(
            config
                .resident_chat_plan(4_096, bfloat16.planned_kv_bytes(), F32)
                .is_err()
        );
    }

    #[test]
    fn plan_precision_follows_the_cache_the_weights_store() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = long_small_config();
        for precision in [
            super::Qwen3FloatPrecision::BFloat16,
            super::Qwen3FloatPrecision::Float16,
            super::Qwen3FloatPrecision::Float32,
        ] {
            let weights: HashMap<String, Array> = deterministic_weights()
                .into_iter()
                .map(|(name, tensor)| {
                    let converted = tensor.as_dtype(precision.dtype()).expect("cast");
                    (name, converted)
                })
                .collect();
            assert_eq!(
                super::kv_precision(&config, &weights).expect("float weights"),
                precision
            );
            let plan = config
                .resident_chat_plan(256, u64::MAX, precision)
                .expect("plan");
            let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
            // 128 positions fill the first storage tier exactly, so the stored
            // bytes equal the plan's bytes for 128 tokens.
            let _ = executor.prefill_last_logits(&[1; 128]).expect("prefill");
            assert_eq!(
                u64::try_from(executor.kv_bytes()).expect("bytes fit u64"),
                config
                    .cached_kv_bytes_at(128, plan.kv_precision())
                    .expect("bytes")
            );
        }
    }

    #[test]
    fn resident_chat_plan_is_separate_from_the_512_token_diagnostic_limit() {
        let config = Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
        let plan = config
            .resident_chat_plan(2_048, super::DEFAULT_RESIDENT_CHAT_KV_BUDGET_BYTES, F32)
            .expect("512 MiB admits Qwen3-0.6B K/V at 2048 tokens");
        assert_eq!(plan.maximum_context_tokens(), 2_048);
        assert_eq!(plan.planned_kv_bytes(), 469_762_048);
        assert!(
            config.resident_chat_plan(16_385, u64::MAX, F32).is_ok(),
            "no resident-chat cap below the model's 40,960 positions"
        );
        assert!(matches!(
            config.resident_chat_plan(40_961, u64::MAX, F32),
            Err(Qwen3ForwardError::ResidentChatContextLimit {
                requested: 40_961,
                maximum: 40_960,
            })
        ));
        assert!(matches!(
            config.resident_chat_plan(2_048, plan.planned_kv_bytes() - 1, F32),
            Err(Qwen3ForwardError::ResidentChatKvBudget {
                required: 469_762_048,
                maximum: 469_762_047,
            })
        ));
        let smaller_model = QWEN3_06B.replace(
            "\"max_position_embeddings\":40960",
            "\"max_position_embeddings\":1024",
        );
        let smaller_model = Qwen3ForwardConfig::parse(&smaller_model).expect("smaller context");
        assert!(matches!(
            smaller_model.resident_chat_plan(2_048, u64::MAX, F32),
            Err(Qwen3ForwardError::ResidentChatContextLimit {
                requested: 2_048,
                maximum: 1_024,
            })
        ));
    }

    #[test]
    fn default_executor_still_rejects_513_tokens_before_gpu_work() {
        let config = Qwen3ForwardConfig::parse(QWEN3_06B).expect("official Qwen3-0.6B layout");
        let weights = HashMap::new();
        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        assert!(matches!(
            executor.prefill_last_logits(&vec![0; 513]),
            Err(Qwen3ForwardError::PromptTooLong {
                actual: 513,
                maximum: 512,
            })
        ));
    }

    #[test]
    fn refuses_layouts_the_dense_path_cannot_implement() {
        let untied = QWEN3_06B.replace(
            "\"tie_word_embeddings\":true",
            "\"tie_word_embeddings\":false",
        );
        let untied = Qwen3ForwardConfig::parse(&untied).expect("untied projects through lm_head");
        assert!(!untied.tied_output_embedding());
        assert_eq!(untied.output_projection(), "lm_head");
        let rope = |parameters: &str| {
            Qwen3ForwardConfig::parse(&QWEN3_06B.replace(
                "\"rope_theta\":1000000,",
                &format!("\"rope_theta\":1000000,\"rope_parameters\":{parameters},"),
            ))
        };
        let parameters_only = QWEN3_06B.replace(
            "\"rope_theta\":1000000,",
            "\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":5000000},",
        );
        let supplied = Qwen3ForwardConfig::parse(&parameters_only).expect("default supplies theta");
        assert!((supplied.rope_theta - 5_000_000.0).abs() < f32::EPSILON);
        // pplx-embed states the same theta in both places.
        let same = rope(r#"{"rope_type":"default","rope_theta":1000000}"#).expect("same theta");
        assert!((same.rope_theta - 1_000_000.0).abs() < f32::EPSILON);
        assert!(matches!(
            rope(r#"{"rope_type":"default","rope_theta":10000}"#),
            Err(Qwen3ForwardError::ConflictingRopeTheta { .. })
        ));
        for scaled in [
            r#"{"rope_type":"linear","factor":8.0,"rope_theta":1000000}"#,
            r#"{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768,"rope_theta":1000000}"#,
            r#"{"rope_type":"default","factor":2.0,"rope_theta":1000000}"#,
            r#"{"rope_type":"default","partial_rotary_factor":0.5,"rope_theta":1000000}"#,
        ] {
            assert!(
                matches!(rope(scaled), Err(Qwen3ForwardError::UnsupportedRopeScaling)),
                "{scaled}"
            );
        }
        let llama = QWEN3_06B.replace("\"model_type\":\"qwen3\"", "\"model_type\":\"llama\"");
        assert_eq!(
            Qwen3ForwardConfig::parse(&llama).expect("llama").family(),
            DecoderFamily::Llama
        );
        assert!(matches!(
            Qwen3ForwardConfig::parse(&llama.replace("\"rope_theta\":1000000,", "")),
            Err(Qwen3ForwardError::MissingRopeTheta)
        ));
        assert!(matches!(
            Qwen3ForwardConfig::parse(
                &llama.replace("\"model_type\":\"llama\"", "\"model_type\":\"mistral\"")
            ),
            Err(Qwen3ForwardError::UnsupportedModelType(_))
        ));
        let activation = QWEN3_06B.replace("\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\"");
        assert!(matches!(
            Qwen3ForwardConfig::parse(&activation),
            Err(Qwen3ForwardError::UnsupportedActivation(_))
        ));
        let window = QWEN3_06B.replace(
            "\"use_sliding_window\":false",
            "\"use_sliding_window\":true",
        );
        assert!(matches!(
            Qwen3ForwardConfig::parse(&window),
            Err(Qwen3ForwardError::UnsupportedSlidingWindow)
        ));
    }

    /// Qwen/Qwen2.5-0.5B-Instruct@7ae5576 config.json, verbatim.
    const QWEN25_05B: &str = r#"{
      "architectures":["Qwen2ForCausalLM"],"attention_dropout":0.0,
      "bos_token_id":151643,"eos_token_id":151645,"hidden_act":"silu",
      "hidden_size":896,"initializer_range":0.02,"intermediate_size":4864,
      "max_position_embeddings":32768,"max_window_layers":21,"model_type":"qwen2",
      "num_attention_heads":14,"num_hidden_layers":24,"num_key_value_heads":2,
      "rms_norm_eps":1e-06,"rope_theta":1000000.0,"sliding_window":32768,
      "tie_word_embeddings":true,"torch_dtype":"bfloat16",
      "transformers_version":"4.43.1","use_cache":true,"use_sliding_window":false,
      "vocab_size":151936
    }"#;

    #[test]
    fn qwen2_parses_with_a_disabled_window_and_refuses_an_enabled_one() {
        let config = Qwen3ForwardConfig::parse(QWEN25_05B).expect("Qwen2.5-0.5B");
        assert_eq!(config.family(), DecoderFamily::Qwen2);
        assert_eq!(config.head_dim, 64);
        assert!(config.tied_output_embedding());
        let enabled = QWEN25_05B.replace(
            "\"use_sliding_window\":false",
            "\"use_sliding_window\":true",
        );
        assert!(matches!(
            Qwen3ForwardConfig::parse(&enabled),
            Err(Qwen3ForwardError::UnsupportedSlidingWindow)
        ));
        assert!(matches!(
            Qwen3ForwardConfig::parse(&QWEN25_05B.replace("\"rope_theta\":1000000.0,", "")),
            Err(Qwen3ForwardError::MissingRopeTheta)
        ));
        // Qwen3 follows the same switch; Llama has none, so any window is refused.
        let qwen3 = QWEN3_06B.replace("\"sliding_window\":null", "\"sliding_window\":4096");
        assert!(Qwen3ForwardConfig::parse(&qwen3).is_ok());
        let llama = qwen3.replace("\"model_type\":\"qwen3\"", "\"model_type\":\"llama\"");
        assert!(matches!(
            Qwen3ForwardConfig::parse(&llama),
            Err(Qwen3ForwardError::UnsupportedSlidingWindow)
        ));
        // Only the fixed Q/K/V bias: a declared MLP or attention bias is refused.
        for flag in ["mlp_bias", "attention_bias"] {
            let biased =
                QWEN25_05B.replace("\"hidden_act\"", &format!("\"{flag}\":true,\"hidden_act\""));
            assert!(
                matches!(
                    Qwen3ForwardConfig::parse(&biased),
                    Err(Qwen3ForwardError::UnsupportedBiasLayout)
                ),
                "{flag}"
            );
        }
    }

    #[test]
    fn qwen2_layout_reads_its_biases_and_caches_like_its_full_forward() {
        let _gpu = GPU_TEST_LOCK.lock().expect("GPU test lock");
        let config = Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen2",
              "num_hidden_layers":2,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "max_position_embeddings":16,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "sliding_window":16,
              "use_sliding_window":false,
              "tie_word_embeddings":true
            }"#,
        )
        .expect("small Qwen2 config");
        // hidden 4 over 2 heads: head_dim 2, so Q is 4 wide and K/V 2 wide.
        let mut weights = HashMap::new();
        insert_matrix(&mut weights, "model.embed_tokens.weight", 8, 4);
        insert_vector(&mut weights, "model.norm.weight", 4);
        for layer in 0..2 {
            let base = format!("model.layers.{layer}");
            insert_vector(&mut weights, &format!("{base}.input_layernorm.weight"), 4);
            insert_vector(
                &mut weights,
                &format!("{base}.post_attention_layernorm.weight"),
                4,
            );
            let attn = format!("{base}.self_attn");
            for (name, rows) in [("q_proj", 4), ("k_proj", 2), ("v_proj", 2)] {
                insert_matrix(&mut weights, &format!("{attn}.{name}.weight"), rows, 4);
                insert_vector(&mut weights, &format!("{attn}.{name}.bias"), rows);
            }
            insert_matrix(&mut weights, &format!("{attn}.o_proj.weight"), 4, 4);
            let mlp = format!("{base}.mlp");
            insert_matrix(&mut weights, &format!("{mlp}.gate_proj.weight"), 8, 4);
            insert_matrix(&mut weights, &format!("{mlp}.up_proj.weight"), 8, 4);
            insert_matrix(&mut weights, &format!("{mlp}.down_proj.weight"), 4, 8);
        }

        let prompt = [1, 2, 3];
        let full = forward_last_logits(&weights, &config, &prompt).expect("full forward");
        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        let _ = executor.prefill_last_logits(&prompt[..2]).expect("prefill");
        assert_logits_match(
            full.clone(),
            executor.decode_last_logits(3).expect("cached decode"),
        );

        // The biases reach the logits: zeroing one changes them.
        let mut unbiased = weights.clone();
        unbiased.insert(
            "model.layers.0.self_attn.v_proj.bias".to_owned(),
            Array::from_slice(&[0.0_f32, 0.0], &[2]),
        );
        let changed = forward_last_logits(&unbiased, &config, &prompt).expect("unbiased");
        assert!(
            full.iter().zip(&changed).any(|(a, b)| (a - b).abs() > 1e-4),
            "a V bias left the logits unchanged"
        );
    }

    #[test]
    fn cached_decode_matches_full_causal_forward_for_nonzero_weights() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3",
              "num_hidden_layers":1,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":16,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "tie_word_embeddings":true,
              "attention_bias":false,
              "mlp_bias":false
            }"#,
        )
        .expect("small dense Qwen3 config");
        let weights = deterministic_weights();
        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        let _ = executor.prefill_last_logits(&[1, 2]).expect("prefill");
        assert_eq!(executor.cached_tokens(), 2);
        assert!(executor.kv_bytes() > 0);
        for (next, prefix) in [
            (3, &[1, 2, 3][..]),
            (4, &[1, 2, 3, 4][..]),
            (5, &[1, 2, 3, 4, 5][..]),
        ] {
            let cached = executor.decode_last_logits(next).expect("cached decode");
            assert_eq!(executor.cached_tokens(), prefix.len());
            assert_logits_match(
                forward_last_logits(&weights, &config, prefix).expect("full forward"),
                cached,
            );
        }

        // A new prefill must discard all prior request-owned KV, including
        // arrays retained through a previous lazy cached-decode graph.
        let _ = executor
            .prefill_last_logits(&[5, 6])
            .expect("second prefill");
        assert_eq!(executor.cached_tokens(), 2);
        let cached = executor
            .decode_last_logits(7)
            .expect("second cached decode");
        assert_eq!(executor.cached_tokens(), 3);
        assert_logits_match(
            forward_last_logits(&weights, &config, &[5, 6, 7]).expect("second full forward"),
            cached,
        );

        // An invalid append fails closed rather than leaving the first layers'
        // KV available for a later request.
        assert!(matches!(
            executor.decode_last_logits(8),
            Err(Qwen3ForwardError::InvalidTokenId { .. })
        ));
        assert_eq!(executor.cached_tokens(), 0);
        assert_eq!(executor.kv_bytes(), 0);
        assert!(matches!(
            executor.decode_last_logits(1),
            Err(Qwen3ForwardError::DecodeWithoutPrefill)
        ));

        executor.reset();
        assert_eq!(executor.cached_tokens(), 0);
        assert_eq!(executor.kv_bytes(), 0);
    }

    fn host_argmax(logits: &[f32]) -> i32 {
        let mut best = 0;
        for (index, &value) in logits.iter().enumerate() {
            if value > logits[best] {
                best = index;
            }
        }
        i32::try_from(best).expect("small vocabulary")
    }

    #[test]
    fn gpu_greedy_breaks_ties_low_and_refuses_non_finite_rows() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let tied = Array::from_slice(
            &[1.0_f32, 3.0, -2.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[2, 5],
        );
        let picks = Qwen3TokenPicks::start(&tied, 0).expect("queued argmax");
        assert_eq!(picks.wait().expect("finite rows"), [1, 0]);
        assert!(matches!(
            picks.wait_one(),
            Err(Qwen3ForwardError::CacheInconsistent)
        ));
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let row = Array::from_slice(&[0.0_f32, 1.0, poison], &[1, 3]);
            let token = Qwen3TokenPicks::start(&row, 0).expect("queued argmax");
            assert!(matches!(
                token.wait(),
                Err(Qwen3ForwardError::NonFiniteLogits)
            ));
        }
    }

    #[test]
    fn truncating_restored_snapshot_storage_keeps_it_compact() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = long_small_config();
        let weights = deterministic_weights();
        let plan = config.resident_chat_plan(256, u64::MAX, F32).expect("plan");
        let prompt: Vec<i32> = (0..100).map(|index| index % 7 + 1).collect();
        let mut source = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        let _ = source.prefill_last_logits(&prompt).expect("prefill");
        let snapshot = source.snapshot_prefix(prompt.len(), 7).expect("snapshot");
        // Restored storage holds exactly 100 positions, below the 128 tier
        // that 90 tokens would occupy; truncation must not widen it.
        let mut restored = Qwen3ForwardExecutor::from_snapshot(&config, &weights, &snapshot);
        restored.truncate_cached_tokens(90).expect("truncate");
        assert_eq!(restored.cached_tokens(), 90);
        let continued = restored.decode_last_logits(5).expect("append re-tiers");

        let mut expected_prompt = prompt[..90].to_vec();
        expected_prompt.push(5);
        let mut fresh = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        let expected = fresh
            .prefill_last_logits(&expected_prompt)
            .expect("fresh prefill");
        assert_logits_match(expected, continued);
    }

    #[test]
    fn pipelined_gpu_greedy_matches_host_greedy_and_discards_the_stop_token() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = long_small_config();
        let weights = deterministic_weights();
        let prompt = [1, 2, 3];
        // Crosses the 128-token storage tier, so the rollback also has to
        // shrink storage back to the tier forks check.
        let steps = 140;

        let mut host = Qwen3ForwardExecutor::new_for_resident_chat(
            &config,
            &weights,
            config.resident_chat_plan(256, u64::MAX, F32).expect("plan"),
        );
        let mut logits = host.prefill_last_logits(&prompt).expect("host prefill");
        let mut expected = Vec::new();
        for _ in 0..steps {
            let token = host_argmax(&logits);
            expected.push(token);
            logits = host.decode_last_logits(token).expect("host decode");
        }

        let mut gpu = Qwen3ForwardExecutor::new_for_resident_chat(
            &config,
            &weights,
            config.resident_chat_plan(256, u64::MAX, F32).expect("plan"),
        );
        let first = host_argmax(&gpu.prefill_last_logits(&prompt).expect("gpu prefill"));
        let mut actual = vec![first];
        let mut pending = gpu.decode_greedy(first).expect("first greedy step");
        while actual.len() < steps {
            // Queue the next step before reading this one back.
            let next = gpu.decode_greedy_after(&pending).expect("pipelined step");
            actual.push(pending.wait_one().expect("finite logits"));
            pending = next;
        }
        assert_eq!(actual, expected);
        assert_eq!(gpu.cached_tokens(), prompt.len() + steps);
        // The last queued step appended a token the caller never kept.
        gpu.truncate_cached_tokens(gpu.cached_tokens() - 1)
            .expect("rollback");
        assert_eq!(gpu.cached_tokens(), host.cached_tokens() - 1);
        gpu.fork_prefilled()
            .expect("rolled-back storage stays consistent");
        let continued = gpu
            .decode_last_logits(*expected.last().expect("steps"))
            .expect("decode after rollback");
        assert_logits_match(logits, continued);
    }

    #[test]
    fn residual_steering_rejects_corrupt_or_mismatched_artifacts() {
        let config = small_dense_config();
        let mut different_config = config.clone();
        different_config.rope_theta = 123_456.0;
        let positions = Qwen3SteeringPositionRange::new(1, 3).expect("nonempty range");
        assert!(matches!(
            Qwen3SteeringPositionRange::new(3, 3),
            Err(Qwen3SteeringError::EmptyPositionRange { .. })
        ));
        assert!(matches!(
            Qwen3ResidualSteeringArtifact::new("", 1, 4, 0, vec![0.0; 4], 1.0, positions),
            Err(Qwen3SteeringError::MissingModelIdentity)
        ));
        assert!(matches!(
            Qwen3ResidualSteeringArtifact::new("tiny", 1, 4, 0, vec![0.0; 3], 1.0, positions),
            Err(Qwen3SteeringError::ResidualDimensionMismatch { .. })
        ));
        assert!(matches!(
            Qwen3ResidualSteeringArtifact::new("tiny", 1, 4, 1, vec![0.0; 4], 1.0, positions),
            Err(Qwen3SteeringError::LayerOutOfDeclaredRange { .. })
        ));
        assert!(matches!(
            Qwen3ResidualSteeringArtifact::new(
                "tiny",
                1,
                4,
                0,
                vec![0.0, f32::NAN, 0.0, 0.0],
                1.0,
                positions
            ),
            Err(Qwen3SteeringError::NonFiniteResidual)
        ));
        assert!(matches!(
            Qwen3ResidualSteeringArtifact::new(
                "tiny",
                1,
                4,
                0,
                vec![0.0; 4],
                f32::INFINITY,
                positions
            ),
            Err(Qwen3SteeringError::NonFiniteCoefficient(_))
        ));
        let wrong_layout =
            Qwen3ResidualSteeringArtifact::new("tiny", 2, 4, 0, vec![0.0; 4], 1.0, positions)
                .expect("well-formed but wrong declared layout");
        assert!(matches!(
            Qwen3ResidualSteering::bind(wrong_layout, &config),
            Err(Qwen3SteeringError::ConfigurationDimensionMismatch { .. })
        ));
        let overflowing = Qwen3ResidualSteeringArtifact::new(
            "tiny",
            1,
            4,
            0,
            vec![f32::MAX, 0.0, 0.0, 0.0],
            2.0,
            positions,
        )
        .expect("finite source artifact");
        assert!(matches!(
            Qwen3ResidualSteering::bind(overflowing, &config),
            Err(Qwen3SteeringError::NonFiniteScaledResidual)
        ));

        let bound_to_first = residual_steering(&config, 1.0, 0, 1);
        let weights = deterministic_weights();
        let mut other_executor = Qwen3ForwardExecutor::new(&different_config, &weights);
        assert!(matches!(
            other_executor.set_residual_steering(Some(bound_to_first.clone())),
            Err(Qwen3SteeringError::BoundConfigurationMismatch)
        ));
        assert!(matches!(
            forward_last_logits_with_residual_steering(
                &weights,
                &different_config,
                &[1],
                Some(&bound_to_first),
            ),
            Err(Qwen3ForwardError::Steering(
                Qwen3SteeringError::BoundConfigurationMismatch
            ))
        ));
    }

    #[test]
    fn residual_steering_disabled_zero_and_chunked_paths_are_qualified() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = small_dense_config();
        let weights = deterministic_weights();
        let prompt = [1_i32, 2, 3];
        let baseline = forward_last_logits(&weights, &config, &prompt).expect("baseline");
        assert_logits_match(
            baseline.clone(),
            forward_last_logits_with_residual_steering(&weights, &config, &prompt, None)
                .expect("disabled forward"),
        );

        let zero = residual_steering(&config, 0.0, 1, 3);
        assert_logits_match(
            baseline.clone(),
            forward_last_logits_with_residual_steering(&weights, &config, &prompt, Some(&zero))
                .expect("zero-coefficient forward"),
        );

        let steering = residual_steering(&config, 2.0, 1, 3);
        let steered =
            forward_last_logits_with_residual_steering(&weights, &config, &prompt, Some(&steering))
                .expect("steered forward");
        assert!(
            baseline
                .iter()
                .zip(&steered)
                .any(|(plain, altered)| (plain - altered).abs() > 5e-5),
            "nonzero residual should perturb this deterministic tiny model"
        );

        let mut chunked = Qwen3ForwardExecutor::new(&config, &weights);
        chunked
            .set_residual_steering(Some(steering.clone()))
            .expect("install before prefill");
        let _ = chunked
            .prefill_last_logits(&prompt[..2])
            .expect("chunk prefill");
        let chunked_logits = chunked.decode_last_logits(prompt[2]).expect("chunk decode");
        assert_logits_match(steered, chunked_logits);
    }

    #[test]
    fn residual_steering_fork_inherits_without_mutating_parent_state() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = small_dense_config();
        let weights = deterministic_weights();
        let steering = residual_steering(&config, 1.5, 0, 4);
        let mut parent = Qwen3ForwardExecutor::new(&config, &weights);
        parent
            .set_residual_steering(Some(steering.clone()))
            .expect("install before prefill");
        let _ = parent.prefill_last_logits(&[1, 2]).expect("parent prefill");
        let mut child = parent.fork_prefilled().expect("fork inherits steering");
        assert_eq!(child.residual_steering(), Some(&steering));
        assert!(matches!(
            child.set_residual_steering(None),
            Err(Qwen3SteeringError::SteeringRequiresEmptyCache)
        ));

        let child_logits = child.decode_last_logits(3).expect("child decode");
        assert_eq!(
            parent.cached_tokens(),
            2,
            "child must not mutate parent cache"
        );
        assert_logits_match(
            forward_last_logits_with_residual_steering(
                &weights,
                &config,
                &[1, 2, 3],
                Some(&steering),
            )
            .expect("child independent replay"),
            child_logits,
        );
        let parent_logits = parent.decode_last_logits(4).expect("parent decode");
        assert_logits_match(
            forward_last_logits_with_residual_steering(
                &weights,
                &config,
                &[1, 2, 4],
                Some(&steering),
            )
            .expect("parent independent replay"),
            parent_logits,
        );
    }

    mod cache_component_profile;
    mod capacity_cache_profile;
    mod causal_mask;
    mod chunked_prefill;
    mod decode_profile;
    mod embedded;
    mod hot_path_profile;
    mod paged_batch;
    mod paged_kv;
    mod particle_replay;
    mod prefix_extend;
    mod score;
    mod speculative_checkpoint;
    mod speculative_verify;
    mod steering_checkpoint;

    #[test]
    fn resident_chat_cached_decode_matches_fresh_prefill_beyond_512_tokens() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = long_small_config();
        let weights = deterministic_weights();
        let plan = config
            .resident_chat_plan(600, u64::MAX, F32)
            .expect("long tiny-model resident plan");
        let mut cached = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        let mut prefix = vec![1; 513];
        let _ = cached.prefill_last_logits(&prefix).expect("long prefill");
        assert_eq!(cached.maximum_context_tokens(), 600);
        assert_eq!(cached.resident_chat_plan(), Some(plan));
        assert_eq!(cached.planned_kv_bytes(), Some(plan.planned_kv_bytes()));

        for token in [2, 3, 4] {
            let cached_logits = cached.decode_last_logits(token).expect("cached decode");
            prefix.push(token);
            let mut fresh = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
            let fresh_logits = fresh
                .prefill_last_logits(&prefix)
                .expect("fresh long prefill");
            assert_logits_match(fresh_logits, cached_logits);
        }
    }

    #[test]
    fn resident_chat_context_error_resets_cached_kv() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = long_small_config();
        let weights = deterministic_weights();
        let plan = config
            .resident_chat_plan(513, u64::MAX, F32)
            .expect("exact long tiny-model resident plan");
        let mut executor = Qwen3ForwardExecutor::new_for_resident_chat(&config, &weights, plan);
        executor
            .prefill_last_logits(&vec![1; 513])
            .expect("exact-limit prefill");
        assert!(matches!(
            executor.decode_last_logits(2),
            Err(Qwen3ForwardError::PromptTooLong {
                actual: 514,
                maximum: 513,
            })
        ));
        assert_eq!(executor.cached_tokens(), 0);
        assert_eq!(executor.kv_bytes(), 0);
    }

    #[test]
    fn composed_nonzero_layer_matches_independent_cached_prefill() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3",
              "num_hidden_layers":2,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":16,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "tie_word_embeddings":true,
              "attention_bias":false,
              "mlp_bias":false
            }"#,
        )
        .expect("two-layer dense Qwen3 config");
        let weights = deterministic_weights_for_layers(2);
        let input_ids = [1_i32, 2, 3];
        let stream = mlx_rs::StreamOrDevice::gpu();
        let embedding = weight(&weights, "model.embed_tokens.weight").expect("embedding");
        let ids = Array::from_slice(&input_ids, &[3]);
        let hidden = embedding
            .take_axis_device(&ids, 0, &stream)
            .expect("embedding lookup")
            .reshape_device(&[1, 3, 4], &stream)
            .expect("residual shape");

        // The cached executor has its own attention and MLP implementation.
        // Its prefill result therefore checks the extracted nonzero layer's
        // contribution instead of comparing this helper to itself.
        let after_layer_zero = forward_layer(&config, &weights, 0, &hidden).expect("layer zero");
        let after_layer_one =
            forward_layer(&config, &weights, 1, &after_layer_zero).expect("nonzero layer");
        let last = after_layer_one
            .take_axis_device(Array::from_slice(&[2_i32], &[1]), 1, &stream)
            .expect("last hidden state");
        let normalized = rms_norm(
            &last,
            weight(&weights, "model.norm.weight").expect("final norm"),
            config.rms_norm_eps,
        )
        .expect("final norm graph");
        let composed = read_last_logits(
            &project(&config, &weights, &normalized, "model.embed_tokens")
                .expect("tied output projection"),
            1,
            8,
        )
        .expect("composed logits");

        let cached = Qwen3ForwardExecutor::new(&config, &weights)
            .prefill_last_logits(&input_ids)
            .expect("independent cached prefill");
        assert_logits_match(composed, cached);
    }

    fn two_layer_config() -> Qwen3ForwardConfig {
        Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3", "num_hidden_layers":2, "hidden_size":4,
              "intermediate_size":8, "vocab_size":8, "num_attention_heads":2,
              "num_key_value_heads":1, "head_dim":4, "max_position_embeddings":16,
              "rms_norm_eps":0.000001, "rope_theta":1000000, "hidden_act":"silu",
              "tie_word_embeddings":true, "attention_bias":false, "mlp_bias":false
            }"#,
        )
        .expect("two-layer dense Qwen3 config")
    }

    fn close(a: &[Vec<f32>], b: &[Vec<f32>]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(left, right)| left.len() == right.len())
            && a.iter()
                .flatten()
                .zip(b.iter().flatten())
                .all(|(x, y)| (x - y).abs() <= 1e-5 * (1.0 + y.abs()))
    }

    fn rows(state: &Array) -> Vec<Vec<f32>> {
        let state = state.as_type::<f32>().expect("f32");
        state.eval().expect("eval");
        state
            .as_slice::<f32>()
            .chunks(4)
            .map(<[f32]>::to_vec)
            .collect()
    }

    // Padding is masked as keys: a later padding row must not see an earlier
    // padding token. Without the mask, changing the token at the first
    // padding position changes every padding row after it.
    #[test]
    fn padded_layer_states_hide_padding_keys() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = two_layer_config();
        let weights = deterministic_weights_for_layers(2);
        let first = forward_layer_states(&weights, &config, &[1, 2, 3, 0, 0, 0], 3, &[1, 2])
            .expect("padded states");
        let changed = forward_layer_states(&weights, &config, &[1, 2, 3, 5, 0, 0], 3, &[1, 2])
            .expect("padded states");
        assert_eq!(first.len(), 2);
        for (a, b) in first.iter().zip(&changed) {
            let (a, b) = (rows(a), rows(b));
            assert!(close(&a[..3], &b[..3]), "real rows");
            assert!(!close(&a[3..4], &b[3..4]), "the changed row itself");
            assert!(close(&a[4..], &b[4..]), "later padding rows");
        }
        // Real rows equal the unpadded causal forward's.
        let unpadded = decoder_states(&weights, &config, &[1, 2, 3], None).expect("unpadded");
        assert!(close(&rows(&first[1])[..3], &rows(&unpadded)));
    }

    #[test]
    fn padded_layer_states_reject_bad_selections() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = two_layer_config();
        let weights = deterministic_weights_for_layers(2);
        for layers in [&[][..], &[0][..], &[2, 1][..], &[3][..]] {
            assert!(matches!(
                forward_layer_states(&weights, &config, &[1, 2], 2, layers),
                Err(Qwen3ForwardError::LayerSelection { .. })
            ));
        }
        assert!(matches!(
            forward_layer_states(&weights, &config, &[1, 2], 3, &[1]),
            Err(Qwen3ForwardError::RealLength { .. })
        ));
    }

    #[test]
    fn selected_layer_rejects_out_of_range_and_invalid_residual_shape() {
        let config = Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3", "num_hidden_layers":1, "hidden_size":4,
              "intermediate_size":8, "vocab_size":8, "num_attention_heads":2,
              "num_key_value_heads":1, "head_dim":4, "max_position_embeddings":16,
              "hidden_act":"silu", "tie_word_embeddings":true
            }"#,
        )
        .expect("small config");
        let weights = deterministic_weights();
        let bad_shape = Array::from_slice(&[1.0_f32; 8], &[2, 4]);
        assert!(matches!(
            forward_layer(&config, &weights, 1, &bad_shape),
            Err(Qwen3ForwardError::LayerOutOfRange { .. })
        ));
        assert!(matches!(
            forward_layer(&config, &weights, 0, &bad_shape),
            Err(Qwen3ForwardError::InvalidLayerInputShape { .. })
        ));
        let overlong = Array::from_slice(&[1.0_f32; 68], &[1, 17, 4]);
        assert!(matches!(
            forward_layer(&config, &weights, 0, &overlong),
            Err(Qwen3ForwardError::PromptTooLong {
                actual: 17,
                maximum: 16
            })
        ));
    }

    /// A Llama layer has no Q/K norms and projects through its own
    /// `lm_head`; cached decode must still match the full forward.
    #[test]
    fn llama_layout_skips_qk_norms_and_projects_through_lm_head() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"llama",
              "num_hidden_layers":2,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":16,
              "rms_norm_eps":0.000001,
              "rope_theta":5000000,
              "hidden_act":"silu",
              "tie_word_embeddings":false
            }"#,
        )
        .expect("small Llama config");
        let mut weights = deterministic_weights_for_layers(2);
        weights
            .retain(|name, _| !name.ends_with("q_norm.weight") && !name.ends_with("k_norm.weight"));
        // An output table unlike the embedding, so a tied projection fails.
        let lm_head: Vec<f32> = nonzero_values(32)
            .iter()
            .rev()
            .map(|value| -value)
            .collect();
        weights.insert(
            "lm_head.weight".to_owned(),
            Array::from_slice(&lm_head, &[8, 4]),
        );

        let prompt = [1, 2, 3];
        let hidden = forward_last_hidden(&weights, &config, &prompt).expect("hidden");
        let expected: Vec<f32> = lm_head
            .chunks_exact(4)
            .map(|row| row.iter().zip(&hidden).map(|(w, h)| w * h).sum())
            .collect();
        assert_logits_match(
            expected,
            forward_last_logits(&weights, &config, &prompt).expect("full forward"),
        );

        let mut executor = Qwen3ForwardExecutor::new(&config, &weights);
        let _ = executor.prefill_last_logits(&prompt[..2]).expect("prefill");
        assert_logits_match(
            forward_last_logits(&weights, &config, &prompt).expect("full forward"),
            executor.decode_last_logits(3).expect("cached decode"),
        );

        // The same weights under a Qwen3 type still require the norms.
        let qwen = config_for_family_check();
        assert!(matches!(
            forward_last_logits(&weights, &qwen, &prompt),
            Err(Qwen3ForwardError::MissingWeight(name)) if name.ends_with("q_norm.weight")
        ));
    }

    fn config_for_family_check() -> Qwen3ForwardConfig {
        Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3",
              "num_hidden_layers":2,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":16,
              "hidden_act":"silu",
              "tie_word_embeddings":false
            }"#,
        )
        .expect("small Qwen3 config")
    }

    fn assert_logits_match(full: Vec<f32>, cached: Vec<f32>) {
        assert_eq!(full.len(), cached.len());
        for (full, cached) in full.into_iter().zip(cached) {
            assert!(
                (full - cached).abs() <= 5e-5,
                "cached logit {cached} differs from full logit {full}"
            );
        }
    }

    fn long_small_config() -> Qwen3ForwardConfig {
        Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3",
              "num_hidden_layers":1,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":1024,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "tie_word_embeddings":true,
              "attention_bias":false,
              "mlp_bias":false
            }"#,
        )
        .expect("long tiny-model config")
    }

    fn small_dense_config() -> Qwen3ForwardConfig {
        Qwen3ForwardConfig::parse(
            r#"{
              "model_type":"qwen3",
              "num_hidden_layers":1,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":16,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "tie_word_embeddings":true,
              "attention_bias":false,
              "mlp_bias":false
            }"#,
        )
        .expect("small dense Qwen3 config")
    }

    fn residual_steering(
        config: &Qwen3ForwardConfig,
        coefficient: f32,
        start_inclusive: usize,
        end_exclusive: usize,
    ) -> Qwen3ResidualSteering {
        let artifact = Qwen3ResidualSteeringArtifact::new(
            "tiny-qwen-test",
            config.hidden_layers,
            config.hidden_size,
            0,
            vec![0.25, -0.5, 0.75, -1.0],
            coefficient,
            Qwen3SteeringPositionRange::new(start_inclusive, end_exclusive)
                .expect("test position range"),
        )
        .expect("test steering artifact");
        Qwen3ResidualSteering::bind(artifact, config).expect("bind test steering")
    }

    #[test]
    fn last_hidden_is_the_input_of_the_tied_output_projection() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = small_dense_config();
        let weights = deterministic_weights();
        let prompt = [1_i32, 2, 3];
        let hidden = forward_last_hidden(&weights, &config, &prompt).expect("last hidden");
        let logits = forward_last_logits(&weights, &config, &prompt).expect("last logits");
        assert_eq!(hidden.len(), 4);
        // The test embedding is `nonzero_values(32)` as an 8 x 4 matrix.
        let embedding = nonzero_values(32);
        for (row, &logit) in logits.iter().enumerate() {
            let projected: f32 = hidden
                .iter()
                .zip(&embedding[row * 4..row * 4 + 4])
                .map(|(h, e)| h * e)
                .sum();
            assert!(
                (projected - logit).abs() <= 1e-5 * (1.0 + logit.abs()),
                "row {row}: {projected} vs {logit}"
            );
        }
    }

    fn bidirectional_config() -> Qwen3ForwardConfig {
        let json = r#"{
              "model_type":"bidirectional_pplx_qwen3",
              "use_bidirectional_attention":true,
              "num_hidden_layers":1,
              "hidden_size":4,
              "intermediate_size":8,
              "vocab_size":8,
              "num_attention_heads":2,
              "num_key_value_heads":1,
              "head_dim":4,
              "max_position_embeddings":16,
              "rms_norm_eps":0.000001,
              "rope_theta":1000000,
              "hidden_act":"silu",
              "tie_word_embeddings":true,
              "attention_bias":false,
              "mlp_bias":false
            }"#;
        Qwen3ForwardConfig::parse(json).expect("bidirectional Qwen3 config")
    }

    #[test]
    fn bidirectional_attention_lets_early_positions_see_later_tokens() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let weights = deterministic_weights();
        let causal = small_dense_config();
        let bidirectional = bidirectional_config();
        assert_eq!(causal.attention(), crate::Qwen3Attention::Causal);
        assert_eq!(
            bidirectional.attention(),
            crate::Qwen3Attention::Bidirectional
        );
        let row = |config: &Qwen3ForwardConfig, ids: &[i32], position: usize| {
            forward_hidden_states(&weights, config, ids).expect("hidden states")
                [position * 4..position * 4 + 4]
                .to_vec()
        };
        // Changing only the last token moves position 0 under bidirectional
        // attention and leaves it untouched under causal attention.
        assert_eq!(row(&causal, &[1, 2, 3], 0), row(&causal, &[1, 2, 6], 0));
        assert_ne!(
            row(&bidirectional, &[1, 2, 3], 0),
            row(&bidirectional, &[1, 2, 6], 0)
        );
        // The last position attends to everything either way.
        for (left, right) in
            row(&causal, &[1, 2, 3], 2)
                .iter()
                .zip(row(&bidirectional, &[1, 2, 3], 2))
        {
            assert!((left - right).abs() <= 1e-6, "{left} vs {right}");
        }
    }

    #[test]
    fn per_token_states_end_with_the_last_hidden_state() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = small_dense_config();
        let weights = deterministic_weights();
        let all = forward_hidden_states(&weights, &config, &[1, 2, 3]).expect("all positions");
        assert_eq!(all.len(), 3 * 4);
        let last = forward_last_hidden(&weights, &config, &[1, 2, 3]).expect("last position");
        for (left, right) in all[8..].iter().zip(&last) {
            assert!((left - right).abs() <= 1e-6, "{left} vs {right}");
        }
    }

    #[test]
    fn batched_last_hidden_matches_each_sequence_run_alone() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = small_dense_config();
        let weights = deterministic_weights();
        // Different lengths, so the shorter rows are padded and must be read
        // at their own last token, not the batch's last position.
        let sequences: [&[i32]; 3] = [&[1, 2, 3], &[4], &[5, 6, 7, 1, 2]];
        let batched =
            forward_last_hidden_batch(&weights, &config, &sequences).expect("batched rows");
        assert_eq!(batched.len(), sequences.len());
        for (ids, row) in sequences.iter().zip(&batched) {
            let alone = forward_last_hidden(&weights, &config, ids).expect("single row");
            assert_eq!(row.len(), alone.len());
            for (left, right) in row.iter().zip(&alone) {
                assert!((left - right).abs() <= 1e-6, "{ids:?}: {left} vs {right}");
            }
        }
        assert!(matches!(
            forward_last_hidden_batch(&weights, &bidirectional_config(), &sequences),
            Err(Qwen3ForwardError::BatchedBidirectional)
        ));
        assert!(matches!(
            forward_last_hidden_batch(&weights, &config, &[]),
            Err(Qwen3ForwardError::EmptyInput)
        ));
        assert!(matches!(
            forward_last_hidden_batch(&weights, &config, &[&[1], &[]]),
            Err(Qwen3ForwardError::EmptyInput)
        ));
        assert!(matches!(
            forward_last_hidden_batch(&weights, &config, &[&[1], &[8]]),
            Err(Qwen3ForwardError::InvalidTokenId { token_id: 8, .. })
        ));
    }

    #[test]
    fn kv_cache_executor_refuses_bidirectional_attention() {
        let _gpu = GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = bidirectional_config();
        let weights = deterministic_weights();
        assert!(matches!(
            Qwen3ForwardExecutor::new(&config, &weights).prefill_last_logits(&[1, 2]),
            Err(Qwen3ForwardError::CachedBidirectional)
        ));
        assert!(matches!(
            config.resident_chat_plan(8, u64::MAX, F32),
            Err(Qwen3ForwardError::CachedBidirectional)
        ));
    }

    fn deterministic_weights() -> HashMap<String, Array> {
        deterministic_weights_for_layers(1)
    }

    fn deterministic_weights_for_layers(layers: usize) -> HashMap<String, Array> {
        let mut weights = HashMap::new();
        insert_matrix(&mut weights, "model.embed_tokens.weight", 8, 4);
        insert_vector(&mut weights, "model.norm.weight", 4);
        for layer in 0..layers {
            let base = format!("model.layers.{layer}");
            insert_vector(&mut weights, &format!("{base}.input_layernorm.weight"), 4);
            insert_vector(
                &mut weights,
                &format!("{base}.post_attention_layernorm.weight"),
                4,
            );
            let attn = format!("{base}.self_attn");
            insert_matrix(&mut weights, &format!("{attn}.q_proj.weight"), 8, 4);
            insert_matrix(&mut weights, &format!("{attn}.k_proj.weight"), 4, 4);
            insert_matrix(&mut weights, &format!("{attn}.v_proj.weight"), 4, 4);
            insert_matrix(&mut weights, &format!("{attn}.o_proj.weight"), 4, 8);
            insert_vector(&mut weights, &format!("{attn}.q_norm.weight"), 4);
            insert_vector(&mut weights, &format!("{attn}.k_norm.weight"), 4);
            let mlp = format!("{base}.mlp");
            insert_matrix(&mut weights, &format!("{mlp}.gate_proj.weight"), 8, 4);
            insert_matrix(&mut weights, &format!("{mlp}.up_proj.weight"), 8, 4);
            insert_matrix(&mut weights, &format!("{mlp}.down_proj.weight"), 4, 8);
        }
        weights
    }

    fn insert_matrix(weights: &mut HashMap<String, Array>, name: &str, rows: i32, columns: i32) {
        let count = usize::try_from(rows * columns).expect("small test shape");
        weights.insert(
            name.to_owned(),
            Array::from_slice(&nonzero_values(count), &[rows, columns]),
        );
    }

    fn insert_vector(weights: &mut HashMap<String, Array>, name: &str, length: i32) {
        weights.insert(
            name.to_owned(),
            Array::from_slice(
                &nonzero_values(usize::try_from(length).expect("small test shape")),
                &[length],
            ),
        );
    }

    fn nonzero_values(length: usize) -> Vec<f32> {
        (0..length)
            .map(|index| {
                (f32::from(u8::try_from(index % 11).expect("bounded test value")) + 1.0) * 0.017
            })
            .collect()
    }
}
