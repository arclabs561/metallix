//! Gemma 4 dense text-model contract, checkpoint validation and Metal forward.
//!
//! Gemma 4 interleaves sliding-window and full-attention decoder layers. This
//! crate implements the dense text path shared by `gemma4` (31B) and
//! `gemma4_unified` (12B) checkpoints and refuses the variants it does not
//! implement: per-layer input embeddings, K/V shared across layers, and the
//! mixture-of-experts block (the E2B/E4B and 26B-A4B models).
//!
//! Source: `modeling_gemma4.py` and `modeling_gemma4_unified.py` in
//! transformers v5.18.0. The numerics that differ from the Qwen3 adapter:
//!
//! - Token embeddings are multiplied by `sqrt(hidden_size)` rounded to the
//!   weight dtype (62.0 for 3840 in bf16, 61.97 in f32).
//! - `RMSNorm` multiplies by the stored weight `w`, not `1 + w` as Gemma 2/3 did.
//! - Each layer applies a norm to the attention output and to the MLP output
//!   before adding them to the residual, then multiplies the whole result,
//!   residual included, by the stored scalar `layer_scalar`.
//! - Q and K get a per-head `RMSNorm`; V gets one with no weight. Attention
//!   uses a softmax scale of 1, with no `1/sqrt(head_dim)`.
//! - Full-attention layers have a wider head (`global_head_dim`), their own
//!   K/V head count, and no `v_proj` when `attention_k_eq_v`: the raw K
//!   projection is also V, normalized separately and never rotated.
//! - Sliding layers use standard `RoPE`. Full layers use "proportional" `RoPE`:
//!   frequencies `theta^(2i / head_dim)` for the first
//!   `partial_rotary_factor * head_dim / 2` pairs and none for the rest, with
//!   pairs `(i, i + head_dim / 2)` across the whole head.
//! - A sliding position `q` sees keys `k` with `q - window < k <= q`.
//! - Final logits are soft-capped: `cap * tanh(logits / cap)`.
//! - The MLP uses tanh-approximated GELU.
//!
//! # Overview
//!
//! * [`Gemma4TextConfig::parse`] validates `config.json` and refuses the
//!   variants above. [`Gemma4GenerationConfig::parse`] reads the stop and
//!   suppressed tokens from `generation_config.json`.
//! * [`checkpoint::Gemma4CheckpointInspection::inspect`] checks every
//!   safetensors header against that configuration without loading a
//!   payload.
//! * With the `metal` feature, `metal::Gemma4MlxWeights::load` loads the text
//!   weights at a chosen precision, and its `executor` method starts a
//!   `forward::Gemma4Executor`, which owns one sequence's K/V cache and
//!   returns the logits after a prefill, a decoded token or an extension.
//!
//! The configuration and header checks need no GPU, so a checkpoint can be
//! qualified on any machine before Metal is involved.
//!
//! # Example: refusing an unimplemented variant
//!
//! ```
//! use gemma::{Gemma4ConfigError, Gemma4GenerationConfig, Gemma4TextConfig};
//!
//! let moe = r#"{"model_type": "gemma4_text", "enable_moe_block": true}"#;
//! assert!(matches!(
//!     Gemma4TextConfig::parse(moe),
//!     Err(Gemma4ConfigError::Unsupported("mixture-of-experts block")),
//! ));
//!
//! let generation = Gemma4GenerationConfig::parse(r#"{"eos_token_id": [1, 106]}"#)?;
//! assert_eq!(generation.eos_token_ids, [1, 106]);
//! # Ok::<(), Gemma4ConfigError>(())
//! ```

#![deny(missing_docs)]
// The workspace allows this lint; crates opt in once their docs are complete.
#![warn(clippy::missing_errors_doc)]

pub mod checkpoint;
#[cfg(feature = "metal")]
pub mod forward;
#[cfg(feature = "metal")]
pub mod metal;
pub mod routing;

// MLX's native test operations share process-global device initialization.
#[cfg(all(test, feature = "metal"))]
pub(crate) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

use serde::Deserialize;
use thiserror::Error;

/// Which attention a decoder layer uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gemma4LayerKind {
    /// Causal attention limited to the last `sliding_window` positions.
    Sliding,
    /// Causal attention over every earlier position.
    Full,
}

/// Rotary position embedding for one layer kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gemma4Rope {
    /// Standard `RoPE` over the whole head with base `theta`.
    Default {
        /// Base wavelength.
        theta: f32,
    },
    /// Rotates only the first `rotated_pairs` of `head_dim / 2` pairs, with
    /// exponents divided by the whole head dimension.
    Proportional {
        /// Base wavelength.
        theta: f32,
        /// Number of `(i, i + head_dim / 2)` pairs that rotate.
        rotated_pairs: usize,
    },
}

/// Attention geometry shared by every layer of one kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemma4Attention {
    /// Query heads.
    pub heads: usize,
    /// K/V heads; queries share them in equal groups.
    pub kv_heads: usize,
    /// Width of one head.
    pub head_dim: usize,
    /// The layer has no `v_proj`; V is the raw K projection.
    pub value_from_key: bool,
    /// Rotary embedding.
    pub rope: Gemma4Rope,
}

impl Gemma4Attention {
    /// `RoPE` wavelengths for MLX's `freqs` argument (one per pair; MLX uses
    /// their reciprocals), or `None` when standard `RoPE` with a base applies.
    /// Non-rotating pairs get an infinite wavelength, which MLX turns into a
    /// zero angle.
    #[must_use]
    pub fn rope_wavelengths(&self) -> Option<Vec<f32>> {
        let Gemma4Rope::Proportional {
            theta,
            rotated_pairs,
        } = self.rope
        else {
            return None;
        };
        let pairs = self.head_dim / 2;
        // Matches the source's float32 `base ** (arange(0, 2n, 2) / head_dim)`.
        #[allow(
            clippy::cast_precision_loss,
            reason = "head dimensions are validated to fit u16"
        )]
        let head_dim = self.head_dim as f32;
        Some(
            (0..pairs)
                .map(|pair| {
                    if pair < rotated_pairs {
                        #[allow(
                            clippy::cast_precision_loss,
                            reason = "pair indices are below u16::MAX"
                        )]
                        let exponent = (2 * pair) as f32 / head_dim;
                        theta.powf(exponent)
                    } else {
                        f32::INFINITY
                    }
                })
                .collect(),
        )
    }
}

/// The validated dense Gemma 4 text layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Gemma4TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    sliding_window: usize,
    final_logit_softcapping: Option<f32>,
    layers: Vec<Gemma4LayerKind>,
    sliding: Gemma4Attention,
    full: Gemma4Attention,
}

impl Gemma4TextConfig {
    /// Parses a checkpoint `config.json`: a multimodal `gemma4` or
    /// `gemma4_unified` document whose `text_config` is used, or a bare text
    /// configuration.
    ///
    /// # Errors
    ///
    /// * [`Gemma4ConfigError::Json`] for malformed JSON.
    /// * [`Gemma4ConfigError::UnexpectedModelType`] for another architecture.
    /// * [`Gemma4ConfigError::Unsupported`] for a Gemma 4 variant this crate
    ///   does not implement: per-layer input embeddings, K/V shared across
    ///   layers, the mixture-of-experts block, bidirectional text attention,
    ///   attention bias, another activation, an untied output embedding, or
    ///   scaled or unknown `RoPE`.
    /// * [`Gemma4ConfigError::MissingDimension`],
    ///   [`Gemma4ConfigError::InvalidValue`],
    ///   [`Gemma4ConfigError::InvalidLayerTypes`] and
    ///   [`Gemma4ConfigError::InvalidGroupedQueryLayout`] for a layout that
    ///   cannot be built.
    pub fn parse(json: &str) -> Result<Self, Gemma4ConfigError> {
        let document: RawDocument = serde_json::from_str(json).map_err(Gemma4ConfigError::Json)?;
        let raw = match document.text_config {
            Some(text) => {
                if !matches!(document.model_type.as_str(), "gemma4" | "gemma4_unified") {
                    return Err(Gemma4ConfigError::UnexpectedModelType(document.model_type));
                }
                text
            }
            None => serde_json::from_str(json).map_err(Gemma4ConfigError::Json)?,
        };
        Self::from_raw(raw)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one ordered list of upstream configuration gates"
    )]
    fn from_raw(raw: RawTextConfig) -> Result<Self, Gemma4ConfigError> {
        if !matches!(
            raw.model_type.as_str(),
            "gemma4_text" | "gemma4_unified_text"
        ) {
            return Err(Gemma4ConfigError::UnexpectedModelType(raw.model_type));
        }
        if raw.hidden_size_per_layer_input.unwrap_or(0) != 0 {
            return Err(Gemma4ConfigError::Unsupported("per-layer input embeddings"));
        }
        if raw.num_kv_shared_layers != 0 {
            return Err(Gemma4ConfigError::Unsupported("K/V shared across layers"));
        }
        if raw.enable_moe_block {
            return Err(Gemma4ConfigError::Unsupported("mixture-of-experts block"));
        }
        if raw.use_bidirectional_attention.as_deref() == Some("all") {
            return Err(Gemma4ConfigError::Unsupported(
                "bidirectional text attention",
            ));
        }
        if raw.attention_bias {
            return Err(Gemma4ConfigError::Unsupported("attention bias"));
        }
        if raw.hidden_activation != "gelu_pytorch_tanh" {
            return Err(Gemma4ConfigError::Unsupported(
                "activation other than gelu_pytorch_tanh",
            ));
        }
        if !raw.tie_word_embeddings {
            return Err(Gemma4ConfigError::Unsupported("untied output embedding"));
        }

        let fields = [
            ("num_hidden_layers", raw.num_hidden_layers),
            ("hidden_size", raw.hidden_size),
            ("intermediate_size", raw.intermediate_size),
            ("vocab_size", raw.vocab_size),
            ("num_attention_heads", raw.num_attention_heads),
            ("num_key_value_heads", raw.num_key_value_heads),
            ("head_dim", raw.head_dim),
            ("global_head_dim", raw.global_head_dim),
            ("max_position_embeddings", raw.max_position_embeddings),
            ("sliding_window", raw.sliding_window),
        ];
        for (name, value) in fields {
            if value == 0 {
                return Err(Gemma4ConfigError::MissingDimension(name));
            }
        }
        if !raw.rms_norm_eps.is_finite() || raw.rms_norm_eps <= 0.0 {
            return Err(Gemma4ConfigError::InvalidValue("rms_norm_eps"));
        }
        if raw
            .final_logit_softcapping
            .is_some_and(|cap| !cap.is_finite() || cap <= 0.0)
        {
            return Err(Gemma4ConfigError::InvalidValue("final_logit_softcapping"));
        }

        let layers = raw
            .layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                "sliding_attention" => Ok(Gemma4LayerKind::Sliding),
                "full_attention" => Ok(Gemma4LayerKind::Full),
                _ => Err(Gemma4ConfigError::InvalidLayerTypes),
            })
            .collect::<Result<Vec<_>, _>>()?;
        // The source forces the last layer to full attention when the list
        // says otherwise; refusing keeps this layout identical to the file.
        if layers.len() != raw.num_hidden_layers || layers.last() != Some(&Gemma4LayerKind::Full) {
            return Err(Gemma4ConfigError::InvalidLayerTypes);
        }

        let rope = raw
            .rope_parameters
            .ok_or(Gemma4ConfigError::InvalidValue("rope_parameters"))?;
        let sliding_rope = rope.sliding_attention.parse(raw.head_dim)?;
        let full_rope = rope.full_attention.parse(raw.global_head_dim)?;
        let full_kv_heads = if raw.attention_k_eq_v {
            raw.num_global_key_value_heads
                .unwrap_or(raw.num_key_value_heads)
        } else {
            raw.num_key_value_heads
        };
        let sliding = Gemma4Attention {
            heads: raw.num_attention_heads,
            kv_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            value_from_key: false,
            rope: sliding_rope,
        };
        let full = Gemma4Attention {
            heads: raw.num_attention_heads,
            kv_heads: full_kv_heads,
            head_dim: raw.global_head_dim,
            value_from_key: raw.attention_k_eq_v,
            rope: full_rope,
        };
        for attention in [sliding, full] {
            if attention.kv_heads == 0 || !attention.heads.is_multiple_of(attention.kv_heads) {
                return Err(Gemma4ConfigError::InvalidGroupedQueryLayout {
                    heads: attention.heads,
                    kv_heads: attention.kv_heads,
                });
            }
            if !attention.head_dim.is_multiple_of(2) || attention.head_dim > usize::from(u16::MAX) {
                return Err(Gemma4ConfigError::InvalidValue("head_dim"));
            }
        }

        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            vocab_size: raw.vocab_size,
            max_position_embeddings: raw.max_position_embeddings,
            rms_norm_eps: raw.rms_norm_eps,
            sliding_window: raw.sliding_window,
            final_logit_softcapping: raw.final_logit_softcapping,
            layers,
            sliding,
            full,
        })
    }

    /// Number of decoder layers.
    #[must_use]
    pub fn hidden_layers(&self) -> usize {
        self.layers.len()
    }

    /// Residual-stream width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// MLP inner width.
    #[must_use]
    pub const fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Rows of the tied embedding and output projection.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Maximum position the checkpoint declares.
    #[must_use]
    pub const fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    /// `RMSNorm` epsilon.
    #[must_use]
    pub const fn rms_norm_eps(&self) -> f32 {
        self.rms_norm_eps
    }

    /// Positions a sliding layer sees, its own included.
    #[must_use]
    pub const fn sliding_window(&self) -> usize {
        self.sliding_window
    }

    /// Final logit soft cap, when the checkpoint declares one.
    #[must_use]
    pub const fn final_logit_softcapping(&self) -> Option<f32> {
        self.final_logit_softcapping
    }

    /// Attention kind of every layer, in order.
    #[must_use]
    pub fn layers(&self) -> &[Gemma4LayerKind] {
        &self.layers
    }

    /// Geometry of one layer kind.
    #[must_use]
    pub const fn attention(&self, kind: Gemma4LayerKind) -> &Gemma4Attention {
        match kind {
            Gemma4LayerKind::Sliding => &self.sliding,
            Gemma4LayerKind::Full => &self.full,
        }
    }

    /// The embedding multiplier before rounding to the weight dtype.
    #[must_use]
    pub fn embed_scale(&self) -> f32 {
        #[allow(
            clippy::cast_precision_loss,
            reason = "hidden sizes are far below f32's exact-integer range"
        )]
        (self.hidden_size as f32).sqrt()
    }

    /// K/V elements retained after `tokens` positions: sliding layers keep at
    /// most `sliding_window - 1` earlier positions, full layers keep all.
    #[must_use]
    pub fn retained_kv_elements(&self, tokens: usize) -> u128 {
        self.layers
            .iter()
            .map(|&kind| {
                let attention = self.attention(kind);
                let positions = match kind {
                    Gemma4LayerKind::Sliding => tokens.min(self.sliding_window - 1),
                    Gemma4LayerKind::Full => tokens,
                };
                2 * attention.kv_heads as u128 * attention.head_dim as u128 * positions as u128
            })
            .sum()
    }
}

/// Stop and suppressed tokens from a checkpoint's `generation_config.json`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4GenerationConfig {
    /// Every token that ends a generated turn (`<eos>`, `<turn|>`, ...).
    pub eos_token_ids: Vec<i32>,
    /// Tokens the publisher never samples.
    pub suppress_token_ids: Vec<i32>,
}

impl Gemma4GenerationConfig {
    /// Parses `generation_config.json`, accepting one EOS ID or a list.
    ///
    /// # Errors
    ///
    /// Returns [`Gemma4ConfigError::Json`] for malformed JSON and
    /// [`Gemma4ConfigError::InvalidValue`] when there is no EOS token or one
    /// is negative.
    pub fn parse(json: &str) -> Result<Self, Gemma4ConfigError> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum OneOrMany {
            One(i32),
            Many(Vec<i32>),
        }
        #[derive(Deserialize)]
        struct Raw {
            eos_token_id: Option<OneOrMany>,
            #[serde(default)]
            suppress_tokens: Vec<i32>,
        }
        let raw: Raw = serde_json::from_str(json).map_err(Gemma4ConfigError::Json)?;
        let eos_token_ids = match raw.eos_token_id {
            Some(OneOrMany::One(id)) => vec![id],
            Some(OneOrMany::Many(ids)) => ids,
            None => Vec::new(),
        };
        if eos_token_ids.is_empty() || eos_token_ids.iter().any(|&id| id < 0) {
            return Err(Gemma4ConfigError::InvalidValue("eos_token_id"));
        }
        Ok(Self {
            eos_token_ids,
            suppress_token_ids: raw.suppress_tokens,
        })
    }
}

#[derive(Deserialize)]
struct RawDocument {
    #[serde(default)]
    model_type: String,
    text_config: Option<RawTextConfig>,
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "the fields directly mirror upstream Gemma 4 JSON gates"
)]
#[derive(Deserialize)]
struct RawTextConfig {
    #[serde(default)]
    model_type: String,
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
    num_global_key_value_heads: Option<usize>,
    #[serde(default)]
    head_dim: usize,
    #[serde(default)]
    global_head_dim: usize,
    #[serde(default)]
    max_position_embeddings: usize,
    #[serde(default)]
    sliding_window: usize,
    #[serde(default)]
    rms_norm_eps: f32,
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    layer_types: Vec<String>,
    rope_parameters: Option<RawRopeParameters>,
    #[serde(default)]
    attention_k_eq_v: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    hidden_activation: String,
    #[serde(default)]
    tie_word_embeddings: bool,
    hidden_size_per_layer_input: Option<usize>,
    #[serde(default)]
    num_kv_shared_layers: usize,
    #[serde(default)]
    enable_moe_block: bool,
    use_bidirectional_attention: Option<String>,
}

#[derive(Deserialize)]
struct RawRopeParameters {
    sliding_attention: RawRope,
    full_attention: RawRope,
}

#[derive(Deserialize)]
struct RawRope {
    rope_type: String,
    rope_theta: f32,
    partial_rotary_factor: Option<f64>,
    factor: Option<f64>,
}

impl RawRope {
    fn parse(&self, head_dim: usize) -> Result<Gemma4Rope, Gemma4ConfigError> {
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Gemma4ConfigError::InvalidValue("rope_theta"));
        }
        #[allow(
            clippy::float_cmp,
            reason = "any factor other than exactly 1 changes the frequencies"
        )]
        if self.factor.is_some_and(|factor| factor != 1.0) {
            return Err(Gemma4ConfigError::Unsupported("scaled RoPE"));
        }
        match (self.rope_type.as_str(), self.partial_rotary_factor) {
            ("default", None) => Ok(Gemma4Rope::Default {
                theta: self.rope_theta,
            }),
            ("proportional", factor) => {
                let factor = factor.unwrap_or(1.0);
                if !(factor > 0.0 && factor <= 1.0) {
                    return Err(Gemma4ConfigError::InvalidValue("partial_rotary_factor"));
                }
                // The source's `int(factor * head_dim // 2)`.
                #[allow(
                    clippy::cast_precision_loss,
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "head dimensions fit u16 and the factor is in (0, 1]"
                )]
                let rotated_pairs = (factor * head_dim as f64 / 2.0).floor() as usize;
                Ok(Gemma4Rope::Proportional {
                    theta: self.rope_theta,
                    rotated_pairs,
                })
            }
            _ => Err(Gemma4ConfigError::Unsupported("RoPE type")),
        }
    }
}

/// An invalid or unsupported Gemma 4 configuration.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Gemma4ConfigError {
    /// The document was not JSON of the expected shape.
    #[error("invalid configuration JSON: {0}")]
    Json(serde_json::Error),
    /// Not a Gemma 4 text configuration.
    #[error("expected a gemma4 or gemma4_unified configuration, got model_type {0:?}")]
    UnexpectedModelType(String),
    /// A Gemma 4 variant this adapter does not implement.
    #[error("Gemma 4 {0} is not implemented by this adapter")]
    Unsupported(&'static str),
    /// A required dimension is absent or zero.
    #[error("Gemma 4 configuration has no usable {0}")]
    MissingDimension(&'static str),
    /// A value is out of range.
    #[error("Gemma 4 configuration has an invalid {0}")]
    InvalidValue(&'static str),
    /// `layer_types` is not one known kind per layer ending in full attention.
    #[error("Gemma 4 layer_types must list one known kind per layer and end with full_attention")]
    InvalidLayerTypes,
    /// Query heads are not an exact multiple of K/V heads.
    #[error("{heads} query heads cannot share {kv_heads} K/V heads")]
    InvalidGroupedQueryLayout {
        /// Query heads.
        heads: usize,
        /// K/V heads.
        kv_heads: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        Gemma4ConfigError, Gemma4GenerationConfig, Gemma4LayerKind, Gemma4Rope, Gemma4TextConfig,
    };

    /// The text configuration of google/gemma-4-12B-it@707f0a3, cut from 48
    /// to 12 layers so tests stay small.
    pub(crate) const GEMMA4_12B_CONFIG: &str = r#"{
      "architectures": ["Gemma4UnifiedForConditionalGeneration"],
      "model_type": "gemma4_unified",
      "text_config": {
        "attention_bias": false, "attention_k_eq_v": true, "enable_moe_block": false,
        "final_logit_softcapping": 30.0, "global_head_dim": 512, "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 3840,
        "hidden_size_per_layer_input": 0, "intermediate_size": 15360,
        "layer_types": ["sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention",
          "sliding_attention","sliding_attention","full_attention"],
        "max_position_embeddings": 262144, "model_type": "gemma4_unified_text",
        "num_attention_heads": 16, "num_global_key_value_heads": 1,
        "num_hidden_layers": 12, "num_key_value_heads": 8, "num_kv_shared_layers": 0,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
          "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
        },
        "sliding_window": 1024, "tie_word_embeddings": true,
        "use_bidirectional_attention": "vision", "vocab_size": 262144
      }
    }"#;

    #[test]
    fn parses_the_interleaved_dense_layout() {
        let config = Gemma4TextConfig::parse(GEMMA4_12B_CONFIG).expect("12B layout");
        assert_eq!(config.hidden_layers(), 12);
        assert_eq!(config.sliding_window(), 1024);
        assert_eq!(config.final_logit_softcapping(), Some(30.0));
        assert_eq!(config.layers()[5], Gemma4LayerKind::Full);
        assert_eq!(config.layers()[4], Gemma4LayerKind::Sliding);

        let sliding = config.attention(Gemma4LayerKind::Sliding);
        assert_eq!(
            (sliding.heads, sliding.kv_heads, sliding.head_dim),
            (16, 8, 256)
        );
        assert!(!sliding.value_from_key);
        assert_eq!(sliding.rope, Gemma4Rope::Default { theta: 10_000.0 });
        assert_eq!(sliding.rope_wavelengths(), None);

        let full = config.attention(Gemma4LayerKind::Full);
        assert_eq!((full.heads, full.kv_heads, full.head_dim), (16, 1, 512));
        assert!(full.value_from_key);
        assert_eq!(
            full.rope,
            Gemma4Rope::Proportional {
                theta: 1_000_000.0,
                rotated_pairs: 64
            }
        );
    }

    #[test]
    fn proportional_rope_divides_by_the_whole_head_and_stops_after_the_rotated_pairs() {
        let config = Gemma4TextConfig::parse(GEMMA4_12B_CONFIG).expect("12B layout");
        let wavelengths = config
            .attention(Gemma4LayerKind::Full)
            .rope_wavelengths()
            .expect("proportional");
        assert_eq!(wavelengths.len(), 256);
        assert_eq!(wavelengths[0], 1.0);
        // Pair 1 rotates with exponent 2/512, not the 2/128 of a 128-wide RoPE.
        let expected = 1_000_000_f32.powf(2.0 / 512.0);
        assert!((wavelengths[1] - expected).abs() <= expected * 1e-6);
        assert!(wavelengths[63].is_finite());
        assert!(wavelengths[64..].iter().all(|value| value.is_infinite()));
    }

    #[test]
    fn retained_kv_caps_sliding_layers_at_the_window() {
        let config = Gemma4TextConfig::parse(GEMMA4_12B_CONFIG).expect("12B layout");
        // 10 sliding layers of 8x256 K and V, 2 full layers of 1x512 K and V.
        let per_sliding = 2 * 8 * 256;
        let per_full = 2 * 512;
        assert_eq!(
            config.retained_kv_elements(100),
            (10 * per_sliding + 2 * per_full) * 100
        );
        assert_eq!(
            config.retained_kv_elements(5000),
            10 * per_sliding * 1023 + 2 * per_full * 5000
        );
    }

    #[test]
    fn refuses_variants_without_a_dense_text_path() {
        for (from, to, expected) in [
            (
                r#""hidden_size_per_layer_input": 0"#,
                r#""hidden_size_per_layer_input": 256"#,
                "per-layer input embeddings",
            ),
            (
                r#""num_kv_shared_layers": 0"#,
                r#""num_kv_shared_layers": 18"#,
                "K/V shared across layers",
            ),
            (
                r#""enable_moe_block": false"#,
                r#""enable_moe_block": true"#,
                "mixture-of-experts block",
            ),
        ] {
            let config = GEMMA4_12B_CONFIG.replace(from, to);
            match Gemma4TextConfig::parse(&config) {
                Err(Gemma4ConfigError::Unsupported(what)) => assert_eq!(what, expected),
                other => panic!("expected {expected} to be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn refuses_another_architecture_and_a_trailing_sliding_layer() {
        let other = GEMMA4_12B_CONFIG.replace(
            r#""model_type": "gemma4_unified","#,
            r#""model_type": "gemma3","#,
        );
        assert!(matches!(
            Gemma4TextConfig::parse(&other),
            Err(Gemma4ConfigError::UnexpectedModelType(_))
        ));
        let trailing = GEMMA4_12B_CONFIG.replace(
            r#""sliding_attention","full_attention"],"#,
            r#""full_attention","sliding_attention"],"#,
        );
        assert!(matches!(
            Gemma4TextConfig::parse(&trailing),
            Err(Gemma4ConfigError::InvalidLayerTypes)
        ));
    }

    #[test]
    fn generation_config_keeps_every_stop_token() {
        let config = Gemma4GenerationConfig::parse(
            r#"{"eos_token_id": [1, 106, 50], "suppress_tokens": [258883, 258882]}"#,
        )
        .expect("generation config");
        assert_eq!(config.eos_token_ids, [1, 106, 50]);
        assert_eq!(config.suppress_token_ids, [258_883, 258_882]);
        assert_eq!(
            Gemma4GenerationConfig::parse(r#"{"eos_token_id": 1}"#)
                .expect("single")
                .eos_token_ids,
            [1]
        );
        assert!(Gemma4GenerationConfig::parse("{}").is_err());
    }
}
