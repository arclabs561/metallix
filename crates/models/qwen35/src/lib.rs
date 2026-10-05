//! Text decoder for the Qwen3.5-family hybrid layout (`model_type` `qwen3_5`).
//!
//! Qwen3.5, Qwen3.6 and Qwen3.8 dense checkpoints share this layout: every
//! fourth decoder layer is gated full attention with a key/value cache, and
//! the others are `GatedDeltaNet` linear attention, whose per-sequence state is
//! a fixed-size matrix per value head plus a short-convolution window. The
//! layer kinds come from the configuration's `layer_types`.
//!
//! # `GatedDeltaNet` layer
//!
//! For one token with input `x` (after the layer's input norm), and `Hk` key
//! heads, `Hv` value heads, head widths `Dk`, `Dv` and convolution width `K`:
//!
//! 1. `qkv = W_qkv x` (width `2 Hk Dk + Hv Dv`), `z = W_z x` (`Hv Dv`),
//!    `b = W_b x` and `a = W_a x` (`Hv` each).
//! 2. Depthwise causal convolution of `qkv` over the last `K` positions, then
//!    `silu`. The previous `K - 1` inputs are the convolution state (zeros at the
//!    start of a sequence).
//! 3. Split into `q`, `k` (`Hk` heads) and `v` (`Hv` heads); L2-normalize `q`
//!    and `k` per head (`x / sqrt(sum x^2 + 1e-6)`), scale `q` by `Dk^-1/2`, and
//!    repeat each key head `Hv / Hk` times consecutively.
//! 4. `beta = sigmoid(b)` and the log-decay `g = -exp(A_log) softplus(a + dt_bias)`.
//! 5. Per value head, with state `S` of shape `[Dk, Dv]` (f32):
//!    `S = exp(g) S`, `S = S + k (beta (v - k^T S))^T`, output `o = q^T S`.
//! 6. `o` is RMS-normalized per head with a plain weight, multiplied by
//!    `silu(z)`, and projected back with `W_out`.
//!
//! Step 5 is the gated delta rule (arXiv 2412.06464). The source decode
//! applies it one token at a time. The source prefill computes the same
//! values in 64-token chunks: within a chunk it solves a unit lower-triangular
//! system (the UT transform) for the corrected values, then carries `S` from
//! chunk to chunk. This crate evaluates the token-by-token form for both
//! prefill and decode.
//!
//! # Sequence state
//!
//! Each linear layer keeps `S` for every value head (`[Hv, Dk, Dv]`, f32) and
//! the last `K - 1` convolution inputs (`[K - 1, 2 Hk Dk + Hv Dv]`); neither
//! grows with the sequence. Each full-attention layer keeps rotated keys and
//! values. A prompt shorter than `K - 1` leaves zeros in the older window
//! positions, as the source's left padding does.
//!
//! # Full-attention layer
//!
//! The query projection is twice as wide as the heads: per head, the first
//! `head_dim` values are the query and the rest are an output gate applied as
//! `sigmoid(gate)` to the attention output before `o_proj`. Queries and keys
//! get per-head RMS norms, then rotary embedding on the first
//! `head_dim * partial_rotary_factor` features only. The configuration's
//! multimodal rotary sections reduce to ordinary rotary embedding for text,
//! because all three position axes carry the same index.
//!
//! Every RMS norm except the `GatedDeltaNet` output norm scales by
//! `1 + weight`, computed in f32.

#[cfg(feature = "metal")]
pub mod forward;

use serde::Deserialize;
use thiserror::Error;

/// The token mixer of one decoder layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35LayerKind {
    /// `GatedDeltaNet` linear attention with recurrent and convolution state.
    LinearAttention,
    /// Gated full attention with a key/value cache.
    FullAttention,
}

/// The validated text-decoder configuration of a `qwen3_5` checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35Config {
    layers: Vec<Qwen35LayerKind>,
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    attention_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_theta: f32,
    rms_norm_eps: f32,
    linear_key_heads: usize,
    linear_value_heads: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    conv_kernel: usize,
    max_position_embeddings: usize,
    tie_word_embeddings: bool,
}

impl Qwen35Config {
    /// Parses a `qwen3_5` checkpoint configuration (with a nested
    /// `text_config`) or a bare `qwen3_5_text` configuration.
    ///
    /// Layout variants this decoder does not implement (mixture of experts,
    /// scaled rotary embedding, attention bias, an ungated attention output,
    /// another gate activation) are refused rather than ignored.
    pub fn parse(json: &str) -> Result<Self, Qwen35ConfigError> {
        let outer: RawOuterConfig = serde_json::from_str(json).map_err(Qwen35ConfigError::Json)?;
        let (text, outer_tie) = match (outer.model_type.as_str(), outer.text_config) {
            ("qwen3_5", Some(text)) => (text, outer.tie_word_embeddings),
            ("qwen3_5_text", None) => (
                serde_json::from_str::<RawTextConfig>(json).map_err(Qwen35ConfigError::Json)?,
                None,
            ),
            _ => return Err(Qwen35ConfigError::UnexpectedModelType(outer.model_type)),
        };
        Self::from_text(text, outer_tie)
    }

    fn from_text(raw: RawTextConfig, outer_tie: Option<bool>) -> Result<Self, Qwen35ConfigError> {
        if !matches!(raw.model_type.as_deref(), None | Some("qwen3_5_text")) {
            return Err(Qwen35ConfigError::UnexpectedModelType(
                raw.model_type.unwrap_or_default(),
            ));
        }
        check_supported(&raw)?;
        let rope = raw
            .rope_parameters
            .ok_or(Qwen35ConfigError::Missing("rope_parameters"))?;
        if rope.rope_type != "default" || rope.factor.is_some() {
            return Err(Qwen35ConfigError::Unsupported("scaled rotary embedding"));
        }
        let tie_word_embeddings = match (raw.tie_word_embeddings, outer_tie) {
            (Some(inner), Some(outer)) if inner != outer => {
                return Err(Qwen35ConfigError::ConflictingTiedEmbeddings);
            }
            (Some(tied), _) | (None, Some(tied)) => tied,
            (None, None) => return Err(Qwen35ConfigError::Missing("tie_word_embeddings")),
        };

        let fields = [
            ("num_hidden_layers", raw.num_hidden_layers),
            ("hidden_size", raw.hidden_size),
            ("intermediate_size", raw.intermediate_size),
            ("vocab_size", raw.vocab_size),
            ("num_attention_heads", raw.num_attention_heads),
            ("num_key_value_heads", raw.num_key_value_heads),
            ("head_dim", raw.head_dim),
            ("linear_num_key_heads", raw.linear_num_key_heads),
            ("linear_num_value_heads", raw.linear_num_value_heads),
            ("linear_key_head_dim", raw.linear_key_head_dim),
            ("linear_value_head_dim", raw.linear_value_head_dim),
            ("linear_conv_kernel_dim", raw.linear_conv_kernel_dim),
            ("max_position_embeddings", raw.max_position_embeddings),
        ];
        for (name, value) in fields {
            if value == 0 {
                return Err(Qwen35ConfigError::Missing(name));
            }
        }
        if !raw
            .num_attention_heads
            .is_multiple_of(raw.num_key_value_heads)
        {
            return Err(Qwen35ConfigError::HeadGrouping("num_attention_heads"));
        }
        if !raw
            .linear_num_value_heads
            .is_multiple_of(raw.linear_num_key_heads)
        {
            return Err(Qwen35ConfigError::HeadGrouping("linear_num_value_heads"));
        }
        let rotary_dim = rotary_dim(raw.head_dim, rope.partial_rotary_factor)?;
        if !rope.rope_theta.is_finite() || rope.rope_theta <= 0.0 {
            return Err(Qwen35ConfigError::Invalid("rope_theta"));
        }
        if !raw.rms_norm_eps.is_finite() || raw.rms_norm_eps <= 0.0 {
            return Err(Qwen35ConfigError::Invalid("rms_norm_eps"));
        }

        let layer_types = raw
            .layer_types
            .ok_or(Qwen35ConfigError::Missing("layer_types"))?;
        if layer_types.len() != raw.num_hidden_layers {
            return Err(Qwen35ConfigError::LayerCount {
                layer_types: layer_types.len(),
                num_hidden_layers: raw.num_hidden_layers,
            });
        }
        let layers = layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                "linear_attention" => Ok(Qwen35LayerKind::LinearAttention),
                "full_attention" => Ok(Qwen35LayerKind::FullAttention),
                _ => Err(Qwen35ConfigError::UnknownLayerType(kind.clone())),
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            layers,
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            vocab_size: raw.vocab_size,
            attention_heads: raw.num_attention_heads,
            key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rotary_dim,
            rope_theta: rope.rope_theta,
            rms_norm_eps: raw.rms_norm_eps,
            linear_key_heads: raw.linear_num_key_heads,
            linear_value_heads: raw.linear_num_value_heads,
            linear_key_head_dim: raw.linear_key_head_dim,
            linear_value_head_dim: raw.linear_value_head_dim,
            conv_kernel: raw.linear_conv_kernel_dim,
            max_position_embeddings: raw.max_position_embeddings,
            tie_word_embeddings,
        })
    }

    /// The token mixer of each decoder layer, in order.
    #[must_use]
    pub fn layers(&self) -> &[Qwen35LayerKind] {
        &self.layers
    }

    /// Residual-stream width.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Output vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Whether the output projection reuses the token embedding.
    #[must_use]
    pub const fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    /// Leading head features that receive rotary embedding.
    #[must_use]
    pub const fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    /// Maximum sequence length declared by the checkpoint.
    #[must_use]
    pub const fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings
    }

    /// Width of the convolved `q`, `k`, `v` projection of a linear layer.
    #[must_use]
    pub const fn conv_dim(&self) -> usize {
        2 * self.linear_key_heads * self.linear_key_head_dim
            + self.linear_value_heads * self.linear_value_head_dim
    }

    /// Logical bytes of all sequence state that does not grow with length:
    /// every linear layer's f32 recurrent matrices and convolution window.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35ConfigError::ShapeOverflow`] if the size overflows.
    pub fn fixed_state_bytes(&self) -> Result<u64, Qwen35ConfigError> {
        let recurrent = self
            .linear_value_heads
            .checked_mul(self.linear_key_head_dim)
            .and_then(|value| value.checked_mul(self.linear_value_head_dim));
        let conv = self.conv_dim().checked_mul(self.conv_kernel - 1);
        let linear_layers = self
            .layers
            .iter()
            .filter(|kind| **kind == Qwen35LayerKind::LinearAttention)
            .count();
        recurrent
            .zip(conv)
            .and_then(|(recurrent, conv)| recurrent.checked_add(conv))
            .and_then(|per_layer| per_layer.checked_mul(linear_layers))
            .and_then(|values| values.checked_mul(size_of::<f32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(Qwen35ConfigError::ShapeOverflow)
    }

    /// Logical f32 bytes of the full-attention key/value cache at `tokens`.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen35ConfigError::ShapeOverflow`] if the size overflows.
    pub fn kv_bytes(&self, tokens: usize) -> Result<u64, Qwen35ConfigError> {
        let full_layers = self
            .layers
            .iter()
            .filter(|kind| **kind == Qwen35LayerKind::FullAttention)
            .count();
        full_layers
            .checked_mul(2 * self.key_value_heads)
            .and_then(|value| value.checked_mul(self.head_dim))
            .and_then(|value| value.checked_mul(tokens))
            .and_then(|values| values.checked_mul(size_of::<f32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(Qwen35ConfigError::ShapeOverflow)
    }
}

/// Refuses layout variants the decoder does not implement.
fn check_supported(raw: &RawTextConfig) -> Result<(), Qwen35ConfigError> {
    let unsupported = |what: &'static str| Err(Qwen35ConfigError::Unsupported(what));
    if raw.num_experts.unwrap_or(0) != 0 {
        return unsupported("mixture-of-experts layers");
    }
    if raw
        .mlp_only_layers
        .as_ref()
        .is_some_and(|layers| !layers.is_empty())
    {
        return unsupported("mlp_only_layers");
    }
    if raw.attention_bias {
        return unsupported("attention bias");
    }
    if !raw.attn_output_gate {
        return unsupported("full attention without an output gate");
    }
    if raw.hidden_act != "silu" {
        return unsupported("an activation other than silu");
    }
    // SGLang passes this to the Gated DeltaNet output norm; "swish" is its
    // default SiLU gate, and Transformers does not read the key.
    if !matches!(
        raw.output_gate_type.as_deref(),
        None | Some("swish" | "silu")
    ) {
        return unsupported("a gated-norm activation other than swish");
    }
    if !matches!(raw.mamba_ssm_dtype.as_deref(), None | Some("float32")) {
        return unsupported("a recurrent state dtype other than float32");
    }
    Ok(())
}

fn rotary_dim(head_dim: usize, factor: f64) -> Result<usize, Qwen35ConfigError> {
    let invalid = Err(Qwen35ConfigError::Invalid("partial_rotary_factor"));
    if !(factor > 0.0 && factor <= 1.0) {
        return invalid;
    }
    // Transformers truncates `int(head_dim * factor)`; the result must be an
    // even feature count for rotate-half pairs.
    let Ok(head) = u32::try_from(head_dim) else {
        return invalid;
    };
    let scaled = (f64::from(head) * factor).floor();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scaled lies in (0, head_dim] after the checks above"
    )]
    let dim = scaled as usize;
    if dim == 0 || !dim.is_multiple_of(2) {
        return invalid;
    }
    Ok(dim)
}

#[derive(Debug, Deserialize)]
struct RawOuterConfig {
    #[serde(default)]
    model_type: String,
    text_config: Option<RawTextConfig>,
    tie_word_embeddings: Option<bool>,
}

#[allow(
    clippy::struct_excessive_bools,
    reason = "the fields directly mirror upstream Qwen JSON gates"
)]
#[derive(Debug, Deserialize)]
struct RawTextConfig {
    model_type: Option<String>,
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
    linear_num_key_heads: usize,
    #[serde(default)]
    linear_num_value_heads: usize,
    #[serde(default)]
    linear_key_head_dim: usize,
    #[serde(default)]
    linear_value_head_dim: usize,
    #[serde(default)]
    linear_conv_kernel_dim: usize,
    #[serde(default)]
    max_position_embeddings: usize,
    #[serde(default = "default_eps")]
    rms_norm_eps: f32,
    layer_types: Option<Vec<String>>,
    rope_parameters: Option<RawRope>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    attn_output_gate: bool,
    #[serde(default)]
    hidden_act: String,
    output_gate_type: Option<String>,
    mamba_ssm_dtype: Option<String>,
    tie_word_embeddings: Option<bool>,
    num_experts: Option<usize>,
    mlp_only_layers: Option<Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct RawRope {
    #[serde(default)]
    rope_type: String,
    #[serde(default)]
    rope_theta: f32,
    #[serde(default = "full_rotary")]
    partial_rotary_factor: f64,
    factor: Option<f64>,
}

const fn default_eps() -> f32 {
    1e-6
}

const fn full_rotary() -> f64 {
    1.0
}

/// An invalid or unsupported `qwen3_5` configuration.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen35ConfigError {
    /// The document was not valid JSON for this schema.
    #[error("invalid configuration JSON: {0}")]
    Json(serde_json::Error),
    /// The configuration is another architecture.
    #[error("expected model_type qwen3_5 with text_config, or qwen3_5_text, got {0:?}")]
    UnexpectedModelType(String),
    /// A layout variant this decoder does not implement.
    #[error("unsupported qwen3_5 layout: {0}")]
    Unsupported(&'static str),
    /// A required field is absent or zero.
    #[error("qwen3_5 configuration has no usable {0}")]
    Missing(&'static str),
    /// A field holds an unusable value.
    #[error("qwen3_5 configuration has an invalid {0}")]
    Invalid(&'static str),
    /// A head count does not divide evenly into its group.
    #[error("{0} is not a multiple of its key head count")]
    HeadGrouping(&'static str),
    /// `layer_types` does not name every layer.
    #[error("layer_types names {layer_types} layers, num_hidden_layers is {num_hidden_layers}")]
    LayerCount {
        /// Entries in `layer_types`.
        layer_types: usize,
        /// Declared decoder layer count.
        num_hidden_layers: usize,
    },
    /// A `layer_types` entry is not a known token mixer.
    #[error("unknown layer type {0:?}")]
    UnknownLayerType(String),
    /// The text and top-level configurations disagree on embedding tying.
    #[error("text_config and the top level disagree on tie_word_embeddings")]
    ConflictingTiedEmbeddings,
    /// A derived size does not fit in memory arithmetic.
    #[error("qwen3_5 state size overflows")]
    ShapeOverflow,
}

#[cfg(test)]
mod tests {
    use super::{Qwen35Config, Qwen35ConfigError, Qwen35LayerKind};

    /// The text fields of Qwen/Qwen3.8-27B@1d4bf0f2, with its 64-entry
    /// `layer_types` generated from the published 3-linear-then-1-full pattern.
    fn qwen38_27b() -> String {
        let layer_types = (0..64)
            .map(|layer| {
                if layer % 4 == 3 {
                    "\"full_attention\""
                } else {
                    "\"linear_attention\""
                }
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{
              "model_type": "qwen3_5",
              "tie_word_embeddings": false,
              "text_config": {{
                "model_type": "qwen3_5_text",
                "attention_bias": false,
                "attn_output_gate": true,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "layer_types": [{layer_types}],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144,
                "num_attention_heads": 24,
                "num_hidden_layers": 64,
                "num_key_value_heads": 4,
                "output_gate_type": "swish",
                "rms_norm_eps": 1e-06,
                "rope_parameters": {{
                  "mrope_interleaved": true,
                  "mrope_section": [11, 11, 10],
                  "partial_rotary_factor": 0.25,
                  "rope_theta": 10000000,
                  "rope_type": "default"
                }},
                "tie_word_embeddings": false,
                "vocab_size": 248320
              }}
            }}"#
        )
    }

    #[test]
    fn parses_the_qwen38_27b_layout() {
        let config = Qwen35Config::parse(&qwen38_27b()).expect("valid config");
        assert_eq!(config.layers().len(), 64);
        let full = config
            .layers()
            .iter()
            .filter(|kind| **kind == Qwen35LayerKind::FullAttention)
            .count();
        assert_eq!(full, 16);
        assert_eq!(config.layers()[3], Qwen35LayerKind::FullAttention);
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.conv_dim(), 2 * 16 * 128 + 48 * 128);
        assert!(!config.tie_word_embeddings());
        // 48 linear layers x (48 x 128 x 128 recurrent + 10240 x 3 conv) x 4 bytes.
        assert_eq!(
            config.fixed_state_bytes().expect("fits"),
            48 * (48 * 128 * 128 + 10_240 * 3) * 4
        );
        // 16 full layers x 2 x 4 KV heads x 256 x tokens x 4 bytes.
        assert_eq!(
            config.kv_bytes(1000).expect("fits"),
            16 * 2 * 4 * 256 * 1000 * 4
        );
    }

    #[test]
    fn refuses_layouts_it_does_not_implement() {
        let base = qwen38_27b();
        for (from, to) in [
            (
                r#""output_gate_type": "swish""#,
                r#""output_gate_type": "sigmoid""#,
            ),
            (
                r#""attn_output_gate": true"#,
                r#""attn_output_gate": false"#,
            ),
            (r#""rope_type": "default""#, r#""rope_type": "yarn""#),
            (r#""attention_bias": false"#, r#""attention_bias": true"#),
            (
                r#""linear_num_value_heads": 48"#,
                r#""linear_num_value_heads": 40"#,
            ),
            (
                r#""model_type": "qwen3_5","#,
                r#""model_type": "qwen3_5_moe","#,
            ),
        ] {
            assert!(base.contains(from), "fixture lacks {from}");
            assert!(
                Qwen35Config::parse(&base.replacen(from, to, 1)).is_err(),
                "accepted {to}"
            );
        }
    }

    #[test]
    fn requires_layer_types_to_cover_every_layer() {
        let config =
            qwen38_27b().replace(r#""num_hidden_layers": 64"#, r#""num_hidden_layers": 63"#);
        assert!(matches!(
            Qwen35Config::parse(&config),
            Err(Qwen35ConfigError::LayerCount {
                layer_types: 64,
                num_hidden_layers: 63
            })
        ));
        let config = qwen38_27b().replacen("\"full_attention\"", "\"sliding_attention\"", 1);
        assert!(matches!(
            Qwen35Config::parse(&config),
            Err(Qwen35ConfigError::UnknownLayerType(_))
        ));
    }

    #[test]
    fn rejects_conflicting_embedding_tying() {
        let config = qwen38_27b().replacen(
            r#""tie_word_embeddings": false,"#,
            r#""tie_word_embeddings": true,"#,
            1,
        );
        assert!(matches!(
            Qwen35Config::parse(&config),
            Err(Qwen35ConfigError::ConflictingTiedEmbeddings)
        ));
    }
}
