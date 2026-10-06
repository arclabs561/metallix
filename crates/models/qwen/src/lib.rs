//! Qwen3 text-model execution-contract parsing and validation.
//!
//! The same dense decoder also runs plain `llama` checkpoints (MiniCPM5-2B):
//! see [`DecoderFamily`].

pub mod checkpoint;
pub mod embedding;
#[cfg(feature = "metal")]
pub mod forward;
pub mod late;
#[cfg(feature = "metal")]
pub mod metal;
pub mod preflight;

// MLX's native test operations share process-global device initialization.
// Serialize GPU tests; pure config/header tests remain parallel.
// Tests take it through `PoisonError::into_inner`, so one failing GPU test
// reports as one failure instead of failing every later test on the lock.
#[cfg(all(test, feature = "metal"))]
pub(crate) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

use serde::Deserialize;
use thiserror::Error;

/// Dense decoder families this crate executes.
///
/// Both use pre-norm GQA blocks, a `SiLU`-gated MLP and rotate-half `RoPE`
/// with no embedding, residual or logit scaling (transformers
/// `modeling_llama.py` and `modeling_qwen3.py`). They differ only in Qwen3's
/// per-head RMS normalization of queries and keys before `RoPE`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecoderFamily {
    /// `qwen3` and `bidirectional_pplx_qwen3`: per-head Q/K RMS norms.
    Qwen3,
    /// `llama`: no Q/K norms.
    Llama,
}

impl DecoderFamily {
    /// Maps a configuration's `model_type`, or `None` for an unknown type.
    #[must_use]
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen3" | "bidirectional_pplx_qwen3" => Some(Self::Qwen3),
            "llama" => Some(Self::Llama),
            _ => None,
        }
    }

    /// Whether each layer carries `self_attn.q_norm` and `self_attn.k_norm`.
    #[must_use]
    pub const fn has_qk_norm(self) -> bool {
        matches!(self, Self::Qwen3)
    }
}

/// Which positions a Qwen3 decoder layer attends to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3Attention {
    /// Each position attends to itself and earlier positions (`qwen3`, `llama`).
    Causal,
    /// Each position attends to every position: `bidirectional_pplx_qwen3`
    /// with `use_bidirectional_attention`, as in pplx-embed.
    Bidirectional,
}

impl Qwen3Attention {
    /// Maps a configuration's model type and bidirectional flag to an attention
    /// layout, or `None` for an unknown type or a flag that contradicts it.
    #[must_use]
    pub fn from_config(model_type: &str, use_bidirectional_attention: bool) -> Option<Self> {
        match (model_type, use_bidirectional_attention) {
            ("qwen3" | "llama", false) => Some(Self::Causal),
            ("bidirectional_pplx_qwen3", true) => Some(Self::Bidirectional),
            _ => None,
        }
    }
}

/// The validated text-execution contract extracted from a Qwen3 configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3TextContract {
    family: DecoderFamily,
    hidden_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    attention_heads: u32,
    key_value_heads: u32,
    head_dim: u32,
    max_position_embeddings: u32,
}

impl Qwen3TextContract {
    /// Parses a Qwen3 configuration and rejects incomplete text layouts.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3ConfigError`] when the document is malformed, is not a
    /// Qwen3 configuration, or omits a positive execution dimension.
    pub fn parse(json: &str) -> Result<Self, Qwen3ConfigError> {
        let mut config: RawConfig = serde_json::from_str(json).map_err(Qwen3ConfigError::Json)?;
        let (Some(family), Some(_)) = (
            DecoderFamily::from_model_type(&config.model_type),
            Qwen3Attention::from_config(&config.model_type, config.use_bidirectional_attention),
        ) else {
            return Err(Qwen3ConfigError::UnexpectedModelType(config.model_type));
        };
        // Llama configurations may omit `head_dim`; transformers then uses
        // `hidden_size / num_attention_heads`. Qwen3 must state it.
        if family == DecoderFamily::Llama
            && config.head_dim == 0
            && config.num_attention_heads != 0
            && config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
        {
            config.head_dim = config.hidden_size / config.num_attention_heads;
        }
        if config.num_hidden_layers == 0 {
            return Err(Qwen3ConfigError::MissingHiddenLayers);
        }
        if config.hidden_size == 0 {
            return Err(Qwen3ConfigError::MissingHiddenSize);
        }
        if config.vocab_size == 0 {
            return Err(Qwen3ConfigError::MissingVocabSize);
        }
        if config.num_attention_heads == 0 {
            return Err(Qwen3ConfigError::MissingAttentionHeads);
        }
        if config.num_key_value_heads == 0 {
            return Err(Qwen3ConfigError::MissingKeyValueHeads);
        }
        if config.head_dim == 0 {
            return Err(Qwen3ConfigError::MissingHeadDimension);
        }
        if config.max_position_embeddings == 0 {
            return Err(Qwen3ConfigError::MissingMaxPositionEmbeddings);
        }

        Ok(Self {
            family,
            hidden_layers: config.num_hidden_layers,
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            attention_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            max_position_embeddings: config.max_position_embeddings,
        })
    }

    /// Returns the decoder family named by `model_type`.
    #[must_use]
    pub const fn family(&self) -> DecoderFamily {
        self.family
    }

    /// Returns the total number of transformer layers.
    #[must_use]
    pub const fn total_layers(&self) -> u32 {
        self.hidden_layers
    }

    /// Returns the token representation width.
    #[must_use]
    pub const fn hidden_size(&self) -> u32 {
        self.hidden_size
    }

    /// Returns the tokenizer vocabulary size used by the tied embeddings.
    #[must_use]
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Returns the number of attention heads.
    #[must_use]
    pub const fn attention_heads(&self) -> u32 {
        self.attention_heads
    }

    /// Returns the number of grouped-query key/value heads.
    #[must_use]
    pub const fn key_value_heads(&self) -> u32 {
        self.key_value_heads
    }

    /// Returns the configured attention-head representation width.
    #[must_use]
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Returns the maximum supported sequence length.
    #[must_use]
    pub const fn max_position_embeddings(&self) -> u32 {
        self.max_position_embeddings
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    use_bidirectional_attention: bool,
    #[serde(default)]
    num_hidden_layers: u32,
    #[serde(default)]
    hidden_size: u32,
    #[serde(default)]
    vocab_size: u32,
    #[serde(default)]
    num_attention_heads: u32,
    #[serde(default)]
    num_key_value_heads: u32,
    #[serde(default)]
    head_dim: u32,
    #[serde(default)]
    max_position_embeddings: u32,
}

/// An invalid or unsupported Qwen3 text-model layout.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3ConfigError {
    /// The supplied document was not JSON.
    #[error("invalid configuration JSON: {0}")]
    Json(serde_json::Error),
    /// The configuration was not a Qwen3 text model, or its attention flag
    /// disagrees with its model type.
    #[error(
        "expected model_type qwen3 or llama, or bidirectional_pplx_qwen3 with use_bidirectional_attention, got {0:?}"
    )]
    UnexpectedModelType(String),
    /// The configuration does not expose transformer layers.
    #[error("Qwen3 configuration has no usable transformer layers")]
    MissingHiddenLayers,
    /// The configuration does not expose a token representation width.
    #[error("Qwen3 configuration has no usable hidden size")]
    MissingHiddenSize,
    /// The configuration does not expose a token vocabulary size.
    #[error("Qwen3 configuration has no usable vocabulary size")]
    MissingVocabSize,
    /// The configuration does not expose attention heads.
    #[error("Qwen3 configuration has no usable attention heads")]
    MissingAttentionHeads,
    /// The configuration does not expose grouped-query key/value heads.
    #[error("Qwen3 configuration has no usable key/value attention heads")]
    MissingKeyValueHeads,
    /// The configuration does not expose an attention-head representation width.
    #[error("Qwen3 configuration has no usable attention head dimension")]
    MissingHeadDimension,
    /// The configuration does not expose a maximum sequence length.
    #[error("Qwen3 configuration has no usable maximum position embeddings")]
    MissingMaxPositionEmbeddings,
}

#[cfg(test)]
mod tests {
    use super::{DecoderFamily, Qwen3ConfigError, Qwen3TextContract};

    const CONFIG: &str = r#"{
      "model_type":"qwen3",
      "num_hidden_layers":28,
      "hidden_size":1024,
      "vocab_size":151936,
      "num_attention_heads":16,
      "num_key_value_heads":8,
      "head_dim":128,
      "max_position_embeddings":40960
    }"#;

    #[test]
    fn parses_a_qwen3_text_contract() {
        let contract = Qwen3TextContract::parse(CONFIG).expect("valid Qwen3 config");
        assert_eq!(contract.total_layers(), 28);
        assert_eq!(contract.hidden_size(), 1024);
        assert_eq!(contract.vocab_size(), 151_936);
        assert_eq!(contract.attention_heads(), 16);
        assert_eq!(contract.key_value_heads(), 8);
        assert_eq!(contract.head_dim(), 128);
        assert_eq!(contract.max_position_embeddings(), 40_960);
        assert_eq!(contract.family(), DecoderFamily::Qwen3);
        assert!(contract.family().has_qk_norm());
    }

    #[test]
    fn parses_a_llama_contract_and_defaults_its_head_dimension() {
        // MiniCPM5-2B's dimensions, with `head_dim` omitted as older Llama
        // configurations do.
        let llama = CONFIG
            .replace(r#""model_type":"qwen3""#, r#""model_type":"llama""#)
            .replace(r#""hidden_size":1024"#, r#""hidden_size":2048"#)
            .replace(r#""head_dim":128,"#, "");
        let contract = Qwen3TextContract::parse(&llama).expect("valid Llama config");
        assert_eq!(contract.family(), DecoderFamily::Llama);
        assert!(!contract.family().has_qk_norm());
        assert_eq!(contract.head_dim(), 128);
        // Qwen3 has no such default.
        let qwen = CONFIG.replace(r#""head_dim":128,"#, "");
        assert!(matches!(
            Qwen3TextContract::parse(&qwen),
            Err(Qwen3ConfigError::MissingHeadDimension)
        ));
    }

    #[test]
    fn maps_model_type_and_bidirectional_flag_to_attention() {
        use super::Qwen3Attention;
        assert_eq!(
            Qwen3Attention::from_config("qwen3", false),
            Some(Qwen3Attention::Causal)
        );
        assert_eq!(
            Qwen3Attention::from_config("llama", false),
            Some(Qwen3Attention::Causal)
        );
        assert_eq!(
            Qwen3Attention::from_config("bidirectional_pplx_qwen3", true),
            Some(Qwen3Attention::Bidirectional)
        );
        for (model_type, flag) in [
            ("qwen3", true),
            ("bidirectional_pplx_qwen3", false),
            ("llama", true),
            ("mistral", false),
        ] {
            assert_eq!(Qwen3Attention::from_config(model_type, flag), None);
        }
        let bidirectional = CONFIG.replace(
            r#""model_type":"qwen3""#,
            r#""model_type":"bidirectional_pplx_qwen3","use_bidirectional_attention":true"#,
        );
        assert!(Qwen3TextContract::parse(&bidirectional).is_ok());
        let contradictory = CONFIG.replace(
            r#""model_type":"qwen3""#,
            r#""model_type":"qwen3","use_bidirectional_attention":true"#,
        );
        assert!(matches!(
            Qwen3TextContract::parse(&contradictory),
            Err(Qwen3ConfigError::UnexpectedModelType(_))
        ));
    }

    #[test]
    fn rejects_another_architecture() {
        let error =
            Qwen3TextContract::parse(r#"{"model_type":"mistral"}"#).expect_err("unsupported");
        assert!(matches!(error, Qwen3ConfigError::UnexpectedModelType(_)));
    }

    #[test]
    fn rejects_missing_execution_dimensions() {
        let config = CONFIG.replace("\"num_attention_heads\":16", "\"num_attention_heads\":0");
        let error = Qwen3TextContract::parse(&config).expect_err("missing attention heads");
        assert!(matches!(error, Qwen3ConfigError::MissingAttentionHeads));
    }

    #[test]
    fn rejects_missing_key_value_heads() {
        let config = CONFIG.replace("\"num_key_value_heads\":8", "\"num_key_value_heads\":0");
        let error = Qwen3TextContract::parse(&config).expect_err("missing key/value heads");
        assert!(matches!(error, Qwen3ConfigError::MissingKeyValueHeads));
    }

    #[test]
    fn rejects_missing_head_dimension() {
        let config = CONFIG.replace("\"head_dim\":128", "\"head_dim\":0");
        let error = Qwen3TextContract::parse(&config).expect_err("missing head dimension");
        assert!(matches!(error, Qwen3ConfigError::MissingHeadDimension));
    }
}
