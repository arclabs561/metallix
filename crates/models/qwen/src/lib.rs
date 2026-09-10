//! Qwen3 text-model execution-contract parsing and validation.

use serde::Deserialize;
use thiserror::Error;

/// The validated text-execution contract extracted from a Qwen3 configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3TextContract {
    hidden_layers: u32,
    hidden_size: u32,
    attention_heads: u32,
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
        let config: RawConfig = serde_json::from_str(json).map_err(Qwen3ConfigError::Json)?;
        if config.model_type != "qwen3" {
            return Err(Qwen3ConfigError::UnexpectedModelType(config.model_type));
        }
        if config.num_hidden_layers == 0 {
            return Err(Qwen3ConfigError::MissingHiddenLayers);
        }
        if config.hidden_size == 0 {
            return Err(Qwen3ConfigError::MissingHiddenSize);
        }
        if config.num_attention_heads == 0 {
            return Err(Qwen3ConfigError::MissingAttentionHeads);
        }
        if config.max_position_embeddings == 0 {
            return Err(Qwen3ConfigError::MissingMaxPositionEmbeddings);
        }

        Ok(Self {
            hidden_layers: config.num_hidden_layers,
            hidden_size: config.hidden_size,
            attention_heads: config.num_attention_heads,
            max_position_embeddings: config.max_position_embeddings,
        })
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

    /// Returns the number of attention heads.
    #[must_use]
    pub const fn attention_heads(&self) -> u32 {
        self.attention_heads
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
    num_hidden_layers: u32,
    #[serde(default)]
    hidden_size: u32,
    #[serde(default)]
    num_attention_heads: u32,
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
    /// The configuration was not a Qwen3 text model.
    #[error("expected model_type qwen3, got {0:?}")]
    UnexpectedModelType(String),
    /// The configuration does not expose transformer layers.
    #[error("Qwen3 configuration has no usable transformer layers")]
    MissingHiddenLayers,
    /// The configuration does not expose a token representation width.
    #[error("Qwen3 configuration has no usable hidden size")]
    MissingHiddenSize,
    /// The configuration does not expose attention heads.
    #[error("Qwen3 configuration has no usable attention heads")]
    MissingAttentionHeads,
    /// The configuration does not expose a maximum sequence length.
    #[error("Qwen3 configuration has no usable maximum position embeddings")]
    MissingMaxPositionEmbeddings,
}

#[cfg(test)]
mod tests {
    use super::{Qwen3ConfigError, Qwen3TextContract};

    const CONFIG: &str = r#"{
      "model_type":"qwen3",
      "num_hidden_layers":28,
      "hidden_size":1024,
      "num_attention_heads":16,
      "max_position_embeddings":40960
    }"#;

    #[test]
    fn parses_a_qwen3_text_contract() {
        let contract = Qwen3TextContract::parse(CONFIG).expect("valid Qwen3 config");
        assert_eq!(contract.total_layers(), 28);
        assert_eq!(contract.hidden_size(), 1024);
        assert_eq!(contract.attention_heads(), 16);
        assert_eq!(contract.max_position_embeddings(), 40_960);
    }

    #[test]
    fn rejects_another_architecture() {
        let error = Qwen3TextContract::parse(r#"{"model_type":"llama"}"#).expect_err("not Qwen3");
        assert!(matches!(error, Qwen3ConfigError::UnexpectedModelType(_)));
    }

    #[test]
    fn rejects_missing_execution_dimensions() {
        let config = CONFIG.replace("\"num_attention_heads\":16", "\"num_attention_heads\":0");
        let error = Qwen3TextContract::parse(&config).expect_err("missing attention heads");
        assert!(matches!(error, Qwen3ConfigError::MissingAttentionHeads));
    }
}
