//! DeepSeek-V4.1 execution-contract parsing and validation.

use serde::Deserialize;
use thiserror::Error;

/// The validated text-execution contract extracted from a V4.1 configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41TextContract {
    encoder_layers: u16,
    decoder_layers: u16,
    local_experts: u16,
    experts_per_token: u16,
    engram_max_ngram_size: u8,
}

impl V41TextContract {
    /// Parses a V4.1 configuration and rejects layouts Metallix does not support.
    ///
    /// # Errors
    ///
    /// Returns [`V41ConfigError`] when the document is malformed or is not the
    /// expected V4.1 CED text layout.
    pub fn parse(json: &str) -> Result<Self, V41ConfigError> {
        let config: RawConfig = serde_json::from_str(json).map_err(V41ConfigError::Json)?;
        let text = config.text_config.as_deref().unwrap_or(&config);

        if text.model_type != "deepseek_v41_text" {
            return Err(V41ConfigError::UnexpectedModelType(text.model_type.clone()));
        }
        if text.encoder_layers == 0 || text.decoder_layers == 0 {
            return Err(V41ConfigError::MissingCedSplit);
        }
        if text.local_experts == 0 || text.experts_per_token == 0 {
            return Err(V41ConfigError::MissingMoERouting);
        }
        if text.engram_max_ngram_size == 0 {
            return Err(V41ConfigError::MissingEngram);
        }

        Ok(Self {
            encoder_layers: text.encoder_layers,
            decoder_layers: text.decoder_layers,
            local_experts: text.local_experts,
            experts_per_token: text.experts_per_token,
            engram_max_ngram_size: text.engram_max_ngram_size,
        })
    }

    /// Returns the total number of transformer layers.
    #[must_use]
    pub const fn total_layers(&self) -> u16 {
        self.encoder_layers + self.decoder_layers
    }

    /// Returns the number of routed experts in the checkpoint.
    #[must_use]
    pub const fn local_experts(&self) -> u16 {
        self.local_experts
    }

    /// Returns how many experts one token routes to.
    #[must_use]
    pub const fn experts_per_token(&self) -> u16 {
        self.experts_per_token
    }

    /// Returns the maximum Engram n-gram size.
    #[must_use]
    pub const fn engram_max_ngram_size(&self) -> u8 {
        self.engram_max_ngram_size
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    encoder_layers: u16,
    #[serde(default)]
    decoder_layers: u16,
    #[serde(default)]
    local_experts: u16,
    #[serde(default)]
    experts_per_token: u16,
    #[serde(default)]
    engram_max_ngram_size: u8,
    #[serde(default)]
    text_config: Option<Box<RawConfig>>,
}

/// An invalid or unsupported V4.1 execution layout.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41ConfigError {
    /// The supplied document was not JSON.
    #[error("invalid configuration JSON: {0}")]
    Json(serde_json::Error),
    /// The configuration was not the expected V4.1 text model.
    #[error("expected model_type deepseek_v41_text, got {0:?}")]
    UnexpectedModelType(String),
    /// The configuration does not expose a non-empty CED split.
    #[error("V4.1 configuration has no usable CED encoder/decoder split")]
    MissingCedSplit,
    /// The configuration does not expose sparse-MoE routing.
    #[error("V4.1 configuration has no usable MoE routing")]
    MissingMoERouting,
    /// The configuration does not expose Engram state.
    #[error("V4.1 configuration has no usable Engram configuration")]
    MissingEngram,
}

#[cfg(test)]
mod tests {
    use super::{V41ConfigError, V41TextContract};

    const CONFIG: &str = r#"{
        "model_type": "deepseek_v41",
        "text_config": {
            "model_type": "deepseek_v41_text",
            "encoder_layers": 20,
            "decoder_layers": 20,
            "local_experts": 384,
            "experts_per_token": 8,
            "engram_max_ngram_size": 4
        }
    }"#;

    #[test]
    fn parses_the_ced_and_sparse_moe_contract() {
        let contract = V41TextContract::parse(CONFIG).expect("valid V4.1 config");
        assert_eq!(contract.total_layers(), 40);
        assert_eq!(contract.local_experts(), 384);
        assert_eq!(contract.experts_per_token(), 8);
        assert_eq!(contract.engram_max_ngram_size(), 4);
    }

    #[test]
    fn rejects_another_architecture() {
        let error = V41TextContract::parse(r#"{"model_type":"llama"}"#)
            .expect_err("other architectures are not V4.1");
        assert!(matches!(error, V41ConfigError::UnexpectedModelType(_)));
    }
}
