//! DeepSeek-V4.1 execution-contract parsing and bounded operator qualifications.

pub mod csa2;
#[cfg(feature = "metal")]
pub mod indexer;
pub mod manifest;
pub mod selection;

pub use csa2::{CandidateError, candidate_mask};
#[cfg(feature = "metal")]
pub use indexer::{IndexScoreError, index_scores_f32};
pub use selection::{SelectionError, select_indices};

// MLX's native test operations share process-global device initialization.
#[cfg(all(test, feature = "metal"))]
pub(crate) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

use serde::Deserialize;
use thiserror::Error;

/// The validated text-execution contract extracted from a V4.1 configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41TextContract {
    hidden_layers: u16,
    routed_experts: u16,
    experts_per_token: u16,
    engram_max_ngram_size: u8,
    quantization: V41QuantizationContract,
}

impl V41TextContract {
    /// Parses a V4.1 configuration and rejects layouts Metallix does not support.
    ///
    /// # Errors
    ///
    /// Returns [`V41ConfigError`] when the document is malformed or is not the
    /// expected V4.1 text layout and official checkpoint quantization.
    pub fn parse(json: &str) -> Result<Self, V41ConfigError> {
        let config: RawConfig = serde_json::from_str(json).map_err(V41ConfigError::Json)?;
        let text = config.text_config.as_deref().unwrap_or(&config);
        if text.model_type != "deepseek_v41_text" {
            return Err(V41ConfigError::UnexpectedModelType(text.model_type.clone()));
        }
        if text.num_hidden_layers == 0 {
            return Err(V41ConfigError::MissingLayers);
        }
        if text.n_routed_experts == 0 || text.num_experts_per_tok == 0 {
            return Err(V41ConfigError::MissingMoERouting);
        }
        if text.engram_max_ngram_size == 0 {
            return Err(V41ConfigError::MissingEngram);
        }
        let quantization = V41QuantizationContract::parse(
            config
                .quantization_config
                .as_ref()
                .ok_or(V41ConfigError::MissingQuantization)?,
        )?;
        Ok(Self {
            hidden_layers: text.num_hidden_layers,
            routed_experts: text.n_routed_experts,
            experts_per_token: text.num_experts_per_tok,
            engram_max_ngram_size: text.engram_max_ngram_size,
            quantization,
        })
    }

    /// Returns the total number of transformer layers.
    #[must_use]
    pub const fn total_layers(&self) -> u16 {
        self.hidden_layers
    }

    /// Returns the number of routed experts in the checkpoint.
    #[must_use]
    pub const fn local_experts(&self) -> u16 {
        self.routed_experts
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

    /// Returns the validated official checkpoint quantization contract.
    #[must_use]
    pub const fn quantization(&self) -> V41QuantizationContract {
        self.quantization
    }
}

/// Quantization layout required by the official V4.1 Flash checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct V41QuantizationContract {
    weight_block_rows: u16,
    weight_block_columns: u16,
}

impl V41QuantizationContract {
    fn parse(config: &RawQuantizationConfig) -> Result<Self, V41ConfigError> {
        if config.quant_method != "fp8"
            || config.activation_scheme != "dynamic"
            || config.scale_fmt != "ue8m0"
            || config.expert_dtype != "fp4"
        {
            return Err(V41ConfigError::UnsupportedQuantization);
        }
        let [rows, columns] = config.weight_block_size.as_slice() else {
            return Err(V41ConfigError::UnsupportedQuantization);
        };
        if *rows == 0 || *columns == 0 {
            return Err(V41ConfigError::UnsupportedQuantization);
        }
        Ok(Self {
            weight_block_rows: *rows,
            weight_block_columns: *columns,
        })
    }

    /// Returns the rows in one checkpoint weight-quantization block.
    #[must_use]
    pub const fn weight_block_rows(self) -> u16 {
        self.weight_block_rows
    }

    /// Returns the columns in one checkpoint weight-quantization block.
    #[must_use]
    pub const fn weight_block_columns(self) -> u16 {
        self.weight_block_columns
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    num_hidden_layers: u16,
    #[serde(default)]
    n_routed_experts: u16,
    #[serde(default)]
    num_experts_per_tok: u16,
    #[serde(default)]
    engram_max_ngram_size: u8,
    #[serde(default)]
    text_config: Option<Box<RawConfig>>,
    #[serde(default)]
    quantization_config: Option<RawQuantizationConfig>,
}

#[derive(Debug, Deserialize)]
struct RawQuantizationConfig {
    #[serde(default)]
    quant_method: String,
    #[serde(default)]
    activation_scheme: String,
    #[serde(default)]
    weight_block_size: Vec<u16>,
    #[serde(default)]
    scale_fmt: String,
    #[serde(default)]
    expert_dtype: String,
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
    /// The configuration does not expose any transformer layers.
    #[error("V4.1 configuration has no usable transformer layers")]
    MissingLayers,
    /// The configuration does not expose sparse-MoE routing.
    #[error("V4.1 configuration has no usable MoE routing")]
    MissingMoERouting,
    /// The configuration does not expose Engram state.
    #[error("V4.1 configuration has no usable Engram configuration")]
    MissingEngram,
    /// The configuration does not declare checkpoint quantization.
    #[error("V4.1 configuration has no checkpoint quantization contract")]
    MissingQuantization,
    /// Metallix does not yet support this checkpoint quantization layout.
    #[error("V4.1 configuration has an unsupported checkpoint quantization layout")]
    UnsupportedQuantization,
}

#[cfg(test)]
mod tests {
    use super::{V41ConfigError, V41TextContract};

    const CONFIG: &str = r#"{
      "model_type":"deepseek_v41",
      "text_config":{"model_type":"deepseek_v41_text","num_hidden_layers":40,"n_routed_experts":384,"num_experts_per_tok":6,"engram_max_ngram_size":4},
      "quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}
    }"#;

    #[test]
    fn parses_the_sparse_moe_and_quantization_contract() {
        let contract = V41TextContract::parse(CONFIG).expect("valid V4.1 config");
        assert_eq!(contract.total_layers(), 40);
        assert_eq!(contract.local_experts(), 384);
        assert_eq!(contract.experts_per_token(), 6);
        assert_eq!(contract.engram_max_ngram_size(), 4);
        assert_eq!(contract.quantization().weight_block_rows(), 32);
        assert_eq!(contract.quantization().weight_block_columns(), 32);
    }

    #[test]
    fn rejects_another_architecture() {
        let error = V41TextContract::parse(r#"{"model_type":"llama"}"#).expect_err("not V4.1");
        assert!(matches!(error, V41ConfigError::UnexpectedModelType(_)));
    }

    #[test]
    fn rejects_a_non_official_quantization_layout() {
        let config = CONFIG.replace("\"expert_dtype\":\"fp4\"", "\"expert_dtype\":\"fp8\"");
        let error = V41TextContract::parse(&config).expect_err("unsupported quantization");
        assert!(matches!(error, V41ConfigError::UnsupportedQuantization));
    }
}
