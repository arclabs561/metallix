//! DeepSeek-V4.1 execution-contract parsing and bounded operator qualifications.

pub mod attention;
pub mod checkpoint;
pub mod csa2;
#[cfg(feature = "metal")]
pub mod indexer;
pub mod manifest;
pub mod precision;
pub mod rotary;
pub mod selection;

pub use attention::{SparseAttentionError, SparseAttentionLayout, sparse_attention_reference};
pub use checkpoint::{
    V41ExpertI8ScalePair, V41ExpertI8ScalePairError, V41ExpertProjection, V41SafetensorsHeader,
    V41SafetensorsHeaderError, V41StorageDtype, V41TensorRange,
};
pub use csa2::{CandidateError, candidate_mask};
#[cfg(feature = "metal")]
pub use indexer::{IndexScoreError, index_scores_f32};
pub use rotary::{
    RotaryDirection, RotaryError, RotaryFrequency, RotaryFrequencyError, RotaryFrequencyParameters,
    RotaryTailLayout, rotate_tail,
};
#[cfg(feature = "metal")]
pub use rotary::{RotaryMetalError, rotate_tail_metal};
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

/// Initial attention/MoE dimensions and CSA2 source lists needed before tensor planning.
///
/// This is intentionally a stronger gate than [`V41TextContract`]: metadata
/// inspection can parse a small config projection, while a loader must know
/// the widths and cache owners it first uses. It does not yet represent
/// Engram table layouts or CED state flow, so it is not an executable load
/// plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41ExecutionShape {
    text: V41TextContract,
    draft_layers: u16,
    hidden_size: u16,
    vocab_size: u32,
    attention_heads: u16,
    key_value_heads: u16,
    head_dim: u16,
    rope_head_dim: u16,
    q_lora_rank: u16,
    o_lora_rank: u16,
    output_groups: u16,
    sliding_window: u32,
    max_position_embeddings: u32,
    moe_intermediate_size: u16,
    shared_experts: u16,
    csa2: V41Csa2Schedule,
}

impl V41ExecutionShape {
    /// Parses initial attention/MoE dimensions and CSA2 source-list consistency.
    ///
    /// This checks configuration shape only. It does not authenticate a model
    /// revision, inspect checkpoint tensor names or bytes, load weights, or
    /// qualify a numerical forward pass. It also does not validate Engram or
    /// CED tensor layout.
    ///
    /// # Errors
    ///
    /// Returns [`V41ConfigError`] when the input is not a supported V4.1 text
    /// layout or omits a dimension or CSA2 cache schedule needed by a loader.
    pub fn parse(json: &str) -> Result<Self, V41ConfigError> {
        let text = V41TextContract::parse(json)?;
        let config: RawConfig = serde_json::from_str(json).map_err(V41ConfigError::Json)?;
        let raw = config.text_config.as_deref().unwrap_or(&config);

        if raw.hidden_size == 0
            || raw.vocab_size == 0
            || raw.num_attention_heads == 0
            || raw.num_key_value_heads == 0
            || raw.head_dim == 0
            || raw.qk_rope_head_dim == 0
            || raw.qk_rope_head_dim > raw.head_dim
            || raw.q_lora_rank == 0
            || raw.o_lora_rank == 0
            || raw.o_groups == 0
            || raw.sliding_window == 0
            || raw.max_position_embeddings == 0
        {
            return Err(V41ConfigError::MissingAttentionLayout);
        }
        // The same rotary tail is applied to attention and indexer vectors;
        // apply_rotary_emb interprets adjacent values as complex pairs.
        if !raw.qk_rope_head_dim.is_multiple_of(2) || raw.qk_rope_head_dim > raw.index_head_dim {
            return Err(V41ConfigError::UnsupportedAttentionLayout);
        }
        if raw.n_shared_experts != 1
            || raw.moe_intermediate_size == 0
            || raw.num_experts_per_tok > raw.n_routed_experts
        {
            return Err(V41ConfigError::UnsupportedMoELayout);
        }
        let output_width = u32::from(raw.num_attention_heads) * u32::from(raw.head_dim);
        if output_width % u32::from(raw.o_groups) != 0 {
            return Err(V41ConfigError::UnsupportedAttentionLayout);
        }

        Ok(Self {
            text,
            draft_layers: raw.num_nextn_predict_layers,
            hidden_size: raw.hidden_size,
            vocab_size: raw.vocab_size,
            attention_heads: raw.num_attention_heads,
            key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rope_head_dim: raw.qk_rope_head_dim,
            q_lora_rank: raw.q_lora_rank,
            o_lora_rank: raw.o_lora_rank,
            output_groups: raw.o_groups,
            sliding_window: raw.sliding_window,
            max_position_embeddings: raw.max_position_embeddings,
            moe_intermediate_size: raw.moe_intermediate_size,
            shared_experts: raw.n_shared_experts,
            csa2: V41Csa2Schedule::parse(raw)?,
        })
    }

    /// Returns the metadata contract from which this execution shape was qualified.
    #[must_use]
    pub const fn text_contract(&self) -> &V41TextContract {
        &self.text
    }

    /// Returns appended `DSpark` draft layers, included in the CSA2 ratio schedule.
    #[must_use]
    pub const fn draft_layers(&self) -> u16 {
        self.draft_layers
    }

    /// Returns transformer and draft layers together, as indexed by CSA2.
    #[must_use]
    pub fn total_layers_with_draft(&self) -> u32 {
        u32::from(self.text.total_layers()) + u32::from(self.draft_layers)
    }

    /// Returns the token representation width.
    #[must_use]
    pub const fn hidden_size(&self) -> u16 {
        self.hidden_size
    }

    /// Returns the tokenizer vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    /// Returns the query attention-head count.
    #[must_use]
    pub const fn attention_heads(&self) -> u16 {
        self.attention_heads
    }

    /// Returns the key/value attention-head count.
    #[must_use]
    pub const fn key_value_heads(&self) -> u16 {
        self.key_value_heads
    }

    /// Returns the latent-attention head width.
    #[must_use]
    pub const fn head_dim(&self) -> u16 {
        self.head_dim
    }

    /// Returns the rotary tail width within an attention head.
    #[must_use]
    pub const fn rope_head_dim(&self) -> u16 {
        self.rope_head_dim
    }

    /// Returns the query low-rank projection width.
    #[must_use]
    pub const fn q_lora_rank(&self) -> u16 {
        self.q_lora_rank
    }

    /// Returns the grouped output low-rank projection width.
    #[must_use]
    pub const fn o_lora_rank(&self) -> u16 {
        self.o_lora_rank
    }

    /// Returns the number of grouped output-projection groups.
    #[must_use]
    pub const fn output_groups(&self) -> u16 {
        self.output_groups
    }

    /// Returns the local attention-window capacity in tokens.
    #[must_use]
    pub const fn sliding_window(&self) -> u32 {
        self.sliding_window
    }

    /// Returns the declared maximum token position.
    #[must_use]
    pub const fn max_position_embeddings(&self) -> u32 {
        self.max_position_embeddings
    }

    /// Returns the routed and shared expert intermediate width.
    #[must_use]
    pub const fn moe_intermediate_size(&self) -> u16 {
        self.moe_intermediate_size
    }

    /// Returns the always-on shared-expert count.
    #[must_use]
    pub const fn shared_experts(&self) -> u16 {
        self.shared_experts
    }

    /// Returns the validated CSA2 cache-owner and sparse-index schedule.
    #[must_use]
    pub const fn csa2(&self) -> &V41Csa2Schedule {
        &self.csa2
    }
}

/// CSA2 fields that define compressed-KV and sparse-index ownership.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct V41Csa2Schedule {
    compress_ratios: Vec<u16>,
    kv_source_layers: Vec<u16>,
    index_source_layers: Vec<u16>,
    index_heads: u16,
    index_head_dim: u16,
    index_topk: u16,
    candidate_source_layer: Option<u16>,
    candidate_topk_blocks: u16,
    candidate_block_size: u16,
}

impl V41Csa2Schedule {
    fn parse(raw: &RawConfig) -> Result<Self, V41ConfigError> {
        let expected_layers = usize::from(raw.num_hidden_layers)
            .checked_add(usize::from(raw.num_nextn_predict_layers))
            .ok_or(V41ConfigError::InvalidCsa2Schedule)?;
        if raw.compress_ratios.len() != expected_layers
            || raw.kv_source_layer_ids.is_empty()
            || raw.index_source_layer_ids.is_empty()
            || raw.index_n_heads == 0
            || raw.index_head_dim == 0
            || raw.index_topk == 0
        {
            return Err(V41ConfigError::InvalidCsa2Schedule);
        }
        let backbone_layers = raw.num_hidden_layers;
        let source_is_valid =
            |layer: u16| layer < backbone_layers && raw.compress_ratios[usize::from(layer)] != 0;
        if !unique_layers(&raw.kv_source_layer_ids)
            || !raw.kv_source_layer_ids.iter().copied().all(source_is_valid)
        {
            return Err(V41ConfigError::InvalidCsa2Schedule);
        }
        if !unique_layers(&raw.index_source_layer_ids)
            || !raw
                .index_source_layer_ids
                .iter()
                .copied()
                .all(source_is_valid)
        {
            return Err(V41ConfigError::InvalidCsa2Schedule);
        }
        for (layer, &ratio) in raw.compress_ratios.iter().enumerate() {
            if ratio == 0 {
                continue;
            }
            // The reference overwrites one shared slot, not a map by ratio.
            // An older compatible publisher cannot rescue a newer mismatch.
            let latest_matches = |sources: &[u16]| {
                sources
                    .iter()
                    .copied()
                    .filter(|&owner| usize::from(owner) <= layer)
                    .max()
                    .is_some_and(|owner| raw.compress_ratios[usize::from(owner)] == ratio)
            };
            if !latest_matches(&raw.kv_source_layer_ids)
                || !latest_matches(&raw.index_source_layer_ids)
            {
                return Err(V41ConfigError::InvalidCsa2Schedule);
            }
        }

        let candidate_source_layer = match raw.candidate_source_layer_id {
            -1 => None,
            layer if layer >= 0 => u16::try_from(layer)
                .ok()
                .filter(|layer| raw.index_source_layer_ids.contains(layer))
                .ok_or(V41ConfigError::InvalidCsa2Schedule)
                .map(Some)?,
            _ => return Err(V41ConfigError::InvalidCsa2Schedule),
        };
        if candidate_source_layer.is_some()
            && (raw.candidate_topk_blocks == 0 || raw.candidate_block_size == 0)
        {
            return Err(V41ConfigError::InvalidCsa2Schedule);
        }

        Ok(Self {
            compress_ratios: raw.compress_ratios.clone(),
            kv_source_layers: raw.kv_source_layer_ids.clone(),
            index_source_layers: raw.index_source_layer_ids.clone(),
            index_heads: raw.index_n_heads,
            index_head_dim: raw.index_head_dim,
            index_topk: raw.index_topk,
            candidate_source_layer,
            candidate_topk_blocks: raw.candidate_topk_blocks,
            candidate_block_size: raw.candidate_block_size,
        })
    }

    /// Returns one compression ratio per transformer and draft layer.
    #[must_use]
    pub fn compress_ratios(&self) -> &[u16] {
        &self.compress_ratios
    }

    /// Returns layers that publish shared compressed KV.
    #[must_use]
    pub fn kv_source_layers(&self) -> &[u16] {
        &self.kv_source_layers
    }

    /// Returns layers that publish sparse-attention Top-K indices.
    #[must_use]
    pub fn index_source_layers(&self) -> &[u16] {
        &self.index_source_layers
    }

    /// Returns sparse-index query-head count.
    #[must_use]
    pub const fn index_heads(&self) -> u16 {
        self.index_heads
    }

    /// Returns the sparse-index head width.
    #[must_use]
    pub const fn index_head_dim(&self) -> u16 {
        self.index_head_dim
    }

    /// Returns the maximum compressed positions retained per query.
    #[must_use]
    pub const fn index_topk(&self) -> u16 {
        self.index_topk
    }

    /// Returns the first-level candidate-source layer, if candidate filtering is enabled.
    #[must_use]
    pub const fn candidate_source_layer(&self) -> Option<u16> {
        self.candidate_source_layer
    }

    /// Returns first-level candidate blocks retained per query.
    #[must_use]
    pub const fn candidate_topk_blocks(&self) -> u16 {
        self.candidate_topk_blocks
    }

    /// Returns compressed positions per candidate block.
    #[must_use]
    pub const fn candidate_block_size(&self) -> u16 {
        self.candidate_block_size
    }
}

fn unique_layers(layers: &[u16]) -> bool {
    let mut sorted = layers.to_vec();
    sorted.sort_unstable();
    sorted.windows(2).all(|pair| pair[0] != pair[1])
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
    num_nextn_predict_layers: u16,
    #[serde(default)]
    hidden_size: u16,
    #[serde(default)]
    vocab_size: u32,
    #[serde(default)]
    num_attention_heads: u16,
    #[serde(default)]
    num_key_value_heads: u16,
    #[serde(default)]
    head_dim: u16,
    #[serde(default)]
    qk_rope_head_dim: u16,
    #[serde(default)]
    q_lora_rank: u16,
    #[serde(default)]
    o_lora_rank: u16,
    #[serde(default)]
    o_groups: u16,
    #[serde(default)]
    sliding_window: u32,
    #[serde(default)]
    max_position_embeddings: u32,
    #[serde(default)]
    n_routed_experts: u16,
    #[serde(default)]
    num_experts_per_tok: u16,
    #[serde(default)]
    n_shared_experts: u16,
    #[serde(default)]
    moe_intermediate_size: u16,
    #[serde(default)]
    compress_ratios: Vec<u16>,
    #[serde(default)]
    kv_source_layer_ids: Vec<u16>,
    #[serde(default)]
    index_source_layer_ids: Vec<u16>,
    #[serde(default)]
    index_n_heads: u16,
    #[serde(default)]
    index_head_dim: u16,
    #[serde(default)]
    index_topk: u16,
    #[serde(default = "default_candidate_source_layer_id")]
    candidate_source_layer_id: i32,
    #[serde(default)]
    candidate_topk_blocks: u16,
    #[serde(default)]
    candidate_block_size: u16,
    #[serde(default)]
    engram_max_ngram_size: u8,
    #[serde(default)]
    text_config: Option<Box<RawConfig>>,
    #[serde(default)]
    quantization_config: Option<RawQuantizationConfig>,
}

const fn default_candidate_source_layer_id() -> i32 {
    -1
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
    /// The configuration omits an attention width, rank, or cache capacity.
    #[error("V4.1 configuration has no usable attention execution layout")]
    MissingAttentionLayout,
    /// The configuration's attention projection groups cannot be planned.
    #[error("V4.1 configuration has an unsupported attention execution layout")]
    UnsupportedAttentionLayout,
    /// The configuration's `MoE` counts do not match the supported source layout.
    #[error("V4.1 configuration has an unsupported MoE execution layout")]
    UnsupportedMoELayout,
    /// The configuration's compressed-KV or sparse-index schedule is inconsistent.
    #[error("V4.1 configuration has an invalid CSA2 cache/index schedule")]
    InvalidCsa2Schedule,
    /// The configuration does not declare checkpoint quantization.
    #[error("V4.1 configuration has no checkpoint quantization contract")]
    MissingQuantization,
    /// Metallix does not yet support this checkpoint quantization layout.
    #[error("V4.1 configuration has an unsupported checkpoint quantization layout")]
    UnsupportedQuantization,
}

#[cfg(test)]
mod tests {
    use super::{V41ConfigError, V41ExecutionShape, V41TextContract};

    const CONFIG: &str = r#"{
      "model_type":"deepseek_v41",
      "text_config":{"model_type":"deepseek_v41_text","num_hidden_layers":40,"n_routed_experts":384,"num_experts_per_tok":6,"engram_max_ngram_size":4},
      "quantization_config":{"quant_method":"fp8","activation_scheme":"dynamic","weight_block_size":[32,32],"scale_fmt":"ue8m0","expert_dtype":"fp4"}
    }"#;

    // Source: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/config.json
    // SHA-256: 8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879.
    // This is the pinned Flash configuration's execution-relevant projection,
    // not a checkpoint identity or a numerical-forward fixture.
    const EXECUTION_CONFIG: &str = r#"{
      "model_type":"deepseek_v41",
      "text_config":{
        "model_type":"deepseek_v41_text",
        "num_hidden_layers":40,
        "num_nextn_predict_layers":3,
        "hidden_size":5120,
        "vocab_size":129280,
        "num_attention_heads":64,
        "num_key_value_heads":1,
        "head_dim":512,
        "qk_rope_head_dim":64,
        "q_lora_rank":1280,
        "o_lora_rank":1024,
        "o_groups":8,
        "sliding_window":128,
        "max_position_embeddings":1048576,
        "n_routed_experts":384,
        "num_experts_per_tok":6,
        "n_shared_experts":1,
        "moe_intermediate_size":2304,
        "compress_ratios":[0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        "kv_source_layer_ids":[2,8,14,20],
        "index_source_layer_ids":[2,8,14,20,24,28,32,36],
        "index_n_heads":32,
        "index_head_dim":128,
        "index_topk":512,
        "candidate_source_layer_id":20,
        "candidate_topk_blocks":2048,
        "candidate_block_size":8,
        "engram_max_ngram_size":4
      },
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

    #[test]
    fn qualifies_the_pinned_flash_execution_shape_projection() {
        let shape = V41ExecutionShape::parse(EXECUTION_CONFIG).expect("valid execution shape");
        assert_eq!(shape.text_contract().total_layers(), 40);
        assert_eq!(shape.draft_layers(), 3);
        assert_eq!(shape.total_layers_with_draft(), 43);
        assert_eq!(shape.hidden_size(), 5120);
        assert_eq!(shape.vocab_size(), 129_280);
        assert_eq!(shape.attention_heads(), 64);
        assert_eq!(shape.key_value_heads(), 1);
        assert_eq!(shape.head_dim(), 512);
        assert_eq!(shape.rope_head_dim(), 64);
        assert_eq!(shape.q_lora_rank(), 1280);
        assert_eq!(shape.o_lora_rank(), 1024);
        assert_eq!(shape.output_groups(), 8);
        assert_eq!(shape.sliding_window(), 128);
        assert_eq!(shape.max_position_embeddings(), 1_048_576);
        assert_eq!(shape.moe_intermediate_size(), 2304);
        assert_eq!(shape.shared_experts(), 1);
        assert_eq!(shape.csa2().compress_ratios().len(), 43);
        assert_eq!(shape.csa2().kv_source_layers(), [2, 8, 14, 20]);
        assert_eq!(
            shape.csa2().index_source_layers(),
            [2, 8, 14, 20, 24, 28, 32, 36]
        );
        assert_eq!(shape.csa2().index_heads(), 32);
        assert_eq!(shape.csa2().index_head_dim(), 128);
        assert_eq!(shape.csa2().index_topk(), 512);
        assert_eq!(shape.csa2().candidate_source_layer(), Some(20));
        assert_eq!(shape.csa2().candidate_topk_blocks(), 2048);
        assert_eq!(shape.csa2().candidate_block_size(), 8);
    }

    #[test]
    fn rejects_unexecutable_rotary_widths() {
        for width in [63, 256] {
            let mut config: serde_json::Value = serde_json::from_str(EXECUTION_CONFIG).unwrap();
            config["text_config"]["qk_rope_head_dim"] = width.into();
            assert!(matches!(
                V41ExecutionShape::parse(&config.to_string()),
                Err(V41ConfigError::UnsupportedAttentionLayout)
            ));
        }
    }

    #[test]
    fn rejects_unexecutable_reuse_of_an_overwritten_cache_ratio() {
        let mut config: serde_json::Value = serde_json::from_str(EXECUTION_CONFIG).unwrap();
        // Layer 20 overwrote the shared slots with ratio 1. Older ratio-2
        // publishers still exist in the list, but their slots are not live.
        config["text_config"]["compress_ratios"][21] = 2.into();
        assert!(matches!(
            V41ExecutionShape::parse(&config.to_string()),
            Err(V41ConfigError::InvalidCsa2Schedule)
        ));
    }

    #[test]
    fn metadata_contract_does_not_treat_a_projection_as_an_execution_shape() {
        V41TextContract::parse(CONFIG).expect("metadata projection remains supported");
        let error = V41ExecutionShape::parse(CONFIG).expect_err("shape fields are required");
        assert!(matches!(error, V41ConfigError::MissingAttentionLayout));
    }

    #[test]
    fn rejects_an_index_source_without_a_prior_matching_kv_owner() {
        let config = EXECUTION_CONFIG.replace(
            "\"kv_source_layer_ids\":[2,8,14,20]",
            "\"kv_source_layer_ids\":[2,8,14]",
        );
        let error = V41ExecutionShape::parse(&config).expect_err("unowned index cache");
        assert!(matches!(error, V41ConfigError::InvalidCsa2Schedule));
    }

    #[test]
    fn rejects_a_candidate_source_that_does_not_publish_indices() {
        let config = EXECUTION_CONFIG.replace(
            "\"candidate_source_layer_id\":20",
            "\"candidate_source_layer_id\":19",
        );
        let error =
            V41ExecutionShape::parse(&config).expect_err("candidate source must be index owner");
        assert!(matches!(error, V41ConfigError::InvalidCsa2Schedule));
    }

    #[test]
    fn rejects_a_missing_initial_attention_dimension() {
        let config = EXECUTION_CONFIG.replace("\"hidden_size\":5120", "\"hidden_size\":0");
        let error = V41ExecutionShape::parse(&config).expect_err("hidden width is required");
        assert!(matches!(error, V41ConfigError::MissingAttentionLayout));
    }

    #[test]
    fn rejects_an_output_group_that_cannot_partition_attention_width() {
        let config = EXECUTION_CONFIG.replace("\"o_groups\":8", "\"o_groups\":7");
        let error = V41ExecutionShape::parse(&config).expect_err("invalid output grouping");
        assert!(matches!(error, V41ConfigError::UnsupportedAttentionLayout));
    }

    #[test]
    fn rejects_duplicate_kv_source_layers() {
        let config = EXECUTION_CONFIG.replace(
            "\"kv_source_layer_ids\":[2,8,14,20]",
            "\"kv_source_layer_ids\":[2,2,14,20]",
        );
        let error = V41ExecutionShape::parse(&config).expect_err("duplicate source");
        assert!(matches!(error, V41ConfigError::InvalidCsa2Schedule));
    }

    #[test]
    fn rejects_an_out_of_range_index_source_layer() {
        let config = EXECUTION_CONFIG.replace(
            "\"index_source_layer_ids\":[2,8,14,20,24,28,32,36]",
            "\"index_source_layer_ids\":[2,8,14,20,24,28,32,40]",
        );
        let error = V41ExecutionShape::parse(&config).expect_err("out-of-range source");
        assert!(matches!(error, V41ConfigError::InvalidCsa2Schedule));
    }

    #[test]
    fn rejects_a_ratio_schedule_with_the_wrong_layer_count() {
        let config = EXECUTION_CONFIG.replace(
            "\"num_nextn_predict_layers\":3",
            "\"num_nextn_predict_layers\":4",
        );
        let error = V41ExecutionShape::parse(&config).expect_err("wrong schedule length");
        assert!(matches!(error, V41ConfigError::InvalidCsa2Schedule));
    }

    #[test]
    fn rejects_a_compressed_layer_without_preceding_shared_state() {
        let config =
            EXECUTION_CONFIG.replace("\"compress_ratios\":[0,0,2", "\"compress_ratios\":[0,2,2");
        let error = V41ExecutionShape::parse(&config).expect_err("unowned compressed state");
        assert!(matches!(error, V41ConfigError::InvalidCsa2Schedule));
    }
}
