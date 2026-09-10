//! Derived Qwen3 execution dimensions, validated before loading weights.

use thiserror::Error;

use crate::Qwen3TextContract;

/// Dimensions needed to construct a Qwen3 text-model execution plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3ExecutionPreflight {
    head_dim: u32,
    key_value_heads: u32,
    kv_bytes_per_token_bf16: u64,
    position_capacity: u32,
}

impl Qwen3ExecutionPreflight {
    /// Validates dimensions derived from a parsed Qwen3 text-model contract.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3PreflightError`] when the attention heads cannot evenly
    /// partition the hidden representation or no positions are available.
    pub const fn from_contract(contract: &Qwen3TextContract) -> Result<Self, Qwen3PreflightError> {
        let attention_heads = contract.attention_heads();

        let position_capacity = contract.max_position_embeddings();
        if position_capacity == 0 {
            return Err(Qwen3PreflightError::MissingPositionCapacity);
        }

        let key_value_heads = contract.key_value_heads();
        if attention_heads % key_value_heads != 0 {
            return Err(
                Qwen3PreflightError::AttentionHeadsNotDivisibleByKeyValueHeads {
                    attention_heads,
                    key_value_heads,
                },
            );
        }
        let kv_bytes_per_token_bf16 = (contract.total_layers() as u64)
            * (key_value_heads as u64)
            * (contract.head_dim() as u64)
            * 2
            * 2;

        Ok(Self {
            head_dim: contract.head_dim(),
            key_value_heads,
            kv_bytes_per_token_bf16,
            position_capacity,
        })
    }

    /// Returns the width of one attention head.
    #[must_use]
    pub const fn head_dim(self) -> u32 {
        self.head_dim
    }

    /// Returns the number of key/value heads per layer.
    #[must_use]
    pub const fn key_value_heads(self) -> u32 {
        self.key_value_heads
    }

    /// Returns the physical BF16 key/value cache footprint for one token.
    ///
    /// This covers keys and values across all decoder layers, but excludes
    /// allocator alignment and any prefix-cache bookkeeping.
    #[must_use]
    pub const fn kv_bytes_per_token_bf16(self) -> u64 {
        self.kv_bytes_per_token_bf16
    }

    /// Returns the maximum number of positions available to execution.
    #[must_use]
    pub const fn position_capacity(self) -> u32 {
        self.position_capacity
    }
}

/// A Qwen3 text-model contract that cannot produce an execution plan.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3PreflightError {
    /// Grouped-query key/value heads must evenly partition attention heads.
    #[error(
        "Qwen3 attention heads {attention_heads} are not divisible by {key_value_heads} key/value heads"
    )]
    AttentionHeadsNotDivisibleByKeyValueHeads {
        /// Number of query heads.
        attention_heads: u32,
        /// Number of key/value heads.
        key_value_heads: u32,
    },
    /// The model does not allow any token positions.
    #[error("Qwen3 configuration has no usable position capacity")]
    MissingPositionCapacity,
}

#[cfg(test)]
mod tests {
    use super::{Qwen3ExecutionPreflight, Qwen3PreflightError};
    use crate::Qwen3TextContract;

    #[test]
    fn derives_attention_and_position_dimensions() {
        let contract = Qwen3TextContract::parse(
            r#"{
                "model_type":"qwen3",
                "num_hidden_layers":28,
                "hidden_size":1024,
                "vocab_size":151936,
                "num_attention_heads":16,
                "num_key_value_heads":8,
                "head_dim":128,
                "max_position_embeddings":40960
            }"#,
        )
        .expect("valid Qwen3 config");

        let preflight = Qwen3ExecutionPreflight::from_contract(&contract)
            .expect("divisible attention dimensions");
        assert_eq!(preflight.head_dim(), 128);
        assert_eq!(preflight.key_value_heads(), 8);
        assert_eq!(preflight.kv_bytes_per_token_bf16(), 114_688);
        assert_eq!(preflight.position_capacity(), 40_960);
    }

    #[test]
    fn rejects_non_groupable_key_value_heads() {
        let contract = Qwen3TextContract::parse(
            r#"{
                "model_type":"qwen3",
                "num_hidden_layers":28,
                "hidden_size":1024,
                "vocab_size":151936,
                "num_attention_heads":16,
                "num_key_value_heads":3,
                "head_dim":128,
                "max_position_embeddings":40960
            }"#,
        )
        .expect("config dimensions are syntactically valid");

        let error = Qwen3ExecutionPreflight::from_contract(&contract)
            .expect_err("GQA heads must divide query heads");
        assert!(matches!(
            error,
            Qwen3PreflightError::AttentionHeadsNotDivisibleByKeyValueHeads {
                attention_heads: 16,
                key_value_heads: 3,
            }
        ));
    }
}
