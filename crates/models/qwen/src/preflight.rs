//! Derived Qwen3 execution dimensions, validated before loading weights.

use thiserror::Error;

use crate::Qwen3TextContract;

/// Dimensions needed to construct a Qwen3 text-model execution plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3ExecutionPreflight {
    head_dim: u32,
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
        let hidden_size = contract.hidden_size();
        let attention_heads = contract.attention_heads();
        if attention_heads == 0 || hidden_size % attention_heads != 0 {
            return Err(
                Qwen3PreflightError::HiddenSizeNotDivisibleByAttentionHeads {
                    hidden_size,
                    attention_heads,
                },
            );
        }

        let position_capacity = contract.max_position_embeddings();
        if position_capacity == 0 {
            return Err(Qwen3PreflightError::MissingPositionCapacity);
        }

        Ok(Self {
            head_dim: hidden_size / attention_heads,
            position_capacity,
        })
    }

    /// Returns the width of one attention head.
    #[must_use]
    pub const fn head_dim(self) -> u32 {
        self.head_dim
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
    /// The hidden representation cannot be divided into equal attention heads.
    #[error(
        "Qwen3 hidden size {hidden_size} is not divisible by {attention_heads} attention heads"
    )]
    HiddenSizeNotDivisibleByAttentionHeads {
        /// Width of the hidden representation.
        hidden_size: u32,
        /// Number of attention heads.
        attention_heads: u32,
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
                "num_attention_heads":16,
                "max_position_embeddings":40960
            }"#,
        )
        .expect("valid Qwen3 config");

        let preflight = Qwen3ExecutionPreflight::from_contract(&contract)
            .expect("divisible attention dimensions");
        assert_eq!(preflight.head_dim(), 64);
        assert_eq!(preflight.position_capacity(), 40_960);
    }

    #[test]
    fn rejects_uneven_attention_dimensions() {
        let contract = Qwen3TextContract::parse(
            r#"{
                "model_type":"qwen3",
                "num_hidden_layers":28,
                "hidden_size":1025,
                "num_attention_heads":16,
                "max_position_embeddings":40960
            }"#,
        )
        .expect("config dimensions are syntactically valid");

        let error = Qwen3ExecutionPreflight::from_contract(&contract)
            .expect_err("uneven attention dimensions must be rejected");
        assert!(matches!(
            error,
            Qwen3PreflightError::HiddenSizeNotDivisibleByAttentionHeads {
                hidden_size: 1025,
                attention_heads: 16,
            }
        ));
    }
}
