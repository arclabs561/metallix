//! Model-local, scalar layer-attention composition for the pinned V4.1 path.
//!
//! This is deliberately a layer-4-shaped adapter, rather than a cache scheduler:
//! it owns one numerical BF16 window ring and consumes an explicitly published
//! compressed-KV slice.  The publication is checked against a source-layer,
//! epoch, and successful-call ordinal before any state is changed.

mod error;
mod layout;
mod ops;
mod qr;
mod state;
#[cfg(test)]
mod tests;

pub use error::{LayerAttentionError, LayerAttentionLayoutError};
pub use layout::LayerAttentionLayout;
pub use qr::{
    AttentionQrDiagnostic, AttentionQrLayout, AttentionQrWeights, Fp8Projection,
    prepare_attention_qr,
};
pub use state::{
    CompressedAttentionPublication, LayerAttentionDiagnostic, LayerAttentionState,
    LayerAttentionWeights,
};

const MAX_LAYER_ATTENTION_ELEMENTS: usize = 1 << 23;
