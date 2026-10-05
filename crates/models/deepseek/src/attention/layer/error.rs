//! Layer-attention configuration and execution errors.

use thiserror::Error;

use crate::{
    RotaryError,
    attention::{
        AttentionOutputError, AttentionOutputLayoutError, SparseAttentionBf16Error,
        SparseAttentionError, window::WindowError,
    },
    precision::{ActivationQuantError, ActivationRoundtripError, Fp8ForwardError, Fp8LinearError},
};

use super::MAX_LAYER_ATTENTION_ELEMENTS;

/// Invalid layer-attention configuration.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum LayerAttentionLayoutError {
    #[error("layer-attention shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("rope width {rope_width} exceeds head dimension {head_dimension}")]
    RopeExceedsHead {
        rope_width: usize,
        head_dimension: usize,
    },
    #[error("heads {heads} are not divisible by groups {groups}")]
    HeadsNotGrouped { heads: usize, groups: usize },
    #[error("{field} width {width} is not divisible by FP8 group 32")]
    UngroupedFp8Reduction { field: &'static str, width: usize },
    #[error(
        "layer-attention {field} has {elements} elements, maximum is {MAX_LAYER_ATTENTION_ELEMENTS}"
    )]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    #[error("norm epsilon must be finite and positive")]
    InvalidNormEpsilon,
    #[error("softmax scale must be finite and positive")]
    InvalidSoftmaxScale,
}

/// A rejected layer-attention call. State is unchanged on every variant.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayerAttentionError {
    #[error(transparent)]
    Layout(#[from] LayerAttentionLayoutError),
    #[error(transparent)]
    ActivationQuant(#[from] ActivationQuantError),
    #[error(transparent)]
    Fp8Linear(#[from] Fp8LinearError),
    /// Inside a device scope, a Metal FP8 projection failed.
    #[cfg(feature = "metal")]
    #[error("Metal FP8 projection failed: {0}")]
    Fp8Device(crate::precision::Fp8MetalError),
    #[error(transparent)]
    Norm(#[from] crate::RmsNormError),
    #[error(transparent)]
    Rotary(#[from] RotaryError),
    #[error(transparent)]
    Roundtrip(#[from] ActivationRoundtripError),
    #[error(transparent)]
    Window(#[from] WindowError),
    #[error(transparent)]
    SparseLayout(#[from] SparseAttentionError),
    #[error(transparent)]
    Sparse(#[from] SparseAttentionBf16Error),
    #[error(transparent)]
    Output(#[from] AttentionOutputError),
    #[error(transparent)]
    OutputLayout(#[from] AttentionOutputLayoutError),
    #[error("attention input length is {actual}; it must be a nonempty multiple of {stride}")]
    InputLength { actual: usize, stride: usize },
    #[error("decode requires exactly one position, got {positions}")]
    DecodeMustHaveOnePosition { positions: usize },
    #[error("attention position discontinuity: expected {expected:?}, got {actual}")]
    DiscontinuousPosition {
        expected: Option<usize>,
        actual: usize,
    },
    #[error("a window-only attention layout requires forward_window_only")]
    WindowOnlyLayoutRequiresWindowMethod,
    #[error("a compressed attention layout requires a compressed publication")]
    CompressedLayoutRequiresPublication,
    #[error("compressed publication source layer {actual} is not expected layer {expected}")]
    WrongSourceLayer { actual: u16, expected: u16 },
    #[error("compressed publication epoch {actual} is not expected epoch {expected}")]
    WrongEpoch { actual: u64, expected: u64 },
    #[error("compressed publication call {actual} is not expected call {expected}")]
    WrongCallId { actual: u64, expected: u64 },
    #[error("compressed numerical BF16 length is {actual}; it must be divisible by {stride}")]
    CompressedValueLength { actual: usize, stride: usize },
    #[error("compressed publication has {actual} keys; source shape requires {expected}")]
    CompressedKeyCount { actual: usize, expected: usize },
    #[error("compressed index length is {actual}; it must be divisible by {rows}")]
    CompressedIndexLength { actual: usize, rows: usize },
    #[error("compressed publication has no keys but supplies {slots} index slots per query")]
    CompressedSlotsWithoutKeys { slots: usize },
    #[error(
        "compressed index {index} at slot {slot} is not -1 or in window-plus-compressed range (window keys {window_keys}, compressed keys {compressed_keys})"
    )]
    InvalidCompressedIndex {
        slot: usize,
        index: i32,
        window_keys: usize,
        compressed_keys: usize,
    },
    #[error("compressed index {index} is duplicated in publication row {row}")]
    DuplicateCompressedIndex { row: usize, index: usize },
    #[error("compressed index {index} at slot {slot} is beyond causal compressed length {causal}")]
    FutureCompressedIndex {
        slot: usize,
        index: usize,
        causal: usize,
    },
    #[error("layer-attention shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error(
        "layer-attention {field} has {elements} elements, maximum is {MAX_LAYER_ATTENTION_ELEMENTS}"
    )]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    #[error("a sparse attention call needs at least one sparse slot")]
    NoSparseSlots,
    #[error("a sparse attention call needs at least one key")]
    NoKeys,
    #[error("epoch counter overflowed")]
    EpochOverflow,
    #[error("successful call counter overflowed")]
    CallIdOverflow,
    #[error("FP8 projection result was nonfinite after BF16 narrowing at element {element}")]
    NonFiniteProjection { element: usize },
    #[error("rotary result was nonfinite after BF16 narrowing at tail element {element}")]
    NonFiniteRotary { element: usize },
}

impl From<Fp8ForwardError> for LayerAttentionError {
    fn from(error: Fp8ForwardError) -> Self {
        match error {
            Fp8ForwardError::Scalar(error) => Self::Fp8Linear(error),
            #[cfg(feature = "metal")]
            Fp8ForwardError::Device(error) => Self::Fp8Device(error),
        }
    }
}
