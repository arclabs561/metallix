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
    /// A derived shape does not fit in `usize`.
    #[error("layer-attention shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// The buffer or derived count's role.
        field: &'static str,
    },
    /// The rotary tail is wider than one head.
    #[error("rope width {rope_width} exceeds head dimension {head_dimension}")]
    RopeExceedsHead {
        /// Rotary tail width.
        rope_width: usize,
        /// Head width.
        head_dimension: usize,
    },
    /// Heads cannot be split evenly into output groups.
    #[error("heads {heads} are not divisible by groups {groups}")]
    HeadsNotGrouped {
        /// Attention heads.
        heads: usize,
        /// Output groups.
        groups: usize,
    },
    /// An FP8 projection's reduction width is not a multiple of 32.
    #[error("{field} width {width} is not divisible by FP8 group 32")]
    UngroupedFp8Reduction {
        /// The projection's role.
        field: &'static str,
        /// Its reduction width.
        width: usize,
    },
    /// A staging buffer would pass the fixed element cap.
    #[error(
        "layer-attention {field} has {elements} elements, maximum is {MAX_LAYER_ATTENTION_ELEMENTS}"
    )]
    ElementLimit {
        /// The buffer or derived count's role.
        field: &'static str,
        /// Its element count.
        elements: usize,
    },
    /// The `RMSNorm` epsilon is not finite and positive.
    #[error("norm epsilon must be finite and positive")]
    InvalidNormEpsilon,
    /// The softmax scale is not finite and positive.
    #[error("softmax scale must be finite and positive")]
    InvalidSoftmaxScale,
}

/// A rejected layer-attention call. State is unchanged on every variant.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayerAttentionError {
    /// The layout is invalid.
    #[error(transparent)]
    Layout(#[from] LayerAttentionLayoutError),
    /// FP8 activation quantization rejected its input.
    #[error(transparent)]
    ActivationQuant(#[from] ActivationQuantError),
    /// A scalar FP8 projection rejected its input or overflowed.
    #[error(transparent)]
    Fp8Linear(#[from] Fp8LinearError),
    /// Inside a device scope, a Metal FP8 projection failed.
    #[cfg(feature = "metal")]
    #[error("Metal FP8 projection failed: {0}")]
    Fp8Device(crate::precision::Fp8MetalError),
    /// An `RMSNorm` stage rejected its input.
    #[error(transparent)]
    Norm(#[from] crate::RmsNormError),
    /// A rotary stage rejected its input.
    #[error(transparent)]
    Rotary(#[from] RotaryError),
    /// The FP8 activation round trip rejected its input.
    #[error(transparent)]
    Roundtrip(#[from] ActivationRoundtripError),
    /// The window ring step was invalid.
    #[error(transparent)]
    Window(#[from] WindowError),
    /// The sparse-attention layout was invalid.
    #[error(transparent)]
    SparseLayout(#[from] SparseAttentionError),
    /// Sparse attention rejected its input or overflowed.
    #[error(transparent)]
    Sparse(#[from] SparseAttentionBf16Error),
    /// The output projection failed.
    #[error(transparent)]
    Output(#[from] AttentionOutputError),
    /// The output-projection layout was invalid.
    #[error(transparent)]
    OutputLayout(#[from] AttentionOutputLayoutError),
    /// The input is empty or not a whole number of rows.
    #[error("attention input length is {actual}; it must be a nonempty multiple of {stride}")]
    InputLength {
        /// Supplied length.
        actual: usize,
        /// Elements per position.
        stride: usize,
    },
    /// A decode call supplied other than one position.
    #[error("decode requires exactly one position, got {positions}")]
    DecodeMustHaveOnePosition {
        /// Positions supplied.
        positions: usize,
    },
    /// The call does not start where the previous call ended.
    #[error("attention position discontinuity: expected {expected:?}, got {actual}")]
    DiscontinuousPosition {
        /// The next position, or `None` when no prefill has run.
        expected: Option<usize>,
        /// The call's start position.
        actual: usize,
    },
    /// A window-only layout was passed to `forward`.
    #[error("a window-only attention layout requires forward_window_only")]
    WindowOnlyLayoutRequiresWindowMethod,
    /// A compressed layout was passed to `forward_window_only`.
    #[error("a compressed attention layout requires a compressed publication")]
    CompressedLayoutRequiresPublication,
    /// The publication came from another layer.
    #[error("compressed publication source layer {actual} is not expected layer {expected}")]
    WrongSourceLayer {
        /// The publication's layer.
        actual: u16,
        /// The layout's expected layer.
        expected: u16,
    },
    /// The publication belongs to an earlier or later epoch.
    #[error("compressed publication epoch {actual} is not expected epoch {expected}")]
    WrongEpoch {
        /// The publication's epoch.
        actual: u64,
        /// The state's epoch for this call.
        expected: u64,
    },
    /// The publication belongs to another call.
    #[error("compressed publication call {actual} is not expected call {expected}")]
    WrongCallId {
        /// The publication's call ordinal.
        actual: u64,
        /// The next successful call.
        expected: u64,
    },
    /// The compressed values are not whole key rows.
    #[error("compressed numerical BF16 length is {actual}; it must be divisible by {stride}")]
    CompressedValueLength {
        /// Supplied length.
        actual: usize,
        /// Elements per key row.
        stride: usize,
    },
    /// The publication's key count differs from the causal prefix.
    #[error("compressed publication has {actual} keys; source shape requires {expected}")]
    CompressedKeyCount {
        /// Keys supplied.
        actual: usize,
        /// Keys the causal prefix requires.
        expected: usize,
    },
    /// The compressed indices are not whole query rows.
    #[error("compressed index length is {actual}; it must be divisible by {rows}")]
    CompressedIndexLength {
        /// Supplied length.
        actual: usize,
        /// Query rows.
        rows: usize,
    },
    /// Index slots were supplied with no compressed keys.
    #[error("compressed publication has no keys but supplies {slots} index slots per query")]
    CompressedSlotsWithoutKeys {
        /// Index slots per query.
        slots: usize,
    },
    /// An index is neither -1 nor inside the concatenated keys.
    #[error(
        "compressed index {index} at slot {slot} is not -1 or in window-plus-compressed range (window keys {window_keys}, compressed keys {compressed_keys})"
    )]
    InvalidCompressedIndex {
        /// Flat index slot.
        slot: usize,
        /// The index.
        index: i32,
        /// Window keys before the compressed ones.
        window_keys: usize,
        /// Compressed keys.
        compressed_keys: usize,
    },
    /// A row names the same compressed key twice.
    #[error("compressed index {index} is duplicated in publication row {row}")]
    DuplicateCompressedIndex {
        /// Query row.
        row: usize,
        /// The repeated index.
        index: usize,
    },
    /// An index points past the causal compressed length.
    #[error("compressed index {index} at slot {slot} is beyond causal compressed length {causal}")]
    FutureCompressedIndex {
        /// Flat index slot.
        slot: usize,
        /// The compressed index.
        index: usize,
        /// The row's causal compressed length.
        causal: usize,
    },
    /// A derived shape does not fit in `usize`.
    #[error("layer-attention shape arithmetic overflowed for {field}")]
    ShapeOverflow {
        /// The buffer or derived count's role.
        field: &'static str,
    },
    /// A staging buffer would pass the fixed element cap.
    #[error(
        "layer-attention {field} has {elements} elements, maximum is {MAX_LAYER_ATTENTION_ELEMENTS}"
    )]
    ElementLimit {
        /// The buffer or derived count's role.
        field: &'static str,
        /// Its element count.
        elements: usize,
    },
    /// The call has no sparse slots.
    #[error("a sparse attention call needs at least one sparse slot")]
    NoSparseSlots,
    /// The call has no keys.
    #[error("a sparse attention call needs at least one key")]
    NoKeys,
    /// The epoch counter overflowed.
    #[error("epoch counter overflowed")]
    EpochOverflow,
    /// The successful-call counter overflowed.
    #[error("successful call counter overflowed")]
    CallIdOverflow,
    /// An FP8 projection narrowed to a non-finite BF16.
    #[error("FP8 projection result was nonfinite after BF16 narrowing at element {element}")]
    NonFiniteProjection {
        /// Flat element index.
        element: usize,
    },
    /// A rotary result narrowed to a non-finite BF16.
    #[error("rotary result was nonfinite after BF16 narrowing at tail element {element}")]
    NonFiniteRotary {
        /// Flat index into the rotary tail.
        element: usize,
    },
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
