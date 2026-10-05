//! Schedule and request errors.

use thiserror::Error;

use crate::{
    attention::layer::LayerAttentionError,
    reduced::{
        AttentionInputError, BlockTailError, EngramSessionError, FinalHeadError,
        LayerFourSessionError, LayerOneSessionError, LayerThreeSessionError, StartupSessionError,
    },
};

/// Why a layer schedule was rejected before request allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum ScheduleError {
    /// More layers than a `u16` source-layer identity can name.
    #[error("schedule has more layers than source identities")]
    LayerCount,
    /// A layer names an Engram definition the model does not hold.
    #[error("Engram index {index} is outside {available} definitions")]
    EngramIndex { index: usize, available: usize },
    /// A window-only layer was given a compressed attention layout.
    #[error("window-only layer has a compressed attention layout")]
    WindowLayoutCompressed,
    /// A compressed kind was given a window-only attention layout.
    #[error("compressed layer has a window-only attention layout")]
    MissingCompression,
    /// The attention layout's ratio disagrees with the layer kind.
    #[error("layer kind needs compression ratio {expected}, layout has {actual}")]
    CompressionRatio { expected: usize, actual: usize },
    /// An owner does not publish under its own layer number.
    #[error("owner must publish as layer {expected}, layout names {actual}")]
    OwnerSource { expected: u16, actual: u16 },
    /// A consumer or indexer has no matching latest preceding owner.
    #[error("no latest preceding ratio-{ratio} owner at source layer {source_layer}")]
    MissingProducer { source_layer: u16, ratio: usize },
    /// A ratio-two layer follows a ratio-one owner.
    #[error("ratio-two layers must precede ratio-one layers")]
    RatioOrder,
    /// Ratio-two owners have no ratio-one keys for incomplete groups.
    #[error("ratio-two owners need a ratio-one owner for incomplete groups")]
    MissingRatioOneOwner,
}

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RequestError {
    /// Block tails, startup, or head disagree on copy or hidden width.
    #[error("request blocks do not share copy/hidden geometry")]
    BlockGeometry,
    /// L2 does not consume the fixed L1 ratio-two publication geometry.
    #[error("reused L2 attention must be batch-one source-one ratio-two with block hidden width")]
    LayerTwoGeometry,
    /// One attention layout disagrees on batch, hidden width, or `RoPE` pairs.
    #[error("request attention layouts do not share batch-one hidden/rotary geometry")]
    AttentionGeometry,
    /// The layer schedule is not a valid sequence of producers and consumers.
    #[error("schedule layer {layer}: {reason}")]
    Schedule { layer: usize, reason: ScheduleError },
    /// A per-layer expert-source slice did not have one entry per model layer.
    #[error("request needs {expected} per-layer expert sources, got {actual}")]
    ExpertSourceCount { expected: usize, actual: usize },
    /// A per-Engram row-source slice did not have one entry per Engram definition.
    #[error("request needs {expected} per-Engram row sources, got {actual}")]
    EngramRowSourceCount { expected: usize, actual: usize },
    /// A requested static or dynamic request surface exceeds its bound.
    #[error("request has {elements} tokens beyond the bounded maximum")]
    ElementLimit { elements: usize },
    /// A prior admitted stage failed after possibly advancing inner state.
    #[error("request is poisoned; restart is required")]
    Poisoned,
    /// The caller supplied an empty token chunk.
    #[error("request needs at least one token ID")]
    EmptyIds,
    /// A post-prefill call included more than one decode token.
    #[error("request decode accepts one token, got {actual}")]
    DecodeChunk { actual: usize },
    /// Absolute token arithmetic overflowed before state mutation.
    #[error("request token position overflowed")]
    PositionOverflow,
    /// The requested call would exceed the immutable request token limit.
    #[error("request end {end} exceeds configured maximum {maximum}")]
    TokenLimit { end: usize, maximum: usize },
    /// The model's full rotary table is too short for a requested span.
    #[error("request needs {required} rotary-frequency elements, table has {available}")]
    FrequencyTable { required: usize, available: usize },
    /// Startup accepts unsigned IDs and rejected a negative input ID.
    #[error("request token ID {id} cannot convert to startup unsigned form")]
    NegativeToken { id: i64 },
    /// An incomplete ratio-two group had no previous-step ratio-one owner keys.
    #[error("a partial ratio-two owner call needs an earlier successful ratio-one publication")]
    MissingPriorLayerThree,
    /// A composed buffer did not match its exact stage geometry.
    #[error("request buffer {field} has {actual} elements, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    /// A bounded owned stage buffer could not be allocated.
    #[error("request allocation failed for {field} with {elements} elements")]
    Allocation {
        field: &'static str,
        elements: usize,
    },
    /// Startup execution rejected its live input or operands.
    #[error(transparent)]
    Startup(#[from] StartupSessionError),
    /// An Engram state rejected its live stream.
    #[error(transparent)]
    Engram(#[from] EngramSessionError),
    /// An HC-collapse or RMS-normalization boundary failed.
    #[error(transparent)]
    Input(#[from] AttentionInputError),
    /// A ratio-two owner, its scoring, or its attention failed.
    #[error(transparent)]
    LayerOne(#[from] LayerOneSessionError),
    /// Window-only or reused-publication consumer attention failed.
    #[error(transparent)]
    LayerTwo(#[from] LayerAttentionError),
    /// A ratio-one owner, candidate, selection, or attention failed.
    #[error(transparent)]
    LayerThree(#[from] LayerThreeSessionError),
    /// A candidate indexer's query, selection, or attention failed.
    #[error(transparent)]
    LayerFour(#[from] LayerFourSessionError),
    /// A block tail failed.
    #[error(transparent)]
    Tail(#[from] BlockTailError),
    /// Final normalization or vocabulary projection failed.
    #[error(transparent)]
    Head(#[from] FinalHeadError),
}
