//! Bounded CPU reference for the pinned Julia decision head.

pub mod head;

pub use head::{
    ATTENTION_HEADS, DecisionHead, FEED_FORWARD_WIDTH, HEAD_LAYERS, HeadInput, HeadLayerWeights,
    HeadWeights, INVALID_MARKER_SCORE, JuliaHeadError, ScorerWeights, WIDTH,
};
