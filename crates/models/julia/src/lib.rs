//! Bounded CPU reference for the pinned Julia decision head.

pub mod encoder;
pub mod head;

pub use encoder::{
    ENCODER_FF_WIDTH, EncoderBlock, EncoderBlockInput, EncoderBlockWeights, JuliaEncoderError,
};

pub use head::{
    ATTENTION_HEADS, DecisionHead, FEED_FORWARD_WIDTH, HEAD_LAYERS, HeadInput, HeadLayerWeights,
    HeadWeights, INVALID_MARKER_SCORE, JuliaHeadError, ScorerWeights, WIDTH,
};
