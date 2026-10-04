//! Bounded CPU reference for the pinned Julia decision model: checkpoint
//! loading, typed request encoding, encoder and decision head.

pub mod checkpoint;
pub mod encoder;
pub mod head;
#[cfg(test)]
mod prefill;
pub mod typed;

pub use encoder::{
    ENCODER_FF_WIDTH, EncoderBlock, EncoderBlockInput, EncoderBlockWeights, EncoderInput,
    FullEncoderWeights, JuliaEncoder, JuliaEncoderError,
};

pub use head::{
    ATTENTION_HEADS, DecisionHead, FEED_FORWARD_WIDTH, HEAD_LAYERS, HeadInput, HeadLayerWeights,
    HeadWeights, INVALID_MARKER_SCORE, JuliaHeadError, ScorerWeights, WIDTH,
};

pub use checkpoint::{JuliaCheckpoint, JuliaCheckpointError};
