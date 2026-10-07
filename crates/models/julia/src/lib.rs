//! Bounded CPU reference for the pinned Julia decision model: checkpoint
//! loading, typed request encoding, encoder and decision head.
//!
//! Julia-1 answers typed questions about a state by scoring each option, not
//! by generating text. The pinned source is
//! [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1/tree/a85b127321d580d65176c89ced8273f305745d85).
//! Everything here is scalar F32 on the CPU, checked against that source.
//!
//! # Overview
//!
//! One question flows through these steps:
//!
//! 1. [`typed::parse_typed_request`] reads the request into one
//!    [`typed::TypedRow`] per question.
//! 2. [`typed::sequence`] serializes a row with the caller's tokenizer into
//!    token IDs and option-marker positions, rejecting any input the
//!    source's strict policy would clean or truncate.
//! 3. [`JuliaCheckpoint::load`] validates and loads the published weights;
//!    [`JuliaCheckpoint::encoder`] builds a [`JuliaEncoder`] holding only the
//!    embedding rows a sequence uses.
//! 4. [`JuliaEncoder::forward`] runs the 22-layer `ModernBERT` encoder, and
//!    [`DecisionHead::scores`] turns its output into one score per option.
//! 5. [`typed::typed_answer`] applies the option softmax and the question
//!    type's readout.
//!
//! The encoder and head accept at most 126 positions, so a serialized
//! question longer than that is refused even though the source's policy
//! allows [`typed::MAX_LENGTH`] tokens.
//!
//! # Example: reading out an answer
//!
//! ```
//! use julia::typed::{parse_typed_request, typed_answer};
//!
//! let request = br#"{"state": "a closed door",
//!     "questions": {"open": {"type": "noul", "instructions": "Is it open?"}}}"#;
//! let parsed = parse_typed_request(request)?;
//! let row = &parsed.rows[0];
//! assert_eq!(row.keys, ["false", "true"]);
//!
//! // Equal scores for both options give the true option probability 0.5.
//! let answer = typed_answer(row, &[1.0, 1.0])?;
//! assert_eq!(answer["noul"], 0.5);
//! # Ok::<(), julia::typed::JuliaRequestError>(())
//! ```

#![deny(missing_docs)]
// The workspace allows this lint; crates opt in once their docs are complete.
#![warn(clippy::missing_errors_doc)]

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
