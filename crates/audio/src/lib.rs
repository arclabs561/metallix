//! Model-free audio input processing: WAV decoding and log-mel features.
//!
//! Nothing here depends on MLX, so these tests run without the `metal`
//! feature. Model crates own their encoders; this crate owns the steps that
//! must match a reference preprocessor exactly before any tensor reaches a
//! model.

#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc)]

mod mel;
mod wav;

pub use mel::{LogMel, LogMelError, LogMelExtractor, LogMelSpec};
pub use wav::{MonoAudio, WavError, WavInfo, decode_wav, decode_wav_limited, probe_wav};
