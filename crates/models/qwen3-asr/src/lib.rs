//! Qwen3-ASR speech recognition: a windowed audio encoder whose output rows
//! replace audio placeholder tokens in a Qwen3 decoder prompt.
//!
//! Configuration, prompt and output handling are model-free and build
//! without MLX; the encoder and the decoder splice need the `metal` feature.
//! The loader implements dense checkpoints only; declared packed/quantized
//! formats are rejected before tokenizer or weight files are read.

#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc)]

pub mod config;
#[cfg(feature = "metal")]
pub mod encoder;
pub mod languages;
pub mod lengths;
#[cfg(feature = "metal")]
pub mod model;
pub mod output;
pub mod tokenizer;

pub use config::{AudioEncoderConfig, AudioTokens, Qwen3AsrConfig, Qwen3AsrConfigError};
pub use output::{Transcript, parse_output};
pub use tokenizer::{AsrPrompt, AsrTokenizer, AsrTokenizerError};

#[cfg(feature = "metal")]
pub use model::{AsrError, Qwen3Asr, TranscribeOptions, TranscribeResult};
