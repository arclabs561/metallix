//! `FLUX.2 [klein]` text-to-image adapter.
//!
//! The pipeline is not a token decoder. A Qwen3 text encoder runs once per
//! prompt; a rectified-flow transformer (5 double-stream and 20 single-stream
//! blocks for klein 4B) predicts a velocity over packed image latents at each
//! scheduler step; a 2D VAE decodes the final latents to pixels.
//!
//! The text encoder reuses the `qwen` crate's dense decoder; only the layer
//! read-out and padding are klein-specific.
//!
//! Mechanism and numerics follow diffusers v0.41.0
//! (`models/transformers/transformer_flux2.py`,
//! `pipelines/flux2/pipeline_flux2_klein.py`).
//!
//! # Entry points
//!
//! [`ImageSize`] bounds the image grid; [`Flux2TransformerConfig`] reads the
//! transformer shape, and [`RopeTables`] constructs position tables on the CPU.
//! With `metal`, `KleinTextEncoder` embeds tokens, `Flux2Transformer` predicts
//! velocity, and `Flux2Pipeline` composes denoising and VAE decoding. Those
//! paths require local checkpoint files and Apple Silicon; the CPU types do not.
//! Component availability does not establish full image-pipeline qualification.
//!
//! # Example
//!
//! ```
//! let size = flux2::ImageSize::new(512, 256)?;
//! assert_eq!(size.grid(), (16, 32));
//! assert_eq!(size.tokens(), 512);
//! # Ok::<(), flux2::ImageSizeError>(())
//! ```

#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc)]

mod config;
pub use config::{ConfigError, Flux2TransformerConfig};
mod rope;
pub use rope::{RopeError, RopeTables, image_position_ids, text_position_ids};
mod size;
pub use size::{ImageSize, ImageSizeError, PIXELS_PER_TOKEN};

#[cfg(feature = "metal")]
mod transformer;
#[cfg(feature = "metal")]
pub use transformer::{Flux2Error, Flux2Precision, Flux2Trace, Flux2Transformer};
#[cfg(feature = "metal")]
mod vae;
#[cfg(feature = "metal")]
pub use vae::{Flux2VaeDecoder, to_rgb8};
#[cfg(feature = "metal")]
mod text_encoder;
#[cfg(feature = "metal")]
pub use text_encoder::{
    KleinTextEncoder, MAX_TEXT_TOKENS, PAD_TOKEN_ID, TEXT_LAYERS, TextEncoderError,
    klein_prompt_text,
};
#[cfg(feature = "metal")]
mod pipeline;
#[cfg(feature = "metal")]
pub use pipeline::{Flux2Pipeline, InitialNoise, PipelineError};
