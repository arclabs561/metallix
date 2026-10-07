//! The klein text-to-image loop: noise, `steps` transformer passes with Euler
//! updates along the flow-matching schedule, then VAE decode to RGB8.
//!
//! Prompt encoding is the caller's: the loop takes the `[1, T, 3 * hidden]`
//! embedding (Qwen3 hidden states at layers 9, 18 and 27), so it runs and is
//! gated without the text encoder loaded.

use std::fs;
use std::path::Path;

use diffusion::{FlowMatchConfig, Sigmas, flux2_empirical_mu};
use mlx_rs::{Array, Dtype};
use serde::Deserialize;
use thiserror::Error;

use crate::rope::{RopeTables, image_position_ids, text_position_ids};
use crate::size::ImageSize;
use crate::transformer::{Flux2Error, Flux2Precision, Flux2Transformer};
use crate::vae::{Flux2VaeDecoder, to_rgb8};

/// Where the starting latents come from.
pub enum InitialNoise {
    /// Standard normal noise from MLX's generator keyed by `seed`, drawn in
    /// diffusers' unpacked layout `[1, 4 * latent_channels, rows, columns]`.
    Seed(u64),
    /// Caller-supplied latents in that same layout (parity gates inject the
    /// reference's noise this way).
    Latents(Array),
}

#[derive(Deserialize)]
struct ModelIndex {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(default)]
    is_distilled: bool,
}

/// A pipeline checkpoint, schedule or latent operation failed.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// A transformer or VAE component failed.
    #[error(transparent)]
    Model(#[from] Flux2Error),
    /// Rotary table dimensions or storage are invalid.
    #[error(transparent)]
    Rope(#[from] crate::RopeError),
    /// The flow-matching schedule was rejected.
    #[error(transparent)]
    Schedule(#[from] diffusion::ScheduleError),
    /// A pipeline file could not be read or parsed.
    #[error("cannot read {path}: {message}")]
    Read {
        /// Input file or component path.
        path: String,
        /// Read or parse diagnostic.
        message: String,
    },
    /// The checkpoint declares a different pipeline class.
    #[error("pipeline class {0:?} is not Flux2KleinPipeline")]
    Class(String),
    /// The checkpoint requires classifier-free guidance, which is unsupported.
    #[error("only step-distilled klein checkpoints run without guidance; this one needs CFG")]
    NotDistilled,
    /// Supplied initial latents do not match the requested grid.
    #[error("initial latents have shape {actual:?}, expected {expected:?}")]
    LatentShape {
        /// Required dimensions in axis order.
        expected: Vec<i32>,
        /// Observed dimensions in axis order.
        actual: Vec<i32>,
    },
    /// An MLX tensor or random-generation operation failed.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
}

/// Loaded denoising and VAE components; text embeddings are supplied separately.
pub struct Flux2Pipeline {
    transformer: Flux2Transformer,
    vae: Flux2VaeDecoder,
    schedule: FlowMatchConfig,
    dtype: Dtype,
}

fn read(path: &Path) -> Result<String, PipelineError> {
    fs::read_to_string(path).map_err(|error| PipelineError::Read {
        path: path.display().to_string(),
        message: error.to_string(),
    })
}

impl Flux2Pipeline {
    /// Loads a diffusers `Flux2KleinPipeline` directory (the text encoder is
    /// not loaded here).
    ///
    /// # Errors
    ///
    /// Returns read/parse errors, [`PipelineError::Class`] for another pipeline,
    /// or [`PipelineError::NotDistilled`] for a checkpoint needing guidance.
    /// Component-loading and scheduler errors are propagated.
    ///
    /// # Panics
    ///
    /// Inherits the component loaders' malformed-dimension preconditions.
    pub fn load(
        model_dir: impl AsRef<Path>,
        precision: Flux2Precision,
    ) -> Result<Self, PipelineError> {
        let dir = model_dir.as_ref();
        let index: ModelIndex = serde_json::from_str(&read(&dir.join("model_index.json"))?)
            .map_err(|error| PipelineError::Read {
                path: "model_index.json".into(),
                message: error.to_string(),
            })?;
        if index.class_name != "Flux2KleinPipeline" {
            return Err(PipelineError::Class(index.class_name));
        }
        if !index.is_distilled {
            return Err(PipelineError::NotDistilled);
        }
        let schedule =
            FlowMatchConfig::from_json(&read(&dir.join("scheduler/scheduler_config.json"))?)?;
        Ok(Self {
            transformer: Flux2Transformer::load(dir.join("transformer"), precision)?,
            vae: Flux2VaeDecoder::load(dir.join("vae"), precision)?,
            schedule,
            dtype: precision.dtype(),
        })
    }

    fn latent_shape(&self, size: ImageSize) -> Result<Vec<i32>, PipelineError> {
        let (rows, columns) = size.grid();
        let channels = i32::try_from(self.transformer.config().in_channels)
            .map_err(|_| Flux2Error::Dimension(0))?;
        Ok(vec![1, channels, rows.cast_signed(), columns.cast_signed()])
    }

    /// The starting latents in the unpacked layout, float32.
    ///
    /// # Errors
    ///
    /// Returns a dimension error for unrepresentable channels,
    /// [`PipelineError::LatentShape`] for an incompatible supplied tensor, or
    /// MLX errors from random generation and conversion.
    pub fn initial_latents(
        &self,
        noise: InitialNoise,
        size: ImageSize,
    ) -> Result<Array, PipelineError> {
        let shape = self.latent_shape(size)?;
        let latents = match noise {
            InitialNoise::Seed(seed) => {
                let key = mlx_rs::random::key(seed)?;
                mlx_rs::random::normal::<f32>(shape.as_slice(), None, None, &key)?
            }
            InitialNoise::Latents(latents) => latents,
        };
        if latents.shape() != shape.as_slice() {
            return Err(PipelineError::LatentShape {
                expected: shape,
                actual: latents.shape().to_vec(),
            });
        }
        Ok(latents.as_dtype(Dtype::Float32)?)
    }

    /// Runs the denoising loop and returns the final packed latents
    /// `[1, tokens, in_channels]`. `on_step` sees each step's updated latents.
    ///
    /// # Errors
    ///
    /// Propagates invalid schedule, rotary-table, component input and tensor-operation errors.
    /// Earlier `on_step` calls remain observable if a later step fails.
    ///
    /// # Panics
    ///
    /// Panics if `text` or `initial` has fewer than two axes, or if the
    /// callback panics.
    pub fn denoise(
        &self,
        text: &Array,
        initial: &Array,
        size: ImageSize,
        steps: usize,
        mut on_step: impl FnMut(usize, &Array),
    ) -> Result<Array, PipelineError> {
        let mu = flux2_empirical_mu(size.tokens(), steps);
        let sigmas = Sigmas::new(&self.schedule, steps, Some(mu))?;

        let (rows, columns) = size.grid();
        let text_len = usize::try_from(text.shape()[1]).map_err(|_| Flux2Error::Dimension(0))?;
        let mut ids = text_position_ids(text_len);
        ids.extend(image_position_ids(rows as usize, columns as usize));
        let config = self.transformer.config();
        let rope = RopeTables::new(&ids, config.axes_dims_rope, config.rope_theta)?;

        // [1, C, rows, columns] -> [1, rows * columns, C], row-major tokens;
        // latents then live in the activation dtype, as in the source.
        let channels = initial.shape()[1];
        let mut latents = initial
            .reshape(&[1, channels, -1])?
            .transpose_axes(&[0, 2, 1])?
            .as_dtype(self.dtype)?;

        for step in sigmas.steps() {
            let velocity = self
                .transformer
                .forward(&latents, text, step.timestep, &rope)?;
            // FlowMatchEulerDiscreteScheduler.step: x + dt * v, with dt a
            // float32 scalar; the product rounds to the latent dtype before
            // the add.
            let update = velocity
                .as_dtype(Dtype::Float32)?
                .multiply(Array::from_f32(step.dt()))?
                .as_dtype(self.dtype)?;
            latents = latents.add(&update)?;
            latents.eval()?;
            on_step(step.index, &latents);
        }
        Ok(latents)
    }

    /// Decodes final packed latents to `height x width` RGB8 bytes.
    ///
    /// # Errors
    ///
    /// Propagates latent reshaping, VAE decoding and pixel-conversion failures.
    ///
    /// # Panics
    ///
    /// Inherits [`crate::Flux2VaeDecoder::latents_to_decoder_input`] shape preconditions.
    pub fn decode_rgb8(&self, packed: &Array, size: ImageSize) -> Result<Vec<u8>, PipelineError> {
        let (rows, columns) = size.grid();
        let input =
            self.vae
                .latents_to_decoder_input(packed, rows.cast_signed(), columns.cast_signed())?;
        let image = self.vae.decode(&input)?;
        Ok(to_rgb8(&image)?)
    }
}

#[cfg(test)]
mod tests;
