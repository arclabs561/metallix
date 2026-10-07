//! `AutoencoderKLFlux2` decoder and the latent post-processing before it.
//!
//! MLX convolutions are channels-last, so everything here is NHWC and conv
//! weights are transposed from `PyTorch`'s `[O, I, kH, kW]` to `[O, kH, kW, I]`
//! at load. The decoder is diffusers' `Decoder` with four `UpDecoderBlock2D`s
//! (three resnets each, nearest-2x upsampling on all but the last), a mid
//! block of resnet, single-head attention, resnet, and `GroupNorm(32)`
//! throughout.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mlx_rs::error::Exception;
use mlx_rs::fast;
use mlx_rs::ops;
use mlx_rs::{Array, Dtype};
use serde::Deserialize;

use crate::transformer::{Flux2Error, Flux2Precision};

const VAE_CLASS: &str = "AutoencoderKLFlux2";
const GROUPS: i32 = 32;
const EPS: f32 = 1e-6;

#[derive(Deserialize)]
struct RawVaeConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    block_out_channels: Vec<usize>,
    layers_per_block: usize,
    latent_channels: usize,
    norm_num_groups: usize,
    batch_norm_eps: f32,
    patch_size: [usize; 2],
    mid_block_add_attention: bool,
    use_post_quant_conv: bool,
}

/// A loaded VAE decoder and its latent normalization statistics.
pub struct Flux2VaeDecoder {
    weights: HashMap<String, Array>,
    dtype: Dtype,
    blocks: usize,
    resnets_per_block: usize,
    latent_channels: i32,
    batch_norm_eps: f32,
}

impl Flux2VaeDecoder {
    /// Loads a diffusers `vae/` directory (decoder, `post_quant_conv` and
    /// the latent `BatchNorm` statistics; the encoder is not loaded).
    ///
    /// # Errors
    ///
    /// Returns configuration or file-loading errors for an unsupported or unreadable
    /// checkpoint, a dimension error for oversized latent channels, or MLX errors.
    ///
    /// # Panics
    ///
    /// An excessive configured layer count can overflow its increment.
    pub fn load(dir: impl AsRef<Path>, precision: Flux2Precision) -> Result<Self, Flux2Error> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        let text = fs::read_to_string(&config_path).map_err(|source| Flux2Error::Read {
            path: config_path.clone(),
            source,
        })?;
        let raw: RawVaeConfig = serde_json::from_str(&text)
            .map_err(|error| Flux2Error::Config(crate::ConfigError::Json(error.to_string())))?;
        if raw.class_name != VAE_CLASS {
            return Err(crate::ConfigError::Class(raw.class_name).into());
        }
        if raw.norm_num_groups != 32
            || raw.patch_size != [2, 2]
            || !raw.mid_block_add_attention
            || !raw.use_post_quant_conv
        {
            return Err(crate::ConfigError::Unsupported("VAE layout other than FLUX.2's").into());
        }

        let path = dir.join("diffusion_pytorch_model.safetensors");
        let tensors = Array::load_safetensors(&path).map_err(|source| Flux2Error::Load {
            path: path.clone(),
            source,
        })?;
        let dtype = precision.dtype();
        let mut weights = HashMap::new();
        for (name, tensor) in tensors {
            if name.starts_with("encoder.")
                || name.starts_with("quant_conv.")
                || name == "bn.num_batches_tracked"
            {
                continue;
            }
            let tensor = tensor.as_dtype(dtype)?;
            // PyTorch conv weights are [O, I, kH, kW]; MLX wants [O, kH, kW, I].
            let tensor = if tensor.ndim() == 4 {
                tensor.transpose_axes(&[0, 2, 3, 1])?
            } else {
                tensor
            };
            weights.insert(name, tensor);
        }
        Ok(Self {
            weights,
            dtype,
            blocks: raw.block_out_channels.len(),
            resnets_per_block: raw.layers_per_block + 1,
            latent_channels: i32::try_from(raw.latent_channels)
                .map_err(|_| Flux2Error::Dimension(raw.latent_channels))?,
            batch_norm_eps: raw.batch_norm_eps,
        })
    }

    fn weight(&self, name: &str) -> Result<&Array, Flux2Error> {
        self.weights
            .get(name)
            .ok_or_else(|| Flux2Error::MissingWeight(name.to_owned()))
    }

    /// Turns the transformer's packed output `[B, h * w, 4 * latent_channels]`
    /// (row-major over an `h x w` grid) into the decoder input
    /// `[B, 2h, 2w, latent_channels]`: undo the latent `BatchNorm`, then split
    /// each token's channels into a 2x2 pixel patch.
    ///
    /// # Errors
    ///
    /// Returns missing-weight errors for absent normalization statistics, or MLX
    /// errors for incompatible packed shapes and tensor operations.
    ///
    /// # Panics
    ///
    /// Panics for scalar `packed` input or when doubled dimensions or packed
    /// channel counts overflow with overflow checks enabled.
    pub fn latents_to_decoder_input(
        &self,
        packed: &Array,
        height: i32,
        width: i32,
    ) -> Result<Array, Flux2Error> {
        let batch = packed.shape()[0];
        let channels = 4 * self.latent_channels;
        let grid = packed
            .as_dtype(self.dtype)?
            .reshape(&[batch, height, width, channels])?;
        let eps = Array::from_f32(self.batch_norm_eps).as_dtype(self.dtype)?;
        let std = self.weight("bn.running_var")?.add(&eps)?.sqrt()?;
        let mean = self.weight("bn.running_mean")?;
        let grid = grid.multiply(&std)?.add(mean)?;
        // Channel index is c * 4 + dy * 2 + dx (diffusers `_unpatchify_latents`).
        Ok(grid
            .reshape(&[batch, height, width, self.latent_channels, 2, 2])?
            .transpose_axes(&[0, 1, 4, 2, 5, 3])?
            .reshape(&[batch, 2 * height, 2 * width, self.latent_channels])?)
    }

    /// Decodes `[B, H, W, latent_channels]` to `[B, 8H, 8W, 3]` in `[-1, 1]`
    /// (not clamped).
    ///
    /// # Errors
    ///
    /// Returns missing-weight errors or MLX errors for incompatible shapes and
    /// failed decoder operations.
    ///
    /// # Panics
    ///
    /// Malformed rank or excessive spatial dimensions can panic in shape handling.
    pub fn decode(&self, latents: &Array) -> Result<Array, Flux2Error> {
        let mut x = self.conv(&latents.as_dtype(self.dtype)?, "post_quant_conv", 0)?;
        x = self.conv(&x, "decoder.conv_in", 1)?;
        x = self.resnet(&x, "decoder.mid_block.resnets.0")?;
        x = self.attention(&x, "decoder.mid_block.attentions.0")?;
        x = self.resnet(&x, "decoder.mid_block.resnets.1")?;
        for block in 0..self.blocks {
            for resnet in 0..self.resnets_per_block {
                x = self.resnet(&x, &format!("decoder.up_blocks.{block}.resnets.{resnet}"))?;
            }
            if block + 1 < self.blocks {
                x = self.conv(
                    &upsample_nearest_2x(&x)?,
                    &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
                    1,
                )?;
            }
        }
        x = silu(&self.group_norm(&x, "decoder.conv_norm_out")?)?;
        self.conv(&x, "decoder.conv_out", 1)
    }

    fn conv(&self, input: &Array, name: &str, padding: i32) -> Result<Array, Flux2Error> {
        let out = ops::conv2d(
            input,
            self.weight(&format!("{name}.weight"))?,
            None,
            (padding, padding),
            None,
            None,
        )?;
        Ok(out.add(self.weight(&format!("{name}.bias"))?)?)
    }

    /// `GroupNorm(32, eps=1e-6)` over NHWC, statistics in f32.
    fn group_norm(&self, input: &Array, name: &str) -> Result<Array, Flux2Error> {
        let shape = input.shape().to_vec();
        let (batch, channels) = (shape[0], shape[3]);
        let grouped =
            input
                .as_dtype(Dtype::Float32)?
                .reshape(&[batch, -1, GROUPS, channels / GROUPS])?;
        let mean = grouped.mean_axes(&[1, 3], true)?;
        let variance = grouped.var_axes(&[1, 3], true, 0)?;
        let normed = grouped
            .subtract(&mean)?
            .multiply(variance.add(Array::from_f32(EPS))?.rsqrt()?)?
            .reshape(&shape)?
            .as_dtype(self.dtype)?;
        Ok(normed
            .multiply(self.weight(&format!("{name}.weight"))?)?
            .add(self.weight(&format!("{name}.bias"))?)?)
    }

    /// `ResnetBlock2D` with the default time-embedding-free path.
    fn resnet(&self, input: &Array, name: &str) -> Result<Array, Flux2Error> {
        let h = silu(&self.group_norm(input, &format!("{name}.norm1"))?)?;
        let h = self.conv(&h, &format!("{name}.conv1"), 1)?;
        let h = silu(&self.group_norm(&h, &format!("{name}.norm2"))?)?;
        let h = self.conv(&h, &format!("{name}.conv2"), 1)?;
        let shortcut_name = format!("{name}.conv_shortcut");
        let shortcut = if self
            .weights
            .contains_key(&format!("{shortcut_name}.weight"))
        {
            self.conv(input, &shortcut_name, 0)?
        } else {
            input.clone()
        };
        Ok(shortcut.add(&h)?)
    }

    /// The mid block's single-head spatial self-attention with a residual.
    fn attention(&self, input: &Array, name: &str) -> Result<Array, Flux2Error> {
        let shape = input.shape().to_vec();
        let (batch, channels) = (shape[0], shape[3]);
        let normed = self.group_norm(input, &format!("{name}.group_norm"))?;
        let tokens = normed.reshape(&[batch, -1, channels])?;
        let project = |part: &str| -> Result<Array, Flux2Error> {
            let weight = self.weight(&format!("{name}.{part}.weight"))?;
            let bias = self.weight(&format!("{name}.{part}.bias"))?;
            Ok(tokens.matmul(weight.transpose()?)?.add(bias)?)
        };
        let heads = |x: Array| x.reshape(&[batch, 1, -1, channels]);
        let q = heads(project("to_q")?)?;
        let k = heads(project("to_k")?)?;
        let v = heads(project("to_v")?)?;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (channels as f32).sqrt();
        let attended = fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?
            .reshape(&[batch, -1, channels])?;
        let out_weight = self.weight(&format!("{name}.to_out.0.weight"))?;
        let out_bias = self.weight(&format!("{name}.to_out.0.bias"))?;
        let out = attended
            .matmul(out_weight.transpose()?)?
            .add(out_bias)?
            .reshape(&shape)?;
        Ok(out.add(input)?)
    }
}

fn silu(input: &Array) -> Result<Array, Exception> {
    ops::sigmoid(input)?.multiply(input)
}

fn upsample_nearest_2x(input: &Array) -> Result<Array, Exception> {
    let shape = input.shape().to_vec();
    let (b, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let expanded = input.reshape(&[b, h, 1, w, 1, c])?;
    ops::broadcast_to(&expanded, &[b, h, 2, w, 2, c])?.reshape(&[b, 2 * h, 2 * w, c])
}

/// diffusers' image post-processing to 8-bit RGB: `(x / 2 + 0.5)` clamped to
/// `[0, 1]` in the decoder's dtype, then `round(x * 255)` in f32 with ties to
/// even (`numpy.round`). Returns `[B, H, W, 3]` row-major bytes.
///
/// # Errors
///
/// Returns an MLX error if conversion, arithmetic or evaluation fails.
/// The function preserves the input layout without validating that it is RGB.
pub fn to_rgb8(image: &Array) -> Result<Vec<u8>, Exception> {
    let dtype = image.dtype();
    let half = Array::from_f32(0.5).as_dtype(dtype)?;
    let two = Array::from_f32(2.0).as_dtype(dtype)?;
    let shifted = image.divide(&two)?.add(&half)?;
    let unit = ops::minimum(
        ops::maximum(&shifted, Array::from_f32(0.0).as_dtype(dtype)?)?,
        Array::from_f32(1.0).as_dtype(dtype)?,
    )?
    .as_dtype(Dtype::Float32)?;
    unit.eval()?;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bytes = unit
        .as_slice::<f32>()
        .iter()
        .map(|&v| (v * 255.0).round_ties_even() as u8)
        .collect();
    Ok(bytes)
}

#[cfg(test)]
mod tests;
