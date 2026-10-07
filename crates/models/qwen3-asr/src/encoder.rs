//! The Qwen3-ASR audio encoder on MLX.
//!
//! Mechanism, from `Qwen3ASRAudioEncoder.forward` in
//! <https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py>:
//!
//! 1. The mel features are cut into 100-frame chunks and the chunks are
//!    zero-padded to the longest one. The padding is not cosmetic: the
//!    second and third convolutions read the first convolution's output at
//!    padded positions, so a short final chunk run alone would differ.
//! 2. Three 3x3 stride-2 convolutions with GELU, then a bias-free linear over
//!    the flattened (channel, frequency) axis.
//! 3. A sinusoid position table restarting at 0 in every chunk.
//! 4. The valid frames of all chunks are concatenated, and Whisper-style
//!    pre-norm layers attend within windows of `window_frames` frames.
//! 5. A final layer norm and a two-layer GELU projection to the decoder's
//!    width.
//!
//! Windows are run as a batch, so attention never spans a window boundary
//! and no mask over the whole clip is built. The final window is padded and
//! its padded keys are masked.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the qwen crate defers the same migration"
)]

use std::collections::HashMap;

use mlx_rs::{Array, Dtype, StreamOrDevice, fast, ops, ops::indexing::IndexOp};

use crate::config::{AudioEncoderConfig, CHUNK_FRAMES, MEL_BINS};
use crate::lengths::{chunk_lengths, encoder_frames};

const LAYER_NORM_EPS: f32 = 1e-5;
/// Prefix of encoder tensors in released checkpoints.
pub const TENSOR_PREFIX: &str = "thinker.audio_tower.";

/// Encoder weights, keyed without [`TENSOR_PREFIX`]. Convolution kernels are
/// stored in MLX's `[out, height, width, in]` layout.
pub struct AudioEncoder {
    config: AudioEncoderConfig,
    tensors: HashMap<String, Array>,
    /// The sinusoid table, `[max_source_positions, d_model]`, f32.
    positions: Array,
}

impl std::fmt::Debug for AudioEncoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AudioEncoder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl AudioEncoder {
    /// Takes the encoder tensors out of a loaded checkpoint map, checking
    /// every name and shape the forward pass reads.
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError`] for a missing or misshapen tensor.
    pub fn from_tensors(
        config: AudioEncoderConfig,
        checkpoint: &mut HashMap<String, Array>,
    ) -> Result<Self, EncoderError> {
        let mut tensors = HashMap::new();
        for (name, shape) in required_tensors(&config) {
            let tensor = checkpoint
                .remove(&format!("{TENSOR_PREFIX}{name}"))
                .ok_or_else(|| EncoderError::MissingTensor(name.clone()))?;
            if tensor.shape() != shape.as_slice() {
                return Err(EncoderError::TensorShape {
                    name,
                    expected: shape,
                    actual: tensor.shape().to_vec(),
                });
            }
            let tensor = if name.starts_with("conv2d") && name.ends_with(".weight") {
                // PyTorch `[out, in, kh, kw]` to MLX `[out, kh, kw, in]`.
                tensor.transpose_axes_device(&[0, 2, 3, 1], StreamOrDevice::gpu())?
            } else {
                tensor
            };
            tensors.insert(name, tensor);
        }
        if let Some(extra) = checkpoint
            .keys()
            .find(|name| name.starts_with(TENSOR_PREFIX))
        {
            return Err(EncoderError::UnexpectedTensor(extra.clone()));
        }
        let positions = sinusoids(config.max_source_positions, config.d_model)?;
        Ok(Self {
            config,
            tensors,
            positions,
        })
    }

    /// Converts every tensor to `dtype` (f32 for parity checks).
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError::Mlx`] if MLX fails.
    pub fn convert(&mut self, dtype: Dtype) -> Result<(), EncoderError> {
        for tensor in self.tensors.values_mut() {
            let converted = tensor.as_dtype_device(dtype, StreamOrDevice::gpu())?;
            converted.eval()?;
            *tensor = converted;
        }
        Ok(())
    }

    /// The dtype the encoder computes in.
    #[must_use]
    pub fn dtype(&self) -> Dtype {
        self.tensors
            .get("conv_out.weight")
            .map_or(Dtype::Float32, Array::dtype)
    }

    /// Logical bytes of the encoder tensors.
    #[must_use]
    pub fn weight_bytes(&self) -> usize {
        self.tensors.values().map(Array::nbytes).sum()
    }

    /// Encodes one clip's `[128, frames]` bin-major log-mel features into
    /// `[encoder_frames(frames), output_dim]` rows in the encoder's dtype.
    ///
    /// The features are cast to the encoder's dtype first, as the reference
    /// casts `input_features` to the model dtype.
    ///
    /// # Errors
    ///
    /// Returns [`EncoderError`] for an empty or misshapen input, or if MLX
    /// fails.
    pub fn encode(&self, mel: &audio::LogMel) -> Result<Array, EncoderError> {
        self.encode_with(mel, Layout::Reference)
    }

    /// [`Self::encode`] with one deliberate deviation from the reference,
    /// for negative controls: a parity gate must fail on these.
    ///
    /// # Errors
    ///
    /// As [`Self::encode`]; [`EncoderControl::ContinuousPositions`] also
    /// fails when the clip has more frames than the sinusoid table.
    #[cfg(feature = "parity-controls")]
    pub fn encode_control(
        &self,
        mel: &audio::LogMel,
        control: EncoderControl,
    ) -> Result<Array, EncoderError> {
        self.encode_with(
            mel,
            match control {
                EncoderControl::ContinuousPositions => Layout::ContinuousPositions,
                EncoderControl::FullAttention => Layout::FullAttention,
            },
        )
    }

    fn encode_with(&self, mel: &audio::LogMel, layout: Layout) -> Result<Array, EncoderError> {
        let stream = StreamOrDevice::gpu();
        if mel.bins() != MEL_BINS || mel.frames() == 0 {
            return Err(EncoderError::Input {
                bins: mel.bins(),
                frames: mel.frames(),
            });
        }
        let dtype = self.dtype();
        let chunks = chunk_lengths(mel.frames());
        let hidden = self.convolve_chunks(mel, &chunks)?;
        let time = hidden.shape()[1];

        // Keep each chunk's valid frames, in order. Adding positions to the
        // kept rows equals the reference's add before masking, elementwise.
        let mut valid = Vec::with_capacity(encoder_frames(mel.frames()));
        let mut position = Vec::with_capacity(valid.capacity());
        let time_usize = usize::try_from(time).map_err(|_| EncoderError::Shape)?;
        for (chunk, &length) in chunks.iter().enumerate() {
            for frame in 0..encoder_frames(length) {
                valid.push(as_i32(chunk * time_usize + frame)?);
                position.push(match layout {
                    Layout::ContinuousPositions => as_i32(position.len())?,
                    Layout::Reference | Layout::FullAttention => as_i32(frame)?,
                });
            }
        }
        let frames = valid.len();
        if matches!(layout, Layout::ContinuousPositions)
            && frames > self.config.max_source_positions
        {
            return Err(EncoderError::Shape);
        }
        let d_model = as_i32(self.config.d_model)?;
        let positions = self
            .positions
            .take_axis_device(Array::from_slice(&position, &[as_i32(frames)?]), 0, &stream)?
            .as_dtype_device(dtype, &stream)?;
        let flat = hidden
            .reshape_device(&[-1, d_model], &stream)?
            .take_axis_device(Array::from_slice(&valid, &[as_i32(frames)?]), 0, &stream)?
            .add_device(&positions, &stream)?;

        // Lay the frames out as [windows, window, d_model]; the last window is
        // padded by repeating frame 0, and those keys are masked.
        let window = match layout {
            Layout::FullAttention => frames,
            Layout::Reference | Layout::ContinuousPositions => self.config.window_frames(),
        };
        let windows = frames.div_ceil(window);
        let padded = windows * window;
        let mut layout_rows: Vec<i32> = (0..as_i32(frames)?).collect();
        layout_rows.resize(padded, 0);
        let mut hidden = flat
            .take_axis_device(
                Array::from_slice(&layout_rows, &[as_i32(padded)?]),
                0,
                &stream,
            )?
            .reshape_device(&[as_i32(windows)?, as_i32(window)?, d_model], &stream)?;
        let key_mask = if padded == frames {
            None
        } else {
            let keys: Vec<bool> = (0..padded).map(|index| index < frames).collect();
            Some(Array::from_slice(
                &keys,
                &[as_i32(windows)?, 1, 1, as_i32(window)?],
            ))
        };

        for layer in 0..self.config.layers {
            hidden = self.layer(layer, &hidden, key_mask.as_ref())?;
        }
        hidden = layer_norm(
            &hidden,
            self.tensor("ln_post.weight")?,
            self.tensor("ln_post.bias")?,
        )?;
        hidden = gelu(&linear(
            &hidden,
            self.tensor("proj1.weight")?,
            Some(self.tensor("proj1.bias")?),
        )?)?;
        hidden = linear(
            &hidden,
            self.tensor("proj2.weight")?,
            Some(self.tensor("proj2.bias")?),
        )?;
        Ok(hidden
            .reshape_device(&[as_i32(padded)?, as_i32(self.config.output_dim)?], &stream)?
            .index_device((0..as_i32(frames)?, ..), &stream))
    }

    /// Zero-pads mel chunks, applies the three convolutions and projects
    /// each chunk's time rows before valid-frame selection.
    fn convolve_chunks(
        &self,
        mel: &audio::LogMel,
        chunks: &[usize],
    ) -> Result<Array, EncoderError> {
        let stream = StreamOrDevice::gpu();
        let dtype = self.dtype();
        let longest = chunks.iter().copied().max().unwrap_or(0);
        let chunk_count = chunks.len();

        // [chunks, 128, longest, 1]: NHWC with frequency as height. A short
        // final chunk is zero-padded to the longest chunk, as the reference
        // does with `pad_sequence(chunk_list, batch_first=True)`
        // (modeling_qwen3_asr.py line 694 at 7c6daf7). This is a parity
        // invariant: conv2 and conv3 read conv1 outputs at padded positions,
        // so running the final chunk unpadded changes its encoder frames.
        let mut host = vec![0.0_f32; chunk_count * MEL_BINS * longest];
        for (chunk, &length) in chunks.iter().enumerate() {
            for bin in 0..MEL_BINS {
                let source = bin * mel.frames() + chunk * CHUNK_FRAMES;
                let target = (chunk * MEL_BINS + bin) * longest;
                host[target..target + length]
                    .copy_from_slice(&mel.values()[source..source + length]);
            }
        }
        let mut hidden = Array::from_slice(
            &host,
            &[as_i32(chunk_count)?, as_i32(MEL_BINS)?, as_i32(longest)?, 1],
        )
        .as_dtype_device(dtype, &stream)?;
        for conv in ["conv2d1", "conv2d2", "conv2d3"] {
            let convolved = ops::conv2d_device(
                &hidden,
                self.tensor(&format!("{conv}.weight"))?,
                (2, 2),
                (1, 1),
                (1, 1),
                1,
                &stream,
            )?
            .add_device(self.tensor(&format!("{conv}.bias"))?, &stream)?;
            hidden = gelu(&convolved)?;
        }
        // [chunks, 16, time, channels] -> [chunks, time, channels * 16], the
        // reference's `permute(0, 3, 1, 2).view(b, t, c * f)`.
        let shape = hidden.shape().to_vec();
        let (frequency, time, channels) = (shape[1], shape[2], shape[3]);
        hidden = hidden
            .transpose_axes_device(&[0, 2, 3, 1], &stream)?
            .reshape_device(&[as_i32(chunk_count)?, time, channels * frequency], &stream)?;
        linear(&hidden, self.tensor("conv_out.weight")?, None)
    }

    fn layer(
        &self,
        layer: usize,
        hidden: &Array,
        key_mask: Option<&Array>,
    ) -> Result<Array, EncoderError> {
        let stream = StreamOrDevice::gpu();
        let base = format!("layers.{layer}");
        let shape = hidden.shape().to_vec();
        let (windows, window) = (shape[0], shape[1]);
        let heads = as_i32(self.config.heads)?;
        let head_dim = as_i32(self.config.d_model / self.config.heads)?;

        let normed = layer_norm(
            hidden,
            self.tensor(&format!("{base}.self_attn_layer_norm.weight"))?,
            self.tensor(&format!("{base}.self_attn_layer_norm.bias"))?,
        )?;
        let project = |name: &str| -> Result<Array, EncoderError> {
            Ok(linear(
                &normed,
                self.tensor(&format!("{base}.self_attn.{name}.weight"))?,
                Some(self.tensor(&format!("{base}.self_attn.{name}.bias"))?),
            )?
            .reshape_device(&[windows, window, heads, head_dim], &stream)?
            .transpose_axes_device(&[0, 2, 1, 3], &stream)?)
        };
        let (query, key, value) = (project("q_proj")?, project("k_proj")?, project("v_proj")?);
        #[allow(
            clippy::cast_precision_loss,
            reason = "head dimensions are small integers"
        )]
        let scale = (head_dim as f32).powf(-0.5);
        let attended = fast::scaled_dot_product_attention_device(
            &query,
            &key,
            &value,
            scale,
            key_mask.map(fast::ScaledDotProductAttentionMask::Array),
            Option::<&Array>::None,
            &stream,
        )?
        .transpose_axes_device(&[0, 2, 1, 3], &stream)?
        .reshape_device(&[windows, window, heads * head_dim], &stream)?;
        let attended = linear(
            &attended,
            self.tensor(&format!("{base}.self_attn.out_proj.weight"))?,
            Some(self.tensor(&format!("{base}.self_attn.out_proj.bias"))?),
        )?;
        let residual = hidden.add_device(&attended, &stream)?;

        let normed = layer_norm(
            &residual,
            self.tensor(&format!("{base}.final_layer_norm.weight"))?,
            self.tensor(&format!("{base}.final_layer_norm.bias"))?,
        )?;
        let expanded = gelu(&linear(
            &normed,
            self.tensor(&format!("{base}.fc1.weight"))?,
            Some(self.tensor(&format!("{base}.fc1.bias"))?),
        )?)?;
        let contracted = linear(
            &expanded,
            self.tensor(&format!("{base}.fc2.weight"))?,
            Some(self.tensor(&format!("{base}.fc2.bias"))?),
        )?;
        Ok(residual.add_device(&contracted, &stream)?)
    }

    fn tensor(&self, name: &str) -> Result<&Array, EncoderError> {
        self.tensors
            .get(name)
            .ok_or_else(|| EncoderError::MissingTensor(name.to_owned()))
    }
}

/// How an encode deviates from the reference; only the reference layout is
/// reachable outside negative controls.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(feature = "parity-controls"),
    allow(
        dead_code,
        reason = "the deviations are built only for negative controls"
    )
)]
enum Layout {
    Reference,
    ContinuousPositions,
    FullAttention,
}

/// Deliberate deviations from the reference encoder, for negative controls.
#[cfg(feature = "parity-controls")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderControl {
    /// Positions count across the clip instead of restarting each chunk.
    ContinuousPositions,
    /// One attention window over the whole clip instead of 8 s windows.
    FullAttention,
}

fn required_tensors(config: &AudioEncoderConfig) -> Vec<(String, Vec<i32>)> {
    let to = |value: usize| i32::try_from(value).unwrap_or(i32::MAX);
    let (d, ffn, channels) = (
        to(config.d_model),
        to(config.ffn_dim),
        to(config.conv_channels),
    );
    let rows = to(AudioEncoderConfig::conv_frequency_rows());
    let mut tensors = vec![
        ("conv2d1.weight".to_owned(), vec![channels, 1, 3, 3]),
        ("conv2d1.bias".to_owned(), vec![channels]),
        ("conv2d2.weight".to_owned(), vec![channels, channels, 3, 3]),
        ("conv2d2.bias".to_owned(), vec![channels]),
        ("conv2d3.weight".to_owned(), vec![channels, channels, 3, 3]),
        ("conv2d3.bias".to_owned(), vec![channels]),
        ("conv_out.weight".to_owned(), vec![d, channels * rows]),
        ("ln_post.weight".to_owned(), vec![d]),
        ("ln_post.bias".to_owned(), vec![d]),
        ("proj1.weight".to_owned(), vec![d, d]),
        ("proj1.bias".to_owned(), vec![d]),
        ("proj2.weight".to_owned(), vec![to(config.output_dim), d]),
        ("proj2.bias".to_owned(), vec![to(config.output_dim)]),
    ];
    for layer in 0..config.layers {
        let base = format!("layers.{layer}");
        for projection in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            tensors.push((format!("{base}.self_attn.{projection}.weight"), vec![d, d]));
            tensors.push((format!("{base}.self_attn.{projection}.bias"), vec![d]));
        }
        for norm in ["self_attn_layer_norm", "final_layer_norm"] {
            tensors.push((format!("{base}.{norm}.weight"), vec![d]));
            tensors.push((format!("{base}.{norm}.bias"), vec![d]));
        }
        tensors.push((format!("{base}.fc1.weight"), vec![ffn, d]));
        tensors.push((format!("{base}.fc1.bias"), vec![ffn]));
        tensors.push((format!("{base}.fc2.weight"), vec![d, ffn]));
        tensors.push((format!("{base}.fc2.bias"), vec![d]));
    }
    tensors
}

/// `SinusoidsPositionEmbedding`: `[sin(p t_i), cos(p t_i)]` with
/// `t_i = exp(-i ln(10000) / (channels / 2 - 1))`, computed in f32 as the
/// reference's float tensors are.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "position tables are small"
)]
fn sinusoids(length: usize, channels: usize) -> Result<Array, EncoderError> {
    let half = channels / 2;
    let increment = 10_000_f64.ln() / (half - 1) as f64;
    let inverse: Vec<f32> = (0..half)
        .map(|index| (-(increment as f32) * index as f32).exp())
        .collect();
    let mut table = vec![0.0_f32; length * channels];
    for position in 0..length {
        for (index, &rate) in inverse.iter().enumerate() {
            let angle = position as f32 * rate;
            table[position * channels + index] = angle.sin();
            table[position * channels + half + index] = angle.cos();
        }
    }
    Ok(Array::from_slice(
        &table,
        &[as_i32(length)?, as_i32(channels)?],
    ))
}

/// Exact (erf) GELU, evaluated in f32 and returned in the input dtype, as
/// torch evaluates reduced-precision GELU.
fn gelu(input: &Array) -> Result<Array, EncoderError> {
    let stream = StreamOrDevice::gpu();
    let dtype = input.dtype();
    let wide = input.as_dtype_device(Dtype::Float32, &stream)?;
    let scaled = wide.multiply_device(Array::from_f32(std::f32::consts::FRAC_1_SQRT_2), &stream)?;
    let gate = ops::erf_device(&scaled, &stream)?
        .add_device(Array::from_f32(1.0), &stream)?
        .multiply_device(Array::from_f32(0.5), &stream)?;
    Ok(wide
        .multiply_device(&gate, &stream)?
        .as_dtype_device(dtype, &stream)?)
}

fn layer_norm(input: &Array, weight: &Array, bias: &Array) -> Result<Array, EncoderError> {
    Ok(fast::layer_norm_device(
        input,
        weight,
        bias,
        LAYER_NORM_EPS,
        StreamOrDevice::gpu(),
    )?)
}

fn linear(input: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array, EncoderError> {
    let stream = StreamOrDevice::gpu();
    let output = input.matmul_device(weight.transpose_device(&stream)?, &stream)?;
    Ok(match bias {
        Some(bias) => output.add_device(bias, &stream)?,
        None => output,
    })
}

fn as_i32(value: usize) -> Result<i32, EncoderError> {
    i32::try_from(value).map_err(|_| EncoderError::Shape)
}

/// Errors while loading or running the encoder.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum EncoderError {
    /// A required tensor is absent.
    #[error("encoder tensor {0} is missing")]
    MissingTensor(String),
    /// A checkpoint tensor under the encoder prefix is not used.
    #[error("unexpected encoder tensor {0}")]
    UnexpectedTensor(String),
    /// A tensor has the wrong shape.
    #[error("encoder tensor {name} has shape {actual:?}; expected {expected:?}")]
    TensorShape {
        /// Tensor name without the prefix.
        name: String,
        /// Shape the forward pass needs.
        expected: Vec<i32>,
        /// Shape in the checkpoint.
        actual: Vec<i32>,
    },
    /// The features are empty or have the wrong bin count.
    #[error("encoder input has {bins} bins and {frames} frames")]
    Input {
        /// Bins in the supplied features.
        bins: usize,
        /// Frames in the supplied features.
        frames: usize,
    },
    /// A dimension does not fit MLX's i32 shapes.
    #[error("encoder shape overflow")]
    Shape,
    /// MLX reported an error.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
}
