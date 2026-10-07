//! `Flux2Transformer2DModel` on MLX.
//!
//! Operations run on MLX's default stream (the GPU). Scalars that meet
//! activations are built in the activation dtype first: an MLX `f32` array
//! would otherwise promote a bf16 graph to f32, which the source never does.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use mlx_rs::error::{Exception, IoError};
use mlx_rs::fast;
use mlx_rs::ops;
use mlx_rs::{Array, Dtype};
use thiserror::Error;

use crate::config::{ConfigError, Flux2TransformerConfig};
use crate::rope::RopeTables;

/// Compute precision for weights and activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flux2Precision {
    /// Weights upcast to f32; the parity reference precision.
    F32,
    /// The checkpoint's own bf16 weights and bf16 activations.
    Bf16,
}

impl Flux2Precision {
    ///
    /// Returns the MLX scalar type used for weights and activations.
    #[must_use]
    pub fn dtype(self) -> Dtype {
        match self {
            Self::F32 => Dtype::Float32,
            Self::Bf16 => Dtype::Bfloat16,
        }
    }
}

/// A transformer checkpoint, input shape or MLX operation failed.
#[derive(Debug, Error)]
pub enum Flux2Error {
    /// The transformer configuration was rejected.
    #[error(transparent)]
    Config(#[from] ConfigError),
    /// A local file or directory could not be read.
    #[error("cannot read {path}: {source}")]
    Read {
        /// Local filesystem path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// A safetensors shard could not be loaded.
    #[error("cannot load safetensors {path}: {source}")]
    Load {
        /// Local shard path.
        /// Local filesystem path.
        path: PathBuf,
        /// Safetensors loading failure.
        source: IoError,
    },
    /// The checkpoint directory contains no safetensors shards.
    #[error("no safetensors files in {0}")]
    NoWeights(PathBuf),
    /// A required tensor name is absent.
    #[error("weight {0} is missing")]
    MissingWeight(String),
    /// A checkpoint tensor has an unexpected shape.
    #[error("weight {name} has shape {actual:?}, expected {expected:?}")]
    WeightShape {
        /// Checkpoint tensor name.
        name: String,
        /// Required dimensions in axis order.
        expected: Vec<i32>,
        /// Observed dimensions in axis order.
        actual: Vec<i32>,
    },
    /// A supplied activation or position table has an unexpected shape.
    #[error("input {name} has shape {actual:?}, expected {expected}")]
    InputShape {
        /// Input role.
        name: &'static str,
        /// Required layout and dimensions.
        expected: String,
        /// Observed dimensions in axis order.
        actual: Vec<i32>,
    },
    /// A dimension cannot be represented by an MLX shape.
    #[error("dimension {0} does not fit an MLX shape")]
    Dimension(usize),
    /// An MLX tensor operation failed.
    #[error(transparent)]
    Mlx(#[from] Exception),
}

/// Intermediate outputs captured for component parity.
pub struct Flux2Trace {
    /// Text stream after double block 0, `[B, T, D]`.
    pub double0_text: Array,
    /// Image stream after double block 0, `[B, S, D]`.
    pub double0_image: Array,
    /// Joint `[text, image]` stream after single block 0, `[B, T + S, D]`.
    pub single0: Array,
    /// Velocity prediction for the image tokens, `[B, S, in_channels]`.
    pub velocity: Array,
}

/// A loaded transformer with weights in one compute precision.
pub struct Flux2Transformer {
    config: Flux2TransformerConfig,
    weights: HashMap<String, Array>,
    dtype: Dtype,
}

fn dim(value: usize) -> Result<i32, Flux2Error> {
    i32::try_from(value).map_err(|_| Flux2Error::Dimension(value))
}

impl Flux2Transformer {
    /// Loads `config.json` and every `*.safetensors` file in a diffusers
    /// `transformer/` directory, converting weights to `precision`.
    ///
    /// # Errors
    ///
    /// Returns configuration, filesystem, missing-weight or shape errors when the
    /// checkpoint cannot load, and MLX errors when weight conversion fails.
    #[tracing::instrument(name = "flux2.transformer.load", level = "info", skip_all, fields(dir = %dir.as_ref().display()))]
    pub fn load(dir: impl AsRef<Path>, precision: Flux2Precision) -> Result<Self, Flux2Error> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        let text = fs::read_to_string(&config_path).map_err(|source| Flux2Error::Read {
            path: config_path,
            source,
        })?;
        let config = Flux2TransformerConfig::from_json(&text)?;

        let mut shards: Vec<PathBuf> = fs::read_dir(dir)
            .map_err(|source| Flux2Error::Read {
                path: dir.to_path_buf(),
                source,
            })?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(Flux2Error::NoWeights(dir.to_path_buf()));
        }

        let dtype = precision.dtype();
        let mut weights = HashMap::new();
        for shard in shards {
            let tensors = Array::load_safetensors(&shard).map_err(|source| Flux2Error::Load {
                path: shard.clone(),
                source,
            })?;
            for (name, tensor) in tensors {
                let tensor = if tensor.dtype() == dtype {
                    tensor
                } else {
                    tensor.as_dtype(dtype)?
                };
                weights.insert(name, tensor);
            }
        }
        let model = Self {
            config,
            weights,
            dtype,
        };
        model.check_weights()?;
        Ok(model)
    }

    ///
    /// Borrows the loaded transformer configuration.
    #[must_use]
    pub fn config(&self) -> &Flux2TransformerConfig {
        &self.config
    }

    fn check_weights(&self) -> Result<(), Flux2Error> {
        let c = &self.config;
        let d = dim(c.inner_dim())?;
        let mlp = dim(c.mlp_hidden)?;
        let mut expected: Vec<(String, Vec<i32>)> = vec![
            ("x_embedder.weight".into(), vec![d, dim(c.in_channels)?]),
            (
                "context_embedder.weight".into(),
                vec![d, dim(c.joint_attention_dim)?],
            ),
            ("proj_out.weight".into(), vec![dim(c.in_channels)?, d]),
            ("norm_out.linear.weight".into(), vec![2 * d, d]),
            (
                "double_stream_modulation_img.linear.weight".into(),
                vec![6 * d, d],
            ),
            (
                "double_stream_modulation_txt.linear.weight".into(),
                vec![6 * d, d],
            ),
            (
                "single_stream_modulation.linear.weight".into(),
                vec![3 * d, d],
            ),
            (
                "time_guidance_embed.timestep_embedder.linear_1.weight".into(),
                vec![d, dim(c.timestep_channels)?],
            ),
            (
                "time_guidance_embed.timestep_embedder.linear_2.weight".into(),
                vec![d, d],
            ),
        ];
        for block in 0..c.double_blocks {
            let base = format!("transformer_blocks.{block}");
            for name in [
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_add_out",
                "to_out.0",
            ] {
                expected.push((format!("{base}.attn.{name}.weight"), vec![d, d]));
            }
            for ff in ["ff", "ff_context"] {
                expected.push((format!("{base}.{ff}.linear_in.weight"), vec![2 * mlp, d]));
                expected.push((format!("{base}.{ff}.linear_out.weight"), vec![d, mlp]));
            }
        }
        for block in 0..c.single_blocks {
            let base = format!("single_transformer_blocks.{block}.attn");
            expected.push((
                format!("{base}.to_qkv_mlp_proj.weight"),
                vec![3 * d + 2 * mlp, d],
            ));
            expected.push((format!("{base}.to_out.weight"), vec![d, d + mlp]));
        }
        for (name, shape) in expected {
            let actual = self.weight(&name)?.shape();
            if actual != shape.as_slice() {
                return Err(Flux2Error::WeightShape {
                    name,
                    expected: shape,
                    actual: actual.to_vec(),
                });
            }
        }
        Ok(())
    }

    fn weight(&self, name: &str) -> Result<&Array, Flux2Error> {
        self.weights
            .get(name)
            .ok_or_else(|| Flux2Error::MissingWeight(name.to_owned()))
    }

    fn linear(&self, input: &Array, name: &str) -> Result<Array, Flux2Error> {
        let weight = self.weight(&format!("{name}.weight"))?;
        Ok(input.matmul(weight.transpose()?)?)
    }

    fn scalar(&self, value: f32) -> Result<Array, Flux2Error> {
        Ok(Array::from_f32(value).as_dtype(self.dtype)?)
    }

    /// Predicts the flow velocity for packed image latents.
    ///
    /// `latents` is `[B, S, in_channels]`, `text` is the prompt embedding
    /// `[B, T, joint_attention_dim]`, `timestep` is the scheduler's
    /// `sigma * 1000`, and `rope` covers the `T + S` text-then-image ids.
    ///
    /// # Errors
    ///
    /// Returns input-shape or dimension errors for incompatible arrays and rotary
    /// tables, missing-weight errors for absent tensors, or MLX operation errors.
    ///
    /// # Panics
    ///
    /// Excessive combined sequence dimensions can overflow shape arithmetic.
    pub fn forward(
        &self,
        latents: &Array,
        text: &Array,
        timestep: f32,
        rope: &RopeTables,
    ) -> Result<Array, Flux2Error> {
        Ok(self.run(latents, text, timestep, rope, false)?.velocity)
    }

    /// [`Self::forward`], also returning the block outputs the parity gate
    /// compares.
    ///
    /// # Errors
    ///
    /// Propagates the same failures as [`Self::forward`].
    ///
    /// # Panics
    ///
    /// Has the same shape preconditions as [`Self::forward`].
    pub fn forward_traced(
        &self,
        latents: &Array,
        text: &Array,
        timestep: f32,
        rope: &RopeTables,
    ) -> Result<Flux2Trace, Flux2Error> {
        self.run(latents, text, timestep, rope, true)
    }

    fn run(
        &self,
        latents: &Array,
        text: &Array,
        timestep: f32,
        rope: &RopeTables,
        trace: bool,
    ) -> Result<Flux2Trace, Flux2Error> {
        let c = &self.config;
        let (batch, image_len) = Self::check_input("latents", latents, c.in_channels)?;
        let (text_batch, text_len) = Self::check_input("text", text, c.joint_attention_dim)?;
        if text_batch != batch {
            return Err(Flux2Error::InputShape {
                name: "text",
                expected: format!("batch {batch}"),
                actual: text.shape().to_vec(),
            });
        }
        let positions =
            usize::try_from(text_len + image_len).map_err(|_| Flux2Error::Dimension(0))?;
        if rope.positions() != positions || rope.head_dim() != c.head_dim {
            return Err(Flux2Error::InputShape {
                name: "rope",
                expected: format!("{positions} positions x {}", c.head_dim),
                actual: vec![dim(rope.positions())?, dim(rope.head_dim())?],
            });
        }
        let latents = latents.as_dtype(self.dtype)?;
        let text = text.as_dtype(self.dtype)?;

        let temb = self.timestep_embedding(timestep, batch)?;
        let mod_img = self.modulation(&temb, "double_stream_modulation_img", 6)?;
        let mod_txt = self.modulation(&temb, "double_stream_modulation_txt", 6)?;
        let mod_single = self.modulation(&temb, "single_stream_modulation", 3)?;

        let head_dim = dim(c.head_dim)?;
        let cos = Array::from_slice(rope.cos(), &[1, text_len + image_len, 1, head_dim]);
        let sin = Array::from_slice(rope.sin(), &[1, text_len + image_len, 1, head_dim]);

        let mut image = self.linear(&latents, "x_embedder")?;
        let mut context = self.linear(&text, "context_embedder")?;

        let mut double0 = None;
        for block in 0..c.double_blocks {
            (context, image) =
                self.double_block(block, &image, &context, &mod_img, &mod_txt, &cos, &sin)?;
            if trace && block == 0 {
                double0 = Some((context.clone(), image.clone()));
            }
        }

        let mut joint = ops::concatenate(&[&context, &image], 1)?;
        let mut single0 = None;
        for block in 0..c.single_blocks {
            joint = self.single_block(block, &joint, &mod_single, &cos, &sin)?;
            if trace && block == 0 {
                single0 = Some(joint.clone());
            }
        }

        let image = ops::split_at_indices(&joint, &[text_len], 1)?.swap_remove(1);
        let velocity = self.output(&image, &temb)?;
        let (double0_text, double0_image) =
            double0.unwrap_or_else(|| (velocity.clone(), velocity.clone()));
        Ok(Flux2Trace {
            double0_text,
            double0_image,
            single0: single0.unwrap_or_else(|| velocity.clone()),
            velocity,
        })
    }

    fn check_input(
        name: &'static str,
        input: &Array,
        width: usize,
    ) -> Result<(i32, i32), Flux2Error> {
        let shape = input.shape();
        if shape.len() != 3 || shape[2] != dim(width)? || shape[0] < 1 || shape[1] < 1 {
            return Err(Flux2Error::InputShape {
                name,
                expected: format!("[B, L, {width}]"),
                actual: shape.to_vec(),
            });
        }
        Ok((shape[0], shape[1]))
    }

    /// `Flux2TimestepGuidanceEmbeddings` without guidance.
    ///
    /// The pipeline hands the model `t / 1000` in the activation dtype and the
    /// model multiplies by 1000 again in that dtype, so in bf16 the embedded
    /// timestep is rounded twice before the f32 sinusoid.
    fn timestep_embedding(&self, timestep: f32, batch: i32) -> Result<Array, Flux2Error> {
        let thousand = self.scalar(1000.0)?;
        let t = self
            .scalar(timestep)?
            .divide(&thousand)?
            .multiply(&thousand)?
            .as_dtype(Dtype::Float32)?;

        let half = self.config.timestep_channels / 2;
        // get_timestep_embedding: exponent = -ln(10000) * k / half, in f32.
        #[allow(clippy::cast_possible_truncation)]
        let log_period = -(10000.0_f64.ln()) as f32;
        #[allow(clippy::cast_precision_loss)]
        let frequencies: Vec<f32> = (0..half)
            .map(|k| (log_period * k as f32 / half as f32).exp())
            .collect();
        let angles = Array::from_slice(&frequencies, &[1, dim(half)?]).multiply(&t)?;
        // flip_sin_to_cos: [cos, sin].
        let projected =
            ops::concatenate(&[angles.cos()?, angles.sin()?], 1)?.as_dtype(self.dtype)?;
        let projected =
            ops::broadcast_to(&projected, &[batch, dim(self.config.timestep_channels)?])?;

        let hidden = self.linear(&projected, "time_guidance_embed.timestep_embedder.linear_1")?;
        self.linear(
            &silu(&hidden)?,
            "time_guidance_embed.timestep_embedder.linear_2",
        )
    }

    /// `Flux2Modulation`: `chunks` shift/scale/gate vectors, each `[B, 1, D]`.
    fn modulation(&self, temb: &Array, name: &str, chunks: i32) -> Result<Vec<Array>, Flux2Error> {
        let projected = self.linear(&silu(temb)?, &format!("{name}.linear"))?;
        let batch = projected.shape()[0];
        let projected = projected.reshape(&[batch, 1, -1])?;
        Ok(projected.split_equal(chunks, -1)?)
    }

    fn modulate(&self, input: &Array, shift: &Array, scale: &Array) -> Result<Array, Flux2Error> {
        let normed = fast::layer_norm(input, None, None, self.config.eps)?;
        // (1 + scale) * x + shift, in the activation dtype.
        let one = self.scalar(1.0)?;
        Ok(one.add(scale)?.multiply(&normed)?.add(shift)?)
    }

    fn heads(&self, input: &Array, norm: &str) -> Result<Array, Flux2Error> {
        let shape = input.shape();
        let split = input.reshape(&[
            shape[0],
            shape[1],
            dim(self.config.heads)?,
            dim(self.config.head_dim)?,
        ])?;
        Ok(fast::rms_norm(
            &split,
            Some(self.weight(&format!("{norm}.weight"))?),
            self.config.eps,
        )?)
    }

    /// Joint attention over `[B, L, H, Dh]` queries, keys and values whose
    /// sequence is text then image; returns `[B, L, H * Dh]`.
    fn attend(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array, Flux2Error> {
        let q = rotate(q, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = rotate(k, cos, sin)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 / (self.config.head_dim as f32).sqrt();
        let attended = fast::scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let shape = attended.shape().to_vec();
        Ok(attended.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            shape[0],
            shape[2],
            shape[1] * shape[3],
        ])?)
    }

    #[allow(clippy::too_many_arguments)]
    fn double_block(
        &self,
        block: usize,
        image: &Array,
        context: &Array,
        mod_img: &[Array],
        mod_txt: &[Array],
        cos: &Array,
        sin: &Array,
    ) -> Result<(Array, Array), Flux2Error> {
        let base = format!("transformer_blocks.{block}");
        let attn = format!("{base}.attn");
        let [shift, scale, gate, shift_mlp, scale_mlp, gate_mlp] = mod_img else {
            unreachable!("modulation returns six chunks")
        };
        let [
            c_shift,
            c_scale,
            c_gate,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ] = mod_txt
        else {
            unreachable!("modulation returns six chunks")
        };

        let normed_image = self.modulate(image, shift, scale)?;
        let normed_context = self.modulate(context, c_shift, c_scale)?;

        let q = self.heads(
            &self.linear(&normed_image, &format!("{attn}.to_q"))?,
            &format!("{attn}.norm_q"),
        )?;
        let k = self.heads(
            &self.linear(&normed_image, &format!("{attn}.to_k"))?,
            &format!("{attn}.norm_k"),
        )?;
        let v = self.split_heads(&self.linear(&normed_image, &format!("{attn}.to_v"))?)?;
        let cq = self.heads(
            &self.linear(&normed_context, &format!("{attn}.add_q_proj"))?,
            &format!("{attn}.norm_added_q"),
        )?;
        let ck = self.heads(
            &self.linear(&normed_context, &format!("{attn}.add_k_proj"))?,
            &format!("{attn}.norm_added_k"),
        )?;
        let cv = self.split_heads(&self.linear(&normed_context, &format!("{attn}.add_v_proj"))?)?;

        let attended = self.attend(
            &ops::concatenate(&[&cq, &q], 1)?,
            &ops::concatenate(&[&ck, &k], 1)?,
            &ops::concatenate(&[&cv, &v], 1)?,
            cos,
            sin,
        )?;
        let text_len = context.shape()[1];
        let mut parts = ops::split_at_indices(&attended, &[text_len], 1)?;
        let image_attn = self.linear(&parts.swap_remove(1), &format!("{attn}.to_out.0"))?;
        let context_attn = self.linear(&parts.swap_remove(0), &format!("{attn}.to_add_out"))?;

        let image = image.add(gate.multiply(&image_attn)?)?;
        let ff = self.feed_forward(
            &self.modulate(&image, shift_mlp, scale_mlp)?,
            &format!("{base}.ff"),
        )?;
        let image = image.add(gate_mlp.multiply(&ff)?)?;

        let context = context.add(c_gate.multiply(&context_attn)?)?;
        let ff = self.feed_forward(
            &self.modulate(&context, c_shift_mlp, c_scale_mlp)?,
            &format!("{base}.ff_context"),
        )?;
        let context = context.add(c_gate_mlp.multiply(&ff)?)?;
        Ok((context, image))
    }

    fn split_heads(&self, input: &Array) -> Result<Array, Flux2Error> {
        let shape = input.shape();
        Ok(input.reshape(&[
            shape[0],
            shape[1],
            dim(self.config.heads)?,
            dim(self.config.head_dim)?,
        ])?)
    }

    fn feed_forward(&self, input: &Array, name: &str) -> Result<Array, Flux2Error> {
        let hidden = self.linear(input, &format!("{name}.linear_in"))?;
        self.linear(&swiglu(&hidden)?, &format!("{name}.linear_out"))
    }

    /// `Flux2SingleTransformerBlock`: attention and MLP in parallel from one
    /// fused input projection, joined by one fused output projection.
    fn single_block(
        &self,
        block: usize,
        joint: &Array,
        modulation: &[Array],
        cos: &Array,
        sin: &Array,
    ) -> Result<Array, Flux2Error> {
        let attn = format!("single_transformer_blocks.{block}.attn");
        let [shift, scale, gate] = modulation else {
            unreachable!("modulation returns three chunks")
        };
        let normed = self.modulate(joint, shift, scale)?;
        let projected = self.linear(&normed, &format!("{attn}.to_qkv_mlp_proj"))?;
        let inner = dim(self.config.inner_dim())?;
        let mut parts = ops::split_at_indices(&projected, &[inner, 2 * inner, 3 * inner], -1)?;
        let mlp = parts.pop().expect("four parts");
        let v = self.split_heads(&parts.pop().expect("v"))?;
        let k = self.heads(&parts.pop().expect("k"), &format!("{attn}.norm_k"))?;
        let q = self.heads(&parts.pop().expect("q"), &format!("{attn}.norm_q"))?;
        let attended = self.attend(&q, &k, &v, cos, sin)?;
        let joined = ops::concatenate(&[attended, swiglu(&mlp)?], -1)?;
        let out = self.linear(&joined, &format!("{attn}.to_out"))?;
        Ok(joint.add(gate.multiply(&out)?)?)
    }

    /// `AdaLayerNormContinuous` then `proj_out`. Note the chunk order here is
    /// scale then shift, unlike the blocks' shift/scale/gate.
    fn output(&self, image: &Array, temb: &Array) -> Result<Array, Flux2Error> {
        let projected = self.linear(&silu(temb)?, "norm_out.linear")?;
        let batch = projected.shape()[0];
        let mut halves = projected.reshape(&[batch, 1, -1])?.split_equal(2, -1)?;
        let shift = halves.pop().expect("shift");
        let scale = halves.pop().expect("scale");
        let normed = self.modulate(image, &shift, &scale)?;
        self.linear(&normed, "proj_out")
    }
}

fn silu(input: &Array) -> Result<Array, Exception> {
    ops::sigmoid(input)?.multiply(input)
}

/// `Flux2SwiGLU`: `silu(first half) * second half` of the last axis.
fn swiglu(input: &Array) -> Result<Array, Exception> {
    let mut halves = input.split_equal(2, -1)?;
    let linear = halves.pop().expect("two halves");
    let gate = halves.pop().expect("two halves");
    silu(&gate)?.multiply(&linear)
}

/// Rotates adjacent lane pairs of `[B, L, H, Dh]` in f32, then returns to the
/// input dtype (`apply_rotary_emb`, `use_real_unbind_dim=-1`).
fn rotate(input: &Array, cos: &Array, sin: &Array) -> Result<Array, Exception> {
    let dtype = input.dtype();
    let shape = input.shape().to_vec();
    let wide = input.as_dtype(Dtype::Float32)?;
    let mut pairs = wide
        .reshape(&[shape[0], shape[1], shape[2], shape[3] / 2, 2])?
        .split_equal(2, -1)?;
    let imaginary = pairs.pop().expect("pair");
    let real = pairs.pop().expect("pair");
    let rotated = ops::concatenate(&[imaginary.negative()?, real], -1)?.reshape(&shape)?;
    wide.multiply(cos)?
        .add(rotated.multiply(sin)?)?
        .as_dtype(dtype)
}

#[cfg(test)]
mod tests;
