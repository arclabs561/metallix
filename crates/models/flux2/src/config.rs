//! `transformer/config.json` for `Flux2Transformer2DModel`.

use serde::Deserialize;
use thiserror::Error;

const TRANSFORMER_CLASS: &str = "Flux2Transformer2DModel";

/// The executed shape of a `FLUX.2` transformer.
///
/// Options this adapter does not implement (a guidance embedding, patching
/// inside the transformer, a different output width) are refused at parse
/// time rather than silently dropped.
#[derive(Clone, Debug, PartialEq)]
pub struct Flux2TransformerConfig {
    /// Number of attention heads.
    pub(crate) heads: usize,
    /// Scalar lanes per attention head.
    pub(crate) head_dim: usize,
    /// Packed image latent channels per token.
    pub(crate) in_channels: usize,
    /// Text embedding channels per token.
    pub(crate) joint_attention_dim: usize,
    /// Hidden channels in each feed-forward branch.
    pub(crate) mlp_hidden: usize,
    /// Number of joint text/image double-stream blocks.
    pub(crate) double_blocks: usize,
    /// Number of single-stream blocks.
    pub(crate) single_blocks: usize,
    /// Rotary lane widths for the time, height, width and text axes.
    pub(crate) axes_dims_rope: [usize; 4],
    /// Rotary frequency base.
    pub(crate) rope_theta: f64,
    /// Normalization epsilon.
    pub(crate) eps: f32,
    /// Channels in the sinusoidal timestep embedding.
    pub(crate) timestep_channels: usize,
}

/// A configuration cannot be parsed or selects an unsupported transformer layout.
#[derive(Debug, Error, PartialEq)]
pub enum ConfigError {
    /// Malformed JSON or field types; contains the decoder diagnostic.
    #[error("transformer config is not valid JSON: {0}")]
    Json(String),
    /// Unsupported transformer class; contains its declared name.
    #[error("transformer class {0:?} is not {TRANSFORMER_CLASS}")]
    Class(String),
    /// Unsupported configuration option; contains its name.
    #[error("transformer option {0} is not implemented")]
    Unsupported(&'static str),
    /// Rotary axis widths are odd or do not sum to the head width.
    #[error("RoPE axes {axes:?} must sum to the head dimension {head_dim}")]
    RopeAxes {
        /// Supplied widths for the four rotary axes.
        axes: [usize; 4],
        /// Required total lane width.
        head_dim: usize,
    },
    /// A dimension is zero, unrepresentable, or overflows a derived width.
    #[error("invalid or overflowing transformer dimension: {0}")]
    Dimension(&'static str),
    /// A numerical parameter is not finite and positive.
    #[error("transformer parameter must be finite and positive: {0}")]
    Parameter(&'static str),
    /// The supplied ratio does not produce a positive integral MLP width.
    #[error("mlp_ratio {0} does not give an integer MLP width")]
    MlpRatio(f64),
}

#[derive(Deserialize)]
struct RawConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    attention_head_dim: usize,
    axes_dims_rope: [usize; 4],
    eps: f32,
    guidance_embeds: bool,
    in_channels: usize,
    joint_attention_dim: usize,
    mlp_ratio: f64,
    num_attention_heads: usize,
    num_layers: usize,
    num_single_layers: usize,
    out_channels: Option<usize>,
    patch_size: usize,
    rope_theta: f64,
    timestep_guidance_channels: usize,
}

impl Flux2TransformerConfig {
    /// Parses a supported transformer configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ConfigError::Json`] for malformed fields, [`ConfigError::Class`]
    /// for another model class, and [`ConfigError::Unsupported`] for unsupported
    /// guidance, patch or output layouts. Rotary widths and the MLP ratio are
    /// checked by [`ConfigError::RopeAxes`] and [`ConfigError::MlpRatio`].
    /// Zero or overflowing dimensions return [`ConfigError::Dimension`];
    /// invalid epsilon and rotary bases return [`ConfigError::Parameter`].
    pub fn from_json(text: &str) -> Result<Self, ConfigError> {
        let raw: RawConfig =
            serde_json::from_str(text).map_err(|error| ConfigError::Json(error.to_string()))?;
        if raw.class_name != TRANSFORMER_CLASS {
            return Err(ConfigError::Class(raw.class_name));
        }
        if raw.guidance_embeds {
            return Err(ConfigError::Unsupported("guidance_embeds"));
        }
        if raw.patch_size != 1 {
            return Err(ConfigError::Unsupported("patch_size other than 1"));
        }
        if raw.out_channels.is_some_and(|out| out != raw.in_channels) {
            return Err(ConfigError::Unsupported(
                "out_channels other than in_channels",
            ));
        }
        for (name, value) in [
            ("attention_head_dim", raw.attention_head_dim),
            ("num_attention_heads", raw.num_attention_heads),
            ("in_channels", raw.in_channels),
            ("joint_attention_dim", raw.joint_attention_dim),
            ("timestep_guidance_channels", raw.timestep_guidance_channels),
        ] {
            if value == 0 || i32::try_from(value).is_err() {
                return Err(ConfigError::Dimension(name));
            }
        }
        if raw.timestep_guidance_channels & 1 != 0 {
            return Err(ConfigError::Dimension("timestep_guidance_channels"));
        }
        if !raw.eps.is_finite() || raw.eps <= 0.0 {
            return Err(ConfigError::Parameter("eps"));
        }
        if !raw.rope_theta.is_finite() || raw.rope_theta <= 0.0 {
            return Err(ConfigError::Parameter("rope_theta"));
        }
        let axes_sum = raw.axes_dims_rope.iter().try_fold(0_usize, |sum, &axis| {
            sum.checked_add(axis)
                .ok_or(ConfigError::Dimension("axes_dims_rope"))
        })?;
        if axes_sum != raw.attention_head_dim || raw.axes_dims_rope.iter().any(|axis| axis % 2 != 0)
        {
            return Err(ConfigError::RopeAxes {
                axes: raw.axes_dims_rope,
                head_dim: raw.attention_head_dim,
            });
        }
        let inner = raw
            .num_attention_heads
            .checked_mul(raw.attention_head_dim)
            .ok_or(ConfigError::Dimension("heads * head_dim"))?;
        // diffusers: int(dim * mlp_ratio); only exact products are accepted.
        #[allow(clippy::cast_precision_loss)]
        let product = inner as f64 * raw.mlp_ratio;
        if !raw.mlp_ratio.is_finite()
            || raw.mlp_ratio <= 0.0
            || !product.is_finite()
            || product.fract() != 0.0
            || product <= 0.0
            || product > f64::from(i32::MAX)
        {
            return Err(ConfigError::MlpRatio(raw.mlp_ratio));
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let mlp_hidden = product as usize;
        // These are the largest packed projection widths used by the executor.
        let modulation = inner.checked_mul(6);
        let packed = inner
            .checked_mul(3)
            .and_then(|d| mlp_hidden.checked_mul(2).and_then(|m| d.checked_add(m)));
        if [modulation, packed]
            .into_iter()
            .any(|width| width.is_none_or(|value| i32::try_from(value).is_err()))
        {
            return Err(ConfigError::Dimension("packed projection width"));
        }
        Ok(Self {
            heads: raw.num_attention_heads,
            head_dim: raw.attention_head_dim,
            in_channels: raw.in_channels,
            joint_attention_dim: raw.joint_attention_dim,
            mlp_hidden,
            double_blocks: raw.num_layers,
            single_blocks: raw.num_single_layers,
            axes_dims_rope: raw.axes_dims_rope,
            rope_theta: raw.rope_theta,
            eps: raw.eps,
            timestep_channels: raw.timestep_guidance_channels,
        })
    }

    /// Number of attention heads.
    #[must_use]
    pub fn heads(&self) -> usize {
        self.heads
    }

    /// Scalar lanes per attention head.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Packed image latent channels per token.
    #[must_use]
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Text embedding channels per token.
    #[must_use]
    pub fn joint_attention_dim(&self) -> usize {
        self.joint_attention_dim
    }

    /// Hidden channels in each feed-forward branch.
    #[must_use]
    pub fn mlp_hidden(&self) -> usize {
        self.mlp_hidden
    }

    /// Number of joint text/image double-stream blocks.
    #[must_use]
    pub fn double_blocks(&self) -> usize {
        self.double_blocks
    }

    /// Number of single-stream blocks.
    #[must_use]
    pub fn single_blocks(&self) -> usize {
        self.single_blocks
    }

    /// Rotary lane widths for the time, height, width and text axes.
    #[must_use]
    pub fn axes_dims_rope(&self) -> [usize; 4] {
        self.axes_dims_rope
    }

    /// Rotary frequency base.
    #[must_use]
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
    }

    /// Normalization epsilon.
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Channels in the sinusoidal timestep embedding.
    #[must_use]
    pub fn timestep_channels(&self) -> usize {
        self.timestep_channels
    }

    /// Width of the residual stream, `heads * head_dim`.
    /// Parsing checks that this product and the packed projection widths fit MLX.
    #[must_use]
    pub fn inner_dim(&self) -> usize {
        self.heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // FLUX.2-klein-4B transformer/config.json at revision e7b7dc27.
    const KLEIN_4B: &str = r#"{
      "_class_name": "Flux2Transformer2DModel", "_diffusers_version": "0.37.0.dev0",
      "attention_head_dim": 128, "axes_dims_rope": [32, 32, 32, 32], "eps": 1e-06,
      "guidance_embeds": false, "in_channels": 128, "joint_attention_dim": 7680,
      "mlp_ratio": 3.0, "num_attention_heads": 24, "num_layers": 5,
      "num_single_layers": 20, "out_channels": null, "patch_size": 1,
      "rope_theta": 2000, "timestep_guidance_channels": 256
    }"#;

    #[test]
    fn klein_4b_parses_to_its_block_shape() {
        let config = Flux2TransformerConfig::from_json(KLEIN_4B).unwrap();
        assert_eq!(config.inner_dim(), 3072);
        assert_eq!(config.mlp_hidden, 9216);
        assert_eq!((config.double_blocks, config.single_blocks), (5, 20));
    }

    #[test]
    fn a_guidance_embedding_is_refused() {
        let guided = KLEIN_4B.replace("\"guidance_embeds\": false", "\"guidance_embeds\": true");
        assert_eq!(
            Flux2TransformerConfig::from_json(&guided),
            Err(ConfigError::Unsupported("guidance_embeds"))
        );
    }
    #[test]
    fn extreme_shapes_return_errors_without_arithmetic_panics() {
        let mut config: serde_json::Value = serde_json::from_str(KLEIN_4B).unwrap();
        config["axes_dims_rope"] = serde_json::json!([usize::MAX - 1, 2, 0, 0]);
        assert!(matches!(
            Flux2TransformerConfig::from_json(&config.to_string()),
            Err(ConfigError::Dimension("axes_dims_rope"))
        ));
        config = serde_json::from_str(KLEIN_4B).unwrap();
        config["num_attention_heads"] = serde_json::json!(i32::MAX);
        assert!(matches!(
            Flux2TransformerConfig::from_json(&config.to_string()),
            Err(ConfigError::Dimension(_) | ConfigError::MlpRatio(_))
        ));
    }

    #[test]
    fn invalid_numeric_parameters_do_not_enter_model_shapes() {
        for (field, value) in [
            ("eps", 0.0),
            ("rope_theta", -1.0),
            ("mlp_ratio", 1e300),
            ("mlp_ratio", -1.0),
        ] {
            let mut config: serde_json::Value = serde_json::from_str(KLEIN_4B).unwrap();
            config[field] = serde_json::json!(value);
            assert!(
                Flux2TransformerConfig::from_json(&config.to_string()).is_err(),
                "{field}"
            );
        }
        let mut config: serde_json::Value = serde_json::from_str(KLEIN_4B).unwrap();
        config["timestep_guidance_channels"] = serde_json::json!(1);
        assert!(matches!(
            Flux2TransformerConfig::from_json(&config.to_string()),
            Err(ConfigError::Dimension("timestep_guidance_channels"))
        ));
    }
}
