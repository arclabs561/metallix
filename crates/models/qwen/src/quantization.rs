//! MLX affine quantization parameters, shared by checkpoint inspection and
//! the Metal forward path.

use serde_json::{Map, Value};

/// Bits per quantized element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3AffineBits {
    /// Four bits, eight elements per packed `u32`.
    Four,
    /// Six bits, sixteen elements per three packed `u32`.
    Six,
    /// Eight bits, four elements per packed `u32`.
    Eight,
}

/// Elements sharing one scale and bias, along each weight row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen3AffineGroupSize {
    /// 32 elements.
    G32,
    /// 64 elements, the MLX and mlx-community default.
    G64,
    /// 128 elements.
    G128,
}

/// MLX affine quantization: `w ≈ scale * q + bias` per group of a row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3AffineQuantization {
    /// Bits per element.
    pub bits: Qwen3AffineBits,
    /// Elements per group.
    pub group_size: Qwen3AffineGroupSize,
}

impl Qwen3AffineQuantization {
    /// MLX's default and mlx-community's usual 4-bit layout.
    pub const FOUR_BIT_G64: Self = Self {
        bits: Qwen3AffineBits::Four,
        group_size: Qwen3AffineGroupSize::G64,
    };

    /// The parameters a config's `quantization` names, if supported.
    #[must_use]
    pub const fn from_parameters(bits: u64, group_size: u64) -> Option<Self> {
        let bits = match bits {
            4 => Qwen3AffineBits::Four,
            6 => Qwen3AffineBits::Six,
            8 => Qwen3AffineBits::Eight,
            _ => return None,
        };
        let group_size = match group_size {
            32 => Qwen3AffineGroupSize::G32,
            64 => Qwen3AffineGroupSize::G64,
            128 => Qwen3AffineGroupSize::G128,
            _ => return None,
        };
        Some(Self { bits, group_size })
    }

    /// Bits per element, as MLX takes it.
    #[must_use]
    pub const fn bits(self) -> i32 {
        match self.bits {
            Qwen3AffineBits::Four => 4,
            Qwen3AffineBits::Six => 6,
            Qwen3AffineBits::Eight => 8,
        }
    }

    /// Elements per group, as MLX takes it.
    #[must_use]
    pub const fn group_size(self) -> i32 {
        match self.group_size {
            Qwen3AffineGroupSize::G32 => 32,
            Qwen3AffineGroupSize::G64 => 64,
            Qwen3AffineGroupSize::G128 => 128,
        }
    }

    const fn bit_count(self) -> u64 {
        match self.bits {
            Qwen3AffineBits::Four => 4,
            Qwen3AffineBits::Six => 6,
            Qwen3AffineBits::Eight => 8,
        }
    }

    const fn group_length(self) -> u64 {
        match self.group_size {
            Qwen3AffineGroupSize::G32 => 32,
            Qwen3AffineGroupSize::G64 => 64,
            Qwen3AffineGroupSize::G128 => 128,
        }
    }

    /// `u32` words holding one row of `columns` quantized elements.
    #[must_use]
    pub const fn packed_columns(self, columns: u64) -> u64 {
        columns * self.bit_count() / 32
    }

    /// Scale (and bias) entries per row of `columns` elements.
    #[must_use]
    pub const fn groups(self, columns: u64) -> u64 {
        columns / self.group_length()
    }

    /// Parses mlx-lm's `config.json` `quantization` object
    /// (`{"group_size": .., "bits": ..}`, optionally `"mode": "affine"`).
    /// Per-module entries describe mixed precision and are refused, as are
    /// layouts this crate does not implement.
    ///
    /// # Errors
    ///
    /// Returns the object's JSON text when it is not one supported uniform
    /// affine layout.
    pub fn from_config(raw: &Map<String, Value>) -> Result<Self, String> {
        let refuse = || Value::Object(raw.clone()).to_string();
        if raw
            .keys()
            .any(|key| !matches!(key.as_str(), "group_size" | "bits" | "mode"))
            || raw
                .get("mode")
                .is_some_and(|mode| mode.as_str() != Some("affine"))
        {
            return Err(refuse());
        }
        let parameter = |key| raw.get(key).and_then(Value::as_u64);
        match (parameter("bits"), parameter("group_size")) {
            (Some(bits), Some(group_size)) => {
                Self::from_parameters(bits, group_size).ok_or_else(refuse)
            }
            _ => Err(refuse()),
        }
    }
}

/// Whether `name` is a projection or embedding weight, the tensors affine
/// quantization packs. Norm weights are not.
pub(crate) fn is_quantizable(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".weight") else {
        return false;
    };
    stem == "model.embed_tokens"
        || stem == "lm_head"
        || [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        .iter()
        .any(|projection| stem.ends_with(&format!(".{projection}")))
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, Value, json};

    use super::{Qwen3AffineBits, Qwen3AffineQuantization, is_quantizable};

    fn parse(value: Value) -> Result<Qwen3AffineQuantization, String> {
        let Value::Object(map) = value else {
            panic!("test configs are objects")
        };
        Qwen3AffineQuantization::from_config(&map)
    }

    #[test]
    fn config_quantization_takes_only_uniform_supported_affine_layouts() {
        assert_eq!(
            parse(json!({"group_size": 64, "bits": 4})),
            Ok(Qwen3AffineQuantization::FOUR_BIT_G64)
        );
        assert!(parse(json!({"group_size": 64, "bits": 4, "mode": "affine"})).is_ok());
        for (bits, expected) in [(6, Qwen3AffineBits::Six), (8, Qwen3AffineBits::Eight)] {
            assert_eq!(
                parse(json!({"group_size": 64, "bits": bits})).map(|layout| layout.bits),
                Ok(expected)
            );
        }
        for refused in [
            json!({"group_size": 64, "bits": 3}),
            json!({"group_size": 64, "bits": 2}),
            json!({"group_size": 48, "bits": 4}),
            json!({"group_size": 64}),
            json!({"group_size": 32, "bits": 4, "mode": "mxfp4"}),
            json!({"group_size": 64, "bits": 4, "lm_head": {"group_size": 64, "bits": 8}}),
        ] {
            assert!(parse(refused).is_err());
        }
        assert!(Qwen3AffineQuantization::from_config(&Map::new()).is_err());
    }

    #[test]
    fn layouts_size_packed_rows_and_groups() {
        let layout = Qwen3AffineQuantization::FOUR_BIT_G64;
        assert_eq!(layout.packed_columns(1_024), 128);
        assert_eq!(layout.groups(3_072), 48);
        let six = Qwen3AffineQuantization::from_parameters(6, 64).expect("6-bit");
        assert_eq!(six.packed_columns(1_024), 192);
        let eight = Qwen3AffineQuantization::from_parameters(8, 64).expect("8-bit");
        assert_eq!(eight.packed_columns(1_024), 256);
    }

    #[test]
    fn only_projections_and_the_embedding_are_quantizable() {
        for name in [
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ] {
            assert!(is_quantizable(name), "{name}");
        }
        for name in [
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.mlp.down_proj.scales",
        ] {
            assert!(!is_quantizable(name), "{name}");
        }
    }
}
