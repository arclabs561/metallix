//! MLX affine quantization as mlx-lm records it in `config.json`.
//!
//! `quantization` holds the default `group_size`, `bits` and `mode`, plus one
//! entry per module that differs: an object with its own `group_size` and
//! `bits`, or `false` for a module left unquantized. Module paths are
//! mlx-lm's (`language_model.model.layers.0.mlp.gate`), which are also the
//! stored tensor stems (`<path>.weight`, `<path>.scales`, `<path>.biases`).
//! Which modules are quantized at all is the checkpoint's business: a module
//! is quantized when its `scales` tensor is stored.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::{CheckpointError, TensorEncoding};

/// Code widths this decoder has been checked against.
const BITS: [u8; 3] = [4, 6, 8];
/// Columns that share one scale and bias.
const GROUP_SIZES: [u16; 3] = [32, 64, 128];

/// One module's affine code width and group size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlxAffine {
    bits: u8,
    group_size: u16,
}

impl MlxAffine {
    /// Settings this decoder reads: 4, 6 or 8 bits and groups of 32, 64 or
    /// 128 columns; `None` otherwise.
    #[must_use]
    pub fn new(bits: u8, group_size: u16) -> Option<Self> {
        (BITS.contains(&bits) && GROUP_SIZES.contains(&group_size))
            .then_some(Self { bits, group_size })
    }

    /// Bits per code.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.bits
    }

    /// Columns per scale and bias.
    #[must_use]
    pub const fn group_size(self) -> u16 {
        self.group_size
    }

    /// `u32` words per row of `columns` codes, or `None` when `columns` is not
    /// a whole number of groups.
    #[must_use]
    pub fn packed_columns(self, columns: usize) -> Option<usize> {
        let group = usize::from(self.group_size);
        (columns.is_multiple_of(group))
            .then(|| columns * usize::from(self.bits))
            .filter(|bits| bits.is_multiple_of(32))
            .map(|bits| bits / 32)
    }

    /// Scale and bias columns per row of `columns` codes.
    #[must_use]
    pub fn groups(self, columns: usize) -> usize {
        columns / usize::from(self.group_size)
    }

    /// The stored encoding of this module's `weight` codes.
    #[must_use]
    pub const fn encoding(self) -> TensorEncoding {
        TensorEncoding::Affine {
            bits: self.bits,
            group_size: self.group_size,
        }
    }

    fn parse(raw: &Map<String, Value>) -> Result<Self, CheckpointError> {
        let invalid = CheckpointError::InvalidQuantization("quantization");
        if raw
            .keys()
            .any(|key| !matches!(key.as_str(), "group_size" | "bits" | "mode"))
        {
            return Err(invalid);
        }
        if raw
            .get("mode")
            .is_some_and(|mode| mode.as_str() != Some("affine"))
        {
            return Err(CheckpointError::UnsupportedQuantization(
                "a quantization mode other than affine",
            ));
        }
        let field = |name| raw.get(name).and_then(Value::as_u64);
        let bits = field("bits")
            .and_then(|bits| u8::try_from(bits).ok())
            .filter(|bits| BITS.contains(bits))
            .ok_or(CheckpointError::UnsupportedQuantization(
                "affine quantization other than 4, 6 or 8 bits",
            ))?;
        let group_size = field("group_size")
            .and_then(|size| u16::try_from(size).ok())
            .filter(|size| GROUP_SIZES.contains(size))
            .ok_or(CheckpointError::UnsupportedQuantization(
                "an affine group size other than 32, 64 or 128",
            ))?;
        Ok(Self { bits, group_size })
    }
}

/// A checkpoint's MLX affine quantization: the default and per-module
/// overrides.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlxQuantization {
    default: MlxAffine,
    /// Overrides by mlx-lm module path; `None` keeps the module unquantized.
    modules: BTreeMap<String, Option<MlxAffine>>,
}

impl MlxQuantization {
    /// The settings of the module whose tensors are `path.weight`,
    /// `path.scales` and `path.biases`, or `None` if the config leaves it
    /// unquantized.
    #[must_use]
    pub fn module(&self, path: &str) -> Option<MlxAffine> {
        self.modules
            .get(path)
            .copied()
            .unwrap_or(Some(self.default))
    }

    /// The default settings.
    #[must_use]
    pub const fn default_affine(&self) -> MlxAffine {
        self.default
    }

    /// The per-module overrides by mlx-lm module path.
    pub fn overrides(&self) -> impl Iterator<Item = (&str, Option<MlxAffine>)> {
        self.modules
            .iter()
            .map(|(path, affine)| (path.as_str(), *affine))
    }

    /// Reads `quantization` and checks `quantization_config` against it.
    /// mlx-lm writes both as the same object; a `quantization_config` alone
    /// is another format (GPTQ, AWQ, FP8) this decoder does not read.
    ///
    /// # Errors
    /// [`CheckpointError::UnsupportedQuantization`] for another format, mode,
    /// width or group size; [`CheckpointError::InvalidQuantization`] for a
    /// malformed entry.
    pub fn from_config(
        quantization: Option<&Value>,
        quantization_config: Option<&Value>,
    ) -> Result<Option<Self>, CheckpointError> {
        let raw = match (quantization, quantization_config) {
            (None, None) => return Ok(None),
            (None, Some(_)) => {
                return Err(CheckpointError::UnsupportedQuantization(
                    "quantization_config without MLX affine quantization",
                ));
            }
            (Some(raw), Some(other)) if raw != other => {
                return Err(CheckpointError::UnsupportedQuantization(
                    "quantization_config unlike quantization",
                ));
            }
            (Some(raw), _) => raw,
        };
        let raw = raw
            .as_object()
            .ok_or(CheckpointError::InvalidQuantization("quantization"))?;
        let (defaults, modules): (Map<String, Value>, Map<String, Value>) = raw
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .partition(|(key, _)| matches!(key.as_str(), "group_size" | "bits" | "mode"));
        let default = MlxAffine::parse(&defaults)?;
        let mut overrides = BTreeMap::new();
        for (path, value) in modules {
            let setting = match value {
                Value::Bool(false) => None,
                Value::Object(module) => Some(MlxAffine::parse(&module)?),
                _ => return Err(CheckpointError::InvalidQuantization("quantization")),
            };
            overrides.insert(path, setting);
        }
        Ok(Some(Self {
            default,
            modules: overrides,
        }))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{MlxAffine, MlxQuantization};
    use crate::{CheckpointError, TensorEncoding};

    fn parse(quantization: &Value) -> Result<Option<MlxQuantization>, CheckpointError> {
        MlxQuantization::from_config(Some(quantization), None)
    }

    #[test]
    fn reads_the_default_and_per_module_overrides() {
        // The shape of mlx-community/Qwen3.6-35B-A3B-4bit's entry.
        let quantization = parse(&json!({
            "group_size": 64, "bits": 4, "mode": "affine",
            "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8},
            "language_model.lm_head": {"group_size": 32, "bits": 6},
            "language_model.model.layers.1.mlp.gate": false,
        }))
        .unwrap()
        .unwrap();
        let four = MlxAffine {
            bits: 4,
            group_size: 64,
        };
        let eight = MlxAffine {
            bits: 8,
            group_size: 64,
        };
        assert_eq!(
            quantization.module("language_model.model.layers.0.mlp.gate"),
            Some(eight)
        );
        assert_eq!(
            quantization.module("language_model.model.layers.0.mlp.switch_mlp.up_proj"),
            Some(four)
        );
        assert_eq!(
            quantization.module("language_model.model.layers.1.mlp.gate"),
            None
        );
        assert_eq!(quantization.default_affine(), four);
        assert_eq!(quantization.overrides().count(), 3);
        assert_eq!(
            four.encoding(),
            TensorEncoding::Affine {
                bits: 4,
                group_size: 64
            }
        );
        assert_eq!(four.packed_columns(2048), Some(256));
        assert_eq!(eight.packed_columns(2048), Some(512));
        assert_eq!(four.packed_columns(2000), None);
        assert_eq!(four.groups(512), 8);
    }

    #[test]
    fn refuses_what_it_does_not_decode() {
        let unsupported = |value: Value| {
            assert!(
                matches!(
                    parse(&value),
                    Err(CheckpointError::UnsupportedQuantization(_))
                ),
                "{value}"
            );
        };
        unsupported(json!({"group_size": 32, "bits": 4, "mode": "mxfp4"}));
        unsupported(json!({"group_size": 64, "bits": 3}));
        unsupported(json!({"group_size": 16, "bits": 4}));
        unsupported(
            json!({"group_size": 64, "bits": 4, "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 2}}),
        );
        let invalid = |value: Value| {
            assert!(
                matches!(parse(&value), Err(CheckpointError::InvalidQuantization(_))),
                "{value}"
            );
        };
        invalid(
            json!({"group_size": 64, "bits": 4, "language_model.model.layers.0.mlp.gate": true}),
        );
        invalid(
            json!({"group_size": 64, "bits": 4, "language_model.model.layers.0.mlp.gate": {"bits": 8, "group_size": 64, "scale": 1}}),
        );
        invalid(json!([4, 64]));

        // Another format's `quantization_config`, with or without an MLX entry.
        let gptq = json!({"quant_method": "gptq", "bits": 4});
        assert!(matches!(
            MlxQuantization::from_config(None, Some(&gptq)),
            Err(CheckpointError::UnsupportedQuantization(_))
        ));
        let mlx = json!({"group_size": 64, "bits": 4});
        assert!(matches!(
            MlxQuantization::from_config(Some(&mlx), Some(&gptq)),
            Err(CheckpointError::UnsupportedQuantization(_))
        ));
        assert!(
            MlxQuantization::from_config(Some(&mlx), Some(&mlx))
                .unwrap()
                .is_some()
        );
        assert_eq!(MlxQuantization::from_config(None, None).unwrap(), None);
    }
}
