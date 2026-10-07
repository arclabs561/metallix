//! Reading a `qwen35` GGUF file: its configuration keys, tensor names, and
//! the value changes llama.cpp's converter made, so a GGUF loads into the
//! same tensors as the Hugging Face checkpoint it came from.
//!
//! Sources, at llama.cpp commit 7fe450e19305b828c199d602c23a8337aaa1f03b:
//! keys and layer kinds from `src/models/qwen35.cpp` (lines 6-21: layer `i`
//! is recurrent unless `(i + 1) % full_attention_interval == 0`); tensor
//! names from `gguf-py/gguf/constants.py`; the converter's changes from
//! `conversion/qwen.py` (`Qwen3NextModel.modify_tensors` and
//! `_LinearAttentionVReorderBase`), after it widens BF16 to F32
//! (`conversion/base.py`):
//!
//! - every `*norm.weight` except the `GatedDeltaNet` output norm is stored as
//!   `1 + weight`, the value this crate's loader computes anyway;
//! - `ssm_a` is `-exp(A_log)`, so the decay rate `exp(A_log)` is `-ssm_a`;
//! - the convolution kernel is squeezed to `[channels, K]`;
//! - with more value heads than key heads, value heads are stored tiled
//!   (`[v0 of every key head, v1 of every key head, ...]`) instead of
//!   grouped by key head, in every tensor indexed by value head.

use checkpoint::{CheckpointError, TensorSource, gguf::GgufMetadata};

use crate::{Qwen35Config, Qwen35ConfigError, Qwen35LayerKind, Qwen35Mlp};

const ARCHITECTURE: &str = "qwen35";

impl Qwen35Config {
    /// The configuration of a `qwen35` GGUF file, from its metadata and
    /// tensor table. Mixture-of-experts files (`qwen35moe`) and scaled rotary
    /// embedding are refused.
    ///
    /// # Errors
    /// Returns an error for unsupported architecture, invalid metadata, or incompatible tensor shapes.
    pub fn from_gguf(
        metadata: &GgufMetadata,
        tensors: &impl TensorSource,
    ) -> Result<Self, Qwen35ConfigError> {
        let architecture = metadata.architecture()?;
        if architecture != ARCHITECTURE {
            return Err(Qwen35ConfigError::UnexpectedModelType(
                architecture.to_owned(),
            ));
        }
        let key = |name: &str| format!("{ARCHITECTURE}.{name}");
        let count = |name: &str| metadata.require_usize(&key(name));
        if metadata
            .string(&key("rope.scaling.type"))
            .is_some_and(|kind| kind != "none")
        {
            return Err(Qwen35ConfigError::Unsupported("scaled rotary embedding"));
        }

        let layers = count("block_count")?;
        let hidden_size = count("embedding_length")?;
        let head_dim = count("attention.key_length")?;
        if count("attention.value_length")? != head_dim {
            return Err(Qwen35ConfigError::Unsupported(
                "attention value width unlike its key width",
            ));
        }
        let linear_value_heads = count("ssm.time_step_rank")?;
        let inner = count("ssm.inner_size")?;
        if linear_value_heads == 0 || !inner.is_multiple_of(linear_value_heads) {
            return Err(Qwen35ConfigError::Invalid("ssm.inner_size"));
        }
        let interval = metadata
            .unsigned(&key("full_attention_interval"))
            .map_or(Ok(4), |value| {
                usize::try_from(value).map_err(|_| Qwen35ConfigError::Invalid("interval"))
            })?;
        if interval == 0 {
            return Err(Qwen35ConfigError::Invalid("full_attention_interval"));
        }
        let float = |name: &str| {
            metadata.float(&key(name)).ok_or(CheckpointError::Metadata {
                key: key(name),
                expected: "a float",
            })
        };
        #[allow(
            clippy::cast_possible_truncation,
            reason = "the file stores these as f32; f64 only widens them"
        )]
        let (rope_theta, rms_norm_eps) = (
            float("rope.freq_base")? as f32,
            float("attention.layer_norm_rms_epsilon")? as f32,
        );
        let embedding = tensors
            .tensor("token_embd.weight")
            .ok_or_else(|| CheckpointError::MissingTensor("token_embd.weight".into()))?;
        let vocab_size = embedding
            .shape()
            .first()
            .and_then(|rows| usize::try_from(*rows).ok())
            .ok_or(Qwen35ConfigError::Invalid("token_embd.weight"))?;

        let config = Self {
            layers: (0..layers)
                .map(|layer| {
                    if (layer + 1).is_multiple_of(interval) {
                        Qwen35LayerKind::FullAttention
                    } else {
                        Qwen35LayerKind::LinearAttention
                    }
                })
                .collect(),
            hidden_size,
            mlp: Qwen35Mlp::Dense {
                intermediate_size: count("feed_forward_length")?,
            },
            vocab_size,
            attention_heads: count("attention.head_count")?,
            key_value_heads: count("attention.head_count_kv")?,
            head_dim,
            rotary_dim: count("rope.dimension_count")?,
            rope_theta,
            rms_norm_eps,
            linear_key_heads: count("ssm.group_count")?,
            linear_value_heads,
            linear_key_head_dim: count("ssm.state_size")?,
            linear_value_head_dim: inner / linear_value_heads,
            conv_kernel: count("ssm.conv_kernel")?,
            max_position_embeddings: count("context_length")?,
            tie_word_embeddings: tensors.tensor("output.weight").is_none(),
        };
        config.check_gguf()?;
        Ok(config)
    }

    /// The checks [`Qwen35Config::parse`] applies to the same fields.
    fn check_gguf(&self) -> Result<(), Qwen35ConfigError> {
        let positive = [
            ("block_count", self.layers.len()),
            ("embedding_length", self.hidden_size),
            ("vocab_size", self.vocab_size),
            ("attention.head_count", self.attention_heads),
            ("attention.head_count_kv", self.key_value_heads),
            ("attention.key_length", self.head_dim),
            ("ssm.group_count", self.linear_key_heads),
            ("ssm.state_size", self.linear_key_head_dim),
            ("ssm.inner_size", self.linear_value_head_dim),
            ("ssm.conv_kernel", self.conv_kernel),
            ("context_length", self.max_position_embeddings),
        ];
        if let Some((name, _)) = positive.iter().find(|(_, value)| *value == 0) {
            return Err(Qwen35ConfigError::Missing(name));
        }
        if !self.attention_heads.is_multiple_of(self.key_value_heads) {
            return Err(Qwen35ConfigError::HeadGrouping("num_attention_heads"));
        }
        if !self
            .linear_value_heads
            .is_multiple_of(self.linear_key_heads)
        {
            return Err(Qwen35ConfigError::HeadGrouping("linear_num_value_heads"));
        }
        if self.rotary_dim == 0
            || self.rotary_dim > self.head_dim
            || !self.rotary_dim.is_multiple_of(2)
        {
            return Err(Qwen35ConfigError::Invalid("rope.dimension_count"));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Qwen35ConfigError::Invalid("rope_theta"));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(Qwen35ConfigError::Invalid("rms_norm_eps"));
        }
        Ok(())
    }
}

/// The Hugging Face tensor name (without the `model.language_model.` prefix,
/// as the loader keys tensors) of a `qwen35` GGUF tensor, or `None` for a
/// name this decoder does not use.
#[must_use]
pub fn canonical_name(gguf: &str) -> Option<String> {
    match gguf {
        "token_embd.weight" => return Some("embed_tokens.weight".into()),
        "output_norm.weight" => return Some("norm.weight".into()),
        "output.weight" => return Some("lm_head.weight".into()),
        _ => {}
    }
    let rest = gguf.strip_prefix("blk.")?;
    let (layer, field) = rest.split_once('.')?;
    layer.parse::<usize>().ok()?;
    let canonical = match field {
        "attn_norm.weight" => "input_layernorm.weight",
        "post_attention_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q_norm.weight" => "self_attn.q_norm.weight",
        "attn_k_norm.weight" => "self_attn.k_norm.weight",
        "attn_qkv.weight" => "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight" => "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight" => "linear_attn.in_proj_a.weight",
        "ssm_beta.weight" => "linear_attn.in_proj_b.weight",
        "ssm_out.weight" => "linear_attn.out_proj.weight",
        "ssm_conv1d.weight" => "linear_attn.conv1d.weight",
        "ssm_a" => "linear_attn.A_log",
        "ssm_dt.bias" => "linear_attn.dt_bias",
        "ssm_norm.weight" => "linear_attn.norm.weight",
        _ => return None,
    };
    Some(format!("layers.{layer}.{canonical}"))
}

/// Where a tensor indexes value heads, as rows (outermost dimension) or
/// columns (innermost), in units of `width` rows or columns per head, after
/// `skip` leading rows that are not value heads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueHeadAxis {
    /// Heads occupy consecutive row groups after a non-value prefix.
    Rows {
        /// Leading rows belonging to query and key heads, left unchanged.
        skip: usize,
        /// Consecutive rows per value head.
        width: usize,
    },
    /// Heads occupy consecutive column groups in every row.
    Columns {
        /// Consecutive columns per value head.
        width: usize,
    },
}

/// The value-head axis of a canonical tensor name, if the converter
/// reordered it.
#[must_use]
pub fn value_head_axis(canonical: &str, config: &Qwen35Config) -> Option<ValueHeadAxis> {
    let field = canonical.split_once(".linear_attn.")?.1;
    let value_width = config.linear_value_head_dim;
    Some(match field {
        "in_proj_qkv.weight" | "conv1d.weight" => ValueHeadAxis::Rows {
            skip: 2 * config.linear_key_heads * config.linear_key_head_dim,
            width: value_width,
        },
        "in_proj_z.weight" => ValueHeadAxis::Rows {
            skip: 0,
            width: value_width,
        },
        "in_proj_a.weight" | "in_proj_b.weight" | "A_log" | "dt_bias" => {
            ValueHeadAxis::Rows { skip: 0, width: 1 }
        }
        "out_proj.weight" => ValueHeadAxis::Columns { width: value_width },
        _ => return None,
    })
}

/// For each value head in grouped order (key head `k` serves value heads
/// `k r .. (k + 1) r`, `r = value_heads / key_heads`), the position of that
/// head in the converter's tiled order (`j * key_heads + k` for the `j`th
/// value head of key head `k`).
#[must_use]
pub fn tiled_positions(key_heads: usize, value_heads: usize) -> Vec<usize> {
    let per_key = value_heads / key_heads;
    (0..value_heads)
        .map(|grouped| {
            let (key, within) = (grouped / per_key, grouped % per_key);
            within * key_heads + key
        })
        .collect()
}

/// Rearranges `units` equal byte spans starting at `start` so that span `i`
/// of the result is span `order[i]` of the input. Each of `rows` rows of
/// `row_bytes` is treated alike, so columns of a row-major matrix move with
/// `rows > 1`.
///
/// # Panics
///
/// If the spans do not fit the rows, or `order` is not a permutation of
/// `0..order.len()`.
pub fn permute_spans(
    bytes: &mut [u8],
    rows: usize,
    row_bytes: usize,
    start: usize,
    span: usize,
    order: &[usize],
) {
    assert_eq!(bytes.len(), rows * row_bytes, "row layout");
    assert!(start + span * order.len() <= row_bytes, "spans fit a row");
    let mut seen = vec![false; order.len()];
    for &source in order {
        assert!(!std::mem::replace(&mut seen[source], true), "permutation");
    }
    for row in bytes.chunks_exact_mut(row_bytes) {
        let region = &mut row[start..start + span * order.len()];
        let original = region.to_vec();
        for (target, &source) in order.iter().enumerate() {
            region[target * span..(target + 1) * span]
                .copy_from_slice(&original[source * span..(source + 1) * span]);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use checkpoint::{
        CheckpointError, GgufEncoding, TensorInfo, TensorSource,
        gguf::{GgufMetadata, GgufValue},
    };

    use super::{canonical_name, permute_spans, tiled_positions};
    use crate::{Qwen35Config, Qwen35ConfigError, Qwen35LayerKind, Qwen35Mlp};

    struct Table(HashMap<String, TensorInfo>);

    impl TensorSource for Table {
        fn names(&self) -> Vec<&str> {
            self.0.keys().map(String::as_str).collect()
        }

        fn tensor(&self, name: &str) -> Option<&TensorInfo> {
            self.0.get(name)
        }

        fn read(&self, name: &str, _: u64) -> Result<Vec<u8>, CheckpointError> {
            Err(CheckpointError::MissingTensor(name.into()))
        }
    }

    /// The metadata of unsloth/Qwen3.5-0.8B-GGUF@6ab4614 (`Q8_0`), with
    /// `overrides` replacing or adding keys.
    fn qwen35_0_8b(overrides: &[(&str, GgufValue)]) -> GgufMetadata {
        let mut values: HashMap<String, GgufValue> = [
            ("general.architecture", GgufValue::String("qwen35".into())),
            ("qwen35.block_count", GgufValue::U32(24)),
            ("qwen35.context_length", GgufValue::U32(262_144)),
            ("qwen35.embedding_length", GgufValue::U32(1024)),
            ("qwen35.feed_forward_length", GgufValue::U32(3584)),
            ("qwen35.attention.head_count", GgufValue::U32(8)),
            ("qwen35.attention.head_count_kv", GgufValue::U32(2)),
            ("qwen35.rope.freq_base", GgufValue::F32(1.0e7)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                GgufValue::F32(1e-6),
            ),
            ("qwen35.attention.key_length", GgufValue::U32(256)),
            ("qwen35.attention.value_length", GgufValue::U32(256)),
            ("qwen35.ssm.conv_kernel", GgufValue::U32(4)),
            ("qwen35.ssm.state_size", GgufValue::U32(128)),
            ("qwen35.ssm.group_count", GgufValue::U32(16)),
            ("qwen35.ssm.time_step_rank", GgufValue::U32(16)),
            ("qwen35.ssm.inner_size", GgufValue::U32(2048)),
            ("qwen35.full_attention_interval", GgufValue::U32(4)),
            ("qwen35.rope.dimension_count", GgufValue::U32(64)),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect();
        for (key, value) in overrides {
            values.insert((*key).to_owned(), value.clone());
        }
        values.into_iter().collect()
    }

    fn embedding_only() -> Table {
        Table(HashMap::from([(
            "token_embd.weight".to_owned(),
            TensorInfo::new(GgufEncoding::Q8_0, vec![248_320, 1024], 0..1),
        )]))
    }

    #[test]
    fn reads_the_qwen35_0_8b_configuration() {
        let config = Qwen35Config::from_gguf(&qwen35_0_8b(&[]), &embedding_only()).expect("config");
        assert_eq!(config.layers().len(), 24);
        assert_eq!(config.layers()[2], Qwen35LayerKind::LinearAttention);
        assert_eq!(config.layers()[3], Qwen35LayerKind::FullAttention);
        assert_eq!(
            config.mlp(),
            Qwen35Mlp::Dense {
                intermediate_size: 3584
            }
        );
        assert_eq!(config.vocab_size(), 248_320);
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.conv_dim(), 2 * 16 * 128 + 16 * 128);
        assert!(config.tie_word_embeddings());
        assert_eq!(config.linear_value_head_dim, 128);
    }

    #[test]
    fn refuses_other_architectures_and_inconsistent_keys() {
        let table = embedding_only();
        for overrides in [
            vec![(
                "general.architecture",
                GgufValue::String("qwen35moe".into()),
            )],
            vec![("qwen35.attention.value_length", GgufValue::U32(128))],
            vec![("qwen35.ssm.time_step_rank", GgufValue::U32(24))],
            vec![("qwen35.rope.scaling.type", GgufValue::String("yarn".into()))],
            vec![("qwen35.rope.dimension_count", GgufValue::U32(512))],
            vec![("qwen35.full_attention_interval", GgufValue::U32(0))],
        ] {
            assert!(
                Qwen35Config::from_gguf(&qwen35_0_8b(&overrides), &table).is_err(),
                "accepted {overrides:?}"
            );
        }
        let mut sparse = qwen35_0_8b(&[]);
        sparse = sparse
            .keys()
            .into_iter()
            .filter(|key| *key != "qwen35.block_count")
            .map(|key| (key.to_owned(), sparse.get(key).cloned().expect("value")))
            .collect();
        assert!(matches!(
            Qwen35Config::from_gguf(&sparse, &table),
            Err(Qwen35ConfigError::Gguf(CheckpointError::Metadata { .. }))
        ));
    }

    /// The converter's `_reorder_v_heads` on a vector of head labels:
    /// reshape `[key_heads, per_key, width]`, swap the first two axes.
    fn converter_tiled(grouped: &[u8], key_heads: usize, width: usize) -> Vec<u8> {
        let per_key = grouped.len() / width / key_heads;
        let mut tiled = Vec::with_capacity(grouped.len());
        for within in 0..per_key {
            for key in 0..key_heads {
                let head = key * per_key + within;
                tiled.extend_from_slice(&grouped[head * width..(head + 1) * width]);
            }
        }
        tiled
    }

    #[test]
    fn tiled_positions_undo_the_converter_reorder() {
        for (key_heads, value_heads, width) in [(2, 4, 3), (16, 32, 1), (16, 48, 2), (4, 4, 5)] {
            let grouped: Vec<u8> = (0..value_heads * width)
                .map(|index| u8::try_from(index % 251).expect("small"))
                .collect();
            let mut stored = converter_tiled(&grouped, key_heads, width);
            assert_eq!(stored.len(), grouped.len());
            let order = tiled_positions(key_heads, value_heads);
            permute_spans(&mut stored, 1, grouped.len(), 0, width, &order);
            assert_eq!(stored, grouped, "{key_heads}/{value_heads}");
        }
    }

    #[test]
    fn permute_spans_moves_columns_in_every_row() {
        // Two rows: a two-byte prefix kept, then spans [a, b, c] -> [c, a, b].
        let mut bytes = b"xxAABBCCyyDDEEFF".to_vec();
        permute_spans(&mut bytes, 2, 8, 2, 2, &[2, 0, 1]);
        assert_eq!(bytes, b"xxCCAABByyFFDDEE");
    }

    #[test]
    fn maps_every_qwen35_tensor_name() {
        assert_eq!(
            canonical_name("blk.3.attn_q.weight").as_deref(),
            Some("layers.3.self_attn.q_proj.weight")
        );
        assert_eq!(
            canonical_name("blk.0.ssm_a").as_deref(),
            Some("layers.0.linear_attn.A_log")
        );
        assert_eq!(
            canonical_name("token_embd.weight").as_deref(),
            Some("embed_tokens.weight")
        );
        assert_eq!(canonical_name("blk.x.attn_q.weight"), None);
        assert_eq!(canonical_name("blk.0.nextn.eh_proj.weight"), None);
        assert_eq!(canonical_name("rope_freqs.weight"), None);
    }
}
