//! Header-only Gemma 4 checkpoint validation.
//!
//! Every safetensors header is read and checked against the configuration
//! before any payload is loaded, so a checkpoint with a missing, extra or
//! misshapen text tensor fails without allocating weights. Multimodal
//! checkpoints keep text weights under `model.language_model.`; vision and
//! audio tensors are ignored.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use thiserror::Error;

use crate::{Gemma4ConfigError, Gemma4TextConfig};

const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_INDEX_BYTES: u64 = 16 * 1024 * 1024;
const MULTIMODAL_TEXT_PREFIX: &str = "model.language_model.";
const TEXT_ONLY_PREFIX: &str = "model.";

/// One text tensor declared by a safetensors header.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gemma4TensorHeader {
    /// Stored element type, such as `BF16`.
    pub dtype: String,
    /// Stored shape.
    pub shape: Vec<usize>,
    /// Name in the file, before the text prefix is removed.
    pub stored_name: String,
}

/// Header facts that passed validation against the configuration.
#[derive(Clone, Debug)]
pub struct Gemma4CheckpointInspection {
    config: Gemma4TextConfig,
    config_json: String,
    shards: Vec<PathBuf>,
    tensors: BTreeMap<String, Gemma4TensorHeader>,
}

impl Gemma4CheckpointInspection {
    /// Reads `config.json` and every shard header in `model_dir`.
    ///
    /// # Errors
    ///
    /// Returns [`Gemma4CheckpointError`] when a file is missing or malformed,
    /// the configuration is unsupported, or the text tensors differ from the
    /// configuration's layout.
    pub fn inspect(model_dir: impl AsRef<Path>) -> Result<Self, Gemma4CheckpointError> {
        let model_dir = model_dir.as_ref();
        let config_path = model_dir.join("config.json");
        let config_json =
            fs::read_to_string(&config_path).map_err(|source| Gemma4CheckpointError::Read {
                path: config_path,
                source,
            })?;
        let config = Gemma4TextConfig::parse(&config_json)?;
        let shards = shard_paths(model_dir)?;
        let mut stored = BTreeMap::new();
        for shard in &shards {
            for (name, header) in read_header(shard)? {
                stored.insert(name, header);
            }
        }
        let tensors = text_tensors(stored)?;
        check_layout(&config, &tensors)?;
        Ok(Self {
            config,
            config_json,
            shards,
            tensors,
        })
    }

    /// The validated configuration.
    #[must_use]
    pub const fn config(&self) -> &Gemma4TextConfig {
        &self.config
    }

    /// Exact `config.json` text that was validated.
    #[must_use]
    pub fn config_json(&self) -> &str {
        &self.config_json
    }

    /// Shards holding the tensors, in load order.
    #[must_use]
    pub fn shards(&self) -> &[PathBuf] {
        &self.shards
    }

    /// Text tensors keyed by name without the text prefix
    /// (`embed_tokens.weight`, `layers.0.mlp.up_proj.weight`, ...).
    #[must_use]
    pub const fn tensors(&self) -> &BTreeMap<String, Gemma4TensorHeader> {
        &self.tensors
    }
}

/// Every text tensor name and shape the configuration requires.
#[must_use]
pub fn expected_text_tensors(config: &Gemma4TextConfig) -> BTreeMap<String, Vec<usize>> {
    let hidden = config.hidden_size();
    let intermediate = config.intermediate_size();
    let mut expected = BTreeMap::from([
        (
            "embed_tokens.weight".to_owned(),
            vec![config.vocab_size(), hidden],
        ),
        ("norm.weight".to_owned(), vec![hidden]),
    ]);
    for (layer, &kind) in config.layers().iter().enumerate() {
        let attention = config.attention(kind);
        let query = attention.heads * attention.head_dim;
        let key_value = attention.kv_heads * attention.head_dim;
        let base = format!("layers.{layer}");
        let mut add = |name: &str, shape: Vec<usize>| {
            expected.insert(format!("{base}.{name}"), shape);
        };
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            add(&format!("{norm}.weight"), vec![hidden]);
        }
        add("layer_scalar", vec![1]);
        add("mlp.gate_proj.weight", vec![intermediate, hidden]);
        add("mlp.up_proj.weight", vec![intermediate, hidden]);
        add("mlp.down_proj.weight", vec![hidden, intermediate]);
        add("self_attn.q_proj.weight", vec![query, hidden]);
        add("self_attn.k_proj.weight", vec![key_value, hidden]);
        if !attention.value_from_key {
            add("self_attn.v_proj.weight", vec![key_value, hidden]);
        }
        add("self_attn.o_proj.weight", vec![hidden, query]);
        add("self_attn.q_norm.weight", vec![attention.head_dim]);
        add("self_attn.k_norm.weight", vec![attention.head_dim]);
    }
    expected
}

fn check_layout(
    config: &Gemma4TextConfig,
    tensors: &BTreeMap<String, Gemma4TensorHeader>,
) -> Result<(), Gemma4CheckpointError> {
    let expected = expected_text_tensors(config);
    if let Some(name) = expected.keys().find(|name| !tensors.contains_key(*name)) {
        return Err(Gemma4CheckpointError::MissingTensor(name.clone()));
    }
    if let Some(name) = tensors.keys().find(|name| !expected.contains_key(*name)) {
        return Err(Gemma4CheckpointError::UnexpectedTensor(name.clone()));
    }
    for (name, shape) in &expected {
        let header = &tensors[name];
        if &header.shape != shape {
            return Err(Gemma4CheckpointError::Shape {
                name: name.clone(),
                expected: shape.clone(),
                actual: header.shape.clone(),
            });
        }
        if !matches!(header.dtype.as_str(), "BF16" | "F16" | "F32") {
            return Err(Gemma4CheckpointError::Dtype {
                name: name.clone(),
                dtype: header.dtype.clone(),
            });
        }
    }
    Ok(())
}

/// Keeps text tensors and strips their prefix. A multimodal checkpoint is
/// recognized by any `model.language_model.` tensor; otherwise every
/// `model.` tensor is text.
fn text_tensors(
    stored: BTreeMap<String, Gemma4TensorHeader>,
) -> Result<BTreeMap<String, Gemma4TensorHeader>, Gemma4CheckpointError> {
    let prefix = if stored
        .keys()
        .any(|name| name.starts_with(MULTIMODAL_TEXT_PREFIX))
    {
        MULTIMODAL_TEXT_PREFIX
    } else {
        TEXT_ONLY_PREFIX
    };
    let tensors: BTreeMap<_, _> = stored
        .into_iter()
        .filter_map(|(name, header)| {
            name.strip_prefix(prefix)
                .map(|canonical| (canonical.to_owned(), header))
        })
        .collect();
    if tensors.is_empty() {
        return Err(Gemma4CheckpointError::MissingTensor(format!(
            "{prefix}embed_tokens.weight"
        )));
    }
    Ok(tensors)
}

/// The checkpoint's shard files: the index's distinct files, or the single
/// `model.safetensors`.
fn shard_paths(model_dir: &Path) -> Result<Vec<PathBuf>, Gemma4CheckpointError> {
    #[derive(Deserialize)]
    struct Index {
        weight_map: BTreeMap<String, String>,
    }
    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(vec![model_dir.join("model.safetensors")]);
    }
    let bytes = read_bounded(&index_path, MAX_INDEX_BYTES)?;
    let index: Index =
        serde_json::from_slice(&bytes).map_err(|source| Gemma4CheckpointError::Json {
            path: index_path.clone(),
            source,
        })?;
    let files: BTreeSet<_> = index.weight_map.into_values().collect();
    files
        .into_iter()
        .map(|file| {
            // Shard names come from a downloaded file; keep them inside the
            // model directory.
            if file.contains('/') || file.contains('\\') || file.starts_with('.') {
                Err(Gemma4CheckpointError::InvalidShardName(file))
            } else {
                Ok(model_dir.join(file))
            }
        })
        .collect()
}

fn read_header(path: &Path) -> Result<BTreeMap<String, Gemma4TensorHeader>, Gemma4CheckpointError> {
    #[derive(Deserialize)]
    struct Entry {
        dtype: String,
        shape: Vec<usize>,
        data_offsets: [u64; 2],
    }
    let io = |source| Gemma4CheckpointError::Read {
        path: path.to_path_buf(),
        source,
    };
    let mut file = File::open(path).map_err(io)?;
    let file_len = file.metadata().map_err(io)?.len();
    let mut length = [0_u8; 8];
    file.read_exact(&mut length).map_err(io)?;
    let header_len = u64::from_le_bytes(length);
    if header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err(Gemma4CheckpointError::InvalidHeader(path.to_path_buf()));
    }
    let mut header = vec![
        0_u8;
        usize::try_from(header_len).map_err(|_| {
            Gemma4CheckpointError::InvalidHeader(path.to_path_buf())
        })?
    ];
    file.read_exact(&mut header).map_err(io)?;
    let mut raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|source| Gemma4CheckpointError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    raw.remove("__metadata__");
    let payload_len = file_len - 8 - header_len;
    raw.into_iter()
        .map(|(name, value)| {
            let entry: Entry =
                serde_json::from_value(value).map_err(|source| Gemma4CheckpointError::Json {
                    path: path.to_path_buf(),
                    source,
                })?;
            if entry.data_offsets[0] > entry.data_offsets[1] || entry.data_offsets[1] > payload_len
            {
                return Err(Gemma4CheckpointError::InvalidHeader(path.to_path_buf()));
            }
            Ok((
                name.clone(),
                Gemma4TensorHeader {
                    dtype: entry.dtype,
                    shape: entry.shape,
                    stored_name: name,
                },
            ))
        })
        .collect()
}

fn read_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, Gemma4CheckpointError> {
    let io = |source| Gemma4CheckpointError::Read {
        path: path.to_path_buf(),
        source,
    };
    let file = File::open(path).map_err(io)?;
    let mut bytes = Vec::new();
    file.take(maximum + 1).read_to_end(&mut bytes).map_err(io)?;
    if bytes.len() as u64 > maximum {
        return Err(Gemma4CheckpointError::InvalidHeader(path.to_path_buf()));
    }
    Ok(bytes)
}

/// A checkpoint that cannot be served by this adapter.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Gemma4CheckpointError {
    /// A file could not be read.
    #[error("could not read {path}: {source}")]
    Read {
        /// The file.
        path: PathBuf,
        /// The I/O failure.
        source: std::io::Error,
    },
    /// A JSON file or header was malformed.
    #[error("could not parse {path}: {source}")]
    Json {
        /// The file.
        path: PathBuf,
        /// The parse failure.
        source: serde_json::Error,
    },
    /// The configuration is unsupported or invalid.
    #[error(transparent)]
    Config(#[from] Gemma4ConfigError),
    /// A safetensors header is oversized or points past the file.
    #[error("invalid safetensors header in {0}")]
    InvalidHeader(PathBuf),
    /// The shard index names a file outside the model directory.
    #[error("shard name {0:?} is not a plain file name")]
    InvalidShardName(String),
    /// A tensor the configuration requires is absent.
    #[error("checkpoint is missing text tensor {0}")]
    MissingTensor(String),
    /// A text tensor the configuration does not explain.
    #[error("checkpoint has unexpected text tensor {0}")]
    UnexpectedTensor(String),
    /// A tensor's shape disagrees with the configuration.
    #[error("text tensor {name} has shape {actual:?}, expected {expected:?}")]
    Shape {
        /// Tensor name without the text prefix.
        name: String,
        /// Shape implied by the configuration.
        expected: Vec<usize>,
        /// Stored shape.
        actual: Vec<usize>,
    },
    /// A tensor's element type is not a supported float.
    #[error("text tensor {name} has unsupported dtype {dtype}")]
    Dtype {
        /// Tensor name without the text prefix.
        name: String,
        /// Stored dtype.
        dtype: String,
    },
}

#[cfg(test)]
pub(crate) mod tests {
    use std::{collections::BTreeMap, fs, path::Path};

    use super::{Gemma4CheckpointError, Gemma4CheckpointInspection, expected_text_tensors};
    use crate::Gemma4TextConfig;

    /// A two-layer layout (sliding, full) small enough to write to disk.
    pub(crate) const TINY_CONFIG: &str = r#"{
      "model_type": "gemma4_unified",
      "text_config": {
        "model_type": "gemma4_unified_text", "attention_k_eq_v": true,
        "final_logit_softcapping": 30.0, "global_head_dim": 8, "head_dim": 4,
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 8,
        "intermediate_size": 12, "layer_types": ["sliding_attention", "full_attention"],
        "max_position_embeddings": 64, "num_attention_heads": 2,
        "num_global_key_value_heads": 1, "num_hidden_layers": 2,
        "num_key_value_heads": 1, "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 10000.0, "rope_type": "proportional"},
          "sliding_attention": {"rope_theta": 100.0, "rope_type": "default"}
        },
        "sliding_window": 3, "tie_word_embeddings": true, "vocab_size": 16
      }
    }"#;

    /// Writes a zero-filled BF16 safetensors file with these tensors.
    fn write_checkpoint(dir: &Path, config: &str, tensors: &BTreeMap<String, Vec<usize>>) {
        fs::write(dir.join("config.json"), config).expect("config");
        let mut header = serde_json::Map::new();
        let mut offset = 0_u64;
        for (name, shape) in tensors {
            let bytes = 2 * shape.iter().product::<usize>() as u64;
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let header = serde_json::to_vec(&header).expect("header");
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(&header);
        file.resize(file.len() + usize::try_from(offset).expect("small"), 0);
        fs::write(dir.join("model.safetensors"), file).expect("weights");
    }

    fn multimodal_tensors() -> BTreeMap<String, Vec<usize>> {
        let config = Gemma4TextConfig::parse(TINY_CONFIG).expect("tiny config");
        let mut tensors: BTreeMap<_, _> = expected_text_tensors(&config)
            .into_iter()
            .map(|(name, shape)| (format!("model.language_model.{name}"), shape))
            .collect();
        tensors.insert(
            "model.embed_audio.embedding_projection.weight".into(),
            vec![8, 4],
        );
        tensors
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("gemma-checkpoint-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    #[test]
    fn accepts_text_tensors_under_the_multimodal_prefix() {
        let dir = scratch("accept");
        write_checkpoint(&dir, TINY_CONFIG, &multimodal_tensors());
        let inspection = Gemma4CheckpointInspection::inspect(&dir).expect("valid layout");
        assert!(
            inspection
                .tensors()
                .contains_key("layers.0.self_attn.v_proj.weight")
        );
        // The full layer reuses its K projection as V.
        assert!(
            !inspection
                .tensors()
                .contains_key("layers.1.self_attn.v_proj.weight")
        );
        assert!(
            !inspection
                .tensors()
                .keys()
                .any(|name| name.contains("audio"))
        );
        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn refuses_missing_extra_and_misshapen_text_tensors() {
        type Mutation = fn(&mut BTreeMap<String, Vec<usize>>);
        let cases: [(&str, Mutation); 3] = [
            ("missing", |tensors| {
                tensors.remove("model.language_model.layers.1.layer_scalar");
            }),
            ("extra", |tensors| {
                tensors.insert(
                    "model.language_model.layers.1.self_attn.v_proj.weight".into(),
                    vec![8, 8],
                );
            }),
            ("shape", |tensors| {
                tensors.insert(
                    "model.language_model.layers.0.mlp.up_proj.weight".into(),
                    vec![8, 12],
                );
            }),
        ];
        for (name, mutate) in cases {
            let dir = scratch(name);
            let mut tensors = multimodal_tensors();
            mutate(&mut tensors);
            write_checkpoint(&dir, TINY_CONFIG, &tensors);
            let error = Gemma4CheckpointInspection::inspect(&dir).expect_err(name);
            assert!(
                matches!(
                    (name, &error),
                    ("missing", Gemma4CheckpointError::MissingTensor(_))
                        | ("extra", Gemma4CheckpointError::UnexpectedTensor(_))
                        | ("shape", Gemma4CheckpointError::Shape { .. })
                ),
                "{name}: {error}"
            );
            fs::remove_dir_all(dir).expect("cleanup");
        }
    }
}
