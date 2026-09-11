//! Local Qwen3 safetensors checkpoint inspection.
//!
//! Inspection reads each safetensors header without allocating payloads. Its
//! adapter-private bounded read path can then fetch one validated raw tensor
//! for loader qualification without loading a whole shard.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use thiserror::Error;

use crate::{Qwen3ConfigError, Qwen3TextContract};

const SAFETENSORS_PREFIX_BYTES: u64 = 8;
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024;

#[cfg(any(feature = "metal", test))]
mod read;

/// Header-only facts about a validated local Qwen3 checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3CheckpointInspection {
    contract: Qwen3TextContract,
    tensor_count: usize,
    tensor_bytes: u64,
    shards: Vec<PathBuf>,
    tensors: BTreeMap<String, TensorLocation>,
}

impl Qwen3CheckpointInspection {
    /// Validates a model directory and reads every safetensors header in it.
    ///
    /// The directory must contain a valid `config.json`, at least one regular
    /// `.safetensors` file, and the dense tensor names required by the Qwen3
    /// decoder layout. Tensor payload bytes are never read.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3CheckpointError`] for missing artifacts, invalid or
    /// inconsistent safetensors headers, or an incomplete Qwen3 tensor layout.
    pub fn inspect(model_dir: impl AsRef<Path>) -> Result<Self, Qwen3CheckpointError> {
        let model_dir = model_dir.as_ref();
        if !model_dir.is_dir() {
            return Err(Qwen3CheckpointError::NotDirectory(model_dir.to_path_buf()));
        }

        let config_path = model_dir.join("config.json");
        let config = fs::read_to_string(&config_path).map_err(|source| {
            Qwen3CheckpointError::ReadConfig {
                path: config_path,
                source,
            }
        })?;
        let contract = Qwen3TextContract::parse(&config).map_err(Qwen3CheckpointError::Config)?;
        let layout: RawCheckpointLayout =
            serde_json::from_str(&config).map_err(Qwen3CheckpointError::CheckpointConfigJson)?;
        if layout.vocab_size == 0 {
            return Err(Qwen3CheckpointError::MissingCheckpointDimension(
                "vocab_size",
            ));
        }
        if layout.intermediate_size == 0 {
            return Err(Qwen3CheckpointError::MissingCheckpointDimension(
                "intermediate_size",
            ));
        }

        let shards = discover_shards(model_dir)?;

        let mut tensors = BTreeMap::new();
        let mut tensor_count = 0_usize;
        let mut tensor_bytes = 0_u64;
        for shard in &shards {
            let header = read_validated_shard(shard)?;
            for (name, tensor) in header.tensors {
                if name.trim().is_empty() {
                    return Err(Qwen3CheckpointError::BlankTensorName(shard.clone()));
                }
                let byte_length = tensor.byte_length;
                if tensors
                    .insert(
                        name.clone(),
                        TensorLocation {
                            #[cfg(any(feature = "metal", test))]
                            shard: shard.clone(),
                            #[cfg(any(feature = "metal", test))]
                            identity: header.identity,
                            range: tensor,
                        },
                    )
                    .is_some()
                {
                    return Err(Qwen3CheckpointError::DuplicateTensorName(name));
                }
                tensor_count = tensor_count
                    .checked_add(1)
                    .ok_or(Qwen3CheckpointError::TensorCountOverflow)?;
                tensor_bytes = tensor_bytes
                    .checked_add(byte_length)
                    .ok_or(Qwen3CheckpointError::TensorBytesOverflow)?;
            }
        }

        for required in required_dense_tensors(&contract, &layout)? {
            let Some(actual) = tensors.get(&required.name) else {
                return Err(Qwen3CheckpointError::MissingRequiredTensor(required.name));
            };
            if actual.range.shape != required.shape {
                return Err(Qwen3CheckpointError::UnexpectedTensorShape {
                    tensor: required.name,
                    expected: required.shape,
                    actual: actual.range.shape.clone(),
                });
            }
        }

        Ok(Self {
            contract,
            tensor_count,
            tensor_bytes,
            shards,
            tensors,
        })
    }

    /// Returns the configuration contract validated for this checkpoint.
    #[must_use]
    pub const fn contract(&self) -> &Qwen3TextContract {
        &self.contract
    }

    /// Returns the number of distinct tensors declared by all shard headers.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Returns the sum of all declared tensor payload ranges in bytes.
    #[must_use]
    pub const fn tensor_bytes(&self) -> u64 {
        self.tensor_bytes
    }

    /// Returns regular safetensors shard paths in deterministic order.
    #[must_use]
    pub fn shards(&self) -> &[PathBuf] {
        &self.shards
    }

    /// Plans a BF16 read without allocating its payload.
    #[cfg(feature = "metal")]
    pub(crate) fn bf16_tensor_bytes(&self, name: &str) -> Result<u64, Qwen3CheckpointError> {
        let location = self
            .tensors
            .get(name)
            .ok_or_else(|| Qwen3CheckpointError::UnknownTensor(name.to_owned()))?;
        if location.range.dtype != "BF16" {
            return Err(Qwen3CheckpointError::UnsupportedTensorDtype {
                path: location.shard.clone(),
                tensor: name.to_owned(),
                dtype: location.range.dtype.clone(),
            });
        }
        Ok(location.range.byte_length)
    }
}

fn discover_shards(model_dir: &Path) -> Result<Vec<PathBuf>, Qwen3CheckpointError> {
    let entries =
        fs::read_dir(model_dir).map_err(|source| Qwen3CheckpointError::ReadDirectory {
            path: model_dir.to_path_buf(),
            source,
        })?;
    let mut shards = Vec::new();
    for entry in entries {
        let entry = entry.map_err(Qwen3CheckpointError::DirectoryEntry)?;
        let path = entry.path();
        let is_safetensors = path
            .extension()
            .is_some_and(|extension| extension == "safetensors");
        if is_safetensors
            && entry
                .file_type()
                .map_err(|source| Qwen3CheckpointError::FileType {
                    path: path.clone(),
                    source,
                })?
                .is_file()
        {
            shards.push(path);
        }
    }
    shards.sort();
    if shards.is_empty() {
        return Err(Qwen3CheckpointError::NoSafetensors(model_dir.to_path_buf()));
    }
    Ok(shards)
}

#[cfg(any(feature = "metal", test))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Qwen3TensorBytes {
    dtype: String,
    shape: Vec<u64>,
    bytes: Vec<u8>,
}

#[cfg(any(feature = "metal", test))]
impl Qwen3TensorBytes {
    /// Returns the safetensors dtype recorded in the validated header.
    #[must_use]
    pub(crate) fn dtype(&self) -> &str {
        &self.dtype
    }
    /// Returns the tensor shape recorded in the validated safetensors header.
    #[must_use]
    pub(crate) fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns exactly the requested tensor payload bytes.
    #[must_use]
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ShardIdentity {
    bytes: u64,
    modified: std::time::SystemTime,
}

impl ShardIdentity {
    fn from_metadata(path: &Path, metadata: &fs::Metadata) -> Result<Self, Qwen3CheckpointError> {
        Ok(Self {
            bytes: metadata.len(),
            modified: metadata.modified().map_err(|source| {
                Qwen3CheckpointError::ShardMetadata {
                    path: path.to_path_buf(),
                    source,
                }
            })?,
        })
    }
}

struct ValidatedShard {
    #[cfg(any(feature = "metal", test))]
    identity: ShardIdentity,
    tensors: Vec<(String, TensorRange)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorLocation {
    #[cfg(any(feature = "metal", test))]
    shard: PathBuf,
    #[cfg(any(feature = "metal", test))]
    identity: ShardIdentity,
    range: TensorRange,
}

fn ensure_shard_identity(
    file: &File,
    path: &Path,
    expected: ShardIdentity,
) -> Result<(), Qwen3CheckpointError> {
    let actual = ShardIdentity::from_metadata(
        path,
        &file
            .metadata()
            .map_err(|source| Qwen3CheckpointError::ShardMetadata {
                path: path.to_path_buf(),
                source,
            })?,
    )?;
    if actual != expected {
        return Err(Qwen3CheckpointError::ShardMetadataDrift {
            path: path.to_path_buf(),
            expected_bytes: expected.bytes,
            actual_bytes: actual.bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
fn read_header(path: &Path) -> Result<Vec<(String, TensorRange)>, Qwen3CheckpointError> {
    Ok(read_validated_shard(path)?.tensors)
}

fn read_validated_shard(path: &Path) -> Result<ValidatedShard, Qwen3CheckpointError> {
    let mut file = File::open(path).map_err(|source| Qwen3CheckpointError::OpenShard {
        path: path.to_path_buf(),
        source,
    })?;
    let identity = ShardIdentity::from_metadata(
        path,
        &file
            .metadata()
            .map_err(|source| Qwen3CheckpointError::ShardMetadata {
                path: path.to_path_buf(),
                source,
            })?,
    )?;
    let file_bytes = identity.bytes;
    if file_bytes < SAFETENSORS_PREFIX_BYTES {
        return Err(Qwen3CheckpointError::TruncatedPrefix(path.to_path_buf()));
    }

    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|source| Qwen3CheckpointError::ReadShard {
            path: path.to_path_buf(),
            source,
        })?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes > MAX_HEADER_BYTES || header_bytes > file_bytes - SAFETENSORS_PREFIX_BYTES {
        return Err(Qwen3CheckpointError::InvalidHeaderLength {
            path: path.to_path_buf(),
            header_bytes,
            file_bytes,
        });
    }
    let header_len =
        usize::try_from(header_bytes).map_err(|_| Qwen3CheckpointError::InvalidHeaderLength {
            path: path.to_path_buf(),
            header_bytes,
            file_bytes,
        })?;
    let mut json = vec![0_u8; header_len];
    file.read_exact(&mut json)
        .map_err(|source| Qwen3CheckpointError::ReadShard {
            path: path.to_path_buf(),
            source,
        })?;
    let header: UniqueHeader =
        serde_json::from_slice(&json).map_err(|source| Qwen3CheckpointError::HeaderJson {
            path: path.to_path_buf(),
            source,
        })?;

    let tensors = validate_header_tensors(
        path,
        file_bytes - SAFETENSORS_PREFIX_BYTES - header_bytes,
        header.0,
    )?;
    #[cfg(any(feature = "metal", test))]
    let tensors = {
        let mut tensors = tensors;
        let payload_offset = SAFETENSORS_PREFIX_BYTES.checked_add(header_bytes).ok_or(
            Qwen3CheckpointError::InvalidHeaderLength {
                path: path.to_path_buf(),
                header_bytes,
                file_bytes,
            },
        )?;
        for (_, tensor) in &mut tensors {
            tensor.file_offset = payload_offset.checked_add(tensor.payload_offset).ok_or(
                Qwen3CheckpointError::TensorOutsidePayload {
                    path: path.to_path_buf(),
                    tensor: "file offset overflow".to_owned(),
                },
            )?;
        }
        tensors
    };
    ensure_shard_identity(&file, path, identity)?;
    Ok(ValidatedShard {
        #[cfg(any(feature = "metal", test))]
        identity,
        tensors,
    })
}

// Reject repeated tensor/metadata names before map insertion can discard them.
// Compare decoded keys so JSON escape spelling cannot bypass uniqueness.
struct UniqueHeader(BTreeMap<String, serde_json::Value>);

impl<'de> Deserialize<'de> for UniqueHeader {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HeaderVisitor;

        impl<'de> serde::de::Visitor<'de> for HeaderVisitor {
            type Value = UniqueHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a safetensors header with unique keys")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut header = BTreeMap::new();
                while let Some(key) = map.next_key::<String>()? {
                    if header.contains_key(&key) {
                        return Err(serde::de::Error::custom("duplicate safetensors header key"));
                    }
                    header.insert(key, map.next_value::<UniqueJsonValue>()?.0);
                }
                Ok(UniqueHeader(header))
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

// Preserve normal JSON value semantics while applying the same uniqueness
// rule recursively, before serde_json::Value can collapse nested map entries.
struct UniqueJsonValue(serde_json::Value);

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ValueVisitor;

        impl<'de> serde::de::Visitor<'de> for ValueVisitor {
            type Value = UniqueJsonValue;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON value with unique object keys")
            }

            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(value.into()))
            }

            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(value.into()))
            }

            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(value.into()))
            }

            fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueJsonValue(number.into()))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                self.visit_string(value.to_owned())
            }

            fn visit_string<E: serde::de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(value.into()))
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueJsonValue(serde_json::Value::Null))
            }

            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<UniqueJsonValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueJsonValue(serde_json::Value::Array(values)))
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                map: M,
            ) -> Result<Self::Value, M::Error> {
                let object =
                    UniqueHeader::deserialize(serde::de::value::MapAccessDeserializer::new(map))?;
                Ok(UniqueJsonValue(serde_json::Value::Object(
                    object.0.into_iter().collect(),
                )))
            }
        }

        deserializer.deserialize_any(ValueVisitor)
    }
}

fn validate_header_tensors(
    path: &Path,
    data_bytes: u64,
    header: BTreeMap<String, serde_json::Value>,
) -> Result<Vec<(String, TensorRange)>, Qwen3CheckpointError> {
    let mut tensors = Vec::new();
    let mut ranges = Vec::new();
    for (name, value) in header {
        if name == "__metadata__" {
            if serde_json::from_value::<BTreeMap<String, String>>(value).is_err() {
                return Err(Qwen3CheckpointError::InvalidMetadata(path.to_path_buf()));
            }
            continue;
        }
        let tensor: RawTensor = serde_json::from_value(value).map_err(|source| {
            Qwen3CheckpointError::InvalidTensorHeader {
                path: path.to_path_buf(),
                tensor: name.clone(),
                source,
            }
        })?;
        if tensor.data_offsets.len() != 2 {
            return Err(Qwen3CheckpointError::InvalidTensorOffsets {
                path: path.to_path_buf(),
                tensor: name,
            });
        }
        let start = tensor.data_offsets[0];
        let end = tensor.data_offsets[1];
        if start > end || end > data_bytes {
            return Err(Qwen3CheckpointError::TensorOutsidePayload {
                path: path.to_path_buf(),
                tensor: name,
            });
        }
        let element_count = tensor.shape.iter().try_fold(1_u64, |total, dimension| {
            total
                .checked_mul(*dimension)
                .ok_or(Qwen3CheckpointError::TensorShapeOverflow {
                    path: path.to_path_buf(),
                    tensor: name.clone(),
                })
        })?;
        let expected_bytes = tensor_byte_length(&tensor.dtype, element_count).ok_or_else(|| {
            Qwen3CheckpointError::UnsupportedTensorDtype {
                path: path.to_path_buf(),
                tensor: name.clone(),
                dtype: tensor.dtype.clone(),
            }
        })?;
        if end - start != expected_bytes {
            return Err(Qwen3CheckpointError::TensorByteLengthMismatch {
                path: path.to_path_buf(),
                tensor: name,
                expected_bytes,
                actual_bytes: end - start,
            });
        }
        ranges.push((start, end));
        tensors.push((
            name,
            TensorRange {
                byte_length: end - start,
                #[cfg(any(feature = "metal", test))]
                dtype: tensor.dtype,
                shape: tensor.shape,
                #[cfg(any(feature = "metal", test))]
                payload_offset: start,
                #[cfg(any(feature = "metal", test))]
                file_offset: 0,
            },
        ));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|ranges| ranges[1].0 < ranges[0].1) {
        return Err(Qwen3CheckpointError::OverlappingTensorRanges(
            path.to_path_buf(),
        ));
    }
    let mut expected_start = 0_u64;
    for (start, end) in ranges {
        if start != expected_start {
            return Err(Qwen3CheckpointError::NonContiguousPayload(
                path.to_path_buf(),
            ));
        }
        expected_start = end;
    }
    if expected_start != data_bytes {
        return Err(Qwen3CheckpointError::NonContiguousPayload(
            path.to_path_buf(),
        ));
    }
    Ok(tensors)
}

fn tensor_byte_length(dtype: &str, element_count: u64) -> Option<u64> {
    let bytes_per_element = match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3FN" | "F8_E4M3FNUZ" | "F8_E5M2" | "F8_E5M2FNUZ"
        | "F8_E8M0FNU" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => return None,
    };
    element_count.checked_mul(bytes_per_element)
}

#[derive(Debug, Deserialize)]
struct RawTensor {
    dtype: String,
    data_offsets: Vec<u64>,
    shape: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TensorRange {
    byte_length: u64,
    #[cfg(any(feature = "metal", test))]
    dtype: String,
    shape: Vec<u64>,
    #[cfg(any(feature = "metal", test))]
    payload_offset: u64,
    #[cfg(any(feature = "metal", test))]
    file_offset: u64,
}

#[derive(Debug, Deserialize)]
struct RawCheckpointLayout {
    #[serde(default)]
    vocab_size: u64,
    #[serde(default)]
    intermediate_size: u64,
    #[serde(default)]
    tie_word_embeddings: bool,
}

struct ExpectedTensor {
    name: String,
    shape: Vec<u64>,
}

fn required_dense_tensors(
    contract: &Qwen3TextContract,
    layout: &RawCheckpointLayout,
) -> Result<Vec<ExpectedTensor>, Qwen3CheckpointError> {
    let head_dim = u64::from(contract.head_dim());
    let hidden = u64::from(contract.hidden_size());
    let query_width = u64::from(contract.attention_heads())
        .checked_mul(head_dim)
        .ok_or(Qwen3CheckpointError::InvalidAttentionLayout)?;
    let kv_width = u64::from(contract.key_value_heads())
        .checked_mul(head_dim)
        .ok_or(Qwen3CheckpointError::InvalidAttentionLayout)?;
    let mut expected = vec![ExpectedTensor {
        name: "model.embed_tokens.weight".to_owned(),
        shape: vec![layout.vocab_size, hidden],
    }];
    for layer in 0..contract.total_layers() {
        let prefix = format!("model.layers.{layer}");
        expected.extend([
            ExpectedTensor {
                name: format!("{prefix}.input_layernorm.weight"),
                shape: vec![hidden],
            },
            ExpectedTensor {
                name: format!("{prefix}.self_attn.q_norm.weight"),
                shape: vec![head_dim],
            },
            ExpectedTensor {
                name: format!("{prefix}.self_attn.k_norm.weight"),
                shape: vec![head_dim],
            },
            ExpectedTensor {
                name: format!("{prefix}.self_attn.q_proj.weight"),
                shape: vec![query_width, hidden],
            },
            ExpectedTensor {
                name: format!("{prefix}.self_attn.k_proj.weight"),
                shape: vec![kv_width, hidden],
            },
            ExpectedTensor {
                name: format!("{prefix}.self_attn.v_proj.weight"),
                shape: vec![kv_width, hidden],
            },
            ExpectedTensor {
                name: format!("{prefix}.self_attn.o_proj.weight"),
                shape: vec![hidden, query_width],
            },
            ExpectedTensor {
                name: format!("{prefix}.post_attention_layernorm.weight"),
                shape: vec![hidden],
            },
            ExpectedTensor {
                name: format!("{prefix}.mlp.gate_proj.weight"),
                shape: vec![layout.intermediate_size, hidden],
            },
            ExpectedTensor {
                name: format!("{prefix}.mlp.up_proj.weight"),
                shape: vec![layout.intermediate_size, hidden],
            },
            ExpectedTensor {
                name: format!("{prefix}.mlp.down_proj.weight"),
                shape: vec![hidden, layout.intermediate_size],
            },
        ]);
    }
    expected.push(ExpectedTensor {
        name: "model.norm.weight".to_owned(),
        shape: vec![hidden],
    });
    if !layout.tie_word_embeddings {
        expected.push(ExpectedTensor {
            name: "lm_head.weight".to_owned(),
            shape: vec![layout.vocab_size, hidden],
        });
    }
    Ok(expected)
}

/// A local Qwen3 checkpoint that cannot be safely inspected.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Qwen3CheckpointError {
    /// The model root did not name a directory.
    #[error("Qwen3 model path is not a directory: {0}")]
    NotDirectory(PathBuf),
    /// The model configuration could not be read.
    #[error("could not read Qwen3 configuration {path}: {source}")]
    ReadConfig {
        /// Path that failed to read.
        path: PathBuf,
        /// I/O failure.
        source: std::io::Error,
    },
    /// The model configuration is not a supported Qwen3 text layout.
    #[error("invalid Qwen3 configuration: {0}")]
    Config(#[source] Qwen3ConfigError),
    /// The checkpoint-only configuration fields were not valid JSON.
    #[error("invalid Qwen3 checkpoint configuration JSON: {0}")]
    CheckpointConfigJson(#[source] serde_json::Error),
    /// A dimension needed to validate checkpoint tensor shapes was absent.
    #[error("Qwen3 checkpoint configuration has no usable {0}")]
    MissingCheckpointDimension(&'static str),
    /// Attention dimensions cannot produce a nonzero head layout.
    #[error("Qwen3 checkpoint has an invalid attention layout")]
    InvalidAttentionLayout,
    /// The model root could not be enumerated.
    #[error("could not read model directory {path}: {source}")]
    ReadDirectory {
        /// Directory that failed to enumerate.
        path: PathBuf,
        /// I/O failure.
        source: std::io::Error,
    },
    /// One model-directory entry could not be read.
    #[error("could not read a model-directory entry: {0}")]
    DirectoryEntry(std::io::Error),
    /// A candidate shard's type could not be read.
    #[error("could not inspect candidate shard {path}: {source}")]
    FileType {
        /// Candidate path.
        path: PathBuf,
        /// I/O failure.
        source: std::io::Error,
    },
    /// The model root contained no regular safetensors shard.
    #[error("Qwen3 model directory has no safetensors shards: {0}")]
    NoSafetensors(PathBuf),
    /// A shard could not be opened.
    #[error("could not open safetensors shard {path}: {source}")]
    OpenShard {
        /// Shard path.
        path: PathBuf,
        /// I/O failure.
        source: std::io::Error,
    },
    /// A shard's metadata could not be read.
    #[error("could not read safetensors shard metadata {path}: {source}")]
    ShardMetadata {
        /// Shard path.
        path: PathBuf,
        /// I/O failure.
        source: std::io::Error,
    },
    /// The safetensors prefix was incomplete.
    #[error("safetensors shard has an incomplete prefix: {0}")]
    TruncatedPrefix(PathBuf),
    /// The claimed header cannot fit safely in the shard.
    #[error("invalid safetensors header length {header_bytes} for {path} ({file_bytes} bytes)")]
    InvalidHeaderLength {
        /// Shard path.
        path: PathBuf,
        /// Declared header length.
        header_bytes: u64,
        /// On-disk file length.
        file_bytes: u64,
    },
    /// The header was not valid JSON.
    #[error("invalid safetensors header JSON in {path}: {source}")]
    HeaderJson {
        /// Shard path.
        path: PathBuf,
        /// JSON failure.
        source: serde_json::Error,
    },
    /// Reserved safetensors metadata was not a JSON object.
    #[error("invalid safetensors metadata in {0}")]
    InvalidMetadata(PathBuf),
    /// A non-metadata header entry did not describe a tensor.
    #[error("invalid tensor header for {tensor:?} in {path}: {source}")]
    InvalidTensorHeader {
        /// Shard path.
        path: PathBuf,
        /// Tensor name.
        tensor: String,
        /// JSON failure.
        source: serde_json::Error,
    },
    /// A tensor did not declare exactly one start/end range.
    #[error("invalid data offsets for tensor {tensor:?} in {path}")]
    InvalidTensorOffsets {
        /// Shard path.
        path: PathBuf,
        /// Tensor name.
        tensor: String,
    },
    /// A tensor range was malformed or outside the shard payload.
    #[error("tensor {tensor:?} is outside the safetensors payload in {path}")]
    TensorOutsidePayload {
        /// Shard path.
        path: PathBuf,
        /// Tensor name.
        tensor: String,
    },
    /// A tensor's dimensions overflowed the element count.
    #[error("tensor {tensor:?} has an overflowing shape in {path}")]
    TensorShapeOverflow {
        /// Shard path.
        path: PathBuf,
        /// Tensor name.
        tensor: String,
    },
    /// A tensor uses a safetensors dtype this inspector does not understand.
    #[error("tensor {tensor:?} has unsupported safetensors dtype {dtype:?} in {path}")]
    UnsupportedTensorDtype {
        /// Shard path.
        path: PathBuf,
        /// Tensor name.
        tensor: String,
        /// Header dtype.
        dtype: String,
    },
    /// A tensor range did not match its dtype and shape.
    #[error("tensor {tensor:?} has {actual_bytes} bytes in {path}, expected {expected_bytes}")]
    TensorByteLengthMismatch {
        /// Shard path.
        path: PathBuf,
        /// Tensor name.
        tensor: String,
        /// Bytes implied by dtype and shape.
        expected_bytes: u64,
        /// Bytes named by the tensor range.
        actual_bytes: u64,
    },
    /// Two tensor ranges overlap in one shard.
    #[error("safetensors shard has overlapping tensor ranges: {0}")]
    OverlappingTensorRanges(PathBuf),
    /// Tensor ranges did not cover the shard payload exactly once.
    #[error("safetensors shard has holes or trailing payload bytes: {0}")]
    NonContiguousPayload(PathBuf),
    /// A shard header named an empty tensor.
    #[error("safetensors shard has a blank tensor name: {0}")]
    BlankTensorName(PathBuf),
    /// A tensor appears in more than one local shard.
    #[error("Qwen3 checkpoint repeats tensor {0:?}")]
    DuplicateTensorName(String),
    /// A required dense decoder tensor was absent.
    #[error("Qwen3 checkpoint is missing required tensor {0:?}")]
    MissingRequiredTensor(String),
    /// A required tensor did not match the configuration-derived shape.
    #[error("Qwen3 tensor {tensor:?} has shape {actual:?}, expected {expected:?}")]
    UnexpectedTensorShape {
        /// Tensor name.
        tensor: String,
        /// Shape inferred from `config.json`.
        expected: Vec<u64>,
        /// Shape declared by the safetensors header.
        actual: Vec<u64>,
    },
    /// The number of tensors exceeded the process counter.
    #[error("Qwen3 checkpoint tensor count overflowed")]
    TensorCountOverflow,
    /// The tensor-byte sum exceeded `u64`.
    #[error("Qwen3 checkpoint tensor bytes overflowed")]
    TensorBytesOverflow,
    /// A shard could not be read fully.
    #[error("could not read safetensors shard {path}: {source}")]
    ReadShard {
        /// Shard path.
        path: PathBuf,
        /// I/O failure.
        source: std::io::Error,
    },
    /// The requested name was not in the checkpoint validated by inspection.
    #[error("Qwen3 checkpoint has no validated tensor {0:?}")]
    UnknownTensor(String),
    /// The tensor would exceed the caller's explicit bounded-read budget.
    #[error("Qwen3 tensor {tensor:?} is {tensor_bytes} bytes, above read budget {max_bytes}")]
    TensorExceedsReadBudget {
        /// Requested tensor name.
        tensor: String,
        /// Requested payload byte count.
        tensor_bytes: u64,
        /// Maximum allocation authorized by the caller.
        max_bytes: u64,
    },
    /// The approved read budget cannot be represented by the process address space.
    #[error("Qwen3 tensor {tensor:?} is {tensor_bytes} bytes and cannot fit this address space")]
    ReadBudgetCannotFitAddressSpace {
        /// Requested tensor name.
        tensor: String,
        /// Validated payload byte count.
        tensor_bytes: u64,
    },
    /// A shard changed since its header and tensor layout were validated.
    #[error(
        "safetensors shard metadata drifted for {path}: expected {expected_bytes} bytes, found {actual_bytes}"
    )]
    ShardMetadataDrift {
        /// Shard whose file metadata no longer matches inspection.
        path: PathBuf,
        /// Byte length captured during inspection.
        expected_bytes: u64,
        /// Byte length observed while reading a selected tensor.
        actual_bytes: u64,
    },
    /// The selected payload range ended before all validated bytes were read.
    #[error("safetensors shard payload truncated while reading tensor {tensor:?} from {path}")]
    TensorPayloadTruncated {
        /// Shard containing the tensor.
        path: PathBuf,
        /// Requested tensor name.
        tensor: String,
    },
    /// A row read was requested for a tensor that is not a rank-two matrix.
    #[error("Qwen3 tensor {tensor:?} has shape {shape:?}; row reads require rank two")]
    TensorRowsRequireMatrix {
        /// Requested tensor name.
        tensor: String,
        /// Validated safetensors shape.
        shape: Vec<u64>,
    },
    /// A row read requires BF16 payload elements.
    #[error("Qwen3 tensor {tensor:?} has dtype {dtype:?}; row reads require BF16")]
    TensorRowsRequireBf16 {
        /// Requested tensor name.
        tensor: String,
        /// Validated safetensors dtype.
        dtype: String,
    },
    /// A row selection was empty or outside the validated matrix bounds.
    #[error(
        "Qwen3 tensor {tensor:?} row range {start}..{end} is outside its {row_count} rows or empty"
    )]
    InvalidTensorRowRange {
        /// Requested tensor name.
        tensor: String,
        /// Inclusive range start supplied by the caller.
        start: usize,
        /// Exclusive range end supplied by the caller.
        end: usize,
        /// Validated matrix row count.
        row_count: u64,
    },
    /// Validated matrix dimensions cannot produce a safe selected byte range.
    #[error("Qwen3 tensor {tensor:?} row layout overflowed")]
    TensorRowLayoutOverflow {
        /// Requested tensor name.
        tensor: String,
    },
}

#[cfg(test)]
mod tests;
