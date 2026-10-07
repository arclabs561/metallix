//! GGUF version 3: a little-endian header of typed metadata and tensor
//! records, then an aligned data section.
//!
//! Layout per llama.cpp `ggml/include/gguf.h` and `gguf-py/gguf/constants.py`
//! at commit 7fe450e19305b828c199d602c23a8337aaa1f03b: magic `GGUF`,
//! version, tensor count, key count, then each key (string, value type,
//! value), then each tensor (name, dimension count, dimensions innermost
//! first, `ggml` type, offset into the data section). Strings are a `u64`
//! length and bytes. The data section starts at the header's end rounded up
//! to `general.alignment` (default 32), and every tensor offset is a
//! multiple of it.
//!
//! The parser bounds every count and length before allocating, rejects
//! duplicate names, unknown types, nested arrays, overlapping or
//! out-of-file tensors, and keeps no payload in memory.

use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufReader, Read},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    time::SystemTime,
};

use crate::{CheckpointError, GgufEncoding, TensorInfo, TensorSource};

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;
/// `GGML_MAX_DIMS`.
const MAX_DIMS: u32 = 4;
const MAX_TENSORS: u64 = 1 << 20;
const MAX_KEYS: u64 = 1 << 16;
const MAX_STRING_BYTES: u64 = 16 << 20;
const MAX_ARRAY_ITEMS: u64 = 1 << 24;
/// Metadata read in total: vocabularies of a few hundred thousand tokens and
/// merges fit with room to spare.
const MAX_HEADER_BYTES: u64 = 512 << 20;

/// One metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    /// Stored U8 values.
    U8(u8),
    /// Stored I8 values.
    I8(i8),
    /// Stored U16 values.
    U16(u16),
    /// Stored I16 values.
    I16(i16),
    /// Stored U32 values.
    U32(u32),
    /// Stored I32 values.
    I32(i32),
    /// Stored F32 values.
    F32(f32),
    /// Stored Bool values.
    Bool(bool),
    /// Stored String values.
    String(String),
    /// Stored Array values.
    Array(GgufArray),
    /// Stored U64 values.
    U64(u64),
    /// Stored I64 values.
    I64(i64),
    /// Stored F64 values.
    F64(f64),
}

/// A homogeneous metadata array.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufArray {
    /// Homogeneous array of U8 values.
    U8(Vec<u8>),
    /// Homogeneous array of I8 values.
    I8(Vec<i8>),
    /// Homogeneous array of U16 values.
    U16(Vec<u16>),
    /// Homogeneous array of I16 values.
    I16(Vec<i16>),
    /// Homogeneous array of U32 values.
    U32(Vec<u32>),
    /// Homogeneous array of I32 values.
    I32(Vec<i32>),
    /// Homogeneous array of F32 values.
    F32(Vec<f32>),
    /// Homogeneous array of Bool values.
    Bool(Vec<bool>),
    /// Homogeneous array of String values.
    String(Vec<String>),
    /// Homogeneous array of U64 values.
    U64(Vec<u64>),
    /// Homogeneous array of I64 values.
    I64(Vec<i64>),
    /// Homogeneous array of F64 values.
    F64(Vec<f64>),
}

impl GgufArray {
    /// Item count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(items) => items.len(),
            Self::I8(items) => items.len(),
            Self::U16(items) => items.len(),
            Self::I16(items) => items.len(),
            Self::U32(items) => items.len(),
            Self::I32(items) => items.len(),
            Self::F32(items) => items.len(),
            Self::Bool(items) => items.len(),
            Self::String(items) => items.len(),
            Self::U64(items) => items.len(),
            Self::I64(items) => items.len(),
            Self::F64(items) => items.len(),
        }
    }

    #[must_use]
    /// Whether the array contains no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Integer items as `i64`, for any integer item type.
    #[must_use]
    pub fn integers(&self) -> Option<Vec<i64>> {
        Some(match self {
            Self::U8(items) => items.iter().map(|&v| i64::from(v)).collect(),
            Self::I8(items) => items.iter().map(|&v| i64::from(v)).collect(),
            Self::U16(items) => items.iter().map(|&v| i64::from(v)).collect(),
            Self::I16(items) => items.iter().map(|&v| i64::from(v)).collect(),
            Self::U32(items) => items.iter().map(|&v| i64::from(v)).collect(),
            Self::I32(items) => items.iter().map(|&v| i64::from(v)).collect(),
            Self::I64(items) => items.clone(),
            Self::U64(items) => items
                .iter()
                .map(|&v| i64::try_from(v).ok())
                .collect::<Option<_>>()?,
            Self::Bool(items) => items.iter().map(|&v| i64::from(v)).collect(),
            _ => return None,
        })
    }
}

/// The key-value metadata of a GGUF file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GgufMetadata {
    values: HashMap<String, GgufValue>,
}

impl GgufMetadata {
    /// The value stored under `key`.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.values.get(key)
    }

    /// Every key, sorted.
    #[must_use]
    pub fn keys(&self) -> Vec<&str> {
        let mut keys: Vec<&str> = self.values.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys
    }

    /// An unsigned integer of any width, or a nonnegative signed one.
    #[must_use]
    pub fn unsigned(&self, key: &str) -> Option<u64> {
        match self.get(key)? {
            GgufValue::U8(v) => Some(u64::from(*v)),
            GgufValue::U16(v) => Some(u64::from(*v)),
            GgufValue::U32(v) => Some(u64::from(*v)),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) => u64::try_from(*v).ok(),
            GgufValue::I16(v) => u64::try_from(*v).ok(),
            GgufValue::I32(v) => u64::try_from(*v).ok(),
            GgufValue::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// [`Self::unsigned`] as `usize`, or a [`CheckpointError::Metadata`].
    ///
    /// # Errors
    /// The key is absent, not an integer, negative, or too large for usize.
    pub fn require_usize(&self, key: &str) -> Result<usize, CheckpointError> {
        self.unsigned(key)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| missing(key, "an unsigned integer"))
    }

    /// A float stored as `f32` or `f64` (narrowed).
    #[must_use]
    pub fn float(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            GgufValue::F32(v) => Some(f64::from(*v)),
            GgufValue::F64(v) => Some(*v),
            _ => None,
        }
    }

    /// A boolean.
    #[must_use]
    pub fn boolean(&self, key: &str) -> Option<bool> {
        match self.get(key)? {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// A string.
    #[must_use]
    pub fn string(&self, key: &str) -> Option<&str> {
        match self.get(key)? {
            GgufValue::String(v) => Some(v),
            _ => None,
        }
    }

    /// An array of any item type.
    #[must_use]
    pub fn array(&self, key: &str) -> Option<&GgufArray> {
        match self.get(key)? {
            GgufValue::Array(v) => Some(v),
            _ => None,
        }
    }

    /// An array of strings.
    #[must_use]
    pub fn strings(&self, key: &str) -> Option<&[String]> {
        match self.array(key)? {
            GgufArray::String(v) => Some(v),
            _ => None,
        }
    }

    /// `general.architecture`, which prefixes the architecture's own keys.
    ///
    /// # Errors
    /// The architecture key is absent or not a string.
    pub fn architecture(&self) -> Result<&str, CheckpointError> {
        self.string("general.architecture")
            .ok_or_else(|| missing("general.architecture", "a string"))
    }

    /// The `general.base_model.N.*` entries, in order.
    #[must_use]
    pub fn base_models(&self) -> Vec<BaseModel> {
        let count = self.unsigned("general.base_model.count").unwrap_or(0);
        (0..count)
            .map(|index| {
                let field = |name: &str| {
                    self.string(&format!("general.base_model.{index}.{name}"))
                        .map(str::to_owned)
                };
                BaseModel {
                    name: field("name"),
                    organization: field("organization"),
                    repo_url: field("repo_url"),
                }
            })
            .collect()
    }

    /// The embedded tokenizer and chat template.
    ///
    /// # Errors
    /// Required tokenizer metadata is absent, mistyped, or contains an invalid token ID.
    pub fn tokenizer(&self) -> Result<GgufTokenizer, CheckpointError> {
        let id = |key: &str| -> Result<Option<u32>, CheckpointError> {
            self.get(key)
                .map(|_| {
                    self.unsigned(key)
                        .and_then(|value| u32::try_from(value).ok())
                        .ok_or_else(|| missing(key, "a token id"))
                })
                .transpose()
        };
        Ok(GgufTokenizer {
            model: self
                .string("tokenizer.ggml.model")
                .ok_or_else(|| missing("tokenizer.ggml.model", "a string"))?
                .to_owned(),
            pre: self.string("tokenizer.ggml.pre").map(str::to_owned),
            tokens: self
                .strings("tokenizer.ggml.tokens")
                .ok_or_else(|| missing("tokenizer.ggml.tokens", "a string array"))?
                .to_vec(),
            merges: self
                .strings("tokenizer.ggml.merges")
                .map(<[String]>::to_vec)
                .unwrap_or_default(),
            bos: id("tokenizer.ggml.bos_token_id")?,
            eos: id("tokenizer.ggml.eos_token_id")?,
            padding: id("tokenizer.ggml.padding_token_id")?,
            add_bos: self.boolean("tokenizer.ggml.add_bos_token"),
            chat_template: self.string("tokenizer.chat_template").map(str::to_owned),
        })
    }
}

impl FromIterator<(String, GgufValue)> for GgufMetadata {
    fn from_iter<I: IntoIterator<Item = (String, GgufValue)>>(values: I) -> Self {
        Self {
            values: values.into_iter().collect(),
        }
    }
}

fn missing(key: &str, expected: &'static str) -> CheckpointError {
    CheckpointError::Metadata {
        key: key.to_owned(),
        expected,
    }
}

/// One `general.base_model.N` entry: the model a file was derived from, as
/// its converter recorded it (from the model card). No revision is stored.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BaseModel {
    /// Human-readable base model name.
    pub name: Option<String>,
    /// Publisher recorded by the converter.
    pub organization: Option<String>,
    /// Source repository recorded by the converter.
    pub repo_url: Option<String>,
}

/// The tokenizer a GGUF file embeds (`tokenizer.ggml.*`) and its chat
/// template. The pre-tokenizer is only a name (`pre`) for a split rule that
/// lives in llama.cpp's code, not in the file.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GgufTokenizer {
    /// `gpt2` for byte-level BPE, `llama` for `SentencePiece`, and so on.
    pub model: String,
    /// Named llama.cpp pre-tokenizer rule.
    pub pre: Option<String>,
    /// Token spellings indexed by ID.
    pub tokens: Vec<String>,
    /// BPE merges as `"left right"`, in priority order.
    pub merges: Vec<String>,
    /// Beginning-of-sequence token ID.
    pub bos: Option<u32>,
    /// End-of-sequence token ID.
    pub eos: Option<u32>,
    /// Padding token ID.
    pub padding: Option<u32>,
    /// Whether the converter requests automatic BOS insertion.
    pub add_bos: Option<bool>,
    /// Embedded Jinja chat template.
    pub chat_template: Option<String>,
}

/// An inspected GGUF file: metadata in memory, tensor payloads on disk.
#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    length: u64,
    modified: SystemTime,
    metadata: GgufMetadata,
    order: Vec<String>,
    tensors: HashMap<String, TensorInfo>,
}

impl GgufFile {
    /// Reads and validates the header of the GGUF file at `path`.
    ///
    /// # Errors
    /// The file is unreadable, malformed, exceeds parser bounds, or has invalid tensor ranges.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let path = path.as_ref().to_owned();
        let io = |source| CheckpointError::Io {
            path: path.clone(),
            source,
        };
        let file = File::open(&path).map_err(io)?;
        let stat = file.metadata().map_err(io)?;
        if !stat.is_file() {
            return Err(CheckpointError::NotGguf(path));
        }
        let mut reader = Reader {
            inner: BufReader::new(file),
            read: 0,
            path: path.clone(),
        };
        let mut magic = [0; 4];
        reader.bytes(&mut magic)?;
        if &magic != MAGIC {
            return Err(CheckpointError::NotGguf(path));
        }
        let version = reader.u32()?;
        if version != 3 {
            return Err(CheckpointError::Version(version));
        }
        let tensor_count = bounded("tensor count", reader.u64()?, MAX_TENSORS)?;
        let key_count = bounded("key count", reader.u64()?, MAX_KEYS)?;

        let mut values = HashMap::new();
        for _ in 0..key_count {
            let key = reader.string("metadata key")?;
            let kind = reader.u32()?;
            let value = reader.value(&key, kind)?;
            if values.insert(key.clone(), value).is_some() {
                return Err(CheckpointError::DuplicateKey(key));
            }
        }
        let metadata = GgufMetadata { values };
        let alignment = match metadata.get("general.alignment") {
            None => DEFAULT_ALIGNMENT,
            Some(_) => metadata
                .unsigned("general.alignment")
                .ok_or_else(|| missing("general.alignment", "an unsigned integer"))?,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CheckpointError::Alignment(alignment));
        }

        let mut records = Vec::new();
        for _ in 0..tensor_count {
            let name = reader.string("tensor name")?;
            let dims = reader.u32()?;
            if dims == 0 || dims > MAX_DIMS {
                return Err(CheckpointError::Limit {
                    what: "tensor dimension count",
                    value: u64::from(dims),
                    limit: u64::from(MAX_DIMS),
                });
            }
            let mut shape = (0..dims)
                .map(|_| reader.u64())
                .collect::<Result<Vec<_>, _>>()?;
            // GGUF writes the innermost dimension first.
            shape.reverse();
            let id = reader.u32()?;
            let encoding = GgufEncoding::from_id(id).ok_or_else(|| CheckpointError::Encoding {
                tensor: name.clone(),
                id,
            })?;
            let offset = reader.u64()?;
            records.push((name, shape, encoding, offset));
        }

        let data_start = reader
            .read
            .checked_next_multiple_of(alignment)
            .ok_or_else(|| CheckpointError::Truncated(path.clone()))?;
        let length = stat.len();
        let (order, tensors) = tensor_table(records, data_start, alignment, length)?;
        Ok(Self {
            path,
            length,
            modified: stat.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            metadata,
            order,
            tensors,
        })
    }

    /// The file's metadata.
    #[must_use]
    pub const fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// The inspected path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn unchanged(&self) -> Result<(), CheckpointError> {
        let stat = fs::metadata(&self.path).map_err(|source| CheckpointError::Io {
            path: self.path.clone(),
            source,
        })?;
        if stat.len() != self.length
            || stat.modified().unwrap_or(SystemTime::UNIX_EPOCH) != self.modified
        {
            return Err(CheckpointError::Changed(self.path.clone()));
        }
        Ok(())
    }
}

impl TensorSource for GgufFile {
    fn names(&self) -> Vec<&str> {
        self.order.iter().map(String::as_str).collect()
    }

    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Checks the file's length and modification time against inspection
    /// before and after the read. That catches replacement and truncation,
    /// not a concurrent in-place write that preserves both.
    fn read(&self, name: &str, max_bytes: u64) -> Result<Vec<u8>, CheckpointError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))?;
        let bytes = info.range.end - info.range.start;
        if bytes > max_bytes {
            return Err(CheckpointError::TooLarge {
                tensor: name.to_owned(),
                bytes,
                max_bytes,
            });
        }
        self.unchanged()?;
        let io = |source| CheckpointError::Io {
            path: self.path.clone(),
            source,
        };
        let file = File::open(&self.path).map_err(io)?;
        let mut payload = vec![
            0;
            usize::try_from(bytes).map_err(|_| CheckpointError::TooLarge {
                tensor: name.to_owned(),
                bytes,
                max_bytes,
            })?
        ];
        file.read_exact_at(&mut payload, info.range.start)
            .map_err(io)?;
        self.unchanged()?;
        Ok(payload)
    }
}

/// Tensor records as `(name, shape outermost first, encoding, offset)`.
type Records = Vec<(String, Vec<u64>, GgufEncoding, u64)>;

/// Validates each record's shape, alignment and extent, and that no two
/// tensors share a name or bytes. Returns names in file order and the table.
fn tensor_table(
    records: Records,
    data_start: u64,
    alignment: u64,
    length: u64,
) -> Result<(Vec<String>, HashMap<String, TensorInfo>), CheckpointError> {
    let mut order = Vec::with_capacity(records.len());
    let mut tensors = HashMap::with_capacity(records.len());
    for (name, shape, encoding, offset) in records {
        let elements = shape.iter().try_fold(1_u64, |total, &dim| {
            (dim > 0).then(|| total.checked_mul(dim)).flatten()
        });
        // A row must hold whole blocks, so a block never spans two rows.
        let row_blocks = shape
            .last()
            .is_some_and(|&inner| inner.is_multiple_of(u64::from(encoding.block_elements())));
        let bytes = elements
            .filter(|_| row_blocks)
            .and_then(|elements| encoding.byte_len(elements))
            .ok_or_else(|| CheckpointError::Shape {
                tensor: name.clone(),
                shape: shape.clone(),
                encoding,
            })?;
        let range = data_start
            .checked_add(offset)
            .and_then(|start| Some(start..start.checked_add(bytes)?))
            .filter(|range| offset.is_multiple_of(alignment) && range.end <= length)
            .ok_or_else(|| CheckpointError::Range(name.clone()))?;
        let info = TensorInfo {
            encoding,
            shape,
            range,
        };
        if tensors.insert(name.clone(), info).is_some() {
            return Err(CheckpointError::DuplicateTensor(name));
        }
        order.push(name);
    }
    let mut spans: Vec<(&str, &TensorInfo)> = tensors
        .iter()
        .map(|(name, info)| (name.as_str(), info))
        .collect();
    spans.sort_by_key(|(_, info)| (info.range.start, info.range.end));
    for pair in spans.windows(2) {
        if pair[0].1.range.end > pair[1].1.range.start {
            return Err(CheckpointError::Overlap(
                pair[0].0.to_owned(),
                pair[1].0.to_owned(),
            ));
        }
    }
    Ok((order, tensors))
}

fn bounded(what: &'static str, value: u64, limit: u64) -> Result<u64, CheckpointError> {
    if value > limit {
        return Err(CheckpointError::Limit { what, value, limit });
    }
    Ok(value)
}

struct Reader {
    inner: BufReader<File>,
    read: u64,
    path: PathBuf,
}

impl Reader {
    fn bytes(&mut self, into: &mut [u8]) -> Result<(), CheckpointError> {
        let count = into.len() as u64;
        bounded("header size", self.read + count, MAX_HEADER_BYTES)?;
        self.inner.read_exact(into).map_err(|source| {
            if source.kind() == std::io::ErrorKind::UnexpectedEof {
                CheckpointError::Truncated(self.path.clone())
            } else {
                CheckpointError::Io {
                    path: self.path.clone(),
                    source,
                }
            }
        })?;
        self.read += count;
        Ok(())
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], CheckpointError> {
        let mut bytes = [0; N];
        self.bytes(&mut bytes)?;
        Ok(bytes)
    }

    fn u32(&mut self) -> Result<u32, CheckpointError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, CheckpointError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn string(&mut self, what: &str) -> Result<String, CheckpointError> {
        let length = bounded("string length", self.u64()?, MAX_STRING_BYTES)?;
        let mut bytes = vec![0; usize::try_from(length).expect("bounded")];
        self.bytes(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| CheckpointError::Utf8(what.to_owned()))
    }

    fn value(&mut self, key: &str, kind: u32) -> Result<GgufValue, CheckpointError> {
        Ok(match kind {
            0 => GgufValue::U8(u8::from_le_bytes(self.array()?)),
            1 => GgufValue::I8(i8::from_le_bytes(self.array()?)),
            2 => GgufValue::U16(u16::from_le_bytes(self.array()?)),
            3 => GgufValue::I16(i16::from_le_bytes(self.array()?)),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(i32::from_le_bytes(self.array()?)),
            6 => GgufValue::F32(f32::from_le_bytes(self.array()?)),
            7 => GgufValue::Bool(self.boolean(key)?),
            8 => GgufValue::String(self.string(key)?),
            9 => GgufValue::Array(self.items(key)?),
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(i64::from_le_bytes(self.array()?)),
            12 => GgufValue::F64(f64::from_le_bytes(self.array()?)),
            id => {
                return Err(CheckpointError::ValueType {
                    key: key.to_owned(),
                    id,
                });
            }
        })
    }

    fn boolean(&mut self, key: &str) -> Result<bool, CheckpointError> {
        match self.array::<1>()?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(missing(key, "a boolean of 0 or 1")),
        }
    }

    fn items(&mut self, key: &str) -> Result<GgufArray, CheckpointError> {
        let kind = self.u32()?;
        let count = bounded("array length", self.u64()?, MAX_ARRAY_ITEMS)?;
        let count = usize::try_from(count).expect("bounded");
        macro_rules! collect {
            ($variant:ident, $read:expr) => {
                GgufArray::$variant((0..count).map(|_| $read).collect::<Result<_, _>>()?)
            };
        }
        Ok(match kind {
            0 => collect!(U8, self.array().map(u8::from_le_bytes)),
            1 => collect!(I8, self.array().map(i8::from_le_bytes)),
            2 => collect!(U16, self.array().map(u16::from_le_bytes)),
            3 => collect!(I16, self.array().map(i16::from_le_bytes)),
            4 => collect!(U32, self.u32()),
            5 => collect!(I32, self.array().map(i32::from_le_bytes)),
            6 => collect!(F32, self.array().map(f32::from_le_bytes)),
            7 => collect!(Bool, self.boolean(key)),
            8 => collect!(String, self.string(key)),
            9 => return Err(CheckpointError::NestedArray(key.to_owned())),
            10 => collect!(U64, self.u64()),
            11 => collect!(I64, self.array().map(i64::from_le_bytes)),
            12 => collect!(F64, self.array().map(f64::from_le_bytes)),
            id => {
                return Err(CheckpointError::ValueType {
                    key: key.to_owned(),
                    id,
                });
            }
        })
    }
}

#[cfg(test)]
mod tests;
