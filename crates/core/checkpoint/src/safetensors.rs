//! Safetensors checkpoints: one file ([`SafetensorsFile`]) or the shards an
//! index names ([`SafetensorsCheckpoint`]), read without loading payloads.
//!
//! A file is an 8-byte little-endian header length, a JSON header mapping
//! each tensor name to its `dtype`, `shape` and `data_offsets` (relative to
//! the end of the header), then the payloads. Each file stays open after
//! inspection, so range reads are one `pread` each and see the inode that
//! was inspected even if the path is replaced. A concurrent in-place write is
//! not detected.

use std::{
    collections::{HashMap, hash_map::Entry},
    fs::{self, File},
    io::Read,
    ops::Range,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::{
    CheckpointError, DType, ReadCache, TensorEncoding, TensorInfo, TensorSource, check_range,
    mlx::MlxQuantization,
};

/// Largest JSON header read (the 35B mlx-community shards are under 200 KiB).
const MAX_HEADER_BYTES: u64 = 100 << 20;

/// An inspected safetensors file: the tensor table in memory, payloads on
/// disk behind a held file handle.
#[derive(Debug)]
pub struct SafetensorsFile {
    path: PathBuf,
    file: File,
    order: Vec<String>,
    tensors: HashMap<String, TensorInfo>,
}

impl SafetensorsFile {
    /// Reads and validates the header of the safetensors file at `path`,
    /// keeping the file open for later reads.
    ///
    /// # Errors
    /// The file is unreadable, its header is malformed or larger than the
    /// parser bound, a dtype is unknown, or a payload range is inconsistent
    /// with its shape, outside the file, or overlaps another.
    pub fn open(path: impl AsRef<Path>, cache: ReadCache) -> Result<Self, CheckpointError> {
        let path = path.as_ref().to_owned();
        let io = |source| CheckpointError::Io {
            path: path.clone(),
            source,
        };
        let mut file = File::open(&path).map_err(io)?;
        let length = file.metadata().map_err(io)?.len();
        let not = || CheckpointError::NotSafetensors(path.clone());
        let mut prefix = [0; 8];
        file.read_exact(&mut prefix).map_err(|_| not())?;
        let header_bytes = u64::from_le_bytes(prefix);
        if header_bytes > MAX_HEADER_BYTES || header_bytes.saturating_add(8) > length {
            return Err(not());
        }
        let mut header = vec![0; usize::try_from(header_bytes).map_err(|_| not())?];
        file.read_exact(&mut header).map_err(io)?;
        let Ok(HeaderEntries(entries)) = serde_json::from_slice::<HeaderEntries>(&header) else {
            return Err(not());
        };
        let data_start = 8 + header_bytes;
        let mut spans = Vec::with_capacity(entries.len());
        let mut tensors = HashMap::with_capacity(entries.len());
        for (name, entry) in entries {
            if name == "__metadata__" {
                continue;
            }
            let info = tensor_info(&name, &entry, data_start, length)?;
            spans.push((info.range(), name.clone()));
            match tensors.entry(name) {
                Entry::Occupied(slot) => {
                    return Err(CheckpointError::DuplicateTensor(slot.key().clone()));
                }
                Entry::Vacant(slot) => {
                    slot.insert(info);
                }
            }
        }
        spans.sort_by_key(|(range, _)| (range.start, range.end));
        for pair in spans.windows(2) {
            if pair[1].0.start < pair[0].0.end {
                return Err(CheckpointError::Overlap(
                    pair[0].1.clone(),
                    pair[1].1.clone(),
                ));
            }
        }
        cache.apply(&file, &path)?;
        Ok(Self {
            path,
            file,
            order: spans.into_iter().map(|(_, name)| name).collect(),
            tensors,
        })
    }

    /// The inspected path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// The header's top-level entries in file order, duplicates included. The
/// format disallows duplicate keys, and a JSON map would silently keep only
/// the last, so [`SafetensorsFile::open`] checks the decoded names itself.
struct HeaderEntries(Vec<(String, Value)>);

impl<'de> serde::Deserialize<'de> for HeaderEntries {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries;

        impl<'de> serde::de::Visitor<'de> for Entries {
            type Value = HeaderEntries;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut entries = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(HeaderEntries(entries))
            }
        }

        deserializer.deserialize_map(Entries)
    }
}

/// One header entry as a [`TensorInfo`] with an absolute payload range.
fn tensor_info(
    name: &str,
    entry: &Value,
    data_start: u64,
    length: u64,
) -> Result<TensorInfo, CheckpointError> {
    let invalid = || CheckpointError::Range(name.to_owned());
    let dtype_name = entry
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(invalid)?;
    let dtype = dtype(dtype_name).ok_or_else(|| CheckpointError::UnknownDType {
        tensor: name.to_owned(),
        dtype: dtype_name.to_owned(),
    })?;
    let shape = entry
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(invalid)?
        .iter()
        .map(Value::as_u64)
        .collect::<Option<Vec<u64>>>()
        .ok_or_else(invalid)?;
    let offsets = entry
        .get("data_offsets")
        .and_then(Value::as_array)
        .filter(|offsets| offsets.len() == 2)
        .ok_or_else(invalid)?;
    let (Some(begin), Some(end)) = (offsets[0].as_u64(), offsets[1].as_u64()) else {
        return Err(invalid());
    };
    let bytes = shape
        .iter()
        .try_fold(1_u64, |total, &dim| total.checked_mul(dim))
        .and_then(|elements| elements.checked_mul(dtype.size()));
    let range = data_start
        .checked_add(begin)
        .zip(data_start.checked_add(end))
        .map(|(start, end)| start..end)
        .filter(|range| {
            range.start <= range.end
                && range.end <= length
                && bytes == Some(range.end - range.start)
        })
        .ok_or_else(invalid)?;
    Ok(TensorInfo::new(TensorEncoding::Plain(dtype), shape, range))
}

fn dtype(name: &str) -> Option<DType> {
    Some(match name {
        "F64" => DType::F64,
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I64" => DType::I64,
        "I32" => DType::I32,
        "I16" => DType::I16,
        "I8" => DType::I8,
        "U64" => DType::U64,
        "U32" => DType::U32,
        "U16" => DType::U16,
        "U8" => DType::U8,
        "BOOL" => DType::Bool,
        "F8_E4M3" => DType::F8E4M3,
        "F8_E8M0" => DType::F8E8M0,
        _ => return None,
    })
}

impl TensorSource for SafetensorsFile {
    fn names(&self) -> Vec<&str> {
        self.order.iter().map(String::as_str).collect()
    }

    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    fn read(&self, name: &str, max_bytes: u64) -> Result<Vec<u8>, CheckpointError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))?;
        let range = info.range();
        let bytes = range.end - range.start;
        let too_large = || CheckpointError::TooLarge {
            tensor: name.to_owned(),
            bytes,
            max_bytes,
        };
        if bytes > max_bytes {
            return Err(too_large());
        }
        let mut payload = vec![0; usize::try_from(bytes).map_err(|_| too_large())?];
        self.read_at(range.start, &mut payload)?;
        Ok(payload)
    }

    fn read_range(
        &self,
        name: &str,
        range: Range<u64>,
        dst: &mut [u8],
    ) -> Result<(), CheckpointError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))?;
        check_range(name, info, &range, dst.len())?;
        self.read_at(range.start, dst)
    }
}

impl SafetensorsFile {
    /// One positional read on the held handle.
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), CheckpointError> {
        self.file
            .read_exact_at(dst, offset)
            .map_err(|source| CheckpointError::Io {
                path: self.path.clone(),
                source,
            })
    }
}

/// A checkpoint directory: `model.safetensors`, or every shard that
/// `model.safetensors.index.json` names, with MLX affine encodings applied
/// from `config.json`'s `quantization` when present.
#[derive(Debug)]
pub struct SafetensorsCheckpoint {
    shards: Vec<SafetensorsFile>,
    /// Name to shard index and the tensor as this checkpoint describes it.
    tensors: HashMap<String, (usize, TensorInfo)>,
    order: Vec<String>,
    quantization: Option<MlxQuantization>,
}

impl SafetensorsCheckpoint {
    /// Opens every shard of the checkpoint in `dir` and reads `config.json`'s
    /// MLX `quantization`. Each `<stem>.weight` whose `<stem>.scales` is
    /// stored becomes [`TensorEncoding::Affine`] with the settings the
    /// configuration gives `<stem>`, after checking that the codes are `u32`,
    /// and that scales and biases have one column per group of the same
    /// logical width.
    ///
    /// # Errors
    /// An unsafe or missing shard name, a tensor in two shards, an index
    /// entry its shard does not store, an unreadable configuration or shard,
    /// a quantized module the configuration does not describe or leaves
    /// unquantized, or inconsistent affine shapes.
    pub fn open(dir: impl AsRef<Path>, cache: ReadCache) -> Result<Self, CheckpointError> {
        let dir = dir.as_ref();
        let quantization = match fs::read(dir.join("config.json")) {
            Ok(bytes) => {
                let config: Value = serde_json::from_slice(&bytes)
                    .map_err(|_| CheckpointError::InvalidQuantization("config.json"))?;
                MlxQuantization::from_config(
                    config.get("quantization"),
                    config.get("quantization_config"),
                )?
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(source) => {
                return Err(CheckpointError::Io {
                    path: dir.join("config.json"),
                    source,
                });
            }
        };
        let (paths, index) = shard_paths(dir)?;
        let mut shards = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::new();
        let mut order = Vec::new();
        for (shard, path) in paths.into_iter().enumerate() {
            let file = SafetensorsFile::open(&path, cache)?;
            for name in file.names() {
                let info = file.tensors[name].clone();
                match tensors.entry(name.to_owned()) {
                    Entry::Occupied(_) => {
                        return Err(CheckpointError::DuplicateTensor(name.to_owned()));
                    }
                    Entry::Vacant(slot) => {
                        slot.insert((shard, info));
                    }
                }
                order.push(name.to_owned());
            }
            shards.push(file);
        }
        for (name, shard_name) in index {
            let stored = tensors
                .get(&name)
                .map(|&(shard, _)| shards[shard].path().file_name());
            if stored != Some(Some(std::ffi::OsStr::new(&shard_name))) {
                return Err(CheckpointError::MissingTensor(name));
            }
        }
        let mut checkpoint = Self {
            shards,
            tensors,
            order,
            quantization,
        };
        checkpoint.apply_affine()?;
        Ok(checkpoint)
    }

    /// The configuration's MLX affine quantization, if any.
    #[must_use]
    pub const fn quantization(&self) -> Option<&MlxQuantization> {
        self.quantization.as_ref()
    }

    fn apply_affine(&mut self) -> Result<(), CheckpointError> {
        let stems: Vec<String> = self
            .order
            .iter()
            .filter_map(|name| name.strip_suffix(".scales"))
            .map(str::to_owned)
            .collect();
        for stem in stems {
            let affine = self
                .quantization
                .as_ref()
                .and_then(|quantization| quantization.module(&stem))
                .ok_or_else(|| CheckpointError::UndeclaredQuantization(stem.clone()))?;
            let part = |suffix: &str| {
                let name = format!("{stem}.{suffix}");
                self.tensors
                    .get(&name)
                    .map(|(_, info)| info.clone())
                    .ok_or(CheckpointError::MissingTensor(name))
            };
            let (codes, scales, biases) = (part("weight")?, part("scales")?, part("biases")?);
            for (suffix, info) in [("scales", &scales), ("biases", &biases)] {
                if !matches!(
                    info.encoding(),
                    TensorEncoding::Plain(DType::F16 | DType::BF16 | DType::F32)
                ) {
                    return Err(CheckpointError::AffineParameterDtype {
                        tensor: format!("{stem}.{suffix}"),
                        encoding: info.encoding(),
                    });
                }
            }
            // The scales fix the logical width; the codes must pack exactly
            // that many `bits`-wide values per row.
            let expected = scales
                .shape()
                .split_last()
                .and_then(|(&groups, outer)| {
                    let bits = groups
                        .checked_mul(u64::from(affine.group_size()))?
                        .checked_mul(u64::from(affine.bits()))?;
                    bits.is_multiple_of(32)
                        .then(|| [outer, &[bits / 32]].concat())
                })
                .unwrap_or_default();
            let consistent = codes.encoding() == TensorEncoding::Plain(DType::U32)
                && scales.encoding() == biases.encoding()
                && scales.shape() == biases.shape()
                && !expected.is_empty()
                && codes.shape() == expected.as_slice();
            if !consistent {
                return Err(CheckpointError::AffineShape {
                    module: stem,
                    expected,
                    codes: codes.shape().to_vec(),
                });
            }
            let name = format!("{stem}.weight");
            if let Some((_, info)) = self.tensors.get_mut(&name) {
                *info = TensorInfo::new(affine.encoding(), codes.shape().to_vec(), codes.range());
            }
        }
        Ok(())
    }

    fn shard_of(&self, name: &str) -> Result<(&SafetensorsFile, &TensorInfo), CheckpointError> {
        self.tensors
            .get(name)
            .map(|(shard, info)| (&self.shards[*shard], info))
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))
    }
}

/// The single-file checkpoint, or every shard named by the index with the
/// index's name-to-shard map. Shard names must be plain file names inside
/// `dir` ending in `.safetensors`.
/// Shard paths, and the index's `(tensor, shard file name)` entries.
type Shards = (Vec<PathBuf>, Vec<(String, String)>);

fn shard_paths(dir: &Path) -> Result<Shards, CheckpointError> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok((vec![single], Vec::new()));
    }
    let path = dir.join("model.safetensors.index.json");
    let bytes = fs::read(&path).map_err(|source| CheckpointError::Io {
        path: path.clone(),
        source,
    })?;
    let index: Value = serde_json::from_slice(&bytes)
        .map_err(|_| CheckpointError::NotSafetensors(path.clone()))?;
    let map = index
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| CheckpointError::NotSafetensors(path.clone()))?;
    let mut entries = Vec::with_capacity(map.len());
    for (name, shard) in map {
        let shard = shard
            .as_str()
            .ok_or_else(|| CheckpointError::NotSafetensors(path.clone()))?;
        entries.push((name.clone(), shard.to_owned()));
    }
    let mut names: Vec<&str> = entries.iter().map(|(_, shard)| shard.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    let paths = names
        .into_iter()
        .map(|name| {
            let flat = Path::new(name).file_name().is_some_and(|file| file == name)
                && name.ends_with(".safetensors");
            if flat {
                Ok(dir.join(name))
            } else {
                Err(CheckpointError::UnsafeShard(name.to_owned()))
            }
        })
        .collect::<Result<_, _>>()?;
    Ok((paths, entries))
}

impl TensorSource for SafetensorsCheckpoint {
    fn names(&self) -> Vec<&str> {
        self.order.iter().map(String::as_str).collect()
    }

    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|(_, info)| info)
    }

    fn read(&self, name: &str, max_bytes: u64) -> Result<Vec<u8>, CheckpointError> {
        self.shard_of(name)?.0.read(name, max_bytes)
    }

    fn read_range(
        &self,
        name: &str,
        range: Range<u64>,
        dst: &mut [u8],
    ) -> Result<(), CheckpointError> {
        self.shard_of(name)?.0.read_range(name, range, dst)
    }
}

#[cfg(test)]
mod tests;
