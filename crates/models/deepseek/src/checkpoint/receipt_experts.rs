//! Receipt-verified local routed tensors for the shared expert pager.

use std::{collections::BTreeMap, ops::Range};

use ::checkpoint::{CheckpointError, DType, TensorEncoding, TensorInfo, TensorSource};

use super::{
    V41SafetensorsHeader, V41StorageDtype, V41TensorRange,
    local_store::V41LocalStoreSource,
    range_cache::{V41RangeCacheError, V41RangeRequest, V41RangeSource},
};
use crate::manifest::V41SafetensorsIndex;

struct Entry {
    shard: String,
    range: V41TensorRange,
    info: TensorInfo,
}

/// Bridges admitted native routed-expert metadata to receipt-checked local reads.
///
/// Only tensors named `layers.*.ffn.experts.*` are exposed, with I8 packed codes
/// or E8M0 scales. Other tensors remain with the existing row/range providers.
/// This source owns no payload cache, performs no acquisition, and never writes
/// receipts or payloads. The caller must admit the index/header digests and
/// pinned revision before construction; supplied headers are checked against
/// the index again here. Shards without supplied headers remain unreadable.
pub struct V41ReceiptExpertSource {
    source: V41LocalStoreSource,
    entries: BTreeMap<String, Entry>,
}

impl V41ReceiptExpertSource {
    /// Builds the metadata bridge without reading payloads or receipts.
    ///
    /// # Errors
    /// Rejects header/index disagreement, duplicate headers, or unsupported
    /// storage dtypes on routed-expert tensors.
    pub fn new(
        source: V41LocalStoreSource,
        index: &V41SafetensorsIndex,
        headers: impl IntoIterator<Item = (String, V41SafetensorsHeader)>,
    ) -> Result<Self, V41RangeCacheError> {
        let mut entries = BTreeMap::new();
        let mut shards = std::collections::BTreeSet::new();
        for (shard, header) in headers {
            if !shards.insert(shard.clone()) {
                return Err(V41RangeCacheError::UnknownTensor(format!(
                    "duplicate shard {shard}"
                )));
            }
            header.validate_index_shard(index, &shard)?;
            for (name, range) in header.tensors() {
                if !name.starts_with("layers.") || !name.contains(".ffn.experts.") {
                    continue;
                }
                let dtype = match range.dtype() {
                    V41StorageDtype::I8 => DType::I8,
                    V41StorageDtype::F8E8M0Fnu => DType::F8E8M0,
                    _ => return Err(V41RangeCacheError::UnknownTensor(name.to_owned())),
                };
                entries.insert(
                    name.to_owned(),
                    Entry {
                        shard: shard.clone(),
                        range: range.clone(),
                        info: TensorInfo::new(
                            TensorEncoding::Plain(dtype),
                            range.shape().to_vec(),
                            range.file_range(),
                        ),
                    },
                );
            }
        }
        Ok(Self { source, entries })
    }

    fn entry(&self, name: &str) -> Result<&Entry, CheckpointError> {
        self.entries
            .get(name)
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))
    }

    fn read_checked(
        &self,
        name: &str,
        entry: &Entry,
        range: Range<u64>,
    ) -> Result<Vec<u8>, CheckpointError> {
        let bytes = self
            .source
            .read_range(&V41RangeRequest {
                tensor: name,
                shard: &entry.shard,
                tensor_range: &entry.range,
                range: range.clone(),
            })
            .map_err(|error| CheckpointError::Range(format!("{name}: {error}")))?;
        if u64::try_from(bytes.len()).ok() != range.end.checked_sub(range.start) {
            return Err(CheckpointError::Range(name.to_owned()));
        }
        Ok(bytes)
    }
}

impl TensorSource for V41ReceiptExpertSource {
    fn names(&self) -> Vec<&str> {
        self.entries.keys().map(String::as_str).collect()
    }

    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.entries.get(name).map(|entry| &entry.info)
    }

    fn read(&self, name: &str, max_bytes: u64) -> Result<Vec<u8>, CheckpointError> {
        let entry = self.entry(name)?;
        if entry.range.byte_length() > max_bytes {
            return Err(CheckpointError::Range(format!(
                "{name}: byte budget exceeded"
            )));
        }
        self.read_checked(name, entry, entry.range.file_range())
    }

    fn read_range(
        &self,
        name: &str,
        range: Range<u64>,
        dst: &mut [u8],
    ) -> Result<(), CheckpointError> {
        let entry = self.entry(name)?;
        let whole = entry.range.file_range();
        if range.start < whole.start
            || range.end > whole.end
            || range.end.checked_sub(range.start) != u64::try_from(dst.len()).ok()
        {
            return Err(CheckpointError::Range(name.to_owned()));
        }
        let bytes = self.read_checked(name, entry, range)?;
        dst.copy_from_slice(&bytes);
        Ok(())
    }
}
