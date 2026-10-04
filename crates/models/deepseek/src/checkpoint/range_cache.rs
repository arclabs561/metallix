//! Bounded in-memory LRU cache of verified DeepSeek-V4.1 checkpoint byte ranges.
//!
//! The cache resolves a tensor name through a validated index and shard
//! headers, asks a [`V41RangeSource`] for the exact absolute shard range, and
//! keeps the result under an explicit byte budget. Whole tensors (routed
//! experts) and row slices of two-dimensional tables (Engram embedding weight
//! and scale tables) share one keyspace: `(shard, absolute range)`.
//!
//! The crate performs no network I/O. [`V41LocalWeightsSource`] serves the
//! per-tensor files and receipts that the route-trace acquisition writes; a
//! network fetcher implements [`V41RangeSource`] outside this crate.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::Read,
    ops::Range,
    path::PathBuf,
    sync::Arc,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{V41SafetensorsHeader, V41SafetensorsHeaderError, V41StorageDtype, V41TensorRange};
use crate::manifest::V41SafetensorsIndex;

/// One exact byte-range request against a checkpoint shard.
#[derive(Clone, Debug)]
pub struct V41RangeRequest<'a> {
    /// Tensor that owns the range.
    pub tensor: &'a str,
    /// Index-assigned shard file name.
    pub shard: &'a str,
    /// The owning tensor's validated header interval.
    pub tensor_range: &'a V41TensorRange,
    /// Requested absolute shard range; lies within `tensor_range`.
    pub range: Range<u64>,
}

/// Supplies exact checkpoint bytes. Implementations return exactly
/// `request.range.end - request.range.start` bytes or an error; the cache
/// rechecks the length.
pub trait V41RangeSource {
    /// Reads the requested range.
    fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError>;
}

/// Serves ranges from route-trace `weights/<tensor>.bin` files, each checked
/// against its `<tensor>.receipt.json` (identity, header metadata, size and
/// SHA-256) before any byte is returned.
///
/// A row request still hashes the whole tensor file, because receipts only
/// digest whole tensors.
#[derive(Clone, Debug)]
pub struct V41LocalWeightsSource {
    dir: PathBuf,
    revision: String,
}

impl V41LocalWeightsSource {
    /// Serves receipts for `revision` from `dir`.
    pub fn new(dir: impl Into<PathBuf>, revision: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            revision: revision.into(),
        }
    }
}

#[derive(Deserialize)]
struct Receipt {
    tensor: String,
    shard: String,
    range: [u64; 2],
    bytes: u64,
    metadata: ReceiptMetadata,
    sha256: String,
    revision: String,
}

#[derive(Deserialize)]
struct ReceiptMetadata {
    dtype: String,
    shape: Vec<u64>,
}

impl V41RangeSource for V41LocalWeightsSource {
    fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        let tensor = request.tensor;
        // Tensor names become file names; refuse anything that could leave `dir`.
        if tensor.is_empty() || tensor.starts_with('.') || tensor.contains(['/', '\\']) {
            return Err(V41RangeCacheError::UnknownTensor(tensor.to_owned()));
        }
        let receipt_path = self.dir.join(format!("{tensor}.receipt.json"));
        let receipt: Receipt = match fs::read(&receipt_path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(V41RangeCacheError::ReceiptJson)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(V41RangeCacheError::NotLocal(tensor.to_owned()));
            }
            Err(error) => return Err(V41RangeCacheError::Io(error)),
        };
        let expected = request.tensor_range;
        let mismatch = |field| V41RangeCacheError::ReceiptMismatch {
            tensor: tensor.to_owned(),
            field,
        };
        if receipt.tensor != tensor {
            return Err(mismatch("tensor"));
        }
        if receipt.shard != request.shard {
            return Err(mismatch("shard"));
        }
        if receipt.revision != self.revision {
            return Err(mismatch("revision"));
        }
        if (receipt.range[0]..receipt.range[1]) != expected.file_range()
            || receipt.bytes != expected.byte_length()
        {
            return Err(mismatch("range"));
        }
        if V41StorageDtype::parse(&receipt.metadata.dtype) != Some(expected.dtype())
            || receipt.metadata.shape != expected.shape()
        {
            return Err(mismatch("metadata"));
        }

        let bin_path = self.dir.join(format!("{tensor}.bin"));
        let file = match fs::File::open(&bin_path) {
            Ok(file) => file,
            // A receipt without payload is an evicted tensor.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Err(V41RangeCacheError::NotLocal(tensor.to_owned()));
            }
            Err(error) => return Err(V41RangeCacheError::Io(error)),
        };
        let actual = file.metadata().map_err(V41RangeCacheError::Io)?.len();
        if actual != receipt.bytes {
            return Err(V41RangeCacheError::SizeMismatch {
                expected: receipt.bytes,
                actual,
            });
        }
        let length = usize::try_from(actual).map_err(|_| V41RangeCacheError::Allocation)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| V41RangeCacheError::Allocation)?;
        file.take(actual)
            .read_to_end(&mut bytes)
            .map_err(V41RangeCacheError::Io)?;
        if format!("{:x}", Sha256::digest(&bytes)) != receipt.sha256 {
            return Err(V41RangeCacheError::HashMismatch(tensor.to_owned()));
        }
        let base = expected.file_range().start;
        let start = usize::try_from(request.range.start - base)
            .map_err(|_| V41RangeCacheError::Allocation)?;
        let end = usize::try_from(request.range.end - base)
            .map_err(|_| V41RangeCacheError::Allocation)?;
        if start == 0 && end == bytes.len() {
            return Ok(bytes);
        }
        Ok(bytes[start..end].to_vec())
    }
}

type Key = (String, u64, u64);

/// LRU cache of verified checkpoint ranges under a fixed byte budget.
///
/// Results are `Arc<[u8]>`: eviction drops only the cache's reference, so a
/// caller's slice stays valid without a copy. Bytes a caller still holds after
/// eviction are no longer counted against the budget. Access takes `&mut self`;
/// wrap the cache in a `Mutex` to share it.
#[derive(Debug)]
pub struct V41RangeCache<S> {
    source: S,
    index: V41SafetensorsIndex,
    headers: BTreeMap<String, V41SafetensorsHeader>,
    budget_bytes: u64,
    used_bytes: u64,
    tick: u64,
    entries: HashMap<Key, (Arc<[u8]>, u64)>,
    order: BTreeMap<u64, Key>,
}

impl<S: V41RangeSource> V41RangeCache<S> {
    /// Builds a cache over `headers`, each checked against `index` for an
    /// exact tensor-name agreement. Shards without a header stay unreadable.
    pub fn new(
        source: S,
        index: V41SafetensorsIndex,
        headers: impl IntoIterator<Item = (String, V41SafetensorsHeader)>,
        budget_bytes: u64,
    ) -> Result<Self, V41RangeCacheError> {
        let headers: BTreeMap<_, _> = headers.into_iter().collect();
        for (shard, header) in &headers {
            header.validate_index_shard(&index, shard)?;
        }
        Ok(Self {
            source,
            index,
            headers,
            budget_bytes,
            used_bytes: 0,
            tick: 0,
            entries: HashMap::new(),
            order: BTreeMap::new(),
        })
    }

    /// Bytes currently held by the cache.
    #[must_use]
    pub const fn used_bytes(&self) -> u64 {
        self.used_bytes
    }

    /// Returns the cached or freshly read bytes of a whole named tensor.
    pub fn get_tensor(&mut self, tensor: &str) -> Result<Arc<[u8]>, V41RangeCacheError> {
        let (shard, range) = self.lookup(tensor)?;
        let file_range = range.file_range();
        self.get(tensor, &shard, &range, file_range)
    }

    /// Returns rows `rows` of a two-dimensional table such as
    /// `layers.N.engram.embed.weight` or its `.scale` companion. Row `r`
    /// occupies `r * row_bytes .. (r + 1) * row_bytes` of the tensor, with
    /// `row_bytes = shape[1] * dtype bytes`.
    pub fn get_rows(
        &mut self,
        table: &str,
        rows: Range<u64>,
    ) -> Result<Arc<[u8]>, V41RangeCacheError> {
        let (shard, range) = self.lookup(table)?;
        let byte_range = row_byte_range(&range, &rows).ok_or_else(|| {
            V41RangeCacheError::RowRangeOutOfBounds {
                table: table.to_owned(),
                rows: rows.clone(),
            }
        })?;
        self.get(table, &shard, &range, byte_range)
    }

    fn lookup(&self, tensor: &str) -> Result<(String, V41TensorRange), V41RangeCacheError> {
        let unknown = || V41RangeCacheError::UnknownTensor(tensor.to_owned());
        let shard = self.index.shard_for_tensor(tensor).ok_or_else(unknown)?;
        let header = self
            .headers
            .get(shard)
            .ok_or_else(|| V41RangeCacheError::MissingShardHeader(shard.to_owned()))?;
        let range = header.tensor(tensor).ok_or_else(unknown)?;
        Ok((shard.to_owned(), range.clone()))
    }

    fn get(
        &mut self,
        tensor: &str,
        shard: &str,
        tensor_range: &V41TensorRange,
        range: Range<u64>,
    ) -> Result<Arc<[u8]>, V41RangeCacheError> {
        let key = (shard.to_owned(), range.start, range.end);
        self.tick += 1;
        if let Some((bytes, last)) = self.entries.get_mut(&key) {
            self.order.remove(last);
            *last = self.tick;
            self.order.insert(self.tick, key);
            return Ok(Arc::clone(bytes));
        }
        let length = range.end - range.start;
        if length > self.budget_bytes {
            return Err(V41RangeCacheError::BudgetTooSmall {
                item_bytes: length,
                budget_bytes: self.budget_bytes,
            });
        }
        let bytes = self.source.read_range(&V41RangeRequest {
            tensor,
            shard,
            tensor_range,
            range,
        })?;
        let actual = bytes.len() as u64;
        if actual != length {
            return Err(V41RangeCacheError::SizeMismatch {
                expected: length,
                actual,
            });
        }
        while self.used_bytes + length > self.budget_bytes {
            let (_, victim) = self
                .order
                .pop_first()
                .expect("used bytes imply a resident entry");
            let (evicted, _) = self.entries.remove(&victim).expect("ordered entry exists");
            self.used_bytes -= evicted.len() as u64;
        }
        let bytes: Arc<[u8]> = bytes.into();
        self.used_bytes += length;
        self.order.insert(self.tick, key.clone());
        self.entries.insert(key, (Arc::clone(&bytes), self.tick));
        Ok(bytes)
    }
}

fn row_byte_range(table: &V41TensorRange, rows: &Range<u64>) -> Option<Range<u64>> {
    let [row_count, columns] = *table.shape() else {
        return None;
    };
    if rows.start >= rows.end || rows.end > row_count {
        return None;
    }
    // The header already proved row_count * row_bytes fits the tensor interval.
    let row_bytes = columns * table.dtype().bytes_per_element();
    let base = table.file_range().start;
    Some(base + rows.start * row_bytes..base + rows.end * row_bytes)
}

/// A checkpoint range could not be served from the cache or its source.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41RangeCacheError {
    /// The tensor is absent from the index or its shard header.
    #[error("unknown checkpoint tensor {0:?}")]
    UnknownTensor(String),
    /// The tensor's shard has no loaded header.
    #[error("no header loaded for shard {0:?}")]
    MissingShardHeader(String),
    /// The row range is empty, past the table, or the table is not 2-D.
    #[error("rows {rows:?} are not a valid range of two-dimensional table {table:?}")]
    RowRangeOutOfBounds {
        /// Table name.
        table: String,
        /// Requested rows.
        rows: Range<u64>,
    },
    /// One item exceeds the whole cache budget.
    #[error("{item_bytes}-byte item exceeds the {budget_bytes}-byte cache budget")]
    BudgetTooSmall {
        /// Requested bytes.
        item_bytes: u64,
        /// Configured budget.
        budget_bytes: u64,
    },
    /// The source returned or stored a different byte count.
    #[error("expected {expected} bytes, found {actual}")]
    SizeMismatch {
        /// Expected bytes.
        expected: u64,
        /// Actual bytes.
        actual: u64,
    },
    /// Stored bytes do not hash to the receipt's SHA-256.
    #[error("tensor {0:?} does not match its receipt SHA-256")]
    HashMismatch(String),
    /// A receipt disagrees with the request or the shard header.
    #[error("receipt for {tensor:?} disagrees on {field}")]
    ReceiptMismatch {
        /// Tensor name.
        tensor: String,
        /// Disagreeing field.
        field: &'static str,
    },
    /// The source holds no payload for this tensor; a fetcher must acquire it.
    #[error("tensor {0:?} is not available locally")]
    NotLocal(String),
    /// A receipt was not valid JSON of the expected shape.
    #[error("invalid receipt JSON: {0}")]
    ReceiptJson(serde_json::Error),
    /// The payload could not be allocated.
    #[error("range payload allocation failed")]
    Allocation,
    /// A header did not agree with the index.
    #[error(transparent)]
    Header(#[from] V41SafetensorsHeaderError),
    /// Local I/O failed.
    #[error("range source I/O failed: {0}")]
    Io(std::io::Error),
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        Digest, Sha256, V41LocalWeightsSource, V41RangeCache, V41RangeCacheError, V41RangeRequest,
        V41RangeSource,
    };
    use crate::{checkpoint::V41SafetensorsHeader, manifest::V41SafetensorsIndex};

    const SHARD: &str = "model-00001-of-00001.safetensors";
    const REV: &str = "rev";
    static UNIQUE: AtomicUsize = AtomicUsize::new(0);

    // Three 16-byte experts and a 4x3 U8 table, payload bytes = their offset.
    const HEADER: &str = r#"{"e0":{"dtype":"U8","shape":[16],"data_offsets":[0,16]},"e1":{"dtype":"U8","shape":[16],"data_offsets":[16,32]},"e2":{"dtype":"U8","shape":[16],"data_offsets":[32,48]},"table":{"dtype":"U8","shape":[4,3],"data_offsets":[48,60]}}"#;

    fn payload_start() -> u64 {
        8 + HEADER.len() as u64
    }

    fn header() -> V41SafetensorsHeader {
        V41SafetensorsHeader::parse(HEADER.as_bytes(), payload_start() + 60).expect("header")
    }

    fn index() -> V41SafetensorsIndex {
        V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":1}},"weight_map":{{"e0":"{SHARD}","e1":"{SHARD}","e2":"{SHARD}","table":"{SHARD}"}}}}"#
        ))
        .expect("index")
    }

    /// Payload byte at absolute offset `o` is `o - payload_start`; records reads.
    #[derive(Default)]
    struct Synthetic(std::cell::RefCell<Vec<String>>);

    impl V41RangeSource for &Synthetic {
        fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
            self.0.borrow_mut().push(request.tensor.to_owned());
            Ok((request.range.clone())
                .map(|o| u8::try_from(o - payload_start()).expect("small"))
                .collect())
        }
    }

    fn cache<S: V41RangeSource>(source: S, budget: u64) -> V41RangeCache<S> {
        V41RangeCache::new(source, index(), [(SHARD.to_owned(), header())], budget).expect("cache")
    }

    #[test]
    fn hits_misses_and_evicts_least_recently_used() {
        let source = Synthetic::default();
        let mut cache = cache(&source, 32);
        let e0 = cache.get_tensor("e0").expect("miss");
        assert_eq!(&*e0, (0..16).collect::<Vec<u8>>());
        cache.get_tensor("e1").expect("miss");
        cache.get_tensor("e0").expect("hit refreshes e0");
        cache.get_tensor("e2").expect("miss evicts e1, the LRU");
        assert_eq!(cache.used_bytes(), 32);
        cache.get_tensor("e0").expect("still resident");
        cache.get_tensor("e1").expect("refetched");
        assert_eq!(*source.0.borrow(), ["e0", "e1", "e2", "e1"]);
        // The evicted e0 slice the caller held stays valid.
        cache.get_tensor("e2").expect("hit");
        assert_eq!(e0[15], 15);
    }

    #[test]
    fn rejects_unknown_tensors_and_items_larger_than_the_budget() {
        let source = Synthetic::default();
        let mut cache = cache(&source, 15);
        assert!(matches!(
            cache.get_tensor("missing"),
            Err(V41RangeCacheError::UnknownTensor(_))
        ));
        assert!(matches!(
            cache.get_tensor("e0"),
            Err(V41RangeCacheError::BudgetTooSmall {
                item_bytes: 16,
                budget_bytes: 15
            })
        ));
        assert!(source.0.borrow().is_empty());
    }

    #[test]
    fn maps_rows_to_exact_table_byte_ranges() {
        let source = Synthetic::default();
        let mut cache = cache(&source, 64);
        // Table starts at payload byte 48 with 3-byte rows.
        assert_eq!(
            &*cache.get_rows("table", 1..3).expect("rows"),
            [51, 52, 53, 54, 55, 56]
        );
        assert_eq!(
            &*cache.get_rows("table", 3..4).expect("last row"),
            [57, 58, 59]
        );
        for (table, rows) in [("table", 2..2), ("table", 3..5), ("e0", 0..1)] {
            assert!(matches!(
                cache.get_rows(table, rows),
                Err(V41RangeCacheError::RowRangeOutOfBounds { .. })
            ));
        }
    }

    #[test]
    fn short_source_reads_fail_closed() {
        struct Short;
        impl V41RangeSource for Short {
            fn read_range(&self, _: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
                Ok(vec![0; 3])
            }
        }
        assert!(matches!(
            cache(Short, 64).get_tensor("e0"),
            Err(V41RangeCacheError::SizeMismatch {
                expected: 16,
                actual: 3
            })
        ));
    }

    fn weights_dir(e0: &[u8], receipt_sha: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "metallix-range-cache-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let sha = receipt_sha.map_or_else(|| format!("{:x}", Sha256::digest(e0)), str::to_owned);
        let start = payload_start();
        std::fs::write(dir.join("e0.bin"), e0).expect("bin");
        std::fs::write(
            dir.join("e0.receipt.json"),
            format!(
                r#"{{"tensor":"e0","shard":"{SHARD}","range":[{start},{}],"bytes":16,"metadata":{{"dtype":"U8","shape":[16],"data_offsets":[0,16]}},"sha256":"{sha}","revision":"{REV}"}}"#,
                start + 16
            ),
        )
        .expect("receipt");
        dir
    }

    #[test]
    fn local_source_verifies_receipt_hash_and_size() {
        let good: Vec<u8> = (0..16).collect();
        let dir = weights_dir(&good, None);
        let mut ok = cache(V41LocalWeightsSource::new(&dir, REV), 64);
        assert_eq!(&*ok.get_tensor("e0").expect("verified"), good);
        assert!(matches!(
            ok.get_tensor("e1"),
            Err(V41RangeCacheError::NotLocal(_))
        ));
        let mut wrong_rev = cache(V41LocalWeightsSource::new(&dir, "other"), 64);
        assert!(matches!(
            wrong_rev.get_tensor("e0"),
            Err(V41RangeCacheError::ReceiptMismatch {
                field: "revision",
                ..
            })
        ));

        let wrong_hash = weights_dir(&good, Some(&"0".repeat(64)));
        assert!(matches!(
            cache(V41LocalWeightsSource::new(&wrong_hash, REV), 64).get_tensor("e0"),
            Err(V41RangeCacheError::HashMismatch(_))
        ));
        let truncated = weights_dir(&good, None);
        std::fs::write(truncated.join("e0.bin"), &good[..8]).expect("truncate");
        assert!(matches!(
            cache(V41LocalWeightsSource::new(&truncated, REV), 64).get_tensor("e0"),
            Err(V41RangeCacheError::SizeMismatch {
                expected: 16,
                actual: 8
            })
        ));
        for dir in [dir, wrong_hash, truncated] {
            std::fs::remove_dir_all(dir).expect("cleanup own temp dir");
        }
    }

    #[test]
    #[ignore = "requires .agents/receipts/route-trace weights, headers and index"]
    fn real_expert_tensor_matches_its_receipt() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.agents/receipts");
        let trace = root.join("route-trace");
        let index = V41SafetensorsIndex::parse(
            &std::fs::read_to_string(root.join(
                "control/receipts/candidate-control/real-expert/model.safetensors.index.json",
            ))
            .expect("index"),
        )
        .expect("pinned index");
        let shards: serde_json::Value = serde_json::from_slice(
            &std::fs::read(trace.join("headers.json")).expect("headers.json"),
        )
        .expect("headers.json");
        let shard = "model-00003-of-00048.safetensors";
        let file_bytes = shards["shards"][shard]["file_bytes"]
            .as_u64()
            .expect("file_bytes");
        let header = V41SafetensorsHeader::parse_prefixed_header(
            &std::fs::read(trace.join(format!("headers/{shard}.header.bin"))).expect("header"),
            file_bytes,
        )
        .expect("pinned header");
        let tensor = "layers.0.ffn.experts.2.w1.weight";
        let source = V41LocalWeightsSource::new(
            trace.join("weights"),
            "dba1be0a40aa45a94ad051997016db3960a90277",
        );
        let mut cache = V41RangeCache::new(source, index, [(shard.to_owned(), header)], 64 << 20)
            .expect("cache");
        let bytes = cache.get_tensor(tensor).expect("verified real expert");
        assert_eq!(bytes.len(), 2_304 * 2_560);
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(trace.join(format!("weights/{tensor}.receipt.json"))).expect("receipt"),
        )
        .expect("receipt json");
        assert_eq!(
            format!("{:x}", Sha256::digest(&*bytes)),
            receipt["sha256"].as_str().expect("sha256")
        );
    }
}
