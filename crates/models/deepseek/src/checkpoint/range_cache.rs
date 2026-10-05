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
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::SystemTime,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::{V41SafetensorsHeader, V41SafetensorsHeaderError, V41StorageDtype, V41TensorRange};
use crate::{
    manifest::{CheckpointManifestError, V41SafetensorsIndex},
    moe::{Fp4ExpertWeights, MoEError, RoutedExpertSource},
};

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
/// Receipts digest whole tensors, so the first read of a tensor hashes its
/// whole file. A tensor whose file then keeps the verified digest, length and
/// modification time is not hashed again: later reads, including row
/// subranges, read only the requested range. Clones share that record.
/// Concurrent first reads of one tensor hash it once: the others wait for it.
/// Where the platform reports no modification time, the record falls back to
/// digest and length alone.
#[derive(Clone, Debug)]
pub struct V41LocalWeightsSource {
    dir: PathBuf,
    revision: String,
    verified: Arc<Mutex<HashMap<String, FileIdentity>>>,
    hashing: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    #[cfg(test)]
    hashes: Arc<std::sync::atomic::AtomicUsize>,
}

/// The receipt digest, length and modification time a whole-file hash verified.
type FileIdentity = (String, u64, Option<SystemTime>);

impl V41LocalWeightsSource {
    /// Serves receipts for `revision` from `dir`.
    pub fn new(dir: impl Into<PathBuf>, revision: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            revision: revision.into(),
            verified: Arc::default(),
            hashing: Arc::default(),
            #[cfg(test)]
            hashes: Arc::default(),
        }
    }

    /// The lock one tensor's whole-file hash holds, shared by its concurrent readers.
    fn hash_gate(&self, tensor: &str) -> Arc<Mutex<()>> {
        let mut gates = self
            .hashing
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(gates.entry(tensor.to_owned()).or_default())
    }

    fn is_verified(&self, tensor: &str, identity: &FileIdentity) -> bool {
        self.verified
            .lock()
            .is_ok_and(|verified| verified.get(tensor) == Some(identity))
    }

    fn mark_verified(&self, tensor: &str, identity: FileIdentity) {
        if let Ok(mut verified) = self.verified.lock() {
            verified.insert(tensor.to_owned(), identity);
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
        let metadata = file.metadata().map_err(V41RangeCacheError::Io)?;
        let actual = metadata.len();
        if actual != receipt.bytes {
            return Err(V41RangeCacheError::SizeMismatch {
                expected: receipt.bytes,
                actual,
            });
        }
        let identity = (receipt.sha256, actual, metadata.modified().ok());
        let base = expected.file_range().start;
        let (start, end) = (request.range.start - base, request.range.end - base);
        let mut file = file;
        let gate = (!self.is_verified(tensor, &identity)).then(|| self.hash_gate(tensor));
        let _hashing = gate.as_ref().map(|gate| {
            gate.lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        });
        // Recheck under the gate: a concurrent reader may have just verified it.
        if gate.is_some() && !self.is_verified(tensor, &identity) {
            #[cfg(test)]
            self.hashes
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if start == 0 && end == actual {
                let bytes = read_exact_range(&mut file, 0, actual)?;
                if format!("{:x}", Sha256::digest(&bytes)) != identity.0 {
                    return Err(V41RangeCacheError::HashMismatch(tensor.to_owned()));
                }
                self.mark_verified(tensor, identity);
                return Ok(bytes);
            }
            // A subrange: hash the file in bounded chunks rather than holding it.
            let mut hasher = Sha256::new();
            let mut chunk = vec![0; 8 << 20];
            loop {
                let read = file.read(&mut chunk).map_err(V41RangeCacheError::Io)?;
                if read == 0 {
                    break;
                }
                hasher.update(&chunk[..read]);
            }
            if format!("{:x}", hasher.finalize()) != identity.0 {
                return Err(V41RangeCacheError::HashMismatch(tensor.to_owned()));
            }
            self.mark_verified(tensor, identity);
        }
        read_exact_range(&mut file, start, end)
    }
}

fn read_exact_range(
    file: &mut fs::File,
    start: u64,
    end: u64,
) -> Result<Vec<u8>, V41RangeCacheError> {
    let length = usize::try_from(end - start).map_err(|_| V41RangeCacheError::Allocation)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| V41RangeCacheError::Allocation)?;
    bytes.resize(length, 0);
    file.seek(SeekFrom::Start(start))
        .and_then(|_| file.read_exact(&mut bytes))
        .map_err(V41RangeCacheError::Io)?;
    Ok(bytes)
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

    /// Builds a cache from the route-trace metadata layout: `index_path` is
    /// the checkpoint index, `trace_dir/headers.json` lists each shard's
    /// `header_bytes`, `header_sha256` and `file_bytes` for `revision`, and
    /// `trace_dir/headers/<shard>.header.bin` holds its prefixed header. The
    /// index and every listed header must match their recorded digests.
    pub fn load(
        source: S,
        index_path: &Path,
        trace_dir: &Path,
        revision: &str,
        budget_bytes: u64,
    ) -> Result<Self, V41RangeCacheError> {
        #[derive(Deserialize)]
        struct Manifest {
            revision: String,
            index_sha256: String,
            shards: BTreeMap<String, ShardHeader>,
        }
        #[derive(Deserialize)]
        struct ShardHeader {
            header_bytes: u64,
            header_sha256: String,
            file_bytes: u64,
        }
        let mismatch = |what: &str| V41RangeCacheError::HeadersManifest(what.to_owned());
        let manifest: Manifest = serde_json::from_slice(
            &fs::read(trace_dir.join("headers.json")).map_err(V41RangeCacheError::Io)?,
        )
        .map_err(V41RangeCacheError::ReceiptJson)?;
        if manifest.revision != revision {
            return Err(mismatch("revision"));
        }
        let index_json = fs::read(index_path).map_err(V41RangeCacheError::Io)?;
        if format!("{:x}", Sha256::digest(&index_json)) != manifest.index_sha256 {
            return Err(mismatch("index sha256"));
        }
        let index = V41SafetensorsIndex::parse(
            std::str::from_utf8(&index_json).map_err(|_| mismatch("index utf-8"))?,
        )?;
        let mut headers = Vec::with_capacity(manifest.shards.len());
        for (shard, expected) in manifest.shards {
            if shard.starts_with('.') || shard.contains(['/', '\\']) {
                return Err(mismatch(&shard));
            }
            let bytes = fs::read(trace_dir.join(format!("headers/{shard}.header.bin")))
                .map_err(V41RangeCacheError::Io)?;
            if bytes.len() as u64 != expected.header_bytes
                || format!("{:x}", Sha256::digest(&bytes)) != expected.header_sha256
            {
                return Err(mismatch(&shard));
            }
            let header = V41SafetensorsHeader::parse_prefixed_header(&bytes, expected.file_bytes)?;
            headers.push((shard, header));
        }
        Self::new(source, index, headers, budget_bytes)
    }

    /// Returns a named tensor's validated header dtype, shape and range.
    pub fn tensor_range(&self, tensor: &str) -> Result<V41TensorRange, V41RangeCacheError> {
        self.lookup(tensor).map(|(_, range)| range)
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
    /// The checkpoint index was invalid.
    #[error(transparent)]
    Index(#[from] CheckpointManifestError),
    /// `headers.json` disagrees with the index, a header file, or the revision.
    #[error("headers manifest mismatch: {0}")]
    HeadersManifest(String),
}

/// Lends one layer's routed experts from a shared [`V41RangeCache`].
///
/// Each call loads `layers.{layer}.ffn.experts.{index}.{w1,w2,w3}.{weight,scale}`
/// under the lock, releases it, then runs the expert over the `Arc` payloads,
/// so one cache can serve every layer and the lock is never held during
/// expert arithmetic.
#[derive(Debug)]
pub struct V41CachedRoutedExperts<'a, S> {
    cache: &'a Mutex<V41RangeCache<S>>,
    layer: usize,
    hidden_width: usize,
    intermediate_width: usize,
}

impl<'a, S> V41CachedRoutedExperts<'a, S> {
    /// Serves `layer`'s routed experts with the given geometry (5120 x 2304
    /// for V4.1 Flash).
    pub const fn new(
        cache: &'a Mutex<V41RangeCache<S>>,
        layer: usize,
        hidden_width: usize,
        intermediate_width: usize,
    ) -> Self {
        Self {
            cache,
            layer,
            hidden_width,
            intermediate_width,
        }
    }
}

impl<S: V41RangeSource> RoutedExpertSource for V41CachedRoutedExperts<'_, S> {
    fn with_expert(
        &self,
        index: usize,
        run: &mut dyn FnMut(Fp4ExpertWeights<'_>) -> Result<Vec<u16>, MoEError>,
    ) -> Result<Vec<u16>, MoEError> {
        let unavailable = |reason: String| MoEError::ExpertUnavailable { index, reason };
        let payloads = {
            let mut cache = self
                .cache
                .lock()
                .map_err(|_| unavailable("range cache lock poisoned".to_owned()))?;
            let mut payloads = Vec::with_capacity(6);
            for projection in ["w1", "w2", "w3"] {
                for kind in ["weight", "scale"] {
                    let name = format!(
                        "layers.{}.ffn.experts.{index}.{projection}.{kind}",
                        self.layer
                    );
                    payloads.push(
                        cache
                            .get_tensor(&name)
                            .map_err(|error| unavailable(error.to_string()))?,
                    );
                }
            }
            payloads
        };
        let [w1, w1_scale, w2, w2_scale, w3, w3_scale] = &payloads[..] else {
            unreachable!("six projection payloads");
        };
        run(Fp4ExpertWeights::new(
            self.hidden_width,
            self.intermediate_width,
            w1,
            w1_scale,
            w2,
            w2_scale,
            w3,
            w3_scale,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        path::{Path, PathBuf},
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use super::{
        Digest, Sha256, V41CachedRoutedExperts, V41LocalWeightsSource, V41RangeCache,
        V41RangeCacheError, V41RangeRequest, V41RangeSource,
    };
    use crate::{
        checkpoint::V41SafetensorsHeader,
        manifest::V41SafetensorsIndex,
        moe::{Fp8ExpertWeights, MoEConfig, MoEError, MoEReference, RoutedExpertSource},
    };

    /// Zero payloads; records every requested tensor name in order.
    #[derive(Default)]
    struct Zeros(RefCell<Vec<String>>);

    impl V41RangeSource for &Zeros {
        fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
            self.0.borrow_mut().push(request.tensor.to_owned());
            Ok(vec![
                0;
                usize::try_from(request.range.end - request.range.start)
                    .expect("small")
            ])
        }
    }

    #[test]
    fn cached_experts_request_exactly_six_tensors_and_report_absent_experts() {
        // Layer 3 holds only expert 0, with 32 x 32 FP4 projections.
        let mut entries = Vec::new();
        let mut offset = 0;
        for projection in ["w1", "w2", "w3"] {
            for (kind, dtype, columns) in [("weight", "I8", 16), ("scale", "F8_E8M0", 1)] {
                let name = format!("layers.3.ffn.experts.0.{projection}.{kind}");
                let end = offset + 32 * columns;
                entries.push((
                    name.clone(),
                    format!(
                        r#""{name}":{{"dtype":"{dtype}","shape":[32,{columns}],"data_offsets":[{offset},{end}]}}"#
                    ),
                ));
                offset = end;
            }
        }
        let header_json = format!(
            "{{{}}}",
            entries
                .iter()
                .map(|(_, json)| json.as_str())
                .collect::<Vec<_>>()
                .join(",")
        );
        let header = V41SafetensorsHeader::parse(
            header_json.as_bytes(),
            8 + header_json.len() as u64 + offset,
        )
        .expect("synthetic expert header");
        let index = V41SafetensorsIndex::parse(&format!(
            r#"{{"metadata":{{"total_size":1}},"weight_map":{{{}}}}}"#,
            entries
                .iter()
                .map(|(name, _)| format!(r#""{name}":"{SHARD}""#))
                .collect::<Vec<_>>()
                .join(",")
        ))
        .expect("synthetic index");
        let source = Zeros::default();
        let cache = Mutex::new(
            V41RangeCache::new(&source, index, [(SHARD.to_owned(), header)], 1 << 20)
                .expect("cache"),
        );
        let experts = V41CachedRoutedExperts::new(&cache, 3, 32, 32);
        let mut runs = 0;
        let output = experts
            .with_expert(0, &mut |_| {
                runs += 1;
                Ok(vec![7])
            })
            .expect("present expert");
        assert_eq!((output, runs), (vec![7], 1));
        let requested: Vec<_> = entries.into_iter().map(|(name, _)| name).collect();
        assert_eq!(*source.0.borrow(), requested);

        let absent = experts.with_expert(1, &mut |_| {
            runs += 1;
            Ok(Vec::new())
        });
        assert!(matches!(
            absent,
            Err(MoEError::ExpertUnavailable { index: 1, ref reason })
                if reason.contains("layers.3.ffn.experts.1.w1.weight")
        ));
        assert_eq!(runs, 1);
    }

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
    fn local_source_hashes_a_tensor_once_until_its_file_changes() {
        let dir = std::env::temp_dir().join(format!(
            "metallix-range-cache-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        // The 4x3 U8 `table` at payload bytes 48..60; row r holds 3r..3r+3.
        let table: Vec<u8> = (0..12).collect();
        let start = payload_start() + 48;
        let bin = dir.join("table.bin");
        std::fs::write(&bin, &table).expect("bin");
        std::fs::write(
            dir.join("table.receipt.json"),
            format!(
                r#"{{"tensor":"table","shard":"{SHARD}","range":[{start},{}],"bytes":12,"metadata":{{"dtype":"U8","shape":[4,3]}},"sha256":"{:x}","revision":"{REV}"}}"#,
                start + 12,
                Sha256::digest(&table)
            ),
        )
        .expect("receipt");
        let source = V41LocalWeightsSource::new(&dir, REV);
        assert_eq!(
            &*cache(source.clone(), 64)
                .get_rows("table", 1..2)
                .expect("verified"),
            [3, 4, 5]
        );

        // Corrupt a byte outside the next row but keep the verified length and
        // modification time: the next uncached row is read without rehashing.
        let modified = std::fs::metadata(&bin)
            .and_then(|m| m.modified())
            .expect("mtime");
        let mut corrupt = table.clone();
        corrupt[0] = 0xff;
        std::fs::write(&bin, &corrupt).expect("corrupt");
        std::fs::File::options()
            .write(true)
            .open(&bin)
            .and_then(|file| file.set_modified(modified))
            .expect("restore mtime");
        assert_eq!(
            &*cache(source.clone(), 64)
                .get_rows("table", 2..3)
                .expect("not rehashed"),
            [6, 7, 8]
        );

        // A new modification time means a different file: hash it again.
        std::fs::File::options()
            .write(true)
            .open(&bin)
            .and_then(|file| file.set_modified(modified + std::time::Duration::from_secs(5)))
            .expect("touch");
        assert!(matches!(
            cache(source, 64).get_rows("table", 3..4),
            Err(V41RangeCacheError::HashMismatch(_))
        ));
        std::fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn concurrent_first_reads_of_one_tensor_hash_it_once() {
        let good: Vec<u8> = (0..16).collect();
        let dir = weights_dir(&good, None);
        let source = V41LocalWeightsSource::new(&dir, REV);
        let header = header();
        let range = header.tensor("e0").expect("e0").clone();
        let readers: Vec<_> = (0..8)
            .map(|row| {
                let (source, range) = (source.clone(), range.clone());
                std::thread::spawn(move || {
                    let start = range.file_range().start + row * 2;
                    source
                        .read_range(&V41RangeRequest {
                            tensor: "e0",
                            shard: SHARD,
                            tensor_range: &range,
                            range: start..start + 2,
                        })
                        .expect("verified range")
                })
            })
            .collect();
        for (row, reader) in readers.into_iter().enumerate() {
            let row = u8::try_from(row).expect("small");
            assert_eq!(reader.join().expect("reader"), [2 * row, 2 * row + 1]);
        }
        assert_eq!(source.hashes.load(Ordering::Relaxed), 1);
        std::fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    #[ignore = "requires .agents/receipts/route-trace weights, headers and index"]
    fn real_expert_tensor_matches_its_receipt() {
        let (trace, mut cache) = real_cache(64 << 20);
        let tensor = "layers.0.ffn.experts.2.w1.weight";
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

    const PINNED: &str = "dba1be0a40aa45a94ad051997016db3960a90277";

    fn real_cache(budget: u64) -> (PathBuf, V41RangeCache<V41LocalWeightsSource>) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.agents/receipts");
        let trace = root.join("route-trace");
        let cache = V41RangeCache::load(
            V41LocalWeightsSource::new(trace.join("weights"), PINNED),
            &root.join(
                "control/receipts/candidate-control/real-expert/model.safetensors.index.json",
            ),
            &trace,
            PINNED,
            budget,
        )
        .expect("pinned index and headers");
        (trace, cache)
    }

    fn le_u16(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }

    #[test]
    #[ignore = "requires .agents/receipts/route-trace weights, headers, index and capture-parity3"]
    fn real_layer_zero_moe_matches_the_captured_source_output_bit_for_bit() {
        const HIDDEN: usize = 5_120;
        const INTERMEDIATE: usize = 2_304;
        const EXPERTS: usize = 384;
        let (trace, mut cache) = real_cache(1 << 30);
        let mut get = |name: &str| cache.get_tensor(name).expect(name);
        let gate = le_u16(&get("layers.0.ffn.gate.weight"));
        let bias: Vec<f32> = get("layers.0.ffn.gate.bias")
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect();
        let shared: Vec<_> = ["w1", "w2", "w3"]
            .into_iter()
            .flat_map(|projection| {
                ["weight", "scale"]
                    .map(|kind| format!("layers.0.ffn.shared_experts.{projection}.{kind}"))
            })
            .map(|name| get(&name))
            .collect();
        let shared = Fp8ExpertWeights::new(
            HIDDEN,
            INTERMEDIATE,
            &shared[0],
            &shared[1],
            &shared[2],
            &shared[3],
            &shared[4],
            &shared[5],
        )
        .expect("shared expert");
        let table = vec![None; EXPERTS];
        let moe = MoEReference::new_sparse(
            MoEConfig::new(HIDDEN, INTERMEDIATE, 10.0, 6, 1.0, true, 1.5).expect("V4.1 config"),
            &gate,
            &bias,
            &table,
            shared,
        )
        .expect("empty sparse layer-0 MoE");
        let cache = Mutex::new(cache);
        let experts = V41CachedRoutedExperts::new(&cache, 0, HIDDEN, INTERMEDIATE);
        let capture = trace.join("capture-parity3");
        let input = le_u16(
            &std::fs::read(capture.join("layer00.ffn_in.torch.bfloat16.bin")).expect("ffn_in"),
        );
        let expected = le_u16(
            &std::fs::read(capture.join("layer00.ffn_out.torch.bfloat16.bin")).expect("ffn_out"),
        );
        assert_eq!(input.len(), 3 * HIDDEN);
        assert_eq!(expected.len(), input.len());
        // The recorder's layer-0 gate selections for the same three tokens.
        let recorded: serde_json::Value = serde_json::from_slice(
            &std::fs::read(trace.join("parity-greedy3.json")).expect("parity-greedy3.json"),
        )
        .expect("recorded routes");
        let recorded = recorded["runs"][0]["routes"]
            .as_array()
            .expect("routes")
            .iter()
            .find(|route| route["layer"] == 0)
            .expect("layer-0 routes")["ids"]
            .clone();
        for (token, (input, expected)) in input
            .chunks_exact(HIDDEN)
            .zip(expected.chunks_exact(HIDDEN))
            .enumerate()
        {
            let output = moe
                .forward_token_with(input, &experts)
                .unwrap_or_else(|error| panic!("token {token}: {error}"));
            let mut ids: Vec<_> = recorded[token]
                .as_array()
                .expect("token routes")
                .iter()
                .map(|id| usize::try_from(id.as_u64().expect("id")).expect("small id"))
                .collect();
            ids.sort_unstable();
            let routed: Vec<_> = output.routes().iter().map(|r| r.expert_index()).collect();
            assert_eq!(routed, ids, "token {token} routes");
            assert_eq!(output.output_bf16(), expected, "token {token}");
        }
    }
}
