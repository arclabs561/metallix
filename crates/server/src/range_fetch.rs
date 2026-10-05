//! Network fallback for `DeepSeek` V4.1 checkpoint range reads.
//!
//! [`FetchingSource`] serves a range from the local route-trace weights store
//! and, only when that store reports a tensor as absent, fetches the exact
//! byte range from the pinned Hugging Face revision. A whole-tensor fetch is
//! written as `<tensor>.bin` plus the receipt the acquisition scripts write,
//! then re-read through the local source so it passes the same receipt and
//! SHA-256 checks as any other local tensor.
//!
//! A sub-tensor read of whole rows (Engram and embedding table rows) is
//! stored per row under `.rows/<tensor>/`, each row with its own receipt, so a
//! repeat request reads rows locally whatever runs they were grouped into.
//! The source publishes no digest for part of a tensor, so a row is trusted
//! on first use: that fetch is checked by exact length and `Content-Range`,
//! and its receipt records the digest every later read and any refetch must
//! reproduce. A read that is not whole rows is fetched and not stored.
//!
//! Stored bytes stay inside an acquisition envelope (total stored bytes and a
//! free-disk floor). A fetch that would cross it first evicts
//! least-recently-used routed experts, one whole expert at a time; experts in
//! the trace directory's `pinned-experts.json`, all non-expert tensors and all
//! stored rows are never evicted, and the fetch is refused only when evicting everything else
//! would still not make room. Replaying 5 recorded runs (58,080 expert
//! accesses) through LRU, a 64 GiB cap missed 23% of accesses and re-downloaded
//! 79 GiB; 128 GiB, the default, missed 16% and re-downloaded 7.6 GiB, against a
//! 15.3% compulsory floor.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    io::{self, Write as _},
    ops::Range,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use deepseek::checkpoint::{
    V41StorageDtype,
    range_cache::{V41LocalWeightsSource, V41RangeCacheError, V41RangeRequest, V41RangeSource},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bounds on bytes a [`FetchingSource`] may store.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Envelope {
    /// Largest total size of `*.bin` payloads in the weights directory.
    pub max_acquired_bytes: u64,
    /// Free bytes that must remain on the weights volume after a write.
    pub min_free_bytes: u64,
}

impl Default for Envelope {
    /// The route-trace acquisition envelope: 128 GiB stored, 150 GiB free.
    fn default() -> Self {
        Self {
            max_acquired_bytes: 128 << 30,
            min_free_bytes: 150 << 30,
        }
    }
}

/// The outside world a fetch needs: HTTP range reads and free disk space.
pub trait RangeHost {
    /// Fetches bytes `range` of `url`, returning the body and the final
    /// response's `Content-Range` value (empty when absent).
    fn get_range(&self, url: &str, range: Range<u64>) -> io::Result<(Vec<u8>, String)>;

    /// Free bytes on the volume holding `path`.
    fn free_bytes(&self, path: &Path) -> Option<u64>;
}

/// [`RangeHost`] over the system `curl`, with the deadlines `record.py` uses.
///
/// The body is buffered in memory, so a fetch costs its full size in RAM.
#[derive(Clone, Copy, Debug, Default)]
pub struct CurlHost;

impl RangeHost for CurlHost {
    fn get_range(&self, url: &str, range: Range<u64>) -> io::Result<(Vec<u8>, String)> {
        let last = range
            .end
            .checked_sub(1)
            .filter(|last| *last >= range.start)
            .ok_or_else(|| io::Error::other("empty range"))?;
        let output = Command::new("curl")
            .args([
                "-sS",
                "-L",
                "--fail",
                "--connect-timeout",
                "10",
                "--max-time",
                "180",
            ])
            // `--retry` alone skips connection resets (exit 35), which the Hub
            // CDN produces under many small range reads.
            .args(["--retry", "3", "--retry-delay", "2", "--retry-all-errors"])
            .args(["-H", "Accept-Encoding: identity", "-D", "/dev/stderr"])
            .arg("-r")
            .arg(format!("{}-{last}", range.start))
            .arg(url)
            .output()?;
        let headers = String::from_utf8_lossy(&output.stderr);
        if !output.status.success() {
            return Err(io::Error::other(format!(
                "curl exited with {}: {}",
                output.status,
                headers.lines().last().unwrap_or_default()
            )));
        }
        // `-L` prints one header block per hop; the last Content-Range wins.
        let content_range = headers
            .lines()
            .filter_map(|line| line.split_once(':'))
            .filter(|(name, _)| name.trim().eq_ignore_ascii_case("content-range"))
            .map(|(_, value)| value.trim().to_owned())
            .next_back()
            .unwrap_or_default();
        Ok((output.stdout, content_range))
    }

    fn free_bytes(&self, path: &Path) -> Option<u64> {
        crate::free_space_bytes(path)
    }
}

#[derive(Deserialize)]
struct HeadersManifest {
    revision: String,
    shards: BTreeMap<String, ShardBytes>,
}

#[derive(Deserialize)]
struct ShardBytes {
    header_bytes: u64,
    file_bytes: u64,
}

#[derive(Deserialize, Serialize)]
struct Receipt {
    tensor: String,
    shard: String,
    range: [u64; 2],
    bytes: u64,
    metadata: ReceiptMetadata,
    sha256: String,
    revision: String,
}

/// The receipt of one stored table row, `.rows/<tensor>/<row>.receipt.json`.
#[derive(Deserialize, Serialize)]
struct RowReceipt {
    tensor: String,
    shard: String,
    row: u64,
    /// Absolute shard byte range of the row.
    range: [u64; 2],
    sha256: String,
    revision: String,
}

#[derive(Deserialize, Serialize)]
struct ReceiptMetadata {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

/// `trace_dir/pinned-experts.json`, written by `scripts/v41_expert_pins.py`.
#[derive(Deserialize)]
struct PinFile {
    experts: Vec<Expert>,
}

/// A routed expert: (layer, expert index).
type Expert = (u32, u32);

/// The tensors of one routed expert, after `layers.L.ffn.experts.E.`.
const EXPERT_PARTS: [&str; 6] = [
    "w1.weight",
    "w1.scale",
    "w2.weight",
    "w2.scale",
    "w3.weight",
    "w3.scale",
];

/// Store-owned recency log, one `layer expert` line per use, in the weights
/// directory. A leading dot keeps it out of the tensor namespace.
const JOURNAL: &str = ".expert-recency.log";

/// Directory of stored table rows in the weights directory, one subdirectory
/// per tensor; hidden like the journal.
const ROWS: &str = ".rows";

/// Journal lines beyond the live expert count before it is rewritten.
const JOURNAL_SLACK: usize = 1 << 16;

/// The routed expert `tensor` belongs to, if it is one of an expert's parts.
fn routed_expert(tensor: &str) -> Option<Expert> {
    let (layer, rest) = tensor.strip_prefix("layers.")?.split_once('.')?;
    let (expert, part) = rest.strip_prefix("ffn.experts.")?.split_once('.')?;
    if !EXPERT_PARTS.contains(&part) {
        return None;
    }
    Some((layer.parse().ok()?, expert.parse().ok()?))
}

/// What the weights directory holds, and when each routed expert was last used.
///
/// Recency is a store-owned journal rather than file atime, which macOS does
/// not maintain reliably. It is an append-only log: one short `write` per use,
/// no timer, and a torn final line after a crash is skipped on replay, so a
/// crash loses at most the last use. It is compacted to one line per stored
/// expert, oldest first, on startup and whenever it grows past
/// [`JOURNAL_SLACK`] extra lines. The journal only orders eviction; losing it
/// costs re-downloads, never correctness.
#[derive(Debug)]
struct Store {
    /// Total `*.bin` bytes in the weights directory, stored rows included.
    acquired: u64,
    /// Stored `*.bin` bytes and last-use tick of each routed expert with any
    /// part on disk.
    experts: HashMap<Expert, (u64, u64)>,
    /// Routed experts by last-use tick, oldest first.
    by_age: BTreeMap<u64, Expert>,
    tick: u64,
    journal_path: PathBuf,
    journal: Option<fs::File>,
    journal_lines: usize,
    /// The expert of the last journal line; repeat uses of one expert's parts
    /// write one line.
    last: Option<Expert>,
}

impl Store {
    fn open(weights_dir: &Path) -> io::Result<Self> {
        let mut acquired = 0;
        let mut experts = HashMap::<Expert, (u64, u64)>::new();
        for entry in fs::read_dir(weights_dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(tensor) = name
                .to_string_lossy()
                .strip_suffix(".bin")
                .map(str::to_owned)
            else {
                continue;
            };
            let len = entry.metadata()?.len();
            acquired += len;
            if let Some(expert) = routed_expert(&tensor) {
                experts.entry(expert).or_default().0 += len;
            }
        }
        acquired += stored_row_bytes(&weights_dir.join(ROWS))?;
        let mut store = Self {
            acquired,
            experts,
            by_age: BTreeMap::new(),
            tick: 0,
            journal_path: weights_dir.join(JOURNAL),
            journal: None,
            journal_lines: 0,
            last: None,
        };
        // Experts the journal never saw are the oldest, in name order.
        let mut unseen: Vec<Expert> = store.experts.keys().copied().collect();
        unseen.sort_unstable();
        for expert in unseen {
            store.mark_used(expert);
        }
        let log = match fs::read_to_string(&store.journal_path) {
            Ok(log) => log,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(store),
            Err(error) => return Err(error),
        };
        for line in log.lines() {
            let mut fields = line.split(' ').map(str::parse::<u32>);
            if let (Some(Ok(layer)), Some(Ok(index)), None) =
                (fields.next(), fields.next(), fields.next())
            {
                store.mark_used((layer, index));
            }
        }
        store.compact_journal()?;
        Ok(store)
    }

    /// Moves a stored `expert` to the most-recent end; false if not stored.
    fn mark_used(&mut self, expert: Expert) -> bool {
        let Some((_, used)) = self.experts.get_mut(&expert) else {
            return false;
        };
        self.by_age.remove(used);
        self.tick += 1;
        *used = self.tick;
        self.by_age.insert(self.tick, expert);
        true
    }

    /// Records a use of `expert` in memory and in the journal.
    fn touch(&mut self, expert: Expert) {
        if self.last == Some(expert) || !self.mark_used(expert) {
            return;
        }
        self.last = Some(expert);
        // A failed journal write only loses recency, so it does not fail the read.
        if self.journal_lines > self.experts.len().saturating_mul(4) + JOURNAL_SLACK {
            let _ = self.compact_journal();
        } else {
            let _ = self.append(expert);
        }
    }

    fn append(&mut self, (layer, index): Expert) -> io::Result<()> {
        if self.journal.is_none() {
            self.journal = Some(
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.journal_path)?,
            );
        }
        if let Some(journal) = &mut self.journal {
            journal.write_all(format!("{layer} {index}\n").as_bytes())?;
        }
        self.journal_lines += 1;
        Ok(())
    }

    /// Rewrites the journal as one line per stored expert, oldest first.
    fn compact_journal(&mut self) -> io::Result<()> {
        let mut log = Vec::new();
        for (layer, index) in self.by_age.values() {
            writeln!(log, "{layer} {index}")?;
        }
        // The open handle would keep appending to the replaced file.
        self.journal = None;
        write_atomically(&self.journal_path, &log).map_err(|error| match error {
            V41RangeCacheError::Io(error) => error,
            other => io::Error::other(other.to_string()),
        })?;
        self.journal_lines = self.by_age.len();
        Ok(())
    }

    /// Least-recently-used experts whose eviction lets `length` more bytes fit
    /// both limits, never `pinned` ones or `keep` (the expert being fetched).
    fn victims(
        &self,
        envelope: Envelope,
        free: u64,
        length: u64,
        pinned: &HashSet<Expert>,
        keep: Option<Expert>,
    ) -> Result<Vec<Expert>, String> {
        let over_cap = (self.acquired + length).saturating_sub(envelope.max_acquired_bytes);
        let over_floor = (envelope.min_free_bytes + length).saturating_sub(free);
        let shortfall = over_cap.max(over_floor);
        let mut need = shortfall;
        let mut chosen = Vec::new();
        for expert in self.by_age.values() {
            if need == 0 {
                break;
            }
            if Some(*expert) == keep || pinned.contains(expert) {
                continue;
            }
            need = need.saturating_sub(self.experts[expert].0);
            chosen.push(*expert);
        }
        if need > 0 {
            return Err(format!(
                "{shortfall} bytes over the envelope ({} stored of {} allowed, {free} free above a {}-byte floor), and only {} of them belong to unpinned routed experts",
                self.acquired,
                envelope.max_acquired_bytes,
                envelope.min_free_bytes,
                shortfall - need
            ));
        }
        Ok(chosen)
    }

    /// Deletes every stored part of `expert`.
    ///
    /// Only `*.bin` payloads are removed; receipts stay, so a refetch must
    /// reproduce the recorded digest, and a part whose payload is gone reads
    /// as absent ([`V41RangeCacheError::NotLocal`]). Each unlink is atomic, so
    /// a crash part-way leaves each part either whole and verified or absent:
    /// the next read of an absent part refetches and re-verifies it, the next
    /// startup counts exactly the payloads that remain, and a later eviction
    /// removes the rest.
    fn evict(&mut self, weights_dir: &Path, expert: Expert) -> io::Result<()> {
        let (layer, index) = expert;
        for part in EXPERT_PARTS {
            let path = weights_dir.join(format!("layers.{layer}.ffn.experts.{index}.{part}.bin"));
            let len = match fs::metadata(&path) {
                Ok(metadata) => metadata.len(),
                Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            fs::remove_file(&path)?;
            self.acquired = self.acquired.saturating_sub(len);
            if let Some((bytes, _)) = self.experts.get_mut(&expert) {
                *bytes = bytes.saturating_sub(len);
            }
        }
        if let Some((_, used)) = self.experts.remove(&expert) {
            self.by_age.remove(&used);
        }
        if self.last == Some(expert) {
            self.last = None;
        }
        Ok(())
    }

    /// Counts a `length`-byte payload that replaced `replaced` bytes.
    fn stored(&mut self, expert: Option<Expert>, length: u64, replaced: u64) {
        self.acquired = self.acquired.saturating_sub(replaced) + length;
        if let Some(expert) = expert {
            let fresh = !self.experts.contains_key(&expert);
            let (bytes, _) = self.experts.entry(expert).or_default();
            *bytes = bytes.saturating_sub(replaced) + length;
            if fresh {
                self.mark_used(expert);
            }
        }
    }
}

/// Local weights first; exact-range fetch from the pinned revision on a miss.
#[derive(Debug)]
pub struct FetchingSource<H> {
    local: V41LocalWeightsSource,
    weights_dir: PathBuf,
    repo: String,
    revision: String,
    shards: BTreeMap<String, (u64, u64)>,
    host: H,
    envelope: Envelope,
    pinned: HashSet<Expert>,
    /// Held across a whole fetch so concurrent misses cannot jointly overrun.
    acquiring: Mutex<()>,
    /// Held briefly, so local reads are not stalled behind a network fetch.
    store: Mutex<Store>,
}

impl<H: RangeHost> FetchingSource<H> {
    /// Serves `weights_dir` for `repo` at `revision`. Shard lengths come from
    /// `trace_dir/headers.json`, whose revision must match, and pinned experts
    /// from `trace_dir/pinned-experts.json` when present. The current `*.bin`
    /// total in `weights_dir` counts against `envelope`.
    pub fn new(
        weights_dir: impl Into<PathBuf>,
        trace_dir: &Path,
        repo: impl Into<String>,
        revision: impl Into<String>,
        host: H,
        envelope: Envelope,
    ) -> io::Result<Self> {
        let weights_dir = weights_dir.into();
        let revision = revision.into();
        let manifest: HeadersManifest =
            serde_json::from_slice(&fs::read(trace_dir.join("headers.json"))?)
                .map_err(io::Error::other)?;
        if manifest.revision != revision {
            return Err(io::Error::other(format!(
                "headers.json is for revision {}, not {revision}",
                manifest.revision
            )));
        }
        let pinned = match fs::read(trace_dir.join("pinned-experts.json")) {
            Ok(json) => serde_json::from_slice::<PinFile>(&json)
                .map_err(io::Error::other)?
                .experts
                .into_iter()
                .collect(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => HashSet::new(),
            Err(error) => return Err(error),
        };
        let store = Store::open(&weights_dir)?;
        Ok(Self {
            local: V41LocalWeightsSource::new(&weights_dir, revision.clone()),
            weights_dir,
            repo: repo.into(),
            revision,
            shards: manifest
                .shards
                .into_iter()
                .map(|(shard, bytes)| (shard, (bytes.header_bytes, bytes.file_bytes)))
                .collect(),
            host,
            envelope,
            pinned,
            acquiring: Mutex::new(()),
            store: Mutex::new(store),
        })
    }

    /// Total `*.bin` bytes counted against the envelope.
    pub fn acquired_bytes(&self) -> u64 {
        self.lock_store().acquired
    }

    fn lock_store(&self) -> std::sync::MutexGuard<'_, Store> {
        self.store
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn free_bytes(&self) -> Result<u64, V41RangeCacheError> {
        self.host
            .free_bytes(&self.weights_dir)
            .ok_or_else(|| fail("cannot determine free space for the weights volume".to_owned()))
    }

    fn fetch(&self, shard: &str, range: Range<u64>) -> Result<Vec<u8>, V41RangeCacheError> {
        let &(_, file_bytes) = self
            .shards
            .get(shard)
            .ok_or_else(|| fail(format!("headers.json has no shard {shard:?}")))?;
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{shard}",
            self.repo, self.revision
        );
        let (bytes, content_range) = self
            .host
            .get_range(&url, range.clone())
            .map_err(V41RangeCacheError::Io)?;
        let expected = format!("bytes {}-{}/{file_bytes}", range.start, range.end - 1);
        if content_range != expected {
            return Err(fail(format!(
                "{shard} answered Content-Range {content_range:?}, expected {expected:?}"
            )));
        }
        let length = range.end - range.start;
        if bytes.len() as u64 != length {
            return Err(V41RangeCacheError::SizeMismatch {
                expected: length,
                actual: bytes.len() as u64,
            });
        }
        Ok(bytes)
    }

    fn acquire_tensor(&self, request: &V41RangeRequest<'_>) -> Result<(), V41RangeCacheError> {
        let tensor = request.tensor;
        let file_range = request.tensor_range.file_range();
        let length = file_range.end - file_range.start;
        let owner = routed_expert(tensor);
        let refuse = |why: String| fail(format!("fetching {tensor} ({length} bytes): {why}"));
        let _acquiring = self
            .acquiring
            .lock()
            .map_err(|_| fail("acquisition lock poisoned".to_owned()))?;
        // Refuse before downloading anything that could never be stored.
        let free = self.free_bytes()?;
        self.lock_store()
            .victims(self.envelope, free, length, &self.pinned, owner)
            .map_err(refuse)?;

        let bytes = self.fetch(request.shard, file_range.clone())?;
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let receipt_path = self.weights_dir.join(format!("{tensor}.receipt.json"));
        // An evicted tensor keeps its receipt; the refetch must reproduce it.
        let prior = match fs::read(&receipt_path) {
            Ok(json) => Some(
                serde_json::from_slice::<Receipt>(&json)
                    .map_err(V41RangeCacheError::ReceiptJson)?,
            ),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(V41RangeCacheError::Io(error)),
        };
        if prior.as_ref().is_some_and(|prior| prior.sha256 != sha256) {
            return Err(V41RangeCacheError::HashMismatch(tensor.to_owned()));
        }
        // Evict only once the bytes are in hand, so a failed download costs nothing.
        let free = self.free_bytes()?;
        let mut store = self.lock_store();
        let victims = store
            .victims(self.envelope, free, length, &self.pinned, owner)
            .map_err(refuse)?;
        for victim in victims {
            store
                .evict(&self.weights_dir, victim)
                .map_err(V41RangeCacheError::Io)?;
        }
        let bin_path = self.weights_dir.join(format!("{tensor}.bin"));
        // A payload left without a receipt is replaced, not added to.
        let replaced = fs::metadata(&bin_path).map_or(0, |metadata| metadata.len());
        write_atomically(&bin_path, &bytes)?;
        store.stored(owner, length, replaced);
        drop(store);
        if prior.is_none() {
            let &(payload_start, _) = &self.shards[request.shard];
            let receipt = Receipt {
                tensor: tensor.to_owned(),
                shard: request.shard.to_owned(),
                range: [file_range.start, file_range.end],
                bytes: length,
                metadata: ReceiptMetadata {
                    dtype: safetensors_dtype(request.tensor_range.dtype())?.to_owned(),
                    shape: request.tensor_range.shape().to_vec(),
                    data_offsets: [
                        file_range.start - payload_start,
                        file_range.end - payload_start,
                    ],
                },
                sha256,
                revision: self.revision.clone(),
            };
            let json = serde_json::to_vec(&receipt).map_err(V41RangeCacheError::ReceiptJson)?;
            write_atomically(&receipt_path, &json)?;
        }
        Ok(())
    }
}

impl<H: RangeHost> FetchingSource<H> {
    /// Serves a sub-tensor read from stored rows, fetching and storing the
    /// missing ones. A read that is not whole rows is fetched, not stored.
    fn read_rows(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        let Some((row_bytes, rows)) = whole_rows(request) else {
            return self.fetch(request.shard, request.range.clone());
        };
        let mut found = rows
            .clone()
            .map(|row| self.stored_row(request, row, row_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let mut index = 0;
        while index < found.len() {
            if found[index].is_some() {
                index += 1;
                continue;
            }
            let mut end = index + 1;
            while end < found.len() && found[end].is_none() {
                end += 1;
            }
            let missing = rows.start + index as u64..rows.start + end as u64;
            let fetched = self.acquire_rows(request, row_bytes, missing)?;
            for (slot, row) in found[index..end].iter_mut().zip(fetched) {
                *slot = Some(row);
            }
            index = end;
        }
        Ok(found.into_iter().flatten().flatten().collect())
    }

    fn row_paths(&self, tensor: &str, row: u64) -> (PathBuf, PathBuf) {
        let dir = self.weights_dir.join(ROWS).join(tensor);
        (
            dir.join(format!("{row}.bin")),
            dir.join(format!("{row}.receipt.json")),
        )
    }

    /// One stored row, checked against its receipt; `None` when it has no
    /// receipt or no payload (a payload without a receipt is replaced).
    fn stored_row(
        &self,
        request: &V41RangeRequest<'_>,
        row: u64,
        row_bytes: u64,
    ) -> Result<Option<Vec<u8>>, V41RangeCacheError> {
        let (bin_path, receipt_path) = self.row_paths(request.tensor, row);
        let Some(receipt) = row_receipt(&receipt_path)? else {
            return Ok(None);
        };
        let bytes = match fs::read(&bin_path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(V41RangeCacheError::Io(error)),
        };
        let start = request.tensor_range.file_range().start + row * row_bytes;
        if receipt.tensor != request.tensor
            || receipt.shard != request.shard
            || receipt.row != row
            || receipt.range != [start, start + row_bytes]
            || receipt.revision != self.revision
        {
            return Err(fail(format!(
                "{} does not describe {} row {row} at revision {}",
                receipt_path.display(),
                request.tensor,
                self.revision
            )));
        }
        if bytes.len() as u64 != row_bytes
            || format!("{:x}", Sha256::digest(&bytes)) != receipt.sha256
        {
            return Err(V41RangeCacheError::HashMismatch(format!(
                "{} row {row}",
                request.tensor
            )));
        }
        Ok(Some(bytes))
    }

    /// Fetches `rows` of `request.tensor` in one range read and stores each
    /// row with a receipt, inside the envelope. A row whose receipt survives
    /// its payload must refetch to the recorded digest.
    fn acquire_rows(
        &self,
        request: &V41RangeRequest<'_>,
        row_bytes: u64,
        rows: Range<u64>,
    ) -> Result<Vec<Vec<u8>>, V41RangeCacheError> {
        let tensor = request.tensor;
        let length = (rows.end - rows.start) * row_bytes;
        let refuse = |why: String| {
            fail(format!(
                "fetching {tensor} rows {rows:?} ({length} bytes): {why}"
            ))
        };
        let _acquiring = self
            .acquiring
            .lock()
            .map_err(|_| fail("acquisition lock poisoned".to_owned()))?;
        let free = self.free_bytes()?;
        self.lock_store()
            .victims(self.envelope, free, length, &self.pinned, None)
            .map_err(refuse)?;

        let start = request.tensor_range.file_range().start + rows.start * row_bytes;
        let bytes = self.fetch(request.shard, start..start + length)?;
        let fetched: Vec<Vec<u8>> = bytes
            .chunks_exact(usize::try_from(row_bytes).map_err(|_| fail("row too large".to_owned()))?)
            .map(<[u8]>::to_vec)
            .collect();
        let mut receipts = Vec::with_capacity(fetched.len());
        for (row, payload) in rows.clone().zip(&fetched) {
            let (_, receipt_path) = self.row_paths(tensor, row);
            let sha256 = format!("{:x}", Sha256::digest(payload));
            let prior = row_receipt(&receipt_path)?;
            if prior.as_ref().is_some_and(|prior| prior.sha256 != sha256) {
                return Err(V41RangeCacheError::HashMismatch(format!(
                    "{tensor} row {row}"
                )));
            }
            receipts.push((prior.is_none(), sha256));
        }

        let free = self.free_bytes()?;
        let mut store = self.lock_store();
        let victims = store
            .victims(self.envelope, free, length, &self.pinned, None)
            .map_err(refuse)?;
        for victim in victims {
            store
                .evict(&self.weights_dir, victim)
                .map_err(V41RangeCacheError::Io)?;
        }
        fs::create_dir_all(self.weights_dir.join(ROWS).join(tensor))
            .map_err(V41RangeCacheError::Io)?;
        for ((row, payload), (fresh, sha256)) in rows.zip(&fetched).zip(receipts) {
            let (bin_path, receipt_path) = self.row_paths(tensor, row);
            let replaced = fs::metadata(&bin_path).map_or(0, |metadata| metadata.len());
            write_atomically(&bin_path, payload)?;
            store.stored(None, row_bytes, replaced);
            if fresh {
                let row_start = request.tensor_range.file_range().start + row * row_bytes;
                let receipt = RowReceipt {
                    tensor: tensor.to_owned(),
                    shard: request.shard.to_owned(),
                    row,
                    range: [row_start, row_start + row_bytes],
                    sha256,
                    revision: self.revision.clone(),
                };
                let json = serde_json::to_vec(&receipt).map_err(V41RangeCacheError::ReceiptJson)?;
                write_atomically(&receipt_path, &json)?;
            }
        }
        Ok(fetched)
    }
}

fn row_receipt(path: &Path) -> Result<Option<RowReceipt>, V41RangeCacheError> {
    match fs::read(path) {
        Ok(json) => Ok(Some(
            serde_json::from_slice(&json).map_err(V41RangeCacheError::ReceiptJson)?,
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(V41RangeCacheError::Io(error)),
    }
}

/// `(row bytes, rows)` when `request` reads whole rows of its tensor's first
/// dimension.
fn whole_rows(request: &V41RangeRequest<'_>) -> Option<(u64, Range<u64>)> {
    let tensor = request.tensor_range.file_range();
    let rows = *request.tensor_range.shape().first()?;
    let tensor_bytes = tensor.end - tensor.start;
    if rows == 0 || !tensor_bytes.is_multiple_of(rows) {
        return None;
    }
    let row_bytes = tensor_bytes / rows;
    let offset = request.range.start.checked_sub(tensor.start)?;
    let length = request.range.end.checked_sub(request.range.start)?;
    if row_bytes == 0
        || length == 0
        || request.range.end > tensor.end
        || !offset.is_multiple_of(row_bytes)
        || !length.is_multiple_of(row_bytes)
    {
        return None;
    }
    Some((row_bytes, offset / row_bytes..(offset + length) / row_bytes))
}

/// Total `*.bin` bytes of stored rows under `rows_dir`, zero if it is absent.
fn stored_row_bytes(rows_dir: &Path) -> io::Result<u64> {
    let tensors = match fs::read_dir(rows_dir) {
        Ok(tensors) => tensors,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(error),
    };
    let mut total = 0;
    for tensor in tensors {
        for row in fs::read_dir(tensor?.path())? {
            let row = row?;
            if row.file_name().to_string_lossy().ends_with(".bin") {
                total += row.metadata()?.len();
            }
        }
    }
    Ok(total)
}

impl<H: RangeHost> V41RangeSource for FetchingSource<H> {
    fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        let result = match self.local.read_range(request) {
            Err(V41RangeCacheError::NotLocal(_)) => {
                if request.range != request.tensor_range.file_range() {
                    return self.read_rows(request);
                }
                self.acquire_tensor(request)?;
                self.local.read_range(request)
            }
            result => result,
        };
        if result.is_ok()
            && let Some(expert) = routed_expert(request.tensor)
        {
            self.lock_store().touch(expert);
        }
        result
    }

    /// A RAM-hot expert stays recent on disk, so it is not evicted as cold.
    fn note_memory_hit(&self, request: &V41RangeRequest<'_>) {
        if let Some(expert) = routed_expert(request.tensor) {
            self.lock_store().touch(expert);
        }
    }
}

fn fail(message: String) -> V41RangeCacheError {
    V41RangeCacheError::Io(io::Error::other(message))
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), V41RangeCacheError> {
    static UNIQUE: AtomicU64 = AtomicU64::new(0);
    let mut part = path.as_os_str().to_owned();
    part.push(format!(
        ".{}.{}.part",
        std::process::id(),
        UNIQUE.fetch_add(1, Ordering::Relaxed)
    ));
    let part = PathBuf::from(part);
    fs::write(&part, bytes)
        .and_then(|()| fs::rename(&part, path))
        .map_err(|error| {
            let _ = fs::remove_file(&part);
            V41RangeCacheError::Io(error)
        })
}

/// Canonical safetensors spelling, as the pinned headers and receipts use.
fn safetensors_dtype(dtype: V41StorageDtype) -> Result<&'static str, V41RangeCacheError> {
    Ok(match dtype {
        V41StorageDtype::Bool => "BOOL",
        V41StorageDtype::U8 => "U8",
        V41StorageDtype::I8 => "I8",
        V41StorageDtype::U16 => "U16",
        V41StorageDtype::I16 => "I16",
        V41StorageDtype::F16 => "F16",
        V41StorageDtype::Bf16 => "BF16",
        V41StorageDtype::U32 => "U32",
        V41StorageDtype::I32 => "I32",
        V41StorageDtype::F32 => "F32",
        V41StorageDtype::U64 => "U64",
        V41StorageDtype::I64 => "I64",
        V41StorageDtype::F64 => "F64",
        V41StorageDtype::F8E4M3Fn => "F8_E4M3",
        V41StorageDtype::F8E4M3Fnuz => "F8_E4M3FNUZ",
        V41StorageDtype::F8E5M2 => "F8_E5M2",
        V41StorageDtype::F8E5M2Fnuz => "F8_E5M2FNUZ",
        V41StorageDtype::F8E8M0Fnu => "F8_E8M0",
        other => return Err(fail(format!("no safetensors spelling for {other:?}"))),
    })
}

#[cfg(test)]
mod tests {
    use std::{
        cell::RefCell,
        collections::BTreeSet,
        fs,
        ops::Range,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use deepseek::checkpoint::range_cache::{V41RangeCache, V41RangeCacheError};
    use sha2::{Digest, Sha256};

    use super::{CurlHost, EXPERT_PARTS, Envelope, FetchingSource, RangeHost};

    const SHARD: &str = "model-00001-of-00001.safetensors";
    const REV: &str = "rev";
    /// Routed experts `layers.0.ffn.experts.0..EXPERTS`, each six 2-byte parts.
    const EXPERTS: u32 = 4;
    const EXPERT_BYTES: u64 = 12;
    const PAYLOAD_BYTES: u64 = 24 + EXPERTS as u64 * EXPERT_BYTES;

    fn part(expert: u32, part: &str) -> String {
        format!("layers.0.ffn.experts.{expert}.{part}")
    }

    /// `a` is a 4x4 U8 table at payload bytes 0..16; `b` is 8 E8M0 scales at
    /// 16..24; the expert parts follow as 2-byte U8 tensors.
    fn header() -> String {
        let mut tensors = vec![
            r#""a":{"dtype":"U8","shape":[4,4],"data_offsets":[0,16]}"#.to_owned(),
            r#""b":{"dtype":"F8_E8M0","shape":[8],"data_offsets":[16,24]}"#.to_owned(),
        ];
        let mut offset = 24;
        for expert in 0..EXPERTS {
            for name in EXPERT_PARTS {
                tensors.push(format!(
                    r#""{}":{{"dtype":"U8","shape":[2],"data_offsets":[{offset},{}]}}"#,
                    part(expert, name),
                    offset + 2
                ));
                offset += 2;
            }
        }
        format!("{{{}}}", tensors.join(","))
    }

    fn payload_start() -> u64 {
        8 + header().len() as u64
    }

    /// Serves one synthetic shard whose payload byte `i` is `i`.
    struct FakeHost {
        free: u64,
        /// When set, free space is this volume size minus the `*.bin` bytes
        /// in the queried directory, so evictions free space.
        volume: Option<u64>,
        content_range: Option<String>,
        calls: RefCell<Vec<Range<u64>>>,
    }

    impl FakeHost {
        fn new(free: u64) -> Self {
            Self {
                free,
                volume: None,
                content_range: None,
                calls: RefCell::default(),
            }
        }
    }

    impl RangeHost for &FakeHost {
        fn get_range(&self, url: &str, range: Range<u64>) -> std::io::Result<(Vec<u8>, String)> {
            assert_eq!(
                url,
                format!("https://huggingface.co/org/model/resolve/{REV}/{SHARD}")
            );
            self.calls.borrow_mut().push(range.clone());
            let total = payload_start() + PAYLOAD_BYTES;
            let body = range
                .clone()
                .map(|offset| u8::try_from(offset - payload_start()).expect("small"))
                .collect();
            let content_range = self
                .content_range
                .clone()
                .unwrap_or_else(|| format!("bytes {}-{}/{total}", range.start, range.end - 1));
            Ok((body, content_range))
        }

        fn free_bytes(&self, path: &Path) -> Option<u64> {
            let Some(volume) = self.volume else {
                return Some(self.free);
            };
            let stored: u64 = fs::read_dir(path)
                .expect("weights")
                .map(|entry| entry.expect("entry"))
                .filter(|entry| entry.file_name().to_string_lossy().ends_with(".bin"))
                .map(|entry| entry.metadata().expect("metadata").len())
                .sum();
            Some(volume - stored)
        }
    }

    /// A trace dir (index, headers.json, header file) and an empty weights dir.
    fn trace_dir() -> PathBuf {
        static UNIQUE: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "metallix-range-fetch-{}-{}",
            std::process::id(),
            UNIQUE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(dir.join("headers")).expect("headers dir");
        fs::create_dir_all(dir.join("weights")).expect("weights dir");
        let mut weight_map = serde_json::json!({"a": SHARD, "b": SHARD});
        for expert in 0..EXPERTS {
            for name in EXPERT_PARTS {
                weight_map[part(expert, name)] = SHARD.into();
            }
        }
        let index = serde_json::json!({
            "metadata": {"total_size": PAYLOAD_BYTES},
            "weight_map": weight_map,
        })
        .to_string();
        fs::write(dir.join("index.json"), &index).expect("index");
        let header_json = header();
        let mut header = (header_json.len() as u64).to_le_bytes().to_vec();
        header.extend_from_slice(header_json.as_bytes());
        fs::write(dir.join(format!("headers/{SHARD}.header.bin")), &header).expect("header");
        fs::write(
            dir.join("headers.json"),
            serde_json::json!({
                "revision": REV,
                "index_sha256": format!("{:x}", Sha256::digest(&index)),
                "shards": {SHARD: {
                    "header_bytes": header.len(),
                    "header_sha256": format!("{:x}", Sha256::digest(&header)),
                    "file_bytes": payload_start() + PAYLOAD_BYTES,
                }},
            })
            .to_string(),
        )
        .expect("headers.json");
        dir
    }

    fn cache<'a>(
        dir: &Path,
        host: &'a FakeHost,
        envelope: Envelope,
    ) -> V41RangeCache<FetchingSource<&'a FakeHost>> {
        let source =
            FetchingSource::new(dir.join("weights"), dir, "org/model", REV, host, envelope)
                .expect("source");
        V41RangeCache::load(source, &dir.join("index.json"), dir, REV, 1 << 20).expect("cache")
    }

    const ROOMY: Envelope = Envelope {
        max_acquired_bytes: 1 << 20,
        min_free_bytes: 100,
    };

    #[test]
    fn fetches_a_missing_tensor_once_and_writes_an_acquire_style_receipt() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        let mut cache = cache(&dir, &host, ROOMY);
        assert_eq!(
            &*cache.get_tensor("b").expect("fetched"),
            [16, 17, 18, 19, 20, 21, 22, 23]
        );
        let start = payload_start() + 16;
        assert_eq!(
            host.calls.borrow().as_slice(),
            std::slice::from_ref(&(start..start + 8))
        );
        let receipt: serde_json::Value =
            serde_json::from_slice(&fs::read(dir.join("weights/b.receipt.json")).expect("receipt"))
                .expect("receipt json");
        assert_eq!(
            receipt,
            serde_json::json!({
                "tensor": "b", "shard": SHARD, "range": [start, start + 8], "bytes": 8,
                "metadata": {"dtype": "F8_E8M0", "shape": [8], "data_offsets": [16, 24]},
                "sha256": format!("{:x}", Sha256::digest([16, 17, 18, 19, 20, 21, 22, 23])),
                "revision": REV,
            })
        );
        // A fresh cache over the same directory reads it locally, verified.
        let host2 = FakeHost::new(1 << 30);
        let mut again = self::cache(&dir, &host2, ROOMY);
        assert_eq!(again.get_tensor("b").expect("local").len(), 8);
        assert!(host2.calls.borrow().is_empty());
        let names: Vec<_> = fs::read_dir(dir.join("weights"))
            .expect("weights")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .into_string()
                    .expect("utf-8")
            })
            .collect();
        assert_eq!(names.len(), 2, "no .part files left behind: {names:?}");
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn refuses_fetches_outside_the_envelope_without_contacting_the_host() {
        let dir = trace_dir();
        fs::write(dir.join("weights/old.bin"), [0; 10]).expect("existing payload");
        for (envelope, free) in [
            (
                Envelope {
                    max_acquired_bytes: 17,
                    min_free_bytes: 0,
                },
                1 << 30,
            ),
            (
                Envelope {
                    max_acquired_bytes: 1 << 20,
                    min_free_bytes: 100,
                },
                107,
            ),
        ] {
            let host = FakeHost::new(free);
            let mut cache = cache(&dir, &host, envelope);
            assert!(matches!(
                cache.get_tensor("b"),
                Err(V41RangeCacheError::Io(_))
            ));
            assert!(host.calls.borrow().is_empty());
        }
        assert!(!dir.join("weights/b.bin").exists());
        assert!(
            dir.join("weights/old.bin").exists(),
            "non-expert data is never evicted"
        );
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn rejects_a_wrong_content_range_and_writes_nothing() {
        let dir = trace_dir();
        let mut host = FakeHost::new(1 << 30);
        host.content_range = Some(format!("bytes 0-7/{}", payload_start() + PAYLOAD_BYTES));
        let mut cache = cache(&dir, &host, ROOMY);
        let error = cache.get_tensor("b").expect_err("wrong range");
        assert!(error.to_string().contains("Content-Range"), "{error}");
        assert_eq!(
            fs::read_dir(dir.join("weights")).expect("weights").count(),
            0
        );
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn an_evicted_tensor_must_refetch_to_its_recorded_digest() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        cache(&dir, &host, ROOMY)
            .get_tensor("b")
            .expect("first fetch");
        fs::remove_file(dir.join("weights/b.bin")).expect("evict payload, keep receipt");
        let receipt_path = dir.join("weights/b.receipt.json");
        let receipt = fs::read_to_string(&receipt_path).expect("receipt");
        let digest = format!("{:x}", Sha256::digest([16, 17, 18, 19, 20, 21, 22, 23]));
        fs::write(&receipt_path, receipt.replace(&digest, &"0".repeat(64))).expect("tamper");
        let host = FakeHost::new(1 << 30);
        assert!(matches!(
            cache(&dir, &host, ROOMY).get_tensor("b"),
            Err(V41RangeCacheError::HashMismatch(_))
        ));
        assert!(!dir.join("weights/b.bin").exists());
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    fn row_names(dir: &Path) -> BTreeSet<String> {
        fs::read_dir(dir.join("weights/.rows/a"))
            .map(|entries| {
                entries
                    .map(|entry| {
                        entry
                            .expect("entry")
                            .file_name()
                            .into_string()
                            .expect("utf-8")
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn row_reads_store_each_row_so_a_regrouped_repeat_fetches_only_new_rows() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        assert_eq!(
            &*cache(&dir, &host, ROOMY).get_rows("a", 1..3).expect("rows"),
            [4, 5, 6, 7, 8, 9, 10, 11]
        );
        let start = payload_start();
        assert_eq!(
            host.calls.borrow().as_slice(),
            std::slice::from_ref(&(start + 4..start + 12))
        );
        assert_eq!(
            row_names(&dir),
            BTreeSet::from(
                ["1.bin", "1.receipt.json", "2.bin", "2.receipt.json"].map(str::to_owned)
            )
        );
        let receipt: serde_json::Value = serde_json::from_slice(
            &fs::read(dir.join("weights/.rows/a/2.receipt.json")).expect("receipt"),
        )
        .expect("receipt json");
        assert_eq!(
            receipt,
            serde_json::json!({
                "tensor": "a", "shard": SHARD, "row": 2, "range": [start + 8, start + 12],
                "sha256": format!("{:x}", Sha256::digest([8, 9, 10, 11])), "revision": REV,
            })
        );
        // A fresh cache, rows grouped differently: only row 3 goes to the network.
        let host = FakeHost::new(1 << 30);
        let source = FetchingSource::new(dir.join("weights"), &dir, "org/model", REV, &host, ROOMY)
            .expect("source");
        assert_eq!(
            source.acquired_bytes(),
            8,
            "stored rows count against the envelope"
        );
        let mut regrouped =
            V41RangeCache::load(source, &dir.join("index.json"), &dir, REV, 1 << 20)
                .expect("cache");
        assert_eq!(
            &*regrouped.get_rows("a", 2..4).expect("rows"),
            [8, 9, 10, 11, 12, 13, 14, 15]
        );
        assert_eq!(
            host.calls.borrow().as_slice(),
            std::slice::from_ref(&(start + 12..start + 16))
        );
        // And a third read of stored rows needs no network at all.
        let host = FakeHost::new(1 << 30);
        assert_eq!(
            cache(&dir, &host, ROOMY)
                .get_rows("a", 1..4)
                .expect("rows")
                .len(),
            12
        );
        assert!(host.calls.borrow().is_empty());
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn a_stored_row_is_checked_on_every_read_and_refetched_to_its_digest() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        cache(&dir, &host, ROOMY)
            .get_rows("a", 2..3)
            .expect("first fetch");
        let row = dir.join("weights/.rows/a/2.bin");
        fs::write(&row, [8, 9, 10, 0]).expect("corrupt");
        let host = FakeHost::new(1 << 30);
        assert!(matches!(
            cache(&dir, &host, ROOMY).get_rows("a", 2..3),
            Err(V41RangeCacheError::HashMismatch(_))
        ));
        assert!(
            host.calls.borrow().is_empty(),
            "a corrupt row is not silently refetched"
        );
        // A lost payload refetches, and the bytes must match the kept receipt.
        fs::remove_file(&row).expect("lose payload");
        let receipt_path = dir.join("weights/.rows/a/2.receipt.json");
        let receipt = fs::read_to_string(&receipt_path).expect("receipt");
        let digest = format!("{:x}", Sha256::digest([8, 9, 10, 11]));
        fs::write(&receipt_path, receipt.replace(&digest, &"0".repeat(64))).expect("tamper");
        let host = FakeHost::new(1 << 30);
        assert!(matches!(
            cache(&dir, &host, ROOMY).get_rows("a", 2..3),
            Err(V41RangeCacheError::HashMismatch(_))
        ));
        assert!(!row.exists());
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn row_fetches_respect_the_envelope() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        let tight = Envelope {
            max_acquired_bytes: 7,
            min_free_bytes: 0,
        };
        assert!(matches!(
            cache(&dir, &host, tight).get_rows("a", 1..3),
            Err(V41RangeCacheError::Io(_))
        ));
        assert!(host.calls.borrow().is_empty());
        assert!(row_names(&dir).is_empty());
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    /// Room for exactly two experts and nothing else.
    const TWO_EXPERTS: Envelope = Envelope {
        max_acquired_bytes: 2 * EXPERT_BYTES,
        min_free_bytes: 0,
    };

    /// A cache that keeps one part in memory, so every expert read reaches
    /// the source.
    fn expert_cache<'a>(
        dir: &Path,
        host: &'a FakeHost,
        envelope: Envelope,
    ) -> V41RangeCache<FetchingSource<&'a FakeHost>> {
        let source =
            FetchingSource::new(dir.join("weights"), dir, "org/model", REV, host, envelope)
                .expect("source");
        V41RangeCache::load(source, &dir.join("index.json"), dir, REV, 2).expect("cache")
    }

    fn read_expert(
        cache: &mut V41RangeCache<FetchingSource<&FakeHost>>,
        expert: u32,
    ) -> Result<(), V41RangeCacheError> {
        for name in EXPERT_PARTS {
            cache.get_tensor(&part(expert, name))?;
        }
        Ok(())
    }

    /// Payloads on disk for each part of `expert`, in [`EXPERT_PARTS`] order.
    fn stored_parts(dir: &Path, expert: u32) -> Vec<bool> {
        EXPERT_PARTS
            .iter()
            .map(|name| {
                dir.join(format!("weights/{}.bin", part(expert, name)))
                    .exists()
            })
            .collect()
    }

    /// Experts with every part on disk; panics on a partially stored expert.
    fn stored_experts(dir: &Path) -> BTreeSet<u32> {
        (0..EXPERTS)
            .filter(|&expert| {
                let parts = stored_parts(dir, expert);
                assert!(
                    parts.iter().all(|&p| p) || parts.iter().all(|&p| !p),
                    "expert {expert} is partially stored: {parts:?}"
                );
                parts[0]
            })
            .collect()
    }

    fn pin(dir: &Path, experts: &[u32]) {
        let experts: Vec<_> = experts.iter().map(|&expert| [0, expert]).collect();
        fs::write(
            dir.join("pinned-experts.json"),
            serde_json::json!({"experts": experts}).to_string(),
        )
        .expect("pin file");
    }

    #[test]
    fn lru_eviction_honors_read_hits() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        let mut cache = expert_cache(&dir, &host, TWO_EXPERTS);
        read_expert(&mut cache, 0).expect("expert 0");
        read_expert(&mut cache, 1).expect("expert 1");
        // A local hit on expert 0 makes expert 1 the least recently used.
        read_expert(&mut cache, 0).expect("expert 0 again");
        assert_eq!(host.calls.borrow().len(), 12, "the hit fetched nothing");
        read_expert(&mut cache, 2).expect("expert 2 evicts rather than refuses");
        assert_eq!(stored_experts(&dir), BTreeSet::from([0, 2]));
        // The evicted expert keeps its receipts, so a refetch is verified.
        for name in EXPERT_PARTS {
            assert!(
                dir.join(format!("weights/{}.receipt.json", part(1, name)))
                    .exists()
            );
        }
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn memory_hits_keep_an_expert_recent_on_disk() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        // Room for every part in memory, so the second read of expert 0
        // never reaches the source.
        let mut cache = cache(&dir, &host, TWO_EXPERTS);
        read_expert(&mut cache, 0).expect("expert 0");
        read_expert(&mut cache, 1).expect("expert 1");
        read_expert(&mut cache, 0).expect("expert 0 from memory");
        read_expert(&mut cache, 2).expect("expert 2");
        assert_eq!(stored_experts(&dir), BTreeSet::from([0, 2]));
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn recency_survives_a_restart_through_the_journal() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        let mut cache = expert_cache(&dir, &host, TWO_EXPERTS);
        read_expert(&mut cache, 0).expect("expert 0");
        read_expert(&mut cache, 1).expect("expert 1");
        read_expert(&mut cache, 0).expect("expert 0 again");
        drop(cache);
        // Without the journal, name order would make expert 0 the oldest.
        let host = FakeHost::new(1 << 30);
        let mut cache = expert_cache(&dir, &host, TWO_EXPERTS);
        read_expert(&mut cache, 2).expect("expert 2");
        assert_eq!(stored_experts(&dir), BTreeSet::from([0, 2]));
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn eviction_removes_a_whole_expert_even_when_one_part_would_do() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        let mut cache = expert_cache(&dir, &host, TWO_EXPERTS);
        read_expert(&mut cache, 0).expect("expert 0");
        read_expert(&mut cache, 1).expect("expert 1");
        // The first 2-byte part of expert 2 needs 2 bytes; all 12 of expert 0 go.
        cache
            .get_tensor(&part(2, EXPERT_PARTS[0]))
            .expect("first part of expert 2");
        assert_eq!(stored_parts(&dir, 0), [false; 6]);
        assert_eq!(stored_parts(&dir, 1), [true; 6]);
        // The rest of expert 2 then fits without touching expert 1.
        read_expert(&mut cache, 2).expect("rest of expert 2");
        assert_eq!(stored_experts(&dir), BTreeSet::from([1, 2]));
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn pinned_experts_survive_pressure() {
        let dir = trace_dir();
        pin(&dir, &[0]);
        let host = FakeHost::new(1 << 30);
        let mut cache = expert_cache(&dir, &host, TWO_EXPERTS);
        read_expert(&mut cache, 0).expect("expert 0");
        read_expert(&mut cache, 1).expect("expert 1");
        // Expert 0 is least recently used but pinned.
        read_expert(&mut cache, 2).expect("expert 2");
        read_expert(&mut cache, 3).expect("expert 3");
        assert_eq!(stored_experts(&dir), BTreeSet::from([0, 3]));
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn free_floor_pressure_evicts_instead_of_refusing() {
        let dir = trace_dir();
        let mut host = FakeHost::new(0);
        // The volume holds the floor plus two experts; the cap is no constraint.
        host.volume = Some(100 + 2 * EXPERT_BYTES);
        let envelope = Envelope {
            max_acquired_bytes: 1 << 20,
            min_free_bytes: 100,
        };
        let mut cache = expert_cache(&dir, &host, envelope);
        read_expert(&mut cache, 0).expect("expert 0");
        read_expert(&mut cache, 1).expect("expert 1");
        read_expert(&mut cache, 2).expect("expert 2 evicts rather than refuses");
        assert_eq!(stored_experts(&dir), BTreeSet::from([1, 2]));
        let free = (&host).free_bytes(&dir.join("weights")).expect("free");
        assert!(free >= 100, "{free}");
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn refuses_only_when_pinned_experts_alone_fill_the_cap() {
        let dir = trace_dir();
        pin(&dir, &[0, 1]);
        let host = FakeHost::new(1 << 30);
        let mut cache = expert_cache(&dir, &host, TWO_EXPERTS);
        read_expert(&mut cache, 0).expect("expert 0");
        read_expert(&mut cache, 1).expect("expert 1");
        let error = read_expert(&mut cache, 2).expect_err("no unpinned expert to evict");
        assert!(error.to_string().contains("unpinned"), "{error}");
        assert_eq!(host.calls.borrow().len(), 12, "refused before downloading");
        assert_eq!(stored_experts(&dir), BTreeSet::from([0, 1]));
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn a_crash_mid_eviction_leaves_a_state_the_next_read_repairs() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        read_expert(&mut expert_cache(&dir, &host, ROOMY), 0).expect("expert 0");
        // Eviction unlinks payloads only, in part order; crash after three.
        for name in &EXPERT_PARTS[..3] {
            fs::remove_file(dir.join(format!("weights/{}.bin", part(0, name)))).expect("unlink");
        }
        let host = FakeHost::new(1 << 30);
        let source = FetchingSource::new(dir.join("weights"), &dir, "org/model", REV, &host, ROOMY)
            .expect("source");
        assert_eq!(
            source.acquired_bytes(),
            6,
            "only the surviving payloads count"
        );
        let mut cache =
            V41RangeCache::load(source, &dir.join("index.json"), &dir, REV, 2).expect("cache");
        read_expert(&mut cache, 0).expect("repaired");
        // Exactly the three missing parts were refetched, against kept receipts.
        let start = payload_start() + 24;
        assert_eq!(
            host.calls.borrow().as_slice(),
            [start..start + 2, start + 2..start + 4, start + 4..start + 6]
        );
        assert_eq!(stored_parts(&dir, 0), [true; 6]);
        assert_eq!(
            &*cache.get_tensor(&part(0, EXPERT_PARTS[0])).expect("part"),
            [24, 25]
        );
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    #[ignore = "fetches 2,560 bytes from huggingface.co; needs .agents/receipts/route-trace metadata"]
    fn fetches_one_real_scale_tensor_matching_its_recorded_receipt() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.agents/receipts");
        let trace = root.join("route-trace");
        let registry = crate::model_registry::deepseek().expect("registry");
        let weights = trace_dir().join("weights");
        let source = FetchingSource::new(
            &weights,
            &trace,
            registry.model_repo,
            registry.model_revision.clone(),
            CurlHost,
            Envelope::default(),
        )
        .expect("source");
        let mut cache = V41RangeCache::load(
            source,
            &root.join(
                "control/receipts/candidate-control/real-expert/model.safetensors.index.json",
            ),
            &trace,
            &registry.model_revision,
            1 << 20,
        )
        .expect("pinned cache");
        let tensor = "layers.0.attn.wkv.scale";
        let bytes = cache.get_tensor(tensor).expect("fetched over the network");
        let recorded = |dir: &Path| -> serde_json::Value {
            serde_json::from_slice(
                &fs::read(dir.join(format!("{tensor}.receipt.json"))).expect("receipt"),
            )
            .expect("receipt json")
        };
        assert_eq!(bytes.len(), 2_560);
        assert_eq!(recorded(&weights), recorded(&trace.join("weights")));
        fs::remove_dir_all(weights.parent().expect("temp root")).expect("cleanup own temp dir");
    }
}
