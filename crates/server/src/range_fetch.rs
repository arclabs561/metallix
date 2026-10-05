//! Network fallback for `DeepSeek` V4.1 checkpoint range reads.
//!
//! [`FetchingSource`] serves a range from the local route-trace weights store
//! and, only when that store reports a tensor as absent, fetches the exact
//! byte range from the pinned Hugging Face revision. A whole-tensor fetch is
//! written as `<tensor>.bin` plus the receipt the acquisition scripts write,
//! then re-read through the local source so it passes the same receipt and
//! SHA-256 checks as any other local tensor. A sub-tensor (table row) fetch is
//! returned without being stored: no receipt can digest a partial tensor, so
//! it is checked only by exact length and `Content-Range`.
//!
//! Stored bytes stay inside an acquisition envelope (total stored bytes and a
//! free-disk floor). A fetch that would cross it is refused; nothing is evicted.

use std::{
    collections::BTreeMap,
    fs, io,
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
    /// The route-trace acquisition envelope: 64 GiB stored, 150 GiB free.
    fn default() -> Self {
        Self {
            max_acquired_bytes: 64 << 30,
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

#[derive(Deserialize, Serialize)]
struct ReceiptMetadata {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
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
    acquired: Mutex<u64>,
}

impl<H: RangeHost> FetchingSource<H> {
    /// Serves `weights_dir` for `repo` at `revision`. Shard lengths come from
    /// `trace_dir/headers.json`, whose revision must match. The current
    /// `*.bin` total in `weights_dir` counts against `envelope`.
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
        let mut acquired = 0;
        for entry in fs::read_dir(&weights_dir)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().ends_with(".bin") {
                acquired += entry.metadata()?.len();
            }
        }
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
            acquired: Mutex::new(acquired),
        })
    }

    /// Total `*.bin` bytes counted against the envelope.
    pub fn acquired_bytes(&self) -> u64 {
        *self
            .acquired
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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
        // Held across the write so concurrent misses cannot jointly overrun.
        let mut acquired = self
            .acquired
            .lock()
            .map_err(|_| fail("acquisition lock poisoned".to_owned()))?;
        if *acquired + length > self.envelope.max_acquired_bytes {
            return Err(fail(format!(
                "fetching {tensor} ({length} bytes) would exceed the {}-byte acquisition envelope ({} stored)",
                self.envelope.max_acquired_bytes, *acquired
            )));
        }
        let free = self
            .host
            .free_bytes(&self.weights_dir)
            .ok_or_else(|| fail("cannot determine free space for the weights volume".to_owned()))?;
        if free.saturating_sub(length) < self.envelope.min_free_bytes {
            return Err(fail(format!(
                "fetching {tensor} ({length} bytes) would leave less than {} bytes free",
                self.envelope.min_free_bytes
            )));
        }

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
        write_atomically(&self.weights_dir.join(format!("{tensor}.bin")), &bytes)?;
        *acquired += length;
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

impl<H: RangeHost> V41RangeSource for FetchingSource<H> {
    fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        match self.local.read_range(request) {
            Err(V41RangeCacheError::NotLocal(_)) => {}
            result => return result,
        }
        if request.range != request.tensor_range.file_range() {
            return self.fetch(request.shard, request.range.clone());
        }
        self.acquire_tensor(request)?;
        self.local.read_range(request)
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
        fs,
        ops::Range,
        path::{Path, PathBuf},
        sync::atomic::{AtomicUsize, Ordering},
    };

    use deepseek::checkpoint::range_cache::{V41RangeCache, V41RangeCacheError};
    use sha2::{Digest, Sha256};

    use super::{CurlHost, Envelope, FetchingSource, RangeHost};

    const SHARD: &str = "model-00001-of-00001.safetensors";
    const REV: &str = "rev";
    // `a` is a 4x4 U8 table at payload bytes 0..16; `b` is 8 E8M0 scales at 16..24.
    const HEADER: &str = r#"{"a":{"dtype":"U8","shape":[4,4],"data_offsets":[0,16]},"b":{"dtype":"F8_E8M0","shape":[8],"data_offsets":[16,24]}}"#;

    fn payload_start() -> u64 {
        8 + HEADER.len() as u64
    }

    /// Serves one synthetic shard whose payload byte `i` is `i`.
    struct FakeHost {
        free: u64,
        content_range: Option<String>,
        calls: RefCell<Vec<Range<u64>>>,
    }

    impl FakeHost {
        fn new(free: u64) -> Self {
            Self {
                free,
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
            let total = payload_start() + 24;
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

        fn free_bytes(&self, _: &Path) -> Option<u64> {
            Some(self.free)
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
        let index = format!(
            r#"{{"metadata":{{"total_size":24}},"weight_map":{{"a":"{SHARD}","b":"{SHARD}"}}}}"#
        );
        fs::write(dir.join("index.json"), &index).expect("index");
        let mut header = (HEADER.len() as u64).to_le_bytes().to_vec();
        header.extend_from_slice(HEADER.as_bytes());
        fs::write(dir.join(format!("headers/{SHARD}.header.bin")), &header).expect("header");
        fs::write(
            dir.join("headers.json"),
            serde_json::json!({
                "revision": REV,
                "index_sha256": format!("{:x}", Sha256::digest(&index)),
                "shards": {SHARD: {
                    "header_bytes": header.len(),
                    "header_sha256": format!("{:x}", Sha256::digest(&header)),
                    "file_bytes": payload_start() + 24,
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
        fs::remove_dir_all(dir).expect("cleanup own temp dir");
    }

    #[test]
    fn rejects_a_wrong_content_range_and_writes_nothing() {
        let dir = trace_dir();
        let mut host = FakeHost::new(1 << 30);
        host.content_range = Some(format!("bytes 0-7/{}", payload_start() + 24));
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

    #[test]
    fn row_reads_fetch_only_the_rows_and_store_nothing() {
        let dir = trace_dir();
        let host = FakeHost::new(1 << 30);
        let mut cache = cache(&dir, &host, ROOMY);
        assert_eq!(&*cache.get_rows("a", 2..3).expect("row"), [8, 9, 10, 11]);
        let start = payload_start() + 8;
        assert_eq!(
            host.calls.borrow().as_slice(),
            std::slice::from_ref(&(start..start + 4))
        );
        assert_eq!(
            fs::read_dir(dir.join("weights")).expect("weights").count(),
            0
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
