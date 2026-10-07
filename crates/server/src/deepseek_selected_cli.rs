//! Local-only, fixed-geometry layer-zero `MoE` comparison against independent captures.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{self, Read},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use clap::Args;
use deepseek::{
    checkpoint::{
        V41SafetensorsHeader, V41StorageDtype,
        range_cache::{
            V41CachedRoutedExperts, V41LocalWeightsSource, V41RangeCache, V41RangeCacheError,
            V41RangeRequest, V41RangeSource,
        },
    },
    manifest::V41SafetensorsIndex,
    moe::{Fp8ExpertWeights, MoEConfig, MoEReference},
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 2304;
const EXPERTS: usize = 384;
const MIB: u64 = 1024 * 1024;

/// Compare up to three captured BF16 rows through the pinned layer-zero `MoE`.
#[derive(Debug, Args)]
pub(crate) struct SelectedArgs {
    #[arg(long)]
    index: PathBuf,
    /// Directory containing headers.json and headers/<shard>.header.bin.
    #[arg(long)]
    headers_dir: PathBuf,
    /// Local per-tensor .bin and .receipt.json files; no fetching is performed.
    #[arg(long)]
    weights_dir: PathBuf,
    #[arg(long)]
    revision: String,
    #[arg(long)]
    input_bf16: PathBuf,
    /// Independent source-captured output, not output from this implementation.
    #[arg(long)]
    expected_bf16: PathBuf,
    #[arg(long)]
    input_sha256: String,
    #[arg(long)]
    expected_sha256: String,
    /// Source capture with revision and runs[0].routes layer-zero IDs.
    #[arg(long)]
    routes: PathBuf,
    #[arg(long)]
    tokens: usize,
    /// In-memory cache membership cap, not a process/retained-memory ceiling.
    #[arg(long, default_value_t = 512)]
    cache_budget_mib: u64,
    /// Cap on cumulative source ranges requested, excluding hashing/receipt I/O.
    #[arg(long, default_value_t = 512)]
    payload_budget_mib: u64,
}

impl SelectedArgs {
    pub(crate) fn run(self) -> ExitCode {
        match run(&self) {
            Ok(true) => ExitCode::SUCCESS,
            Ok(false) => {
                eprintln!("deepseek selected comparison failed");
                ExitCode::FAILURE
            }
            Err(error) => {
                eprintln!("deepseek selected refused: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bounded_read(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|error| error.to_string())?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err("input is not a regular file within its byte bound".to_owned());
    }
    let mut bytes = Vec::new();
    file.take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > maximum || bytes.len() as u64 != metadata.len() {
        return Err("input changed size or exceeded its byte bound".to_owned());
    }
    Ok(bytes)
}

fn validate_args(args: &SelectedArgs) -> Result<(), String> {
    if args.revision != REVISION {
        return Err("revision is not the pinned V4.1 source".to_owned());
    }
    if !(1..=3).contains(&args.tokens) {
        return Err("tokens must be in 1..=3".to_owned());
    }
    if !(1..=512).contains(&args.cache_budget_mib) || !(1..=512).contains(&args.payload_budget_mib)
    {
        return Err("cache and payload budgets must be in 1..=512 MiB".to_owned());
    }
    for value in [&args.input_sha256, &args.expected_sha256] {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("input/expected SHA256 must have 64 lowercase hex characters".to_owned());
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct Capture {
    revision: String,
    runs: Vec<CaptureRun>,
}
#[derive(Deserialize)]
struct CaptureRun {
    routes: Vec<CaptureRoutes>,
}
#[derive(Deserialize)]
struct CaptureRoutes {
    layer: usize,
    ids: Vec<Vec<usize>>,
}

fn expected_routes(bytes: &[u8], tokens: usize) -> Result<Vec<Vec<usize>>, String> {
    let capture: Capture = serde_json::from_slice(bytes).map_err(|error| error.to_string())?;
    if capture.revision != REVISION || capture.runs.len() != 1 {
        return Err("route capture must have pinned revision and one run".to_owned());
    }
    let mut layers = capture
        .runs
        .into_iter()
        .flat_map(|run| run.routes)
        .filter(|route| route.layer == 0);
    let mut rows = layers.next().ok_or("route capture lacks layer zero")?.ids;
    if layers.next().is_some() || rows.len() != tokens {
        return Err(
            "route capture must contain one layer-zero entry with exactly tokens rows".to_owned(),
        );
    }
    for row in &mut rows {
        row.sort_unstable();
        if row.len() != 6
            || row.iter().any(|id| *id >= EXPERTS)
            || row.windows(2).any(|pair| pair[0] == pair[1])
        {
            return Err("route row must contain six distinct expert IDs below 384".to_owned());
        }
    }
    Ok(rows)
}

struct TensorSpec {
    dtype: V41StorageDtype,
    shape: Vec<u64>,
    bytes: u64,
}
fn tensor_specs(routes: &[Vec<usize>]) -> BTreeMap<String, TensorSpec> {
    use V41StorageDtype::{Bf16, F8E4M3Fn, F8E8M0Fnu, F32, I8};
    let mut specs = BTreeMap::new();
    let mut add = |name: String, dtype, shape: Vec<u64>, bytes| {
        specs.insert(
            name,
            TensorSpec {
                dtype,
                shape,
                bytes,
            },
        );
    };
    add(
        "layers.0.ffn.gate.weight".to_owned(),
        Bf16,
        vec![384, 5120],
        3_932_160,
    );
    add("layers.0.ffn.gate.bias".to_owned(), F32, vec![384], 1536);
    for projection in ["w1", "w2", "w3"] {
        let shape = if projection == "w2" {
            vec![5120, 2304]
        } else {
            vec![2304, 5120]
        };
        let scale = if projection == "w2" {
            vec![160, 72]
        } else {
            vec![72, 160]
        };
        add(
            format!("layers.0.ffn.shared_experts.{projection}.weight"),
            F8E4M3Fn,
            shape,
            11_796_480,
        );
        add(
            format!("layers.0.ffn.shared_experts.{projection}.scale"),
            F8E8M0Fnu,
            scale,
            11_520,
        );
    }
    for id in routes.iter().flatten().collect::<BTreeSet<_>>() {
        for projection in ["w1", "w2", "w3"] {
            let shape = if projection == "w2" {
                vec![5120, 1152]
            } else {
                vec![2304, 2560]
            };
            let scale = if projection == "w2" {
                vec![5120, 72]
            } else {
                vec![2304, 160]
            };
            add(
                format!("layers.0.ffn.experts.{id}.{projection}.weight"),
                I8,
                shape,
                5_898_240,
            );
            add(
                format!("layers.0.ffn.experts.{id}.{projection}.scale"),
                F8E8M0Fnu,
                scale,
                368_640,
            );
        }
    }
    specs
}

#[derive(Default)]
struct Counters {
    charged: Mutex<u64>,
    returned: AtomicU64,
    hits: AtomicU64,
}
struct SelectedSource {
    inner: V41LocalWeightsSource,
    allowed: BTreeMap<String, TensorSpec>,
    budget: u64,
    counters: Arc<Counters>,
}
impl V41RangeSource for SelectedSource {
    fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        let refused = || {
            V41RangeCacheError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                "selected payload allowlist or cumulative range budget exceeded",
            ))
        };
        let expected = self.allowed.get(request.tensor).ok_or_else(refused)?;
        if request.range != request.tensor_range.file_range()
            || request.tensor_range.byte_length() != expected.bytes
            || request.tensor_range.dtype() != expected.dtype
            || request.tensor_range.shape() != expected.shape
        {
            return Err(refused());
        }
        {
            let mut charged = self.counters.charged.lock().map_err(|_| {
                V41RangeCacheError::Io(io::Error::other("selected budget lock poisoned"))
            })?;
            *charged = charged
                .checked_add(expected.bytes)
                .filter(|total| *total <= self.budget)
                .ok_or_else(refused)?;
        }
        let bytes = self.inner.read_range(request)?;
        self.counters
            .returned
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }
    fn note_memory_hit(&self, _request: &V41RangeRequest<'_>) {
        self.counters.hits.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Deserialize)]
struct HeaderManifest {
    revision: String,
    index_sha256: String,
    shards: BTreeMap<String, HeaderRecord>,
}
#[derive(Deserialize)]
struct HeaderRecord {
    header_bytes: u64,
    header_sha256: String,
    file_bytes: u64,
}

fn load_cache(
    args: &SelectedArgs,
    source: SelectedSource,
) -> Result<(V41RangeCache<SelectedSource>, String, String), String> {
    let manifest_bytes = bounded_read(&args.headers_dir.join("headers.json"), MIB)?;
    let manifest: HeaderManifest =
        serde_json::from_slice(&manifest_bytes).map_err(|error| error.to_string())?;
    if manifest.revision != REVISION || manifest.shards.is_empty() || manifest.shards.len() > 64 {
        return Err("header manifest revision/count is invalid".to_owned());
    }
    let total = manifest
        .shards
        .values()
        .try_fold(0_u64, |sum, shard| sum.checked_add(shard.header_bytes))
        .ok_or("header size overflow")?;
    if total > 32 * MIB {
        return Err("aggregate headers exceed 32 MiB".to_owned());
    }
    let index_bytes = bounded_read(&args.index, 16 * MIB)?;
    let index_sha = digest(&index_bytes);
    if index_sha != manifest.index_sha256 {
        return Err("index SHA256 disagrees with header manifest".to_owned());
    }
    let index = V41SafetensorsIndex::parse(
        std::str::from_utf8(&index_bytes).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let mut headers = Vec::new();
    for (shard, record) in manifest.shards {
        if shard.is_empty() || shard.starts_with('.') || shard.contains(['/', '\\']) {
            return Err("invalid header shard name".to_owned());
        }
        let bytes = bounded_read(
            &args.headers_dir.join(format!("headers/{shard}.header.bin")),
            record.header_bytes,
        )?;
        if bytes.len() as u64 != record.header_bytes || digest(&bytes) != record.header_sha256 {
            return Err("header length/SHA256 mismatch".to_owned());
        }
        let header = V41SafetensorsHeader::parse_prefixed_header(&bytes, record.file_bytes)
            .map_err(|error| error.to_string())?;
        headers.push((shard, header));
    }
    let cache = V41RangeCache::new(source, index, headers, args.cache_budget_mib * MIB)
        .map_err(|error| error.to_string())?;
    Ok((cache, index_sha, digest(&manifest_bytes)))
}

fn preflight_payloads(
    args: &SelectedArgs,
    cache: &V41RangeCache<SelectedSource>,
    specs: &BTreeMap<String, TensorSpec>,
) -> Result<(), String> {
    let mut total = 0_u64;
    for (name, spec) in specs {
        let range = cache
            .tensor_range(name)
            .map_err(|error| error.to_string())?;
        if range.dtype() != spec.dtype
            || range.shape() != spec.shape
            || range.byte_length() != spec.bytes
        {
            return Err(format!("fixed V4.1 tensor geometry mismatch: {name}"));
        }
        total = total
            .checked_add(spec.bytes)
            .ok_or("payload size overflow")?;
        if total > args.payload_budget_mib * MIB || spec.bytes > args.cache_budget_mib * MIB {
            return Err("selected payload inventory exceeds explicit budgets".to_owned());
        }
        let metadata = fs::metadata(args.weights_dir.join(format!("{name}.bin")))
            .map_err(|error| error.to_string())?;
        if !metadata.is_file() || metadata.len() != spec.bytes {
            return Err(format!("payload file length mismatch: {name}"));
        }
        // Inputs are operator-owned immutable captures. The delegated reader rechecks payload stat/hash.
        bounded_read(
            &args.weights_dir.join(format!("{name}.receipt.json")),
            16 * 1024,
        )?;
    }
    Ok(())
}

fn bf16_words(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

fn run(args: &SelectedArgs) -> Result<bool, String> {
    validate_args(args)?;
    let bytes_per_capture = (args.tokens * HIDDEN * 2) as u64;
    let input_bytes = bounded_read(&args.input_bf16, bytes_per_capture)?;
    let expected_bytes = bounded_read(&args.expected_bf16, bytes_per_capture)?;
    if input_bytes.len() as u64 != bytes_per_capture
        || expected_bytes.len() as u64 != bytes_per_capture
    {
        return Err(
            "input and expected BF16 captures must have exactly tokens * 5120 values".to_owned(),
        );
    }
    if digest(&input_bytes) != args.input_sha256 || digest(&expected_bytes) != args.expected_sha256
    {
        return Err("input/expected SHA256 mismatch".to_owned());
    }
    let routes_bytes = bounded_read(&args.routes, MIB)?;
    let routes = expected_routes(&routes_bytes, args.tokens)?;
    let specs = tensor_specs(&routes);
    let counters = Arc::new(Counters::default());
    let source = SelectedSource {
        inner: V41LocalWeightsSource::new(&args.weights_dir, REVISION),
        allowed: tensor_specs(&routes),
        budget: args.payload_budget_mib * MIB,
        counters: Arc::clone(&counters),
    };
    let (cache, index_sha, headers_sha) = load_cache(args, source)?;
    preflight_payloads(args, &cache, &specs)?;
    let (rows, matched, cache_bytes) = evaluate(cache, &input_bytes, &expected_bytes, &routes)?;
    let charged_bytes = *counters
        .charged
        .lock()
        .map_err(|_| "selected budget lock poisoned")?;
    println!(
        "{}",
        json!({
            "schema_version": 1,
            "operation": "deepseek-selected-layer0-moe",
            "scope": "local selected weights, scalar library MoE; not full checkpoint inference",
            "revision": REVISION,
            "tokens": args.tokens,
            "input_sha256": args.input_sha256,
            "expected_sha256": args.expected_sha256,
            "routes_sha256": digest(&routes_bytes),
            "index_sha256": index_sha,
            "headers_manifest_sha256": headers_sha,
            "comparison": "source expert IDs and exact BF16 bits",
            "matched": matched,
            "rows": rows,
            "cache_membership_bytes": cache_bytes,
            "source_returned_range_bytes": counters.returned.load(Ordering::Relaxed),
            "source_range_budget_charged_bytes": charged_bytes,
            "memory_cache_hits": counters.hits.load(Ordering::Relaxed),
            "physical_read_bytes": null,
            "physical_read_bytes_status": "unavailable; returned ranges exclude hash and receipt I/O",
            "cache_budget_bytes": args.cache_budget_mib * MIB,
            "payload_budget_bytes": args.payload_budget_mib * MIB,
        })
    );
    Ok(matched)
}

fn evaluate(
    mut cache: V41RangeCache<SelectedSource>,
    input_bytes: &[u8],
    expected_bytes: &[u8],
    routes: &[Vec<usize>],
) -> Result<(Vec<serde_json::Value>, bool, u64), String> {
    let mut get = |name: &str| cache.get_tensor(name).map_err(|error| error.to_string());
    let gate = bf16_words(&get("layers.0.ffn.gate.weight")?);
    let bias = get("layers.0.ffn.gate.bias")?
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect::<Vec<_>>();
    let shared = ["w1", "w2", "w3"]
        .into_iter()
        .flat_map(|projection| {
            ["weight", "scale"]
                .map(|kind| format!("layers.0.ffn.shared_experts.{projection}.{kind}"))
        })
        .map(|name| get(&name))
        .collect::<Result<Vec<_>, _>>()?;
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
    .map_err(|error| error.to_string())?;
    let table = vec![None; EXPERTS];
    let moe = MoEReference::new_sparse(
        MoEConfig::new(HIDDEN, INTERMEDIATE, 10.0, 6, 1.0, true, 1.5)
            .map_err(|error| error.to_string())?,
        &gate,
        &bias,
        &table,
        shared,
    )
    .map_err(|error| error.to_string())?;
    let cache = Mutex::new(cache);
    let experts = V41CachedRoutedExperts::new(&cache, 0, HIDDEN, INTERMEDIATE);
    let input = bf16_words(input_bytes);
    let expected = bf16_words(expected_bytes);
    let mut rows = Vec::new();
    let mut matched = true;
    for (token, (input, expected)) in input
        .chunks_exact(HIDDEN)
        .zip(expected.chunks_exact(HIDDEN))
        .enumerate()
    {
        let result = moe
            .forward_token_with(input, &experts)
            .map_err(|error| error.to_string())?;
        let actual_ids: Vec<_> = result
            .routes()
            .iter()
            .map(|route| route.expert_index())
            .collect();
        let (row_matched, mismatch_count) =
            compare_row(&actual_ids, &routes[token], result.output_bf16(), expected);
        matched &= row_matched;
        rows.push(json!({"row": token, "expected_ids": routes[token], "actual_ids": actual_ids, "bf16_mismatches": mismatch_count, "matched": row_matched}));
    }
    let cache_bytes = cache
        .lock()
        .map_err(|_| "range cache lock poisoned")?
        .used_bytes();
    Ok((rows, matched, cache_bytes))
}

fn compare_row(
    actual_ids: &[usize],
    expected_ids: &[usize],
    actual: &[u16],
    expected: &[u16],
) -> (bool, usize) {
    let mismatches = actual
        .iter()
        .zip(expected)
        .filter(|(left, right)| left != right)
        .count()
        + actual.len().abs_diff(expected.len());
    (actual_ids == expected_ids && mismatches == 0, mismatches)
}

#[cfg(test)]
mod tests {
    use super::compare_row;

    #[test]
    fn comparison_rejects_changed_bits_routes_and_missing_output() {
        assert_eq!(
            compare_row(&[1, 2], &[1, 2], &[0x3f80], &[0x3f80]),
            (true, 0)
        );
        assert_eq!(
            compare_row(&[1, 2], &[1, 2], &[0x3f81], &[0x3f80]),
            (false, 1)
        );
        assert_eq!(
            compare_row(&[1, 3], &[1, 2], &[0x3f80], &[0x3f80]),
            (false, 0)
        );
        assert_eq!(compare_row(&[1, 2], &[1, 2], &[], &[0x3f80]), (false, 1));
    }
}
