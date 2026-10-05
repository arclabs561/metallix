//! `V41RangeCache` hit versus miss for one expert-projection-sized tensor.
//!
//! Workloads, each a `get_tensor` of a 2304 x 2560 byte tensor (one packed FP4
//! routed-expert `w1.weight`, 5,898,240 bytes):
//! - `synthetic_*`: an in-memory source that copies the requested range, so a
//!   miss costs lookup, the copy, `Arc` construction and LRU eviction.
//! - `local_*`: route-trace `weights/` through `V41LocalWeightsSource`, so a
//!   miss adds the file read and the receipt SHA-256. Skipped (no samples)
//!   when `.agents/receipts/route-trace` is absent.
//!
//! A hit is a resident tensor. A miss alternates two tensors under a one-tensor
//! budget, so every timed call reads and evicts.
//!
//! Startup embedding rows, one prefill step of 3 distinct token rows of a
//! 5120-wide BF16 `embed.weight` read through `V41CachedEmbeddingRows`:
//! - `embedding_rows_cold` / `_warm`: a synthetic 1024-row table in memory;
//!   cold builds a fresh cache per call, warm re-reads resident rows.
//! - `local_embedding_rows_cold`: the real 129280-row table through a fresh
//!   `V41LocalWeightsSource`, whose first read of the tensor hashes the whole
//!   1.3 GB file once.
//! - `local_embedding_rows_verified`: the same rows uncached in a fresh cache,
//!   through one long-lived source that verified the tensor during setup, so
//!   only the rows are read (a server's decode case).
//!
//! Both `local_embedding_rows_*` run only with `METALLIX_BENCH_LOCAL=1`.
//! - `startup_selected_rows`: the startup lookup itself
//!   (`startup_selected_bf16_reference`, 3 tokens x 4 HC copies) over rows
//!   already read; layer 0's attention and FFN are not included.

use std::{
    hint::black_box,
    path::{Path, PathBuf},
    sync::Mutex,
};

use deepseek::{
    StartupLayout,
    checkpoint::{
        V41SafetensorsHeader,
        embedding_rows::V41CachedEmbeddingRows,
        range_cache::{
            V41LocalWeightsSource, V41RangeCache, V41RangeCacheError, V41RangeRequest,
            V41RangeSource,
        },
    },
    manifest::V41SafetensorsIndex,
    reduced::EmbeddingRowSource,
    startup_selected_bf16_reference,
};

const TENSOR_BYTES: u64 = 2_304 * 2_560;
const SHARD: &str = "model-00001-of-00001.safetensors";
const PINNED: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const REAL_A: &str = "layers.0.ffn.experts.2.w1.weight";
const REAL_B: &str = "layers.0.ffn.experts.4.w1.weight";

/// Serves any absolute range from one in-memory shard image.
struct Memory {
    shard: Vec<u8>,
}

impl V41RangeSource for Memory {
    fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        let start = usize::try_from(request.range.start).expect("in-memory range");
        let end = usize::try_from(request.range.end).expect("in-memory range");
        Ok(self.shard[start..end].to_vec())
    }
}

fn synthetic_cache(budget: u64) -> V41RangeCache<Memory> {
    let header = format!(
        r#"{{"a":{{"dtype":"U8","shape":[{TENSOR_BYTES}],"data_offsets":[0,{TENSOR_BYTES}]}},"b":{{"dtype":"U8","shape":[{TENSOR_BYTES}],"data_offsets":[{TENSOR_BYTES},{}]}}}}"#,
        2 * TENSOR_BYTES
    );
    let file_bytes = 8 + header.len() as u64 + 2 * TENSOR_BYTES;
    let header =
        V41SafetensorsHeader::parse(header.as_bytes(), file_bytes).expect("synthetic header");
    let index = V41SafetensorsIndex::parse(&format!(
        r#"{{"metadata":{{"total_size":1}},"weight_map":{{"a":"{SHARD}","b":"{SHARD}"}}}}"#
    ))
    .expect("synthetic index");
    let shard = (0..file_bytes)
        .map(|offset| offset.wrapping_mul(2_654_435_761).to_le_bytes()[3])
        .collect();
    V41RangeCache::new(
        Memory { shard },
        index,
        [(SHARD.to_owned(), header)],
        budget,
    )
    .expect("synthetic cache")
}

fn receipts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.agents/receipts")
}

fn local_cache(budget: u64) -> Option<V41RangeCache<V41LocalWeightsSource>> {
    let root = receipts();
    let trace = root.join("route-trace");
    let index =
        root.join("control/receipts/candidate-control/real-expert/model.safetensors.index.json");
    if ![REAL_A, REAL_B]
        .iter()
        .all(|tensor| trace.join(format!("weights/{tensor}.bin")).exists())
        || !index.exists()
    {
        eprintln!("range_cache: local route-trace weights absent; local_* benches skipped");
        return None;
    }
    Some(
        V41RangeCache::load(
            V41LocalWeightsSource::new(trace.join("weights"), PINNED),
            &index,
            &trace,
            PINNED,
            budget,
        )
        .expect("pinned index and headers"),
    )
}

fn bench_hit<S: V41RangeSource>(bencher: divan::Bencher, mut cache: V41RangeCache<S>, a: &str) {
    cache.get_tensor(a).expect("warm");
    bencher.bench_local(|| black_box(cache.get_tensor(black_box(a)).expect("hit")));
}

fn bench_miss<S: V41RangeSource>(
    bencher: divan::Bencher,
    mut cache: V41RangeCache<S>,
    a: &str,
    b: &str,
) {
    let mut flip = false;
    bencher
        .counter(divan::counter::BytesCount::new(TENSOR_BYTES))
        .bench_local(|| {
            flip = !flip;
            black_box(
                cache
                    .get_tensor(black_box(if flip { a } else { b }))
                    .expect("miss"),
            )
        });
}

#[divan::bench]
fn synthetic_hit(bencher: divan::Bencher) {
    bench_hit(bencher, synthetic_cache(2 * TENSOR_BYTES), "a");
}

#[divan::bench]
fn synthetic_miss(bencher: divan::Bencher) {
    bench_miss(bencher, synthetic_cache(TENSOR_BYTES), "a", "b");
}

#[divan::bench]
fn local_hit(bencher: divan::Bencher) {
    if let Some(cache) = local_cache(2 * TENSOR_BYTES) {
        bench_hit(bencher, cache, REAL_A);
    }
}

#[divan::bench(sample_count = 20)]
fn local_miss(bencher: divan::Bencher) {
    if let Some(cache) = local_cache(TENSOR_BYTES) {
        bench_miss(bencher, cache, REAL_A, REAL_B);
    }
}

const WIDTH: usize = 5120;
const SYNTHETIC_ROWS: usize = 1024;
/// One prefill step's distinct token rows (ascending), as startup requests them.
const STEP_ROWS: [usize; 3] = [7, 42, 913];

fn embedding_cache() -> V41RangeCache<Memory> {
    let bytes = SYNTHETIC_ROWS * WIDTH * 2;
    let header = format!(
        r#"{{"embed.weight":{{"dtype":"BF16","shape":[{SYNTHETIC_ROWS},{WIDTH}],"data_offsets":[0,{bytes}]}}}}"#
    );
    let file_bytes = 8 + header.len() as u64 + bytes as u64;
    let header =
        V41SafetensorsHeader::parse(header.as_bytes(), file_bytes).expect("synthetic header");
    let index = V41SafetensorsIndex::parse(&format!(
        r#"{{"metadata":{{"total_size":1}},"weight_map":{{"embed.weight":"{SHARD}"}}}}"#
    ))
    .expect("synthetic index");
    // Finite BF16 values around 1.0.
    let shard = (0..file_bytes)
        .map(|offset| {
            if offset % 2 == 1 {
                0x3f
            } else {
                offset.to_le_bytes()[0]
            }
        })
        .collect();
    V41RangeCache::new(
        Memory { shard },
        index,
        [(SHARD.to_owned(), header)],
        1 << 26,
    )
    .expect("synthetic cache")
}

fn read_step<S: V41RangeSource>(cache: &Mutex<V41RangeCache<S>>) -> Vec<u16> {
    let mut rows = vec![0; STEP_ROWS.len() * WIDTH];
    V41CachedEmbeddingRows::new(cache, WIDTH)
        .read_rows(&STEP_ROWS, &mut rows)
        .expect("embedding rows");
    rows
}

#[divan::bench]
fn embedding_rows_cold(bencher: divan::Bencher) {
    bencher
        .with_inputs(|| Mutex::new(embedding_cache()))
        .bench_local_values(|cache| black_box(read_step(&cache)));
}

#[divan::bench]
fn embedding_rows_warm(bencher: divan::Bencher) {
    let cache = Mutex::new(embedding_cache());
    read_step(&cache);
    bencher.bench_local(|| black_box(read_step(&cache)));
}

/// A cache over the pinned real `embed.weight` through `source`, or `None`
/// (bench skipped) when the local data is absent.
fn local_embedding_cache(
    source: V41LocalWeightsSource,
) -> Option<Mutex<V41RangeCache<V41LocalWeightsSource>>> {
    let root = receipts();
    let trace = root.join("route-trace");
    let index =
        root.join("control/receipts/candidate-control/real-expert/model.safetensors.index.json");
    // Each run hashes the 1.3 GB table; keep it out of `cargo test --benches`.
    if std::env::var_os("METALLIX_BENCH_LOCAL").is_none() {
        eprintln!("range_cache: set METALLIX_BENCH_LOCAL=1 to run local_embedding_rows_*");
        return None;
    }
    if !trace.join("weights/embed.weight.bin").exists() || !index.exists() {
        eprintln!("range_cache: local embed.weight absent; local_embedding_rows_* skipped");
        return None;
    }
    Some(Mutex::new(
        V41RangeCache::load(source, &index, &trace, PINNED, 1 << 26)
            .expect("pinned index and headers"),
    ))
}

fn local_source() -> V41LocalWeightsSource {
    V41LocalWeightsSource::new(receipts().join("route-trace/weights"), PINNED)
}

#[divan::bench(sample_count = 3, sample_size = 1)]
fn local_embedding_rows_cold(bencher: divan::Bencher) {
    if local_embedding_cache(local_source()).is_some() {
        bencher
            .with_inputs(|| local_embedding_cache(local_source()).expect("present"))
            .bench_local_values(|cache| black_box(read_step(&cache)));
    }
}

#[divan::bench(sample_count = 20)]
fn local_embedding_rows_verified(bencher: divan::Bencher) {
    let source = local_source();
    if let Some(cache) = local_embedding_cache(source.clone()) {
        read_step(&cache);
        bencher
            .with_inputs(|| local_embedding_cache(source.clone()).expect("present"))
            .bench_local_values(|cache| black_box(read_step(&cache)));
    }
}

#[divan::bench]
fn startup_selected_rows(bencher: divan::Bencher) {
    let rows = read_step(&Mutex::new(embedding_cache()));
    let selected: Vec<u64> = STEP_ROWS.iter().map(|&row| row as u64).collect();
    let ids = [selected[1], selected[0], selected[2]];
    let layout = StartupLayout::new(STEP_ROWS.len(), WIDTH, 4).expect("layout");
    bencher.bench_local(|| {
        black_box(
            startup_selected_bf16_reference(black_box(&ids), &selected, &rows, layout)
                .expect("startup"),
        )
    });
}

fn main() {
    divan::main();
}
