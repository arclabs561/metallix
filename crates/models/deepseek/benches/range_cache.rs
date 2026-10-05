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

use std::{
    hint::black_box,
    path::{Path, PathBuf},
};

use deepseek::{
    checkpoint::{
        V41SafetensorsHeader,
        range_cache::{
            V41LocalWeightsSource, V41RangeCache, V41RangeCacheError, V41RangeRequest,
            V41RangeSource,
        },
    },
    manifest::V41SafetensorsIndex,
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

fn main() {
    divan::main();
}
