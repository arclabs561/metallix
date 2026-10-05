//! One V4.1 `MoE` token (5120 x 2304, 384 routed experts, top 6, FP8 shared
//! expert) through three routed-expert providers, to measure provider overhead:
//! - `table`: `forward_token` over a construction-time sparse table.
//! - `source_table`: `forward_token_with` over a `RoutedExpertSource` that lends
//!   the same table.
//! - `source_cached_warm`: `forward_token_with` over `V41CachedRoutedExperts`
//!   with every selected expert already resident (lock, six `get_tensor` hits).
//!
//! Workload `real` uses layer 0's gate, bias, shared expert and selected routed
//! experts from `.agents/receipts/route-trace/weights` and token 0 of
//! `capture-parity3/layer00.ffn_in`; otherwise `synthetic` uses deterministic
//! finite codes at the same geometry. Setup checks that all three providers
//! return identical BF16 output. The workload in use is printed to stderr.

use std::{
    hint::black_box,
    path::Path,
    sync::{Arc, LazyLock, Mutex},
};

use deepseek::{
    checkpoint::{
        V41SafetensorsHeader,
        range_cache::{
            V41CachedRoutedExperts, V41LocalWeightsSource, V41RangeCache, V41RangeCacheError,
            V41RangeRequest, V41RangeSource,
        },
    },
    manifest::V41SafetensorsIndex,
    moe::{
        Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEError, MoEReference, RoutedExpertSource,
    },
    routing::flash_bf16_gate_routes,
};

const HIDDEN: usize = 5_120;
const INTERMEDIATE: usize = 2_304;
const EXPERTS: usize = 384;
const LAYER: usize = 0;
const PINNED: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const SHARD: &str = "model-00001-of-00001.safetensors";

fn config() -> MoEConfig {
    MoEConfig::new(HIDDEN, INTERMEDIATE, 10.0, 6, 1.0, true, 1.5).expect("V4.1 config")
}

/// Leaked so experts and the reference can borrow for the process lifetime.
struct Workload {
    name: &'static str,
    token: Vec<u16>,
    moe: MoEReference<'static>,
    table: &'static [Option<Fp4ExpertWeights<'static>>],
    cached: Box<dyn RoutedExpertSource + Send + Sync>,
}

struct TableSource(&'static [Option<Fp4ExpertWeights<'static>>]);

impl RoutedExpertSource for TableSource {
    fn with_expert(
        &self,
        index: usize,
        run: &mut dyn FnMut(Fp4ExpertWeights<'_>) -> Result<Vec<u16>, MoEError>,
    ) -> Result<Vec<u16>, MoEError> {
        match self.0.get(index) {
            Some(Some(expert)) => run(*expert),
            _ => Err(MoEError::ExpertUnavailable {
                index,
                reason: "not in table".to_owned(),
            }),
        }
    }
}

fn leak<T: ?Sized>(value: Box<T>) -> &'static T {
    Box::leak(value)
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

fn mix(seed: u64) -> u64 {
    let mut x = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// BF16 value in `[-scale, scale)`.
fn bf16(seed: u64, scale: f32) -> u16 {
    // [1, 2) from 23 random mantissa bits, then truncated to BF16.
    let mantissa = u32::try_from(mix(seed) >> 41).expect("23 bits");
    let unit = f32::from_bits(0x3f80_0000 | mantissa) - 1.0;
    u16::try_from(((unit * 2.0 - 1.0) * scale).to_bits() >> 16).expect("high half")
}

/// Deterministic payload: E8M0 scales near 1.0, FP8 E4M3 codes with no NaN
/// encoding, arbitrary packed FP4 codes.
fn payload(tensor: &str, start: u64, len: usize) -> Vec<u8> {
    (start..start + len as u64)
        .map(|offset| {
            let byte = (mix(offset) >> 56) as u8;
            if tensor.rsplit('.').next() == Some("scale") {
                120 + byte % 8
            } else if tensor.contains("shared_experts") {
                byte & 0x77
            } else {
                byte
            }
        })
        .collect()
}

/// Synthetic shard holding every layer-0 routed-expert tensor.
struct SyntheticShard;

impl V41RangeSource for SyntheticShard {
    fn read_range(&self, request: &V41RangeRequest<'_>) -> Result<Vec<u8>, V41RangeCacheError> {
        let len = usize::try_from(request.range.end - request.range.start).expect("small");
        Ok(payload(request.tensor, request.range.start, len))
    }
}

fn expert_tensors() -> Vec<(String, &'static str, [usize; 2])> {
    let mut tensors = Vec::new();
    for expert in 0..EXPERTS {
        for (projection, rows, reduction) in [
            ("w1", INTERMEDIATE, HIDDEN),
            ("w2", HIDDEN, INTERMEDIATE),
            ("w3", INTERMEDIATE, HIDDEN),
        ] {
            let prefix = format!("layers.{LAYER}.ffn.experts.{expert}.{projection}");
            tensors.push((format!("{prefix}.weight"), "I8", [rows, reduction / 2]));
            tensors.push((format!("{prefix}.scale"), "F8_E8M0", [rows, reduction / 32]));
        }
    }
    tensors
}

fn synthetic_cache() -> V41RangeCache<SyntheticShard> {
    let tensors = expert_tensors();
    let mut offset = 0usize;
    let mut entries = Vec::with_capacity(tensors.len());
    for (name, dtype, [rows, columns]) in &tensors {
        let end = offset + rows * columns;
        entries.push(format!(
            r#""{name}":{{"dtype":"{dtype}","shape":[{rows},{columns}],"data_offsets":[{offset},{end}]}}"#
        ));
        offset = end;
    }
    let header = format!("{{{}}}", entries.join(","));
    let header =
        V41SafetensorsHeader::parse(header.as_bytes(), 8 + header.len() as u64 + offset as u64)
            .expect("synthetic expert header");
    let map = tensors
        .iter()
        .map(|(name, _, _)| format!(r#""{name}":"{SHARD}""#))
        .collect::<Vec<_>>()
        .join(",");
    let index = V41SafetensorsIndex::parse(&format!(
        r#"{{"metadata":{{"total_size":1}},"weight_map":{{{map}}}}}"#
    ))
    .expect("synthetic index");
    V41RangeCache::new(
        SyntheticShard,
        index,
        [(SHARD.to_owned(), header)],
        256 << 20,
    )
    .expect("synthetic cache")
}

fn experts_from<S: V41RangeSource>(
    cache: &mut V41RangeCache<S>,
    ids: &[usize],
) -> &'static [Option<Fp4ExpertWeights<'static>>] {
    let mut table = vec![None; EXPERTS];
    for &id in ids {
        let bytes: Vec<&'static [u8]> = ["w1", "w2", "w3"]
            .into_iter()
            .flat_map(|projection| ["weight", "scale"].map(|kind| (projection, kind)))
            .map(|(projection, kind)| {
                let name = format!("layers.{LAYER}.ffn.experts.{id}.{projection}.{kind}");
                let arc: Arc<[u8]> = cache.get_tensor(&name).expect("routed expert tensor");
                leak(Box::<[u8]>::from(&*arc))
            })
            .collect();
        table[id] = Some(
            Fp4ExpertWeights::new(
                HIDDEN,
                INTERMEDIATE,
                bytes[0],
                bytes[1],
                bytes[2],
                bytes[3],
                bytes[4],
                bytes[5],
            )
            .expect("expert geometry"),
        );
    }
    leak(table.into_boxed_slice())
}

fn routes(token: &[u16], gate: &[u16], bias: &[f32]) -> Vec<usize> {
    flash_bf16_gate_routes(token, gate, EXPERTS, HIDDEN, bias, 6, 1.0, true, 1.5)
        .expect("routes")
        .iter()
        .map(|route| route.expert_index())
        .collect()
}

fn real() -> Option<Workload> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.agents/receipts");
    let trace = root.join("route-trace");
    let index =
        root.join("control/receipts/candidate-control/real-expert/model.safetensors.index.json");
    let input = trace.join("capture-parity3/layer00.ffn_in.torch.bfloat16.bin");
    if !index.exists() || !input.exists() || !trace.join("headers.json").exists() {
        return None;
    }
    let mut cache = V41RangeCache::load(
        V41LocalWeightsSource::new(trace.join("weights"), PINNED),
        &index,
        &trace,
        PINNED,
        1 << 30,
    )
    .ok()?;
    let mut get = |name: &str| cache.get_tensor(name).ok();
    let gate: &'static [u16] = leak(u16s(&get("layers.0.ffn.gate.weight")?).into_boxed_slice());
    let bias: &'static [f32] = leak(
        get("layers.0.ffn.gate.bias")?
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("four bytes")))
            .collect::<Box<[f32]>>(),
    );
    let shared: Vec<&'static [u8]> = ["w1", "w2", "w3"]
        .into_iter()
        .flat_map(|projection| ["weight", "scale"].map(|kind| (projection, kind)))
        .map(|(projection, kind)| {
            get(&format!("layers.0.ffn.shared_experts.{projection}.{kind}"))
                .map(|arc| leak(Box::<[u8]>::from(&*arc)))
        })
        .collect::<Option<_>>()?;
    let token = u16s(&std::fs::read(input).ok()?)[..HIDDEN].to_vec();
    let ids = routes(&token, gate, bias);
    let names: Vec<String> = ids
        .iter()
        .flat_map(|id| {
            ["w1", "w2", "w3"].into_iter().flat_map(move |projection| {
                ["weight", "scale"]
                    .map(|kind| format!("layers.0.ffn.experts.{id}.{projection}.{kind}"))
            })
        })
        .collect();
    if names.iter().any(|name| cache.get_tensor(name).is_err()) {
        eprintln!("moe_provider: a routed layer-0 expert is not local; using synthetic");
        return None;
    }
    let table = experts_from(&mut cache, &ids);
    let shared = Fp8ExpertWeights::new(
        HIDDEN,
        INTERMEDIATE,
        shared[0],
        shared[1],
        shared[2],
        shared[3],
        shared[4],
        shared[5],
    )
    .expect("shared expert");
    let moe = MoEReference::new_sparse(config(), gate, bias, table, shared).expect("real MoE");
    let cache = leak(Box::new(Mutex::new(cache)));
    Some(Workload {
        name: "real layer 0, capture-parity3 token 0",
        token,
        moe,
        table,
        cached: Box::new(V41CachedRoutedExperts::new(
            cache,
            LAYER,
            HIDDEN,
            INTERMEDIATE,
        )),
    })
}

fn synthetic() -> Workload {
    let gate: &'static [u16] = leak(
        (0..EXPERTS * HIDDEN)
            .map(|i| bf16(i as u64, 0.05))
            .collect::<Box<[u16]>>(),
    );
    let bias: &'static [f32] = leak(vec![0.0; EXPERTS].into_boxed_slice());
    let shared: Vec<&'static [u8]> = [
        ("w1", INTERMEDIATE, HIDDEN),
        ("w2", HIDDEN, INTERMEDIATE),
        ("w3", INTERMEDIATE, HIDDEN),
    ]
    .into_iter()
    .flat_map(|(projection, rows, reduction)| {
        [
            (
                format!("shared_experts.{projection}.weight"),
                rows * reduction,
            ),
            (
                format!("shared_experts.{projection}.scale"),
                rows.div_ceil(32) * (reduction / 32),
            ),
        ]
    })
    .map(|(name, len)| leak(payload(&name, 0, len).into_boxed_slice()))
    .collect();
    let token: Vec<u16> = (0..HIDDEN).map(|i| bf16(1 << 40 | i as u64, 1.0)).collect();
    let mut cache = synthetic_cache();
    let table = experts_from(&mut cache, &routes(&token, gate, bias));
    let shared = Fp8ExpertWeights::new(
        HIDDEN,
        INTERMEDIATE,
        shared[0],
        shared[1],
        shared[2],
        shared[3],
        shared[4],
        shared[5],
    )
    .expect("shared expert");
    let moe = MoEReference::new_sparse(config(), gate, bias, table, shared).expect("synthetic MoE");
    let cache = leak(Box::new(Mutex::new(cache)));
    Workload {
        name: "synthetic 5120x2304, deterministic codes",
        token,
        moe,
        table,
        cached: Box::new(V41CachedRoutedExperts::new(
            cache,
            LAYER,
            HIDDEN,
            INTERMEDIATE,
        )),
    }
}

static WORKLOAD: LazyLock<Workload> = LazyLock::new(|| {
    let workload = real().unwrap_or_else(synthetic);
    let table = workload.moe.forward_token(&workload.token).expect("table");
    let lent = workload
        .moe
        .forward_token_with(&workload.token, &TableSource(workload.table))
        .expect("source table");
    // Also warms the cache with the selected experts.
    let cached = workload
        .moe
        .forward_token_with(&workload.token, workload.cached.as_ref())
        .expect("cached source");
    assert_eq!(table.output_bf16(), lent.output_bf16());
    assert_eq!(table.output_bf16(), cached.output_bf16());
    eprintln!("moe_provider workload: {}", workload.name);
    workload
});

#[divan::bench(sample_count = 15, sample_size = 1)]
fn table() {
    let w = &*WORKLOAD;
    black_box(w.moe.forward_token(black_box(&w.token)).expect("table"));
}

#[divan::bench(sample_count = 15, sample_size = 1)]
fn source_table() {
    let w = &*WORKLOAD;
    black_box(
        w.moe
            .forward_token_with(black_box(&w.token), &TableSource(w.table))
            .expect("source table"),
    );
}

#[divan::bench(sample_count = 15, sample_size = 1)]
fn source_cached_warm() {
    let w = &*WORKLOAD;
    black_box(
        w.moe
            .forward_token_with(black_box(&w.token), w.cached.as_ref())
            .expect("cached source"),
    );
}

fn main() {
    LazyLock::force(&WORKLOAD);
    divan::main();
}
