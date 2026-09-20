//! Release microbenchmark for source-captured ratio-one owner publications.
//!
//! Setup, resets, and the preceding publication are deliberately outside the
//! timed phase. Run this executable at least three times; it prints per-phase
//! median and sample standard deviation from its bounded internal iterations.

use std::{
    hint::black_box,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use deepseek::{
    RotaryFrequency,
    indexer::{
        cache::IndexKeyPublicationId,
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{RatioOneCompressedOwner, RatioOneOwnerCall, RatioOneOwnerWeights},
    },
};
use serde_json::Value;

const ITERATIONS: usize = 200;
const OWNER_LAYER: u16 = 3;
const BATCHES: usize = 1;
const INPUT_DIMENSION: usize = 128;
const LATENT_DIMENSION: usize = 64;
const KEY_DIMENSION: usize = 64;
const CACHE_CAPACITY: usize = 8;
const ROPE_PAIRS: usize = 16;
const SOURCE_REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const SOURCE_MODEL_SHA256: &str =
    "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65";
const OWNER_CAPTURE_SHA256: &str =
    "e27dde6ead409c74f7bb2c9e08d4cd5a2b0cfc3c9505c7d6b8908b1cd78b1cc6";

struct Fixture {
    wkv: Vec<u16>,
    compressor_norm: Vec<u16>,
    wk: Vec<u16>,
    key_norm: Vec<u16>,
    prefill_input: Vec<u16>,
    prefill_frequencies: Vec<RotaryFrequency>,
    decode_five_input: Vec<u16>,
    decode_five_frequencies: Vec<RotaryFrequency>,
    decode_six_input: Vec<u16>,
    decode_six_frequencies: Vec<RotaryFrequency>,
    expected_prefill_latent: Vec<u16>,
    expected_prefill_keys: Vec<u16>,
    expected_prefill_kv: Vec<u16>,
}

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("fixed benchmark geometry is nonzero")
}

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    value
        .get(name)
        .unwrap_or_else(|| panic!("fixture lacks {name}"))
}

fn bf16(value: &Value) -> Vec<u16> {
    let hex = field(value, "storage_hex")
        .as_str()
        .expect("fixture BF16 storage is hex");
    assert!(
        hex.len().is_multiple_of(4),
        "BF16 storage has complete hex words"
    );
    let bytes: Vec<_> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("hex UTF-8"), 16).expect("hex byte")
        })
        .collect();
    assert_eq!(bytes.len() % 2, 0, "BF16 storage has complete words");
    bytes
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
        .collect()
}

fn assert_source(value: &Value, capture: Option<&str>) {
    let source = field(value, "source");
    assert_eq!(field(source, "revision").as_str(), Some(SOURCE_REVISION));
    assert_eq!(
        field(source, "model_sha256").as_str(),
        Some(SOURCE_MODEL_SHA256)
    );
    assert_eq!(field(source, "storage_byteorder").as_str(), Some("little"));
    if let Some(capture) = capture {
        assert_eq!(
            field(source, "complete_capture_sha256").as_str(),
            Some(capture)
        );
    }
}

fn frequencies(value: &Value, start: usize, positions: usize) -> Vec<RotaryFrequency> {
    let pairs = field(field(value, "frequencies"), "fp32_pairs")
        .as_array()
        .expect("fixture frequency pairs");
    pairs[start * ROPE_PAIRS..(start + positions) * ROPE_PAIRS]
        .iter()
        .map(|pair| {
            let pair = pair.as_array().expect("frequency pair");
            RotaryFrequency::new(
                f32::from_bits(u32::try_from(pair[0].as_u64().expect("real bits")).expect("u32")),
                f32::from_bits(
                    u32::try_from(pair[1].as_u64().expect("imaginary bits")).expect("u32"),
                ),
            )
            .expect("finite source frequency")
        })
        .collect()
}

fn case(cases: &[Value], start: usize) -> &Value {
    cases
        .iter()
        .find(|candidate| field(candidate, "start_pos").as_u64() == Some(start as u64))
        .unwrap_or_else(|| panic!("fixture lacks call at {start}"))
}

fn fixture() -> Fixture {
    let compressor: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-compressor-reference.json"
    ))
    .expect("compressor fixture JSON");
    let owner: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
    ))
    .expect("owner fixture JSON");
    let attention: Value = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-attention-reference.json"
    ))
    .expect("attention fixture JSON");
    assert_source(&compressor, None);
    assert_source(&owner, Some(OWNER_CAPTURE_SHA256));
    assert_source(&attention, Some(OWNER_CAPTURE_SHA256));
    assert_eq!(
        field(&owner, "model")["cache_capacity"].as_u64(),
        Some(CACHE_CAPACITY as u64)
    );
    assert_eq!(
        field(&owner, "model")["key_dimension"].as_u64(),
        Some(KEY_DIMENSION as u64)
    );
    assert_eq!(
        field(&compressor, "model")["input_dimension"].as_u64(),
        Some(INPUT_DIMENSION as u64)
    );
    assert_eq!(
        field(&compressor, "model")["latent_dimension"].as_u64(),
        Some(LATENT_DIMENSION as u64)
    );
    let compressor_cases = field(&compressor, "cases").as_array().expect("cases");
    let owner_cases = field(&owner, "cases").as_array().expect("owner cases");
    let attention_cases = field(&attention, "cases")
        .as_array()
        .expect("attention cases");
    let weights = field(&compressor, "weights");
    let owner_weights = field(&owner, "weights");
    let input = |start| bf16(field(case(compressor_cases, start), "attention_input"));
    Fixture {
        wkv: bf16(field(weights, "wkv")),
        compressor_norm: bf16(field(weights, "norm")),
        wk: bf16(field(owner_weights, "wk")),
        key_norm: bf16(field(owner_weights, "norm")),
        prefill_input: input(0),
        prefill_frequencies: frequencies(&attention, 0, 5),
        decode_five_input: input(5),
        decode_five_frequencies: frequencies(&attention, 5, 1),
        decode_six_input: input(6),
        decode_six_frequencies: frequencies(&attention, 6, 1),
        expected_prefill_latent: bf16(field(case(owner_cases, 0), "latent")),
        expected_prefill_keys: bf16(field(case(owner_cases, 0), "index_cache_after")),
        expected_prefill_kv: bf16(field(case(attention_cases, 0), "compressed_kv")),
    }
}

fn owner(fixture: &Fixture) -> RatioOneCompressedOwner {
    RatioOneCompressedOwner::new(
        IndexKeyLayout::new(
            nz(BATCHES),
            nz(LATENT_DIMENSION),
            nz(KEY_DIMENSION),
            nz(ROPE_PAIRS),
            1.0e-20,
        )
        .expect("fixture key layout"),
        nz(INPUT_DIMENSION),
        nz(CACHE_CAPACITY),
        OWNER_LAYER,
        &fixture.compressor_norm,
        1.0e-20,
    )
    .expect("fixture owner")
}

fn call<'a>(
    fixture: &'a Fixture,
    epoch: u64,
    call_id: u64,
    start: usize,
    input: &'a [u16],
    frequencies: &'a [RotaryFrequency],
) -> RatioOneOwnerCall<'a> {
    RatioOneOwnerCall::new(
        IndexKeyPublicationId::new(OWNER_LAYER, epoch, call_id),
        start,
        nz(input.len() / INPUT_DIMENSION),
        input,
        frequencies,
        RatioOneOwnerWeights::new(
            &fixture.wkv,
            IndexKeyWeights::new(&fixture.wk, &fixture.key_norm),
        ),
    )
}

fn prime(owner: &mut RatioOneCompressedOwner, fixture: &Fixture) {
    let diagnostic = owner
        .forward(call(
            fixture,
            owner.epoch(),
            0,
            0,
            &fixture.prefill_input,
            &fixture.prefill_frequencies,
        ))
        .expect("captured prefill publishes");
    assert_eq!(diagnostic.owner.latent.len(), 5 * LATENT_DIMENSION);
    assert_eq!(diagnostic.owner.latent, fixture.expected_prefill_latent);
    assert_eq!(
        owner.key_prefix(0).expect("prefill key prefix"),
        fixture.expected_prefill_keys
    );
    assert_eq!(
        owner.kv_prefix(0).expect("prefill KV prefix"),
        fixture.expected_prefill_kv
    );
    assert_eq!(owner.valid_positions(), 5);
}

fn reset_and_prime(owner: &mut RatioOneCompressedOwner, fixture: &Fixture) {
    owner.reset().expect("reset succeeds");
    prime(owner, fixture);
}

fn duration_ns(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000_000.0
}

fn report(label: &str, values: &mut [f64]) {
    values.sort_by(f64::total_cmp);
    let count = u32::try_from(values.len()).expect("bounded iteration count fits u32");
    assert!(count > 1, "sample standard deviation needs two values");
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        values[middle - 1].midpoint(values[middle])
    } else {
        values[middle]
    };
    let mean = values.iter().sum::<f64>() / f64::from(count);
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / f64::from(count - 1);
    println!(
        "{label}: iterations={} median_ns={median:.1} sample_stdev_ns={:.1}",
        values.len(),
        variance.sqrt()
    );
}

fn prepare_and_drop(
    label: &str,
    fixture: &Fixture,
    start: usize,
    call_id: u64,
    input: &[u16],
    frequencies: &[RotaryFrequency],
) {
    let mut owner = owner(fixture);
    prime(&mut owner, fixture);
    if start == 6 {
        let diagnostic = owner
            .forward(call(
                fixture,
                owner.epoch(),
                1,
                5,
                &fixture.decode_five_input,
                &fixture.decode_five_frequencies,
            ))
            .expect("captured decode at prefix five publishes");
        black_box(diagnostic);
        assert_eq!(owner.valid_positions(), 6);
    }
    let mut prepare = Vec::with_capacity(ITERATIONS);
    let mut drop_cost = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let pending = owner
            .prepare(call(
                fixture,
                owner.epoch(),
                call_id,
                start,
                input,
                frequencies,
            ))
            .expect("captured transaction prepares");
        prepare.push(duration_ns(started.elapsed()));
        assert_eq!(
            pending.key_prefix(0).expect("staged key prefix").len(),
            (start + 1) * KEY_DIMENSION
        );
        assert_eq!(
            pending.kv_prefix(0).expect("staged KV prefix").len(),
            (start + 1) * LATENT_DIMENSION
        );
        black_box(pending.diagnostic());
        let started = Instant::now();
        drop(pending);
        drop_cost.push(duration_ns(started.elapsed()));
        assert_eq!(owner.valid_positions(), start);
    }
    report(&format!("{label}.prepare"), &mut prepare);
    report(&format!("{label}.drop"), &mut drop_cost);
}

fn prepare_prefill_and_drop(fixture: &Fixture) {
    let mut owner = owner(fixture);
    let mut prepare = Vec::with_capacity(ITERATIONS);
    let mut drop_cost = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let pending = owner
            .prepare(call(
                fixture,
                owner.epoch(),
                0,
                0,
                &fixture.prefill_input,
                &fixture.prefill_frequencies,
            ))
            .expect("captured prefill transaction prepares");
        prepare.push(duration_ns(started.elapsed()));
        assert_eq!(
            pending.key_prefix(0).expect("staged key prefix").len(),
            5 * KEY_DIMENSION
        );
        assert_eq!(
            pending.kv_prefix(0).expect("staged KV prefix").len(),
            5 * LATENT_DIMENSION
        );
        black_box(pending.diagnostic());
        let started = Instant::now();
        drop(pending);
        drop_cost.push(duration_ns(started.elapsed()));
        assert_eq!(owner.valid_positions(), 0);
    }
    report("prefill_5.prepare", &mut prepare);
    report("prefill_5.drop", &mut drop_cost);
}

fn commit_and_forward(label: &str, fixture: &Fixture) {
    let mut owner = owner(fixture);
    prime(&mut owner, fixture);
    let mut commit = Vec::with_capacity(ITERATIONS);
    let mut forward = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let pending = owner
            .prepare(call(
                fixture,
                owner.epoch(),
                1,
                5,
                &fixture.decode_five_input,
                &fixture.decode_five_frequencies,
            ))
            .expect("captured transaction prepares outside commit timing");
        let started = Instant::now();
        let diagnostic = pending.commit().expect("captured transaction commits");
        commit.push(duration_ns(started.elapsed()));
        black_box(diagnostic);
        assert_eq!(owner.valid_positions(), 6);
        reset_and_prime(&mut owner, fixture);

        let started = Instant::now();
        let diagnostic = owner
            .forward(call(
                fixture,
                owner.epoch(),
                1,
                5,
                &fixture.decode_five_input,
                &fixture.decode_five_frequencies,
            ))
            .expect("forward publishes captured decode");
        forward.push(duration_ns(started.elapsed()));
        black_box(diagnostic);
        assert_eq!(owner.valid_positions(), 6);
        reset_and_prime(&mut owner, fixture);
    }
    report(&format!("{label}.commit"), &mut commit);
    report(&format!("{label}.forward"), &mut forward);
}

fn main() {
    let total_started = Instant::now();
    let fixture = fixture();
    println!(
        "owner_transaction fixture_prefixes=5,6 geometry=batch={BATCHES},input={INPUT_DIMENSION},latent={LATENT_DIMENSION},key={KEY_DIMENSION},capacity={CACHE_CAPACITY}"
    );
    prepare_prefill_and_drop(&fixture);
    prepare_and_drop(
        "decode_at_prefix_5",
        &fixture,
        5,
        1,
        &fixture.decode_five_input,
        &fixture.decode_five_frequencies,
    );
    prepare_and_drop(
        "decode_at_prefix_6",
        &fixture,
        6,
        2,
        &fixture.decode_six_input,
        &fixture.decode_six_frequencies,
    );
    commit_and_forward("decode_at_prefix_5", &fixture);
    println!(
        "owner_transaction total_wall_ms={:.3}",
        total_started.elapsed().as_secs_f64() * 1_000.0
    );
}
