//! Native ratio-one owner over the alternate 4/1/1/1 source partition.
//!
//! This is a small transaction integration: the source fixture supplies only
//! owner operands and observed prefixes, while `RatioOneCompressedOwner`
//! performs the WKV, compressor, key, KV, prepare, and commit path.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::{
        cache::IndexKeyPublicationId,
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{RatioOneCompressedOwner, RatioOneOwnerCall, RatioOneOwnerWeights},
    },
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

const SOURCE_RECEIPT_SHA256: &str =
    "9613150fea8010a7435dab0443a1f9e0d73fd8d0f32455b8d67dd572617f3906";
const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const MODEL_SHA256: &str = "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65";
const SCHEDULE: &[(usize, usize)] = &[(0, 4), (4, 1), (5, 1), (6, 1)];

#[derive(Deserialize)]
struct Fixture {
    schema_version: u8,
    source_receipt_sha256: String,
    source: Value,
    capture_identity: Value,
    model: Value,
    weights: Weights,
    frequencies: Tensor,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Weights {
    wkv: Tensor,
    compressor_norm: Tensor,
    wk: Tensor,
    key_norm: Tensor,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    token_count: usize,
    input: Tensor,
    projected: Tensor,
    latent: Tensor,
    index_key_prefix: Tensor,
    compressed_kv_prefix: Tensor,
    next_layer1_score_prefix: Option<Tensor>,
}

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    finite: bool,
    shape: Vec<usize>,
    numel: usize,
    storage_hex: String,
    storage_sha256: String,
}

impl Tensor {
    fn bytes(&self, width: usize) -> Vec<u8> {
        assert!(self.finite, "source tensor must be finite");
        let elements = self.shape.iter().copied().product::<usize>();
        assert_eq!(self.numel, elements, "source tensor element count");
        assert_eq!(self.storage_hex.len(), elements * width * 2, "source bytes");
        let bytes: Vec<_> = self
            .storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("fixture UTF-8"), 16)
                    .expect("fixture hex")
            })
            .collect();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), self.storage_sha256);
        bytes
    }

    fn bf16(&self) -> Vec<u16> {
        assert_eq!(self.dtype, "torch.bfloat16");
        self.bytes(2)
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
            .collect()
    }

    fn frequencies(&self) -> Vec<RotaryFrequency> {
        assert_eq!(self.dtype, "torch.complex64");
        self.bytes(8)
            .chunks_exact(8)
            .map(|pair| {
                RotaryFrequency::new(
                    f32::from_le_bytes(pair[..4].try_into().expect("complex real")),
                    f32::from_le_bytes(pair[4..].try_into().expect("complex imaginary")),
                )
                .expect("finite source frequency")
            })
            .collect()
    }
}

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("captured nonzero dimension")
}

fn object<'a>(value: &'a Value, label: &str) -> &'a serde_json::Map<String, Value> {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{label} must be an object"))
}

fn usize_field(value: &Value, field: &str) -> usize {
    usize::try_from(
        object(value, "source model")[field]
            .as_u64()
            .unwrap_or_else(|| panic!("source model {field}")),
    )
    .expect("source model usize")
}

fn fixture() -> Fixture {
    let raw = include_str!("../../../../fixtures/deepseek-v41/partition-owner-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "3a788074b5de4597104a673851cddc616a977b47e59fd35b5f383d33372b0a8f"
    );
    let fixture: Fixture = serde_json::from_str(raw).expect("partition owner fixture JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source_receipt_sha256, SOURCE_RECEIPT_SHA256);
    let source = object(&fixture.source, "source");
    assert_eq!(source["revision"].as_str(), Some(REVISION));
    assert_eq!(source["model_sha256"].as_str(), Some(MODEL_SHA256));
    assert_eq!(
        object(&fixture.capture_identity, "capture identity")["schedule"].as_array(),
        Some(&vec![
            Value::from(4),
            Value::from(1),
            Value::from(1),
            Value::from(1)
        ]),
        "source capture identifies the alternate partition"
    );
    assert_eq!(usize_field(&fixture.model, "batches"), 1);
    assert_eq!(usize_field(&fixture.model, "input_dimension"), 128);
    assert_eq!(usize_field(&fixture.model, "latent_dimension"), 64);
    assert_eq!(usize_field(&fixture.model, "key_dimension"), 64);
    assert_eq!(usize_field(&fixture.model, "rope_pairs"), 16);
    assert_eq!(usize_field(&fixture.model, "cache_capacity"), 8);
    assert_eq!(usize_field(&fixture.model, "owner_layer"), 3);
    let epsilon = object(&fixture.model, "source model")["norm_epsilon"]
        .as_f64()
        .expect("source epsilon");
    assert_eq!(epsilon.to_bits(), 1.0e-20_f64.to_bits());
    assert_eq!(fixture.weights.wkv.shape, [64, 128]);
    assert_eq!(fixture.weights.compressor_norm.shape, [64]);
    assert_eq!(fixture.weights.wk.shape, [64, 64]);
    assert_eq!(fixture.weights.key_norm.shape, [64]);
    assert_eq!(fixture.frequencies.shape, [8, 16]);
    assert_eq!(fixture.cases.len(), SCHEDULE.len());
    for (case, &(start, count)) in fixture.cases.iter().zip(SCHEDULE) {
        assert_eq!((case.start_pos, case.token_count), (start, count));
        assert_eq!(case.input.shape, [1, count, 128]);
        assert_eq!(case.projected.shape, [1, count, 64]);
        assert_eq!(case.latent.shape, [1, count, 64]);
        assert_eq!(case.index_key_prefix.shape, [1, start + count, 64]);
        assert_eq!(case.compressed_kv_prefix.shape, [1, start + count, 64]);
        assert_eq!(
            case.next_layer1_score_prefix.is_some(),
            matches!(start, 0 | 5),
            "both partial handoffs are mandatory"
        );
        if let Some(next) = &case.next_layer1_score_prefix {
            let end = start.checked_add(count).expect("source token endpoint");
            assert_eq!(next.shape, [1, end / 2, 64]);
        }
    }
    fixture
}

fn owner(fixture: &Fixture) -> RatioOneCompressedOwner {
    let layout =
        IndexKeyLayout::new(nz(1), nz(64), nz(64), nz(16), 1.0e-20).expect("captured owner layout");
    RatioOneCompressedOwner::new(
        layout,
        nz(128),
        nz(8),
        3,
        &fixture.weights.compressor_norm.bf16(),
        1.0e-20,
    )
    .expect("captured ratio-one owner")
}

fn commit_case(owner: &mut RatioOneCompressedOwner, fixture: &Fixture, index: usize) {
    let case = &fixture.cases[index];
    let frequencies = fixture.frequencies.frequencies();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    let input = case.input.bf16();
    let start = case.start_pos;
    let end = start + case.token_count;
    let diagnostic = owner
        .prepare(RatioOneOwnerCall::new(
            IndexKeyPublicationId::new(3, owner.epoch(), owner.next_call_id()),
            start,
            nz(case.token_count),
            &input,
            &frequencies[start * 16..end * 16],
            RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
        ))
        .expect("source-shaped owner preparation")
        .commit()
        .expect("source-shaped owner commit");
    assert_eq!(
        diagnostic.owner.projected,
        case.projected.bf16(),
        "native WKV"
    );
    assert_eq!(
        diagnostic.owner.latent,
        case.latent.bf16(),
        "native compressor"
    );
    assert_eq!(
        owner.key_prefix(0).expect("native index prefix"),
        case.index_key_prefix.bf16(),
        "native index keys"
    );
    assert_eq!(
        owner.kv_prefix(0).expect("native compressed-KV prefix"),
        case.compressed_kv_prefix.bf16(),
        "native compressed KV"
    );
    if let Some(next_layer_one) = &case.next_layer1_score_prefix {
        let prefix = owner.key_prefix(0).expect("published key prefix");
        assert_eq!(
            &prefix[..next_layer_one.numel],
            next_layer_one.bf16(),
            "native committed L3 keys feed the next L1 partial score prefix"
        );
    }
}

fn state(owner: &RatioOneCompressedOwner) -> (u64, u64, usize, usize, Vec<u16>, Vec<u16>) {
    (
        owner.epoch(),
        owner.next_call_id(),
        owner.next_position(),
        owner.valid_positions(),
        owner.key_prefix(0).expect("key prefix").to_vec(),
        owner.kv_prefix(0).expect("KV prefix").to_vec(),
    )
}

#[test]
fn alternate_partition_owner_matches_source_prefixes_and_partial_bridges() {
    let fixture = fixture();
    let mut owner = owner(&fixture);
    for index in 0..fixture.cases.len() {
        commit_case(&mut owner, &fixture, index);
    }
}

#[test]
fn rejected_calls_leave_the_owner_retryable() {
    let fixture = fixture();
    let mut owner = owner(&fixture);
    let before = state(&owner);
    let case = &fixture.cases[1];
    let frequencies = fixture.frequencies.frequencies();
    let input = case.input.bf16();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    assert!(
        owner
            .prepare(RatioOneOwnerCall::new(
                IndexKeyPublicationId::new(3, 0, 0),
                case.start_pos,
                nz(case.token_count),
                &input,
                &frequencies[case.start_pos * 16..(case.start_pos + case.token_count) * 16],
                RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
            ))
            .is_err()
    );
    assert_eq!(state(&owner), before, "out-of-order call is invisible");

    commit_case(&mut owner, &fixture, 0);
    commit_case(&mut owner, &fixture, 1);
    commit_case(&mut owner, &fixture, 2);
    let before_late = state(&owner);
    let late = &fixture.cases[3];
    let mut malformed = late.input.bf16();
    malformed.pop();
    let frequencies = fixture.frequencies.frequencies();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    assert!(
        owner
            .prepare(RatioOneOwnerCall::new(
                IndexKeyPublicationId::new(3, owner.epoch(), owner.next_call_id()),
                late.start_pos,
                nz(late.token_count),
                &malformed,
                &frequencies[late.start_pos * 16..(late.start_pos + late.token_count) * 16],
                RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
            ))
            .is_err()
    );
    assert_eq!(
        state(&owner),
        before_late,
        "malformed late call is invisible"
    );
    commit_case(&mut owner, &fixture, 3);
}

#[test]
fn reset_requires_fresh_publications_and_replays_the_source_partition() {
    let fixture = fixture();
    let mut owner = owner(&fixture);
    for index in 0..fixture.cases.len() {
        commit_case(&mut owner, &fixture, index);
    }
    let previous_epoch = owner.epoch();
    owner.reset().expect("reset coupled owner state");
    assert_eq!(owner.epoch(), previous_epoch + 1);
    assert_eq!(owner.next_call_id(), 0);
    assert_eq!(owner.next_position(), 0);
    assert_eq!(owner.valid_positions(), 0);
    assert_eq!(state(&owner).4, Vec::<u16>::new(), "reset key prefix");
    assert_eq!(state(&owner).5, Vec::<u16>::new(), "reset KV prefix");

    let first = &fixture.cases[0];
    let input = first.input.bf16();
    let frequencies = fixture.frequencies.frequencies();
    let wkv = fixture.weights.wkv.bf16();
    let wk = fixture.weights.wk.bf16();
    let key_norm = fixture.weights.key_norm.bf16();
    let after_reset = state(&owner);
    assert!(
        owner
            .prepare(RatioOneOwnerCall::new(
                IndexKeyPublicationId::new(3, 0, 0),
                0,
                nz(first.token_count),
                &input,
                &frequencies[..first.token_count * 16],
                RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm)),
            ))
            .is_err()
    );
    assert_eq!(state(&owner), after_reset, "stale publication is invisible");
    for index in 0..fixture.cases.len() {
        commit_case(&mut owner, &fixture, index);
    }
}
