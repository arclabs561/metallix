//! Same-trace layer-three native key publication consumed by partial layer one.

#[path = "support/layer1_owner_capture.rs"]
#[allow(
    dead_code,
    reason = "the bridge test uses only the supplied-prefix score helper from this shared fixture oracle"
)]
mod layer1_owner_capture;

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    indexer::{
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::{RatioOneIndexKeyOwner, RatioOneOwnerCall, RatioOneOwnerWeights},
    },
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const CAPTURE: &str = "887dbe4b2bc4330f0c3fef3efab57b7e0878827c76108a3b2422f7806c710dee";

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    producer: Producer,
    consumer: Consumer,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    complete_capture_sha256: String,
}

#[derive(Deserialize)]
struct Producer {
    keys: KeyFixture,
    compressor: CompressorFixture,
}

#[derive(Deserialize)]
struct KeyFixture {
    source: Source,
    model: KeyModel,
    weights: KeyWeights,
    frequencies: Tensor,
    cases: Vec<KeyCase>,
}

#[derive(Deserialize)]
struct KeyModel {
    batches: usize,
    latent_dimension: usize,
    key_dimension: usize,
    rope_pairs: usize,
    norm_epsilon: f32,
    owner_layer: usize,
    cache_capacity: usize,
}

#[derive(Deserialize)]
struct KeyWeights {
    wk: Tensor,
    norm: Tensor,
}

#[derive(Deserialize)]
struct KeyCase {
    start_pos: usize,
    latent: Tensor,
    index_cache_after: Tensor,
}

#[derive(Deserialize)]
struct CompressorFixture {
    source: Source,
    model: CompressorModel,
    weights: CompressorWeights,
    cases: Vec<CompressorCase>,
}

#[derive(Deserialize)]
struct CompressorModel {
    input_dimension: usize,
    latent_dimension: usize,
    norm_epsilon: f32,
    owner_layer: usize,
    compression_ratio: usize,
}

#[derive(Deserialize)]
struct CompressorWeights {
    wkv: Tensor,
    norm: Tensor,
}

#[derive(Deserialize)]
struct CompressorCase {
    start_pos: usize,
    attention_input: Tensor,
    projected: Tensor,
    latent: Tensor,
}

#[derive(Deserialize)]
struct Consumer {
    layer: usize,
    start_pos: usize,
    consumed_prefix_positions: usize,
    score_key_prefix: Tensor,
    owner_fixture: serde_json::Value,
}

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    numel: usize,
    storage_hex: String,
    storage_sha256: String,
}

impl Tensor {
    fn bytes(&self, width: usize) -> Vec<u8> {
        let count = self.shape.iter().product::<usize>();
        assert_eq!(self.numel, count, "source tensor element count");
        assert_eq!(self.storage_hex.len(), count * width * 2, "source bytes");
        let bytes: Vec<_> = self
            .storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("UTF-8"), 16).expect("hex")
            })
            .collect();
        assert_eq!(format!("{:x}", Sha256::digest(&bytes)), self.storage_sha256);
        bytes
    }

    fn bf16(&self) -> Vec<u16> {
        assert_eq!(self.dtype, "torch.bfloat16");
        self.bytes(2)
            .chunks_exact(2)
            .map(|word| u16::from_le_bytes(word.try_into().expect("BF16")))
            .collect()
    }

    fn frequencies(&self) -> Vec<RotaryFrequency> {
        assert_eq!(self.dtype, "torch.complex64");
        self.bytes(8)
            .chunks_exact(8)
            .map(|word| {
                RotaryFrequency::new(
                    f32::from_le_bytes(word[..4].try_into().expect("real")),
                    f32::from_le_bytes(word[4..].try_into().expect("imaginary")),
                )
                .expect("finite source frequency")
            })
            .collect()
    }
}

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("captured nonzero geometry")
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/layer3-to-layer1-reference.json"
    ))
    .expect("bridge fixture JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source.revision, REVISION);
    assert_eq!(fixture.source.complete_capture_sha256, CAPTURE);
    assert_eq!(
        fixture.producer.keys.source.complete_capture_sha256,
        CAPTURE
    );
    assert_eq!(
        fixture.producer.compressor.source.complete_capture_sha256,
        CAPTURE
    );
    fixture
}

#[test]
fn native_layer_three_publication_feeds_next_partial_layer_one_score_prefix() {
    let fixture = fixture();
    let keys = &fixture.producer.keys;
    let compressor = &fixture.producer.compressor;
    assert_eq!(keys.model.owner_layer, 3);
    assert_eq!(compressor.model.owner_layer, 3);
    assert_eq!(compressor.model.compression_ratio, 1);
    assert_eq!(
        keys.model.latent_dimension,
        compressor.model.latent_dimension
    );
    assert_eq!(
        keys.model.norm_epsilon.to_bits(),
        compressor.model.norm_epsilon.to_bits()
    );
    assert_eq!(keys.cases.len(), 3);
    assert_eq!(compressor.cases.len(), 3);

    let layout = IndexKeyLayout::new(
        nz(keys.model.batches),
        nz(keys.model.latent_dimension),
        nz(keys.model.key_dimension),
        nz(keys.model.rope_pairs),
        keys.model.norm_epsilon,
    )
    .expect("captured key layout");
    let wkv = compressor.weights.wkv.bf16();
    let compressor_norm = compressor.weights.norm.bf16();
    let wk = keys.weights.wk.bf16();
    let key_norm = keys.weights.norm.bf16();
    let weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm));
    let frequencies = keys.frequencies.frequencies();
    let mut owner = RatioOneIndexKeyOwner::new(
        layout,
        nz(compressor.model.input_dimension),
        nz(keys.model.cache_capacity),
        3,
        &compressor_norm,
        compressor.model.norm_epsilon,
    )
    .expect("bounded layer-three producer");

    // The source executes layer three at prefill then start five before its
    // leading three keys become the start-six layer-one score operand.
    for (call_id, (key_case, compressor_case)) in
        keys.cases.iter().zip(&compressor.cases).take(2).enumerate()
    {
        assert_eq!(key_case.start_pos, compressor_case.start_pos);
        let positions = compressor_case.attention_input.shape[1];
        let start = compressor_case.start_pos;
        let diagnostic = owner
            .forward(RatioOneOwnerCall::new(
                deepseek::indexer::cache::IndexKeyPublicationId::new(
                    3,
                    0,
                    u64::try_from(call_id).expect("call id"),
                ),
                start,
                nz(positions),
                &compressor_case.attention_input.bf16(),
                &frequencies
                    [start * keys.model.rope_pairs..(start + positions) * keys.model.rope_pairs],
                weights,
            ))
            .expect("native layer-three source-shaped call");
        assert_eq!(diagnostic.projected, compressor_case.projected.bf16());
        assert_eq!(diagnostic.latent, compressor_case.latent.bf16());
        assert_eq!(diagnostic.latent, key_case.latent.bf16());
        assert_eq!(
            owner.prefix(0).expect("producer prefix"),
            key_case.index_cache_after.bf16()
        );
    }

    assert_eq!(fixture.consumer.layer, 1);
    assert_eq!(fixture.consumer.start_pos, 6);
    assert_eq!(fixture.consumer.consumed_prefix_positions, 3);
    let prefix = owner.prefix(0).expect("six-key producer prefix");
    assert_eq!(prefix.len(), 6 * keys.model.key_dimension);
    assert_eq!(
        &prefix[..fixture.consumer.consumed_prefix_positions * keys.model.key_dimension],
        fixture.consumer.score_key_prefix.bf16(),
        "native layer-three prefix must drive the next partial layer-one score"
    );
    let owner_fixture_json =
        serde_json::to_string(&fixture.consumer.owner_fixture).expect("owner fixture JSON");
    assert_eq!(
        layer1_owner_capture::partial_score_with_native_layer_three_prefix(
            &owner_fixture_json,
            CAPTURE,
            prefix,
        ),
        vec![6],
        "partial layer-one must select using the native layer-three publication"
    );
}
