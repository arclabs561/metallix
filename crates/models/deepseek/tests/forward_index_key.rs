//! Owner-layer key preparation against the reduced source-forward capture.
//! Source inputs feed the native ratio-one compressor, key preparation and cache.

use std::num::NonZeroUsize;

use deepseek::{
    RotaryFrequency,
    compressor::{CompressorInput, CompressorState},
    indexer::{
        cache::{IndexKeyPublicationId, IndexKeyState},
        key::{IndexKeyLayout, IndexKeyWeights, prepare_index_keys},
    },
    precision::bf16_linear_reference,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    model: Model,
    weights: Weights,
    frequencies: Tensor,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    complete_capture_sha256: String,
}

#[derive(Deserialize)]
struct Model {
    batches: usize,
    latent_dimension: usize,
    key_dimension: usize,
    rope_pairs: usize,
    norm_epsilon: f32,
    owner_layer: usize,
    cache_capacity: usize,
}

#[derive(Deserialize)]
struct Weights {
    wk: Tensor,
    norm: Tensor,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    latent: Tensor,
    index_cache_after: Tensor,
}

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    numel: usize,
    storage_hex: String,
    storage_sha256: String,
}

#[derive(Deserialize)]
struct CompressorFixture {
    schema_version: u32,
    source: Source,
    model: CompressorModel,
    weights: CompressorWeights,
    cases: Vec<CompressorCase>,
}

#[derive(Deserialize)]
struct CompressorModel {
    batches: usize,
    input_dimension: usize,
    latent_dimension: usize,
    norm_epsilon: f32,
    owner_layer: usize,
    compression_ratio: usize,
    expected_start_positions: Vec<usize>,
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

fn native_compressor_latents() -> Vec<(usize, Vec<u16>)> {
    let fixture: CompressorFixture = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-compressor-reference.json"
    ))
    .expect("supplementary compressor fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture.source.complete_capture_sha256,
        "2f2ff3f1734f959b33a673773cf6fe9c056fabb06a82531e5465562af5480c39"
    );
    assert_eq!(fixture.weights.wkv.shape, [64, 128]);
    assert_eq!(fixture.weights.norm.shape, [64]);
    let model = fixture.model;
    assert_eq!(
        (
            model.batches,
            model.input_dimension,
            model.latent_dimension,
            model.owner_layer,
            model.compression_ratio
        ),
        (1, 128, 64, 3, 1)
    );
    assert_eq!(model.norm_epsilon.to_bits(), 1e-20_f32.to_bits());
    assert_eq!(model.expected_start_positions, [0, 5, 6]);
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| case.start_pos)
            .collect::<Vec<_>>(),
        model.expected_start_positions
    );
    let wkv = fixture.weights.wkv.bf16();
    let norm = fixture.weights.norm.bf16();
    let mut compressor =
        CompressorState::new(1, 64, 1, &norm, 1e-20).expect("captured ratio-one compressor");
    fixture
        .cases
        .into_iter()
        .map(|case| {
            let positions = if case.start_pos == 0 { 5 } else { 1 };
            assert_eq!(case.attention_input.shape, [1, positions, 128]);
            assert_eq!(case.projected.shape, [1, positions, 64]);
            assert_eq!(case.latent.shape, [1, positions, 64]);
            let input = case.attention_input.bf16();
            let mut projected = vec![0; positions * 64];
            bf16_linear_reference(&input, &wkv, positions, 128, 64, &mut projected)
                .expect("native compressor projection");
            assert_eq!(projected, case.projected.bf16(), "source wkv output");
            let latent = compressor
                .forward(
                    CompressorInput::ProjectedBf16(&projected),
                    positions,
                    case.start_pos,
                )
                .expect("native compressor normalization")
                .expect("ratio one always completes");
            assert_eq!(
                latent,
                case.latent.bf16(),
                "source pre-mutation compressor latent"
            );
            let mut wrong_projected = vec![0; positions * 64];
            bf16_linear_reference(
                &input,
                &vec![0; wkv.len()],
                positions,
                128,
                64,
                &mut wrong_projected,
            )
            .expect("zero-weight control");
            let mut wrong =
                CompressorState::new(1, 64, 1, &norm, 1e-20).expect("control compressor");
            let wrong_latent = wrong
                .forward(
                    CompressorInput::ProjectedBf16(&wrong_projected),
                    positions,
                    0,
                )
                .expect("control normalization")
                .expect("completed control");
            assert_ne!(
                wrong_latent, latent,
                "oracle detects dropped compressor projection"
            );
            (case.start_pos, latent)
        })
        .collect()
}

impl Tensor {
    fn bytes(&self, width: usize) -> Vec<u8> {
        let count = self
            .shape
            .iter()
            .try_fold(1_usize, |n, d| n.checked_mul(*d))
            .expect("bounded shape");
        assert_eq!(count, self.numel);
        assert!(count <= 16_384, "small owner-key fixture");
        assert_eq!(self.storage_hex.len(), count * width * 2);
        let bytes: Vec<_> = self
            .storage_hex
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                u8::from_str_radix(std::str::from_utf8(pair).expect("hex UTF-8"), 16)
                    .expect("hex byte")
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
            .map(|word| {
                RotaryFrequency::new(
                    f32::from_le_bytes(word[..4].try_into().expect("real FP32")),
                    f32::from_le_bytes(word[4..].try_into().expect("imaginary FP32")),
                )
                .expect("finite source frequency")
            })
            .collect()
    }
}

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("nonzero captured dimension")
}

fn fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../../fixtures/deepseek-v41/forward-index-key-reference.json"
    ))
    .expect("owner-key fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(
        fixture.source.revision,
        "dba1be0a40aa45a94ad051997016db3960a90277"
    );
    assert_eq!(
        fixture.source.complete_capture_sha256,
        "e27dde6ead409c74f7bb2c9e08d4cd5a2b0cfc3c9505c7d6b8908b1cd78b1cc6"
    );
    let model = &fixture.model;
    assert_eq!(
        (
            model.owner_layer,
            model.batches,
            model.latent_dimension,
            model.key_dimension,
            model.rope_pairs
        ),
        (3, 1, 64, 64, 16)
    );
    assert_eq!(model.norm_epsilon.to_bits(), 1e-20_f32.to_bits());
    assert_eq!(fixture.weights.wk.shape, [64, 64]);
    assert_eq!(fixture.weights.norm.shape, [64]);
    assert_eq!(fixture.frequencies.shape, [8, 16]);
    assert_eq!(
        fixture
            .cases
            .iter()
            .map(|case| case.start_pos)
            .collect::<Vec<_>>(),
        [0, 5, 6]
    );
    fixture
}

#[test]
fn native_owner_keys_match_captured_cache_append_regions() {
    let fixture = fixture();
    let model = fixture.model;
    let layout = IndexKeyLayout::new(nz(1), nz(64), nz(64), nz(16), model.norm_epsilon)
        .expect("source layout");
    let wk = fixture.weights.wk.bf16();
    let norm = fixture.weights.norm.bf16();
    let weights = IndexKeyWeights::new(&wk, &norm);
    let frequencies = fixture.frequencies.frequencies();
    let mut previous_prefix = Vec::new();
    let owner = u16::try_from(model.owner_layer).expect("source layer");
    let mut state = IndexKeyState::new(
        nz(model.batches),
        nz(model.key_dimension),
        nz(model.cache_capacity),
        owner,
    )
    .expect("bounded source cache");
    let mut wrong_frequency_detected = false;
    let native_latents = native_compressor_latents();
    assert_eq!(native_latents.len(), fixture.cases.len());
    for (call_id, (case, (start_pos, latent))) in
        fixture.cases.into_iter().zip(native_latents).enumerate()
    {
        let positions = if case.start_pos == 0 { 5 } else { 1 };
        assert_eq!(case.latent.shape, [1, positions, 64]);
        assert_eq!(
            case.index_cache_after.shape,
            [1, case.start_pos + positions, 64]
        );
        assert_eq!(start_pos, case.start_pos);
        assert_eq!(
            latent,
            case.latent.bf16(),
            "new capture agrees with prior key oracle"
        );
        let cache = case.index_cache_after.bf16();
        let start = case.start_pos * 64;
        let end = (case.start_pos + positions) * 64;
        assert_eq!(
            &cache[..start],
            previous_prefix,
            "source cache preserves prior keys"
        );
        let prepared = prepare_index_keys(
            &latent,
            &frequencies[case.start_pos * 16..(case.start_pos + positions) * 16],
            weights,
            layout,
        )
        .expect("native captured owner-key preparation");
        assert_eq!(
            prepared.post_fp4,
            cache[start..end],
            "owner keys at start {}",
            case.start_pos
        );
        assert!(
            prepared.post_fp4.iter().any(|&bits| bits & 0x7fff != 0),
            "nontrivial captured keys"
        );
        state
            .append_prepared(
                IndexKeyPublicationId::new(owner, 0, u64::try_from(call_id).expect("call ID")),
                case.start_pos,
                &prepared.post_fp4,
            )
            .expect("atomic native append");
        assert_eq!(
            state.prefix(0).expect("batch zero"),
            cache,
            "complete native prefix"
        );
        if case.start_pos != 0 {
            let wrong = prepare_index_keys(&latent, &frequencies[..16], weights, layout)
                .expect("wrong but finite position");
            wrong_frequency_detected |= wrong.post_fp4 != prepared.post_fp4;
        }
        previous_prefix = cache[..end].to_vec();
    }
    assert!(
        wrong_frequency_detected,
        "decode oracle detects replaying position-zero rotary frequencies"
    );
}
