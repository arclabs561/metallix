//! Test-only native replay of the captured layer-one ratio-two owner publication.

use std::{collections::BTreeMap, num::NonZeroUsize};

use deepseek::{
    RotaryFrequency,
    compressor::{CompressorInput, CompressorState},
    indexer::{
        bf16::index_scores_bf16_reference,
        compressed_kv::{CompressedKvLayout, prepare_compressed_kv},
        key::{IndexKeyLayout, IndexKeyWeights, prepare_index_keys},
        query::{IndexQueryLayout, IndexQueryWeights, prepare_index_query},
    },
    precision::fp32_linear_reference,
    select_indices,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
const CAPTURE_SHA256: &str = "16c949df47afd5cffc5f8ce95612f2d27003dcb23a80c764f7a8d02bef79be16";
const FIXTURE_SHA256: &str = "966122b3fd74fe3f5965d17aad1f863bb67164c69fd66bc52cc7df836db9407a";
const FP32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    source: Source,
    model: Model,
    frequency_table: Tensor,
    encoded_parameters: BTreeMap<String, Tensor>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Source {
    revision: String,
    complete_capture_sha256: String,
}

#[derive(Deserialize)]
struct Model {
    owner_layer: usize,
    ratio: usize,
    norm_eps: f32,
    index_topk: usize,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    sequence: usize,
    compressed_prefix: usize,
    group_frequency_positions: Vec<usize>,
    input: Tensor,
    index_input: Tensor,
    index_qr: Tensor,
    index_operations: IndexOperations,
    offset: usize,
    selected_indices: Tensor,
    wkv_projection: Tensor,
    wgate_projection: Tensor,
    latent: Option<Tensor>,
    index_key_prefix: Tensor,
    index_score_key_prefix: Tensor,
    compressed_kv_prefix: Tensor,
}

#[derive(Deserialize)]
struct IndexOperations {
    q_after_rope_fp4: Tensor,
    weights_proj_output: Tensor,
    scaled_weights: Tensor,
    scores_einsum: Tensor,
    scores_after_relu: Tensor,
    scores_weighted_per_head: Tensor,
    scores_after_head_sum: Tensor,
    scores_after_causal_mask: Option<Tensor>,
}

#[derive(Deserialize)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    numel: usize,
    storage_hex: String,
    storage_sha256: String,
}

/// One native owner publication and its source-qualified selection after a call.
pub(super) struct NativeCase {
    pub start_pos: usize,
    pub latent: Option<Vec<u16>>,
    pub key_prefix: Vec<u16>,
    pub kv_prefix: Vec<u16>,
    pub source_score_key_prefix: Vec<u16>,
    pub selected_indices: Vec<i32>,
}

fn fixture() -> Fixture {
    let raw =
        include_str!("../../../../../fixtures/deepseek-v41/layer1-ratio2-owner-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let fixture: Fixture = serde_json::from_str(raw).expect("layer-one owner fixture JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.source.revision, REVISION);
    assert_eq!(fixture.source.complete_capture_sha256, CAPTURE_SHA256);
    assert_eq!(fixture.model.owner_layer, 1);
    assert_eq!(fixture.model.ratio, 2);
    assert_eq!(fixture.model.index_topk, 1);
    assert_eq!(fixture.cases.len(), 3);
    fixture
}

fn nonzero(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).expect("captured geometry is nonzero")
}

fn bytes(tensor: &Tensor, bytes_per_element: usize) -> Vec<u8> {
    let expected = tensor.shape.iter().copied().product::<usize>();
    assert_eq!(tensor.numel, expected);
    let bytes: Vec<_> = tensor
        .storage_hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).expect("hex"), 16).expect("hex byte")
        })
        .collect();
    assert_eq!(bytes.len(), expected * bytes_per_element);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        tensor.storage_sha256
    );
    bytes
}

fn bf16(tensor: &Tensor) -> Vec<u16> {
    assert_eq!(tensor.dtype, "torch.bfloat16");
    bytes(tensor, 2)
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("BF16 word")))
        .collect()
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn fp32(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype, "torch.float32");
    bytes(tensor, 4)
        .chunks_exact(4)
        .map(|word| f32::from_bits(u32::from_le_bytes(word.try_into().expect("FP32 word"))))
        .collect()
}

fn fp8(tensor: &Tensor) -> Vec<u8> {
    assert!(matches!(
        tensor.dtype.as_str(),
        "torch.float8_e4m3fn" | "torch.float8_e8m0fnu"
    ));
    bytes(tensor, 1)
}

fn i32s(tensor: &Tensor) -> Vec<i32> {
    assert_eq!(tensor.dtype, "torch.int32");
    bytes(tensor, 4)
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes(word.try_into().expect("i32 word")))
        .collect()
}

fn prior_layer_three_prefix() -> Vec<u16> {
    let root: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/forward-candidate-reference.json"
    ))
    .expect("layer-three candidate fixture");
    assert_eq!(root["source"]["revision"].as_str(), Some(REVISION));
    assert_eq!(
        root["source"]["complete_capture_sha256"].as_str(),
        Some("7c5cc8541da338fa3426d63e32b9a66e9132e07ab68ee26d86fbf9e29f62f48d")
    );
    let case = root["cases"]
        .as_array()
        .expect("candidate cases")
        .iter()
        .find(|case| case["start_pos"].as_u64() == Some(5))
        .expect("layer-three start five");
    let tensor = &case["inputs"]["shared_index_k_prefix"];
    assert_eq!(tensor["dtype"].as_str(), Some("torch.bfloat16"));
    assert_eq!(tensor["shape"], serde_json::json!([1, 6, 64]));
    assert_eq!(tensor["numel"].as_u64(), Some(384));
    let hex = tensor["storage_hex"].as_str().expect("storage hex");
    assert!(
        hex.len().is_multiple_of(2),
        "candidate storage hex alignment"
    );
    let raw: Vec<_> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).expect("hex"), 16).expect("byte"))
        .collect();
    assert_eq!(raw.len(), 384 * 2, "candidate BF16 storage length");
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw)),
        tensor["storage_sha256"].as_str().expect("storage hash")
    );
    raw[..3 * 64 * 2]
        .chunks_exact(2)
        .map(|word| u16::from_le_bytes(word.try_into().expect("BF16")))
        .collect()
}

fn frequencies(tensor: &Tensor) -> Vec<RotaryFrequency> {
    assert_eq!(tensor.dtype, "torch.complex64");
    assert_eq!(tensor.shape, [8, 16]);
    bytes(tensor, 8)
        .chunks_exact(8)
        .map(|pair| {
            RotaryFrequency::new(
                f32::from_bits(u32::from_le_bytes(pair[..4].try_into().expect("real"))),
                f32::from_bits(u32::from_le_bytes(pair[4..].try_into().expect("imag"))),
            )
            .expect("finite source frequency")
        })
        .collect()
}

fn parameter<'a>(fixture: &'a Fixture, name: &str) -> &'a Tensor {
    fixture
        .encoded_parameters
        .get(name)
        .unwrap_or_else(|| panic!("missing layer-one owner parameter {name}"))
}

/// Bounds two valid FP32 dot reductions against their shared exact dot.
///
/// Each stream is bounded by a multiply/add reduction with `2n` roundings;
/// their difference is bounded by `gamma(4n) * Σ|xᵢwᵢ|`.  This comes only
/// from operands, term count, and IEEE FP32 unit roundoff, never an observed
/// source/native delta.
fn assert_fp32_projection_envelope(
    name: &str,
    activations: &[f32],
    weights: &[f32],
    rows: usize,
    outputs: usize,
    native: &[f32],
    source: &[f32],
) -> Result<(), String> {
    if rows == 0 || outputs != 64 {
        return Err("layer-one projection envelope has unexpected row or output geometry".into());
    }
    let terms = activations.len() / rows;
    if terms != 128 {
        return Err("layer-one projection envelope has unexpected reduction geometry".into());
    }
    assert_eq!(activations.len(), rows * terms);
    assert_eq!(weights.len(), outputs * terms);
    assert_eq!(native.len(), rows * outputs);
    assert_eq!(source.len(), rows * outputs);
    let operations = u32::try_from(terms.checked_mul(4).expect("bounded dot operation count"))
        .map_err(|_| "layer-one projection envelope operation count exceeds u32".to_owned())?;
    let roundoff = f64::from(operations) * FP32_UNIT_ROUNDOFF;
    if !roundoff.is_finite() || roundoff >= 1.0 {
        return Err("layer-one projection envelope has invalid gamma denominator".into());
    }
    let gamma = roundoff / (1.0 - roundoff);
    if !gamma.is_finite() || gamma <= 0.0 {
        return Err("layer-one projection envelope has nonfinite gamma".into());
    }
    for (field, values) in [
        ("activation", activations),
        ("weight", weights),
        ("native", native),
        ("source", source),
    ] {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(format!(
                "{name} projection has nonfinite {field} at {index}"
            ));
        }
    }
    let mut maximum_error = 0.0_f64;
    let mut maximum_index = 0_usize;
    for row in 0..rows {
        for output in 0..outputs {
            let index = row * outputs + output;
            let magnitude = activations[row * terms..(row + 1) * terms]
                .iter()
                .zip(&weights[output * terms..(output + 1) * terms])
                .map(|(&left, &right)| f64::from(left).abs() * f64::from(right).abs())
                .sum::<f64>();
            let bound = gamma * magnitude;
            if !magnitude.is_finite() || !bound.is_finite() {
                return Err(format!(
                    "{name} projection has nonfinite magnitude or bound at {index}"
                ));
            }
            let error = (f64::from(native[index]) - f64::from(source[index])).abs();
            if error > maximum_error {
                maximum_error = error;
                maximum_index = index;
            }
            if error > bound {
                return Err(format!(
                    "{name} projection envelope rejected element {index}: error {error:e}, bound {bound:e}, maximum error {maximum_error:e} at {maximum_index}"
                ));
            }
        }
    }
    Ok(())
}

fn source_projections(case: &Case, wkv: &[f32], wgate: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(case.input.shape, [1, case.sequence, 128]);
    let input = bf16(&case.input);
    let input_f32: Vec<_> = input.iter().copied().map(bf16_to_f32).collect();
    let mut projected = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input_f32, wkv, case.sequence, 128, 64, &mut projected)
        .expect("bounded native WKV projection");
    assert_fp32_projection_envelope(
        "WKV",
        &input_f32,
        wkv,
        case.sequence,
        64,
        &projected,
        &fp32(&case.wkv_projection),
    )
    .unwrap_or_else(|error| panic!("start {} {error}", case.start_pos));
    let mut gate = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input_f32, wgate, case.sequence, 128, 64, &mut gate)
        .expect("bounded native gate projection");
    assert_fp32_projection_envelope(
        "gate",
        &input_f32,
        wgate,
        case.sequence,
        64,
        &gate,
        &fp32(&case.wgate_projection),
    )
    .unwrap_or_else(|error| panic!("start {} {error}", case.start_pos));
    (projected, gate)
}

fn call_frequencies(all: &[RotaryFrequency], case: &Case) -> Vec<RotaryFrequency> {
    all[case.start_pos * 16..(case.start_pos + case.sequence) * 16].to_vec()
}

fn score_positions(
    query: &[u16],
    keys: &[u16],
    weights: &[u16],
    positions: usize,
) -> (Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    assert_eq!(query.len(), positions * 2 * 64);
    assert_eq!(weights.len(), positions * 2);
    let mut dots = Vec::new();
    let mut relu = Vec::new();
    let mut weighted = Vec::new();
    let mut scores = Vec::new();
    for position in 0..positions {
        let diagnostic = index_scores_bf16_reference(
            &query[position * 128..(position + 1) * 128],
            keys,
            &weights[position * 2..(position + 1) * 2],
            nonzero(64),
        )
        .expect("bounded ratio-two BF16 index score");
        dots.extend(diagnostic.dot_products);
        relu.extend(diagnostic.rectified);
        weighted.extend(diagnostic.weighted);
        scores.extend(diagnostic.scores);
    }
    (dots, relu, weighted, scores)
}

fn causal_scores(scores: &[u16], case: &Case, keys: usize, ratio: usize) -> Vec<u16> {
    let mut causal = scores.to_vec();
    for position in 0..case.sequence {
        let reachable = (case.start_pos + position + 1) / ratio;
        for key in reachable..keys {
            causal[position * keys + key] = 0xff80;
        }
    }
    causal
}

fn selected_indices(
    causal: &[u16],
    case: &Case,
    keys: usize,
    ratio: usize,
    topk: usize,
) -> Vec<i32> {
    (0..case.sequence)
        .flat_map(|position| {
            select_indices(
                &causal[position * keys..(position + 1) * keys]
                    .iter()
                    .copied()
                    .map(bf16_to_f32)
                    .collect::<Vec<_>>(),
                (case.start_pos + position + 1) / ratio,
                topk,
                case.offset,
            )
            .expect("source ratio-two selection cutoff")
        })
        .collect()
}

fn assert_native_score(
    fixture: &Fixture,
    case: &Case,
    all_frequencies: &[RotaryFrequency],
    keys: &[u16],
) -> Vec<i32> {
    let layout = IndexQueryLayout::new(
        nonzero(1),
        nonzero(128),
        nonzero(32),
        nonzero(2),
        nonzero(64),
        nonzero(16),
    )
    .expect("ratio-two query layout");
    let query = prepare_index_query(
        &bf16(&case.index_qr),
        &bf16(&case.index_input),
        &call_frequencies(all_frequencies, case),
        IndexQueryWeights {
            wq_b_codes: &fp8(parameter(fixture, "layers.1.attn.indexer.wq_b.weight")),
            wq_b_scales: &fp8(parameter(fixture, "layers.1.attn.indexer.wq_b.scale")),
            weights_proj: &bf16(parameter(
                fixture,
                "layers.1.attn.indexer.weights_proj.weight",
            )),
        },
        layout,
    )
    .expect("native ratio-two index query");
    let operations = &case.index_operations;
    assert_eq!(query.query_post_fp4, bf16(&operations.q_after_rope_fp4));
    assert_eq!(
        query.projected_head_weights,
        bf16(&operations.weights_proj_output)
    );
    assert_eq!(query.scaled_head_weights, bf16(&operations.scaled_weights));
    let key_count = keys.len() / 64;
    let (dots, relu, weighted, scores) = score_positions(
        &query.query_post_fp4,
        keys,
        &query.scaled_head_weights,
        case.sequence,
    );
    assert_eq!(dots, bf16(&operations.scores_einsum));
    assert_eq!(relu, bf16(&operations.scores_after_relu));
    assert_eq!(weighted, bf16(&operations.scores_weighted_per_head));
    assert_eq!(scores, bf16(&operations.scores_after_head_sum));
    let causal = causal_scores(&scores, case, key_count, fixture.model.ratio);
    if let Some(expected) = &operations.scores_after_causal_mask {
        assert_eq!(causal, bf16(expected));
    } else {
        assert_eq!(
            causal, scores,
            "start {} has no future score",
            case.start_pos
        );
    }
    let indices = selected_indices(
        &causal,
        case,
        key_count,
        fixture.model.ratio,
        fixture.model.index_topk,
    );
    assert_eq!(indices, i32s(&case.selected_indices));
    indices
}

fn layer_three_score_keys(
    case: &Case,
    owned_keys: &[u16],
    source_keys: &[u16],
) -> Option<Vec<u16>> {
    match case.start_pos {
        6 => {
            assert_ne!(
                owned_keys, source_keys,
                "partial layer-one owner prefix stays distinct"
            );
            let previous = prior_layer_three_prefix();
            assert_eq!(
                previous, source_keys,
                "start six previous layer-three score prefix"
            );
            Some(previous)
        }
        0 | 5 => {
            assert_eq!(
                owned_keys, source_keys,
                "start {} owned score prefix",
                case.start_pos
            );
            None
        }
        other => panic!("unexpected layer-one source call start {other}"),
    }
}

/// Replays source-owned KV/key publication through direct index selection.
///
/// The final partial call deliberately exposes its captured score-key operand
/// separately: source global score state was subsequently owned by layer three,
/// whereas this owner retains its own key and KV prefixes.
pub(super) fn native_publications() -> Vec<NativeCase> {
    let fixture = fixture();
    let all_frequencies = frequencies(&fixture.frequency_table);
    let wkv = fp32(parameter(&fixture, "layers.1.attn.compressor.wkv.weight"));
    let wgate = fp32(parameter(&fixture, "layers.1.attn.compressor.wgate.weight"));
    let compressor_norm = bf16(parameter(&fixture, "layers.1.attn.compressor.norm.weight"));
    let wk = bf16(parameter(&fixture, "layers.1.attn.indexer.wk.weight"));
    let key_norm = bf16(parameter(&fixture, "layers.1.attn.indexer.k_norm.weight"));
    let mut compressor = CompressorState::new(1, 64, 2, &compressor_norm, fixture.model.norm_eps)
        .expect("source ratio-two compressor");
    let key_layout = IndexKeyLayout::new(
        nonzero(1),
        nonzero(64),
        nonzero(64),
        nonzero(16),
        fixture.model.norm_eps,
    )
    .expect("source index-key layout");
    let kv_layout = CompressedKvLayout::new(nonzero(1), nonzero(64), nonzero(16))
        .expect("source compressed-KV layout");
    let mut keys = Vec::new();
    let mut kv = Vec::new();
    let mut outputs = Vec::new();
    for case in &fixture.cases {
        let (projected, gate) = source_projections(case, &wkv, &wgate);
        let latent = compressor
            .forward(
                CompressorInput::Gated {
                    kv: &projected,
                    scores: &gate,
                },
                case.sequence,
                case.start_pos,
            )
            .expect("source ratio-two stream call");
        match (&latent, &case.latent) {
            (Some(actual), Some(expected)) => assert_eq!(
                actual,
                &bf16(expected),
                "start {} compressor latent",
                case.start_pos
            ),
            (None, None) => {}
            _ => panic!("start {} latent publication disagrees", case.start_pos),
        }
        if let Some(latent) = &latent {
            assert_eq!(case.group_frequency_positions.len(), latent.len() / 64);
            let mut selected = Vec::new();
            for &position in &case.group_frequency_positions {
                selected.extend_from_slice(&all_frequencies[position * 16..(position + 1) * 16]);
            }
            let prepared_keys = prepare_index_keys(
                latent,
                &selected,
                IndexKeyWeights::new(&wk, &key_norm),
                key_layout,
            )
            .expect("native source-shaped index keys");
            let prepared_kv = prepare_compressed_kv(latent, &selected, kv_layout)
                .expect("native source-shaped compressed KV");
            keys.extend_from_slice(&prepared_keys.post_fp4);
            kv.extend_from_slice(&prepared_kv.post_fp4);
        }
        assert_eq!(
            keys,
            bf16(&case.index_key_prefix),
            "start {} owned key prefix",
            case.start_pos
        );
        assert_eq!(
            kv,
            bf16(&case.compressed_kv_prefix),
            "start {} owned KV prefix",
            case.start_pos
        );
        assert_eq!(keys.len(), case.compressed_prefix * 64);
        let source_score_keys = bf16(&case.index_score_key_prefix);
        let layer_three_score_keys = layer_three_score_keys(case, &keys, &source_score_keys);
        let score_keys = layer_three_score_keys.as_deref().unwrap_or(&keys);
        let selected_indices = assert_native_score(&fixture, case, &all_frequencies, score_keys);
        outputs.push(NativeCase {
            start_pos: case.start_pos,
            latent,
            key_prefix: keys.clone(),
            kv_prefix: kv.clone(),
            source_score_key_prefix: source_score_keys,
            selected_indices,
        });
    }
    outputs
}

/// Replacing the source's partial-decode score keys with the native layer-one
/// owner prefix must trip an observed score-stage gate.
pub(super) fn partial_owner_keys_fail_score_gate() -> bool {
    let outputs = native_publications();
    let fixture = fixture();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.start_pos == 6)
        .expect("captured partial decode");
    let frequencies = frequencies(&fixture.frequency_table);
    let partial = outputs
        .iter()
        .find(|output| output.start_pos == 6)
        .expect("native partial owner publication");
    assert_ne!(partial.key_prefix, partial.source_score_key_prefix);
    std::panic::catch_unwind(|| {
        let _ = assert_native_score(&fixture, case, &frequencies, &partial.key_prefix);
    })
    .is_err()
}

/// The first gate projection is numerically load-bearing to ratio-two pooling.
pub(super) fn first_gate_mutation_changes_latent() -> bool {
    let fixture = fixture();
    let case = &fixture.cases[0];
    let input: Vec<_> = bf16(&case.input).into_iter().map(bf16_to_f32).collect();
    let wkv = fp32(parameter(&fixture, "layers.1.attn.compressor.wkv.weight"));
    let mut wgate = fp32(parameter(&fixture, "layers.1.attn.compressor.wgate.weight"));
    wgate[0] = -wgate[0];
    let mut projected = vec![0.0; case.sequence * 64];
    let mut gate = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input, &wkv, case.sequence, 128, 64, &mut projected)
        .expect("native WKV projection");
    fp32_linear_reference(&input, &wgate, case.sequence, 128, 64, &mut gate)
        .expect("mutated native gate projection");
    let norm = bf16(parameter(&fixture, "layers.1.attn.compressor.norm.weight"));
    let mut compressor = CompressorState::new(1, 64, 2, &norm, fixture.model.norm_eps)
        .expect("source ratio-two compressor");
    let mutated = compressor
        .forward(
            CompressorInput::Gated {
                kv: &projected,
                scores: &gate,
            },
            5,
            0,
        )
        .expect("mutated compressor call")
        .expect("prefill completes two groups");
    mutated != bf16(case.latent.as_ref().expect("prefill latent"))
}

/// A finite source-projection mutation outside the declared FP32 envelope fails.
pub(super) fn projection_envelope_rejects_corruption() -> bool {
    let fixture = fixture();
    let case = &fixture.cases[0];
    let input: Vec<_> = bf16(&case.input).into_iter().map(bf16_to_f32).collect();
    let weights = fp32(parameter(&fixture, "layers.1.attn.compressor.wkv.weight"));
    let mut native = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input, &weights, case.sequence, 128, 64, &mut native)
        .expect("native WKV projection");
    let mut corrupted = native.clone();
    corrupted[0] = native[0] + native[0].abs().max(1.0) * 0.01;
    assert_fp32_projection_envelope(
        "corrupted WKV",
        &input,
        &weights,
        case.sequence,
        64,
        &native,
        &corrupted,
    )
    .is_err()
}

/// Nonfinite values are rejected before a floating-point comparison can mask them.
pub(super) fn projection_envelope_rejects_nonfinite() -> bool {
    let fixture = fixture();
    let case = &fixture.cases[0];
    let input: Vec<_> = bf16(&case.input).into_iter().map(bf16_to_f32).collect();
    let weights = fp32(parameter(&fixture, "layers.1.attn.compressor.wkv.weight"));
    let mut native = vec![0.0; case.sequence * 64];
    fp32_linear_reference(&input, &weights, case.sequence, 128, 64, &mut native)
        .expect("native WKV projection");
    let mut source = native.clone();
    source[0] = f32::NAN;
    let source_rejected = assert_fp32_projection_envelope(
        "nonfinite WKV",
        &input,
        &weights,
        case.sequence,
        64,
        &native,
        &source,
    )
    .is_err();
    source[0] = native[0];
    native[0] = f32::INFINITY;
    let native_rejected = assert_fp32_projection_envelope(
        "nonfinite WKV",
        &input,
        &weights,
        case.sequence,
        64,
        &native,
        &source,
    )
    .is_err();
    source_rejected && native_rejected
}
