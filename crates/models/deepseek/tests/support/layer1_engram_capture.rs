//! Native layer-one Engram capture composition.

use deepseek::{
    engram::{
        CompressedToken, EngramHashLayout, EngramHashState,
        embedding::{EngramEmbeddingLayout, engram_embedding_bf16_reference},
        gate::{
            EngramGateInputs, EngramGateLayout, EngramGateParams,
            engram_residual_gate_bf16_reference,
        },
    },
    precision::{ActivationGroup, fp8_linear_runtime_f32, quantize_bf16_activations_e4m3fn},
};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn field<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap_or_else(|| panic!("missing {key}"))
}
fn usize_field(v: &Value, key: &str) -> usize {
    field(v, key)
        .as_u64()
        .and_then(|x| x.try_into().ok())
        .expect("usize")
}
fn shape(v: &Value) -> Vec<usize> {
    field(v, "shape")
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap().try_into().unwrap())
        .collect()
}
fn bytes(v: &Value) -> Vec<u8> {
    let n: usize = shape(v).iter().product();
    assert_eq!(
        usize::try_from(field(v, "numel").as_u64().unwrap()).unwrap(),
        n
    );
    let hex = field(v, "storage_hex").as_str().unwrap();
    assert!(hex.len().is_multiple_of(2));
    let b: Vec<_> = hex
        .as_bytes()
        .chunks_exact(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect();
    assert_eq!(
        format!("{:x}", Sha256::digest(&b)),
        field(v, "storage_sha256").as_str().unwrap()
    );
    b
}
fn i64s(v: &Value) -> Vec<i64> {
    assert_eq!(field(v, "dtype").as_str(), Some("torch.int64"));
    let b = bytes(v);
    assert_eq!(b.len(), shape(v).iter().product::<usize>() * 8);
    b.chunks_exact(8)
        .map(|x| i64::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn bf16(v: &Value) -> Vec<u16> {
    assert_eq!(field(v, "dtype").as_str(), Some("torch.bfloat16"));
    let b = bytes(v);
    assert_eq!(b.len(), shape(v).iter().product::<usize>() * 2);
    b.chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn fp8_codes(v: &Value) -> Vec<u8> {
    assert_eq!(field(v, "dtype").as_str(), Some("torch.float8_e4m3fn"));
    bytes(v)
}
fn fp8_scales(v: &Value) -> Vec<u8> {
    assert_eq!(field(v, "dtype").as_str(), Some("torch.float8_e8m0fnu"));
    bytes(v)
}
fn f32_from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
fn bf16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = ((bits >> 16) & 1) + 0x7fff;
    ((bits.wrapping_add(round)) >> 16) as u16
}

fn fixture() -> Value {
    let raw = include_str!("../../../../../fixtures/deepseek-v41/layer1-engram-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "6ea65d4f9127b235db1dad1e91eaee4101551a2d7efe53a5c9105b1d980a2062"
    );
    let root: Value = serde_json::from_str(raw).unwrap();
    assert_eq!(field(&root, "schema_version").as_u64(), Some(1));
    assert_eq!(
        field(field(&root, "source"), "revision").as_str(),
        Some("dba1be0a40aa45a94ad051997016db3960a90277")
    );
    assert!(
        field(&root, "cases")
            .as_array()
            .unwrap()
            .iter()
            .any(|case| { bf16(field(case, "stream")) != bf16(field(case, "output")) }),
        "omitting Engram must fail at least one source trace boundary"
    );
    root
}

/// Produces exact gated layer-one block entries keyed by source start position.
pub(super) fn native_layer_one_block_entries() -> Vec<(usize, Vec<u16>)> {
    native_layer_one_block_entries_from_streams(None)
}

/// Recomputes the layer-one Engram gate from native upstream residuals.
pub(super) fn native_layer_one_block_entries_from_streams(
    supplied_streams: Option<&[(usize, Vec<u16>)]>,
) -> Vec<(usize, Vec<u16>)> {
    let root = fixture();
    native_block_entries_from_streams(&root, 1, 0, supplied_streams)
}

/// Replays one layer-qualified Engram source capture.
///
/// `hash_column` indexes the model-qualified Engram columns in the shared
/// hash-state result. The layer-one fixture uses hash column zero without changing the stateful
/// hash transition.
fn native_block_entries_from_streams(
    root: &Value,
    layer: usize,
    hash_column: usize,
    supplied_streams: Option<&[(usize, Vec<u16>)]>,
) -> Vec<(usize, Vec<u16>)> {
    let model = field(root, "model");
    let engram = field(root, "engram");
    let state = field(engram, "hash_state");
    let layout = field(engram, "layout");
    let layer_ids = field(layout, "layer_ids").as_array().unwrap();
    assert_eq!(layer_ids[hash_column].as_u64(), Some(layer as u64));
    let token_map = i64s(field(state, "token_map"));
    let hash_layout = EngramHashLayout::new(
        usize_field(layout, "max_ngram_size"),
        usize_field(layout, "n_heads"),
        layer_ids.len(),
        field(state, "pad_id").as_i64().unwrap(),
        i64s(field(state, "primes")),
        i64s(field(state, "offsets")),
        i64s(field(state, "multipliers")),
    )
    .unwrap();
    let cases = field(root, "cases").as_array().unwrap();
    let capacity = cases
        .iter()
        .map(|c| usize_field(c, "start_pos") + shape(field(c, "input_ids"))[1])
        .max()
        .unwrap();
    let mut hashes = EngramHashState::new(hash_layout, 1, capacity).unwrap();
    let parameters = field(root, "encoded_parameters");
    let prefix = format!("layers.{layer}.engram");
    let embed_codes = fp8_codes(field(parameters, &format!("{prefix}.embed.weight")));
    let embed_scales = fp8_scales(field(parameters, &format!("{prefix}.embed.scale")));
    let embed_rows = field(layout, "num_embeddings").as_array().unwrap()[hash_column]
        .as_u64()
        .unwrap()
        .try_into()
        .unwrap();
    let embed =
        EngramEmbeddingLayout::new(embed_rows, usize_field(model, "embedding_dim"), 32).unwrap();
    let wkv_codes = fp8_codes(field(parameters, &format!("{prefix}.wkv.weight")));
    let wkv_scales = fp8_scales(field(parameters, &format!("{prefix}.wkv.scale")));
    let q: Vec<f32> = bf16(field(parameters, &format!("{prefix}.q_weight")))
        .into_iter()
        .map(f32_from_bf16)
        .collect();
    let k: Vec<f32> = bf16(field(parameters, &format!("{prefix}.k_weight")))
        .into_iter()
        .map(f32_from_bf16)
        .collect();
    if let Some(streams) = supplied_streams {
        assert_eq!(streams.len(), cases.len(), "native Engram stream count");
    }
    cases
        .iter()
        .enumerate()
        .map(|(case_index, case)| {
            let positions = shape(field(case, "input_ids"))[1];
            let start = usize_field(case, "start_pos");
            let tokens: Vec<_> = i64s(field(case, "input_ids"))
                .into_iter()
                .map(|id| CompressedToken::Live(token_map[usize::try_from(id).unwrap()]))
                .collect();
            let all = hashes.write_and_hash(&tokens, positions, start).unwrap();
            let columns = usize_field(model, "hash_columns");
            let ids: Vec<_> = all
                .chunks_exact(layer_ids.len() * columns)
                .flat_map(|row| {
                    row[hash_column * columns..(hash_column + 1) * columns]
                        .iter()
                        .copied()
                })
                .collect();
            assert_eq!(ids, i64s(field(case, "captured_hash_ids")));
            let mut looked = vec![0; ids.len() * usize_field(model, "embedding_dim")];
            engram_embedding_bf16_reference(&ids, &embed_codes, &embed_scales, embed, &mut looked)
                .unwrap();
            assert_eq!(looked, bf16(field(case, "embedding")));
            check_masked_lookup(&ids, &embed_codes, &embed_scales, embed, &looked);
            let wkv = project_wkv(
                &looked,
                positions,
                columns * usize_field(model, "embedding_dim"),
                usize_field(model, "wkv_width"),
                &wkv_codes,
                &wkv_scales,
            );
            assert_eq!(wkv, bf16(field(case, "wkv_output")));
            let (key, value) = split_wkv(case, model, &wkv);
            let captured_stream = bf16(field(case, "stream"));
            let stream = if let Some(streams) = supplied_streams {
                let (supplied_start, supplied) = &streams[case_index];
                assert_eq!(*supplied_start, start, "native Engram stream start");
                assert_eq!(supplied, &captured_stream, "native Engram stream boundary");
                supplied.as_slice()
            } else {
                captured_stream.as_slice()
            };
            let output = gate_output(case, model, stream, &key, &value, &q, &k);
            (start, output)
        })
        .collect()
}

fn split_wkv(case: &Value, model: &Value, wkv: &[u16]) -> (Vec<u16>, Vec<u16>) {
    let mut key = Vec::new();
    let mut value = Vec::new();
    for row in wkv.chunks_exact(usize_field(model, "wkv_width")) {
        key.extend_from_slice(&row[..256]);
        value.extend_from_slice(&row[256..]);
    }
    assert_eq!(key, bf16(field(case, "key")));
    assert_eq!(value, bf16(field(case, "value")));
    (key, value)
}

fn check_masked_lookup(
    ids: &[i64],
    codes: &[u8],
    scales: &[u8],
    layout: EngramEmbeddingLayout,
    expected: &[u16],
) {
    let mut wrong_ids = ids.to_vec();
    wrong_ids[0] = -1;
    let mut wrong_lookup = vec![0; expected.len()];
    engram_embedding_bf16_reference(&wrong_ids, codes, scales, layout, &mut wrong_lookup).unwrap();
    assert_ne!(
        wrong_lookup, expected,
        "masked hash must change native embedding"
    );
}

fn project_wkv(
    looked: &[u16],
    rows: usize,
    reduction: usize,
    outputs: usize,
    codes: &[u8],
    scales: &[u8],
) -> Vec<u16> {
    let mut ac = vec![0; rows * reduction];
    let mut ascale = vec![0; rows * (reduction / 32)];
    quantize_bf16_activations_e4m3fn(
        looked,
        rows,
        reduction,
        ActivationGroup::Elements32,
        &mut ac,
        &mut ascale,
    )
    .unwrap();
    let mut projected = vec![0.0; rows * outputs];
    fp8_linear_runtime_f32(
        &ac,
        &ascale,
        codes,
        scales,
        rows,
        reduction,
        outputs,
        ActivationGroup::Elements32,
        &mut projected,
    )
    .unwrap();
    projected.into_iter().map(bf16_rne).collect()
}

fn gate_output(
    case: &Value,
    model: &Value,
    stream: &[u16],
    key: &[u16],
    value: &[u16],
    q: &[f32],
    k: &[f32],
) -> Vec<u16> {
    let positions = shape(field(case, "input_ids"))[1];
    let mut output = vec![0; stream.len()];
    engram_residual_gate_bf16_reference(
        EngramGateInputs {
            stream,
            key,
            value,
            q_weight: q,
            k_weight: k,
            mask: None,
        },
        EngramGateLayout::new(
            1,
            positions,
            usize_field(model, "copies"),
            usize_field(model, "dim"),
        )
        .unwrap(),
        EngramGateParams::new(
            serde_json::from_value(field(model, "norm_eps").clone()).unwrap(),
            serde_json::from_value(field(model, "gate_clamp").clone()).unwrap(),
        )
        .unwrap(),
        &mut output,
    )
    .unwrap();
    assert_eq!(output, bf16(field(case, "output")));
    assert_eq!(output, bf16(field(case, "block_entry")));
    output
}
