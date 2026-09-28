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
pub(crate) fn native_layer_one_block_entries() -> Vec<(usize, Vec<u16>)> {
    let mut session = NativeLayerOneEngramSession::new();
    (0..3).map(|_| session.step(None)).collect()
}

/// Recomputes the layer-one Engram gate from native upstream residuals.
pub(crate) fn native_layer_one_block_entries_from_streams(
    supplied_streams: Option<&[(usize, Vec<u16>)]>,
) -> Vec<(usize, Vec<u16>)> {
    let count = supplied_streams.map_or(3, <[_]>::len);
    assert!(
        (1..=3).contains(&count),
        "native Engram stream prefix count"
    );
    let mut session = NativeLayerOneEngramSession::new();
    (0..count)
        .map(|index| session.step(supplied_streams.map(|streams| &streams[index])))
        .collect()
}

/// Test-private layer-one Engram request session.  The hash history is kept
/// live so start five and six extend the same source request rather than
/// replaying its bootstrap prefix.
pub(crate) struct NativeLayerOneEngramSession {
    root: Value,
    hashes: EngramHashState,
    next_case: usize,
}

impl NativeLayerOneEngramSession {
    pub(crate) fn new() -> Self {
        let root = fixture();
        let engram = field(&root, "engram");
        let state = field(engram, "hash_state");
        let layout = field(engram, "layout");
        let layer_ids = field(layout, "layer_ids").as_array().unwrap();
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
        let capacity = field(&root, "cases")
            .as_array()
            .unwrap()
            .iter()
            .map(|case| usize_field(case, "start_pos") + shape(field(case, "input_ids"))[1])
            .max()
            .unwrap();
        Self {
            root,
            hashes: EngramHashState::new(hash_layout, 1, capacity).unwrap(),
            next_case: 0,
        }
    }

    pub(crate) fn step(
        &mut self,
        supplied_stream: Option<&(usize, Vec<u16>)>,
    ) -> (usize, Vec<u16>) {
        let model = field(&self.root, "model");
        let engram = field(&self.root, "engram");
        let state = field(engram, "hash_state");
        let layout = field(engram, "layout");
        let layer_ids = field(layout, "layer_ids").as_array().unwrap();
        let case = &field(&self.root, "cases").as_array().unwrap()[self.next_case];
        let start = usize_field(case, "start_pos");
        assert_eq!(start, [0, 5, 6][self.next_case], "native Engram call order");
        // Reject caller input before publishing any token history.
        let captured = bf16(field(case, "stream"));
        let stream = supplied_stream.map_or(captured.as_slice(), |(supplied_start, supplied)| {
            assert_eq!(*supplied_start, start, "native Engram stream start");
            assert_eq!(supplied, &captured, "native Engram stream boundary");
            supplied.as_slice()
        });
        let positions = shape(field(case, "input_ids"))[1];
        let token_map = i64s(field(state, "token_map"));
        let tokens: Vec<_> = i64s(field(case, "input_ids"))
            .into_iter()
            .map(|id| CompressedToken::Live(token_map[usize::try_from(id).unwrap()]))
            .collect();
        let all = self
            .hashes
            .write_and_hash(&tokens, positions, start)
            .unwrap();
        let columns = usize_field(model, "hash_columns");
        let ids: Vec<_> = all
            .chunks_exact(layer_ids.len() * columns)
            .flat_map(|row| row[..columns].iter().copied())
            .collect();
        assert_eq!(ids, i64s(field(case, "captured_hash_ids")));
        let parameters = field(&self.root, "encoded_parameters");
        let embed_codes = fp8_codes(field(parameters, "layers.1.engram.embed.weight"));
        let embed_scales = fp8_scales(field(parameters, "layers.1.engram.embed.scale"));
        let embed_rows = field(layout, "num_embeddings").as_array().unwrap()[0]
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap();
        let embed = EngramEmbeddingLayout::new(embed_rows, usize_field(model, "embedding_dim"), 32)
            .unwrap();
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
            &fp8_codes(field(parameters, "layers.1.engram.wkv.weight")),
            &fp8_scales(field(parameters, "layers.1.engram.wkv.scale")),
        );
        assert_eq!(wkv, bf16(field(case, "wkv_output")));
        let (key, value) = split_wkv(case, model, &wkv);
        let q = bf16(field(parameters, "layers.1.engram.q_weight"))
            .into_iter()
            .map(f32_from_bf16)
            .collect::<Vec<_>>();
        let k = bf16(field(parameters, "layers.1.engram.k_weight"))
            .into_iter()
            .map(f32_from_bf16)
            .collect::<Vec<_>>();
        let output = gate_output(case, model, stream, &key, &value, &q, &k);
        self.next_case += 1;
        (start, output)
    }
}

#[test]
fn rejected_engram_stream_does_not_publish_hash_history() {
    for rejected_case in [0, 1] {
        let mut session = NativeLayerOneEngramSession::new();
        let mut control = NativeLayerOneEngramSession::new();
        for _ in 0..rejected_case {
            assert_eq!(session.step(None), control.step(None));
        }
        let case = &field(&session.root, "cases").as_array().unwrap()[rejected_case];
        let start = usize_field(case, "start_pos");
        let following_start = start + shape(field(case, "input_ids"))[1];
        let mut invalid = bf16(field(case, "stream"));
        invalid[0] ^= 1;
        let mut before = session.hashes.clone();
        assert!(
            before
                .write_and_hash(&[CompressedToken::Live(0)], 1, following_start)
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                session.step(Some(&(start, invalid)));
            }))
            .is_err()
        );
        assert_eq!(session.next_case, rejected_case);
        let mut after = session.hashes.clone();
        assert!(
            after
                .write_and_hash(&[CompressedToken::Live(0)], 1, following_start)
                .is_err(),
            "rejected stream must not make the next token's history available"
        );
        for _ in rejected_case..3 {
            assert_eq!(session.step(None), control.step(None));
        }
    }
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
    let call_count = supplied_streams.map_or(cases.len(), <[_]>::len);
    assert!(
        (1..=cases.len()).contains(&call_count),
        "native Engram stream prefix count"
    );
    cases
        .iter()
        .take(call_count)
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
