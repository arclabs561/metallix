//! Native layer-three Engram capture composition.

use serde_json::Value;

use crate::layer_zero::runtime_engram;
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
fn fixture() -> Value {
    let raw = include_str!("../../../../../fixtures/deepseek-v41/layer3-engram-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        "df6778841940c4861d991ec90a7394b84bf27d923028b348ebb74e1a7d41e4f7"
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

/// Produces exact gated layer-three block entries keyed by source start position.
pub(super) fn native_layer_three_block_entries() -> Vec<(usize, Vec<u16>)> {
    let mut session = NativeLayerThreeEngramSession::new();
    (0..3).map(|_| session.step(None)).collect()
}

/// Recomputes the layer-three Engram gate from native upstream residuals.
pub(super) fn native_layer_three_block_entries_from_streams(
    supplied_streams: Option<&[(usize, Vec<u16>)]>,
) -> Vec<(usize, Vec<u16>)> {
    let count = supplied_streams.map_or(3, <[_]>::len);
    assert!(
        (1..=3).contains(&count),
        "native Engram stream prefix count"
    );
    let mut session = NativeLayerThreeEngramSession::new();
    (0..count)
        .map(|index| session.step(supplied_streams.map(|streams| &streams[index])))
        .collect()
}

/// Test-private layer-three Engram request state. Hash history is retained
/// across the source partitions rather than rebuilding the [0, 5] prefix.
pub(super) struct NativeLayerThreeEngramSession {
    root: Value,
    runtime: deepseek::reduced::EngramSession,
    next_case: usize,
    next_start: usize,
}

impl NativeLayerThreeEngramSession {
    pub(super) fn new() -> Self {
        Self::from_root(fixture())
    }

    /// Starts the persistent layer-three Engram from the unified reduced
    /// bundle. Metadata remains pinned to the checked-in capture, while every
    /// numerical operand comes from the caller's projection.
    pub(super) fn from_bundle(bundle: &Value) -> Self {
        assert_eq!(field(bundle, "schema_version").as_u64(), Some(1));
        let pinned: Value = serde_json::from_str(include_str!(
            "../../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
        ))
        .expect("pinned reduced bundle metadata");
        let source = field(bundle, "source");
        assert_eq!(source, field(&pinned, "source"), "bundle source metadata");
        let projection = field(field(bundle, "projections"), "layer3_engram");
        assert_eq!(field(projection, "schema_version").as_u64(), Some(1));
        assert_eq!(
            field(projection, "source"),
            field(&field(&pinned, "projections")["layer3_engram"], "source"),
            "layer3 Engram source metadata"
        );
        Self::from_root(projection.clone())
    }

    pub(super) fn from_alternate(projection: &Value, layer_two: &Value) -> Self {
        for name in ["source", "source_receipt_sha256", "capture_identity"] {
            assert_eq!(
                projection[name], layer_two[name],
                "alternate L3 Engram provenance"
            );
        }
        Self::from_root(projection.clone())
    }

    fn from_root(root: Value) -> Self {
        assert_eq!(field(&root, "schema_version").as_u64(), Some(1));
        assert_eq!(
            field(field(&root, "source"), "revision").as_str(),
            Some("dba1be0a40aa45a94ad051997016db3960a90277")
        );
        assert!(
            field(&root, "cases")
                .as_array()
                .expect("layer-three Engram cases")
                .iter()
                .any(|case| bf16(field(case, "stream")) != bf16(field(case, "output"))),
            "omitting Engram must fail at least one source trace boundary"
        );
        let runtime = runtime_engram::session(&root, 3);
        Self {
            root,
            runtime,
            next_case: 0,
            next_start: 0,
        }
    }
    pub(super) fn step(&mut self, supplied: Option<&(usize, Vec<u16>)>) -> (usize, Vec<u16>) {
        let case = &field(&self.root, "cases").as_array().unwrap()[self.next_case];
        let start = usize_field(case, "start_pos");
        assert_eq!(start, self.next_start, "native Engram call order");
        let captured = bf16(field(case, "stream"));
        let stream = supplied.map_or(captured.as_slice(), |(given, input)| {
            assert_eq!(*given, start, "native Engram stream start");
            assert_eq!(input, &captured, "native Engram stream boundary");
            input.as_slice()
        });
        let tokens = i64s(field(case, "input_ids"));
        let output = self
            .runtime
            .step(start, &tokens, stream)
            .expect("native runtime Engram");
        runtime_engram::assert_output(case, &output);
        runtime_engram::assert_masked_embedding(&self.root, 3, &output);
        self.next_case += 1;
        self.next_start = self.runtime.next_start();
        (start, output.output().to_vec())
    }
}

#[test]
fn rejected_engram_stream_does_not_publish_hash_history() {
    for rejected_case in [0, 1] {
        let mut session = NativeLayerThreeEngramSession::new();
        let mut control = NativeLayerThreeEngramSession::new();
        for _ in 0..rejected_case {
            assert_eq!(session.step(None), control.step(None));
        }
        let case = &field(&session.root, "cases").as_array().unwrap()[rejected_case];
        let start = usize_field(case, "start_pos");
        let following_start = start + shape(field(case, "input_ids"))[1];
        let mut invalid = bf16(field(case, "stream"));
        invalid[0] ^= 1;
        let mut before = session.runtime.clone();
        assert!(before.step(following_start, &[0], &[0; 256]).is_err());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                session.step(Some(&(start, invalid)));
            }))
            .is_err()
        );
        assert_eq!(session.next_case, rejected_case);
        let mut after = session.runtime.clone();
        assert!(
            after.step(following_start, &[0], &[0; 256]).is_err(),
            "rejected stream must not admit the following start"
        );
        for _ in rejected_case..3 {
            assert_eq!(session.step(None), control.step(None));
        }
    }
}
