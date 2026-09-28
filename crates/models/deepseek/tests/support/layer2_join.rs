//! Layer-two HC/attention/FFN join into the already-qualified final suffix.
//! Earlier block residuals and incoming HC state remain captured boundaries.

use std::collections::BTreeMap;

use deepseek::hc::{
    HcCoefficients, mixing::hc_post_bf16_reference, projection::project_hc_diagnostics,
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{
    BlockConfig, BlockControl, Coefficients, Tensor, assert_final_suffix, block_tail_from_entries,
    derive_attention_input, engram_capture, fixture, hc_coefficient_bounds, hc_projection_bounds,
    layer_three_fixture, layer1_owner_capture, layer2_attention_capture, layer2_ffn,
    native_layer_three_block_tail_from_entries,
};

#[derive(Deserialize)]
struct HcFixture {
    schema_version: u32,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
    cases: Vec<HcCase>,
}

#[derive(Deserialize)]
struct HcCase {
    start_pos: usize,
    residual: Tensor,
    incoming_pre: Tensor,
    attention_input: Tensor,
    after_attention_residual: Tensor,
    attention_pre: Tensor,
    attention_hc_mixes: Tensor,
    attention_coefficients: Coefficients,
}

fn hc_fixture() -> HcFixture {
    let raw = include_str!("../../../../../fixtures/deepseek-v41/layer2-hc-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw)),
        "9bc422171d741516879dd3e14dd856fc1b099e9521190d4da75e1c8ba1722edf"
    );
    let fixture: HcFixture = serde_json::from_str(raw).unwrap();
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.block_config.copies, 2);
    assert_eq!(fixture.block_config.hc_sinkhorn_iters, 20);
    assert_eq!(fixture.block_config.norm_eps.to_bits(), 1e-20_f32.to_bits());
    assert_eq!(fixture.block_config.hc_eps.to_bits(), 1e-6_f32.to_bits());
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

/// Decodes L2's HC projection from the unified reduced-runner bundle.  The
/// source record is pinned for identity while all retained HC tensors come
/// from the caller's projection.
fn hc_fixture_from_bundle(bundle: &Value) -> HcFixture {
    assert_eq!(bundle["schema_version"].as_u64(), Some(1));
    let pinned: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("pinned reduced bundle metadata");
    assert_eq!(bundle["source"], pinned["source"], "bundle source metadata");
    let raw = bundle["projections"]["layer2_hc"].clone();
    assert_eq!(raw["schema_version"].as_u64(), Some(1));
    assert_eq!(
        raw["source"], pinned["projections"]["layer2_hc"]["source"],
        "layer2_hc source metadata"
    );
    assert_eq!(
        raw["source"]["complete_capture_sha256"], bundle["source"]["complete_capture_sha256"],
        "layer2_hc capture"
    );
    let fixture: HcFixture = serde_json::from_value(raw).expect("bundled layer-two HC");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.block_config.copies, 2);
    assert_eq!(fixture.block_config.hc_sinkhorn_iters, 20);
    assert_eq!(fixture.block_config.norm_eps.to_bits(), 1e-20_f32.to_bits());
    assert_eq!(fixture.block_config.hc_eps.to_bits(), 1e-6_f32.to_bits());
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

pub(super) type BlockEntries = [(usize, Vec<u16>, Vec<f32>)];

fn native_inputs(fixture: &HcFixture, entries: Option<&BlockEntries>) -> Vec<(usize, Vec<u16>)> {
    let call_count = entries.map_or(fixture.cases.len(), <[_]>::len);
    assert!(
        (1..=fixture.cases.len()).contains(&call_count),
        "native layer-two input call prefix"
    );
    let norm = fixture.block_parameters["layers.2.attn_norm.weight"].bf16();
    fixture
        .cases
        .iter()
        .take(call_count)
        .enumerate()
        .map(|(index, case)| {
            let captured_residual = case.residual.bf16();
            let captured_incoming = case.incoming_pre.fp32();
            let (residual, incoming) = if let Some(entries) = entries {
                let (start, residual, incoming) = &entries[index];
                assert_eq!(*start, case.start_pos);
                assert_eq!(
                    residual, &captured_residual,
                    "native layer-one terminal residual at layer-two boundary"
                );
                assert_eq!(incoming.len(), captured_incoming.len());
                assert!(incoming.iter().all(|value| value.is_finite()));
                (residual.as_slice(), incoming.as_slice())
            } else {
                (captured_residual.as_slice(), captured_incoming.as_slice())
            };
            let input: Vec<_> = residual
                .chunks_exact(256)
                .enumerate()
                .flat_map(|(position, row)| {
                    derive_attention_input(
                        row,
                        &incoming[position * 2..(position + 1) * 2],
                        &norm,
                        fixture.block_config.norm_eps,
                    )
                })
                .collect();
            assert_eq!(
                input,
                case.attention_input.bf16(),
                "native layer-two HC attention input"
            );
            (case.start_pos, input)
        })
        .collect()
}

fn check_coefficients(
    fixture: &HcFixture,
    case: &HcCase,
    position: usize,
    residual: &[u16],
) -> HcCoefficients {
    let projection = fixture.block_parameters["layers.2.hc_attn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.2.hc_attn_scale"]
        .fp32()
        .try_into()
        .unwrap();
    let base = fixture.block_parameters["layers.2.hc_attn_base"].fp32();
    let config = &fixture.block_config;
    let native = project_hc_diagnostics(
        residual,
        &projection,
        &scale,
        &base,
        config.copies,
        config.norm_eps,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .unwrap();
    let spans = hc_projection_bounds::normalized_projection_envelopes(
        residual,
        &projection,
        config.norm_eps,
    )
    .unwrap();
    let source = case.attention_hc_mixes.fp32();
    for ((span, &actual), &expected) in spans
        .iter()
        .zip(native.mixes())
        .zip(&source[position * 8..(position + 1) * 8])
    {
        assert!(
            span.contains(actual) && span.contains(expected),
            "layer-two attention HC projection envelope"
        );
    }
    let bounds = hc_coefficient_bounds::coefficient_envelopes(
        &spans
            .iter()
            .map(|span| [span.lo, span.hi])
            .collect::<Vec<_>>(),
        &scale,
        &base,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .unwrap();
    for (bounds, actual, source, width) in [
        (
            &bounds.pre,
            native.coefficients().pre(),
            case.attention_coefficients.pre.fp32(),
            2,
        ),
        (
            &bounds.post,
            native.coefficients().post(),
            case.attention_coefficients.post.fp32(),
            2,
        ),
        (
            &bounds.comb,
            native.coefficients().comb(),
            case.attention_coefficients.comb.fp32(),
            4,
        ),
    ] {
        for ((span, &actual), &expected) in bounds
            .iter()
            .zip(actual)
            .zip(&source[position * width..(position + 1) * width])
        {
            for value in [actual, expected] {
                assert!(
                    value.is_finite() && span[0] <= f64::from(value) && f64::from(value) <= span[1],
                    "layer-two attention HC coefficient envelope"
                );
            }
        }
    }
    native.coefficients().clone()
}

fn handoffs(
    fixture: &HcFixture,
    outputs: &[(usize, Vec<u16>)],
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    assert!(
        (1..=fixture.cases.len()).contains(&outputs.len()),
        "native layer-two attention handoff prefix"
    );
    outputs
        .iter()
        .map(|(start, attention)| {
            let case = fixture
                .cases
                .iter()
                .find(|case| case.start_pos == *start)
                .expect("native layer-two attention handoff source start");
            let residual = case.residual.bf16();
            assert_eq!(attention.len() * 2, residual.len());
            let mut joined = Vec::new();
            let mut pre = Vec::new();
            for (position, row) in residual.chunks_exact(256).enumerate() {
                let coefficients = check_coefficients(fixture, case, position, row);
                let mut output = vec![0; 256];
                hc_post_bf16_reference(
                    &attention[position * 128..(position + 1) * 128],
                    row,
                    coefficients.post(),
                    coefficients.comb(),
                    &mut output,
                )
                .unwrap();
                joined.extend(output);
                pre.extend_from_slice(coefficients.pre());
            }
            assert!(
                joined == case.after_attention_residual.bf16(),
                "native layer-two attention post-mix residual at start {}",
                case.start_pos
            );
            assert_eq!(
                case.attention_pre.fp32(),
                case.attention_coefficients.pre.fp32()
            );
            (case.start_pos, joined, pre)
        })
        .collect()
}

fn through_final_suffix(layer_two: Vec<(usize, Vec<u16>, Vec<f32>)>) {
    let streams: Vec<_> = layer_two
        .iter()
        .map(|(start, residual, _)| (*start, residual.clone()))
        .collect();
    let engram = engram_capture::native_layer_three_block_entries_from_streams(Some(&streams));
    let pre: Vec<_> = layer_two
        .into_iter()
        .map(|(start, _, pre)| (start, pre))
        .collect();
    let third = native_layer_three_block_tail_from_entries(
        &layer_three_fixture(),
        Some(&engram),
        Some(&pre),
    );
    let fourth = fixture();
    let output =
        block_tail_from_entries(&fourth, BlockControl::NativeAttention, true, Some(&third));
    assert_final_suffix(&fourth, &output);
}

pub(super) fn native_layer_two_from_entries(entries: &BlockEntries) {
    let layer_two = native_layer_two_entries_from_entries(entries);
    through_final_suffix(layer_two);
}

/// Continues the supplied native layer-one handoff through layer two, retaining
/// the resulting residual/pre-mix pair for the layer-three producer.
pub(super) fn native_layer_two_entries_from_entries(
    entries: &BlockEntries,
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    let fixture = hc_fixture();
    let inputs = native_inputs(&fixture, Some(entries));
    let outputs = layer2_attention_capture::native_outputs_from_inputs(&inputs);
    let attention = handoffs(&fixture, &outputs);
    layer2_ffn::native_layer_two_entries_from_attention(Some(&attention))
}

/// Test-private continuation that keeps L2 attention/window state across the
/// live L1 partitions. FFN remains the existing exact source boundary.
pub(super) struct NativeLayerTwoSession {
    fixture: HcFixture,
    ffn: super::LayerTwoFixture,
    attention: layer2_attention_capture::NativeLayerTwoAttentionSession,
    next_case: usize,
}

impl NativeLayerTwoSession {
    /// Starts the persistent L2 join from the caller's unified bundle.  HC,
    /// attention, and FFN each retain their own validated projection.
    pub(super) fn from_bundle(bundle: &Value) -> Self {
        Self {
            fixture: hc_fixture_from_bundle(bundle),
            ffn: layer2_ffn::fixture_from_bundle(bundle),
            attention: layer2_attention_capture::NativeLayerTwoAttentionSession::from_bundle(
                bundle,
            ),
            next_case: 0,
        }
    }

    pub(super) fn step(
        &mut self,
        (start, residual, incoming): &(usize, Vec<u16>, Vec<f32>),
        live_owner: Option<&layer1_owner_capture::NativeCase>,
    ) -> (usize, Vec<u16>, Vec<f32>) {
        let case = &self.fixture.cases[self.next_case];
        assert_eq!(*start, case.start_pos, "native L2 live start");
        assert_eq!(
            residual,
            &case.residual.bf16(),
            "native L1 terminal residual at layer-two boundary"
        );
        let norm = self.fixture.block_parameters["layers.2.attn_norm.weight"].bf16();
        let input = residual
            .chunks_exact(256)
            .enumerate()
            .flat_map(|(position, row)| {
                derive_attention_input(
                    row,
                    &incoming[position * 2..(position + 1) * 2],
                    &norm,
                    self.fixture.block_config.norm_eps,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            input,
            case.attention_input.bf16(),
            "native layer-two HC attention input"
        );
        let output = self
            .attention
            .step_with_publication(&(*start, input), live_owner);
        let handoff = handoffs(&self.fixture, &[output]);
        let result = layer2_ffn::native_layer_two_entries_from_attention_with_fixture(
            &self.ffn,
            Some(&handoff),
        );
        self.next_case += 1;
        result.into_iter().next().expect("one native L2 result")
    }
}

#[test]
fn native_layer_two_attention_hc_ffn_reaches_final_logits() {
    let fixture = hc_fixture();
    let inputs = native_inputs(&fixture, None);
    let outputs = layer2_attention_capture::native_outputs_from_inputs(&inputs);
    let attention = handoffs(&fixture, &outputs);
    through_final_suffix(layer2_ffn::native_layer_two_entries_from_attention(Some(
        &attention,
    )));
}

#[test]
#[should_panic(expected = "native layer-two attention post-mix residual")]
fn discarded_native_attention_fails_before_ffn() {
    let fixture = hc_fixture();
    let inputs = native_inputs(&fixture, None);
    let mut outputs = layer2_attention_capture::native_outputs_from_inputs(&inputs);
    outputs[0].1.fill(0);
    handoffs(&fixture, &outputs);
}

proptest::proptest! {
    #![proptest_config(proptest::test_runner::Config::with_cases(16))]

    #[test]
    fn dropping_selected_trace_token_attention_is_rejected(mask in 1_u8..128) {
        let fixture = hc_fixture();
        let inputs = native_inputs(&fixture, None);
        let mut outputs = layer2_attention_capture::native_outputs_from_inputs(&inputs);
        for token in 0_usize..7 {
            if mask & (1 << token) != 0 {
                let (call, position) = if token < 5 { (0, token) } else { (token - 4, 0) };
                outputs[call].1[position * 128..(position + 1) * 128].fill(0);
            }
        }
        proptest::prop_assert!(std::panic::catch_unwind(|| handoffs(&fixture, &outputs)).is_err());
    }
}
