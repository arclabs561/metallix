//! Layer-one native attention/HC/FFN join into the qualified layer-two suffix.
//!
//! The layer-one block entry remains source-captured. From its derived attention
//! input onward, this uses native owner-backed attention, HC, and FFN operators.

use std::{collections::BTreeMap, num::NonZeroUsize};

use deepseek::{
    ffn::FfnSublayerReference,
    hc::{HcCoefficients, mixing::hc_post_bf16_reference, projection::project_hc_diagnostics},
    indexer::{cache::IndexKeyPublicationId, query::CandidateQueryWeights},
    reduced::{
        LayerOneCall, LayerOneConfig, LayerOneSession, LayerOneStepOutput, PreviousLayerThreeKeys,
    },
};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::{
    BlockConfig, Coefficients, Model, Tensor, derive_attention_input, hc_coefficient_bounds,
    hc_projection_bounds, layer1_attention_capture, layer1_engram_capture, layer1_owner_capture,
    layer2_join, with_model_parameters,
};

const FIXTURE_SHA256: &str = "5f31036c71b797e7195a6b93cf2d656b8744b1e6a80a04dc68007d26e326b89b";

#[derive(Deserialize)]
struct Fixture {
    schema_version: u32,
    model: Model,
    encoded_parameters: BTreeMap<String, Tensor>,
    block_parameters: BTreeMap<String, Tensor>,
    block_config: BlockConfig,
    comparison_policy: ComparisonPolicy,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct ComparisonPolicy {
    route_weight_abs_error_max: f32,
}

#[derive(Deserialize)]
struct Case {
    start_pos: usize,
    residual: Tensor,
    incoming_pre: Tensor,
    attention_input: Tensor,
    attention_output: Tensor,
    after_attention_residual: Tensor,
    attention_pre: Tensor,
    attention_coefficients: Coefficients,
    attention_hc_mixes: Tensor,
    ffn_collapsed: Tensor,
    moe_input: Tensor,
    moe_output: Tensor,
    gate_indices: Tensor,
    gate_weights: Tensor,
    ffn_coefficients: Coefficients,
    ffn_hc_mixes: Tensor,
    output: Tensor,
    next_pre: Tensor,
    layer_two_residual: Tensor,
    layer_two_incoming_pre: Tensor,
}

fn fixture() -> Fixture {
    let raw = include_str!("../../../../../fixtures/deepseek-v41/layer1-tail-reference.json");
    assert_eq!(
        format!("{:x}", Sha256::digest(raw.as_bytes())),
        FIXTURE_SHA256
    );
    let fixture: Fixture = serde_json::from_str(raw).expect("layer-one tail fixture JSON");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.block_config.copies, 2);
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

/// Decodes the L1 tail projection from a unified reduced-runner bundle.
/// Identity stays bound to the checked-in source record; all retained tail
/// arithmetic operands are decoded from the caller's projection.
fn fixture_from_bundle(bundle: &Value) -> Fixture {
    assert_eq!(bundle["schema_version"].as_u64(), Some(1));
    let pinned: Value = serde_json::from_str(include_str!(
        "../../../../../fixtures/deepseek-v41/reduced-runner-reference.json"
    ))
    .expect("pinned reduced bundle metadata");
    assert_eq!(bundle["source"], pinned["source"], "bundle source metadata");
    let raw = bundle["projections"]["layer1_tail"].clone();
    assert_eq!(raw["schema_version"].as_u64(), Some(1));
    assert_eq!(
        raw["source"], pinned["projections"]["layer1_tail"]["source"],
        "layer1_tail source metadata"
    );
    assert_eq!(
        raw["source"]["complete_capture_sha256"], bundle["source"]["complete_capture_sha256"],
        "layer1_tail capture"
    );
    let fixture: Fixture = serde_json::from_value(raw).expect("bundled layer-one tail fixture");
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.block_config.copies, 2);
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

fn native_inputs(fixture: &Fixture, entries: Option<&BlockEntries>) -> Vec<(usize, Vec<u16>)> {
    let call_count = entries.map_or(fixture.cases.len(), <[_]>::len);
    assert!(
        (1..=fixture.cases.len()).contains(&call_count),
        "native layer-one input call prefix"
    );
    let norm = fixture.block_parameters["layers.1.attn_norm.weight"].bf16();
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
                    "native Engram layer-one residual at block boundary"
                );
                assert_eq!(incoming.len(), captured_incoming.len());
                assert!(incoming.iter().all(|value| value.is_finite()));
                (residual.as_slice(), incoming.as_slice())
            } else {
                (captured_residual.as_slice(), captured_incoming.as_slice())
            };
            let input = residual
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
                .collect::<Vec<_>>();
            assert_eq!(
                input,
                case.attention_input.bf16(),
                "native layer-one HC attention input"
            );
            (case.start_pos, input)
        })
        .collect()
}

fn coefficients(
    fixture: &Fixture,
    case: &Case,
    position: usize,
    residual: &[u16],
) -> HcCoefficients {
    let projection = fixture.block_parameters["layers.1.hc_attn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.1.hc_attn_scale"]
        .fp32()
        .try_into()
        .expect("three HC scales");
    let base = fixture.block_parameters["layers.1.hc_attn_base"].fp32();
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
    .expect("native layer-one attention HC");
    let projection_bounds = hc_projection_bounds::normalized_projection_envelopes(
        residual,
        &projection,
        config.norm_eps,
    )
    .expect("source-derived layer-one HC projection bounds");
    let source_mixes = case.attention_hc_mixes.fp32();
    for ((bound, &source), &actual) in projection_bounds
        .iter()
        .zip(&source_mixes[position * 8..(position + 1) * 8])
        .zip(native.mixes())
    {
        assert!(
            bound.contains(source) && bound.contains(actual),
            "layer-one attention HC projection envelope"
        );
    }
    let bounds = hc_coefficient_bounds::coefficient_envelopes(
        &projection_bounds
            .iter()
            .map(|span| [span.lo, span.hi])
            .collect::<Vec<_>>(),
        &scale,
        &base,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    )
    .expect("source-derived layer-one HC coefficient bounds");
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
        for ((bound, &actual), &source) in bounds
            .iter()
            .zip(actual)
            .zip(&source[position * width..(position + 1) * width])
        {
            assert!(
                actual.is_finite()
                    && source.is_finite()
                    && bound[0] <= f64::from(actual)
                    && f64::from(actual) <= bound[1]
                    && bound[0] <= f64::from(source)
                    && f64::from(source) <= bound[1],
                "layer-one attention HC coefficient envelope"
            );
        }
    }
    native.coefficients().clone()
}

fn attention_handoffs(
    fixture: &Fixture,
    outputs: &[(usize, Vec<u16>)],
    supplied_entries: Option<&BlockEntries>,
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    assert!(
        (1..=fixture.cases.len()).contains(&outputs.len()),
        "native layer-one attention handoff prefix"
    );
    outputs
        .iter()
        .map(|(start, attention)| {
            let case = fixture
                .cases
                .iter()
                .find(|case| case.start_pos == *start)
                .expect("native layer-one attention handoff source start");
            assert_eq!(
                attention,
                &case.attention_output.bf16(),
                "native layer-one attention output"
            );
            let captured_residual = case.residual.bf16();
            let residual = if let Some(entries) = supplied_entries {
                let (_, residual, _) = entries
                    .iter()
                    .find(|(entry_start, _, _)| entry_start == start)
                    .expect("supplied native L1 HC residual");
                assert_eq!(
                    residual, &captured_residual,
                    "native L1 HC residual boundary"
                );
                residual.as_slice()
            } else {
                captured_residual.as_slice()
            };
            let mut after_attention = Vec::new();
            let mut pre = Vec::new();
            for (position, row) in residual.chunks_exact(256).enumerate() {
                let coefficients = coefficients(fixture, case, position, row);
                let mut output = vec![0; 256];
                hc_post_bf16_reference(
                    &attention[position * 128..(position + 1) * 128],
                    row,
                    coefficients.post(),
                    coefficients.comb(),
                    &mut output,
                )
                .expect("native layer-one HC post mix");
                after_attention.extend(output);
                pre.extend_from_slice(coefficients.pre());
            }
            assert_eq!(
                after_attention,
                case.after_attention_residual.bf16(),
                "native layer-one attention post-mix residual"
            );
            assert_eq!(
                case.attention_pre.fp32(),
                case.attention_coefficients.pre.fp32(),
                "source layer-one attention pre receipt"
            );
            assert_eq!(pre.len(), case.attention_pre.fp32().len());
            assert!(pre.iter().all(|value| value.is_finite()));
            (case.start_pos, after_attention, pre)
        })
        .collect()
}

fn native_ffn(
    fixture: &Fixture,
    entries: &[(usize, Vec<u16>, Vec<f32>)],
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    assert!(
        (1..=fixture.cases.len()).contains(&entries.len()),
        "native layer-one FFN call prefix"
    );
    let norm = fixture.block_parameters["layers.1.ffn_norm.weight"].bf16();
    let projection = fixture.block_parameters["layers.1.hc_ffn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.1.hc_ffn_scale"]
        .fp32()
        .try_into()
        .expect("three FFN HC scales");
    let base = fixture.block_parameters["layers.1.hc_ffn_base"].fp32();
    with_model_parameters(
        &fixture.model,
        &fixture.encoded_parameters,
        1,
        false,
        |model| {
            let ffn = FfnSublayerReference::new(
                model,
                &norm,
                &projection,
                &scale,
                &base,
                fixture.block_config.copies,
                fixture.block_config.norm_eps,
                fixture.block_config.hc_sinkhorn_iters,
                fixture.block_config.hc_eps,
            )
            .expect("native layer-one FFN");
            entries
                .iter()
                .map(|(start, residual, pre)| {
                    let case = fixture
                        .cases
                        .iter()
                        .find(|case| case.start_pos == *start)
                        .expect("native layer-one FFN source start");
                    assert_eq!(residual, &case.after_attention_residual.bf16());
                    assert_eq!(pre.len(), case.attention_pre.fp32().len());
                    assert!(pre.iter().all(|value| value.is_finite()));
                    let positions = case.after_attention_residual.shape[1];
                    let mut output = Vec::new();
                    let mut next_pre = Vec::new();
                    for position in 0..positions {
                        let result = ffn
                            .forward_token(
                                &residual[position * 256..(position + 1) * 256],
                                &pre[position * 2..(position + 1) * 2],
                            )
                            .expect("native layer-one FFN token");
                        assert_eq!(
                            result.collapsed_bf16(),
                            &case.ffn_collapsed.bf16()[position * 128..(position + 1) * 128]
                        );
                        assert_eq!(
                            result.normalized_bf16(),
                            &case.moe_input.bf16()[position * 128..(position + 1) * 128]
                        );
                        assert_eq!(
                            result.moe().output_bf16(),
                            &case.moe_output.bf16()[position * 128..(position + 1) * 128]
                        );
                        assert_routes(fixture, case, position, result.moe().routes());
                        assert_ffn_envelope(fixture, case, position, &result);
                        output.extend_from_slice(result.output_bf16());
                        next_pre.extend_from_slice(result.coefficients().pre());
                    }
                    assert_eq!(
                        output,
                        case.output.bf16(),
                        "native layer-one terminal residual"
                    );
                    assert_eq!(next_pre.len(), case.next_pre.fp32().len());
                    assert!(next_pre.iter().all(|value| value.is_finite()));
                    assert_eq!(
                        case.next_pre.fp32(),
                        case.layer_two_incoming_pre.fp32(),
                        "source layer-one terminal pre feeds layer two"
                    );
                    assert_eq!(
                        output,
                        case.layer_two_residual.bf16(),
                        "native layer-one feeds layer-two residual"
                    );
                    (case.start_pos, output, next_pre)
                })
                .collect()
        },
    )
}

fn assert_routes(
    fixture: &Fixture,
    case: &Case,
    position: usize,
    routes: &[deepseek::ExpertRoute],
) {
    let ids = case.gate_indices.indices();
    let weights = case.gate_weights.fp32();
    let ids = &ids[position * 2..(position + 1) * 2];
    let weights = &weights[position * 2..(position + 1) * 2];
    assert_eq!(
        routes.len(),
        ids.len(),
        "layer-one native routed expert count"
    );
    let mut expected = ids.to_vec();
    expected.sort_unstable();
    let actual: Vec<_> = routes.iter().map(|route| route.expert_index()).collect();
    assert_eq!(actual, expected, "layer-one native selected expert set");
    for route in routes {
        let index = ids
            .iter()
            .position(|&id| id == route.expert_index())
            .expect("native selected expert is selected by layer-one source");
        assert!(
            (route.weight() - weights[index]).abs()
                <= fixture.comparison_policy.route_weight_abs_error_max,
            "layer-one route weight at position {position} expert {}",
            route.expert_index()
        );
    }
}

fn assert_ffn_envelope(
    fixture: &Fixture,
    case: &Case,
    position: usize,
    result: &deepseek::ffn::FfnDiagnostic,
) {
    let residual = case.after_attention_residual.bf16();
    let residual = &residual[position * 256..(position + 1) * 256];
    let projection = fixture.block_parameters["layers.1.hc_ffn_fn"].fp32();
    let scale: [f32; 3] = fixture.block_parameters["layers.1.hc_ffn_scale"]
        .fp32()
        .try_into()
        .expect("three FFN HC scales");
    let base = fixture.block_parameters["layers.1.hc_ffn_base"].fp32();
    let projection_bounds = hc_projection_bounds::normalized_projection_envelopes(
        residual,
        &projection,
        fixture.block_config.norm_eps,
    )
    .expect("source-derived layer-one FFN projection bounds");
    let observed = project_hc_diagnostics(
        residual,
        &projection,
        &scale,
        &base,
        fixture.block_config.copies,
        fixture.block_config.norm_eps,
        fixture.block_config.hc_sinkhorn_iters,
        fixture.block_config.hc_eps,
    )
    .expect("native layer-one FFN HC");
    assert_eq!(observed.coefficients(), result.coefficients());
    let source_mixes = case.ffn_hc_mixes.fp32();
    let source_mixes = &source_mixes[position * 8..(position + 1) * 8];
    for (index, ((bound, &source), &native)) in projection_bounds
        .iter()
        .zip(source_mixes)
        .zip(observed.mixes())
        .enumerate()
    {
        assert!(bound.contains(source), "layer-one source FFN mix {index}");
        assert!(bound.contains(native), "native layer-one FFN mix {index}");
    }
    let bounds = hc_coefficient_bounds::coefficient_envelopes(
        &projection_bounds
            .iter()
            .map(|span| [span.lo, span.hi])
            .collect::<Vec<_>>(),
        &scale,
        &base,
        fixture.block_config.hc_sinkhorn_iters,
        fixture.block_config.hc_eps,
    )
    .expect("source-derived layer-one FFN coefficient bounds");
    let source_pre = case.ffn_coefficients.pre.fp32();
    let source_pre = &source_pre[position * 2..(position + 1) * 2];
    for (index, ((bound, &source), &native)) in bounds
        .pre
        .iter()
        .zip(source_pre)
        .zip(result.coefficients().pre())
        .enumerate()
    {
        assert!(
            source.is_finite() && bound[0] <= f64::from(source) && f64::from(source) <= bound[1],
            "layer-one source FFN pre {index}"
        );
        assert!(
            native.is_finite() && bound[0] <= f64::from(native) && f64::from(native) <= bound[1],
            "native layer-one FFN pre at layer-two boundary {index}"
        );
    }
}

pub(super) fn native_layer_one_entries() -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    native_layer_one_entries_from_block_entries(None)
}

pub(super) fn native_layer_one_entries_from_block_entries(
    entries: Option<&BlockEntries>,
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    native_layer_one_entries_from_block_entries_with_previous_layer_three_prefix(entries, None)
}

pub(super) fn native_layer_one_entries_from_block_entries_with_previous_layer_three_prefix(
    entries: Option<&BlockEntries>,
    previous_layer_three_prefix: Option<&[u16]>,
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    let fixture = fixture();
    let inputs = native_inputs(&fixture, entries);
    let outputs =
        layer1_attention_capture::native_outputs_from_inputs_with_previous_layer_three_prefix(
            &inputs,
            previous_layer_three_prefix,
        );
    native_ffn(&fixture, &attention_handoffs(&fixture, &outputs, entries))
}

pub(super) fn native_layer_one_entries_from_engram_entries(
    entries: &[(usize, Vec<u16>)],
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    let fixture = fixture();
    let incoming_pre = fixture
        .cases
        .iter()
        .map(|case| (case.start_pos, case.incoming_pre.fp32()))
        .collect::<Vec<_>>();
    native_layer_one_entries_from_engram_entries_with_pre(entries, &incoming_pre)
}

/// Continues layer one from an upstream native Engram residual and its native
/// HC pre-mix state.  The stand-alone overload retains the historical fixture
/// regression; composition callers must use this entry point instead.
pub(super) fn native_layer_one_entries_from_engram_entries_with_pre(
    entries: &[(usize, Vec<u16>)],
    incoming_pre: &[(usize, Vec<f32>)],
) -> Vec<(usize, Vec<u16>, Vec<f32>)> {
    let fixture = fixture();
    assert!(
        (1..=fixture.cases.len()).contains(&entries.len()),
        "native Engram entry prefix count"
    );
    assert_eq!(
        incoming_pre.len(),
        entries.len(),
        "native Engram incoming HC pre prefix count"
    );
    let block_entries = fixture
        .cases
        .iter()
        .take(entries.len())
        .zip(entries)
        .zip(incoming_pre)
        .map(|((case, (start, residual)), (pre_start, pre))| {
            assert_eq!(*start, case.start_pos, "native Engram entry start");
            assert_eq!(
                *pre_start, case.start_pos,
                "native Engram incoming HC pre start"
            );
            assert_eq!(
                pre.len(),
                case.incoming_pre.fp32().len(),
                "native Engram incoming HC pre width"
            );
            assert!(pre.iter().all(|value| value.is_finite()));
            (*start, residual.clone(), pre.clone())
        })
        .collect::<Vec<_>>();
    native_layer_one_entries_from_block_entries(Some(&block_entries))
}

/// Test-private composition of the live L1 request state.  One `step` consumes
/// precisely one source partition through Engram, the ratio-two owner, layer
/// attention, HC, and FFN, returning the residual and HC pre-mix consumed by
/// layer two. The reduced graph carries this session's live publication into
/// layer two and retains Engram3 state across the same calls.
pub(super) struct NativeLayerOneSession {
    fixture: Fixture,
    engram: layer1_engram_capture::NativeLayerOneEngramSession,
    runtime: LayerOneSession,
    operands: RuntimeOperandSource,
    last_publication: Option<layer1_owner_capture::NativeCase>,
    previous_layer_three: Option<PreviousLayerThreePublication>,
    next_case: usize,
}

#[derive(Clone)]
enum RuntimeOperandSource {
    Bundle(Value),
    Alternate { projection: Value, startup: Value },
}

struct PreviousLayerThreePublication {
    publication: IndexKeyPublicationId,
    keys: Vec<u16>,
}

impl NativeLayerOneSession {
    /// Starts the persistent L1 join from the caller's unified bundle. Each
    /// child owns its validated projection for the lifetime of this request.
    pub(super) fn from_bundle(bundle: &Value) -> Self {
        Self {
            fixture: fixture_from_bundle(bundle),
            engram: layer1_engram_capture::NativeLayerOneEngramSession::from_bundle(bundle),
            runtime: runtime_from_bundle(bundle),
            operands: RuntimeOperandSource::Bundle(bundle.clone()),
            last_publication: None,
            previous_layer_three: None,
            next_case: 0,
        }
    }

    pub(super) fn from_alternate(projection: &Value, startup: &Value) -> Self {
        for name in ["source", "source_receipt_sha256", "capture_identity"] {
            assert_eq!(
                projection["tail"][name], projection[name],
                "alternate L1 tail provenance"
            );
            assert_eq!(
                projection[name], startup[name],
                "alternate startup provenance"
            );
        }
        let fixture: Fixture =
            serde_json::from_value(projection["tail"].clone()).expect("alternate L1 tail");
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.block_config.copies, 2);
        assert_eq!(fixture.block_config.norm_eps.to_bits(), 1e-20_f32.to_bits());
        assert_eq!(fixture.block_config.hc_eps.to_bits(), 1e-6_f32.to_bits());
        assert_eq!(
            fixture
                .cases
                .iter()
                .map(|case| case.start_pos)
                .collect::<Vec<_>>(),
            [0, 4, 5, 6]
        );
        Self {
            fixture,
            engram: layer1_engram_capture::NativeLayerOneEngramSession::from_alternate_startup(
                startup,
            ),
            runtime: runtime_from_alternate(projection, startup),
            operands: RuntimeOperandSource::Alternate {
                projection: projection.clone(),
                startup: startup.clone(),
            },
            last_publication: None,
            previous_layer_three: None,
            next_case: 0,
        }
    }

    /// Supplies L3's complete preceding-call publication before a partial
    /// L1 compression group consumes its score keys.
    pub(super) fn supply_previous_layer_three_prefix(
        &mut self,
        publication: IndexKeyPublicationId,
        keys: &[u16],
    ) {
        assert!(
            self.previous_layer_three.is_none(),
            "one prior L3 publication per L1 call"
        );
        self.previous_layer_three = Some(PreviousLayerThreePublication {
            publication,
            keys: keys.to_vec(),
        });
    }

    pub(super) fn step(
        &mut self,
        stream: &(usize, Vec<u16>),
        incoming_pre: &(usize, Vec<f32>),
    ) -> (usize, Vec<u16>, Vec<f32>) {
        let case = &self.fixture.cases[self.next_case];
        assert_eq!(stream.0, case.start_pos, "native L1 stream start");
        assert_eq!(incoming_pre.0, case.start_pos, "native L1 HC pre start");
        assert_eq!(
            incoming_pre.1.len(),
            case.incoming_pre.fp32().len(),
            "native L1 HC pre width"
        );
        assert!(incoming_pre.1.iter().all(|value| value.is_finite()));
        let (start, residual) = self.engram.step(Some(stream));
        let norm = self.fixture.block_parameters["layers.1.attn_norm.weight"].bf16();
        let input = residual
            .chunks_exact(256)
            .enumerate()
            .flat_map(|(position, row)| {
                derive_attention_input(
                    row,
                    &incoming_pre.1[position * 2..(position + 1) * 2],
                    &norm,
                    self.fixture.block_config.norm_eps,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            input,
            case.attention_input.bf16(),
            "native L1 HC attention input"
        );
        assert_eq!(self.runtime.next_start(), start, "runtime L1 request start");
        let prior = self.previous_layer_three.take();
        let output = self.step_runtime(&input, prior.as_ref());
        self.last_publication = Some(layer1_owner_capture::NativeCase {
            start_pos: start,
            latent: output.owner().latent().map(<[u16]>::to_vec),
            key_prefix: output.key_prefix().to_vec(),
            kv_prefix: output.kv_prefix().to_vec(),
            source_score_key_prefix: output.score_key_prefix().to_vec(),
            selected_indices: output.selected_indices().to_vec(),
        });
        match &self.operands {
            RuntimeOperandSource::Bundle(bundle) => {
                layer1_owner_capture::assert_bundle_runtime_step(
                    bundle,
                    self.next_case,
                    &input,
                    &output,
                );
                layer1_attention_capture::assert_bundle_runtime_attention(
                    bundle,
                    self.next_case,
                    &input,
                    &output,
                );
            }
            RuntimeOperandSource::Alternate {
                projection,
                startup,
            } => {
                layer1_owner_capture::assert_alternate_runtime_step(
                    projection,
                    startup,
                    self.next_case,
                    &input,
                    &output,
                );
                layer1_attention_capture::assert_alternate_runtime_attention(
                    projection,
                    self.next_case,
                    &input,
                    &output,
                );
            }
        }
        let attention = (start, output.attention().final_output.clone());
        let native_entry = (start, residual, incoming_pre.1.clone());
        let handoff = attention_handoffs(&self.fixture, &[attention], Some(&[native_entry]));
        let mut output = native_ffn(&self.fixture, &handoff);
        self.next_case += 1;
        output.pop().expect("one native L1 FFN result")
    }

    fn step_runtime(
        &mut self,
        input: &[u16],
        previous: Option<&PreviousLayerThreePublication>,
    ) -> LayerOneStepOutput {
        let source = self.operands.clone();
        match source {
            RuntimeOperandSource::Bundle(bundle) => {
                layer1_owner_capture::with_bundle_runtime_owner_operands(
                    &bundle,
                    self.next_case,
                    |owner| {
                        layer1_attention_capture::with_bundle_runtime_attention_operands(
                            &bundle,
                            |attention| {
                                assert_eq!(
                                    owner.frequencies, attention.frequencies,
                                    "bundle L1 owner and attention frequency table"
                                );
                                let query_weights = CandidateQueryWeights {
                                    wq_a: owner.query_wq_a.unwrap_or(attention.candidate_wq_a),
                                    q_norm: owner.query_norm.unwrap_or(attention.candidate_q_norm),
                                    index: owner.index_weights,
                                };
                                let previous = previous.map(|publication| {
                                    PreviousLayerThreeKeys::new(
                                        publication.publication,
                                        &publication.keys,
                                    )
                                });
                                self.runtime
                                    .step(LayerOneCall::new(
                                        input,
                                        owner.positions,
                                        owner.frequencies,
                                        owner.owner_weights,
                                        query_weights,
                                        owner.query_layout,
                                        attention.weights,
                                        previous,
                                    ))
                                    .expect("live bundle layer-one runtime step")
                            },
                        )
                    },
                )
            }
            RuntimeOperandSource::Alternate {
                projection,
                startup,
            } => layer1_owner_capture::with_alternate_runtime_owner_operands(
                &projection,
                &startup,
                self.next_case,
                |owner| {
                    layer1_attention_capture::with_alternate_runtime_attention_operands(
                        &projection,
                        |attention| {
                            assert_eq!(
                                owner.frequencies, attention.frequencies,
                                "alternate L1 owner and attention frequency table"
                            );
                            let query_weights = CandidateQueryWeights {
                                wq_a: owner.query_wq_a.unwrap_or(attention.candidate_wq_a),
                                q_norm: owner.query_norm.unwrap_or(attention.candidate_q_norm),
                                index: owner.index_weights,
                            };
                            let previous = previous.map(|publication| {
                                PreviousLayerThreeKeys::new(
                                    publication.publication,
                                    &publication.keys,
                                )
                            });
                            self.runtime
                                .step(LayerOneCall::new(
                                    input,
                                    owner.positions,
                                    owner.frequencies,
                                    owner.owner_weights,
                                    query_weights,
                                    owner.query_layout,
                                    attention.weights,
                                    previous,
                                ))
                                .expect("live alternate layer-one runtime step")
                        },
                    )
                },
            ),
        }
    }

    pub(super) fn last_publication(&self) -> &layer1_owner_capture::NativeCase {
        self.last_publication
            .as_ref()
            .expect("live L1 owner publication")
    }
}

fn runtime_from_bundle(bundle: &Value) -> LayerOneSession {
    layer1_owner_capture::with_bundle_runtime_owner_operands(bundle, 0, |owner| {
        layer1_attention_capture::with_bundle_runtime_attention_operands(bundle, |attention| {
            LayerOneSession::new(
                LayerOneConfig::new(
                    owner.layout,
                    attention.layout,
                    NonZeroUsize::new(1).expect("source index topk"),
                )
                .expect("bundle layer-one runtime geometry"),
                owner.compressor_norm,
            )
            .expect("bundle layer-one runtime")
        })
    })
}

fn runtime_from_alternate(projection: &Value, startup: &Value) -> LayerOneSession {
    layer1_owner_capture::with_alternate_runtime_owner_operands(projection, startup, 0, |owner| {
        layer1_attention_capture::with_alternate_runtime_attention_operands(
            projection,
            |attention| {
                LayerOneSession::new(
                    LayerOneConfig::new(
                        owner.layout,
                        attention.layout,
                        NonZeroUsize::new(1).expect("source index topk"),
                    )
                    .expect("alternate layer-one runtime geometry"),
                    owner.compressor_norm,
                )
                .expect("alternate layer-one runtime")
            },
        )
    })
}

#[test]
fn native_layer_one_attention_hc_ffn_reaches_final_logits() {
    let entries = native_layer_one_entries();
    layer2_join::native_layer_two_from_entries(&entries);
}

#[test]
#[should_panic(expected = "native layer-one attention output")]
fn discarded_native_layer_one_attention_fails_before_ffn() {
    let fixture = fixture();
    let inputs = native_inputs(&fixture, None);
    let mut outputs = layer1_attention_capture::native_outputs_from_inputs(&inputs);
    outputs[0].1.fill(0);
    attention_handoffs(&fixture, &outputs, None);
}
