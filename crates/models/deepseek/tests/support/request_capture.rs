//! Source-only operands and final-output gates for the reduced request runner.
//!
//! This module is nested under `forward_moe`, so it may reuse that test
//! binary's fixture types while keeping JSON decoding out of the library.

use std::collections::BTreeMap;

use deepseek::indexer::{key::IndexKeyRotaryExecution, query::IndexScoreExecution};
use deepseek::reduced::FinalHeadExecution;

use super::{HeadFixture, attention_capture};
use serde_json::Value;

#[path = "request_alternate.rs"]
mod request_alternate;
#[path = "request_tail.rs"]
mod request_tail;

fn raw_canonical_bundle() -> Value {
    let raw = include_str!("../../../../../fixtures/deepseek-v41/reduced-runner-reference.json");
    let root: Value = serde_json::from_str(raw).expect("canonical reduced request bundle JSON");
    assert_eq!(root["schema_version"].as_u64(), Some(1));
    assert_eq!(root["trace"]["starts"], serde_json::json!([0, 5, 6]));
    root
}

pub(super) fn canonical_trace() -> Vec<Vec<i64>> {
    let root = raw_canonical_bundle();
    root["trace"]["input_ids"]
        .as_array()
        .expect("canonical input ID batches")
        .iter()
        .map(|chunk| {
            chunk
                .as_array()
                .expect("canonical ID chunk")
                .iter()
                .map(|id| id.as_i64().expect("canonical signed ID"))
                .collect()
        })
        .collect()
}

/// Validates that the bundle carries all request operands. This is a
/// fail-closed adapter precondition: no missing projection may silently fall
/// back to captured intermediate rows.
pub(super) fn canonical_projections() -> BTreeMap<String, Value> {
    let root = raw_canonical_bundle();
    let projections = root["projections"]
        .as_object()
        .expect("canonical projections");
    let names = [
        "head",
        "layer0_to_layer1",
        "layer1_attention",
        "layer1_engram",
        "layer1_owner",
        "layer1_tail",
        "layer2_attention",
        "layer2_ffn",
        "layer2_hc",
        "layer3_attention",
        "layer3_candidate",
        "layer3_compressor",
        "layer3_engram",
        "layer3_index_key",
        "layer3_moe",
        "layer4_attention",
        "layer4_moe",
    ];
    names
        .into_iter()
        .map(|name| {
            let projection = projections
                .get(name)
                .unwrap_or_else(|| panic!("missing canonical {name} projection"));
            assert_eq!(
                projection["schema_version"].as_u64(),
                Some(1),
                "{name} schema"
            );
            (name.to_owned(), projection.clone())
        })
        .collect()
}

/// Returns canonical head source data only as an expected-output oracle.
pub(super) fn canonical_head_oracle() -> HeadFixture {
    let projections = canonical_projections();
    serde_json::from_value(projections["head"].clone()).expect("canonical head projection")
}

/// Owned source operands for one compressed-attention layer. `weights` keeps
/// the decoded storage alive for the borrowed runtime definition returned by
/// its caller.
pub(super) struct AttentionOperands {
    pub(super) weights: attention_capture::EncodedWeights,
    pub(super) frequencies: Vec<deepseek::RotaryFrequency>,
}

pub(super) fn canonical_attention(layer: usize) -> AttentionOperands {
    assert!((1..=4).contains(&layer));
    let projections = canonical_projections();
    let name = format!("layer{layer}_attention");
    let fixture: attention_capture::Fixture =
        serde_json::from_value(projections[&name].clone()).expect("canonical attention projection");
    let weights = attention_capture::weights_for_layer(&fixture.encoded_parameters, layer);
    let frequencies = attention_capture::frequencies(&fixture);
    AttentionOperands {
        weights,
        frequencies,
    }
}

fn tail_definition(projection: &Value) -> request_tail::TailDefinition {
    serde_json::from_value(projection.clone()).expect("canonical tail definition")
}

/// Builds and executes the canonical source partition through the production
/// request composition. All closures retain decoded operands until the request
/// has finished; no fixture observation is supplied as an intermediate input.
fn with_canonical_request_model(
    corrupt_l4: bool,
    execution: IndexScoreExecution,
    key_rotary: IndexKeyRotaryExecution,
    head_execution: FinalHeadExecution,
    body: impl FnOnce(&deepseek::reduced::RequestModel<'_>, &super::Fixture, &HeadFixture, &[i64]),
) {
    let mut bundle = raw_canonical_bundle();
    if corrupt_l4 {
        use sha2::{Digest, Sha256};
        let norm = &mut bundle["projections"]["layer4_attention"]["encoded_parameters"]["layers.4.attn.q_norm.weight"];
        norm["shape"] = serde_json::json!([0]);
        norm["numel"] = serde_json::json!(0);
        norm["storage_hex"] = serde_json::json!("");
        norm["storage_sha256"] = serde_json::json!(format!("{:x}", Sha256::digest([])));
    }
    let projections = canonical_projections();
    let l1_tail = tail_definition(&projections["layer1_tail"]);
    let mut l2_tail = tail_definition(&projections["layer2_ffn"]);
    let l2_hc: BTreeMap<String, super::Tensor> =
        serde_json::from_value(projections["layer2_hc"]["block_parameters"].clone()).unwrap();
    for (name, tensor) in l2_hc {
        assert!(l2_tail.block_parameters.insert(name, tensor).is_none());
    }
    let l3_tail = tail_definition(&projections["layer3_moe"]);
    let l4_definition = tail_definition(&projections["layer4_moe"]);
    let l4_tail: super::Fixture =
        serde_json::from_value(projections["layer4_moe"].clone()).unwrap();
    let a1 = canonical_attention(1);
    let a2 = canonical_attention(2);
    let head = canonical_head_oracle();
    let trace = canonical_trace();
    let n1 = l1_tail.block_parameters["layers.1.attn_norm.weight"].bf16();
    let n2 = l2_tail.block_parameters["layers.2.attn_norm.weight"].bf16();
    let n3 = l3_tail.block_parameters["layers.3.attn_norm.weight"].bf16();
    let n4 = l4_definition.block_parameters["layers.4.attn_norm.weight"].bf16();
    assert_eq!(trace, vec![vec![0, 1, 2, 3, 4, 5, 6]]);

    // The operand lenders are deliberately nested: each owns vectors borrowed
    // by exactly one immutable RequestModel.
    super::layer_zero::with_runtime_startup_definition(
        &projections["layer0_to_layer1"],
        |startup| {
            request_tail::with_tail(&l1_tail, 1, |tail1| {
                request_tail::with_tail(&l2_tail, 2, |tail2| {
                    request_tail::with_tail(&l3_tail, 3, |tail3| {
                        request_tail::with_tail(&l4_definition, 4, |tail4| {
                            super::layer1_owner_capture::with_bundle_runtime_owner_operands(
                                &bundle,
                                0,
                                |owner| {
                                    super::owner_attention_capture::with_bundle_runtime_l3_l4_operands(&bundle, |layer_three, layer_four, frequencies| {
                                assert_eq!(a1.frequencies, frequencies, "L1 RoPE table is request-wide");
                                assert_eq!(a2.frequencies, frequencies, "L2 RoPE table is request-wide");
                                let e1 = super::layer_zero::runtime_engram::definition(&projections["layer1_engram"], 1);
                                let e3 = super::layer_zero::runtime_engram::definition(&projections["layer3_engram"], 3);
                                let l1 = deepseek::reduced::LayerOneDefinition::new(
                                    deepseek::reduced::LayerOneConfig::new(
                                        owner.layout, super::layer1_attention_capture::layout(&projections["layer1_attention"]),
                                        std::num::NonZeroUsize::new(
                                            usize::try_from(projections["layer1_owner"]["model"]["index_topk"].as_u64().unwrap()).unwrap(),
                                        ).unwrap(),
                                    ).unwrap(),
                                    owner.compressor_norm, owner.owner_weights,
                                    deepseek::indexer::query::CandidateQueryWeights {
                                        wq_a: owner.query_wq_a.unwrap_or(a1.weights.borrowed().wq_a),
                                        q_norm: owner.query_norm.unwrap_or(a1.weights.borrowed().q_norm),
                                        index: owner.index_weights,
                                    },
                                    owner.query_layout, a1.weights.borrowed(),
                                );
                                let l2 = deepseek::reduced::ReusedAttentionDefinition::new(
                                    super::layer2_attention_capture::layout(&projections["layer2_attention"]), a2.weights.borrowed(),
                                );
                                let blocks = [
                                    deepseek::reduced::BlockDefinition::new(deepseek::reduced::AttentionInput::new(&n1, 2, l1_tail.block_config.norm_eps).unwrap(), tail1),
                                    deepseek::reduced::BlockDefinition::new(deepseek::reduced::AttentionInput::new(&n2, 2, l2_tail.block_config.norm_eps).unwrap(), tail2),
                                    deepseek::reduced::BlockDefinition::new(deepseek::reduced::AttentionInput::new(&n3, 2, l3_tail.block_config.norm_eps).unwrap(), tail3),
                                    deepseek::reduced::BlockDefinition::new(deepseek::reduced::AttentionInput::new(&n4, 2, l4_definition.block_config.norm_eps).unwrap(), tail4),
                                ];
                                let weights: Vec<f32> = head.weight_fp32_bits.iter().copied().map(f32::from_bits).collect();
                                let final_head = deepseek::reduced::FinalHead::new(
                                    &head.norm_weight_bf16, &weights, head.weight_shape[0], 2,
                                    f32::from_bits(head.norm_epsilon_bits),
                                ).unwrap();
                                let model = deepseek::reduced::RequestModel::new(
                                    startup, blocks,
                                    [deepseek::reduced::EngramDefinition::new(e1.0, e1.1), deepseek::reduced::EngramDefinition::new(e3.0, e3.1)],
                                    l1, l2, layer_three, layer_four, final_head,
                                    frequencies, std::num::NonZeroUsize::new(7).unwrap(),
                                ).expect("canonical request model").with_score_execution(execution).with_key_rotary_execution(key_rotary).with_head_execution(head_execution);
                                body(&model, &l4_tail, &head, &trace[0]);
                            });
                                },
                            );
                        });
                    });
                });
            });
        },
    );
}

fn check_request_recovery(
    model: &deepseek::reduced::RequestModel<'_>,
    fixture: &super::Fixture,
    head: &HeadFixture,
    ids: &[i64],
) {
    use deepseek::reduced::{RequestError, RequestSession};
    let mut request = RequestSession::new(model).unwrap();
    assert_eq!(request.score_execution(), model.score_execution());
    assert_eq!(request.key_rotary_execution(), model.key_rotary_execution());
    assert_eq!(request.head_execution(), model.head_execution());
    let run_canonical = |request: &mut RequestSession<'_>| {
        [
            request.step(&ids[..5]).unwrap(),
            request.step(&ids[5..6]).unwrap(),
            request.step(&ids[6..7]).unwrap(),
        ]
    };
    let outputs = run_canonical(&mut request);
    assert_eq!(request.next_start(), 7);
    assert!(!request.is_poisoned());
    request_tail::assert_source_outputs(fixture, head, &outputs);

    // An admitted error must invalidate every later call, even after a valid
    // prefill published both owner chains and both Engram histories.
    request.restart().unwrap();
    assert_eq!(request.score_execution(), model.score_execution());
    assert_eq!(request.key_rotary_execution(), model.key_rotary_execution());
    assert_eq!(request.head_execution(), model.head_execution());
    request.step(&ids[..5]).unwrap();
    assert!(matches!(
        request.step(&[-1]),
        Err(RequestError::NegativeToken { .. })
    ));
    assert!(request.is_poisoned());
    assert_eq!(request.next_start(), 5);
    assert!(matches!(
        request.step(&ids[5..6]),
        Err(RequestError::Poisoned)
    ));
    request.restart().unwrap();
    assert_eq!(request.score_execution(), model.score_execution());
    assert_eq!(request.key_rotary_execution(), model.key_rotary_execution());
    assert_eq!(request.head_execution(), model.head_execution());
    assert_eq!(request.next_start(), 0);
    let replay = run_canonical(&mut request);
    request_tail::assert_source_outputs(fixture, head, &replay);
    for (original, replayed) in outputs.iter().zip(&replay) {
        assert_eq!(original.residual(), replayed.residual());
        assert_eq!(original.incoming_pre(), replayed.incoming_pre());
        assert_eq!(original.heads(), replayed.heads());
        assert_eq!(
            original.layer_one().publication(),
            replayed.layer_one().publication()
        );
        assert_eq!(
            original.layer_three().publication(),
            replayed.layer_three().publication()
        );
    }
}

fn check_alternate_recovery(
    model: &deepseek::reduced::RequestModel<'_>,
    head: &HeadFixture,
    ids: &[i64],
) {
    use deepseek::reduced::{RequestError, RequestSession};
    let mut request = RequestSession::new(model).unwrap();
    assert_eq!(request.score_execution(), model.score_execution());
    assert_eq!(request.key_rotary_execution(), model.key_rotary_execution());
    assert_eq!(request.head_execution(), model.head_execution());
    let run = |request: &mut RequestSession<'_>| {
        [
            request.step(&ids[..4]).unwrap(),
            request.step(&ids[4..5]).unwrap(),
            request.step(&ids[5..6]).unwrap(),
            request.step(&ids[6..7]).unwrap(),
        ]
    };
    let outputs = run(&mut request);
    request_alternate::assert_source_outputs(&outputs, head);
    request.restart().unwrap();
    assert_eq!(request.score_execution(), model.score_execution());
    assert_eq!(request.key_rotary_execution(), model.key_rotary_execution());
    assert_eq!(request.head_execution(), model.head_execution());
    request.step(&ids[..4]).unwrap();
    request.step(&ids[4..5]).unwrap();
    assert!(matches!(
        request.step(&[-1]),
        Err(RequestError::NegativeToken { .. })
    ));
    assert!(matches!(
        request.step(&ids[5..6]),
        Err(RequestError::Poisoned)
    ));
    request.restart().unwrap();
    assert_eq!(request.score_execution(), model.score_execution());
    assert_eq!(request.key_rotary_execution(), model.key_rotary_execution());
    assert_eq!(request.head_execution(), model.head_execution());
    let replay = run(&mut request);
    request_alternate::assert_source_outputs(&replay, head);
    for (original, replayed) in outputs.iter().zip(&replay) {
        assert_eq!(original.residual(), replayed.residual());
        assert_eq!(original.incoming_pre(), replayed.incoming_pre());
        assert_eq!(original.heads(), replayed.heads());
        assert_eq!(
            original.layer_one().publication(),
            replayed.layer_one().publication()
        );
        assert_eq!(
            original.layer_three().publication(),
            replayed.layer_three().publication()
        );
    }
}

#[test]
fn runtime_request_matches_both_source_schedules_and_restarts() {
    with_canonical_request_model(
        false,
        IndexScoreExecution::Scalar,
        IndexKeyRotaryExecution::Scalar,
        FinalHeadExecution::Scalar,
        |model, fixture, head, ids| {
            check_request_recovery(model, fixture, head, ids);
            check_alternate_recovery(model, head, ids);
        },
    );
}

#[test]
fn late_l4_failure_poison_requires_whole_request_reconstruction() {
    check_late_l4_failure(
        IndexScoreExecution::Scalar,
        IndexKeyRotaryExecution::Scalar,
        FinalHeadExecution::Scalar,
    );
}

fn check_late_l4_failure(
    execution: IndexScoreExecution,
    key_rotary: IndexKeyRotaryExecution,
    head_execution: FinalHeadExecution,
) {
    use deepseek::reduced::{RequestError, RequestSession};
    with_canonical_request_model(
        true,
        execution,
        key_rotary,
        head_execution,
        |model, _, _, ids| {
            let mut request = RequestSession::new(model).unwrap();
            assert_eq!(request.score_execution(), model.score_execution());
            assert_eq!(request.key_rotary_execution(), model.key_rotary_execution());
            assert_eq!(request.head_execution(), model.head_execution());
            assert!(matches!(
                request.step(&ids[..5]),
                Err(RequestError::LayerFour(_))
            ));
            assert_eq!(request.next_start(), 0);
            assert!(request.is_poisoned());
            assert!(matches!(
                request.step(&ids[..5]),
                Err(RequestError::Poisoned)
            ));
            request.restart().unwrap();
            assert_eq!(request.score_execution(), model.score_execution());
            assert_eq!(request.key_rotary_execution(), model.key_rotary_execution());
            assert_eq!(request.head_execution(), model.head_execution());
            assert_eq!(request.next_start(), 0);
            assert!(!request.is_poisoned());
            // The immutable malformed L4 weight stays malformed. Reaching L4 again
            // proves all earlier owners were reconstructed instead of retaining
            // their committed prefill cursors after the first failure.
            assert!(matches!(
                request.step(&ids[..5]),
                Err(RequestError::LayerFour(_))
            ));
        },
    );
}

#[cfg(feature = "metal")]
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one ordered source-bound request and restart comparison"
)]
fn metal_scored_request_matches_both_source_schedules_and_restarts() {
    for (score, key_rotary, head_execution) in [
        (
            IndexScoreExecution::MetalBf16,
            IndexKeyRotaryExecution::Scalar,
            FinalHeadExecution::Scalar,
        ),
        (
            IndexScoreExecution::Scalar,
            IndexKeyRotaryExecution::MetalFp32,
            FinalHeadExecution::Scalar,
        ),
        (
            IndexScoreExecution::MetalBf16,
            IndexKeyRotaryExecution::MetalFp32,
            FinalHeadExecution::Scalar,
        ),
        (
            IndexScoreExecution::Scalar,
            IndexKeyRotaryExecution::Scalar,
            FinalHeadExecution::MetalFp32,
        ),
    ] {
        with_canonical_request_model(
            false,
            score,
            key_rotary,
            head_execution,
            |model, fixture, head, ids| {
                assert_eq!(model.score_execution(), score);
                assert_eq!(model.key_rotary_execution(), key_rotary);
                assert_eq!(model.head_execution(), head_execution);
                let scalar_model = model
                    .clone()
                    .with_score_execution(IndexScoreExecution::Scalar)
                    .with_key_rotary_execution(IndexKeyRotaryExecution::Scalar)
                    .with_head_execution(FinalHeadExecution::Scalar);
                for prefill in [4, 5] {
                    let mut scalar = deepseek::reduced::RequestSession::new(&scalar_model).unwrap();
                    let mut metal = deepseek::reduced::RequestSession::new(model).unwrap();
                    let calls = std::iter::once(&ids[..prefill]).chain(ids[prefill..].chunks(1));
                    for call in calls {
                        let expected = scalar.step(call).unwrap();
                        let actual = metal.step(call).unwrap();
                        if head_execution == FinalHeadExecution::Scalar {
                            assert_eq!(actual.heads(), expected.heads());
                        }
                        assert_eq!(actual.residual(), expected.residual());
                        assert_eq!(actual.layer_three().owner(), expected.layer_three().owner());
                        assert_eq!(
                            actual.layer_one().owner().index_keys(),
                            expected.layer_one().owner().index_keys()
                        );
                        assert_eq!(
                            actual.layer_one().key_prefix(),
                            expected.layer_one().key_prefix()
                        );
                        assert_eq!(
                            actual.layer_one().kv_prefix(),
                            expected.layer_one().kv_prefix()
                        );
                        assert_eq!(actual.layer_one().scored(), expected.layer_one().scored());
                        assert_eq!(
                            actual.layer_one().selected_indices(),
                            expected.layer_one().selected_indices()
                        );
                        assert_eq!(
                            actual.layer_one().publication(),
                            expected.layer_one().publication()
                        );
                        assert_eq!(
                            actual.layer_three().candidate(),
                            expected.layer_three().candidate()
                        );
                        assert_eq!(
                            actual.layer_three().selection(),
                            expected.layer_three().selection()
                        );
                        assert_eq!(
                            actual.layer_three().publication(),
                            expected.layer_three().publication()
                        );
                        assert_eq!(
                            actual.layer_three().key_prefix(),
                            expected.layer_three().key_prefix()
                        );
                        assert_eq!(
                            actual.layer_three().kv_prefix(),
                            expected.layer_three().kv_prefix()
                        );
                        assert_eq!(actual.layer_four().scored(), expected.layer_four().scored());
                        assert_eq!(
                            actual.layer_four().selection(),
                            expected.layer_four().selection()
                        );
                    }
                }
                check_request_recovery(model, fixture, head, ids);
                check_alternate_recovery(model, head, ids);
            },
        );
        check_late_l4_failure(score, key_rotary, head_execution);
    }
}

#[test]
fn exported_numerical_artifact_matches_both_source_schedules() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let exported = std::process::Command::new("python3")
        .arg(root.join("scripts/export_v41_reduced_artifact.py"))
        .arg("--source")
        .arg(root.join("fixtures/deepseek-v41/reduced-runner-reference.json"))
        .args(["--output", "-"])
        .output()
        .expect("Python 3 is required by the repository check runner");
    assert!(
        exported.status.success(),
        "exporter rejected pinned source: {}",
        String::from_utf8_lossy(&exported.stderr)
    );
    let artifact = deepseek::reduced::ReducedArtifact::parse(&exported.stdout)
        .expect("bounded numerical artifact");
    let projections = canonical_projections();
    let fixture: super::Fixture =
        serde_json::from_value(projections["layer4_moe"].clone()).unwrap();
    let head = canonical_head_oracle();
    let ids = &canonical_trace()[0];
    let canonical = artifact.run(ids, 5).expect("artifact canonical request");
    request_tail::assert_source_outputs(&fixture, &head, &canonical);
    let alternate = artifact.run(ids, 4).expect("artifact alternate request");
    request_alternate::assert_source_outputs(&alternate, &head);
    #[cfg(feature = "metal")]
    {
        let canonical_metal = artifact
            .run_with_score_execution(ids, 5, IndexScoreExecution::MetalBf16)
            .expect("artifact canonical mixed request");
        request_tail::assert_source_outputs(&fixture, &head, &canonical_metal);
        let alternate_metal = artifact
            .run_with_score_execution(ids, 4, IndexScoreExecution::MetalBf16)
            .expect("artifact alternate mixed request");
        request_alternate::assert_source_outputs(&alternate_metal, &head);
    }
    // Endpoint stability is a separate property from per-call source qualification.
    let terminal_bits = |calls: &[deepseek::reduced::RequestStepOutput]| {
        calls
            .last()
            .unwrap()
            .heads()
            .last()
            .unwrap()
            .logits()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    for prefill in [2, 3, 6] {
        let calls = artifact.run(ids, prefill).expect("bounded prefill sweep");
        assert_eq!(calls.len(), ids.len() - prefill + 1);
        assert_eq!(terminal_bits(&calls), terminal_bits(&canonical));
    }

    // This otherwise-valid schedule reaches a reachable L4 cutoff tie. Keep
    // the qualification policy fail-closed instead of inventing source Top-K order.
    assert!(matches!(
        artifact.run(ids, 7),
        Err(deepseek::reduced::ArtifactError::Request(
            deepseek::reduced::RequestError::LayerFour(
                deepseek::reduced::LayerFourSessionError::Selection(
                    deepseek::indexer::selection::SelectionAdapterError::FinalSelection(
                        deepseek::selection::SelectionError::AmbiguousCutoffTie
                    )
                )
            )
        ))
    ));
    assert!(artifact.run(ids, 1).is_err());
    assert!(artifact.run(&[-1, 0], 2).is_err());
    assert!(artifact.run(&[0, 8], 2).is_err());

    let mut malformed: Value = serde_json::from_slice(&exported.stdout).unwrap();
    malformed["cases"] = serde_json::json!([]);
    assert!(
        deepseek::reduced::ReducedArtifact::parse(&serde_json::to_vec(&malformed).unwrap())
            .is_err()
    );
    malformed.as_object_mut().unwrap().remove("cases");
    malformed["tensors"]
        .as_object_mut()
        .unwrap()
        .remove("head.weight");
    assert!(
        deepseek::reduced::ReducedArtifact::parse(&serde_json::to_vec(&malformed).unwrap())
            .is_err()
    );
}
