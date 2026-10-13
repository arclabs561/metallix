//! Pager integration fixtures use native I8 packed E2M1 and E8M0 tensor
//! layouts, with explicit synthetic code patterns. Scalar expert arithmetic is
//! the independent oracle; these tests qualify routing/storage, not Metal.
use std::{
    collections::BTreeMap,
    num::NonZeroU16,
    ops::Range,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use super::*;
use ::checkpoint::{TensorInfo, TensorSource};

const HIDDEN: usize = 32;
const INTERMEDIATE: usize = 64;
const EXPERT_BYTES: u64 = 3 * (HIDDEN * INTERMEDIATE / 2 + HIDDEN * INTERMEDIATE / 32) as u64;

struct Source {
    tensors: BTreeMap<String, (TensorInfo, Vec<u8>)>,
    reads: AtomicUsize,
    seen: Mutex<Vec<ExpertRange>>,
    fail: Option<String>,
}

impl Source {
    fn new() -> Self {
        let mut tensors = BTreeMap::new();
        for layer in 0..2 {
            for expert in 0..3 {
                for (projection, rows, columns, pattern) in [
                    ("w1", INTERMEDIATE, HIDDEN, 0x12_u8),
                    ("w2", HIDDEN, INTERMEDIATE, 0x93_u8),
                    ("w3", INTERMEDIATE, HIDDEN, 0x21_u8),
                ] {
                    for (part, dtype, divisor) in
                        [("weight", DType::I8, 2), ("scale", DType::F8E8M0, 32)]
                    {
                        let bytes = if part == "weight" {
                            // Distinct signs/codes across experts, layers and rows.
                            (0..rows * columns / divisor)
                                .map(|i| {
                                    pattern
                                        ^ u8::try_from((i + expert * 3 + layer * 5) % 8).unwrap()
                                })
                                .collect()
                        } else {
                            let projection_scale = match projection {
                                "w1" => 0,
                                "w2" => 1,
                                _ => 2,
                            };
                            (0..rows * columns / divisor)
                                .map(|i| {
                                    122 + projection_scale
                                        + u8::try_from((i + expert + layer) % 3).unwrap()
                                })
                                .collect()
                        };
                        let name =
                            format!("layers.{layer}.ffn.experts.{expert}.{projection}.{part}");
                        let info = TensorInfo::new(
                            TensorEncoding::Plain(dtype),
                            vec![rows as u64, (columns / divisor) as u64],
                            97..97 + (rows * columns / divisor) as u64,
                        );
                        tensors.insert(name, (info, bytes));
                    }
                }
            }
        }
        Self {
            tensors,
            reads: AtomicUsize::new(0),
            seen: Mutex::new(Vec::new()),
            fail: None,
        }
    }

    fn weights(&self, key: ExpertKey) -> Fp4ExpertWeights<'_> {
        let part = |projection, kind| {
            &self.tensors[&format!(
                "layers.{}.ffn.experts.{}.{projection}.{kind}",
                key.layer, key.expert
            )]
                .1[..]
        };
        // Deliberately assemble the independent non-paged leaf by projection name,
        // rather than consuming locator output or the adapter's six-part array.
        Fp4ExpertWeights::new(
            HIDDEN,
            INTERMEDIATE,
            part("w1", "weight"),
            part("w1", "scale"),
            part("w2", "weight"),
            part("w2", "scale"),
            part("w3", "weight"),
            part("w3", "scale"),
        )
        .unwrap()
    }

    fn reference(&self, key: ExpertKey) -> Vec<u16> {
        self.weights(key)
            .forward_token(&input(), 0.0, Some(0.7))
            .unwrap()
    }
}

impl TensorSource for Source {
    fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }
    fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|(info, _)| info)
    }
    fn read(&self, name: &str, _max_bytes: u64) -> Result<Vec<u8>, CheckpointError> {
        Err(CheckpointError::Range(format!(
            "unexpected whole read {name}"
        )))
    }
    fn read_range(
        &self,
        name: &str,
        range: Range<u64>,
        dst: &mut [u8],
    ) -> Result<(), CheckpointError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if self.fail.as_deref() == Some(name) {
            return Err(CheckpointError::Range(name.to_owned()));
        }
        let (info, bytes) = self
            .tensors
            .get(name)
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))?;
        // The V4.1 format stores each part as its own whole tensor. This also
        // kills a locator mutant shifting a range without changing its length.
        if range != info.range() || dst.len() != bytes.len() {
            return Err(CheckpointError::Range(name.to_owned()));
        }
        self.seen.lock().unwrap().push(ExpertRange {
            tensor: name.to_owned(),
            range,
        });
        dst.copy_from_slice(bytes);
        Ok(())
    }
}

fn input() -> Vec<u16> {
    (0..HIDDEN)
        .map(|i| {
            crate::precision::f32_to_bf16_rne((f32::from(u16::try_from(i).unwrap()) - 12.0) / 17.0)
        })
        .collect()
}

fn locator(source: Arc<Source>) -> V41ExpertLocator<Source> {
    V41ExpertLocator::new(
        source,
        NonZeroU16::new(2).unwrap(),
        NonZeroU16::new(3).unwrap(),
        HIDDEN,
        INTERMEDIATE,
    )
    .unwrap()
}

#[test]
fn six_native_ranges_preserve_pairing_and_nonzero_offsets() {
    let source = Arc::new(Source::new());
    let locator = locator(Arc::clone(&source));
    assert_eq!(locator.expert_bytes(), EXPERT_BYTES);
    let ranges = locator
        .ranges(ExpertKey {
            layer: 1,
            expert: 2,
        })
        .unwrap();
    assert_eq!(
        ranges.iter().map(|r| r.tensor.as_str()).collect::<Vec<_>>(),
        [
            "layers.1.ffn.experts.2.w1.weight",
            "layers.1.ffn.experts.2.w1.scale",
            "layers.1.ffn.experts.2.w2.weight",
            "layers.1.ffn.experts.2.w2.scale",
            "layers.1.ffn.experts.2.w3.weight",
            "layers.1.ffn.experts.2.w3.scale",
        ]
    );
    assert!(ranges.iter().all(|r| r.range.start == 97));
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
}

#[test]
fn eviction_reload_matches_scalar_with_exact_read_and_callback_counts() {
    for (capacity, expected_reads) in [(1, 36), (4, 24)] {
        let source = Arc::new(Source::new());
        let paged =
            V41PagedExperts::new(locator(Arc::clone(&source)), capacity * EXPERT_BYTES).unwrap();
        let mut callbacks = 0;
        let keys = [(0, 0), (0, 0), (0, 1), (0, 0), (1, 0), (0, 2), (0, 1)];
        let mut distinct = Vec::new();
        for (layer, expert) in keys {
            let expected = source.reference(ExpertKey { layer, expert });
            let actual = paged
                .layer(layer)
                .unwrap()
                .with_expert(usize::from(expert), &mut |weights| {
                    callbacks += 1;
                    weights.forward_token(&input(), 0.0, Some(0.7))
                })
                .unwrap();
            assert_eq!(
                actual, expected,
                "layer {layer} expert {expert} budget {capacity}"
            );
            assert!(actual.iter().any(|&v| v & 0x7fff != 0), "nonzero oracle");
            if !distinct.contains(&actual) {
                distinct.push(actual);
            }
            assert!(paged.cache.lock().unwrap().used_bytes() <= capacity * EXPERT_BYTES);
        }
        assert_eq!(
            distinct.len(),
            4,
            "stale-slot substitutions must be observable"
        );
        assert_eq!(callbacks, keys.len());
        assert_eq!(source.reads.load(Ordering::Relaxed), expected_reads);
    }
}

#[test]
fn invalid_metadata_and_too_small_budget_fail_before_reads() {
    let source = Arc::new(Source::new());
    assert!(matches!(
        V41PagedExperts::new(locator(Arc::clone(&source)), EXPERT_BYTES - 1),
        Err(V41ExpertPagingError::Budget { .. })
    ));
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
    for mutation in 0..3 {
        let mut source = Source::new();
        let (info, _) = source
            .tensors
            .get_mut("layers.0.ffn.experts.0.w3.scale")
            .unwrap();
        *info = match mutation {
            0 => TensorInfo::new(
                TensorEncoding::Plain(DType::U8),
                info.shape().to_vec(),
                info.range(),
            ),
            1 => TensorInfo::new(info.encoding(), vec![32, 2], info.range()),
            _ => TensorInfo::new(info.encoding(), info.shape().to_vec(), 98..info.range().end),
        };
        let source = Arc::new(source);
        let paged = V41PagedExperts::new(locator(Arc::clone(&source)), EXPERT_BYTES).unwrap();
        let result = paged
            .layer(0)
            .unwrap()
            .with_expert(0, &mut |_| panic!("invalid metadata reached callback"));
        assert!(matches!(result, Err(MoEError::ExpertUnavailable { .. })));
        assert_eq!(source.reads.load(Ordering::Relaxed), 0);
        assert!(paged.cache.lock().unwrap().is_empty());
    }
}

#[test]
fn read_failure_does_not_publish_or_call_and_callback_error_is_not_retried() {
    let mut source = Source::new();
    source.fail = Some("layers.0.ffn.experts.0.w2.scale".into());
    let source = Arc::new(source);
    let paged = V41PagedExperts::new(locator(Arc::clone(&source)), EXPERT_BYTES).unwrap();
    assert!(
        paged
            .layer(0)
            .unwrap()
            .with_expert(0, &mut |_| panic!("failed read reached callback"))
            .is_err()
    );
    assert_eq!(
        source.reads.load(Ordering::Relaxed),
        6,
        "pager joins every queued read"
    );
    assert!(paged.cache.lock().unwrap().is_empty());
    let source = Arc::new(Source::new());
    let paged = V41PagedExperts::new(locator(Arc::clone(&source)), EXPERT_BYTES).unwrap();
    let mut callbacks = 0;
    let result = paged.layer(0).unwrap().with_expert(0, &mut |_| {
        callbacks += 1;
        Err(MoEError::InvalidRouteWeight)
    });
    assert!(matches!(result, Err(MoEError::InvalidRouteWeight)));
    assert_eq!(callbacks, 1);
    assert_eq!(source.reads.load(Ordering::Relaxed), 6);
}

#[test]
fn routed_moe_consumer_matches_dense_experts_across_evictions() {
    use crate::moe::{Fp8ExpertWeights, MoEReference};
    let source = Arc::new(Source::new());
    let paged = V41PagedExperts::new(locator(Arc::clone(&source)), EXPERT_BYTES).unwrap();
    let routed: Vec<_> = (0..3)
        .map(|expert| source.weights(ExpertKey { layer: 0, expert }))
        .collect();
    // Explicit zero shared branch isolates storage substitution of selected
    // routed experts; distinct biases select IDs 2 and 1 without cutoff ties.
    let shared_codes = vec![0; HIDDEN * INTERMEDIATE];
    // Shared FP8 uses one scale per 32-by-32 block, unlike the
    // routed FP4 matrices whose scales are per row and 32 columns.
    let shared_up_scales = vec![127; INTERMEDIATE.div_ceil(32) * (HIDDEN / 32)];
    let shared_down_scales = vec![127; HIDDEN.div_ceil(32) * (INTERMEDIATE / 32)];
    let shared = Fp8ExpertWeights::new(
        HIDDEN,
        INTERMEDIATE,
        &shared_codes,
        &shared_up_scales,
        &shared_codes,
        &shared_down_scales,
        &shared_codes,
        &shared_up_scales,
    )
    .unwrap();
    let gate = vec![0; 3 * HIDDEN];
    let reference = MoEReference::new(
        MoEConfig::new(HIDDEN, INTERMEDIATE, 0.0, 2, 1.0, true, 1.0).unwrap(),
        &gate,
        &[0.0, 1.0, 2.0],
        &routed,
        shared,
    )
    .unwrap();
    let expected = reference.forward_token(&input()).unwrap();
    for _ in 0..2 {
        let actual = reference
            .forward_token_with(&input(), &paged.layer(0).unwrap())
            .unwrap();
        assert_eq!(actual.routes(), expected.routes());
        assert_eq!(
            actual.selected_outputs_bf16(),
            expected.selected_outputs_bf16()
        );
        assert_eq!(
            actual
                .accumulator_f32()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .accumulator_f32()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(actual.output_bf16(), expected.output_bf16());
    }
    assert_eq!(
        source.reads.load(Ordering::Relaxed),
        24,
        "two selected experts per call at one-expert capacity"
    );
}

#[test]
fn a_live_callback_survives_eviction_without_holding_cache_mutex() {
    let source = Arc::new(Source::new());
    let paged = V41PagedExperts::new(locator(Arc::clone(&source)), EXPERT_BYTES).unwrap();
    let layer = paged.layer(0).unwrap();
    let actual = layer
        .with_expert(0, &mut |first| {
            layer.with_expert(1, &mut |second| {
                second.forward_token(&input(), 0.0, Some(0.7))
            })?;
            first.forward_token(&input(), 0.0, Some(0.7))
        })
        .unwrap();
    assert_eq!(
        actual,
        source.reference(ExpertKey {
            layer: 0,
            expert: 0
        })
    );
    assert_eq!(source.reads.load(Ordering::Relaxed), 12);
    assert_eq!(paged.cache.lock().unwrap().used_bytes(), EXPERT_BYTES);
}

#[test]
fn scalar_fixture_distinguishes_swapped_gate_and_up_scales() {
    let source = Source::new();
    let part = |projection, kind| {
        &source.tensors[&format!("layers.0.ffn.experts.0.{projection}.{kind}")].1[..]
    };
    let swapped = Fp4ExpertWeights::new(
        HIDDEN,
        INTERMEDIATE,
        part("w1", "weight"),
        part("w3", "scale"),
        part("w2", "weight"),
        part("w2", "scale"),
        part("w3", "weight"),
        part("w1", "scale"),
    )
    .unwrap()
    .forward_token(&input(), 0.0, Some(0.7))
    .unwrap();
    assert_ne!(
        swapped,
        source.reference(ExpertKey {
            layer: 0,
            expert: 0
        }),
        "the paging oracle must detect a scale-pairing mutation"
    );
}
