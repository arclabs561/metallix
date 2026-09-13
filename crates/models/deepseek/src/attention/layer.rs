//! Model-local, scalar layer-attention composition for the pinned V4.1 path.
//!
//! This is deliberately a layer-4-shaped adapter, rather than a cache scheduler:
//! it owns one numerical BF16 window ring and consumes an explicitly published
//! compressed-KV slice.  The publication is checked against a source-layer,
//! epoch, and successful-call ordinal before any state is changed.

use std::{collections::HashSet, num::NonZeroUsize};

use thiserror::Error;

use crate::{
    RotaryDirection, RotaryError, RotaryFrequency, RotaryTailLayout,
    precision::{
        ActivationGroup, ActivationQuantError, ActivationRoundtripError, Fp8LinearError,
        f32_to_bf16_rne, fp8_linear_runtime_f32, quantize_bf16_activations_e4m3fn,
        requantize_bf16_activations_e4m3fn,
    },
    rms_norm_bf16_reference, rotate_tail,
};

use super::{
    AttentionOutputError, AttentionOutputLayout, AttentionOutputLayoutError,
    SparseAttentionBf16Error, SparseAttentionError, SparseAttentionLayout,
    attention_output_reference, sparse_attention_bf16_reference,
    window::{WindowError, WindowStep, window_topk_indices, write_window_kv_bf16},
};

const MAX_LAYER_ATTENTION_ELEMENTS: usize = 1 << 20;

/// The model-local dimensions and provenance contract for one layer attention.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerAttentionLayout {
    batches: NonZeroUsize,
    hidden_dimension: NonZeroUsize,
    heads: NonZeroUsize,
    head_dimension: NonZeroUsize,
    rope_pairs: NonZeroUsize,
    q_rank: NonZeroUsize,
    window: NonZeroUsize,
    groups: NonZeroUsize,
    output_rank: NonZeroUsize,
    expected_source_layer: u16,
    compressed_ratio: NonZeroUsize,
    norm_epsilon_bits: u32,
    softmax_scale_bits: u32,
}

impl LayerAttentionLayout {
    /// Creates a bounded source-shaped layout.
    ///
    /// `expected_source_layer` identifies the only producer whose borrowed
    /// compressed numerical values may be used. `compressed_ratio` controls
    /// the per-query causal limit during a prefill (a value of one is the
    /// pinned initial layer-4 arrangement).
    #[allow(
        clippy::too_many_arguments,
        reason = "source dimensions stay explicit at the adapter boundary"
    )]
    pub fn new(
        batches: NonZeroUsize,
        hidden_dimension: NonZeroUsize,
        heads: NonZeroUsize,
        head_dimension: NonZeroUsize,
        rope_pairs: NonZeroUsize,
        q_rank: NonZeroUsize,
        window: NonZeroUsize,
        groups: NonZeroUsize,
        output_rank: NonZeroUsize,
        expected_source_layer: u16,
        compressed_ratio: NonZeroUsize,
        norm_epsilon: f32,
        softmax_scale: f32,
    ) -> Result<Self, LayerAttentionLayoutError> {
        if !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
            return Err(LayerAttentionLayoutError::InvalidNormEpsilon);
        }
        if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
            return Err(LayerAttentionLayoutError::InvalidSoftmaxScale);
        }
        let rope_width =
            rope_pairs
                .get()
                .checked_mul(2)
                .ok_or(LayerAttentionLayoutError::ShapeOverflow {
                    field: "rope width",
                })?;
        if rope_width > head_dimension.get() {
            return Err(LayerAttentionLayoutError::RopeExceedsHead {
                rope_width,
                head_dimension: head_dimension.get(),
            });
        }
        if !heads.get().is_multiple_of(groups.get()) {
            return Err(LayerAttentionLayoutError::HeadsNotGrouped {
                heads: heads.get(),
                groups: groups.get(),
            });
        }
        for (field, width) in [
            ("hidden dimension", hidden_dimension.get()),
            ("q rank", q_rank.get()),
            ("head dimension", head_dimension.get()),
            (
                "groups * output rank",
                checked_product(&[groups.get(), output_rank.get()], "flattened output rank")?,
            ),
        ] {
            if !width.is_multiple_of(32) {
                return Err(LayerAttentionLayoutError::UngroupedFp8Reduction { field, width });
            }
        }
        let ring_elements = checked_product(
            &[batches.get(), window.get(), head_dimension.get()],
            "window ring",
        )?;
        for (field, elements) in [
            ("window ring", ring_elements),
            (
                "one-position input",
                checked_product(
                    &[batches.get(), hidden_dimension.get()],
                    "one-position input",
                )?,
            ),
            (
                "one-position query",
                checked_product(
                    &[batches.get(), heads.get(), head_dimension.get()],
                    "one-position query",
                )?,
            ),
            (
                "one-position q rank",
                checked_product(&[batches.get(), q_rank.get()], "one-position q rank")?,
            ),
        ] {
            if elements > MAX_LAYER_ATTENTION_ELEMENTS {
                return Err(LayerAttentionLayoutError::ElementLimit { field, elements });
            }
        }
        Ok(Self {
            batches,
            hidden_dimension,
            heads,
            head_dimension,
            rope_pairs,
            q_rank,
            window,
            groups,
            output_rank,
            expected_source_layer,
            compressed_ratio,
            norm_epsilon_bits: norm_epsilon.to_bits(),
            softmax_scale_bits: softmax_scale.to_bits(),
        })
    }

    /// The producer layer accepted for compressed numerical publications.
    #[must_use]
    pub const fn expected_source_layer(self) -> u16 {
        self.expected_source_layer
    }
}

/// One borrowed FP8 checkpoint projection, stored in its runtime orientation.
#[derive(Clone, Copy, Debug)]
pub struct Fp8Projection<'a> {
    /// E4M3FN codes `[outputs, reduction]`.
    pub codes: &'a [u8],
    /// E8M0 scales `[ceil(outputs / 32), reduction / 32]`.
    pub scales: &'a [u8],
}

/// Borrowed weights needed by the source-shaped attention path.
#[derive(Clone, Copy, Debug)]
pub struct LayerAttentionWeights<'a> {
    pub wq_a: Fp8Projection<'a>,
    pub q_norm: &'a [u16],
    pub wq_b: Fp8Projection<'a>,
    pub wkv: Fp8Projection<'a>,
    pub kv_norm: &'a [u16],
    pub attn_sink: &'a [f32],
    pub wo_a: &'a [u16],
    pub wo_b: Fp8Projection<'a>,
}

/// A source-published compressed KV view for one attention call.
///
/// The adapter validates producer identity, epoch, call ordinal, exact prefix
/// length, index range, causality, and per-row index uniqueness. It does not
/// establish how the source scored candidates or which sparse slot count its
/// indexer chose.
#[derive(Clone, Copy, Debug)]
pub struct CompressedAttentionPublication<'a> {
    pub source_layer: u16,
    pub epoch: u64,
    pub call_id: u64,
    /// Numerical, already-quantized-and-reconstructed BF16 `[batch, key, head_dim]`.
    pub numerical_bf16: &'a [u16],
    /// Source indices `[batch, query, compressed_slot]`, offset into concatenated KV.
    pub indices: &'a [i32],
}

/// Every fixture-visible intermediate produced by [`LayerAttentionState::forward`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerAttentionDiagnostic {
    pub wq_a: Vec<u16>,
    pub qr: Vec<u16>,
    pub wq_b_pre_rope: Vec<u16>,
    pub q_after_rope: Vec<u16>,
    pub prepared_window: Vec<u16>,
    pub window_read: Vec<u16>,
    pub window_indices: Vec<i32>,
    pub ring_after: Vec<u16>,
    pub sparse_output: Vec<u16>,
    pub final_output: Vec<u16>,
}

#[derive(Debug)]
struct QueryStages {
    wq_a: Vec<u16>,
    qr: Vec<u16>,
    wq_b_pre_rope: Vec<u16>,
    q_after_rope: Vec<u16>,
}

#[derive(Debug)]
struct WindowStages {
    prepared_window: Vec<u16>,
    window_read: Vec<u16>,
    staged_ring: Vec<u16>,
    shared_kv: Vec<u16>,
    window_indices: Vec<i32>,
    indices: Vec<i32>,
    window_keys: usize,
    compressed_keys: usize,
}

#[derive(Clone, Copy)]
struct TailShape {
    batches: NonZeroUsize,
    positions: usize,
    heads: NonZeroUsize,
    head_dimension: NonZeroUsize,
    rope_pairs: NonZeroUsize,
}

/// A single model-local numerical window owner.
#[derive(Clone, Debug)]
pub struct LayerAttentionState {
    layout: LayerAttentionLayout,
    ring: Vec<u16>,
    next_position: Option<usize>,
    epoch: u64,
    next_call_id: u64,
}

impl LayerAttentionState {
    /// Makes an empty state at epoch zero. Its first prefill expects `(epoch=0, call_id=0)`.
    #[must_use]
    pub fn new(layout: LayerAttentionLayout) -> Self {
        let ring_elements =
            layout.batches.get() * layout.window.get() * layout.head_dimension.get();
        Self {
            layout,
            ring: vec![0; ring_elements],
            next_position: None,
            epoch: 0,
            next_call_id: 0,
        }
    }

    /// Invalidates all prior borrowed publications and clears cache continuity.
    pub fn reset(&mut self) -> Result<(), LayerAttentionError> {
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or(LayerAttentionError::EpochOverflow)?;
        self.ring.fill(0);
        self.next_position = None;
        self.next_call_id = 0;
        Ok(())
    }

    /// Computes a source-shaped attention call and atomically publishes its staged ring.
    ///
    /// `frequencies` is the call-local `[positions, rope_pairs]` slice beginning
    /// at `start_position`; callers must not pass a full position table here.
    #[allow(
        clippy::too_many_arguments,
        reason = "the adapter retains source-visible inputs"
    )]
    pub fn forward(
        &mut self,
        attention_input: &[u16],
        start_position: usize,
        frequencies: &[RotaryFrequency],
        weights: LayerAttentionWeights<'_>,
        publication: CompressedAttentionPublication<'_>,
    ) -> Result<LayerAttentionDiagnostic, LayerAttentionError> {
        let positions = self.input_positions(attention_input)?;
        let (step, reset_epoch) = self.validate_transition(start_position, positions)?;
        let expected_epoch = if reset_epoch {
            self.epoch
                .checked_add(1)
                .ok_or(LayerAttentionError::EpochOverflow)?
        } else {
            self.epoch
        };
        let expected_call_id = if start_position == 0 {
            0
        } else {
            self.next_call_id
        };
        self.validate_publication(
            publication,
            expected_epoch,
            expected_call_id,
            start_position,
            positions,
            step,
        )?;

        let query = self.prepare_query(attention_input, positions, frequencies, weights)?;
        let window = self.prepare_window(
            attention_input,
            start_position,
            positions,
            frequencies,
            weights,
            publication,
            step,
        )?;
        let sparse_output = self.sparse_output(&query.q_after_rope, &window, positions, weights)?;
        let output_layout = AttentionOutputLayout::new(
            self.layout.batches.get(),
            positions,
            self.layout.heads.get(),
            self.layout.head_dimension.get(),
            self.layout.rope_pairs.get(),
            self.layout.groups.get(),
            self.layout.output_rank.get(),
            self.layout.hidden_dimension.get(),
        )?;
        let final_output = attention_output_reference(
            &sparse_output,
            frequencies,
            weights.wo_a,
            weights.wo_b.codes,
            weights.wo_b.scales,
            output_layout,
        )?;

        let next_position =
            start_position
                .checked_add(positions)
                .ok_or(LayerAttentionError::ShapeOverflow {
                    field: "next position",
                })?;
        let next_call_id = expected_call_id
            .checked_add(1)
            .ok_or(LayerAttentionError::CallIdOverflow)?;
        // Take the diagnostic snapshot before publishing the staged cache: no
        // fallible computation is intentionally left after this point.
        let ring_after = window.staged_ring.clone();
        self.ring = window.staged_ring;
        self.next_position = Some(next_position);
        self.epoch = expected_epoch;
        self.next_call_id = next_call_id;
        Ok(LayerAttentionDiagnostic {
            wq_a: query.wq_a,
            qr: query.qr,
            wq_b_pre_rope: query.wq_b_pre_rope,
            q_after_rope: query.q_after_rope,
            prepared_window: window.prepared_window,
            window_read: window.window_read,
            window_indices: window.window_indices,
            ring_after,
            sparse_output,
            final_output,
        })
    }

    fn prepare_query(
        &self,
        attention_input: &[u16],
        positions: usize,
        frequencies: &[RotaryFrequency],
        weights: LayerAttentionWeights<'_>,
    ) -> Result<QueryStages, LayerAttentionError> {
        let rows = checked_product(&[self.layout.batches.get(), positions], "input rows")?;
        let wq_a = fp8_project_bf16(
            attention_input,
            rows,
            self.layout.hidden_dimension.get(),
            self.layout.q_rank.get(),
            weights.wq_a,
        )?;
        let qr = rms_norm_rows(
            &wq_a,
            rows,
            self.layout.q_rank.get(),
            weights.q_norm,
            self.layout.norm_epsilon(),
        )?;
        let wq_b_pre_rope = fp8_project_bf16(
            &qr,
            rows,
            self.layout.q_rank.get(),
            self.query_width()?,
            weights.wq_b,
        )?;
        let q_after_rope = rotate_bf16_tail(
            &wq_b_pre_rope,
            TailShape {
                batches: self.layout.batches,
                positions,
                heads: self.layout.heads,
                head_dimension: self.layout.head_dimension,
                rope_pairs: self.layout.rope_pairs,
            },
            frequencies,
            RotaryDirection::Forward,
        )?;
        Ok(QueryStages {
            wq_a,
            qr,
            wq_b_pre_rope,
            q_after_rope,
        })
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "one source-visible cache preparation phase"
    )]
    fn prepare_window(
        &self,
        attention_input: &[u16],
        start_position: usize,
        positions: usize,
        frequencies: &[RotaryFrequency],
        weights: LayerAttentionWeights<'_>,
        publication: CompressedAttentionPublication<'_>,
        step: WindowStep,
    ) -> Result<WindowStages, LayerAttentionError> {
        let rows = checked_product(&[self.layout.batches.get(), positions], "input rows")?;
        let kv = fp8_project_bf16(
            attention_input,
            rows,
            self.layout.hidden_dimension.get(),
            self.layout.head_dimension.get(),
            weights.wkv,
        )?;
        let normalized_kv = rms_norm_rows(
            &kv,
            rows,
            self.layout.head_dimension.get(),
            weights.kv_norm,
            self.layout.norm_epsilon(),
        )?;
        let rotated_kv = rotate_bf16_tail(
            &normalized_kv,
            TailShape {
                batches: self.layout.batches,
                positions,
                heads: NonZeroUsize::MIN,
                head_dimension: self.layout.head_dimension,
                rope_pairs: self.layout.rope_pairs,
            },
            frequencies,
            RotaryDirection::Forward,
        )?;
        let mut prepared_window = vec![0; rotated_kv.len()];
        requantize_bf16_activations_e4m3fn(
            &rotated_kv,
            rows,
            self.layout.head_dimension.get(),
            ActivationGroup::Elements32,
            &mut prepared_window,
        )?;
        let mut staged_ring = if start_position == 0 {
            vec![0; self.ring.len()]
        } else {
            self.ring.clone()
        };
        write_window_kv_bf16(
            step,
            &prepared_window,
            self.layout.batches,
            self.layout.window,
            self.layout.head_dimension,
            &mut staged_ring,
        )?;
        let (window_read, window_keys) = if start_position == 0 {
            (prepared_window.clone(), positions)
        } else {
            (staged_ring.clone(), self.layout.window.get())
        };
        let compressed_keys = self.compressed_keys(publication)?;
        let shared_kv = concatenate_kv(
            &window_read,
            window_keys,
            publication.numerical_bf16,
            compressed_keys,
            self.layout.batches.get(),
            self.layout.head_dimension.get(),
        )?;
        let window_indices = window_topk_indices(step, self.layout.window, self.layout.batches)?;
        let indices = concatenate_indices(
            &window_indices,
            publication.indices,
            self.layout.batches.get(),
            positions,
        )?;
        Ok(WindowStages {
            prepared_window,
            window_read,
            staged_ring,
            shared_kv,
            window_indices,
            indices,
            window_keys,
            compressed_keys,
        })
    }

    fn sparse_output(
        &self,
        query: &[u16],
        window: &WindowStages,
        positions: usize,
        weights: LayerAttentionWeights<'_>,
    ) -> Result<Vec<u16>, LayerAttentionError> {
        let sparse_slots =
            NonZeroUsize::new(window.indices.len() / (self.layout.batches.get() * positions))
                .ok_or(LayerAttentionError::NoSparseSlots)?;
        let key_positions = NonZeroUsize::new(
            window
                .window_keys
                .checked_add(window.compressed_keys)
                .ok_or(LayerAttentionError::ShapeOverflow {
                    field: "key positions",
                })?,
        )
        .ok_or(LayerAttentionError::NoKeys)?;
        let sparse_layout = SparseAttentionLayout::new(
            self.layout.batches,
            NonZeroUsize::new(positions).expect("checked nonzero positions"),
            self.layout.heads,
            self.layout.head_dimension,
            key_positions,
            sparse_slots,
        )?;
        Ok(sparse_attention_bf16_reference(
            query,
            &window.shared_kv,
            weights.attn_sink,
            &window.indices,
            self.layout.softmax_scale(),
            sparse_layout,
        )?)
    }

    fn input_positions(&self, input: &[u16]) -> Result<usize, LayerAttentionError> {
        let stride = checked_product(
            &[
                self.layout.batches.get(),
                self.layout.hidden_dimension.get(),
            ],
            "input stride",
        )?;
        if input.is_empty() || !input.len().is_multiple_of(stride) {
            return Err(LayerAttentionError::InputLength {
                actual: input.len(),
                stride,
            });
        }
        if input.len() > MAX_LAYER_ATTENTION_ELEMENTS {
            return Err(LayerAttentionError::ElementLimit {
                field: "attention input",
                elements: input.len(),
            });
        }
        let positions = input.len() / stride;
        if positions > MAX_LAYER_ATTENTION_ELEMENTS {
            return Err(LayerAttentionError::ElementLimit {
                field: "input positions",
                elements: positions,
            });
        }
        Ok(positions)
    }

    fn validate_transition(
        &self,
        start: usize,
        positions: usize,
    ) -> Result<(WindowStep, bool), LayerAttentionError> {
        if start == 0 {
            return Ok((
                WindowStep::Prefill {
                    tokens: NonZeroUsize::new(positions).expect("nonempty input"),
                },
                self.next_position.is_some(),
            ));
        }
        if positions != 1 {
            return Err(LayerAttentionError::DecodeMustHaveOnePosition { positions });
        }
        if self.next_position != Some(start) {
            return Err(LayerAttentionError::DiscontinuousPosition {
                expected: self.next_position,
                actual: start,
            });
        }
        Ok((
            WindowStep::Decode {
                position: NonZeroUsize::new(start).expect("nonzero start"),
            },
            false,
        ))
    }

    fn validate_publication(
        &self,
        publication: CompressedAttentionPublication<'_>,
        epoch: u64,
        call_id: u64,
        start: usize,
        positions: usize,
        step: WindowStep,
    ) -> Result<(), LayerAttentionError> {
        self.validate_publication_identity(publication, epoch, call_id)?;
        let compressed_keys = self.compressed_keys(publication)?;
        let expected_compressed_keys =
            start
                .checked_add(positions)
                .ok_or(LayerAttentionError::ShapeOverflow {
                    field: "compressed key position",
                })?
                / self.layout.compressed_ratio.get();
        if compressed_keys != expected_compressed_keys {
            return Err(LayerAttentionError::CompressedKeyCount {
                actual: compressed_keys,
                expected: expected_compressed_keys,
            });
        }
        self.validate_compressed_indices(publication, compressed_keys, start, positions, step)
    }

    fn validate_publication_identity(
        &self,
        publication: CompressedAttentionPublication<'_>,
        epoch: u64,
        call_id: u64,
    ) -> Result<(), LayerAttentionError> {
        if publication.source_layer != self.layout.expected_source_layer {
            return Err(LayerAttentionError::WrongSourceLayer {
                actual: publication.source_layer,
                expected: self.layout.expected_source_layer,
            });
        }
        if publication.epoch != epoch {
            return Err(LayerAttentionError::WrongEpoch {
                actual: publication.epoch,
                expected: epoch,
            });
        }
        if publication.call_id != call_id {
            return Err(LayerAttentionError::WrongCallId {
                actual: publication.call_id,
                expected: call_id,
            });
        }
        Ok(())
    }

    fn validate_compressed_indices(
        &self,
        publication: CompressedAttentionPublication<'_>,
        compressed_keys: usize,
        start: usize,
        positions: usize,
        step: WindowStep,
    ) -> Result<(), LayerAttentionError> {
        if publication.indices.len() > MAX_LAYER_ATTENTION_ELEMENTS {
            return Err(LayerAttentionError::ElementLimit {
                field: "compressed indices",
                elements: publication.indices.len(),
            });
        }
        let denominator = checked_product(
            &[self.layout.batches.get(), positions],
            "compressed index rows",
        )?;
        if !publication.indices.len().is_multiple_of(denominator) {
            return Err(LayerAttentionError::CompressedIndexLength {
                actual: publication.indices.len(),
                rows: denominator,
            });
        }
        let compressed_slots = publication.indices.len() / denominator;
        if compressed_keys == 0 && compressed_slots != 0 {
            return Err(LayerAttentionError::CompressedSlotsWithoutKeys {
                slots: compressed_slots,
            });
        }
        let window_keys = match step {
            WindowStep::Prefill { .. } => positions,
            WindowStep::Decode { .. } => self.layout.window.get(),
        };
        for row in 0..denominator {
            let query = row % positions;
            let mut seen = HashSet::with_capacity(compressed_slots);
            for (offset, &raw_index) in publication.indices
                [row * compressed_slots..(row + 1) * compressed_slots]
                .iter()
                .enumerate()
            {
                if raw_index == -1 {
                    continue;
                }
                let slot = row * compressed_slots + offset;
                let Ok(index) = usize::try_from(raw_index) else {
                    return Err(LayerAttentionError::InvalidCompressedIndex {
                        slot,
                        index: raw_index,
                        window_keys,
                        compressed_keys,
                    });
                };
                if index < window_keys || index >= window_keys + compressed_keys {
                    return Err(LayerAttentionError::InvalidCompressedIndex {
                        slot,
                        index: raw_index,
                        window_keys,
                        compressed_keys,
                    });
                }
                if !seen.insert(index) {
                    return Err(LayerAttentionError::DuplicateCompressedIndex { row, index });
                }
                let causal = start
                    .checked_add(query)
                    .and_then(|position| position.checked_add(1))
                    .ok_or(LayerAttentionError::ShapeOverflow {
                        field: "causal position",
                    })?
                    / self.layout.compressed_ratio.get();
                if index - window_keys >= causal {
                    return Err(LayerAttentionError::FutureCompressedIndex {
                        slot,
                        index,
                        causal,
                    });
                }
            }
        }
        Ok(())
    }

    fn compressed_keys(
        &self,
        publication: CompressedAttentionPublication<'_>,
    ) -> Result<usize, LayerAttentionError> {
        let stride = checked_product(
            &[self.layout.batches.get(), self.layout.head_dimension.get()],
            "compressed key stride",
        )?;
        if !publication.numerical_bf16.len().is_multiple_of(stride) {
            return Err(LayerAttentionError::CompressedValueLength {
                actual: publication.numerical_bf16.len(),
                stride,
            });
        }
        if publication.numerical_bf16.len() > MAX_LAYER_ATTENTION_ELEMENTS {
            return Err(LayerAttentionError::ElementLimit {
                field: "compressed numerical values",
                elements: publication.numerical_bf16.len(),
            });
        }
        Ok(publication.numerical_bf16.len() / stride)
    }

    fn query_width(&self) -> Result<usize, LayerAttentionError> {
        Ok(checked_product(
            &[self.layout.heads.get(), self.layout.head_dimension.get()],
            "query width",
        )?)
    }
}

impl LayerAttentionLayout {
    fn norm_epsilon(self) -> f32 {
        f32::from_bits(self.norm_epsilon_bits)
    }
    fn softmax_scale(self) -> f32 {
        f32::from_bits(self.softmax_scale_bits)
    }
}

/// Invalid layer-attention configuration.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum LayerAttentionLayoutError {
    #[error("layer-attention shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("rope width {rope_width} exceeds head dimension {head_dimension}")]
    RopeExceedsHead {
        rope_width: usize,
        head_dimension: usize,
    },
    #[error("heads {heads} are not divisible by groups {groups}")]
    HeadsNotGrouped { heads: usize, groups: usize },
    #[error("{field} width {width} is not divisible by FP8 group 32")]
    UngroupedFp8Reduction { field: &'static str, width: usize },
    #[error(
        "layer-attention {field} has {elements} elements, maximum is {MAX_LAYER_ATTENTION_ELEMENTS}"
    )]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    #[error("norm epsilon must be finite and positive")]
    InvalidNormEpsilon,
    #[error("softmax scale must be finite and positive")]
    InvalidSoftmaxScale,
}

/// A rejected layer-attention call. State is unchanged on every variant.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LayerAttentionError {
    #[error(transparent)]
    Layout(#[from] LayerAttentionLayoutError),
    #[error(transparent)]
    ActivationQuant(#[from] ActivationQuantError),
    #[error(transparent)]
    Fp8Linear(#[from] Fp8LinearError),
    #[error(transparent)]
    Norm(#[from] crate::RmsNormError),
    #[error(transparent)]
    Rotary(#[from] RotaryError),
    #[error(transparent)]
    Roundtrip(#[from] ActivationRoundtripError),
    #[error(transparent)]
    Window(#[from] WindowError),
    #[error(transparent)]
    SparseLayout(#[from] SparseAttentionError),
    #[error(transparent)]
    Sparse(#[from] SparseAttentionBf16Error),
    #[error(transparent)]
    Output(#[from] AttentionOutputError),
    #[error(transparent)]
    OutputLayout(#[from] AttentionOutputLayoutError),
    #[error("attention input length is {actual}; it must be a nonempty multiple of {stride}")]
    InputLength { actual: usize, stride: usize },
    #[error("decode requires exactly one position, got {positions}")]
    DecodeMustHaveOnePosition { positions: usize },
    #[error("attention position discontinuity: expected {expected:?}, got {actual}")]
    DiscontinuousPosition {
        expected: Option<usize>,
        actual: usize,
    },
    #[error("compressed publication source layer {actual} is not expected layer {expected}")]
    WrongSourceLayer { actual: u16, expected: u16 },
    #[error("compressed publication epoch {actual} is not expected epoch {expected}")]
    WrongEpoch { actual: u64, expected: u64 },
    #[error("compressed publication call {actual} is not expected call {expected}")]
    WrongCallId { actual: u64, expected: u64 },
    #[error("compressed numerical BF16 length is {actual}; it must be divisible by {stride}")]
    CompressedValueLength { actual: usize, stride: usize },
    #[error("compressed publication has {actual} keys; source shape requires {expected}")]
    CompressedKeyCount { actual: usize, expected: usize },
    #[error("compressed index length is {actual}; it must be divisible by {rows}")]
    CompressedIndexLength { actual: usize, rows: usize },
    #[error("compressed publication has no keys but supplies {slots} index slots per query")]
    CompressedSlotsWithoutKeys { slots: usize },
    #[error(
        "compressed index {index} at slot {slot} is not -1 or in window-plus-compressed range (window keys {window_keys}, compressed keys {compressed_keys})"
    )]
    InvalidCompressedIndex {
        slot: usize,
        index: i32,
        window_keys: usize,
        compressed_keys: usize,
    },
    #[error("compressed index {index} is duplicated in publication row {row}")]
    DuplicateCompressedIndex { row: usize, index: usize },
    #[error("compressed index {index} at slot {slot} is beyond causal compressed length {causal}")]
    FutureCompressedIndex {
        slot: usize,
        index: usize,
        causal: usize,
    },
    #[error("layer-attention shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error(
        "layer-attention {field} has {elements} elements, maximum is {MAX_LAYER_ATTENTION_ELEMENTS}"
    )]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    #[error("a sparse attention call needs at least one sparse slot")]
    NoSparseSlots,
    #[error("a sparse attention call needs at least one key")]
    NoKeys,
    #[error("epoch counter overflowed")]
    EpochOverflow,
    #[error("successful call counter overflowed")]
    CallIdOverflow,
    #[error("FP8 projection result was nonfinite after BF16 narrowing at element {element}")]
    NonFiniteProjection { element: usize },
    #[error("rotary result was nonfinite after BF16 narrowing at tail element {element}")]
    NonFiniteRotary { element: usize },
}

fn fp8_project_bf16(
    input: &[u16],
    rows: usize,
    reduction: usize,
    outputs: usize,
    projection: Fp8Projection<'_>,
) -> Result<Vec<u16>, LayerAttentionError> {
    if input.len() > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "FP8 projection input",
            elements: input.len(),
        });
    }
    let output_elements = checked_product(&[rows, outputs], "projection output")?;
    if output_elements > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "FP8 projection output",
            elements: output_elements,
        });
    }
    let mut codes = vec![0; input.len()];
    let mut scales = vec![0; rows * (reduction / 32)];
    quantize_bf16_activations_e4m3fn(
        input,
        rows,
        reduction,
        ActivationGroup::Elements32,
        &mut codes,
        &mut scales,
    )?;
    let mut fp32 = vec![0.0; output_elements];
    fp8_linear_runtime_f32(
        &codes,
        &scales,
        projection.codes,
        projection.scales,
        rows,
        reduction,
        outputs,
        ActivationGroup::Elements32,
        &mut fp32,
    )?;
    fp32.into_iter()
        .enumerate()
        .map(|(element, value)| {
            let bits = f32_to_bf16_rne(value);
            if bf16_to_f32(bits).is_finite() {
                Ok(bits)
            } else {
                Err(LayerAttentionError::NonFiniteProjection { element })
            }
        })
        .collect()
}

fn rms_norm_rows(
    input: &[u16],
    rows: usize,
    width: usize,
    weight: &[u16],
    epsilon: f32,
) -> Result<Vec<u16>, LayerAttentionError> {
    let expected = checked_product(&[rows, width], "RMS norm input")?;
    if input.len() != expected {
        return Err(LayerAttentionError::ShapeOverflow {
            field: "RMS norm input",
        });
    }
    let mut output = vec![0; input.len()];
    for row in 0..rows {
        let start = row * width;
        rms_norm_bf16_reference(
            &input[start..start + width],
            weight,
            epsilon,
            &mut output[start..start + width],
        )?;
    }
    Ok(output)
}

fn rotate_bf16_tail(
    values: &[u16],
    shape: TailShape,
    frequencies: &[RotaryFrequency],
    direction: RotaryDirection,
) -> Result<Vec<u16>, LayerAttentionError> {
    let prefix = shape.head_dimension.get() - shape.rope_pairs.get() * 2;
    let mut tail =
        Vec::with_capacity(values.len() / shape.head_dimension.get() * shape.rope_pairs.get() * 2);
    for head in values.chunks_exact(shape.head_dimension.get()) {
        tail.extend(head[prefix..].iter().map(|&bits| bf16_to_f32(bits)));
    }
    let layout = RotaryTailLayout::new(
        shape.batches,
        NonZeroUsize::new(shape.positions).expect("positions validated"),
        shape.heads,
        shape.rope_pairs,
    )?;
    rotate_tail(&mut tail, layout, frequencies, direction)?;
    let mut result = values.to_vec();
    for (head, rotated) in result
        .chunks_exact_mut(shape.head_dimension.get())
        .zip(tail.chunks_exact(shape.rope_pairs.get() * 2))
    {
        for (offset, &value) in rotated.iter().enumerate() {
            let bits = f32_to_bf16_rne(value);
            if !bf16_to_f32(bits).is_finite() {
                return Err(LayerAttentionError::NonFiniteRotary { element: offset });
            }
            head[prefix + offset] = bits;
        }
    }
    Ok(result)
}

fn concatenate_kv(
    window: &[u16],
    window_keys: usize,
    compressed: &[u16],
    compressed_keys: usize,
    batches: usize,
    width: usize,
) -> Result<Vec<u16>, LayerAttentionError> {
    let keys =
        window_keys
            .checked_add(compressed_keys)
            .ok_or(LayerAttentionError::ShapeOverflow {
                field: "shared KV keys",
            })?;
    let per_batch = checked_product(&[keys, width], "shared KV per batch")?;
    let total = checked_product(&[batches, per_batch], "shared KV")?;
    if total > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "shared KV",
            elements: total,
        });
    }
    let mut result = Vec::with_capacity(total);
    for batch in 0..batches {
        result.extend_from_slice(
            &window[batch * window_keys * width..(batch + 1) * window_keys * width],
        );
        result.extend_from_slice(
            &compressed[batch * compressed_keys * width..(batch + 1) * compressed_keys * width],
        );
    }
    Ok(result)
}

fn concatenate_indices(
    window: &[i32],
    compressed: &[i32],
    batches: usize,
    positions: usize,
) -> Result<Vec<i32>, LayerAttentionError> {
    let rows = checked_product(&[batches, positions], "index rows")?;
    if !window.len().is_multiple_of(rows) || !compressed.len().is_multiple_of(rows) {
        return Err(LayerAttentionError::ShapeOverflow {
            field: "index rows",
        });
    }
    let window_slots = window.len() / rows;
    let compressed_slots = compressed.len() / rows;
    let total =
        window
            .len()
            .checked_add(compressed.len())
            .ok_or(LayerAttentionError::ShapeOverflow {
                field: "concatenated indices",
            })?;
    if total > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "concatenated indices",
            elements: total,
        });
    }
    let mut result = Vec::with_capacity(total);
    for row in 0..rows {
        result.extend_from_slice(&window[row * window_slots..(row + 1) * window_slots]);
        result.extend_from_slice(&compressed[row * compressed_slots..(row + 1) * compressed_slots]);
    }
    Ok(result)
}

fn checked_product(
    values: &[usize],
    field: &'static str,
) -> Result<usize, LayerAttentionLayoutError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(LayerAttentionLayoutError::ShapeOverflow { field })
    })
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::{
        CompressedAttentionPublication, LayerAttentionError, LayerAttentionLayout,
        LayerAttentionLayoutError, LayerAttentionState, WindowStep,
    };
    use std::num::NonZeroUsize;

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("test dimensions are nonzero")
    }

    fn layout() -> LayerAttentionLayout {
        LayerAttentionLayout::new(
            nonzero(1),
            nonzero(32),
            nonzero(2),
            nonzero(32),
            nonzero(1),
            nonzero(32),
            nonzero(4),
            nonzero(1),
            nonzero(32),
            7,
            nonzero(1),
            1e-5,
            0.25,
        )
        .expect("small grouped layout")
    }

    #[test]
    fn layout_retains_explicit_compressed_producer() {
        assert_eq!(layout().expected_source_layer(), 7);
    }

    #[test]
    fn fp8_reduction_boundary_is_validated_up_front() {
        let error = LayerAttentionLayout::new(
            nonzero(1),
            nonzero(31),
            nonzero(1),
            nonzero(32),
            nonzero(1),
            nonzero(32),
            nonzero(4),
            nonzero(1),
            nonzero(32),
            0,
            nonzero(1),
            1e-5,
            0.25,
        )
        .expect_err("ungrouped input cannot enter FP8 projection");
        assert!(matches!(
            error,
            LayerAttentionLayoutError::UngroupedFp8Reduction {
                field: "hidden dimension",
                width: 31
            }
        ));
    }

    #[test]
    fn reset_is_an_explicit_epoch_transition() {
        let mut state = LayerAttentionState::new(layout());
        state.reset().expect("first epoch advance fits");
        assert_eq!(state.epoch, 1);
        assert_eq!(state.next_position, None);
        assert_eq!(state.next_call_id, 0);
    }

    #[test]
    fn zero_compressed_prefix_is_valid_and_mismatched_prefix_is_atomic() {
        let layout = LayerAttentionLayout::new(
            nonzero(1),
            nonzero(32),
            nonzero(2),
            nonzero(32),
            nonzero(1),
            nonzero(32),
            nonzero(4),
            nonzero(1),
            nonzero(32),
            7,
            nonzero(2),
            1e-5,
            0.25,
        )
        .expect("small ratio-two layout");
        let state = LayerAttentionState::new(layout);
        let step = WindowStep::Prefill {
            tokens: NonZeroUsize::MIN,
        };
        state
            .validate_publication(
                CompressedAttentionPublication {
                    source_layer: 7,
                    epoch: 0,
                    call_id: 0,
                    numerical_bf16: &[],
                    indices: &[],
                },
                0,
                0,
                0,
                1,
                step,
            )
            .expect("ratio-two first token has a valid window-only prefix");
        let before = (
            state.ring.clone(),
            state.next_position,
            state.epoch,
            state.next_call_id,
        );
        let phantom_slots = state
            .validate_publication(
                CompressedAttentionPublication {
                    source_layer: 7,
                    epoch: 0,
                    call_id: 0,
                    numerical_bf16: &[],
                    indices: &[-1],
                },
                0,
                0,
                0,
                1,
                step,
            )
            .expect_err("zero compressed keys cannot carry placeholder slots");
        assert!(matches!(
            phantom_slots,
            LayerAttentionError::CompressedSlotsWithoutKeys { slots: 1 }
        ));
        let error = state
            .validate_publication(
                CompressedAttentionPublication {
                    source_layer: 7,
                    epoch: 0,
                    call_id: 0,
                    numerical_bf16: &[0; 32],
                    indices: &[],
                },
                0,
                0,
                0,
                1,
                step,
            )
            .expect_err("one compressed key is too long for the first ratio-two token");
        assert!(matches!(
            error,
            LayerAttentionError::CompressedKeyCount {
                actual: 1,
                expected: 0
            }
        ));
        assert_eq!(
            (
                state.ring,
                state.next_position,
                state.epoch,
                state.next_call_id
            ),
            before
        );
    }

    #[test]
    fn duplicate_compressed_indices_are_rejected_per_query_row() {
        let layout = LayerAttentionLayout::new(
            nonzero(1),
            nonzero(32),
            nonzero(2),
            nonzero(32),
            nonzero(1),
            nonzero(32),
            nonzero(4),
            nonzero(1),
            nonzero(32),
            7,
            nonzero(2),
            1e-5,
            0.25,
        )
        .expect("small ratio-two layout");
        let state = LayerAttentionState::new(layout);
        let error = state
            .validate_publication(
                CompressedAttentionPublication {
                    source_layer: 7,
                    epoch: 0,
                    call_id: 0,
                    numerical_bf16: &[0; 32],
                    indices: &[4, 4],
                },
                0,
                0,
                1,
                1,
                WindowStep::Decode {
                    position: NonZeroUsize::MIN,
                },
            )
            .expect_err("a source top-k row cannot select one compressed key twice");
        assert!(matches!(
            error,
            LayerAttentionError::DuplicateCompressedIndex { row: 0, index: 4 }
        ));
    }
}
