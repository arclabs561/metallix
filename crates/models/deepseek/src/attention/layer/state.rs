//! The request-local window ring and its forward paths.

use std::{collections::HashSet, num::NonZeroUsize};

use crate::{
    RotaryDirection, RotaryFrequency,
    attention::{
        AttentionOutputLayout, SparseAttentionLayout, attention_output_reference,
        sparse_attention_bf16_reference,
        window::{WindowStep, window_topk_indices, write_window_kv_bf16},
    },
    precision::{ActivationGroup, requantize_bf16_activations_e4m3fn},
};

use super::{
    AttentionQrLayout, AttentionQrWeights, Fp8Projection, LayerAttentionError,
    LayerAttentionLayout, MAX_LAYER_ATTENTION_ELEMENTS,
    ops::{
        checked_product, concatenate_indices, concatenate_kv, fp8_project_bf16, rms_norm_rows,
        rotate_bf16_tail,
    },
    prepare_attention_qr,
};

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
pub(super) struct TailShape {
    pub(super) batches: NonZeroUsize,
    pub(super) positions: usize,
    pub(super) heads: NonZeroUsize,
    pub(super) head_dimension: NonZeroUsize,
    pub(super) rope_pairs: NonZeroUsize,
}

/// A single model-local numerical window owner.
#[derive(Clone, Debug)]
pub struct LayerAttentionState {
    layout: LayerAttentionLayout,
    pub(super) ring: Vec<u16>,
    pub(super) next_position: Option<usize>,
    pub(super) epoch: u64,
    pub(super) next_call_id: u64,
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
        if self.layout.compressed_ratio.is_none() {
            return Err(LayerAttentionError::WindowOnlyLayoutRequiresWindowMethod);
        }
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
            Some(publication),
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

    /// Computes one source-shaped attention call with only its local window.
    ///
    /// This path deliberately has no compressed-KV publication, producer
    /// identity, or index concatenation.  It retains the same request-local
    /// ring transition and all query, sparse, and output stages as [`Self::forward`].
    pub fn forward_window_only(
        &mut self,
        attention_input: &[u16],
        start_position: usize,
        frequencies: &[RotaryFrequency],
        weights: LayerAttentionWeights<'_>,
    ) -> Result<LayerAttentionDiagnostic, LayerAttentionError> {
        if self.layout.compressed_ratio.is_some() {
            return Err(LayerAttentionError::CompressedLayoutRequiresPublication);
        }
        let positions = self.input_positions(attention_input)?;
        let (step, reset_epoch) = self.validate_transition(start_position, positions)?;
        let query = self.prepare_query(attention_input, positions, frequencies, weights)?;
        let window = self.prepare_window(
            attention_input,
            start_position,
            positions,
            frequencies,
            weights,
            None,
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
        let next_call_id = if start_position == 0 {
            1
        } else {
            self.next_call_id
                .checked_add(1)
                .ok_or(LayerAttentionError::CallIdOverflow)?
        };
        let next_epoch = if reset_epoch {
            self.epoch
                .checked_add(1)
                .ok_or(LayerAttentionError::EpochOverflow)?
        } else {
            self.epoch
        };
        let ring_after = window.staged_ring.clone();
        self.ring = window.staged_ring;
        self.next_position = Some(next_position);
        self.epoch = next_epoch;
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
        let qr_stages = prepare_attention_qr(
            attention_input,
            AttentionQrWeights {
                wq_a: weights.wq_a,
                q_norm: weights.q_norm,
            },
            AttentionQrLayout::new(
                self.layout.batches,
                self.layout.hidden_dimension,
                self.layout.q_rank,
                self.layout.norm_epsilon(),
            )?,
        )?;
        let rows = checked_product(&[self.layout.batches.get(), positions], "input rows")?;
        let wq_b_pre_rope = fp8_project_bf16(
            &qr_stages.qr,
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
            wq_a: qr_stages.wq_a,
            qr: qr_stages.qr,
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
        publication: Option<CompressedAttentionPublication<'_>>,
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
        let window_indices = window_topk_indices(step, self.layout.window, self.layout.batches)?;
        let (shared_kv, indices, compressed_keys) = if let Some(publication) = publication {
            let compressed_keys = self.compressed_keys(publication)?;
            (
                concatenate_kv(
                    &window_read,
                    window_keys,
                    publication.numerical_bf16,
                    compressed_keys,
                    self.layout.batches.get(),
                    self.layout.head_dimension.get(),
                )?,
                concatenate_indices(
                    &window_indices,
                    publication.indices,
                    self.layout.batches.get(),
                    positions,
                )?,
                compressed_keys,
            )
        } else {
            (window_read.clone(), window_indices.clone(), 0)
        };
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

    pub(super) fn validate_publication(
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
                / self
                    .layout
                    .compressed_ratio
                    .expect("compressed forward validated its layout")
                    .get();
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
        let expected_source_layer = self
            .layout
            .expected_source_layer
            .expect("compressed forward validated its layout");
        if publication.source_layer != expected_source_layer {
            return Err(LayerAttentionError::WrongSourceLayer {
                actual: publication.source_layer,
                expected: expected_source_layer,
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
                    / self
                        .layout
                        .compressed_ratio
                        .expect("compressed forward validated its layout")
                        .get();
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
