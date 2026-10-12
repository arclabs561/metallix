//! The request-local window ring and its forward paths.

use std::{collections::HashSet, num::NonZeroUsize};

use crate::{
    RotaryDirection, RotaryFrequency,
    attention::{
        AttentionOutputLayout, SparseAttentionError, SparseAttentionLayout,
        attention_output_reference, sparse_attention_bf16_reference,
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
    /// Query down-projection to `q_rank`.
    pub wq_a: Fp8Projection<'a>,
    /// `RMSNorm` weight on the query rank, BF16.
    pub q_norm: &'a [u16],
    /// Query up-projection to every head.
    pub wq_b: Fp8Projection<'a>,
    /// Shared key-value projection to one head width.
    pub wkv: Fp8Projection<'a>,
    /// `RMSNorm` weight on the key-value head, BF16.
    pub kv_norm: &'a [u16],
    /// Per-head attention sink logit.
    pub attn_sink: &'a [f32],
    /// Group-local BF16 output projection.
    pub wo_a: &'a [u16],
    /// FP8 output projection back to the hidden width.
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
    /// The layer that produced the compressed keys.
    pub source_layer: u16,
    /// The consuming state's epoch the producer wrote for.
    pub epoch: u64,
    /// The consuming state's successful-call ordinal the producer wrote for.
    pub call_id: u64,
    /// Numerical, already-quantized-and-reconstructed BF16 `[batch, key, head_dim]`.
    pub numerical_bf16: &'a [u16],
    /// Source indices `[batch, query, compressed_slot]`, offset into concatenated KV.
    pub indices: &'a [i32],
}

/// Every fixture-visible intermediate produced by [`LayerAttentionState::forward`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayerAttentionDiagnostic {
    /// `wq_a` projection output, BF16.
    pub wq_a: Vec<u16>,
    /// `q_norm` applied to `wq_a`, the indexer's `qr` operand, BF16.
    pub qr: Vec<u16>,
    /// `wq_b` projection output before rotary embedding, BF16.
    pub wq_b_pre_rope: Vec<u16>,
    /// Queries after rotary embedding, BF16.
    pub q_after_rope: Vec<u16>,
    /// This call's key-value rows as written to the window, BF16.
    pub prepared_window: Vec<u16>,
    /// Window keys read for attention, BF16.
    pub window_read: Vec<u16>,
    /// Window indices into the concatenated keys.
    pub window_indices: Vec<i32>,
    /// The window ring after the call, BF16.
    pub ring_after: Vec<u16>,
    /// Sparse attention output before the output projection, BF16.
    pub sparse_output: Vec<u16>,
    /// Layer output after the output projection, BF16.
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
    ///
    /// # Errors
    ///
    /// Returns [`LayerAttentionError::EpochOverflow`], leaving the state
    /// unchanged, when the epoch counter would overflow.
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
    ///
    /// # Errors
    ///
    /// The state is unchanged on every error.
    ///
    /// * [`LayerAttentionError::WindowOnlyLayoutRequiresWindowMethod`] for a
    ///   window-only layout.
    /// * [`LayerAttentionError::WrongSourceLayer`],
    ///   [`LayerAttentionError::WrongEpoch`], [`LayerAttentionError::WrongCallId`],
    ///   [`LayerAttentionError::CompressedValueLength`],
    ///   [`LayerAttentionError::CompressedKeyCount`],
    ///   [`LayerAttentionError::CompressedIndexLength`],
    ///   [`LayerAttentionError::CompressedSlotsWithoutKeys`],
    ///   [`LayerAttentionError::InvalidCompressedIndex`],
    ///   [`LayerAttentionError::DuplicateCompressedIndex`] and
    ///   [`LayerAttentionError::FutureCompressedIndex`] when the publication
    ///   is not the one this call expects.
    /// * [`LayerAttentionError::InputLength`],
    ///   [`LayerAttentionError::DecodeMustHaveOnePosition`] and
    ///   [`LayerAttentionError::DiscontinuousPosition`] when the input or its
    ///   position does not continue the sequence (a call at position 0 starts
    ///   a new one).
    /// * [`LayerAttentionError::ShapeOverflow`],
    ///   [`LayerAttentionError::ElementLimit`], [`LayerAttentionError::NoKeys`]
    ///   and [`LayerAttentionError::NoSparseSlots`] when the call's shapes do
    ///   not fit the bounded reference.
    /// * [`LayerAttentionError::NonFiniteProjection`],
    ///   [`LayerAttentionError::NonFiniteRotary`] and the wrapped stage errors
    ///   when a projection, norm, rotary, quantization, window, sparse or
    ///   output stage rejects its input or overflows.
    /// * [`LayerAttentionError::EpochOverflow`] and
    ///   [`LayerAttentionError::CallIdOverflow`] when a counter would overflow.
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
    ///
    /// # Errors
    ///
    /// The state is unchanged on every error.
    ///
    /// * [`LayerAttentionError::CompressedLayoutRequiresPublication`] for a
    ///   compressed layout.
    /// * [`LayerAttentionError::InputLength`],
    ///   [`LayerAttentionError::DecodeMustHaveOnePosition`] and
    ///   [`LayerAttentionError::DiscontinuousPosition`] when the input or its
    ///   position does not continue the sequence (a call at position 0 starts
    ///   a new one).
    /// * [`LayerAttentionError::ShapeOverflow`],
    ///   [`LayerAttentionError::ElementLimit`], [`LayerAttentionError::NoKeys`]
    ///   and [`LayerAttentionError::NoSparseSlots`] when the call's shapes do
    ///   not fit the bounded reference.
    /// * [`LayerAttentionError::NonFiniteProjection`],
    ///   [`LayerAttentionError::NonFiniteRotary`] and the wrapped stage errors
    ///   when a projection, norm, rotary, quantization, window, sparse or
    ///   output stage rejects its input or overflows.
    /// * [`LayerAttentionError::EpochOverflow`] and
    ///   [`LayerAttentionError::CallIdOverflow`] when a counter would overflow.
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
        let rows = checked_product(&[self.layout.batches.get(), positions], "sparse query rows")?;
        let (indices, sparse_slots) = compact_sparse_indices(&window.indices, rows)?;
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
            &indices,
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

/// Storage slots do not define numerical execution blocks. Keep every selected
/// key in its original order (including duplicates), then pad only row tails.
/// In particular, a decode ring's unused slots must not separate its window
/// keys from compressed keys into a different BF16 online-normalization block.
fn compact_sparse_indices(
    indices: &[i32],
    rows: usize,
) -> Result<(Vec<i32>, NonZeroUsize), LayerAttentionError> {
    if rows == 0 || indices.is_empty() || !indices.len().is_multiple_of(rows) {
        return Err(LayerAttentionError::NoSparseSlots);
    }
    if rows > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "sparse query rows",
            elements: rows,
        });
    }
    if indices.len() > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "sparse indices",
            elements: indices.len(),
        });
    }
    let storage_slots = indices.len() / rows;
    let row = |index: usize| &indices[index * storage_slots..(index + 1) * storage_slots];
    // Remove only the -1 mask sentinel. Invalid indices remain visible to the
    // primitive's validation, instead of being silently treated as padding.
    let slots = (0..rows)
        .map(|index| row(index).iter().filter(|&&key| key != -1).count())
        .max()
        .unwrap_or(0)
        .max(1);
    let elements = checked_product(&[rows, slots], "compact sparse indices")?;
    if elements > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "compact sparse indices",
            elements,
        });
    }
    let mut compact = Vec::new();
    compact
        .try_reserve_exact(elements)
        .map_err(|_| SparseAttentionError::AllocationFailed { elements })?;
    for index in 0..rows {
        compact.extend(row(index).iter().copied().filter(|&key| key != -1));
        compact.resize((index + 1) * slots, -1);
    }
    Ok((
        compact,
        NonZeroUsize::new(slots).expect("at least one slot"),
    ))
}

#[cfg(test)]
mod sparse_execution_tests {
    use super::*;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).unwrap()
    }

    fn native_sparse(
        indices: Vec<i32>,
        window_keys: usize,
    ) -> Result<Vec<u16>, LayerAttentionError> {
        let layout = LayerAttentionLayout::new(
            nz(1),
            nz(32),
            nz(1),
            nz(32),
            nz(1),
            nz(32),
            nz(128),
            nz(1),
            nz(32),
            7,
            nz(2),
            1e-5,
            1.0,
        )
        .unwrap();
        let state = LayerAttentionState::new(layout);
        // Independent frozen BF16 primitive counterexample, zero-extended to
        // native layer head width: q=[1,0], KV=[[0,1],[1,0]], sink=-100.
        // The two-key numerator uses BF16(exp(-1)), giving [3f3b,3e89].
        // Putting the second key after mask padding at slot128 instead would
        // rescale an earlier FP32 numerator, giving [3f3b,3e8a] instead of
        // the native packing policy's result.
        let mut query = vec![0; 32];
        query[0] = 0x3f80;
        let mut shared_kv = vec![0; (window_keys + 1) * 32];
        shared_kv[1] = 0x3f80;
        shared_kv[window_keys * 32] = 0x3f80;
        let window = WindowStages {
            prepared_window: Vec::new(),
            window_read: Vec::new(),
            staged_ring: Vec::new(),
            shared_kv,
            window_indices: Vec::new(),
            indices,
            window_keys,
            compressed_keys: 1,
        };
        // Only the sink participates in this adapter-stage call; projection
        // operands are deliberately empty rather than invented numerical weights.
        let unused = Fp8Projection {
            codes: &[],
            scales: &[],
        };
        state.sparse_output(
            &query,
            &window,
            1,
            LayerAttentionWeights {
                wq_a: unused,
                q_norm: &[],
                wq_b: unused,
                wkv: unused,
                kv_norm: &[],
                attn_sink: &[-100.0],
                wo_a: &[],
                wo_b: unused,
            },
        )
    }

    #[test]
    fn native_sparse_prefill_and_window128_decode_share_execution_blocks() {
        let prefill = native_sparse(vec![0, -1, -1, 3], 3).unwrap();
        let mut decoded = vec![-1; 129];
        decoded[0] = 0;
        decoded[128] = 128;
        let decode = native_sparse(decoded, 128).unwrap();
        let mut frozen = vec![0; 32];
        frozen[..2].copy_from_slice(&[0x3f3b, 0x3e89]);
        assert_eq!(prefill, frozen, "independent one-block BF16 oracle");
        assert_eq!(decode, frozen, "ring padding must not partition valid keys");
    }

    #[test]
    fn compact_execution_preserves_row_order_duplicates_and_masked_rows() {
        let storage = [2, -1, 0, 2, -1, -1, -1, -1];
        let (execution, slots) = compact_sparse_indices(&storage, 2).unwrap();
        assert_eq!(slots.get(), 3);
        assert_eq!(execution, [2, 0, 2, -1, -1, -1]);
        assert_eq!(storage, [2, -1, 0, 2, -1, -1, -1, -1]);
        assert_eq!(native_sparse(vec![-1; 129], 128).unwrap(), [0; 32]);
        assert!(matches!(
            native_sparse(Vec::new(), 128),
            Err(LayerAttentionError::NoSparseSlots)
        ));
        let duplicate = native_sparse(vec![0, 128, 0], 128).unwrap();
        assert_ne!(duplicate, native_sparse(vec![0, 128], 128).unwrap());
    }

    #[test]
    fn compact_execution_packs_mixed_rows_across_block_boundaries() {
        for valid in [63, 64, 65, 127, 128, 129] {
            let expected: Vec<i32> = (0..valid).rev().collect();
            let mut storage = Vec::new();
            for keys in [&expected[..], &expected[..expected.len() - 1]] {
                let start = storage.len();
                for &key in keys {
                    storage.extend([-1, key, -1]);
                }
                storage.resize(start + expected.len() * 3, -1);
            }
            let original = storage.clone();
            let (execution, slots) = compact_sparse_indices(&storage, 2).unwrap();
            assert_eq!(slots.get(), expected.len());
            assert_eq!(&execution[..expected.len()], expected);
            assert_eq!(
                &execution[expected.len()..execution.len() - 1],
                &expected[..expected.len() - 1]
            );
            assert_eq!(execution.last(), Some(&-1));
            assert_eq!(storage, original);
        }
    }

    #[test]
    fn compact_execution_keeps_invalid_indices_visible_and_bounds_empty_geometry() {
        for index in [-2, 129] {
            assert!(matches!(
                native_sparse(vec![-1, index], 128),
                Err(LayerAttentionError::Sparse(
                    crate::attention::SparseAttentionBf16Error::Input(
                        SparseAttentionError::InvalidIndex { .. }
                    )
                ))
            ));
        }
        assert!(matches!(
            compact_sparse_indices(&[], 0),
            Err(LayerAttentionError::NoSparseSlots)
        ));
        assert!(matches!(
            compact_sparse_indices(&[-1], MAX_LAYER_ATTENTION_ELEMENTS + 1),
            Err(LayerAttentionError::NoSparseSlots)
        ));
        assert!(matches!(
            compact_sparse_indices(&[0, -1, 1], 2),
            Err(LayerAttentionError::NoSparseSlots)
        ));
    }
}
