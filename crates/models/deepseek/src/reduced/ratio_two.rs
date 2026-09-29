//! Request-local ratio-two compressed owner for layer-one numerical operands.
//!
//! The owner projects a live BF16 attention input, pools it with the
//! ratio-two compressor, and publishes paired prepared index-key and compressed
//! KV prefixes. Candidate scoring and the prior-layer-three score-prefix policy
//! remain with the caller: they consume this owner's completed publication but
//! must retain the producer request identity themselves.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    RotaryFrequency,
    compressor::{CompressorError, CompressorInput, CompressorState},
    indexer::{
        cache::{CompressedKvState, IndexKeyPublicationId, IndexKeyState, IndexKeyStateError},
        compressed_kv::{
            CompressedKvDiagnostic, CompressedKvError, CompressedKvLayout, CompressedKvLayoutError,
            prepare_compressed_kv,
        },
        key::{
            IndexKeyDiagnostic, IndexKeyError, IndexKeyLayout, IndexKeyLayoutError,
            IndexKeyWeights, prepare_index_keys,
        },
    },
    precision::{Fp32LinearError, MAX_FP32_LINEAR_ELEMENTS, bf16_to_f32, fp32_linear_reference},
};

/// Fixed source-shaped geometry for one ratio-two compressed owner request.
#[derive(Clone, Copy, Debug)]
pub struct RatioTwoOwnerLayout {
    batches: NonZeroUsize,
    input_dimension: NonZeroUsize,
    latent_dimension: NonZeroUsize,
    key_dimension: NonZeroUsize,
    rope_pairs: NonZeroUsize,
    capacity: NonZeroUsize,
    compressor_epsilon: f32,
}

impl RatioTwoOwnerLayout {
    /// Validates fixed L1 owner geometry before request state is allocated.
    #[allow(
        clippy::too_many_arguments,
        reason = "each source storage dimension is independently named"
    )]
    pub fn new(
        batches: NonZeroUsize,
        input_dimension: NonZeroUsize,
        latent_dimension: NonZeroUsize,
        key_dimension: NonZeroUsize,
        rope_pairs: NonZeroUsize,
        capacity: NonZeroUsize,
        compressor_epsilon: f32,
    ) -> Result<Self, RatioTwoOwnerError> {
        if !compressor_epsilon.is_finite() || compressor_epsilon <= 0.0 {
            return Err(RatioTwoOwnerError::InvalidCompressorEpsilon);
        }
        let layout = Self {
            batches,
            input_dimension,
            latent_dimension,
            key_dimension,
            rope_pairs,
            capacity,
            compressor_epsilon,
        };
        let _ = IndexKeyLayout::new(
            batches,
            latent_dimension,
            key_dimension,
            rope_pairs,
            compressor_epsilon,
        )?;
        let _ = CompressedKvLayout::new(batches, latent_dimension, rope_pairs)?;
        for (field, elements) in [
            (
                "one owner input",
                checked_product(
                    checked_product(batches.get(), input_dimension.get(), "one owner input")?,
                    1,
                    "one owner input",
                )?,
            ),
            (
                "one projected latent",
                checked_product(
                    batches.get(),
                    latent_dimension.get(),
                    "one projected latent",
                )?,
            ),
        ] {
            if elements > MAX_FP32_LINEAR_ELEMENTS {
                return Err(RatioTwoOwnerError::ElementLimit { field, elements });
            }
        }
        Ok(layout)
    }

    fn key_layout(self) -> Result<IndexKeyLayout, RatioTwoOwnerError> {
        Ok(IndexKeyLayout::new(
            self.batches,
            self.latent_dimension,
            self.key_dimension,
            self.rope_pairs,
            self.compressor_epsilon,
        )?)
    }

    fn kv_layout(self) -> Result<CompressedKvLayout, RatioTwoOwnerError> {
        Ok(CompressedKvLayout::new(
            self.batches,
            self.latent_dimension,
            self.rope_pairs,
        )?)
    }

    /// Returns the fixed batch count for request-session composition.
    #[must_use]
    pub(crate) const fn batches(self) -> NonZeroUsize {
        self.batches
    }

    /// Returns the BF16 attention-input row width.
    #[must_use]
    pub(crate) const fn input_dimension(self) -> NonZeroUsize {
        self.input_dimension
    }

    /// Returns the compressed latent width.
    #[must_use]
    pub(crate) const fn latent_dimension(self) -> NonZeroUsize {
        self.latent_dimension
    }

    /// Returns the prepared index-key row width.
    #[must_use]
    pub(crate) const fn key_dimension(self) -> NonZeroUsize {
        self.key_dimension
    }

    /// Returns the rotary-pair count used by prepared keys and KV.
    #[must_use]
    pub(crate) const fn rope_pairs(self) -> NonZeroUsize {
        self.rope_pairs
    }

    /// Returns the maximum completed compressed positions in this request.
    #[must_use]
    pub(crate) const fn capacity(self) -> NonZeroUsize {
        self.capacity
    }
}

/// Borrowed immutable numerical operands for the ratio-two owner.
#[derive(Clone, Copy, Debug)]
pub struct RatioTwoOwnerWeights<'a> {
    wkv: &'a [f32],
    wgate: &'a [f32],
    index_key: IndexKeyWeights<'a>,
}

impl<'a> RatioTwoOwnerWeights<'a> {
    /// Borrows FP32 owner projections and BF16 index-key operands.
    #[must_use]
    pub const fn new(wkv: &'a [f32], wgate: &'a [f32], index_key: IndexKeyWeights<'a>) -> Self {
        Self {
            wkv,
            wgate,
            index_key,
        }
    }
}

/// One ratio-two owner call over a live attention input.
#[derive(Clone, Copy, Debug)]
pub struct RatioTwoOwnerCall<'a> {
    publication: IndexKeyPublicationId,
    token_start: usize,
    positions: NonZeroUsize,
    input: &'a [u16],
    completed_frequencies: &'a [RotaryFrequency],
    weights: RatioTwoOwnerWeights<'a>,
}

impl<'a> RatioTwoOwnerCall<'a> {
    /// Groups request identity, token schedule, input, and completed-group `RoPE` rows.
    #[must_use]
    pub const fn new(
        publication: IndexKeyPublicationId,
        token_start: usize,
        positions: NonZeroUsize,
        input: &'a [u16],
        completed_frequencies: &'a [RotaryFrequency],
        weights: RatioTwoOwnerWeights<'a>,
    ) -> Self {
        Self {
            publication,
            token_start,
            positions,
            input,
            completed_frequencies,
            weights,
        }
    }
}

/// Source-visible numerical stages and prefix snapshots from one committed owner call.
#[derive(Clone, Debug, PartialEq)]
pub struct RatioTwoOwnerDiagnostic {
    projected: Vec<f32>,
    gate: Vec<f32>,
    latent: Option<Vec<u16>>,
    index_keys: Option<IndexKeyDiagnostic>,
    compressed_kv: Option<CompressedKvDiagnostic>,
    key_prefixes: Vec<Vec<u16>>,
    kv_prefixes: Vec<Vec<u16>>,
}

impl RatioTwoOwnerDiagnostic {
    /// Returns FP32 `wkv` output before compressor pooling.
    #[must_use]
    pub fn projected(&self) -> &[f32] {
        &self.projected
    }

    /// Returns FP32 `wgate` scores before compressor pooling.
    #[must_use]
    pub fn gate(&self) -> &[f32] {
        &self.gate
    }

    /// Returns a completed BF16 compressor latent, if this call closed a group.
    #[must_use]
    pub fn latent(&self) -> Option<&[u16]> {
        self.latent.as_deref()
    }

    /// Returns prepared key stages if this call closed a compression group.
    #[must_use]
    pub const fn index_keys(&self) -> Option<&IndexKeyDiagnostic> {
        self.index_keys.as_ref()
    }

    /// Returns prepared compressed-KV stages if this call closed a group.
    #[must_use]
    pub const fn compressed_kv(&self) -> Option<&CompressedKvDiagnostic> {
        self.compressed_kv.as_ref()
    }

    /// Returns one valid prepared key prefix per batch after the publication.
    #[must_use]
    pub fn key_prefixes(&self) -> &[Vec<u16>] {
        &self.key_prefixes
    }

    /// Returns one valid prepared compressed-KV prefix per batch after publication.
    #[must_use]
    pub fn kv_prefixes(&self) -> &[Vec<u16>] {
        &self.kv_prefixes
    }
}

/// Request-local ratio-two owner state. It has no candidate scorer or L3 score cache.
#[derive(Clone, Debug)]
pub struct RatioTwoCompressedOwner {
    layout: RatioTwoOwnerLayout,
    pristine_compressor: CompressorState,
    compressor: CompressorState,
    keys: IndexKeyState,
    kv: CompressedKvState,
    key_layout: IndexKeyLayout,
    kv_layout: CompressedKvLayout,
}

impl RatioTwoCompressedOwner {
    /// Creates empty paired owner prefixes and a ratio-two streaming compressor.
    pub fn new(
        layout: RatioTwoOwnerLayout,
        source_layer: u16,
        compressor_norm: &[u16],
    ) -> Result<Self, RatioTwoOwnerError> {
        let key_layout = layout.key_layout()?;
        let kv_layout = layout.kv_layout()?;
        let compressor = CompressorState::new(
            layout.batches.get(),
            layout.latent_dimension.get(),
            2,
            compressor_norm,
            layout.compressor_epsilon,
        )?;
        let keys = IndexKeyState::new(
            layout.batches,
            layout.key_dimension,
            layout.capacity,
            source_layer,
        )?;
        let kv = CompressedKvState::new(
            layout.batches,
            layout.latent_dimension,
            layout.capacity,
            source_layer,
        )?;
        Ok(Self {
            layout,
            pristine_compressor: compressor.clone(),
            compressor,
            keys,
            kv,
            key_layout,
            kv_layout,
        })
    }

    fn validate_call_identity(
        &self,
        call: RatioTwoOwnerCall<'_>,
    ) -> Result<(), RatioTwoOwnerError> {
        let expected_start = self.compressor.next_position();
        if call.token_start != expected_start {
            return Err(RatioTwoOwnerError::UnexpectedTokenStart {
                actual: call.token_start,
                expected: expected_start,
            });
        }
        if self.keys.epoch() != self.kv.epoch()
            || self.keys.next_call_id() != self.kv.next_call_id()
            || self.keys.valid_positions() != self.kv.valid_positions()
        {
            return Err(RatioTwoOwnerError::DivergentPrefixes);
        }
        if call.publication.source_layer() != self.keys.expected_source_layer()
            || call.publication.epoch() != self.keys.epoch()
            || call.publication.call_id() != self.keys.next_call_id()
        {
            return Err(RatioTwoOwnerError::UnexpectedPublication {
                source_layer: call.publication.source_layer(),
                expected_source_layer: self.keys.expected_source_layer(),
                epoch: call.publication.epoch(),
                expected_epoch: self.keys.epoch(),
                call_id: call.publication.call_id(),
                expected_call_id: self.keys.next_call_id(),
            });
        }
        Ok(())
    }

    /// Projects, compresses, prepares, and atomically replaces paired owner state.
    ///
    /// A partial ratio-two call publishes empty key/KV appends and therefore
    /// advances both request identities while retaining the existing prefixes.
    /// Any rejection leaves compressor and both prefixes unchanged, so the same
    /// publication can retry. Prior-L3 score-prefix selection is intentionally
    /// outside this component and must consume the returned key prefix later.
    pub fn forward(
        &mut self,
        call: RatioTwoOwnerCall<'_>,
    ) -> Result<RatioTwoOwnerDiagnostic, RatioTwoOwnerError> {
        self.validate_call_identity(call)?;
        let rows = checked_product(
            self.layout.batches.get(),
            call.positions.get(),
            "projection rows",
        )?;
        let input_elements = checked_product(rows, self.layout.input_dimension.get(), "input")?;
        if call.input.len() != input_elements {
            return Err(RatioTwoOwnerError::Length {
                field: "input",
                actual: call.input.len(),
                expected: input_elements,
            });
        }
        let projection_elements = checked_product(
            rows,
            self.layout.latent_dimension.get(),
            "projection output",
        )?;
        let input = bf16_to_f32_owned(call.input)?;
        let mut projected = allocate_f32(projection_elements, "wkv projection")?;
        projected.resize(projection_elements, 0.0);
        fp32_linear_reference(
            &input,
            call.weights.wkv,
            rows,
            self.layout.input_dimension.get(),
            self.layout.latent_dimension.get(),
            &mut projected,
        )?;
        let mut gate = allocate_f32(projection_elements, "wgate projection")?;
        gate.resize(projection_elements, 0.0);
        fp32_linear_reference(
            &input,
            call.weights.wgate,
            rows,
            self.layout.input_dimension.get(),
            self.layout.latent_dimension.get(),
            &mut gate,
        )?;

        let mut compressor = self.compressor.clone();
        let latent = compressor.forward(
            CompressorInput::Gated {
                kv: &projected,
                scores: &gate,
            },
            call.positions.get(),
            call.token_start,
        )?;
        let (index_keys, compressed_kv) = if let Some(latent) = latent.as_deref() {
            let index_keys = prepare_index_keys(
                latent,
                call.completed_frequencies,
                call.weights.index_key,
                self.key_layout,
            )?;
            let compressed_kv =
                prepare_compressed_kv(latent, call.completed_frequencies, self.kv_layout)?;
            (Some(index_keys), Some(compressed_kv))
        } else {
            if !call.completed_frequencies.is_empty() {
                return Err(RatioTwoOwnerError::UnexpectedPartialFrequencies {
                    actual: call.completed_frequencies.len(),
                });
            }
            (None, None)
        };
        let key_append = index_keys
            .as_ref()
            .map_or(&[][..], |diagnostic| diagnostic.post_fp4.as_slice());
        let kv_append = compressed_kv
            .as_ref()
            .map_or(&[][..], |diagnostic| diagnostic.post_fp4.as_slice());

        let mut keys = self.keys.clone();
        let mut kv = self.kv.clone();
        if keys.epoch() != kv.epoch()
            || keys.next_call_id() != kv.next_call_id()
            || keys.valid_positions() != kv.valid_positions()
        {
            return Err(RatioTwoOwnerError::DivergentPrefixes);
        }
        let prefix_start = keys.valid_positions();
        keys.append_prepared(call.publication, prefix_start, key_append)?;
        kv.append_prepared(call.publication, prefix_start, kv_append)?;
        let key_prefixes = collect_prefixes(&keys, self.layout.batches.get(), "index-key prefix")?;
        let kv_prefixes = collect_prefixes(&kv, self.layout.batches.get(), "compressed-KV prefix")?;

        self.compressor = compressor;
        self.keys = keys;
        self.kv = kv;
        Ok(RatioTwoOwnerDiagnostic {
            projected,
            gate,
            latent,
            index_keys,
            compressed_kv,
            key_prefixes,
            kv_prefixes,
        })
    }

    /// Begins one coordinated new epoch for compressor, key, and KV state.
    pub fn reset(&mut self) -> Result<(), RatioTwoOwnerError> {
        let mut keys = self.keys.clone();
        let mut kv = self.kv.clone();
        keys.reset()?;
        kv.reset()?;
        self.compressor = self.pristine_compressor.clone();
        self.keys = keys;
        self.kv = kv;
        Ok(())
    }

    /// Borrows one exact valid prepared key prefix.
    pub fn key_prefix(&self, batch: usize) -> Result<&[u16], RatioTwoOwnerError> {
        Ok(self.keys.prefix(batch)?)
    }

    /// Borrows one exact valid prepared compressed-KV prefix.
    pub fn kv_prefix(&self, batch: usize) -> Result<&[u16], RatioTwoOwnerError> {
        Ok(self.kv.prefix(batch)?)
    }

    /// Returns the synchronized owner epoch.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.keys.epoch()
    }

    /// Returns the next synchronized owner call ID.
    #[must_use]
    pub const fn next_call_id(&self) -> u64 {
        self.keys.next_call_id()
    }

    /// Returns the next token position required by the ratio-two compressor.
    #[must_use]
    pub const fn next_position(&self) -> usize {
        self.compressor.next_position()
    }
}

/// Rejected ratio-two owner construction, call, or reset.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RatioTwoOwnerError {
    #[error("ratio-two owner compressor epsilon must be finite and positive")]
    InvalidCompressorEpsilon,
    #[error("ratio-two owner {field} has {elements} elements beyond the FP32 bound")]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    #[error("ratio-two owner {field} shape overflowed")]
    ShapeOverflow { field: &'static str },
    #[error("ratio-two owner {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("ratio-two owner call starts at {actual}, expected {expected}")]
    UnexpectedTokenStart { actual: usize, expected: usize },
    #[error(
        "ratio-two owner publication ({source_layer}, {epoch}, {call_id}) does not match expected ({expected_source_layer}, {expected_epoch}, {expected_call_id})"
    )]
    UnexpectedPublication {
        source_layer: u16,
        expected_source_layer: u16,
        epoch: u64,
        expected_epoch: u64,
        call_id: u64,
        expected_call_id: u64,
    },
    #[error("ratio-two owner partial call supplied {actual} completed-group frequencies")]
    UnexpectedPartialFrequencies { actual: usize },
    #[error("ratio-two owner key and KV prefix state diverged")]
    DivergentPrefixes,
    #[error("could not allocate {elements} ratio-two owner {field} elements")]
    AllocationFailed {
        field: &'static str,
        elements: usize,
    },
    #[error(transparent)]
    Compressor(#[from] CompressorError),
    #[error(transparent)]
    KeyLayout(#[from] IndexKeyLayoutError),
    #[error(transparent)]
    KvLayout(#[from] CompressedKvLayoutError),
    #[error(transparent)]
    Key(#[from] IndexKeyError),
    #[error(transparent)]
    CompressedKv(#[from] CompressedKvError),
    #[error(transparent)]
    Cache(#[from] IndexKeyStateError),
    #[error(transparent)]
    Projection(#[from] Fp32LinearError),
}

fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, RatioTwoOwnerError> {
    let elements = left
        .checked_mul(right)
        .ok_or(RatioTwoOwnerError::ShapeOverflow { field })?;
    if elements > MAX_FP32_LINEAR_ELEMENTS {
        return Err(RatioTwoOwnerError::ElementLimit { field, elements });
    }
    Ok(elements)
}

fn allocate_f32(elements: usize, field: &'static str) -> Result<Vec<f32>, RatioTwoOwnerError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| RatioTwoOwnerError::AllocationFailed { field, elements })?;
    Ok(values)
}

fn bf16_to_f32_owned(values: &[u16]) -> Result<Vec<f32>, RatioTwoOwnerError> {
    let mut converted = allocate_f32(values.len(), "input FP32")?;
    converted.extend(values.iter().copied().map(bf16_to_f32));
    Ok(converted)
}

fn collect_prefixes(
    state: &impl PrefixState,
    batches: usize,
    field: &'static str,
) -> Result<Vec<Vec<u16>>, RatioTwoOwnerError> {
    let mut prefixes = Vec::new();
    prefixes
        .try_reserve_exact(batches)
        .map_err(|_| RatioTwoOwnerError::AllocationFailed {
            field,
            elements: batches,
        })?;
    for batch in 0..batches {
        let prefix = state.prefix(batch)?;
        let mut copied = Vec::new();
        copied.try_reserve_exact(prefix.len()).map_err(|_| {
            RatioTwoOwnerError::AllocationFailed {
                field,
                elements: prefix.len(),
            }
        })?;
        copied.extend_from_slice(prefix);
        prefixes.push(copied);
    }
    Ok(prefixes)
}

trait PrefixState {
    fn prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError>;
}

impl PrefixState for IndexKeyState {
    fn prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        IndexKeyState::prefix(self, batch)
    }
}

impl PrefixState for CompressedKvState {
    fn prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        CompressedKvState::prefix(self, batch)
    }
}
