//! Per-request ownership of V4.1 prepared index keys.
//!
//! The upstream indexer keeps one key cache at the layer which owns compressed
//! KV.  This type represents that narrow, model-local responsibility: append
//! already prepared BF16 keys at contiguous *compressed-position* offsets and
//! lend valid prefixes to a consumer.  It deliberately does not decide when a
//! token group completes, prepare keys, choose candidates, select indices, or
//! coordinate another cache.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::precision::bf16_to_f32;

/// Largest bounded index-key cache allocation, in BF16 elements.
pub const MAX_INDEX_KEY_CACHE_ELEMENTS: usize = 1 << 20;

/// Identifies one source publication in a request-local index-key stream.
///
/// These values are grouped because an epoch or successful-call ordinal is not
/// an interchangeable position. `source_layer` is the layer that owns the key
/// projection, not a candidate or selection producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct IndexKeyPublicationId {
    source_layer: u16,
    epoch: u64,
    call_id: u64,
}

impl IndexKeyPublicationId {
    /// Creates source identity and request-local ordering metadata.
    #[must_use]
    pub const fn new(source_layer: u16, epoch: u64, call_id: u64) -> Self {
        Self {
            source_layer,
            epoch,
            call_id,
        }
    }

    /// The owner layer that prepared these keys.
    #[must_use]
    pub const fn source_layer(self) -> u16 {
        self.source_layer
    }

    /// The request-local epoch.
    #[must_use]
    pub const fn epoch(self) -> u64 {
        self.epoch
    }

    /// The source's successful-call ordinal in this epoch.
    #[must_use]
    pub const fn call_id(self) -> u64 {
        self.call_id
    }
}

/// A bounded, per-request owner of prepared V4.1 index keys.
///
/// Storage is batch-major `[batch, capacity_position, key_dimension]`.
/// [`prefix`](Self::prefix) intentionally exposes only the valid prefix for a
/// batch, so callers cannot accidentally score uninitialized capacity padding.
/// The state has no global registry and is independent of attention's KV
/// window.  A caller must translate token positions and incomplete compression
/// groups to the `start_position` and prepared key slice supplied here.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexKeyState {
    batches: NonZeroUsize,
    key_dimension: NonZeroUsize,
    capacity_positions: NonZeroUsize,
    expected_source_layer: u16,
    keys: Vec<u16>,
    valid_positions: usize,
    epoch: u64,
    next_call_id: u64,
}

impl IndexKeyState {
    /// Creates an empty epoch-zero cache with a fixed source identity.
    ///
    /// The allocation is checked and bounded before it is reserved.  This
    /// performs no model loading and does not establish an attention cache.
    pub fn new(
        batches: NonZeroUsize,
        key_dimension: NonZeroUsize,
        capacity_positions: NonZeroUsize,
        expected_source_layer: u16,
    ) -> Result<Self, IndexKeyStateError> {
        let elements = product(
            &[batches.get(), capacity_positions.get(), key_dimension.get()],
            "index-key cache",
        )?;
        if elements > MAX_INDEX_KEY_CACHE_ELEMENTS {
            return Err(IndexKeyStateError::ElementLimit {
                elements,
                maximum: MAX_INDEX_KEY_CACHE_ELEMENTS,
            });
        }
        let mut keys = Vec::new();
        keys.try_reserve_exact(elements)
            .map_err(|_| IndexKeyStateError::AllocationFailed { elements })?;
        keys.resize(elements, 0);
        Ok(Self {
            batches,
            key_dimension,
            capacity_positions,
            expected_source_layer,
            keys,
            valid_positions: 0,
            epoch: 0,
            next_call_id: 0,
        })
    }

    /// Appends prepared BF16 `[batch, compressed_position, key_dimension]` keys.
    ///
    /// `start_position` is in compressed-position units, never token units.
    /// An empty `keys` slice is a valid no-key publication for an incomplete
    /// token group: it preserves the valid prefix while advancing the successful
    /// call ordinal.  All validation happens before copying, so an error leaves
    /// this state unchanged.
    pub fn append_prepared(
        &mut self,
        publication: IndexKeyPublicationId,
        start_position: usize,
        keys: &[u16],
    ) -> Result<(), IndexKeyStateError> {
        self.validate_publication(publication)?;
        if start_position != self.valid_positions {
            return Err(IndexKeyStateError::PositionDiscontinuity {
                actual: start_position,
                expected: self.valid_positions,
            });
        }

        let batch_stride = product(
            &[self.batches.get(), self.key_dimension.get()],
            "prepared index-key row",
        )?;
        if !keys.len().is_multiple_of(batch_stride) {
            return Err(IndexKeyStateError::PreparedLength {
                actual: keys.len(),
                stride: batch_stride,
            });
        }
        let positions = keys.len() / batch_stride;
        let end_position = start_position
            .checked_add(positions)
            .ok_or(IndexKeyStateError::PositionOverflow)?;
        if end_position > self.capacity_positions.get() {
            return Err(IndexKeyStateError::CapacityExceeded {
                end_position,
                capacity: self.capacity_positions.get(),
            });
        }
        if let Some(position) = keys.iter().position(|&bits| !bf16_to_f32(bits).is_finite()) {
            return Err(IndexKeyStateError::NonFinitePrepared { position });
        }
        let next_call_id = self
            .next_call_id
            .checked_add(1)
            .ok_or(IndexKeyStateError::CallIdOverflow)?;

        let position_width = product(&[positions, self.key_dimension.get()], "prepared key span")?;
        let destination_width = product(
            &[self.capacity_positions.get(), self.key_dimension.get()],
            "index-key cache batch",
        )?;
        let destination_offset = product(
            &[start_position, self.key_dimension.get()],
            "prepared key offset",
        )?;
        for batch in 0..self.batches.get() {
            let source_start = batch * position_width;
            let destination_start = batch * destination_width + destination_offset;
            self.keys[destination_start..destination_start + position_width]
                .copy_from_slice(&keys[source_start..source_start + position_width]);
        }
        self.valid_positions = end_position;
        self.next_call_id = next_call_id;
        Ok(())
    }

    /// Invalidates borrowed prefixes and begins a new checked epoch.
    pub fn reset(&mut self) -> Result<(), IndexKeyStateError> {
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(IndexKeyStateError::EpochOverflow)?;
        self.keys.fill(0);
        self.valid_positions = 0;
        self.epoch = epoch;
        self.next_call_id = 0;
        Ok(())
    }

    /// Borrows one batch's exact valid `[compressed_position, key_dimension]` prefix.
    pub fn prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        if batch >= self.batches.get() {
            return Err(IndexKeyStateError::BatchOutOfRange {
                batch,
                batches: self.batches.get(),
            });
        }
        let batch_width = product(
            &[self.capacity_positions.get(), self.key_dimension.get()],
            "index-key cache batch",
        )?;
        let valid_width = product(
            &[self.valid_positions, self.key_dimension.get()],
            "valid index-key prefix",
        )?;
        let start = batch * batch_width;
        Ok(&self.keys[start..start + valid_width])
    }

    /// The only source layer accepted by [`append_prepared`](Self::append_prepared).
    #[must_use]
    pub const fn expected_source_layer(&self) -> u16 {
        self.expected_source_layer
    }

    /// The current request-local epoch.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The successful-call ordinal required by the next append.
    #[must_use]
    pub const fn next_call_id(&self) -> u64 {
        self.next_call_id
    }

    /// Number of valid compressed positions in each batch prefix.
    #[must_use]
    pub const fn valid_positions(&self) -> usize {
        self.valid_positions
    }

    fn validate_publication(
        &self,
        publication: IndexKeyPublicationId,
    ) -> Result<(), IndexKeyStateError> {
        if publication.source_layer != self.expected_source_layer {
            return Err(IndexKeyStateError::UnexpectedSourceLayer {
                actual: publication.source_layer,
                expected: self.expected_source_layer,
            });
        }
        if publication.epoch != self.epoch {
            return Err(IndexKeyStateError::UnexpectedEpoch {
                actual: publication.epoch,
                expected: self.epoch,
            });
        }
        if publication.call_id != self.next_call_id {
            return Err(IndexKeyStateError::UnexpectedCallId {
                actual: publication.call_id,
                expected: self.next_call_id,
            });
        }
        Ok(())
    }
}

/// Rejected construction, publication, or prefix access for [`IndexKeyState`].
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexKeyStateError {
    #[error("index-key cache shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("index-key cache has {elements} BF16 elements, maximum is {maximum}")]
    ElementLimit { elements: usize, maximum: usize },
    #[error("could not allocate {elements} BF16 index-key cache elements")]
    AllocationFailed { elements: usize },
    #[error("index-key publication source layer {actual} is not expected source layer {expected}")]
    UnexpectedSourceLayer { actual: u16, expected: u16 },
    #[error("index-key publication epoch {actual} is not expected epoch {expected}")]
    UnexpectedEpoch { actual: u64, expected: u64 },
    #[error("index-key publication call {actual} is not expected call {expected}")]
    UnexpectedCallId { actual: u64, expected: u64 },
    #[error(
        "prepared index-key start position {actual} is not contiguous with valid prefix {expected}"
    )]
    PositionDiscontinuity { actual: usize, expected: usize },
    #[error("prepared index-key length {actual} is not a multiple of batch/key stride {stride}")]
    PreparedLength { actual: usize, stride: usize },
    #[error("prepared index-key position count overflowed usize")]
    PositionOverflow,
    #[error(
        "prepared index keys end at compressed position {end_position}, capacity is {capacity}"
    )]
    CapacityExceeded {
        end_position: usize,
        capacity: usize,
    },
    #[error("prepared BF16 index key at flat input position {position} is not finite")]
    NonFinitePrepared { position: usize },
    #[error("index-key publication call ordinal overflowed")]
    CallIdOverflow,
    #[error("index-key epoch counter overflowed")]
    EpochOverflow,
    #[error("batch {batch} is outside index-key cache batch count {batches}")]
    BatchOutOfRange { batch: usize, batches: usize },
}

fn product(values: &[usize], field: &'static str) -> Result<usize, IndexKeyStateError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(IndexKeyStateError::ShapeOverflow { field })
    })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        IndexKeyPublicationId, IndexKeyState, IndexKeyStateError, MAX_INDEX_KEY_CACHE_ELEMENTS,
    };

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("nonzero test dimension")
    }

    fn state() -> IndexKeyState {
        IndexKeyState::new(nz(2), nz(2), nz(3), 3).expect("bounded state")
    }

    fn publication(epoch: u64, call_id: u64) -> IndexKeyPublicationId {
        IndexKeyPublicationId::new(3, epoch, call_id)
    }

    #[test]
    fn constructor_rejects_element_limits_and_shape_overflow() {
        assert_eq!(
            IndexKeyState::new(nz(1), nz(1), nz(MAX_INDEX_KEY_CACHE_ELEMENTS + 1), 3),
            Err(IndexKeyStateError::ElementLimit {
                elements: MAX_INDEX_KEY_CACHE_ELEMENTS + 1,
                maximum: MAX_INDEX_KEY_CACHE_ELEMENTS,
            })
        );
        assert_eq!(
            IndexKeyState::new(nz(usize::MAX), nz(2), nz(1), 3),
            Err(IndexKeyStateError::ShapeOverflow {
                field: "index-key cache",
            })
        );
    }

    #[test]
    fn appends_batch_major_keys_and_lends_only_valid_prefixes() {
        let mut state = state();
        state
            .append_prepared(publication(0, 0), 0, &[1, 2, 3, 4, 5, 6, 7, 8])
            .expect("two positions for each batch");
        state
            .append_prepared(publication(0, 1), 2, &[9, 10, 11, 12])
            .expect("one position for each batch");
        assert_eq!(state.valid_positions(), 3);
        assert_eq!(state.prefix(0).expect("first batch"), [1, 2, 3, 4, 9, 10]);
        assert_eq!(state.prefix(1).expect("second batch"), [5, 6, 7, 8, 11, 12]);
    }

    #[test]
    fn invalid_publications_are_atomic() {
        let mut state = state();
        state
            .append_prepared(publication(0, 0), 0, &[1, 2, 3, 4])
            .expect("initial append");
        let before = state.clone();
        assert_eq!(
            state.append_prepared(IndexKeyPublicationId::new(4, 0, 1), 1, &[5, 6, 7, 8]),
            Err(IndexKeyStateError::UnexpectedSourceLayer {
                actual: 4,
                expected: 3,
            })
        );
        assert_eq!(
            state.prefix(0).expect("unchanged prefix"),
            before.prefix(0).expect("before")
        );
        assert_eq!(state.valid_positions(), before.valid_positions());
        assert_eq!(state.next_call_id(), before.next_call_id());
        assert_eq!(state, before);

        assert_eq!(
            state.append_prepared(publication(0, 1), 1, &[0x7fc0, 6, 7, 8]),
            Err(IndexKeyStateError::NonFinitePrepared { position: 0 })
        );
        assert_eq!(
            state.prefix(0).expect("still unchanged"),
            before.prefix(0).expect("before")
        );
        assert_eq!(state.next_call_id(), before.next_call_id());
        assert_eq!(state, before);
        state
            .append_prepared(publication(0, 1), 1, &[5, 6, 7, 8])
            .expect("same ordinal retries after rejected input");
    }

    #[test]
    fn wrong_ordinal_and_prepared_length_are_atomic() {
        let mut state = state();
        state
            .append_prepared(publication(0, 0), 0, &[1, 2, 3, 4])
            .expect("initial append");
        let before = state.prefix(0).expect("before").to_vec();
        assert_eq!(
            state.append_prepared(publication(0, 2), 1, &[5, 6, 7, 8]),
            Err(IndexKeyStateError::UnexpectedCallId {
                actual: 2,
                expected: 1,
            })
        );
        assert_eq!(state.prefix(0).expect("ordinal failure unchanged"), before);
        assert_eq!(state.next_call_id(), 1);
        let before_state = state.clone();
        assert_eq!(
            state.append_prepared(publication(0, 1), 1, &[5, 6, 7]),
            Err(IndexKeyStateError::PreparedLength {
                actual: 3,
                stride: 4,
            })
        );
        assert_eq!(state.prefix(0).expect("shape failure unchanged"), before);
        assert_eq!(state.next_call_id(), 1);
        assert_eq!(state, before_state);
        state
            .append_prepared(publication(0, 1), 1, &[5, 6, 7, 8])
            .expect("same ordinal retries after rejected shape");
    }

    #[test]
    fn empty_incomplete_group_advances_ordinal_without_padding_visibility() {
        let mut state = state();
        state
            .append_prepared(publication(0, 0), 0, &[])
            .expect("incomplete group is a successful no-key publication");
        assert_eq!(state.valid_positions(), 0);
        assert_eq!(state.next_call_id(), 1);
        assert!(state.prefix(0).expect("empty valid prefix").is_empty());
    }

    #[test]
    fn reset_requires_a_new_epoch_and_clears_prefix() {
        let mut state = state();
        state
            .append_prepared(publication(0, 0), 0, &[1, 2, 3, 4])
            .expect("initial append");
        state.reset().expect("epoch increments");
        assert_eq!(state.epoch(), 1);
        assert_eq!(state.next_call_id(), 0);
        assert!(state.prefix(0).expect("cleared prefix").is_empty());
        assert_eq!(
            state.append_prepared(publication(0, 0), 0, &[1, 2, 3, 4]),
            Err(IndexKeyStateError::UnexpectedEpoch {
                actual: 0,
                expected: 1,
            })
        );
        state
            .append_prepared(publication(1, 0), 0, &[1, 2, 3, 4])
            .expect("new epoch append");
    }

    #[test]
    fn capacity_and_prefix_batch_errors_are_rejected() {
        let mut state = state();
        assert_eq!(
            state.append_prepared(publication(0, 0), 3, &[1, 2, 3, 4]),
            Err(IndexKeyStateError::PositionDiscontinuity {
                actual: 3,
                expected: 0,
            })
        );
        assert_eq!(
            state.prefix(2),
            Err(IndexKeyStateError::BatchOutOfRange {
                batch: 2,
                batches: 2,
            })
        );
        state
            .append_prepared(
                publication(0, 0),
                0,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )
            .expect("fill cache exactly");
        assert_eq!(
            state.append_prepared(publication(0, 1), 3, &[13, 14, 15, 16]),
            Err(IndexKeyStateError::CapacityExceeded {
                end_position: 4,
                capacity: 3,
            })
        );
    }

    #[test]
    fn counter_overflows_are_atomic() {
        let mut epoch_state = state();
        epoch_state.epoch = u64::MAX;
        let epoch_before = epoch_state.clone();
        assert_eq!(epoch_state.reset(), Err(IndexKeyStateError::EpochOverflow));
        assert_eq!(epoch_state.epoch(), u64::MAX);
        assert_eq!(epoch_state.next_call_id(), 0);
        assert!(epoch_state.prefix(0).expect("unchanged prefix").is_empty());
        assert_eq!(epoch_state, epoch_before);

        let mut call_state = state();
        call_state.next_call_id = u64::MAX;
        let call_before = call_state.clone();
        assert_eq!(
            call_state.append_prepared(publication(0, u64::MAX), 0, &[1, 2, 3, 4]),
            Err(IndexKeyStateError::CallIdOverflow)
        );
        assert_eq!(call_state.valid_positions(), 0);
        assert_eq!(call_state.next_call_id(), u64::MAX);
        assert!(call_state.prefix(0).expect("unchanged prefix").is_empty());
        assert_eq!(call_state, call_before);
    }
}
