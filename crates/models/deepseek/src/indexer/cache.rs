//! Per-request ownership of V4.1 owner-layer prefixes.
//!
//! The upstream indexer keeps one key cache at the layer which owns compressed
//! KV. [`IndexKeyState`] and [`CompressedKvState`] represent the two distinct,
//! model-local prefixes: prepared index keys and compressed KV values. They
//! deliberately do not decide when a token group completes, prepare values,
//! choose candidates, or select indices. Their crate-private pending operations
//! let one owner validate both publications before either cache mutates.

use std::num::NonZeroUsize;

use thiserror::Error;

mod storage;

use storage::{PrefixStore, StorePendingAppend, StorePendingReset};

/// Largest bounded index-key cache allocation, in BF16 elements.
pub const MAX_INDEX_KEY_CACHE_ELEMENTS: usize = 1 << 20;

/// Largest bounded compressed-KV cache allocation, in BF16 elements.
///
/// This has the same model-local storage ceiling as index keys. It is a
/// separate constant so a caller need not infer a compressed-KV limit from an
/// index-key type.
pub const MAX_COMPRESSED_KV_CACHE_ELEMENTS: usize = MAX_INDEX_KEY_CACHE_ELEMENTS;

/// Identifies one source publication in a request-local owner-prefix stream.
///
/// These values are grouped because an epoch or successful-call ordinal is not
/// an interchangeable position. `source_layer` is the layer that owns the key
/// projection and compressed-KV preparation, not a candidate or selection
/// producer. The name is retained because it was already the public identity
/// type for the index-key cache.
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

    /// The owner layer that prepared these prefixes.
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
    store: PrefixStore,
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
        Ok(Self {
            store: PrefixStore::new(
                batches,
                key_dimension,
                capacity_positions,
                expected_source_layer,
                "index-key cache",
            )?,
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
        self.prepare_append(publication, start_position, keys)?
            .commit();
        Ok(())
    }

    /// Validates an append and holds an exclusive cache borrow until commit.
    ///
    /// This is crate-private so an owner can prepare one index-key append and
    /// one compressed-KV append before committing either. Its commit
    /// performs only bounded copies and scalar assignments.
    pub(crate) fn prepare_append<'state, 'values>(
        &'state mut self,
        publication: IndexKeyPublicationId,
        start_position: usize,
        keys: &'values [u16],
    ) -> Result<PendingAppend<'state, 'values>, IndexKeyStateError> {
        Ok(PendingAppend {
            pending: self
                .store
                .prepare_append(publication, start_position, keys)?,
        })
    }

    /// Invalidates borrowed prefixes and begins a new checked epoch.
    pub fn reset(&mut self) -> Result<(), IndexKeyStateError> {
        self.prepare_reset()?.commit();
        Ok(())
    }

    /// Validates a reset before mutating this cache.
    pub(crate) fn prepare_reset(&mut self) -> Result<PendingReset<'_>, IndexKeyStateError> {
        Ok(PendingReset {
            pending: self.store.prepare_reset()?,
        })
    }

    /// Borrows one batch's exact valid `[compressed_position, key_dimension]` prefix.
    pub fn prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        self.store.prefix(batch)
    }

    /// The only source layer accepted by [`append_prepared`](Self::append_prepared).
    #[must_use]
    pub const fn expected_source_layer(&self) -> u16 {
        self.store.expected_source_layer()
    }

    /// The current request-local epoch.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.store.epoch()
    }

    /// The successful-call ordinal required by the next append.
    #[must_use]
    pub const fn next_call_id(&self) -> u64 {
        self.store.next_call_id()
    }

    /// Number of valid compressed positions in each batch prefix.
    #[must_use]
    pub const fn valid_positions(&self) -> usize {
        self.store.valid_positions()
    }

    #[cfg(test)]
    fn set_counters_for_test(&mut self, epoch: u64, next_call_id: u64) {
        self.store.set_counters_for_test(epoch, next_call_id);
    }
}

/// A bounded, per-request owner of prepared V4.1 compressed KV values.
///
/// Storage and publication metadata are intentionally parallel to
/// [`IndexKeyState`] while the values remain a separate cache: key and KV
/// widths, contents, and consumers are not interchangeable. The shared
/// [`IndexKeyStateError`] names a storage invariant, not an assertion that the
/// failed values were index keys.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompressedKvState {
    store: PrefixStore,
}

impl CompressedKvState {
    /// Creates an empty epoch-zero compressed-KV prefix with fixed source identity.
    pub fn new(
        batches: NonZeroUsize,
        value_dimension: NonZeroUsize,
        capacity_positions: NonZeroUsize,
        expected_source_layer: u16,
    ) -> Result<Self, IndexKeyStateError> {
        Ok(Self {
            store: PrefixStore::new(
                batches,
                value_dimension,
                capacity_positions,
                expected_source_layer,
                "compressed-KV cache",
            )?,
        })
    }

    /// Appends prepared BF16 `[batch, compressed_position, value_dimension]` values.
    ///
    /// An empty slice advances the successful source call ordinal without
    /// exposing capacity padding, matching [`IndexKeyState::append_prepared`].
    pub fn append_prepared(
        &mut self,
        publication: IndexKeyPublicationId,
        start_position: usize,
        values: &[u16],
    ) -> Result<(), IndexKeyStateError> {
        self.prepare_append(publication, start_position, values)?
            .commit();
        Ok(())
    }

    /// Validates a compressed-KV append without mutating the prefix.
    pub(crate) fn prepare_append<'state, 'values>(
        &'state mut self,
        publication: IndexKeyPublicationId,
        start_position: usize,
        values: &'values [u16],
    ) -> Result<PendingAppend<'state, 'values>, IndexKeyStateError> {
        Ok(PendingAppend {
            pending: self
                .store
                .prepare_append(publication, start_position, values)?,
        })
    }

    /// Begins a new checked epoch and invalidates all borrowed prefixes.
    pub fn reset(&mut self) -> Result<(), IndexKeyStateError> {
        self.prepare_reset()?.commit();
        Ok(())
    }

    /// Validates a compressed-KV reset without mutating the prefix.
    pub(crate) fn prepare_reset(&mut self) -> Result<PendingReset<'_>, IndexKeyStateError> {
        Ok(PendingReset {
            pending: self.store.prepare_reset()?,
        })
    }

    /// Borrows one batch's exact valid `[compressed_position, value_dimension]` prefix.
    pub fn prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        self.store.prefix(batch)
    }

    /// The owner layer this cache accepts.
    #[must_use]
    pub const fn expected_source_layer(&self) -> u16 {
        self.store.expected_source_layer()
    }

    /// Current request-local epoch.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.store.epoch()
    }

    /// Successful-call ordinal required by the next append.
    #[must_use]
    pub const fn next_call_id(&self) -> u64 {
        self.store.next_call_id()
    }

    /// Valid compressed positions in each batch prefix.
    #[must_use]
    pub const fn valid_positions(&self) -> usize {
        self.store.valid_positions()
    }
}

/// A validated append that remains invisible until commit.
///
/// Constructed only inside this module; it borrows its cache mutably and its
/// input immutably, preventing an owner from changing either between prepare
/// and the infallible commit.
pub(crate) struct PendingAppend<'state, 'values> {
    pending: StorePendingAppend<'state, 'values>,
}

impl PendingAppend<'_, '_> {
    /// Copies validated values into their already-bounded destination and advances metadata.
    pub(crate) fn commit(self) {
        self.pending.commit();
    }
}

/// A validated reset that remains invisible until commit.
pub(crate) struct PendingReset<'state> {
    pending: StorePendingReset<'state>,
}

impl PendingReset<'_> {
    /// Clears the bounded prefix and advances its already-checked epoch.
    pub(crate) fn commit(self) {
        self.pending.commit();
    }
}

/// Rejected construction, publication, or prefix access for a bounded owner prefix.
///
/// Kept under its original public name for compatibility. Both
/// [`IndexKeyState`] and [`CompressedKvState`] use these storage invariants.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexKeyStateError {
    #[error("owner-prefix cache shape arithmetic overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("owner-prefix cache has {elements} BF16 elements, maximum is {maximum}")]
    ElementLimit { elements: usize, maximum: usize },
    #[error("could not allocate {elements} BF16 owner-prefix cache elements")]
    AllocationFailed { elements: usize },
    #[error(
        "owner-prefix publication source layer {actual} is not expected source layer {expected}"
    )]
    UnexpectedSourceLayer { actual: u16, expected: u16 },
    #[error("owner-prefix publication epoch {actual} is not expected epoch {expected}")]
    UnexpectedEpoch { actual: u64, expected: u64 },
    #[error("owner-prefix publication call {actual} is not expected call {expected}")]
    UnexpectedCallId { actual: u64, expected: u64 },
    #[error(
        "prepared owner-prefix start position {actual} is not contiguous with valid prefix {expected}"
    )]
    PositionDiscontinuity { actual: usize, expected: usize },
    #[error(
        "prepared owner-prefix length {actual} is not a multiple of batch/value stride {stride}"
    )]
    PreparedLength { actual: usize, stride: usize },
    #[error("prepared owner-prefix position count overflowed usize")]
    PositionOverflow,
    #[error(
        "prepared owner-prefix values end at compressed position {end_position}, capacity is {capacity}"
    )]
    CapacityExceeded {
        end_position: usize,
        capacity: usize,
    },
    #[error("prepared BF16 owner-prefix value at flat input position {position} is not finite")]
    NonFinitePrepared { position: usize },
    #[error("owner-prefix publication call ordinal overflowed")]
    CallIdOverflow,
    #[error("owner-prefix epoch counter overflowed")]
    EpochOverflow,
    #[error("batch {batch} is outside owner-prefix cache batch count {batches}")]
    BatchOutOfRange { batch: usize, batches: usize },
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        CompressedKvState, IndexKeyPublicationId, IndexKeyState, IndexKeyStateError,
        MAX_INDEX_KEY_CACHE_ELEMENTS,
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
        epoch_state.set_counters_for_test(u64::MAX, 0);
        let epoch_before = epoch_state.clone();
        assert_eq!(epoch_state.reset(), Err(IndexKeyStateError::EpochOverflow));
        assert_eq!(epoch_state.epoch(), u64::MAX);
        assert_eq!(epoch_state.next_call_id(), 0);
        assert!(epoch_state.prefix(0).expect("unchanged prefix").is_empty());
        assert_eq!(epoch_state, epoch_before);

        let mut call_state = state();
        call_state.set_counters_for_test(0, u64::MAX);
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

    #[test]
    fn compressed_kv_prefixes_are_distinct_but_follow_the_same_stream_contract() {
        let mut state =
            CompressedKvState::new(nz(2), nz(3), nz(3), 3).expect("bounded compressed KV state");
        state
            .append_prepared(
                publication(0, 0),
                0,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )
            .expect("two batch-major positions");
        assert_eq!(state.prefix(0).expect("first KV batch"), [1, 2, 3, 4, 5, 6]);
        assert_eq!(
            state.prefix(1).expect("second KV batch"),
            [7, 8, 9, 10, 11, 12]
        );
        state
            .append_prepared(publication(0, 1), 2, &[])
            .expect("incomplete call");
        assert_eq!(state.next_call_id(), 2);
        state.reset().expect("checked reset");
        assert_eq!(state.epoch(), 1);
        assert!(state.prefix(0).expect("reset prefix").is_empty());
    }

    #[test]
    fn paired_append_can_reject_second_cache_without_publishing_first() {
        let mut keys = state();
        let mut kv =
            CompressedKvState::new(nz(2), nz(3), nz(3), 3).expect("bounded compressed KV state");
        let key_before = keys.clone();
        let kv_before = kv.clone();

        {
            let _pending_keys = keys
                .prepare_append(publication(0, 0), 0, &[1, 2, 3, 4])
                .expect("key append is fully prepared");
            assert!(matches!(
                kv.prepare_append(publication(0, 0), 0, &[1, 2, 3, 4, 5, 0x7f80]),
                Err(IndexKeyStateError::NonFinitePrepared { position: 5 })
            ));
        }

        assert_eq!(keys, key_before);
        assert_eq!(kv, kv_before);
    }

    #[test]
    fn paired_prepared_appends_commit_without_a_second_validation_step() {
        let mut keys = state();
        let mut kv =
            CompressedKvState::new(nz(2), nz(3), nz(3), 3).expect("bounded compressed KV state");
        let pending_keys = keys
            .prepare_append(publication(0, 0), 0, &[1, 2, 3, 4])
            .expect("validated key append");
        let pending_kv = kv
            .prepare_append(
                publication(0, 0),
                0,
                &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            )
            .expect("validated KV append");

        pending_keys.commit();
        pending_kv.commit();
        assert_eq!(keys.next_call_id(), 1);
        assert_eq!(kv.next_call_id(), 1);
        assert_eq!(keys.prefix(0).expect("key prefix"), [1, 2]);
        assert_eq!(kv.prefix(1).expect("KV prefix"), [7, 8, 9, 10, 11, 12]);
    }
}
