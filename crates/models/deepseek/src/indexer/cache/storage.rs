//! Small shared backing store for V4.1 owner-layer BF16 prefixes.
//!
//! This is deliberately private to the indexer cache module. It is not a
//! generic engine cache: the two public wrappers retain their distinct key/KV
//! meanings while sharing only bounded batch-major storage invariants.

use std::num::NonZeroUsize;

use crate::precision::bf16_to_f32;

use super::{IndexKeyPublicationId, IndexKeyStateError, MAX_INDEX_KEY_CACHE_ELEMENTS};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PrefixStore {
    batches: NonZeroUsize,
    value_dimension: NonZeroUsize,
    capacity_positions: NonZeroUsize,
    expected_source_layer: u16,
    values: Vec<u16>,
    valid_positions: usize,
    epoch: u64,
    next_call_id: u64,
}

impl PrefixStore {
    pub(super) fn new(
        batches: NonZeroUsize,
        value_dimension: NonZeroUsize,
        capacity_positions: NonZeroUsize,
        expected_source_layer: u16,
        allocation_field: &'static str,
    ) -> Result<Self, IndexKeyStateError> {
        let elements = product(
            &[
                batches.get(),
                capacity_positions.get(),
                value_dimension.get(),
            ],
            allocation_field,
        )?;
        if elements > MAX_INDEX_KEY_CACHE_ELEMENTS {
            return Err(IndexKeyStateError::ElementLimit {
                elements,
                maximum: MAX_INDEX_KEY_CACHE_ELEMENTS,
            });
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(elements)
            .map_err(|_| IndexKeyStateError::AllocationFailed { elements })?;
        values.resize(elements, 0);
        Ok(Self {
            batches,
            value_dimension,
            capacity_positions,
            expected_source_layer,
            values,
            valid_positions: 0,
            epoch: 0,
            next_call_id: 0,
        })
    }

    pub(super) fn prepare_append<'state, 'values>(
        &'state mut self,
        publication: IndexKeyPublicationId,
        start_position: usize,
        values: &'values [u16],
    ) -> Result<StorePendingAppend<'state, 'values>, IndexKeyStateError> {
        self.validate_publication(publication)?;
        if start_position != self.valid_positions {
            return Err(IndexKeyStateError::PositionDiscontinuity {
                actual: start_position,
                expected: self.valid_positions,
            });
        }
        let batch_stride = product(
            &[self.batches.get(), self.value_dimension.get()],
            "prepared owner-prefix row",
        )?;
        if !values.len().is_multiple_of(batch_stride) {
            return Err(IndexKeyStateError::PreparedLength {
                actual: values.len(),
                stride: batch_stride,
            });
        }
        let positions = values.len() / batch_stride;
        let end_position = start_position
            .checked_add(positions)
            .ok_or(IndexKeyStateError::PositionOverflow)?;
        if end_position > self.capacity_positions.get() {
            return Err(IndexKeyStateError::CapacityExceeded {
                end_position,
                capacity: self.capacity_positions.get(),
            });
        }
        if let Some(position) = values
            .iter()
            .position(|&bits| !bf16_to_f32(bits).is_finite())
        {
            return Err(IndexKeyStateError::NonFinitePrepared { position });
        }
        let next_call_id = self
            .next_call_id
            .checked_add(1)
            .ok_or(IndexKeyStateError::CallIdOverflow)?;
        let position_width = product(
            &[positions, self.value_dimension.get()],
            "prepared owner-prefix span",
        )?;
        let destination_width = product(
            &[self.capacity_positions.get(), self.value_dimension.get()],
            "owner-prefix cache batch",
        )?;
        let destination_offset = product(
            &[start_position, self.value_dimension.get()],
            "prepared owner-prefix offset",
        )?;

        Ok(StorePendingAppend {
            store: self,
            values,
            position_width,
            destination_width,
            destination_offset,
            end_position,
            next_call_id,
        })
    }

    pub(super) fn prepare_reset(&mut self) -> Result<StorePendingReset<'_>, IndexKeyStateError> {
        let epoch = self
            .epoch
            .checked_add(1)
            .ok_or(IndexKeyStateError::EpochOverflow)?;
        Ok(StorePendingReset { store: self, epoch })
    }

    pub(super) fn prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        if batch >= self.batches.get() {
            return Err(IndexKeyStateError::BatchOutOfRange {
                batch,
                batches: self.batches.get(),
            });
        }
        let batch_width = product(
            &[self.capacity_positions.get(), self.value_dimension.get()],
            "owner-prefix cache batch",
        )?;
        let valid_width = product(
            &[self.valid_positions, self.value_dimension.get()],
            "valid owner-prefix",
        )?;
        let start = batch * batch_width;
        Ok(&self.values[start..start + valid_width])
    }

    pub(super) const fn expected_source_layer(&self) -> u16 {
        self.expected_source_layer
    }

    pub(super) const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub(super) const fn next_call_id(&self) -> u64 {
        self.next_call_id
    }

    pub(super) const fn valid_positions(&self) -> usize {
        self.valid_positions
    }

    #[cfg(test)]
    pub(super) fn set_counters_for_test(&mut self, epoch: u64, next_call_id: u64) {
        self.epoch = epoch;
        self.next_call_id = next_call_id;
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

pub(super) struct StorePendingAppend<'state, 'values> {
    store: &'state mut PrefixStore,
    values: &'values [u16],
    position_width: usize,
    destination_width: usize,
    destination_offset: usize,
    end_position: usize,
    next_call_id: u64,
}

impl StorePendingAppend<'_, '_> {
    pub(super) fn commit(self) {
        for batch in 0..self.store.batches.get() {
            let source_start = batch * self.position_width;
            let destination_start = batch * self.destination_width + self.destination_offset;
            self.store.values[destination_start..destination_start + self.position_width]
                .copy_from_slice(&self.values[source_start..source_start + self.position_width]);
        }
        self.store.valid_positions = self.end_position;
        self.store.next_call_id = self.next_call_id;
    }
}

pub(super) struct StorePendingReset<'state> {
    store: &'state mut PrefixStore,
    epoch: u64,
}

impl StorePendingReset<'_> {
    pub(super) fn commit(self) {
        self.store.values.fill(0);
        self.store.valid_positions = 0;
        self.store.epoch = self.epoch;
        self.store.next_call_id = 0;
    }
}

fn product(values: &[usize], field: &'static str) -> Result<usize, IndexKeyStateError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(IndexKeyStateError::ShapeOverflow { field })
    })
}
