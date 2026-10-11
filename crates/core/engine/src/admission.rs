//! Bounded, logical KV-cache reservations for request admission.

use std::{collections::BTreeMap, num::NonZeroU64};

use thiserror::Error;

use crate::kv::KvPagePlan;

/// An opaque identifier for one active logical KV reservation.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct KvReservationId(NonZeroU64);

impl KvReservationId {
    /// Returns the stable numeric representation of this identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// The result of successfully reserving logical KV pages for one request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvReservation {
    id: KvReservationId,
    page_count: u32,
}

impl KvReservation {
    /// Returns the identifier required to release this reservation.
    #[must_use]
    pub const fn id(self) -> KvReservationId {
        self.id
    }

    /// Returns the number of logical KV pages held by this reservation.
    #[must_use]
    pub const fn page_count(self) -> u32 {
        self.page_count
    }
}

/// A finite pool of logical KV pages with explicit request ownership.
#[derive(Debug)]
pub struct KvAdmissionPool {
    plan: KvPagePlan,
    used_pages: u32,
    next_reservation: u64,
    reservations: BTreeMap<KvReservationId, u32>,
}

impl KvAdmissionPool {
    /// Creates an empty admission pool governed by `plan`.
    #[must_use]
    pub fn new(plan: KvPagePlan) -> Self {
        Self {
            plan,
            used_pages: 0,
            next_reservation: 1,
            reservations: BTreeMap::new(),
        }
    }

    /// Returns the logical page plan that bounds this pool.
    #[must_use]
    pub const fn plan(&self) -> KvPagePlan {
        self.plan
    }

    /// Returns the number of logical KV pages currently held by active requests.
    #[must_use]
    pub const fn used_pages(&self) -> u32 {
        self.used_pages
    }

    /// Returns the number of logical KV pages that remain reservable.
    #[must_use]
    pub const fn available_pages(&self) -> u32 {
        self.plan.page_count().saturating_sub(self.used_pages)
    }

    /// Reserves enough whole logical KV pages to hold `tokens`.
    ///
    /// # Errors
    ///
    /// Returns an error when `tokens` is zero or the pool lacks enough pages.
    pub fn reserve_tokens(&mut self, tokens: u32) -> Result<KvReservation, KvAdmissionError> {
        if tokens == 0 {
            return Err(KvAdmissionError::ZeroTokenRequest);
        }

        let requested_pages = self.plan.pages_for(tokens);
        let available_pages = self.available_pages();
        if requested_pages > available_pages {
            return Err(KvAdmissionError::InsufficientCapacity {
                requested_pages,
                available_pages,
            });
        }

        let reservation = KvReservation {
            id: self.allocate_id(),
            page_count: requested_pages,
        };
        self.used_pages = self
            .used_pages
            .checked_add(requested_pages)
            .ok_or(KvAdmissionError::AccountingOverflow)?;
        self.reservations.insert(reservation.id, requested_pages);
        Ok(reservation)
    }

    /// Releases an active reservation exactly once.
    ///
    /// # Errors
    ///
    /// Returns [`KvAdmissionError::UnknownReservation`] for an unknown or already
    /// released identifier.
    pub fn release(&mut self, id: KvReservationId) -> Result<(), KvAdmissionError> {
        let page_count = *self
            .reservations
            .get(&id)
            .ok_or(KvAdmissionError::UnknownReservation(id))?;
        let used_pages = self
            .used_pages
            .checked_sub(page_count)
            .ok_or(KvAdmissionError::AccountingInvariantViolation)?;

        self.reservations.remove(&id);
        self.used_pages = used_pages;
        Ok(())
    }

    fn allocate_id(&mut self) -> KvReservationId {
        loop {
            let value = self.next_reservation;
            self.next_reservation = self.next_reservation.checked_add(1).unwrap_or(1);
            let Some(value) = NonZeroU64::new(value) else {
                continue;
            };
            let id = KvReservationId(value);
            if !self.reservations.contains_key(&id) {
                return id;
            }
        }
    }
}

/// A failed logical KV reservation or release.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum KvAdmissionError {
    /// A request must reserve at least one token.
    #[error("a KV reservation must contain at least one token")]
    ZeroTokenRequest,
    /// The pool has fewer pages than the request requires.
    #[error(
        "insufficient KV capacity: requested {requested_pages} pages, {available_pages} available"
    )]
    InsufficientCapacity {
        /// The full-page demand for the request.
        requested_pages: u32,
        /// The currently unreserved page count.
        available_pages: u32,
    },
    /// The identifier was never reserved or was already released.
    #[error("unknown or already released KV reservation {0:?}")]
    UnknownReservation(KvReservationId),
    /// Internal arithmetic exceeded the bounded logical-page representation.
    #[error("KV reservation accounting overflowed")]
    AccountingOverflow,
    /// Active reservation bookkeeping was internally inconsistent.
    #[error("KV reservation accounting was internally inconsistent")]
    AccountingInvariantViolation,
}

#[cfg(test)]
mod tests {
    use super::{KvAdmissionError, KvAdmissionPool};
    use crate::kv::{KvPagePlan, KvPageTokens};

    fn pool() -> KvAdmissionPool {
        let width = KvPageTokens::new(16).expect("positive page width");
        KvAdmissionPool::new(KvPagePlan::new(width, 3).expect("positive page count"))
    }

    #[test]
    fn reserves_rounded_up_page_demand() {
        let mut pool = pool();

        let reservation = pool.reserve_tokens(17).expect("capacity for two pages");

        assert_eq!(reservation.page_count(), 2);
        assert_eq!(pool.used_pages(), 2);
        assert_eq!(pool.available_pages(), 1);
    }

    #[test]
    fn rejects_zero_tokens_and_excess_capacity_without_mutating_state() {
        let mut pool = pool();
        assert_eq!(
            pool.reserve_tokens(0),
            Err(KvAdmissionError::ZeroTokenRequest)
        );
        assert_eq!(pool.used_pages(), 0);

        assert_eq!(
            pool.reserve_tokens(49),
            Err(KvAdmissionError::InsufficientCapacity {
                requested_pages: 4,
                available_pages: 3,
            })
        );
        assert_eq!(pool.used_pages(), 0);
        assert_eq!(pool.available_pages(), 3);
    }

    #[test]
    fn release_restores_capacity_once() {
        let mut pool = pool();
        let reservation = pool.reserve_tokens(32).expect("capacity for two pages");

        pool.release(reservation.id()).expect("active reservation");
        assert_eq!(pool.used_pages(), 0);
        assert_eq!(pool.available_pages(), 3);
        assert_eq!(
            pool.release(reservation.id()),
            Err(KvAdmissionError::UnknownReservation(reservation.id()))
        );
    }

    #[test]
    fn reservations_have_distinct_ids_and_cannot_exceed_pool_capacity() {
        let mut pool = pool();
        let first = pool.reserve_tokens(1).expect("first page");
        let second = pool.reserve_tokens(16).expect("second page");
        let third = pool.reserve_tokens(16).expect("third page");

        assert_ne!(first.id(), second.id());
        assert_ne!(second.id(), third.id());
        assert_eq!(pool.available_pages(), 0);
        assert!(matches!(
            pool.reserve_tokens(1),
            Err(KvAdmissionError::InsufficientCapacity {
                available_pages: 0,
                ..
            })
        ));
    }
}
