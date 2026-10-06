//! Host-side accounting for a paged KV pool: block tables, reference counts,
//! an LRU free queue, and hash-chained prefix caching.
//!
//! The pool holds a fixed number of blocks, sized in slabs of
//! [`SLAB_BLOCKS`]; a device side may store one tensor per slab or one for
//! the whole pool. A block
//! holds `block_tokens` positions (16 by default). This module never touches
//! device memory; it hands the device side block IDs, slot positions and
//! copy-on-write copies to perform.
//!
//! The design follows vLLM v1's `BlockPool` and `KVCacheManager` (also ported
//! to Rust by mistral.rs): a block whose reference count drops to zero keeps
//! its hash and goes on the free queue, and it is evicted from the prefix
//! cache only when the queue hands it out again. Cached prefixes therefore
//! occupy only otherwise-free blocks and never reduce admission capacity.
//!
//! # Call sequence
//!
//! The scheduler owns one [`BlockManager`] on the engine thread.
//!
//! 1. Admission. For a waiting request, [`BlockManager::lookup_prefix`]
//!    returns the longest run of cached full blocks, capped so that at least
//!    the last prompt token is recomputed (its logits are needed).
//!    [`BlockManager::admit_cost`] says how many free blocks admitting it with
//!    a first chunk would consume; the scheduler compares that against its
//!    watermark. [`BlockManager::admit`] pins the hit and allocates the chunk,
//!    all or nothing.
//! 2. Each step. For every running sequence the scheduler calls
//!    [`BlockManager::allocate`] with the tokens it schedules (a prefill chunk
//!    or one decode token). The returned [`Allocation`] gives the positions to
//!    write; [`BlockManager::slots`] maps them to `(block, offset)` for the KV
//!    scatter. When the allocation carries a [`BlockCopy`], the device side
//!    copies that block prefix before writing.
//! 3. After the forward ran, [`BlockManager::commit`] marks the scheduled
//!    tokens computed and publishes newly full blocks to the prefix cache.
//!    Blocks become shareable only after their K/V exists.
//! 4. Retirement, cancellation and recompute preemption all call
//!    [`BlockManager::free`]. A preempted sequence re-enters at step 1 and
//!    usually hits its own cached blocks.
//! 5. Forks (SMC resampling, parallel samples) call [`BlockManager::fork`]:
//!    the child shares every block; whichever sequence next appends into a
//!    shared partial tail block receives a fresh block and a [`BlockCopy`].
//!
//! Allocation failure is a value ([`BlockError::OutOfBlocks`]) and leaves the
//! manager unchanged, so the scheduler can preempt and retry.

mod free_queue;
mod hash;
mod manager;
#[cfg(test)]
mod tests;

use std::num::NonZeroU32;

use thiserror::Error;

pub use hash::{BlockHash, HashKeys, block_hashes, hash_block};
pub use manager::{Allocation, BlockCopy, BlockManager, KvCounters, PrefixHit, Slot};

/// Blocks per device slab.
pub const SLAB_BLOCKS: u32 = 32;

/// A validated number of token positions per block: a power of two.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockTokens(NonZeroU32);

impl BlockTokens {
    /// The default block width.
    pub const DEFAULT: Self = Self(NonZeroU32::new(16).expect("16 is non-zero"));

    /// Creates a block width.
    ///
    /// # Errors
    ///
    /// Returns [`BlockConfigError::BlockTokens`] unless `tokens` is a power of
    /// two.
    pub const fn new(tokens: u32) -> Result<Self, BlockConfigError> {
        match NonZeroU32::new(tokens) {
            Some(tokens) if tokens.is_power_of_two() => Ok(Self(tokens)),
            _ => Err(BlockConfigError::BlockTokens(tokens)),
        }
    }

    /// Returns the number of positions in one block.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

impl Default for BlockTokens {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// The fixed shape of one KV pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PoolConfig {
    block_tokens: BlockTokens,
    slabs: NonZeroU32,
    prefix_caching: bool,
}

impl PoolConfig {
    /// Creates a pool of `slabs` slabs with prefix caching enabled.
    ///
    /// # Errors
    ///
    /// Returns an error when `slabs` is zero or the block count would not fit
    /// a `u32` block ID.
    pub fn new(block_tokens: BlockTokens, slabs: u32) -> Result<Self, BlockConfigError> {
        let slabs = NonZeroU32::new(slabs).ok_or(BlockConfigError::ZeroSlabs)?;
        // u32::MAX itself is reserved as the free queue's null link.
        if slabs.get() >= u32::MAX / SLAB_BLOCKS {
            return Err(BlockConfigError::TooManySlabs(slabs.get()));
        }
        Ok(Self {
            block_tokens,
            slabs,
            prefix_caching: true,
        })
    }

    /// Sizes a pool from a byte budget, rounding down to whole slabs.
    ///
    /// `bytes_per_token` is the K and V footprint of one position across all
    /// layers (layers × 2 × KV heads × head dim × dtype size).
    ///
    /// # Errors
    ///
    /// Returns [`BlockConfigError::ZeroBytesPerToken`], or
    /// [`BlockConfigError::BudgetBelowOneSlab`] when the budget cannot hold one
    /// slab.
    pub fn from_budget(
        budget_bytes: u64,
        bytes_per_token: u64,
        block_tokens: BlockTokens,
    ) -> Result<Self, BlockConfigError> {
        if bytes_per_token == 0 {
            return Err(BlockConfigError::ZeroBytesPerToken);
        }
        let slab_bytes = bytes_per_token
            .saturating_mul(u64::from(block_tokens.get()))
            .saturating_mul(u64::from(SLAB_BLOCKS));
        let slabs = budget_bytes / slab_bytes;
        if slabs == 0 {
            return Err(BlockConfigError::BudgetBelowOneSlab {
                budget_bytes,
                slab_bytes,
            });
        }
        let slabs = u32::try_from(slabs).map_err(|_| BlockConfigError::TooManySlabs(u32::MAX))?;
        Self::new(block_tokens, slabs)
    }

    /// Enables or disables prefix caching. Disabled, full blocks are never
    /// published and lookups always miss.
    #[must_use]
    pub const fn with_prefix_caching(mut self, enabled: bool) -> Self {
        self.prefix_caching = enabled;
        self
    }

    /// Returns the block width.
    #[must_use]
    pub const fn block_tokens(self) -> BlockTokens {
        self.block_tokens
    }

    /// Returns the number of slabs.
    #[must_use]
    pub const fn slabs(self) -> u32 {
        self.slabs.get()
    }

    /// Returns the number of blocks in the pool.
    #[must_use]
    pub const fn num_blocks(self) -> u32 {
        self.slabs.get() * SLAB_BLOCKS
    }

    /// Returns the token capacity of the pool.
    #[must_use]
    pub fn token_capacity(self) -> u64 {
        u64::from(self.num_blocks()) * u64::from(self.block_tokens.get())
    }

    /// Returns whether prefix caching is enabled.
    #[must_use]
    pub const fn prefix_caching(self) -> bool {
        self.prefix_caching
    }
}

/// One block of the pool.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BlockId(u32);

impl BlockId {
    /// Returns the block index in `0..num_blocks`.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }

    /// Returns the slab holding this block.
    #[must_use]
    pub const fn slab(self) -> u32 {
        self.0 / SLAB_BLOCKS
    }

    /// Returns the block's index within its slab.
    #[must_use]
    pub const fn index_in_slab(self) -> u32 {
        self.0 % SLAB_BLOCKS
    }
}

/// A token's index in its sequence, counted from the first prompt token.
///
/// It is the `RoPE` position and the index the block table maps to a
/// [`Slot`]. Block IDs, slot offsets and positions are distinct types, so
/// passing one where another is expected does not compile:
///
/// ```compile_fail
/// # use engine::blocks::{BlockManager, BlockTokens, PoolConfig, SequenceId};
/// let manager = BlockManager::new(PoolConfig::new(BlockTokens::DEFAULT, 1).unwrap());
/// // A raw range of integers is not a span of token positions.
/// let _ = manager.slots(SequenceId(1), 0..4);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct TokenPosition(usize);

impl TokenPosition {
    /// The first position of a sequence.
    pub const ZERO: Self = Self(0);

    /// Creates a position.
    #[must_use]
    pub const fn new(position: usize) -> Self {
        Self(position)
    }

    /// Returns the position as an index.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// A half-open run of token positions, `start..end`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TokenSpan {
    start: usize,
    end: usize,
}

impl TokenSpan {
    /// The `len` positions starting at `start`.
    ///
    /// # Panics
    ///
    /// Panics when the end position overflows `usize`.
    #[must_use]
    pub const fn new(start: TokenPosition, len: usize) -> Self {
        Self {
            start: start.0,
            end: start.0.checked_add(len).expect("token span end overflows"),
        }
    }

    /// The first `len` positions of a sequence.
    #[must_use]
    pub const fn prefix(len: usize) -> Self {
        Self { start: 0, end: len }
    }

    /// Returns the first position.
    #[must_use]
    pub const fn start(self) -> TokenPosition {
        TokenPosition(self.start)
    }

    /// Returns the position after the last.
    #[must_use]
    pub const fn end(self) -> TokenPosition {
        TokenPosition(self.end)
    }

    /// Returns the number of positions.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end - self.start
    }

    /// Returns whether the span is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    /// Iterates the positions in order.
    pub fn iter(self) -> impl ExactSizeIterator<Item = TokenPosition> {
        (self.start..self.end).map(TokenPosition)
    }
}

/// A scheduler-assigned sequence identifier.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SequenceId(pub u64);

/// Invalid pool configuration.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum BlockConfigError {
    /// Block width must be a non-zero power of two.
    #[error("KV block width must be a power of two, got {0}")]
    BlockTokens(u32),
    /// The pool needs at least one slab.
    #[error("KV pool must hold at least one slab")]
    ZeroSlabs,
    /// The block count does not fit a block ID.
    #[error("KV pool of {0} slabs exceeds the block ID range")]
    TooManySlabs(u32),
    /// A position must occupy memory.
    #[error("KV bytes per token must be greater than zero")]
    ZeroBytesPerToken,
    /// The budget is smaller than one slab.
    #[error("KV budget of {budget_bytes} bytes is below one slab ({slab_bytes} bytes)")]
    BudgetBelowOneSlab {
        /// The requested budget.
        budget_bytes: u64,
        /// The size of one slab.
        slab_bytes: u64,
    },
}

/// A block-manager operation that could not be applied. The manager is
/// unchanged after every error.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum BlockError {
    /// Not enough free blocks; the scheduler may preempt and retry.
    #[error("KV pool needs {needed} free blocks, has {free}")]
    OutOfBlocks {
        /// Blocks the operation would consume.
        needed: usize,
        /// Free blocks, cached ones included.
        free: usize,
    },
    /// No sequence with this ID.
    #[error("unknown sequence {0:?}")]
    UnknownSequence(SequenceId),
    /// A sequence with this ID already exists.
    #[error("sequence {0:?} already exists")]
    SequenceExists(SequenceId),
    /// A block of the prefix hit was evicted or re-keyed after the lookup.
    #[error("prefix hit is stale; look it up again")]
    StalePrefixHit,
    /// The sequence has tokens reserved by
    /// [`BlockManager::allocate_unresolved`] whose values are still unknown,
    /// or fewer than were supplied.
    #[error("sequence {0:?} has unresolved tokens out of order")]
    Unresolved(SequenceId),
    /// Fork needs every scheduled token committed first.
    #[error("sequence {0:?} has uncommitted tokens")]
    Uncommitted(SequenceId),
    /// A position past the sequence's scheduled tokens.
    #[error("position {} is past the {tokens} scheduled tokens", position.get())]
    PositionOutOfRange {
        /// The requested position.
        position: TokenPosition,
        /// Tokens with slots.
        tokens: usize,
    },
}
