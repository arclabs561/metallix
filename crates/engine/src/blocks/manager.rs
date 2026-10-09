//! The block manager: pool accounting, block tables and the prefix cache.

use std::{collections::HashMap, sync::Arc};

use super::free_queue::FreeQueue;
use super::hash::{BlockHash, HashKeys, hash_block};
use super::{BlockError, BlockId, PoolConfig, SequenceId, TokenPosition, TokenSpan};

/// Positions reserved by [`BlockManager::admit`] or [`BlockManager::allocate`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Allocation {
    /// Positions to write this step.
    pub positions: TokenSpan,
    /// A copy the device side must perform before writing.
    pub copy: Option<BlockCopy>,
}

/// Copy-on-write of a shared partial tail block: copy positions
/// `0..tokens` of `src` into `dst` before writing new tokens into `dst`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockCopy {
    /// The shared block.
    pub src: BlockId,
    /// The sequence's new private block.
    pub dst: BlockId,
    /// Filled positions to copy.
    pub tokens: usize,
}

/// Where one token position lives in the pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Slot {
    /// The block.
    pub block: BlockId,
    /// The position within the block.
    pub offset: u32,
}

impl Slot {
    /// Returns the flat pool index `block * block_tokens + offset`.
    #[must_use]
    pub fn flat(self, block_tokens: u32) -> u64 {
        u64::from(self.block.index()) * u64::from(block_tokens) + u64::from(self.offset)
    }
}

/// Unforgeable identity retained by hits even after their pool is dropped.
#[derive(Clone, Debug)]
struct PoolId(Arc<()>);

impl PartialEq for PoolId {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for PoolId {}

/// The cached prefix found for a prompt. Pass it to [`BlockManager::admit`]
/// before any other mutation of the manager, or the admit may fail with
/// [`BlockError::StalePrefixHit`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixHit {
    owner: PoolId,
    keys: HashKeys,
    blocks: Vec<BlockId>,
    hashes: Vec<BlockHash>,
    cached_tokens: usize,
    prompt_tokens: usize,
}

impl PrefixHit {
    /// Returns the cached blocks, in prompt order.
    #[must_use]
    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    /// Returns the number of prompt tokens the hit covers.
    #[must_use]
    pub const fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    /// Returns the prompt length the lookup was made for.
    #[must_use]
    pub const fn prompt_tokens(&self) -> usize {
        self.prompt_tokens
    }

    /// Returns the keys the lookup used.
    #[must_use]
    pub const fn keys(&self) -> &HashKeys {
        &self.keys
    }
}

/// Monotonic counters for metrics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KvCounters {
    /// Prompt tokens of admitted requests (`vllm:prefix_cache_queries`).
    pub prefix_queries: u64,
    /// Of those, tokens served from cache (`vllm:prefix_cache_hits`).
    pub prefix_hits: u64,
    /// Cached blocks evicted by reallocation.
    pub evictions: u64,
}

#[derive(Clone, Debug)]
struct Sequence {
    keys: HashKeys,
    table: Vec<BlockId>,
    /// Tokens with slots, committed or not.
    scheduled: usize,
    /// Tokens whose K/V the forward has written.
    computed: usize,
    /// Keys of the leading full computed blocks.
    hashes: Vec<BlockHash>,
    /// Tokens after the last hashed block, up to `scheduled`, except the
    /// last `unresolved`.
    unhashed: Vec<u32>,
    /// Trailing scheduled positions whose token values are not known yet: a
    /// pipelined step reserves a slot for an input the device has not read
    /// back. See [`BlockManager::allocate_unresolved`].
    unresolved: usize,
}

/// Block tables, reference counts, the free queue, and the prefix cache for
/// one KV pool. See the [module docs](super) for the call sequence.
#[derive(Debug)]
pub struct BlockManager {
    owner: PoolId,
    config: PoolConfig,
    pool: Pool,
    sequences: HashMap<SequenceId, Sequence>,
}

/// Per-block state, kept apart from the sequence map so an append can borrow
/// one sequence and the pool at once.
#[derive(Clone, Debug)]
struct Pool {
    block_tokens: usize,
    prefix_caching: bool,
    ref_counts: Vec<u32>,
    block_hashes: Vec<Option<BlockHash>>,
    free: FreeQueue,
    cached: HashMap<BlockHash, BlockId>,
    counters: KvCounters,
    published_tokens: usize,
}

// A manager clone owns independent mutable pool state, so old hits cannot
// cross into it even while its blocks and hashes happen to be identical.
impl Clone for BlockManager {
    fn clone(&self) -> Self {
        Self {
            owner: PoolId(Arc::new(())),
            config: self.config,
            pool: self.pool.clone(),
            sequences: self.sequences.clone(),
        }
    }
}

impl BlockManager {
    /// Creates a manager with every block free.
    #[must_use]
    pub fn new(config: PoolConfig) -> Self {
        let blocks = config.num_blocks();
        Self {
            owner: PoolId(Arc::new(())),
            config,
            pool: Pool {
                block_tokens: config.block_tokens().get() as usize,
                prefix_caching: config.prefix_caching(),
                ref_counts: vec![0; blocks as usize],
                block_hashes: vec![None; blocks as usize],
                free: FreeQueue::full(blocks),
                cached: HashMap::new(),
                counters: KvCounters::default(),
                published_tokens: 0,
            },
            sequences: HashMap::new(),
        }
    }

    /// Returns the pool configuration.
    #[must_use]
    pub const fn config(&self) -> PoolConfig {
        self.config
    }

    /// Returns the number of blocks in the pool.
    #[must_use]
    pub fn total_blocks(&self) -> usize {
        self.pool.ref_counts.len()
    }

    /// Tokens per block.
    #[must_use]
    pub const fn block_tokens(&self) -> usize {
        self.pool.block_tokens
    }

    /// Where a prefill chunk of at most `budget` tokens from `start` in a
    /// prompt of `len` tokens ends: the prompt's end when it fits, otherwise
    /// the last block boundary within the budget, so each chunk's full blocks
    /// can be published as it completes. Returns `start` when the budget does
    /// not reach the next boundary.
    #[must_use]
    pub const fn chunk_end(&self, start: usize, len: usize, budget: usize) -> usize {
        let end = start.saturating_add(budget);
        if end >= len {
            return len;
        }
        let aligned = end / self.pool.block_tokens * self.pool.block_tokens;
        if aligned > start { aligned } else { start }
    }

    /// Returns the number of free blocks, cached ones included.
    #[must_use]
    pub const fn free_blocks(&self) -> usize {
        self.pool.free.len()
    }

    /// Returns the number of blocks holding a published prefix key.
    #[must_use]
    pub fn cached_blocks(&self) -> usize {
        self.pool.cached.len()
    }

    /// Returns the fraction of blocks referenced by a sequence, in `0..=1`
    /// (`vllm:kv_cache_usage_perc` is this fraction).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn usage(&self) -> f64 {
        1.0 - self.free_blocks() as f64 / self.total_blocks() as f64
    }

    /// Returns the metric counters.
    #[must_use]
    pub const fn counters(&self) -> KvCounters {
        self.pool.counters
    }

    /// Tokens in full blocks newly published by successful commits. Repeated
    /// commits and duplicate block hashes do not increment this counter.
    #[must_use]
    pub const fn published_tokens(&self) -> usize {
        self.pool.published_tokens
    }

    /// Returns the number of live sequences.
    #[must_use]
    pub fn sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Returns a block's reference count, or `None` for a block outside the
    /// pool.
    #[must_use]
    pub fn ref_count(&self, block: BlockId) -> Option<u32> {
        self.pool.ref_counts.get(block.index() as usize).copied()
    }

    /// Returns the prefix key a block is published under.
    #[must_use]
    pub fn cached_hash(&self, block: BlockId) -> Option<BlockHash> {
        self.pool
            .block_hashes
            .get(block.index() as usize)
            .copied()
            .flatten()
    }

    /// Returns a sequence's block table.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::UnknownSequence`].
    pub fn block_table(&self, seq: SequenceId) -> Result<&[BlockId], BlockError> {
        Ok(&self.sequence(seq)?.table)
    }

    /// Returns the number of tokens with slots (scheduled, committed or not).
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::UnknownSequence`].
    pub fn num_tokens(&self, seq: SequenceId) -> Result<usize, BlockError> {
        Ok(self.sequence(seq)?.scheduled)
    }

    /// Returns the number of scheduled tokens whose values are not known yet
    /// (see [`Self::allocate_unresolved`]).
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::UnknownSequence`].
    pub fn num_unresolved(&self, seq: SequenceId) -> Result<usize, BlockError> {
        Ok(self.sequence(seq)?.unresolved)
    }

    /// Returns the number of committed tokens.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::UnknownSequence`].
    pub fn num_computed(&self, seq: SequenceId) -> Result<usize, BlockError> {
        Ok(self.sequence(seq)?.computed)
    }

    /// Maps positions of a sequence to pool slots.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::UnknownSequence`], or
    /// [`BlockError::PositionOutOfRange`] when `positions` reaches past the
    /// scheduled tokens.
    pub fn slots(
        &self,
        seq: SequenceId,
        positions: TokenSpan,
    ) -> Result<impl ExactSizeIterator<Item = Slot> + '_, BlockError> {
        let sequence = self.sequence(seq)?;
        if positions.end().get() > sequence.scheduled {
            return Err(BlockError::PositionOutOfRange {
                position: TokenPosition::new(positions.end().get() - 1),
                tokens: sequence.scheduled,
            });
        }
        let block_tokens = self.pool.block_tokens;
        Ok(positions
            .iter()
            .map(TokenPosition::get)
            .map(move |position| Slot {
                block: sequence.table[position / block_tokens],
                // The offset is below block_tokens, which is a u32.
                #[allow(clippy::cast_possible_truncation)]
                offset: (position % block_tokens) as u32,
            }))
    }

    /// Finds the longest run of cached full blocks at the start of `prompt`.
    ///
    /// The hit never covers the last prompt token: a prompt that is entirely
    /// cached still recomputes its final token to produce logits, which can
    /// mean recomputing one block. Lookups have no side effects.
    #[must_use]
    pub fn lookup_prefix(&self, prompt: &[u32], keys: HashKeys) -> PrefixHit {
        let block_tokens = self.pool.block_tokens;
        let mut blocks = Vec::new();
        let mut hashes: Vec<BlockHash> = Vec::new();
        if self.pool.prefix_caching {
            let max_blocks = prompt.len().saturating_sub(1) / block_tokens;
            for tokens in prompt.chunks_exact(block_tokens).take(max_blocks) {
                let hash = hash_block(hashes.last(), tokens, &keys);
                let Some(&block) = self.pool.cached.get(&hash) else {
                    break;
                };
                blocks.push(block);
                hashes.push(hash);
            }
        }
        PrefixHit {
            owner: self.owner.clone(),
            keys,
            cached_tokens: blocks.len() * block_tokens,
            blocks,
            hashes,
            prompt_tokens: prompt.len(),
        }
    }

    /// Returns how many free blocks [`BlockManager::admit`] would consume for
    /// `hit` plus a first chunk of `chunk_tokens`: new blocks for the chunk
    /// plus hit blocks that are currently free.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::StalePrefixHit`] when the hit belongs to another
    /// manager (including a clone), or its blocks were evicted or re-keyed.
    pub fn admit_cost(&self, hit: &PrefixHit, chunk_tokens: usize) -> Result<usize, BlockError> {
        self.validate_hit(hit)?;
        let pinned_free = hit
            .blocks
            .iter()
            .filter(|block| self.pool.ref_counts[block.index() as usize] == 0)
            .count();
        let blocks = (hit.cached_tokens + chunk_tokens).div_ceil(self.pool.block_tokens);
        Ok(pinned_free + blocks - hit.blocks.len())
    }

    fn validate_hit(&self, hit: &PrefixHit) -> Result<(), BlockError> {
        // Check identity before using any block ID as an index.
        if self.owner != hit.owner
            || hit.blocks.iter().zip(&hit.hashes).any(|(block, hash)| {
                self.pool.block_hashes[block.index() as usize] != Some(*hash)
                    || self.pool.cached.get(hash) != Some(block)
            })
        {
            return Err(BlockError::StalePrefixHit);
        }
        Ok(())
    }

    /// Admits a sequence: pins the cached prefix in `hit` and allocates slots
    /// for `chunk`, the prompt tokens that follow it (possibly a partial
    /// chunk; later chunks go through [`BlockManager::allocate`]).
    ///
    /// Records the request's prompt length and cached tokens in
    /// [`KvCounters`].
    ///
    /// # Errors
    ///
    /// [`BlockError::SequenceExists`], [`BlockError::StalePrefixHit`], or
    /// [`BlockError::OutOfBlocks`]; the manager is unchanged on error.
    pub fn admit(
        &mut self,
        seq: SequenceId,
        hit: PrefixHit,
        chunk: &[u32],
    ) -> Result<Allocation, BlockError> {
        if self.sequences.contains_key(&seq) {
            return Err(BlockError::SequenceExists(seq));
        }
        let needed = self.admit_cost(&hit, chunk.len())?;
        let pool = &mut self.pool;
        if needed > pool.free.len() {
            return Err(BlockError::OutOfBlocks {
                needed,
                free: pool.free.len(),
            });
        }
        for block in &hit.blocks {
            pool.add_ref(*block);
        }
        let counters = &mut pool.counters;
        counters.prefix_queries = counters
            .prefix_queries
            .saturating_add(hit.prompt_tokens as u64);
        counters.prefix_hits = counters
            .prefix_hits
            .saturating_add(hit.cached_tokens as u64);
        let mut sequence = Sequence {
            keys: hit.keys,
            table: hit.blocks,
            scheduled: hit.cached_tokens,
            computed: hit.cached_tokens,
            hashes: hit.hashes,
            unhashed: Vec::new(),
            unresolved: 0,
        };
        let allocation = pool.append(&mut sequence, chunk);
        self.sequences.insert(seq, sequence);
        Ok(allocation)
    }

    /// Returns how many free blocks appending `tokens` new tokens to `seq`
    /// would consume, including a copy-on-write tail.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::UnknownSequence`].
    pub fn append_cost(&self, seq: SequenceId, tokens: usize) -> Result<usize, BlockError> {
        Ok(self.pool.cost(self.sequence(seq)?, tokens))
    }

    /// Reserves slots for `tokens`, the next tokens of `seq` scheduled this
    /// step. A shared partial tail block is replaced by a private copy first
    /// (see [`Allocation::copy`]).
    ///
    /// # Errors
    ///
    /// [`BlockError::UnknownSequence`] or [`BlockError::OutOfBlocks`]; the
    /// manager is unchanged on error.
    pub fn allocate(&mut self, seq: SequenceId, tokens: &[u32]) -> Result<Allocation, BlockError> {
        let sequence = self
            .sequences
            .get_mut(&seq)
            .ok_or(BlockError::UnknownSequence(seq))?;
        if sequence.unresolved > 0 {
            return Err(BlockError::Unresolved(seq));
        }
        let needed = self.pool.cost(sequence, tokens.len());
        if needed > self.pool.free.len() {
            return Err(BlockError::OutOfBlocks {
                needed,
                free: self.pool.free.len(),
            });
        }
        Ok(self.pool.append(sequence, tokens))
    }

    /// Reserves slots for `count` next tokens of `seq` whose values are not
    /// known yet, such as the input of a decode step queued before the
    /// previous step's pick is read back. [`Self::resolve`] supplies the
    /// values, in order, before they can be hashed.
    ///
    /// # Errors
    ///
    /// [`BlockError::UnknownSequence`] or [`BlockError::OutOfBlocks`]; the
    /// manager is unchanged on error.
    pub fn allocate_unresolved(
        &mut self,
        seq: SequenceId,
        count: usize,
    ) -> Result<Allocation, BlockError> {
        let sequence = self
            .sequences
            .get_mut(&seq)
            .ok_or(BlockError::UnknownSequence(seq))?;
        let needed = self.pool.cost(sequence, count);
        if needed > self.pool.free.len() {
            return Err(BlockError::OutOfBlocks {
                needed,
                free: self.pool.free.len(),
            });
        }
        let allocation = self.pool.reserve(sequence, count);
        sequence.unresolved += count;
        Ok(allocation)
    }

    /// Supplies the values of the oldest unresolved tokens of `seq`.
    ///
    /// # Errors
    ///
    /// [`BlockError::UnknownSequence`], or [`BlockError::Unresolved`] when
    /// `tokens` is longer than the unresolved run.
    pub fn resolve(&mut self, seq: SequenceId, tokens: &[u32]) -> Result<(), BlockError> {
        let sequence = self
            .sequences
            .get_mut(&seq)
            .ok_or(BlockError::UnknownSequence(seq))?;
        if tokens.len() > sequence.unresolved {
            return Err(BlockError::Unresolved(seq));
        }
        sequence.unresolved -= tokens.len();
        if self.pool.prefix_caching {
            sequence.unhashed.extend_from_slice(tokens);
        }
        Ok(())
    }

    /// Marks every scheduled token of `seq` computed and publishes newly full
    /// blocks to the prefix cache. Call after the forward that wrote them.
    ///
    /// A block whose key is already published under another block (two
    /// sequences computed the same prefix concurrently) stays private.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::UnknownSequence`].
    pub fn commit(&mut self, seq: SequenceId) -> Result<(), BlockError> {
        let scheduled = self.num_tokens(seq)?;
        self.commit_through(seq, scheduled)
    }

    /// Marks the first `computed` tokens of `seq` computed, for a pipelined
    /// step that finished while a later one is still queued, and publishes
    /// newly full blocks whose tokens are all computed and resolved.
    ///
    /// # Errors
    ///
    /// [`BlockError::UnknownSequence`], or
    /// [`BlockError::PositionOutOfRange`] when `computed` is past the
    /// scheduled tokens.
    pub fn commit_through(&mut self, seq: SequenceId, computed: usize) -> Result<(), BlockError> {
        let sequence = self
            .sequences
            .get_mut(&seq)
            .ok_or(BlockError::UnknownSequence(seq))?;
        if computed > sequence.scheduled {
            return Err(BlockError::PositionOutOfRange {
                position: TokenPosition::new(computed.saturating_sub(1)),
                tokens: sequence.scheduled,
            });
        }
        sequence.computed = sequence.computed.max(computed);
        let pool = &mut self.pool;
        let block_tokens = pool.block_tokens;
        // Only tokens whose K/V exists may be published.
        let hashable = sequence
            .computed
            .saturating_sub(sequence.hashes.len() * block_tokens)
            .min(sequence.unhashed.len());
        let mut hashed = 0;
        for tokens in sequence.unhashed[..hashable].chunks_exact(block_tokens) {
            let hash = hash_block(sequence.hashes.last(), tokens, &sequence.keys);
            let block = sequence.table[sequence.hashes.len()];
            sequence.hashes.push(hash);
            hashed += block_tokens;
            let key = &mut pool.block_hashes[block.index() as usize];
            if key.is_none() && !pool.cached.contains_key(&hash) {
                *key = Some(hash);
                pool.cached.insert(hash, block);
                pool.published_tokens = pool.published_tokens.saturating_add(block_tokens);
            }
        }
        sequence.unhashed.drain(..hashed);
        Ok(())
    }

    /// Gives `child` the same blocks and tokens as `parent`. Neither sequence
    /// writes a shared block afterwards: the first append into the shared
    /// partial tail is copied on write.
    ///
    /// # Errors
    ///
    /// [`BlockError::UnknownSequence`] for `parent`,
    /// [`BlockError::SequenceExists`] for `child`, or
    /// [`BlockError::Uncommitted`] when `parent` has scheduled tokens not yet
    /// committed.
    pub fn fork(&mut self, parent: SequenceId, child: SequenceId) -> Result<(), BlockError> {
        let source = self.sequence(parent)?;
        if self.sequences.contains_key(&child) {
            return Err(BlockError::SequenceExists(child));
        }
        if source.scheduled != source.computed || source.unresolved > 0 {
            return Err(BlockError::Uncommitted(parent));
        }
        let copy = source.clone();
        for block in &copy.table {
            self.pool.add_ref(*block);
        }
        self.sequences.insert(child, copy);
        Ok(())
    }

    /// Releases a sequence's blocks: on retirement, cancellation, or
    /// recompute preemption. Blocks with a published key stay cached at the
    /// back of the free queue, the sequence's last block first, so a chain is
    /// evicted from its tail. Unkeyed blocks go to the front, reused before
    /// any cached block.
    ///
    /// # Errors
    ///
    /// Returns [`BlockError::UnknownSequence`].
    pub fn free(&mut self, seq: SequenceId) -> Result<(), BlockError> {
        let sequence = self
            .sequences
            .remove(&seq)
            .ok_or(BlockError::UnknownSequence(seq))?;
        for block in sequence.table.iter().rev() {
            self.pool.release(*block);
        }
        Ok(())
    }

    fn sequence(&self, seq: SequenceId) -> Result<&Sequence, BlockError> {
        self.sequences
            .get(&seq)
            .ok_or(BlockError::UnknownSequence(seq))
    }

    /// Checks every structural invariant; used by tests.
    #[cfg(test)]
    pub(super) fn check_invariants(&self) -> Result<(), String> {
        let pool = &self.pool;
        let mut expected = vec![0_u32; self.total_blocks()];
        for (id, sequence) in &self.sequences {
            for block in &sequence.table {
                expected[block.index() as usize] += 1;
            }
            let blocks = sequence.scheduled.div_ceil(pool.block_tokens);
            if sequence.table.len() != blocks {
                return Err(format!(
                    "{id:?}: table has {} blocks for {blocks}",
                    sequence.table.len()
                ));
            }
            if sequence.computed > sequence.scheduled {
                return Err(format!("{id:?}: computed past scheduled"));
            }
            if pool.prefix_caching {
                let tail = sequence.scheduled - sequence.hashes.len() * pool.block_tokens;
                if sequence.unhashed.len() + sequence.unresolved != tail {
                    return Err(format!(
                        "{id:?}: {} unhashed and {} unresolved tokens for a {tail}-token tail",
                        sequence.unhashed.len(),
                        sequence.unresolved
                    ));
                }
            }
        }
        for (index, (&count, &want)) in pool.ref_counts.iter().zip(&expected).enumerate() {
            if count != want {
                return Err(format!(
                    "block {index}: ref count {count}, tables hold {want}"
                ));
            }
            #[allow(clippy::cast_possible_truncation)]
            let queued = pool.free.contains(index as u32);
            if queued != (count == 0) {
                return Err(format!("block {index}: ref count {count}, queued {queued}"));
            }
        }
        if pool.free.iter().count() != pool.free.len() {
            return Err("free queue length disagrees with its links".into());
        }
        for (hash, block) in &pool.cached {
            if pool.block_hashes[block.index() as usize] != Some(*hash) {
                return Err(format!("{block:?} is mapped but not keyed"));
            }
        }
        let keyed = pool.block_hashes.iter().flatten().count();
        if keyed != pool.cached.len() {
            return Err(format!(
                "{keyed} keyed blocks, {} mapped",
                pool.cached.len()
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn free_order(&self) -> Vec<BlockId> {
        self.pool.free.iter().map(BlockId).collect()
    }
}

impl Pool {
    fn needs_copy(&self, sequence: &Sequence) -> bool {
        !sequence.scheduled.is_multiple_of(self.block_tokens)
            && sequence
                .table
                .last()
                .is_some_and(|block| self.ref_counts[block.index() as usize] > 1)
    }

    fn cost(&self, sequence: &Sequence, tokens: usize) -> usize {
        if tokens == 0 {
            return 0;
        }
        let blocks = (sequence.scheduled + tokens).div_ceil(self.block_tokens);
        blocks - sequence.table.len() + usize::from(self.needs_copy(sequence))
    }

    /// Appends after the caller checked `cost` against the free queue.
    fn append(&mut self, sequence: &mut Sequence, tokens: &[u32]) -> Allocation {
        let allocation = self.reserve(sequence, tokens.len());
        if self.prefix_caching {
            sequence.unhashed.extend_from_slice(tokens);
        }
        allocation
    }

    /// Reserves slots for `count` tokens after the caller checked `cost`.
    fn reserve(&mut self, sequence: &mut Sequence, count: usize) -> Allocation {
        let start = sequence.scheduled;
        let mut copy = None;
        if count > 0 && self.needs_copy(sequence) {
            let last = sequence.table.len() - 1;
            let src = sequence.table[last];
            let dst = self.take_free();
            sequence.table[last] = dst;
            self.release(src);
            copy = Some(BlockCopy {
                src,
                dst,
                tokens: start % self.block_tokens,
            });
        }
        let blocks = (start + count).div_ceil(self.block_tokens);
        while sequence.table.len() < blocks {
            let block = self.take_free();
            sequence.table.push(block);
        }
        sequence.scheduled += count;
        Allocation {
            positions: TokenSpan::new(TokenPosition::new(start), count),
            copy,
        }
    }

    /// Takes the front free block, evicting its cached key.
    fn take_free(&mut self) -> BlockId {
        let block = BlockId(
            self.free
                .pop_front()
                .expect("callers check the free count before allocating"),
        );
        let index = block.index() as usize;
        if let Some(hash) = self.block_hashes[index].take() {
            self.cached.remove(&hash);
            self.counters.evictions = self.counters.evictions.saturating_add(1);
        }
        debug_assert_eq!(self.ref_counts[index], 0);
        self.ref_counts[index] = 1;
        block
    }

    fn add_ref(&mut self, block: BlockId) {
        let index = block.index() as usize;
        if self.ref_counts[index] == 0 {
            self.free.remove(block.index());
        }
        self.ref_counts[index] += 1;
    }

    fn release(&mut self, block: BlockId) {
        let index = block.index() as usize;
        self.ref_counts[index] -= 1;
        if self.ref_counts[index] == 0 {
            if self.block_hashes[index].is_some() {
                self.free.push_back(block.index());
            } else {
                self.free.push_front(block.index());
            }
        }
    }
}

#[cfg(test)]
mod provenance_tests {
    use super::*;
    use crate::blocks::BlockTokens;

    #[test]
    fn cloned_manager_rejects_original_hit_and_accepts_own_lookup() {
        let config = PoolConfig::new(BlockTokens::new(4).unwrap(), 1).unwrap();
        let mut original = BlockManager::new(config);
        let prompt = [1, 2, 3, 4, 5];
        let hit = original.lookup_prefix(&prompt, HashKeys::new());
        original.admit(SequenceId(1), hit, &prompt).unwrap();
        original.commit(SequenceId(1)).unwrap();
        original.free(SequenceId(1)).unwrap();
        let hit = original.lookup_prefix(&prompt, HashKeys::new());
        assert_eq!(hit.cached_tokens(), 4);

        let mut cloned = original.clone();
        let before = format!("{cloned:?}");
        assert_eq!(cloned.admit_cost(&hit, 1), Err(BlockError::StalePrefixHit));
        assert_eq!(
            cloned.admit(SequenceId(2), hit.clone(), &prompt[4..]),
            Err(BlockError::StalePrefixHit)
        );
        assert_eq!(format!("{cloned:?}"), before);
        assert_eq!(original.admit_cost(&hit, 1), Ok(2));
        let own = cloned.lookup_prefix(&prompt, HashKeys::new());
        assert_eq!(own.cached_tokens(), 4);
        assert_eq!(cloned.admit_cost(&own, 1), Ok(2));
        cloned.admit(SequenceId(2), own, &prompt[4..]).unwrap();
        cloned.commit(SequenceId(2)).unwrap();
        assert_eq!(cloned.num_computed(SequenceId(2)), Ok(prompt.len()));
        assert_eq!(original.sequences(), 0);
    }

    #[test]
    fn admit_cost_rejects_foreign_empty_hit_without_mutation() {
        let config = PoolConfig::new(BlockTokens::new(4).unwrap(), 1).unwrap();
        let first = BlockManager::new(config);
        let second = BlockManager::new(config);
        let hit = first.lookup_prefix(&[1], HashKeys::new());
        let before = format!("{second:?}");
        assert_eq!(second.admit_cost(&hit, 1), Err(BlockError::StalePrefixHit));
        assert_eq!(format!("{second:?}"), before);
    }
}
