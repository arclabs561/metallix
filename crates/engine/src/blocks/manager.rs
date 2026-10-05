//! The block manager: pool accounting, block tables and the prefix cache.

use std::collections::HashMap;

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

/// The cached prefix found for a prompt. Pass it to [`BlockManager::admit`]
/// before any other mutation of the manager, or the admit may fail with
/// [`BlockError::StalePrefixHit`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrefixHit {
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
    /// Tokens after the last hashed block, up to `scheduled`.
    unhashed: Vec<u32>,
}

/// Block tables, reference counts, the free queue, and the prefix cache for
/// one KV pool. See the [module docs](super) for the call sequence.
#[derive(Clone, Debug)]
pub struct BlockManager {
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
}

impl BlockManager {
    /// Creates a manager with every block free.
    #[must_use]
    pub fn new(config: PoolConfig) -> Self {
        let blocks = config.num_blocks();
        Self {
            config,
            pool: Pool {
                block_tokens: config.block_tokens().get() as usize,
                prefix_caching: config.prefix_caching(),
                ref_counts: vec![0; blocks as usize],
                block_hashes: vec![None; blocks as usize],
                free: FreeQueue::full(blocks),
                cached: HashMap::new(),
                counters: KvCounters::default(),
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
    #[must_use]
    pub fn admit_cost(&self, hit: &PrefixHit, chunk_tokens: usize) -> usize {
        let pinned_free = hit
            .blocks
            .iter()
            .filter(|block| self.pool.ref_counts[block.index() as usize] == 0)
            .count();
        let blocks = (hit.cached_tokens + chunk_tokens).div_ceil(self.pool.block_tokens);
        pinned_free + blocks - hit.blocks.len()
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
        let pool = &mut self.pool;
        let stale = hit.blocks.iter().zip(&hit.hashes).any(|(block, hash)| {
            pool.block_hashes[block.index() as usize] != Some(*hash)
                || pool.cached.get(hash) != Some(block)
        });
        if stale {
            return Err(BlockError::StalePrefixHit);
        }
        let needed = self.admit_cost(&hit, chunk.len());
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
        let needed = self.pool.cost(sequence, tokens.len());
        if needed > self.pool.free.len() {
            return Err(BlockError::OutOfBlocks {
                needed,
                free: self.pool.free.len(),
            });
        }
        Ok(self.pool.append(sequence, tokens))
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
        let sequence = self
            .sequences
            .get_mut(&seq)
            .ok_or(BlockError::UnknownSequence(seq))?;
        sequence.computed = sequence.scheduled;
        let pool = &mut self.pool;
        let block_tokens = pool.block_tokens;
        let mut hashed = 0;
        for tokens in sequence.unhashed.chunks_exact(block_tokens) {
            let hash = hash_block(sequence.hashes.last(), tokens, &sequence.keys);
            let block = sequence.table[sequence.hashes.len()];
            sequence.hashes.push(hash);
            hashed += block_tokens;
            let key = &mut pool.block_hashes[block.index() as usize];
            if key.is_none() && !pool.cached.contains_key(&hash) {
                *key = Some(hash);
                pool.cached.insert(hash, block);
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
        if source.scheduled != source.computed {
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
        let start = sequence.scheduled;
        let mut copy = None;
        if !tokens.is_empty() && self.needs_copy(sequence) {
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
        let blocks = (start + tokens.len()).div_ceil(self.block_tokens);
        while sequence.table.len() < blocks {
            let block = self.take_free();
            sequence.table.push(block);
        }
        sequence.scheduled += tokens.len();
        if self.prefix_caching {
            sequence.unhashed.extend_from_slice(tokens);
        }
        Allocation {
            positions: TokenSpan::new(TokenPosition::new(start), tokens.len()),
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
