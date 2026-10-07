//! Paged K/V storage for Qwen3: one preallocated device pool shared by many
//! sequences, addressed through [`engine::blocks`] block tables.
//!
//! Each layer stores K and V slot-major, `[blocks * block_tokens, kv_heads,
//! head_dim]`, where a token's slot is `block * block_tokens + offset`. New
//! K/V rows go in with one scatter per array. MLX reuses the array's buffer
//! for that scatter only while the pool holds its sole reference, so nothing
//! outside this module may clone a pool handle; forks share blocks through
//! the manager's reference counts instead.
//!
//! Attention gathers every row's tokens with one `take` of a `[tokens, rows]`
//! slot matrix, giving `[tokens, rows, kv_heads, head_dim]`. Viewed as
//! `[rows, kv_heads, tokens, head_dim]`, that is a layout MLX's fused SDPA
//! reads without another copy. A single sequence runs the same SDPA and masks
//! as the contiguous executor, so it produces the same logits as
//! [`super::Qwen3ForwardExecutor`].

use std::{collections::HashMap, hash::BuildHasher};

use engine::blocks::{
    BlockCopy, BlockId, BlockManager, BlockTokens, HashKeys, PoolConfig, SequenceId, Slot,
    TokenPosition, TokenSpan,
};
use mlx_rs::{Array, Dtype, StreamOrDevice, fast, ops, ops::indexing::IndexMutOp};

use super::{
    Qwen3ForwardConfig, Qwen3ForwardError, RopePositions, as_i32, attention_output,
    attention_scale, chunk_causal_mask, embed_rows, kv_precision, mlp_residual, project,
    read_last_logits, rms_norm, rotated_qkv, validate_input_ids, weight,
};
use crate::Qwen3Attention;

/// A Qwen3 decoder whose sequences share one paged K/V pool.
///
/// It mirrors [`super::Qwen3ForwardExecutor`]'s prefill, extend and decode
/// calls, with a [`SequenceId`] naming the sequence, plus block-level
/// [`Self::fork`] and [`Self::free`]. The pool is allocated and evaluated once
/// at construction.
pub struct PagedQwen3Session<'a, S: BuildHasher> {
    config: &'a Qwen3ForwardConfig,
    weights: &'a HashMap<String, Array, S>,
    blocks: BlockManager,
    pool: KvPool,
}

/// Where a queued decode row's input token comes from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepInput {
    /// A token the host holds.
    Host(i32),
    /// The token the previous queued step picked for this row index, still
    /// on the device.
    Previous(usize),
}

/// A decode step queued by [`PagedQwen3Session::queue_decode`] and not yet
/// read back.
pub struct QueuedDecode {
    picks: super::Qwen3TokenPicks,
    rows: Vec<SequenceId>,
    /// Each row's length once this step is computed.
    lengths: Vec<usize>,
}

impl QueuedDecode {
    /// The rows, in step order.
    #[must_use]
    pub fn rows(&self) -> &[SequenceId] {
        &self.rows
    }

    /// Fault injection for the real readback error path, after GPU completion.
    #[cfg(test)]
    pub(super) fn invalidate_commit_length(&mut self) {
        self.lengths[0] = usize::MAX;
    }
}

/// The result of [`PagedQwen3Session::prefill_with_keys`].
#[derive(Clone, Debug, PartialEq)]
pub struct PagedPrefill {
    /// The last prompt token's logits.
    pub logits: Vec<f32>,
    /// Leading prompt tokens served from the prefix cache, not computed.
    pub cached_tokens: usize,
    /// Prompt tokens in full blocks newly published during this prefill.
    /// Later decode may complete a partial prompt block; that is excluded.
    pub cache_write_tokens: usize,
}

/// What [`PagedQwen3Session::decode_batch`] reads back per row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchReadback {
    /// Each row's full logits.
    Logits,
    /// Each row's greedy token, chosen on the GPU (`argmax`, lowest index on
    /// ties); only one ID per row crosses to the host.
    Greedy,
}

/// The per-row result of [`PagedQwen3Session::decode_batch`], in row order.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchDecoded {
    /// `vocab_size` logits per row.
    Logits(Vec<Vec<f32>>),
    /// One token ID per row.
    Greedy(Vec<i32>),
}

/// The device side of the pool: per layer, K and V as `[slots, kv_heads,
/// head_dim]`.
struct KvPool {
    dtype: Dtype,
    block_tokens: u32,
    layers: Vec<LayerKv>,
}

struct LayerKv {
    keys: Array,
    values: Array,
}

/// A token's index along the pool's slot axis, `block * block_tokens +
/// offset`. Kept apart from token positions and block IDs by type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PoolSlot(i32);

impl PoolSlot {
    fn of(slot: Slot, block_tokens: u32) -> Result<Self, Qwen3ForwardError> {
        i32::try_from(slot.flat(block_tokens))
            .map(Self)
            .map_err(|_| Qwen3ForwardError::ShapeOverflow)
    }
}

/// The pool slots one forward writes, in the order of its new K/V rows.
struct WritePlan {
    slots: Array,
}

impl WritePlan {
    fn new(slots: &[PoolSlot]) -> Result<Self, Qwen3ForwardError> {
        let slots = slots.iter().map(|slot| slot.0).collect::<Vec<_>>();
        let len = as_i32(slots.len())?;
        Ok(Self {
            slots: Array::from_slice(&slots, &[len]),
        })
    }
}

/// One decode slot per batch row, from [`PagedQwen3Session::decode_batch`].
struct RowSlots {
    /// Pool slot of each row's new token.
    slots: Vec<PoolSlot>,
    /// Each row's `RoPE` position.
    positions: Vec<i32>,
    /// Each row's length after the step.
    lengths: Vec<usize>,
}

/// Every row's K/V slots for one forward, as a `[tokens, rows]` matrix.
/// Rows shorter than the longest repeat their first slot; attention masks
/// those positions out.
struct GatherPlan {
    slots: Array,
}

impl GatherPlan {
    fn new(
        blocks: &BlockManager,
        seqs: &[SequenceId],
        lengths: &[usize],
    ) -> Result<Self, Qwen3ForwardError> {
        let block_tokens = blocks.config().block_tokens().get();
        let longest = lengths.iter().copied().max().unwrap_or(0);
        let rows = seqs.len();
        let mut matrix = vec![0_i32; longest * rows];
        for (row, (&seq, &length)) in seqs.iter().zip(lengths).enumerate() {
            let mut first = None;
            for (token, slot) in blocks.slots(seq, TokenSpan::prefix(length))?.enumerate() {
                let slot = PoolSlot::of(slot, block_tokens)?;
                first.get_or_insert(slot);
                matrix[token * rows + row] = slot.0;
            }
            let pad = first.ok_or(Qwen3ForwardError::CacheInconsistent)?;
            for token in length..longest {
                matrix[token * rows + row] = pad.0;
            }
        }
        Ok(Self {
            slots: Array::from_slice(&matrix, &[as_i32(longest)?, as_i32(rows)?]),
        })
    }
}

/// Validate device dimensions while the pool is still only a host plan.
fn pool_shape(
    pool: PoolConfig,
    key_value_heads: usize,
    head_dim: usize,
) -> Result<[i32; 3], Qwen3ForwardError> {
    let slots = u64::from(pool.num_blocks()) * u64::from(pool.block_tokens().get());
    Ok([
        i32::try_from(slots).map_err(|_| Qwen3ForwardError::ShapeOverflow)?,
        as_i32(key_value_heads)?,
        as_i32(head_dim)?,
    ])
}

impl KvPool {
    fn new(
        config: &Qwen3ForwardConfig,
        pool: PoolConfig,
        dtype: Dtype,
    ) -> Result<Self, Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        let shape = pool_shape(pool, config.key_value_heads, config.head_dim)?;
        let mut layers = Vec::with_capacity(config.hidden_layers);
        for _ in 0..config.hidden_layers {
            let keys = ops::zeros_dtype_device(&shape, dtype, &stream)?;
            let values = ops::zeros_dtype_device(&shape, dtype, &stream)?;
            // Materialize now: the pool is a startup allocation, not a lazy
            // graph that the first request would pay for.
            keys.eval()?;
            values.eval()?;
            layers.push(LayerKv { keys, values });
        }
        Ok(Self {
            dtype,
            block_tokens: pool.block_tokens().get(),
            layers,
        })
    }

    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.keys.nbytes() + layer.values.nbytes())
            .sum()
    }

    /// Copies the first `tokens` slots of `copy.src` into `copy.dst` in every
    /// layer. The rows are gathered into their own arrays and evaluated
    /// before the update: a lazy slice of the pool would still hold the pool
    /// buffer when the update runs, and MLX would then copy the whole layer
    /// array instead of writing in place.
    fn copy_block(&mut self, copy: BlockCopy) -> Result<(), Qwen3ForwardError> {
        if copy.tokens == 0 {
            return Ok(());
        }
        let stream = StreamOrDevice::gpu();
        let first = |block: BlockId| PoolSlot::of(Slot { block, offset: 0 }, self.block_tokens);
        let (src, dst) = (first(copy.src)?.0, first(copy.dst)?.0);
        let tokens = as_i32(copy.tokens)?;
        let source = Array::arange_device::<i32, i32>(src, src + tokens, None, &stream)?;
        let mut rows = Vec::with_capacity(self.layers.len() * 2);
        for layer in &self.layers {
            for array in [&layer.keys, &layer.values] {
                rows.push(array.take_axis_device(&source, 0, &stream)?);
            }
        }
        mlx_rs::transforms::eval(&rows)?;
        let mut rows = rows.into_iter();
        for layer in &mut self.layers {
            for array in [&mut layer.keys, &mut layer.values] {
                let block = rows.next().ok_or(Qwen3ForwardError::CacheInconsistent)?;
                array.index_mut_device(dst..dst + tokens, &block, &stream);
            }
        }
        Ok(())
    }

    /// Scatters `[tokens, 1, kv_heads, head_dim]` key and value rows into
    /// their slots.
    fn write(
        &mut self,
        layer: usize,
        key: &Array,
        value: &Array,
        plan: &WritePlan,
    ) -> Result<(), Qwen3ForwardError> {
        for array in [key, value] {
            if array.dtype() != self.dtype {
                return Err(Qwen3ForwardError::KvPoolDtype {
                    pool: self.dtype,
                    activation: array.dtype(),
                });
            }
        }
        let stream = StreamOrDevice::gpu();
        let layer = self
            .layers
            .get_mut(layer)
            .ok_or(Qwen3ForwardError::CacheInconsistent)?;
        layer.keys =
            ops::indexing::scatter_single_device(&layer.keys, &plan.slots, key, 0, &stream)?;
        layer.values =
            ops::indexing::scatter_single_device(&layer.values, &plan.slots, value, 0, &stream)?;
        Ok(())
    }

    /// Gathers each row's keys and values as a `[rows, kv_heads, tokens,
    /// head_dim]` view of one `take`.
    fn gather(&self, layer: usize, plan: &GatherPlan) -> Result<(Array, Array), Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        let layer = self
            .layers
            .get(layer)
            .ok_or(Qwen3ForwardError::CacheInconsistent)?;
        let gather = |array: &Array| -> Result<Array, Qwen3ForwardError> {
            // [tokens, rows, kv_heads, head_dim] -> [rows, kv_heads, tokens, head_dim]
            Ok(array
                .take_axis_device(&plan.slots, 0, &stream)?
                .transpose_axes_device(&[1, 2, 0, 3], &stream)?)
        };
        Ok((gather(&layer.keys)?, gather(&layer.values)?))
    }
}

impl<'a, S: BuildHasher> PagedQwen3Session<'a, S> {
    /// Sizes a pool from a K/V byte budget at the K/V precision these
    /// weights produce, rounding down to whole slabs.
    pub fn pool_for_budget(
        config: &Qwen3ForwardConfig,
        weights: &HashMap<String, Array, S>,
        budget_bytes: u64,
        block_tokens: BlockTokens,
    ) -> Result<PoolConfig, Qwen3ForwardError> {
        let per_token = config.cached_kv_bytes_at(1, kv_precision(config, weights)?)?;
        let pool = PoolConfig::from_budget(budget_bytes, per_token, block_tokens)?;
        pool_shape(pool, config.key_value_heads, config.head_dim)?;
        Ok(pool)
    }

    /// Allocates the pool at the K/V precision these weights produce.
    pub fn new(
        config: &'a Qwen3ForwardConfig,
        weights: &'a HashMap<String, Array, S>,
        pool: PoolConfig,
    ) -> Result<Self, Qwen3ForwardError> {
        if config.attention != Qwen3Attention::Causal {
            return Err(Qwen3ForwardError::CachedBidirectional);
        }
        let dtype = kv_precision(config, weights)?.dtype();
        Ok(Self {
            config,
            weights,
            blocks: BlockManager::new(pool),
            pool: KvPool::new(config, pool, dtype)?,
        })
    }

    /// The host-side block accounting.
    #[must_use]
    pub const fn blocks(&self) -> &BlockManager {
        &self.blocks
    }

    /// Bytes held by the device pool, fixed at construction.
    #[must_use]
    pub fn pool_bytes(&self) -> usize {
        self.pool.bytes()
    }

    /// Tokens whose K/V `seq` holds.
    pub fn cached_tokens(&self, seq: SequenceId) -> Result<usize, Qwen3ForwardError> {
        Ok(self.blocks.num_computed(seq)?)
    }

    /// Starts `seq` (replacing any earlier sequence with that ID), fills its
    /// K/V and returns the last prompt token's logits. Cached prefix blocks
    /// are reused when the pool has prefix caching enabled; this prefill is
    /// keyed with no salt and no extra keys.
    pub fn prefill_last_logits(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        Ok(self
            .prefill_with_keys(seq, input_ids, HashKeys::new())?
            .logits)
    }

    /// Free blocks a [`Self::prefill_with_keys`] of `input_ids` under `keys`
    /// would consume now, counting cached prefix blocks it would pin.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3ForwardError::KvBlocks`] if the cached prefix cannot
    /// be validated for this pool.
    pub fn prefill_cost(
        &self,
        input_ids: &[i32],
        keys: HashKeys,
    ) -> Result<usize, Qwen3ForwardError> {
        let tokens = token_ids(input_ids);
        let hit = self.blocks.lookup_prefix(&tokens, keys);
        let cached = hit.cached_tokens();
        Ok(self.blocks.admit_cost(&hit, tokens.len() - cached)?)
    }

    /// [`Self::prefill_last_logits`] under `keys`: blocks are shared only
    /// with prompts hashed under equal keys (the same cache salt and extra
    /// keys). Also reports how many leading prompt tokens came from cache.
    ///
    /// A pool without room returns [`Qwen3ForwardError::KvBlocks`] and leaves
    /// the manager unchanged.
    pub fn prefill_with_keys(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
        keys: HashKeys,
    ) -> Result<PagedPrefill, Qwen3ForwardError> {
        validate_input_ids(
            self.config,
            input_ids,
            0,
            self.config.max_position_embeddings,
        )?;
        if self.blocks.num_tokens(seq).is_ok() {
            self.blocks.free(seq)?;
        }
        let tokens = token_ids(input_ids);
        let hit = self.blocks.lookup_prefix(&tokens, keys);
        let cached = hit.cached_tokens();
        let allocation = self.blocks.admit(seq, hit, &tokens[cached..])?;
        let published_before = self.blocks.published_tokens();
        let logits = self.run(
            seq,
            &input_ids[cached..],
            allocation.positions,
            allocation.copy,
        )?;
        Ok(PagedPrefill {
            logits,
            cached_tokens: cached,
            cache_write_tokens: self.blocks.published_tokens() - published_before,
        })
    }

    /// Appends one or more tokens to a prefilled `seq` and returns the last
    /// one's logits.
    ///
    /// A pool without free blocks returns [`Qwen3ForwardError::KvBlocks`] and
    /// leaves `seq` unchanged, so a scheduler can preempt and retry.
    pub fn extend_last_logits(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        let cached = self.blocks.num_computed(seq)?;
        if cached == 0 {
            return Err(Qwen3ForwardError::DecodeWithoutPrefill);
        }
        validate_input_ids(
            self.config,
            input_ids,
            cached,
            self.config.max_position_embeddings,
        )?;
        let allocation = self.blocks.allocate(seq, &token_ids(input_ids))?;
        self.run(seq, input_ids, allocation.positions, allocation.copy)
    }

    /// Appends exactly one token to `seq` and returns its logits.
    pub fn decode_last_logits(
        &mut self,
        seq: SequenceId,
        input_id: i32,
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        self.extend_last_logits(seq, &[input_id])
    }

    /// Starts `child` as a copy of `parent` that shares every block. The
    /// first append by either into the shared partial tail block copies it.
    pub fn fork(&mut self, parent: SequenceId, child: SequenceId) -> Result<(), Qwen3ForwardError> {
        Ok(self.blocks.fork(parent, child)?)
    }

    /// Releases `seq`'s blocks. Before calling this, finish every queued step
    /// that references `seq`, including steps left outstanding after errors.
    /// A completed turn alone does not mean its queued writes have settled.
    pub fn free(&mut self, seq: SequenceId) -> Result<(), Qwen3ForwardError> {
        Ok(self.blocks.free(seq)?)
    }

    /// Appends one token to each of several prefilled sequences in one
    /// forward and reads back each row's logits or greedy token.
    ///
    /// Non-attention ops run on the packed `[1, rows, hidden]` batch, so the
    /// weights stream once per step. Attention gathers each row's blocks,
    /// applies `RoPE` at each row's own position, and masks positions past
    /// each row's length when the lengths differ. Rows run different kernels
    /// than a single decode (matrix-matrix instead of matrix-vector), so
    /// logits match [`Self::decode_last_logits`] to rounding, not bit for
    /// bit, except for a one-row batch.
    ///
    /// The block check is all or nothing: when the pool cannot take every
    /// row, nothing changes and [`Qwen3ForwardError::KvBlocks`] is returned.
    /// The check counts a copy for each row whose tail is shared, so two forks
    /// of one parent may be refused one block early. Any later failure frees
    /// every row's sequence.
    pub fn decode_batch(
        &mut self,
        rows: &[(SequenceId, i32)],
        readback: BatchReadback,
    ) -> Result<BatchDecoded, Qwen3ForwardError> {
        if rows.is_empty() {
            return Err(Qwen3ForwardError::EmptyInput);
        }
        let mut needed = 0;
        for (index, &(seq, token)) in rows.iter().enumerate() {
            if rows[..index].iter().any(|(earlier, _)| *earlier == seq) {
                return Err(Qwen3ForwardError::RepeatedBatchSequence(seq.0));
            }
            let cached = self.blocks.num_computed(seq)?;
            if cached == 0 {
                return Err(Qwen3ForwardError::DecodeWithoutPrefill);
            }
            validate_input_ids(
                self.config,
                &[token],
                cached,
                self.config.max_position_embeddings,
            )?;
            needed += self.blocks.append_cost(seq, 1)?;
        }
        let free = self.blocks.free_blocks();
        if needed > free {
            return Err(engine::blocks::BlockError::OutOfBlocks { needed, free }.into());
        }
        let result = self.batch_step(rows, readback).and_then(|decoded| {
            for &(seq, _) in rows {
                self.blocks.commit(seq)?;
            }
            Ok(decoded)
        });
        if result.is_err() {
            for &(seq, _) in rows {
                let _ = self.blocks.free(seq);
            }
        }
        result
    }

    /// Queues one greedy decode step without waiting for it, so the next step
    /// can be built before this one is read back. Each row's input is a host
    /// token, or [`StepInput::Previous`]: the token `previous` picked for one
    /// of its rows, still on the device. Such a row's slot is reserved
    /// unresolved, and [`Self::finish_decode`] of `previous` resolves it.
    ///
    /// Picks are the argmax of each row's logits, lowest ID on ties, as
    /// [`BatchReadback::Greedy`]. The block check is all or nothing, as for
    /// [`Self::decode_batch`]. On a later failure, rows remain allocated but
    /// their state may be partially advanced. Do not resume them: finish any
    /// outstanding steps that reference them, then explicitly [`Self::free`]
    /// each failed row. An error cannot free rows owned by `previous`.
    pub fn queue_decode(
        &mut self,
        rows: &[(SequenceId, StepInput)],
        previous: Option<&QueuedDecode>,
    ) -> Result<QueuedDecode, Qwen3ForwardError> {
        if rows.is_empty() {
            return Err(Qwen3ForwardError::EmptyInput);
        }
        let mut needed = 0;
        for (index, &(seq, input)) in rows.iter().enumerate() {
            if rows[..index].iter().any(|(earlier, _)| *earlier == seq) {
                return Err(Qwen3ForwardError::RepeatedBatchSequence(seq.0));
            }
            let scheduled = self.blocks.num_tokens(seq)?;
            if scheduled == 0 {
                return Err(Qwen3ForwardError::DecodeWithoutPrefill);
            }
            match input {
                StepInput::Host(token) => validate_input_ids(
                    self.config,
                    &[token],
                    scheduled,
                    self.config.max_position_embeddings,
                )?,
                StepInput::Previous(row) => {
                    let previous = previous.ok_or(Qwen3ForwardError::CacheInconsistent)?;
                    if previous.rows.get(row) != Some(&seq) {
                        return Err(Qwen3ForwardError::CacheInconsistent);
                    }
                    if scheduled >= self.config.max_position_embeddings {
                        return Err(Qwen3ForwardError::PromptTooLong {
                            actual: scheduled + 1,
                            maximum: self.config.max_position_embeddings,
                        });
                    }
                }
            }
            needed += self.blocks.append_cost(seq, 1)?;
        }
        let free = self.blocks.free_blocks();
        if needed > free {
            return Err(engine::blocks::BlockError::OutOfBlocks { needed, free }.into());
        }
        self.queue_step(rows, previous)
    }

    fn queue_step(
        &mut self,
        rows: &[(SequenceId, StepInput)],
        previous: Option<&QueuedDecode>,
    ) -> Result<QueuedDecode, Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        let batch = as_i32(rows.len())?;
        let reserved = rows
            .iter()
            .map(|&(seq, input)| {
                (
                    seq,
                    match input {
                        StepInput::Host(token) => Some(token),
                        StepInput::Previous(_) => None,
                    },
                )
            })
            .collect::<Vec<_>>();
        let slots = self.allocate_rows(&reserved)?;
        let lengths = slots.lengths.clone();
        let seqs = rows.iter().map(|&(seq, _)| seq).collect::<Vec<_>>();
        let host = rows
            .iter()
            .map(|&(_, input)| match input {
                StepInput::Host(token) => token,
                StepInput::Previous(_) => 0,
            })
            .collect::<Vec<_>>();
        let host = Array::from_slice(&host, &[batch]);
        let ids = match previous {
            Some(previous)
                if rows
                    .iter()
                    .any(|(_, input)| matches!(input, StepInput::Previous(_))) =>
            {
                let (from, carried): (Vec<i32>, Vec<bool>) = rows
                    .iter()
                    .map(|&(_, input)| match input {
                        StepInput::Previous(row) => as_i32(row).map(|row| (row, true)),
                        StepInput::Host(_) => Ok((0, false)),
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .unzip();
                let carried_tokens = previous
                    .picks
                    .tokens
                    .as_type_device::<i32>(&stream)?
                    .take_axis_device(Array::from_slice(&from, &[batch]), 0, &stream)?;
                ops::r#where_device(
                    Array::from_slice(&carried, &[batch]),
                    &carried_tokens,
                    &host,
                    &stream,
                )?
            }
            _ => host,
        };
        let logits = self.batch_logits(&seqs, slots, &ids)?;
        Ok(QueuedDecode {
            picks: super::Qwen3TokenPicks::start(&logits, 0)?,
            rows: seqs,
            lengths,
        })
    }

    /// Waits for a queued step, commits each live row through its step, and
    /// returns each row's picked token in row order. A row whose next step
    /// was queued with [`StepInput::Previous`] gets that slot's token
    /// resolved. Rows freed since queueing (a finished or cancelled turn) are
    /// skipped; their picks are returned but unused.
    ///
    /// On an error rows remain allocated but must not be resumed. The caller
    /// must finish all outstanding steps referencing them before explicitly
    /// freeing them; an already queued next step may still write their slots.
    pub fn finish_decode(&mut self, step: &QueuedDecode) -> Result<Vec<i32>, Qwen3ForwardError> {
        step.picks.wait().and_then(|tokens| {
            for ((&seq, &length), &token) in step.rows.iter().zip(&step.lengths).zip(&tokens) {
                if self.blocks.num_tokens(seq).is_err() {
                    continue;
                }
                self.blocks.commit_through(seq, length)?;
                if self.blocks.num_unresolved(seq)? > 0 {
                    self.blocks.resolve(seq, &token_ids(&[token]))?;
                }
            }
            Ok(tokens)
        })
    }

    /// Reserves one slot per row and performs any copy-on-write. A row with
    /// no host token reserves a slot whose input the device holds.
    fn allocate_rows(
        &mut self,
        rows: &[(SequenceId, Option<i32>)],
    ) -> Result<RowSlots, Qwen3ForwardError> {
        let mut slots = Vec::with_capacity(rows.len());
        let mut positions = Vec::with_capacity(rows.len());
        let mut lengths = Vec::with_capacity(rows.len());
        for &(seq, token) in rows {
            let allocation = match token {
                Some(token) => self.blocks.allocate(seq, &token_ids(&[token]))?,
                None => self.blocks.allocate_unresolved(seq, 1)?,
            };
            if let Some(copy) = allocation.copy {
                self.pool.copy_block(copy)?;
            }
            let slot = self
                .blocks
                .slots(seq, allocation.positions)?
                .next()
                .ok_or(Qwen3ForwardError::CacheInconsistent)?;
            slots.push(PoolSlot::of(slot, self.pool.block_tokens)?);
            positions.push(rope_offset(allocation.positions.start())?);
            lengths.push(allocation.positions.end().get());
        }
        Ok(RowSlots {
            slots,
            positions,
            lengths,
        })
    }

    fn batch_step(
        &mut self,
        rows: &[(SequenceId, i32)],
        readback: BatchReadback,
    ) -> Result<BatchDecoded, Qwen3ForwardError> {
        #[cfg(test)]
        let started = std::time::Instant::now();
        let batch = as_i32(rows.len())?;
        let host = rows
            .iter()
            .map(|&(seq, token)| (seq, Some(token)))
            .collect::<Vec<_>>();
        let slots = self.allocate_rows(&host)?;
        let seqs = rows.iter().map(|&(seq, _)| seq).collect::<Vec<_>>();
        let ids = Array::from_slice(
            &rows.iter().map(|&(_, token)| token).collect::<Vec<_>>(),
            &[batch],
        );
        let logits = self.batch_logits(&seqs, slots, &ids)?;
        #[cfg(test)]
        let built = std::time::Instant::now();
        let decoded = read_rows(&logits, self.config.vocab_size, readback);
        #[cfg(test)]
        profile::record(built - started, built.elapsed());
        decoded
    }

    /// Builds one batched decode step over `seqs`, whose slots are already
    /// reserved, with `ids` (`[rows]` int32, on the host or the device) as
    /// each row's input, and returns its `[rows, vocab]` logits, unevaluated.
    fn batch_logits(
        &mut self,
        seqs: &[SequenceId],
        slots: RowSlots,
        ids: &Array,
    ) -> Result<Array, Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        let batch = as_i32(seqs.len())?;
        let RowSlots {
            slots,
            positions,
            lengths,
        } = slots;
        let writes = WritePlan::new(&slots)?;
        let plan = GatherPlan::new(&self.blocks, seqs, &lengths)?;
        let longest = lengths.iter().copied().max().unwrap_or(0);
        let mask = length_mask(&lengths, longest)?;
        let offsets = Array::from_slice(&positions, &[batch]);

        let hidden = as_i32(self.config.hidden_size)?;
        let mut hidden_states = embed_rows(self.config, self.weights, ids)?
            .reshape_device(&[1, batch, hidden], &stream)?;
        for layer in 0..self.config.hidden_layers {
            let base = format!("model.layers.{layer}");
            let attn = format!("{base}.self_attn");
            let attention_input = rms_norm(
                &hidden_states,
                weight(self.weights, &format!("{base}.input_layernorm.weight"))?,
                self.config.rms_norm_eps,
            )?;
            // [rows, heads, 1, head_dim]
            let (query, key, value) = rotated_qkv(
                self.config,
                self.weights,
                &attn,
                &attention_input,
                batch,
                1,
                RopePositions::PerRow(&offsets),
            )?;
            // [rows, kv_heads, 1, head_dim] -> [rows, 1, kv_heads, head_dim]
            let shape = [batch, 1, key.shape()[1], key.shape()[3]];
            let key = key.reshape_device(&shape, &stream)?;
            let value = value.reshape_device(&shape, &stream)?;
            self.pool.write(layer, &key, &value, &writes)?;
            let (keys, values) = self.pool.gather(layer, &plan)?;
            let output = fast::scaled_dot_product_attention_device(
                &query,
                &keys,
                &values,
                attention_scale(self.config)?,
                mask.as_ref()
                    .map(fast::ScaledDotProductAttentionMask::Array),
                Option::<&Array>::None,
                &stream,
            )?;
            let attention = attention_output(self.config, self.weights, &attn, &output, batch)?;
            let residual = hidden_states.add_device(&attention, &stream)?;
            hidden_states = mlp_residual(self.config, self.weights, &base, &residual, batch)?;
        }

        let normalized = rms_norm(
            &hidden_states,
            weight(self.weights, "model.norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        let vocab = as_i32(self.config.vocab_size)?;
        Ok(project(
            self.config,
            self.weights,
            &normalized,
            self.config.output_projection(),
        )?
        .reshape_device(&[batch, vocab], &stream)?)
    }

    /// Runs the forward for tokens already given slots, then commits them.
    /// Any failure after the slots were assigned frees `seq`: its blocks may
    /// hold partially written K/V.
    fn run(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
        positions: TokenSpan,
        copy: Option<BlockCopy>,
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        let result = self
            .forward(seq, input_ids, positions, copy)
            .and_then(|logits| {
                self.blocks.commit(seq)?;
                Ok(logits)
            });
        if result.is_err() {
            let _ = self.blocks.free(seq);
        }
        result
    }

    fn forward(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
        positions: TokenSpan,
        copy: Option<BlockCopy>,
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
        if positions.len() != input_ids.len() {
            return Err(Qwen3ForwardError::CacheInconsistent);
        }
        let stream = StreamOrDevice::gpu();
        let slots = self
            .blocks
            .slots(seq, positions)?
            .map(|slot| PoolSlot::of(slot, self.pool.block_tokens))
            .collect::<Result<Vec<_>, _>>()?;
        let writes = WritePlan::new(&slots)?;
        let plan = GatherPlan::new(&self.blocks, &[seq], &[positions.end().get()])?;
        if let Some(copy) = copy {
            self.pool.copy_block(copy)?;
        }

        let seq_len = as_i32(input_ids.len())?;
        let start = rope_offset(positions.start())?;
        // The contiguous executor's masks: plain causal for a sequence's
        // first chunk, an explicit offset mask for a later multi-token chunk,
        // none for one token.
        let chunk_mask = if start > 0 && seq_len > 1 {
            Some(chunk_causal_mask(start, seq_len, &stream)?)
        } else {
            None
        };
        let mask = || {
            if start == 0 {
                Some(fast::ScaledDotProductAttentionMask::Causal)
            } else {
                chunk_mask
                    .as_ref()
                    .map(fast::ScaledDotProductAttentionMask::Array)
            }
        };

        let hidden = as_i32(self.config.hidden_size)?;
        let ids = Array::from_slice(input_ids, &[seq_len]);
        let mut hidden_states = embed_rows(self.config, self.weights, &ids)?
            .reshape_device(&[1, seq_len, hidden], &stream)?;
        for layer in 0..self.config.hidden_layers {
            let base = format!("model.layers.{layer}");
            let attn = format!("{base}.self_attn");
            let attention_input = rms_norm(
                &hidden_states,
                weight(self.weights, &format!("{base}.input_layernorm.weight"))?,
                self.config.rms_norm_eps,
            )?;
            let (query, key, value) = rotated_qkv(
                self.config,
                self.weights,
                &attn,
                &attention_input,
                1,
                seq_len,
                RopePositions::Shared(start),
            )?;
            // [1, kv_heads, tokens, head_dim] -> [tokens, 1, kv_heads, head_dim]
            let key = key.transpose_axes_device(&[2, 0, 1, 3], &stream)?;
            let value = value.transpose_axes_device(&[2, 0, 1, 3], &stream)?;
            self.pool.write(layer, &key, &value, &writes)?;
            let (keys, values) = self.pool.gather(layer, &plan)?;
            let output = fast::scaled_dot_product_attention_device(
                &query,
                &keys,
                &values,
                attention_scale(self.config)?,
                mask(),
                Option::<&Array>::None,
                &stream,
            )?;
            let attention = attention_output(self.config, self.weights, &attn, &output, seq_len)?;
            let residual = hidden_states.add_device(&attention, &stream)?;
            hidden_states = mlp_residual(self.config, self.weights, &base, &residual, seq_len)?;
        }

        let last_hidden =
            hidden_states.take_axis_device(Array::from_slice(&[seq_len - 1], &[1]), 1, &stream)?;
        let normalized = rms_norm(
            &last_hidden,
            weight(self.weights, "model.norm.weight")?,
            self.config.rms_norm_eps,
        )?;
        let logits = project(
            self.config,
            self.weights,
            &normalized,
            self.config.output_projection(),
        )?;
        read_last_logits(&logits, 1, self.config.vocab_size)
    }

    /// The device arrays of one layer, for donation checks.
    #[cfg(test)]
    pub(super) fn layer_arrays(&self, layer: usize) -> (&Array, &Array) {
        let layer = &self.pool.layers[layer];
        (&layer.keys, &layer.values)
    }
}

/// `[rows, 1, 1, longest]` mask of each row's valid keys, or `None` when
/// every row has the longest length and every gathered key is valid.
fn length_mask(lengths: &[usize], longest: usize) -> Result<Option<Array>, Qwen3ForwardError> {
    if lengths.iter().all(|&length| length == longest) {
        return Ok(None);
    }
    let stream = StreamOrDevice::gpu();
    let rows = as_i32(lengths.len())?;
    let lengths = lengths
        .iter()
        .map(|&length| as_i32(length))
        .collect::<Result<Vec<_>, _>>()?;
    let keys = Array::arange_device::<i32, i32>(0, as_i32(longest)?, None, &stream)?
        .reshape_device(&[1, 1, 1, as_i32(longest)?], &stream)?;
    Ok(Some(keys.lt_device(
        Array::from_slice(&lengths, &[rows, 1, 1, 1]),
        &stream,
    )?))
}

/// Evaluates `[rows, vocab]` logits and reads back what `readback` asks for.
fn read_rows(
    logits: &Array,
    vocab_size: usize,
    readback: BatchReadback,
) -> Result<BatchDecoded, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    match readback {
        BatchReadback::Logits => {
            let logits = logits.as_type_device::<f32>(&stream)?;
            logits.eval()?;
            Ok(BatchDecoded::Logits(
                logits
                    .as_slice::<f32>()
                    .chunks_exact(vocab_size)
                    .map(<[f32]>::to_vec)
                    .collect(),
            ))
        }
        // The same pick as single-sequence greedy decode, including its
        // refusal of non-finite rows. Binding 0 matches no executor, so these
        // picks cannot be fed to `decode_greedy_after`.
        BatchReadback::Greedy => Ok(BatchDecoded::Greedy(
            super::Qwen3TokenPicks::start(logits, 0)?.wait()?,
        )),
    }
}

fn rope_offset(position: TokenPosition) -> Result<i32, Qwen3ForwardError> {
    as_i32(position.get())
}

/// Token IDs as the block manager hashes them. Callers validate the IDs
/// against the vocabulary first, so none is negative.
fn token_ids(input_ids: &[i32]) -> Vec<u32> {
    input_ids.iter().map(|&id| id.cast_unsigned()).collect()
}

/// Test-only host timings of [`PagedQwen3Session::decode_batch`] steps:
/// building the lazy graph, and evaluating plus reading it back.
#[cfg(test)]
pub(super) mod profile {
    use std::{cell::RefCell, time::Duration};

    thread_local! {
        static STEPS: RefCell<Vec<(Duration, Duration)>> = const { RefCell::new(Vec::new()) };
    }

    pub(in super::super) fn record(build: Duration, evaluate: Duration) {
        STEPS.with(|steps| steps.borrow_mut().push((build, evaluate)));
    }

    /// Returns and clears the recorded `(build, evaluate)` pairs.
    pub(in super::super) fn take() -> Vec<(Duration, Duration)> {
        STEPS.with(|steps| std::mem::take(&mut *steps.borrow_mut()))
    }
}

#[cfg(test)]
mod tests {
    use engine::blocks::{
        BlockManager, BlockTokens, HashKeys, PoolConfig, SequenceId, Slot, TokenSpan,
    };

    use super::{GatherPlan, PoolSlot};

    #[test]
    fn pool_plan_rejects_device_shape_overflow_without_allocating() {
        let pool = PoolConfig::new(BlockTokens::DEFAULT, 1).unwrap();
        assert_eq!(super::pool_shape(pool, 8, 128).unwrap(), [512, 8, 128]);
        let oversized = PoolConfig::new(BlockTokens::DEFAULT, 4_194_304).unwrap();
        assert!(matches!(
            super::pool_shape(oversized, 8, 128),
            Err(super::Qwen3ForwardError::ShapeOverflow)
        ));
        assert!(matches!(
            super::pool_shape(pool, 2_147_483_648, 128),
            Err(super::Qwen3ForwardError::ShapeOverflow)
        ));
    }

    #[test]
    fn gather_plan_is_token_major_and_pads_with_the_first_slot() {
        let mut blocks = BlockManager::new(PoolConfig::new(BlockTokens::DEFAULT, 2).expect("pool"));
        for (seq, len) in [(1, 20), (2, 3)] {
            let prompt = vec![7_u32; len];
            let hit = blocks.lookup_prefix(&prompt, HashKeys::new());
            blocks.admit(SequenceId(seq), hit, &prompt).expect("admit");
        }
        let slots = |seq: u64, len: usize| -> Vec<i32> {
            blocks
                .slots(SequenceId(seq), TokenSpan::prefix(len))
                .expect("in range")
                .map(|slot: Slot| PoolSlot::of(slot, 16).expect("small").0)
                .collect()
        };
        let (first, second) = (slots(1, 20), slots(2, 3));
        let plan =
            GatherPlan::new(&blocks, &[SequenceId(1), SequenceId(2)], &[20, 3]).expect("plan");
        assert_eq!(plan.slots.shape(), &[20, 2]);
        let matrix = plan.slots.as_slice::<i32>().to_vec();
        for token in 0..20 {
            assert_eq!(matrix[token * 2], first[token], "row 0 token {token}");
            let expected = second.get(token).copied().unwrap_or(second[0]);
            assert_eq!(matrix[token * 2 + 1], expected, "row 1 token {token}");
        }
    }
}
