//! Paged K/V storage for Qwen3: one preallocated device pool shared by many
//! sequences, addressed through [`engine::blocks`] block tables.
//!
//! Each layer stores K and V as slabs of [`SLAB_BLOCKS`] blocks,
//! `[SLAB_BLOCKS, kv_heads, block_tokens, head_dim]`. New K/V rows are written
//! with `slice_update` at the slots the block manager assigns. MLX reuses the
//! slab's buffer for that update only while the pool holds the slab's sole
//! reference, so nothing outside this module may clone a slab handle; forks
//! share blocks through the manager's reference counts instead.
//!
//! Attention gathers a sequence's blocks into `[1, kv_heads, blocks *
//! block_tokens, head_dim]`, slices the valid length, and runs the same fused
//! SDPA and masks as the contiguous executor, so a single sequence produces
//! the same logits as [`super::Qwen3ForwardExecutor`].

use std::{collections::HashMap, hash::BuildHasher};

use engine::blocks::{
    BlockCopy, BlockId, BlockManager, BlockTokens, HashKeys, PoolConfig, SLAB_BLOCKS, SequenceId,
    Slot, TokenPosition, TokenSpan,
};
use mlx_rs::{
    Array, Dtype, StreamOrDevice, fast, ops,
    ops::indexing::{IndexMutOp, IndexOp},
};

use super::{
    Qwen3ForwardConfig, Qwen3ForwardError, as_i32, attention_output, attention_scale,
    chunk_causal_mask, linear, mlp_residual, read_last_logits, rms_norm, rotated_qkv,
    validate_input_ids, weight,
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
    keys: HashKeys,
}

/// The device side of the pool: per layer, one K and one V array per slab.
struct KvPool {
    dtype: Dtype,
    block_tokens: i32,
    layers: Vec<LayerSlabs>,
}

struct LayerSlabs {
    keys: Vec<Array>,
    values: Vec<Array>,
}

/// Where a slot lives in the device arrays. Named fields keep the slab, the
/// block row inside it, and the token offset inside that block apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SlabCoordinate {
    slab: usize,
    row: i32,
    offset: i32,
}

impl SlabCoordinate {
    fn of(block: BlockId, offset: u32) -> Result<Self, Qwen3ForwardError> {
        Ok(Self {
            slab: block.slab() as usize,
            row: i32::try_from(block.index_in_slab())
                .map_err(|_| Qwen3ForwardError::ShapeOverflow)?,
            offset: i32::try_from(offset).map_err(|_| Qwen3ForwardError::ShapeOverflow)?,
        })
    }
}

/// Consecutive new tokens that land in one block: chunk rows
/// `chunk_start..chunk_start + len` go to `start` and the offsets after it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WriteRun {
    start: SlabCoordinate,
    chunk_start: i32,
    len: i32,
}

/// Groups slots, in position order, into per-block runs.
fn write_runs(slots: &[Slot]) -> Result<Vec<WriteRun>, Qwen3ForwardError> {
    let mut runs: Vec<WriteRun> = Vec::new();
    let mut previous: Option<Slot> = None;
    for (row, slot) in slots.iter().enumerate() {
        let row = as_i32(row)?;
        match (previous, runs.last_mut()) {
            (Some(last), Some(run))
                if last.block == slot.block && last.offset + 1 == slot.offset =>
            {
                run.len += 1;
            }
            _ => runs.push(WriteRun {
                start: SlabCoordinate::of(slot.block, slot.offset)?,
                chunk_start: row,
                len: 1,
            }),
        }
        previous = Some(*slot);
    }
    Ok(runs)
}

/// The blocks one sequence reads, grouped into runs that share a slab so
/// each run is one `take`.
struct GatherPlan {
    runs: Vec<(usize, Array)>,
    blocks: i32,
    tokens: i32,
}

impl GatherPlan {
    fn new(table: &[BlockId], tokens: usize) -> Result<Self, Qwen3ForwardError> {
        let runs = table
            .chunk_by(|left, right| left.slab() == right.slab())
            .map(|run| {
                let rows = run
                    .iter()
                    .map(|block| {
                        i32::try_from(block.index_in_slab())
                            .map_err(|_| Qwen3ForwardError::ShapeOverflow)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let len = as_i32(rows.len())?;
                Ok((run[0].slab() as usize, Array::from_slice(&rows, &[len])))
            })
            .collect::<Result<Vec<_>, Qwen3ForwardError>>()?;
        Ok(Self {
            runs,
            blocks: as_i32(table.len())?,
            tokens: as_i32(tokens)?,
        })
    }
}

impl KvPool {
    fn new(
        config: &Qwen3ForwardConfig,
        pool: PoolConfig,
        dtype: Dtype,
    ) -> Result<Self, Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        let block_tokens = i32::try_from(pool.block_tokens().get())
            .map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
        let shape = [
            i32::try_from(SLAB_BLOCKS).map_err(|_| Qwen3ForwardError::ShapeOverflow)?,
            as_i32(config.key_value_heads)?,
            block_tokens,
            as_i32(config.head_dim)?,
        ];
        let slabs = pool.slabs() as usize;
        let mut layers = Vec::with_capacity(config.hidden_layers);
        for _ in 0..config.hidden_layers {
            let mut keys = Vec::with_capacity(slabs);
            let mut values = Vec::with_capacity(slabs);
            for _ in 0..slabs {
                let key = ops::zeros_dtype_device(&shape, dtype, &stream)?;
                let value = ops::zeros_dtype_device(&shape, dtype, &stream)?;
                // Materialize now: the pool is a startup allocation, not a
                // lazy graph that the first request would pay for.
                key.eval()?;
                value.eval()?;
                keys.push(key);
                values.push(value);
            }
            layers.push(LayerSlabs { keys, values });
        }
        Ok(Self {
            dtype,
            block_tokens,
            layers,
        })
    }

    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|layer| layer.keys.iter().chain(&layer.values))
            .map(Array::nbytes)
            .sum()
    }

    /// Copies the first `tokens` positions of `copy.src` into `copy.dst` in
    /// every layer. When both blocks share a slab the read keeps the old slab
    /// alive during the update, so MLX copies that one slab; forks are rare
    /// enough that this is not worth a two-step write.
    fn copy_block(&mut self, copy: BlockCopy) -> Result<(), Qwen3ForwardError> {
        if copy.tokens == 0 {
            return Ok(());
        }
        let stream = StreamOrDevice::gpu();
        let src = SlabCoordinate::of(copy.src, 0)?;
        let dst = SlabCoordinate::of(copy.dst, 0)?;
        let tokens = as_i32(copy.tokens)?;
        for layer in &mut self.layers {
            for slabs in [&mut layer.keys, &mut layer.values] {
                let rows = slabs[src.slab]
                    .index_device((src.row..src.row + 1, .., 0..tokens, ..), &stream);
                slabs[dst.slab].index_mut_device(
                    (dst.row..dst.row + 1, .., 0..tokens, ..),
                    &rows,
                    &stream,
                );
            }
        }
        Ok(())
    }

    /// Writes `[1, kv_heads, chunk, head_dim]` key and value rows into their
    /// slots.
    fn write(
        &mut self,
        layer: usize,
        key: &Array,
        value: &Array,
        runs: &[WriteRun],
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
        let slabs = self
            .layers
            .get_mut(layer)
            .ok_or(Qwen3ForwardError::CacheInconsistent)?;
        for run in runs {
            let rows = run.chunk_start..run.chunk_start + run.len;
            let target = (
                run.start.row..run.start.row + 1,
                ..,
                run.start.offset..run.start.offset + run.len,
                ..,
            );
            slabs.keys[run.start.slab].index_mut_device(
                target.clone(),
                &key.index_device((.., .., rows.clone(), ..), &stream),
                &stream,
            );
            slabs.values[run.start.slab].index_mut_device(
                target,
                &value.index_device((.., .., rows, ..), &stream),
                &stream,
            );
        }
        Ok(())
    }

    /// Gathers one sequence's keys and values as `[1, kv_heads, tokens,
    /// head_dim]`.
    fn gather(&self, layer: usize, plan: &GatherPlan) -> Result<(Array, Array), Qwen3ForwardError> {
        let stream = StreamOrDevice::gpu();
        let slabs = self
            .layers
            .get(layer)
            .ok_or(Qwen3ForwardError::CacheInconsistent)?;
        let gather = |arrays: &[Array]| -> Result<Array, Qwen3ForwardError> {
            let mut parts = plan
                .runs
                .iter()
                .map(|(slab, rows)| arrays[*slab].take_axis_device(rows, 0, &stream))
                .collect::<Result<Vec<_>, _>>()?;
            // [blocks, kv_heads, block_tokens, head_dim]
            let blocks = if parts.len() == 1 {
                parts.pop().ok_or(Qwen3ForwardError::CacheInconsistent)?
            } else {
                ops::concatenate_axis_device(&parts, 0, &stream)?
            };
            let shape = blocks.shape();
            let (kv_heads, head_dim) = (shape[1], shape[3]);
            let tokens = plan
                .blocks
                .checked_mul(self.block_tokens)
                .ok_or(Qwen3ForwardError::ShapeOverflow)?;
            Ok(blocks
                .transpose_axes_device(&[1, 0, 2, 3], &stream)?
                .reshape_device(&[1, kv_heads, tokens, head_dim], &stream)?
                .index_device((.., .., 0..plan.tokens, ..), &stream))
        };
        Ok((gather(&slabs.keys)?, gather(&slabs.values)?))
    }
}

impl<'a, S: BuildHasher> PagedQwen3Session<'a, S> {
    /// Sizes a pool from a K/V byte budget for these weights' activation
    /// dtype, rounding down to whole slabs.
    pub fn pool_for_budget(
        config: &Qwen3ForwardConfig,
        weights: &HashMap<String, Array, S>,
        budget_bytes: u64,
        block_tokens: BlockTokens,
    ) -> Result<PoolConfig, Qwen3ForwardError> {
        let item_size = weight(weights, "model.embed_tokens.weight")?.item_size();
        // `cached_kv_bytes` counts f32 elements; rescale to the pool dtype.
        let per_token = (config.cached_kv_bytes(1)? / 4)
            .checked_mul(u64::try_from(item_size).map_err(|_| Qwen3ForwardError::ShapeOverflow)?)
            .ok_or(Qwen3ForwardError::ShapeOverflow)?;
        Ok(PoolConfig::from_budget(
            budget_bytes,
            per_token,
            block_tokens,
        )?)
    }

    /// Allocates the pool. K/V take the token embedding's dtype, which is the
    /// activation dtype of every layer.
    pub fn new(
        config: &'a Qwen3ForwardConfig,
        weights: &'a HashMap<String, Array, S>,
        pool: PoolConfig,
    ) -> Result<Self, Qwen3ForwardError> {
        if config.attention != Qwen3Attention::Causal {
            return Err(Qwen3ForwardError::CachedBidirectional);
        }
        let dtype = weight(weights, "model.embed_tokens.weight")?.dtype();
        Ok(Self {
            config,
            weights,
            blocks: BlockManager::new(pool),
            pool: KvPool::new(config, pool, dtype)?,
            keys: HashKeys::new(),
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
    /// are reused when the pool has prefix caching enabled.
    pub fn prefill_last_logits(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
    ) -> Result<Vec<f32>, Qwen3ForwardError> {
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
        let hit = self.blocks.lookup_prefix(&tokens, self.keys.clone());
        let cached = hit.cached_tokens();
        let allocation = self.blocks.admit(seq, hit, &tokens[cached..])?;
        self.run(
            seq,
            &input_ids[cached..],
            allocation.positions,
            allocation.copy,
        )
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

    /// Releases `seq`'s blocks.
    pub fn free(&mut self, seq: SequenceId) -> Result<(), Qwen3ForwardError> {
        Ok(self.blocks.free(seq)?)
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
        let slots = self.blocks.slots(seq, positions)?.collect::<Vec<_>>();
        let runs = write_runs(&slots)?;
        let plan = GatherPlan::new(self.blocks.block_table(seq)?, positions.end().get())?;
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
        let mut hidden_states = weight(self.weights, "model.embed_tokens.weight")?
            .take_axis_device(&ids, 0, &stream)?
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
                seq_len,
                start,
            )?;
            self.pool.write(layer, &key, &value, &runs)?;
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
        let logits = linear(
            &normalized,
            weight(self.weights, self.config.output_weight_name())?,
        )?;
        read_last_logits(&logits, 1, self.config.vocab_size)
    }

    /// The device arrays of one layer's slab, for donation checks.
    #[cfg(test)]
    pub(super) fn slab(&self, layer: usize, slab: usize) -> (&Array, &Array) {
        let layer = &self.pool.layers[layer];
        (&layer.keys[slab], &layer.values[slab])
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

#[cfg(test)]
mod tests {
    use engine::blocks::{
        BlockManager, BlockTokens, HashKeys, PoolConfig, SequenceId, TokenPosition, TokenSpan,
    };

    use super::{SlabCoordinate, WriteRun, write_runs};

    #[test]
    fn write_runs_split_at_block_boundaries() {
        // 33 full blocks plus 3 tokens: the span 526..531 ends block 32 and
        // starts block 33, which sit in different slabs when the free queue
        // hands blocks out in index order.
        let mut blocks = BlockManager::new(PoolConfig::new(BlockTokens::DEFAULT, 2).expect("pool"));
        let prompt = vec![7_u32; 16 * 33 + 3];
        let hit = blocks.lookup_prefix(&prompt, HashKeys::new());
        blocks.admit(SequenceId(1), hit, &prompt).expect("admit");
        let table = blocks.block_table(SequenceId(1)).expect("live").to_vec();
        let span = TokenSpan::new(TokenPosition::new(16 * 33 - 2), 5);
        let slots = blocks
            .slots(SequenceId(1), span)
            .expect("in range")
            .collect::<Vec<_>>();
        assert_eq!(
            write_runs(&slots).expect("runs"),
            [
                WriteRun {
                    start: SlabCoordinate::of(table[32], 14).expect("coordinate"),
                    chunk_start: 0,
                    len: 2,
                },
                WriteRun {
                    start: SlabCoordinate::of(table[33], 0).expect("coordinate"),
                    chunk_start: 2,
                    len: 3,
                },
            ]
        );
        assert_eq!(
            SlabCoordinate::of(table[33], 1).expect("coordinate"),
            SlabCoordinate {
                slab: table[33].slab() as usize,
                row: i32::try_from(table[33].index_in_slab()).expect("small"),
                offset: 1,
            }
        );
    }
}
