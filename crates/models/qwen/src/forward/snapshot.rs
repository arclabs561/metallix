//! Detached resident K/V prefixes that outlive one executor.
//!
//! A snapshot holds compact copies of the first `tokens` positions of every
//! layer's K/V, or, when it extends a shared base snapshot, copies of only the
//! positions after the base's. It carries the resident plan it was taken under and an opaque
//! binding chosen by the weights owner, so a later restore can refuse a
//! snapshot from another checkpoint load. Fields stay private: the only way to
//! apply one is [`crate::metal::Qwen3MlxWeights::resident_chat_executor_from`].

use std::{hash::BuildHasher, rc::Rc};

use mlx_rs::{Array, StreamOrDevice, ops};

use super::{
    Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor, Qwen3LayerKv,
    Qwen3ResidentChatPlan, as_i32,
};

/// Compact K/V for a token prefix of one resident-chat sequence.
pub struct Qwen3KvSnapshot {
    /// Positions `base.tokens()..tokens`, or `0..tokens` without a base.
    layers: Vec<Qwen3LayerKv>,
    /// The shared snapshot holding this prefix's leading positions.
    base: Option<Rc<Qwen3KvSnapshot>>,
    tokens: usize,
    plan: Qwen3ResidentChatPlan,
    binding: u64,
}

impl Qwen3KvSnapshot {
    /// Number of prompt positions this snapshot covers.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.tokens
    }

    /// Byte total of this snapshot's own K and V arrays, excluding allocator
    /// overhead and any shared base, which is counted once by its owner.
    #[must_use]
    pub fn kv_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.keys.nbytes() + layer.values.nbytes())
            .sum()
    }

    /// The shared snapshot this one extends, if any.
    #[must_use]
    pub fn base(&self) -> Option<&Rc<Self>> {
        self.base.as_ref()
    }

    pub(crate) const fn binding(&self) -> u64 {
        self.binding
    }

    pub(crate) const fn plan(&self) -> Qwen3ResidentChatPlan {
        self.plan
    }
}

impl<'a, S: BuildHasher> Qwen3ForwardExecutor<'a, S> {
    /// Whether this executor borrows exactly `weights`.
    pub(crate) fn borrows(&self, weights: &std::collections::HashMap<String, Array, S>) -> bool {
        std::ptr::eq(self.weights, weights)
    }

    /// Copies the first `tokens` cached positions into a detached snapshot.
    ///
    /// Only an unsteered resident-chat executor qualifies: a residual
    /// intervention would make the K/V depend on more than the token prefix.
    /// The copies are evaluated before return, so the snapshot never pins this
    /// executor's larger stepped storage.
    pub(crate) fn snapshot_prefix(
        &self,
        tokens: usize,
        binding: u64,
    ) -> Result<Qwen3KvSnapshot, Qwen3ForwardError> {
        self.snapshot_after(None, tokens, binding)
    }

    /// Like [`Self::snapshot_prefix`], but copies only the positions after
    /// `base`'s and shares `base` for the rest.
    ///
    /// The caller owns the claim that this executor's first `base.tokens()`
    /// positions hold exactly `base`'s K/V: it was restored from `base` (or
    /// from a snapshot extending it), or `base` was taken from this executor.
    /// Binding, plan and length are checked here; the values are not.
    pub(crate) fn snapshot_extending(
        &self,
        base: Rc<Qwen3KvSnapshot>,
        tokens: usize,
        binding: u64,
    ) -> Result<Qwen3KvSnapshot, Qwen3ForwardError> {
        if base.binding != binding
            || Some(base.plan) != self.resident_chat_plan
            || base.tokens >= tokens
        {
            return Err(Qwen3ForwardError::KvSnapshotMismatch);
        }
        self.snapshot_after(Some(base), tokens, binding)
    }

    fn snapshot_after(
        &self,
        base: Option<Rc<Qwen3KvSnapshot>>,
        tokens: usize,
        binding: u64,
    ) -> Result<Qwen3KvSnapshot, Qwen3ForwardError> {
        let plan = self
            .resident_chat_plan
            .ok_or(Qwen3ForwardError::KvSnapshotUnsupported)?;
        if self.residual_steering.is_some() || tokens == 0 || tokens > self.cached_tokens {
            return Err(Qwen3ForwardError::KvSnapshotUnsupported);
        }
        let stream = StreamOrDevice::gpu();
        let start = as_i32(base.as_ref().map_or(0, |base| base.tokens))?;
        let rows = as_i32(tokens)?;
        // A gather writes fresh rows. A slice written into a new array still
        // kept the whole stepped storage alive after evaluation, so each
        // snapshot held several times its `kv_bytes`.
        let positions = Array::arange_device::<i32, i32>(start, rows, None, &stream)?;
        let compact = |storage: &Array| -> Result<Array, Qwen3ForwardError> {
            let copy = storage.take_axis_device(&positions, 2, &stream)?;
            copy.eval()?;
            Ok(copy)
        };
        let layers = self
            .cache
            .iter()
            .map(|entry| {
                let layer = entry.as_ref().ok_or(Qwen3ForwardError::CacheInconsistent)?;
                Ok(Qwen3LayerKv {
                    keys: compact(&layer.keys)?,
                    values: compact(&layer.values)?,
                })
            })
            .collect::<Result<Vec<_>, Qwen3ForwardError>>()?;
        Ok(Qwen3KvSnapshot {
            layers,
            base,
            tokens,
            plan,
            binding,
        })
    }

    /// Starts a resident executor whose cache is `snapshot`'s prefix.
    ///
    /// The caller has already matched the binding and plan. Without a base,
    /// the compact arrays are shared, not copied; with one, each layer is the
    /// base's positions followed by the snapshot's own, in one new array.
    /// Either way the next append re-tiers them into new storage, so the
    /// snapshot stays unchanged. Until that append the executor cannot be
    /// forked.
    pub(crate) fn from_snapshot(
        config: &'a Qwen3ForwardConfig,
        weights: &'a std::collections::HashMap<String, Array, S>,
        snapshot: &Qwen3KvSnapshot,
    ) -> Result<Self, Qwen3ForwardError> {
        let mut executor = Self::new_for_resident_chat(config, weights, snapshot.plan);
        executor.cache = (0..snapshot.layers.len())
            .map(|layer| snapshot.layer_kv(layer).map(Some))
            .collect::<Result<_, _>>()?;
        executor.cached_tokens = snapshot.tokens;
        Ok(executor)
    }
}

impl Qwen3KvSnapshot {
    /// One layer's K/V over all `tokens` positions, joining a base's
    /// positions ahead of this snapshot's own.
    fn layer_kv(&self, layer: usize) -> Result<Qwen3LayerKv, Qwen3ForwardError> {
        let own = self
            .layers
            .get(layer)
            .ok_or(Qwen3ForwardError::CacheInconsistent)?;
        let Some(base) = &self.base else {
            return Ok(Qwen3LayerKv {
                keys: own.keys.clone(),
                values: own.values.clone(),
            });
        };
        let leading = base.layer_kv(layer)?;
        let stream = StreamOrDevice::gpu();
        let join =
            |head: &Array, tail: &Array| ops::concatenate_axis_device(&[head, tail], 2, &stream);
        Ok(Qwen3LayerKv {
            keys: join(&leading.keys, &own.keys)?,
            values: join(&leading.values, &own.values)?,
        })
    }
}
