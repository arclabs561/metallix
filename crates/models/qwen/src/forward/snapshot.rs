//! Detached resident K/V prefixes that outlive one executor.
//!
//! A snapshot holds compact copies of the first `tokens` positions of every
//! layer's K/V. It carries the resident plan it was taken under and an opaque
//! binding chosen by the weights owner, so a later restore can refuse a
//! snapshot from another checkpoint load. Fields stay private: the only way to
//! apply one is [`crate::metal::Qwen3MlxWeights::resident_chat_executor_from`].

use std::hash::BuildHasher;

use mlx_rs::{Array, StreamOrDevice};

use super::{
    Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor, Qwen3LayerKv,
    Qwen3ResidentChatPlan, as_i32,
};

/// Compact K/V for a token prefix of one resident-chat sequence.
pub struct Qwen3KvSnapshot {
    layers: Vec<Qwen3LayerKv>,
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

    /// Byte total of the retained K and V arrays, excluding allocator
    /// overhead.
    #[must_use]
    pub fn kv_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.keys.nbytes() + layer.values.nbytes())
            .sum()
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
        let plan = self
            .resident_chat_plan
            .ok_or(Qwen3ForwardError::KvSnapshotUnsupported)?;
        if self.residual_steering.is_some() || tokens == 0 || tokens > self.cached_tokens {
            return Err(Qwen3ForwardError::KvSnapshotUnsupported);
        }
        let stream = StreamOrDevice::gpu();
        let rows = as_i32(tokens)?;
        // A gather writes fresh rows. A slice written into a new array still
        // kept the whole stepped storage alive after evaluation, so each
        // snapshot held several times its `kv_bytes`.
        let positions = Array::arange_device::<i32, i32>(0, rows, None, &stream)?;
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
            tokens,
            plan,
            binding,
        })
    }

    /// Starts a resident executor whose cache is `snapshot`'s prefix.
    ///
    /// The caller has already matched the binding and plan. The compact
    /// arrays are shared, not copied: the next append re-tiers them into new
    /// storage, so the snapshot stays unchanged. Until that append the
    /// executor cannot be forked.
    pub(crate) fn from_snapshot(
        config: &'a Qwen3ForwardConfig,
        weights: &'a std::collections::HashMap<String, Array, S>,
        snapshot: &Qwen3KvSnapshot,
    ) -> Self {
        let mut executor = Self::new_for_resident_chat(config, weights, snapshot.plan);
        executor.cache = snapshot
            .layers
            .iter()
            .map(|layer| {
                Some(Qwen3LayerKv {
                    keys: layer.keys.clone(),
                    values: layer.values.clone(),
                })
            })
            .collect();
        executor.cached_tokens = snapshot.tokens;
        executor
    }
}
