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
    Qwen3FloatPrecision, Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor, Qwen3LayerKv,
    Qwen3ResidentChatPlan, as_i32,
};

/// Leading bytes of an encoded snapshot.
const MAGIC: &[u8; 8] = b"MXQ3KVS\0";
/// Changed whenever the encoded layout changes, so older bytes are refused.
const FORMAT_VERSION: u32 = 1;
/// Magic, version, token count, layer count, kv heads, head width, dtype.
const HEADER_BYTES: usize = 8 + 4 + 8 + 4 + 4 + 4 + 1;

const fn dtype_code(precision: Qwen3FloatPrecision) -> u8 {
    match precision {
        Qwen3FloatPrecision::BFloat16 => 1,
        Qwen3FloatPrecision::Float16 => 2,
        Qwen3FloatPrecision::Float32 => 3,
    }
}

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

    /// The snapshot as bytes for storage outside this process: a header
    /// (format version, token count, layer count, kv heads, head width and
    /// element dtype), then each layer's keys and values as their stored
    /// element bits, little-endian. `decode` rebuilds the same bits.
    /// A snapshot that extends a base encodes the joined positions, so the
    /// bytes stand alone. The bytes carry no checkpoint identity; whoever
    /// stores them keys them by one.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3ForwardError::CacheInconsistent`] when a layer's arrays
    /// differ from the snapshot's shape or dtype, and MLX errors from reading
    /// them.
    pub fn encode(&self) -> Result<Vec<u8>, Qwen3ForwardError> {
        let layers = (0..self.layers.len())
            .map(|layer| self.layer_kv(layer))
            .collect::<Result<Vec<_>, _>>()?;
        let first = layers.first().ok_or(Qwen3ForwardError::CacheInconsistent)?;
        let shape = first.keys.shape().to_vec();
        let [1, kv_heads, rows, head_dim] = shape[..] else {
            return Err(Qwen3ForwardError::CacheInconsistent);
        };
        if usize::try_from(rows).ok() != Some(self.tokens) {
            return Err(Qwen3ForwardError::CacheInconsistent);
        }
        let precision = self.plan.kv_precision();
        let narrow =
            |value: usize| u32::try_from(value).map_err(|_| Qwen3ForwardError::ShapeOverflow);
        let payload: usize = layers
            .iter()
            .map(|layer| layer.keys.nbytes() + layer.values.nbytes())
            .sum();
        let mut bytes = Vec::with_capacity(HEADER_BYTES + payload);
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(
            &u64::try_from(self.tokens)
                .map_err(|_| Qwen3ForwardError::ShapeOverflow)?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(&narrow(layers.len())?.to_le_bytes());
        for dim in [kv_heads, head_dim] {
            let dim = u32::try_from(dim).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.push(dtype_code(precision));
        for layer in &layers {
            for array in [&layer.keys, &layer.values] {
                if array.shape() != shape.as_slice() || array.dtype() != precision.dtype() {
                    return Err(Qwen3ForwardError::CacheInconsistent);
                }
                // A layer's arrays need not be row-major, and only a
                // row-major array can be read as a slice.
                let array = array.contiguous()?;
                if precision.bytes_per_element() == 4 {
                    let bits = array.view::<u32>()?;
                    bits.eval()?;
                    for word in bits.as_slice::<u32>() {
                        bytes.extend_from_slice(&word.to_le_bytes());
                    }
                } else {
                    let bits = array.view::<u16>()?;
                    bits.eval()?;
                    for word in bits.as_slice::<u16>() {
                        bytes.extend_from_slice(&word.to_le_bytes());
                    }
                }
            }
        }
        Ok(bytes)
    }

    /// Rebuilds a snapshot from [`Self::encode`]'s bytes for a checkpoint
    /// with `config`, under the current resident `plan` and weights
    /// `binding`. Checks the format version, that the layer count, kv heads,
    /// head width and dtype match `config` and `plan`, that the token count
    /// fits the plan's context, and that the payload is exactly as long as
    /// the header says. It cannot check which weights produced the bytes.
    pub(crate) fn decode(
        config: &Qwen3ForwardConfig,
        plan: Qwen3ResidentChatPlan,
        binding: u64,
        bytes: &[u8],
    ) -> Result<Self, Qwen3ForwardError> {
        let format = Qwen3ForwardError::KvSnapshotFormat;
        let header = bytes
            .get(..HEADER_BYTES)
            .ok_or(format("shorter than its header"))?;
        if &header[..8] != MAGIC {
            return Err(format("not a Qwen3 K/V snapshot"));
        }
        let word = |at: usize| u32::from_le_bytes(header[at..at + 4].try_into().expect("4 bytes"));
        if word(8) != FORMAT_VERSION {
            return Err(format("another format version"));
        }
        let tokens = u64::from_le_bytes(header[12..20].try_into().expect("8 bytes"));
        let tokens = usize::try_from(tokens).map_err(|_| format("token count overflows"))?;
        let precision = plan.kv_precision();
        let layers = usize::try_from(word(20)).map_err(|_| format("layer count overflows"))?;
        let kv_heads = usize::try_from(word(24)).map_err(|_| format("kv heads overflow"))?;
        let head_dim = usize::try_from(word(28)).map_err(|_| format("head width overflows"))?;
        if layers != config.hidden_layers
            || kv_heads != config.key_value_heads
            || head_dim != config.head_dim
        {
            return Err(format("another layer count, kv heads or head width"));
        }
        if header[32] != dtype_code(precision) {
            return Err(format("another K/V dtype than the resident plan"));
        }
        if tokens == 0 || tokens > plan.maximum_context_tokens() {
            return Err(format("token count outside the resident context"));
        }
        let elements = kv_heads * tokens * head_dim;
        let width = usize::try_from(precision.bytes_per_element()).expect("2 or 4");
        let array_bytes = elements * width;
        if bytes.len() != HEADER_BYTES + layers * 2 * array_bytes {
            return Err(format("payload length does not match its header"));
        }
        let shape = [1, as_i32(kv_heads)?, as_i32(tokens)?, as_i32(head_dim)?];
        let mut payload = bytes[HEADER_BYTES..].chunks_exact(array_bytes);
        let mut next = || -> Result<Array, Qwen3ForwardError> {
            let chunk = payload.next().ok_or(format("payload ended early"))?;
            let bits = if width == 4 {
                let words = chunk
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().expect("4 bytes")))
                    .collect::<Vec<_>>();
                Array::from_slice(&words, &shape)
            } else {
                let words = chunk
                    .chunks_exact(2)
                    .map(|word| u16::from_le_bytes(word.try_into().expect("2 bytes")))
                    .collect::<Vec<_>>();
                Array::from_slice(&words, &shape)
            };
            let array = bits.view_dtype(precision.dtype())?;
            array.eval()?;
            Ok(array)
        };
        let layers = (0..layers)
            .map(|_| {
                Ok(Qwen3LayerKv {
                    keys: next()?,
                    values: next()?,
                })
            })
            .collect::<Result<Vec<_>, Qwen3ForwardError>>()?;
        Ok(Self {
            layers,
            base: None,
            tokens,
            plan,
            binding,
        })
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
