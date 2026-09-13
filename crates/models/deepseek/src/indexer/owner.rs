//! Atomic ratio-one compressor-to-index-key ownership.
//!
//! V4.1's captured owner layer uses a ratio-one compressor.  This narrow,
//! request-local adapter joins that completed compressor output to prepared
//! index keys and their cache without making the cache, selection, or an
//! arbitrary compression scheduler generic.  A call either commits both the
//! compressor's stream progress and its prepared key prefix, or commits neither.

use std::num::NonZeroUsize;

use thiserror::Error;

use crate::{
    RotaryFrequency,
    compressor::{CompressorError, CompressorInput, CompressorState},
    precision::{Bf16LinearError, MAX_BF16_LINEAR_ELEMENTS, bf16_linear_reference},
};

use super::{
    cache::{CompressedKvState, IndexKeyPublicationId, IndexKeyState, IndexKeyStateError},
    compressed_kv::{
        CompressedKvDiagnostic, CompressedKvError, CompressedKvLayout, CompressedKvLayoutError,
        prepare_compressed_kv,
    },
    key::{IndexKeyDiagnostic, IndexKeyError, IndexKeyLayout, IndexKeyWeights, prepare_index_keys},
};

const MAX_RATIO_ONE_OWNER_WORK: usize = 1 << 24;

/// Borrowed BF16 parameters for one ratio-one owner call.
///
/// `wkv` has source layout `[latent_dimension, input_dimension]`; key weights
/// retain their own typed source layout in [`IndexKeyWeights`].
#[derive(Clone, Copy, Debug)]
pub struct RatioOneOwnerWeights<'a> {
    wkv: &'a [u16],
    key: IndexKeyWeights<'a>,
}

impl<'a> RatioOneOwnerWeights<'a> {
    /// Borrows the owner `wkv` projection and index-key weights.
    #[must_use]
    pub const fn new(wkv: &'a [u16], key: IndexKeyWeights<'a>) -> Self {
        Self { wkv, key }
    }
}

/// One ratio-one owner-layer call before its key cache publication.
///
/// `input` is BF16 owner-layer attention input in
/// `[batch, token_position, input_dimension]` order.  `token_start` is also
/// the compressed-key offset because this adapter is deliberately ratio one.
/// `frequencies` contains one rotary frequency per `[token_position, rope_pair]`
/// at the first token of each completed group (there is one such token here).
#[derive(Clone, Copy, Debug)]
pub struct RatioOneOwnerCall<'a> {
    publication: IndexKeyPublicationId,
    token_start: usize,
    positions: NonZeroUsize,
    input: &'a [u16],
    frequencies: &'a [RotaryFrequency],
    weights: RatioOneOwnerWeights<'a>,
}

impl<'a> RatioOneOwnerCall<'a> {
    /// Groups one call's source identity, input, frequencies, and borrowed weights.
    #[must_use]
    pub const fn new(
        publication: IndexKeyPublicationId,
        token_start: usize,
        positions: NonZeroUsize,
        input: &'a [u16],
        frequencies: &'a [RotaryFrequency],
        weights: RatioOneOwnerWeights<'a>,
    ) -> Self {
        Self {
            publication,
            token_start,
            positions,
            input,
            frequencies,
            weights,
        }
    }
}

/// Precision-staged output from one committed ratio-one owner call.
///
/// The cache itself remains borrowed through [`RatioOneIndexKeyOwner::prefix`];
/// `latent` and `keys` expose the source-visible values that led to that append.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct RatioOneOwnerDiagnostic {
    /// BF16 owner `wkv` output before compressor RMS normalization.
    pub projected: Vec<u16>,
    /// BF16 compressor output before index-key projection.
    pub latent: Vec<u16>,
    /// Source-visible index-key precision stages.
    pub keys: IndexKeyDiagnostic,
}

/// Rejected ratio-one owner construction, call, or reset.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RatioOneIndexKeyOwnerError {
    /// The bounded ratio-one compressor could not be constructed or staged.
    #[error("ratio-one index-key compressor failed: {0}")]
    Compressor(#[from] CompressorError),
    /// The bounded BF16 owner `wkv` projection rejected its operands.
    #[error("ratio-one owner projection failed: {0}")]
    Projection(#[from] Bf16LinearError),
    /// The raw owner input does not match this call's exact `[batch, position, input_dimension]` shape.
    #[error("ratio-one owner input length is {actual}, expected {expected}")]
    InputLength { actual: usize, expected: usize },
    /// Index-key staging rejected the completed compressor output.
    #[error("ratio-one index-key preparation failed: {0}")]
    Key(#[from] IndexKeyError),
    /// The owner-key cache rejected a source identity, ordering, or prepared prefix.
    #[error("ratio-one index-key cache failed: {0}")]
    Cache(#[from] IndexKeyStateError),
    /// A ratio-one compressor unexpectedly did not produce a completed latent.
    #[error("ratio-one compressor returned no completed latent")]
    MissingLatent,
    /// An owner projection dimension product overflowed before allocation.
    #[error("ratio-one owner projection shape arithmetic overflowed for {field}")]
    ProjectionShapeOverflow { field: &'static str },
    /// An owner projection buffer exceeds the explicit bounded reference limit.
    #[error("ratio-one owner projection {field} has {elements} elements, maximum is {maximum}")]
    ProjectionElementLimit {
        field: &'static str,
        elements: usize,
        maximum: usize,
    },
    /// An owner projection would exceed the scalar multiply-accumulate budget.
    #[error("ratio-one owner projection work {terms} exceeds {maximum} scalar terms")]
    ProjectionWorkloadTooLarge { terms: usize, maximum: usize },
    /// The bounded owner-projection output could not be reserved.
    #[error("could not allocate {elements} BF16 ratio-one owner projection elements")]
    ProjectionAllocationFailed { elements: usize },
}

/// Request-local owner for V4.1's qualified ratio-one compressor/key/cache path.
///
/// Construction fixes compression ratio to one.  The adapter owns its `wkv`
/// projection staging, but does not contain candidate construction, index scoring, selection, an
/// attention cache, or a generic cross-layer scheduler.  [`forward`](Self::forward)
/// stages a cloned compressor, then key preparation, then the cache append; only
/// after every fallible step succeeds does it publish the staged compressor.
#[derive(Clone, Debug)]
pub struct RatioOneIndexKeyOwner {
    compressor: CompressorState,
    pristine_compressor: CompressorState,
    keys: IndexKeyState,
    layout: IndexKeyLayout,
    input_dimension: NonZeroUsize,
}

/// Fully prepared but not yet committed ratio-one owner progress.
///
/// Kept private so only the two owner adapters can decide which coupled cache
/// publications commit with this compressor state.
struct StagedRatioOneOwner {
    compressor: CompressorState,
    diagnostic: RatioOneOwnerDiagnostic,
}

/// Diagnostics from one atomically committed ratio-one owner publication.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct RatioOneCompressedOwnerDiagnostic {
    /// Owner projection, compressor, and index-key stages.
    pub owner: RatioOneOwnerDiagnostic,
    /// Compressed KV rotary and FP4 stages from the same compressor latent.
    pub compressed_kv: CompressedKvDiagnostic,
}

/// Rejected combined ratio-one owner construction, call, or reset.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RatioOneCompressedOwnerError {
    /// Ratio-one projection, compressor, or index-key staging failed.
    #[error("combined ratio-one key owner failed: {0}")]
    Owner(#[from] RatioOneIndexKeyOwnerError),
    /// The source-shaped compressed-KV layout could not be derived.
    #[error("combined ratio-one compressed-KV layout failed: {0}")]
    CompressedKvLayout(#[from] CompressedKvLayoutError),
    /// Compressed-KV precision staging failed.
    #[error("combined ratio-one compressed-KV preparation failed: {0}")]
    CompressedKv(#[from] CompressedKvError),
    /// Either coupled owner-prefix cache rejected the publication.
    #[error("combined ratio-one owner-prefix cache failed: {0}")]
    Cache(#[from] IndexKeyStateError),
}

/// Request-local owner for V4.1's coupled ratio-one key and compressed-KV prefixes.
///
/// This is intentionally a source-specific composition, not a generic cache
/// transaction framework: KV preparation uses the producer's original latent
/// width and rotary layout. Both prefixes share batch, position, source, epoch,
/// and call identity; their stored feature widths can differ.
/// Each call validates both bounded cache appends before either becomes visible.
#[derive(Clone, Debug)]
pub struct RatioOneCompressedOwner {
    key_owner: RatioOneIndexKeyOwner,
    compressed_kv: CompressedKvState,
    compressed_kv_layout: CompressedKvLayout,
}

impl RatioOneCompressedOwner {
    /// Creates coupled ratio-one key and compressed-KV owners.
    ///
    /// The compressed-KV layout is derived and validated from `key_layout`
    /// before either cache allocates, keeping this narrow adapter tied to the
    /// source's shared compressor latent rather than accepting arbitrary cache
    /// layouts that might later diverge.
    pub fn new(
        key_layout: IndexKeyLayout,
        input_dimension: NonZeroUsize,
        capacity: NonZeroUsize,
        source_layer: u16,
        compressor_norm: &[u16],
        compressor_epsilon: f32,
    ) -> Result<Self, RatioOneCompressedOwnerError> {
        let compressed_kv_layout = CompressedKvLayout::new(
            key_layout.batches(),
            key_layout.latent_dimension(),
            key_layout.rope_pairs(),
        )?;
        let key_owner = RatioOneIndexKeyOwner::new(
            key_layout,
            input_dimension,
            capacity,
            source_layer,
            compressor_norm,
            compressor_epsilon,
        )?;
        let compressed_kv = CompressedKvState::new(
            key_layout.batches(),
            key_layout.latent_dimension(),
            capacity,
            source_layer,
        )?;
        Ok(Self {
            key_owner,
            compressed_kv,
            compressed_kv_layout,
        })
    }

    /// Stages and atomically publishes source-coupled index keys and compressed KV.
    ///
    /// The input call and its source identity are shared verbatim by both
    /// cache publications. A rejected key, compressed-KV, or cache stage leaves
    /// both prefixes and the compressor position unchanged, so the same call ID
    /// can be retried. As with [`RatioOneIndexKeyOwner`], starting a new prefill
    /// at zero requires [`reset`](Self::reset) after a successful call.
    pub fn forward(
        &mut self,
        call: RatioOneOwnerCall<'_>,
    ) -> Result<RatioOneCompressedOwnerDiagnostic, RatioOneCompressedOwnerError> {
        let staged = self.key_owner.stage(call)?;
        let compressed_kv = prepare_compressed_kv(
            &staged.diagnostic.latent,
            call.frequencies,
            self.compressed_kv_layout,
        )?;

        // Both pending values hold exclusive borrows and have completed every
        // fallible validation. Their commits are bounded copies and metadata
        // assignments only, so neither prefix becomes visible on rejection.
        let key_append = self.key_owner.keys.prepare_append(
            call.publication,
            call.token_start,
            &staged.diagnostic.keys.post_fp4,
        )?;
        let kv_append = self.compressed_kv.prepare_append(
            call.publication,
            call.token_start,
            &compressed_kv.post_fp4,
        )?;
        key_append.commit();
        kv_append.commit();
        self.key_owner.compressor = staged.compressor;
        Ok(RatioOneCompressedOwnerDiagnostic {
            owner: staged.diagnostic,
            compressed_kv,
        })
    }

    /// Begins a checked new epoch for both coupled prefixes and the compressor.
    pub fn reset(&mut self) -> Result<(), RatioOneCompressedOwnerError> {
        // Clone before any checked cache mutation: a failed allocation cannot
        // leave either prefix reset while compressor state remains old.
        let pristine_compressor = self.key_owner.pristine_compressor.clone();
        let key_reset = self.key_owner.keys.prepare_reset()?;
        let kv_reset = self.compressed_kv.prepare_reset()?;
        key_reset.commit();
        kv_reset.commit();
        self.key_owner.compressor = pristine_compressor;
        Ok(())
    }

    /// Borrows one batch's valid prepared index-key prefix.
    pub fn key_prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        self.key_owner.prefix(batch)
    }

    /// Borrows one batch's valid prepared compressed-KV prefix.
    pub fn kv_prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        self.compressed_kv.prefix(batch)
    }

    /// Coupled source layer accepted by both prefixes.
    #[must_use]
    pub const fn source_layer(&self) -> u16 {
        self.key_owner.source_layer()
    }

    /// Coupled request-local epoch.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.key_owner.epoch()
    }

    /// Coupled successful-call ordinal required by the next call.
    #[must_use]
    pub const fn next_call_id(&self) -> u64 {
        self.key_owner.next_call_id()
    }

    /// Coupled number of valid compressed positions in each prefix.
    #[must_use]
    pub const fn valid_positions(&self) -> usize {
        self.key_owner.valid_positions()
    }

    /// Token position required by the ratio-one compressor.
    #[must_use]
    pub const fn next_position(&self) -> usize {
        self.key_owner.next_position()
    }
}

impl RatioOneIndexKeyOwner {
    /// Creates a bounded owner with a fixed source layer and ratio-one compressor.
    ///
    /// The supplied compressor normalization weights must have one BF16 value
    /// per [`IndexKeyLayout`] latent feature.  `key_capacity` is expressed in
    /// compressed positions, which equal token positions only for this ratio-one
    /// adapter.
    pub fn new(
        layout: IndexKeyLayout,
        input_dimension: NonZeroUsize,
        key_capacity: NonZeroUsize,
        source_layer: u16,
        compressor_norm: &[u16],
        compressor_epsilon: f32,
    ) -> Result<Self, RatioOneIndexKeyOwnerError> {
        let compressor = CompressorState::new(
            layout.batches().get(),
            layout.latent_dimension().get(),
            1,
            compressor_norm,
            compressor_epsilon,
        )?;
        let keys = IndexKeyState::new(
            layout.batches(),
            layout.key_dimension(),
            key_capacity,
            source_layer,
        )?;
        Ok(Self {
            pristine_compressor: compressor.clone(),
            compressor,
            keys,
            layout,
            input_dimension,
        })
    }

    /// Stages and atomically commits one ratio-one owner-layer publication.
    ///
    /// The caller must call [`reset`](Self::reset) before a new prefill at token
    /// zero.  In particular, `CompressorState` itself accepts a zero start as a
    /// reset, but this adapter refuses it when the key prefix is nonempty so
    /// compressor and cache epochs cannot diverge.  Any rejected call leaves
    /// both stream states unchanged and is safe to retry with the same identity.
    pub fn forward(
        &mut self,
        call: RatioOneOwnerCall<'_>,
    ) -> Result<RatioOneOwnerDiagnostic, RatioOneIndexKeyOwnerError> {
        let staged = self.stage(call)?;
        self.keys.append_prepared(
            call.publication,
            call.token_start,
            &staged.diagnostic.keys.post_fp4,
        )?;
        // All remaining work is infallible ownership transfer. The cache and
        // compressor become visible together without a full cache clone per call.
        self.compressor = staged.compressor;
        Ok(staged.diagnostic)
    }

    fn stage(
        &self,
        call: RatioOneOwnerCall<'_>,
    ) -> Result<StagedRatioOneOwner, RatioOneIndexKeyOwnerError> {
        let projected_elements = self.validate_projection_shape(call)?;
        let mut projected = Vec::new();
        projected
            .try_reserve_exact(projected_elements)
            .map_err(|_| RatioOneIndexKeyOwnerError::ProjectionAllocationFailed {
                elements: projected_elements,
            })?;
        projected.resize(projected_elements, 0);
        let rows = self
            .layout
            .batches()
            .get()
            .checked_mul(call.positions.get())
            .ok_or(RatioOneIndexKeyOwnerError::ProjectionShapeOverflow { field: "rows" })?;
        bf16_linear_reference(
            call.input,
            call.weights.wkv,
            rows,
            self.input_dimension.get(),
            self.layout.latent_dimension().get(),
            &mut projected,
        )?;
        let mut staged_compressor = self.compressor.clone();
        let latent = staged_compressor
            .forward(
                CompressorInput::ProjectedBf16(&projected),
                call.positions.get(),
                call.token_start,
            )?
            .ok_or(RatioOneIndexKeyOwnerError::MissingLatent)?;
        let prepared =
            prepare_index_keys(&latent, call.frequencies, call.weights.key, self.layout)?;
        Ok(StagedRatioOneOwner {
            compressor: staged_compressor,
            diagnostic: RatioOneOwnerDiagnostic {
                projected,
                latent,
                keys: prepared,
            },
        })
    }

    /// Begins a checked new request epoch and restores the pristine compressor.
    ///
    /// The key cache checks its epoch counter before mutation.  Only after that
    /// checked step succeeds is the small ratio-one compressor restored; no full
    /// key cache is cloned.  Existing borrowed prefixes must not be retained
    /// across this mutable call.
    pub fn reset(&mut self) -> Result<(), RatioOneIndexKeyOwnerError> {
        // `Vec::clone` may need to reserve the small compressor's fixed norm
        // buffer. Do it before the cache's checked mutation so an allocation
        // failure cannot leave the two owners at different epochs.
        let pristine_compressor = self.pristine_compressor.clone();
        self.keys.reset()?;
        self.compressor = pristine_compressor;
        Ok(())
    }

    /// Borrows one batch's exact valid prepared-key prefix.
    pub fn prefix(&self, batch: usize) -> Result<&[u16], IndexKeyStateError> {
        self.keys.prefix(batch)
    }

    /// Fixed source layer accepted by this owner's cache.
    #[must_use]
    pub const fn source_layer(&self) -> u16 {
        self.keys.expected_source_layer()
    }

    /// Current request-local key-cache epoch.
    #[must_use]
    pub const fn epoch(&self) -> u64 {
        self.keys.epoch()
    }

    /// Source call ordinal required by the next successful owner call.
    #[must_use]
    pub const fn next_call_id(&self) -> u64 {
        self.keys.next_call_id()
    }

    /// Number of valid compressed keys per batch.
    #[must_use]
    pub const fn valid_positions(&self) -> usize {
        self.keys.valid_positions()
    }

    /// Token position required by the staged ratio-one compressor.
    #[must_use]
    pub const fn next_position(&self) -> usize {
        self.compressor.next_position()
    }

    fn validate_projection_shape(
        &self,
        call: RatioOneOwnerCall<'_>,
    ) -> Result<usize, RatioOneIndexKeyOwnerError> {
        let rows = checked_product(self.layout.batches().get(), call.positions.get(), "rows")?;
        let input_elements = checked_product(rows, self.input_dimension.get(), "input")?;
        if call.input.len() != input_elements {
            return Err(RatioOneIndexKeyOwnerError::InputLength {
                actual: call.input.len(),
                expected: input_elements,
            });
        }
        let latent_elements = checked_product(
            rows,
            self.layout.latent_dimension().get(),
            "projection output",
        )?;
        let weight_elements = checked_product(
            self.layout.latent_dimension().get(),
            self.input_dimension.get(),
            "projection weight",
        )?;
        let terms = checked_product(
            latent_elements,
            self.input_dimension.get(),
            "projection work",
        )?;
        for (field, elements) in [
            ("input", input_elements),
            ("projection output", latent_elements),
            ("projection weight", weight_elements),
        ] {
            if elements > MAX_BF16_LINEAR_ELEMENTS {
                return Err(RatioOneIndexKeyOwnerError::ProjectionElementLimit {
                    field,
                    elements,
                    maximum: MAX_BF16_LINEAR_ELEMENTS,
                });
            }
        }
        if terms > MAX_RATIO_ONE_OWNER_WORK {
            return Err(RatioOneIndexKeyOwnerError::ProjectionWorkloadTooLarge {
                terms,
                maximum: MAX_RATIO_ONE_OWNER_WORK,
            });
        }
        Ok(latent_elements)
    }
}

fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, RatioOneIndexKeyOwnerError> {
    left.checked_mul(right)
        .ok_or(RatioOneIndexKeyOwnerError::ProjectionShapeOverflow { field })
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::{
        RatioOneCompressedOwner, RatioOneCompressedOwnerError, RatioOneIndexKeyOwner,
        RatioOneIndexKeyOwnerError, RatioOneOwnerCall, RatioOneOwnerWeights,
    };
    use crate::{
        RotaryFrequency,
        indexer::{
            cache::{IndexKeyPublicationId, IndexKeyStateError},
            key::{IndexKeyLayout, IndexKeyWeights},
        },
    };

    const UNIT_WKV: [u16; 1] = [0x3f80];

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("nonzero test dimension")
    }

    fn layout() -> IndexKeyLayout {
        IndexKeyLayout::new(nz(1), nz(1), nz(32), nz(1), 1.0e-6).expect("test layout")
    }

    fn owner(capacity: usize) -> RatioOneIndexKeyOwner {
        RatioOneIndexKeyOwner::new(layout(), nz(1), nz(capacity), 3, &[0x3f80], 1.0e-6)
            .expect("bounded ratio-one owner")
    }

    fn call<'a>(
        epoch: u64,
        call_id: u64,
        start: usize,
        input: &'a [u16],
        frequencies: &'a [RotaryFrequency],
        wk: &'a [u16],
        norm: &'a [u16],
    ) -> RatioOneOwnerCall<'a> {
        RatioOneOwnerCall::new(
            IndexKeyPublicationId::new(3, epoch, call_id),
            start,
            nz(input.len()),
            input,
            frequencies,
            RatioOneOwnerWeights::new(&UNIT_WKV, IndexKeyWeights::new(wk, norm)),
        )
    }

    fn frequency() -> RotaryFrequency {
        RotaryFrequency::new(1.0, 0.0).expect("finite frequency")
    }

    fn weights() -> (Vec<u16>, Vec<u16>) {
        let mut wk = vec![0_u16; 32];
        wk[0] = 0x3f80;
        (wk, vec![0x3f80; 32])
    }

    #[test]
    fn commits_latent_keys_and_prefix_together() {
        let (wk, norm) = weights();
        let frequencies = [frequency()];
        let mut owner = owner(2);
        let diagnostic = owner
            .forward(call(0, 0, 0, &[0x4000], &frequencies, &wk, &norm))
            .expect("finite owner call");
        assert_eq!(diagnostic.latent, vec![0x3f80]);
        assert_eq!(diagnostic.keys.post_fp4.len(), 32);
        assert_eq!(owner.prefix(0).expect("prefix"), diagnostic.keys.post_fp4);
        assert_eq!((owner.next_position(), owner.next_call_id()), (1, 1));
    }

    #[test]
    fn failed_key_preparation_rolls_back_compressor_and_cache() {
        let (wk, norm) = weights();
        let frequencies = [frequency()];
        let mut owner = owner(2);
        let before = owner.clone();
        let error = owner
            .forward(call(0, 0, 0, &[0x4000], &[], &wk, &norm))
            .expect_err("missing rotary frequency follows staged compressor");
        assert!(matches!(error, RatioOneIndexKeyOwnerError::Key(_)));
        assert_eq!(owner.next_position(), before.next_position());
        assert_eq!(owner.next_call_id(), before.next_call_id());
        assert_eq!(
            owner.prefix(0).expect("prefix"),
            before.prefix(0).expect("prefix")
        );
        owner
            .forward(call(0, 0, 0, &[0x4000], &frequencies, &wk, &norm))
            .expect("same call retries after failed key staging");
    }

    #[test]
    fn cache_capacity_and_publication_errors_roll_back_compressor() {
        let (wk, norm) = weights();
        let frequencies = [frequency(), frequency()];
        let mut owner = owner(1);
        let before = owner.clone();
        let error = owner
            .forward(call(0, 0, 0, &[0x3f80, 0x3f80], &frequencies, &wk, &norm))
            .expect_err("two keys exceed one-position cache");
        assert!(matches!(
            error,
            RatioOneIndexKeyOwnerError::Cache(IndexKeyStateError::CapacityExceeded { .. })
        ));
        assert_eq!(owner.next_position(), before.next_position());
        assert_eq!(owner.next_call_id(), before.next_call_id());

        let error = owner
            .forward(RatioOneOwnerCall::new(
                IndexKeyPublicationId::new(4, 0, 0),
                0,
                nz(1),
                &[0x3f80],
                &frequencies[..1],
                RatioOneOwnerWeights::new(&UNIT_WKV, IndexKeyWeights::new(&wk, &norm)),
            ))
            .expect_err("wrong source leaves staged compressor private");
        assert!(matches!(
            error,
            RatioOneIndexKeyOwnerError::Cache(IndexKeyStateError::UnexpectedSourceLayer {
                actual: 4,
                expected: 3
            })
        ));
        assert_eq!(owner.next_position(), 0);
        assert_eq!(owner.next_call_id(), 0);
    }

    #[test]
    fn reset_checks_epoch_then_restores_ratio_one_stream() {
        let (wk, norm) = weights();
        let frequencies = [frequency()];
        let mut owner = owner(2);
        owner
            .forward(call(0, 0, 0, &[0x3f80], &frequencies, &wk, &norm))
            .expect("initial call");
        let prefix = owner.prefix(0).expect("initial prefix").to_vec();
        assert!(matches!(
            owner.forward(call(0, 1, 0, &[0x3f80], &frequencies, &wk, &norm)),
            Err(RatioOneIndexKeyOwnerError::Cache(
                IndexKeyStateError::PositionDiscontinuity { .. }
            ))
        ));
        assert_eq!(owner.next_position(), 1, "rejected reset did not advance");
        assert_eq!(owner.prefix(0).expect("retained prefix"), prefix);
        owner.reset().expect("new epoch");
        assert_eq!(
            (
                owner.epoch(),
                owner.next_call_id(),
                owner.next_position(),
                owner.valid_positions()
            ),
            (1, 0, 0, 0)
        );
        owner
            .forward(call(1, 0, 0, &[0x3f80], &frequencies, &wk, &norm))
            .expect("new epoch begins at zero");
    }

    #[test]
    fn coupled_owner_rolls_back_late_kv_cache_failure_and_retries_same_id() {
        let layout =
            IndexKeyLayout::new(nz(1), nz(32), nz(32), nz(1), 1.0e-6).expect("coupled layout");
        let mut owner =
            RatioOneCompressedOwner::new(layout, nz(1), nz(2), 3, &[0x3f80; 32], 1.0e-6)
                .expect("coupled owner");
        let mut wkv = vec![0_u16; 32];
        wkv[0] = 0x3f80;
        let mut wk = vec![0_u16; 32 * 32];
        wk[0] = 0x3f80;
        let key_norm = vec![0x3f80; 32];
        let weights = RatioOneOwnerWeights::new(&wkv, IndexKeyWeights::new(&wk, &key_norm));
        let frequencies = [frequency()];
        let publication = IndexKeyPublicationId::new(3, 0, 0);

        // A module-private adversarial setup advances only the KV cache. The
        // combined call therefore gets through key staging and key append
        // validation before the KV append rejects its stale ordinal.
        let pristine_kv = owner.compressed_kv.clone();
        owner
            .compressed_kv
            .append_prepared(publication, 0, &[0_u16; 32])
            .expect("test-only divergent KV cache");
        let keys_before = owner.key_owner.keys.clone();
        let kv_before = owner.compressed_kv.clone();
        let error = owner
            .forward(RatioOneOwnerCall::new(
                publication,
                0,
                nz(1),
                &[0x4000],
                &frequencies,
                weights,
            ))
            .expect_err("late KV cache validation fails");
        assert!(matches!(
            error,
            RatioOneCompressedOwnerError::Cache(IndexKeyStateError::UnexpectedCallId { .. })
        ));
        assert_eq!(owner.key_owner.keys, keys_before, "key cache is atomic");
        assert_eq!(owner.compressed_kv, kv_before, "KV cache is atomic");
        assert!(
            owner
                .key_prefix(0)
                .expect("unchanged key prefix")
                .is_empty()
        );
        assert_eq!((owner.next_position(), owner.next_call_id()), (0, 0));

        // Restore only the deliberately corrupted private test fixture. The
        // externally visible owner state was never advanced, so the original
        // source identity can be retried unchanged.
        owner.compressed_kv = pristine_kv;
        let diagnostic = owner
            .forward(RatioOneOwnerCall::new(
                publication,
                0,
                nz(1),
                &[0x4000],
                &frequencies,
                weights,
            ))
            .expect("same ID retry");
        assert_eq!(
            owner.key_prefix(0).expect("key prefix"),
            diagnostic.owner.keys.post_fp4
        );
        assert_eq!(
            owner.kv_prefix(0).expect("KV prefix"),
            diagnostic.compressed_kv.post_fp4
        );
        assert_eq!((owner.next_position(), owner.next_call_id()), (1, 1));

        owner.reset().expect("coupled reset");
        assert_eq!(
            (
                owner.epoch(),
                owner.next_position(),
                owner.next_call_id(),
                owner.valid_positions()
            ),
            (1, 0, 0, 0)
        );
        assert!(owner.key_prefix(0).expect("cleared key prefix").is_empty());
        assert!(owner.kv_prefix(0).expect("cleared KV prefix").is_empty());
    }
}
