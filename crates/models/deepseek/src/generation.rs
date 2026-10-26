//! Local-only scalar generation from admitted checkpoint receipts.
//!
//! A store root contains `config.json` (native Hugging Face configuration),
//! `inference-config.json` (the pinned execution configuration),
//! `model.safetensors.index.json`, `headers.json`, prefixed saved headers at
//! `headers/<shard>.header.bin`, `v41-engram-inputs.bin`, and `weights/`.
//! `headers.json` uses the range cache manifest: revision, index SHA-256 and
//! per-shard header length/hash and original file length. Whole-tensor receipts
//! and packed rows use [`crate::checkpoint::local_store::V41LocalStoreSource`].
//! Missing bytes fail locally; opening or executing never fetches, migrates,
//! repairs, evicts or writes files. Tokenization, selection and stopping belong
//! to the caller. This facade does not qualify upstream token parity.

use std::{
    num::{NonZeroU32, NonZeroU64, NonZeroUsize},
    path::Path,
    sync::Mutex,
};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{
    TokenId, V41ArtifactInspectionError,
    checkpoint::{
        embedding_rows::V41CachedEmbeddingRows,
        engram_rows::V41CachedEngramRows,
        local_store::V41LocalStoreSource,
        range_cache::{V41CachedRoutedExperts, V41RangeCache, V41RangeCacheError, read_metadata},
    },
    engram::{
        embedding::EngramRowSource,
        inputs::{EngramHashInputs, EngramInputsError, V41_ENGRAM_INPUTS_IDENTITIES},
    },
    moe::RoutedExpertSource,
    reduced::{
        RequestError, RequestSession, StepSources,
        checkpoint_model::{
            V41Backend, V41CheckpointModelError, V41CheckpointWeights, V41Engrams,
            V41InferenceConfig,
        },
    },
};

type LocalCache = V41RangeCache<V41LocalStoreSource>;

/// Admission and execution limits for the local scalar consumer.
///
/// Weight and cache limits are separate from process footprint: request state,
/// Engram definition copies, activations, temporary conversions and allocator
/// overhead still need a caller-owned process budget.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct V41ModelOptions {
    /// Scalar is supported; other backends are refused before metadata/payload reads.
    pub backend: V41Backend,
    /// Total appended positions, at most 32,768.
    pub context_tokens: NonZeroU32,
    /// Maximum tokens per prefill chunk, 2 through 128.
    pub prefill_step_tokens: NonZeroU32,
    /// Retained raw range-cache budget, default 2 GiB.
    pub cache_bytes: NonZeroU64,
    /// Admission cap on retained scalar checkpoint-weight bytes, default 12 GiB.
    pub resident_bytes: NonZeroU64,
}

impl Default for V41ModelOptions {
    fn default() -> Self {
        Self {
            backend: V41Backend::Scalar,
            context_tokens: NonZeroU32::new(32_768).expect("nonzero"),
            prefill_step_tokens: NonZeroU32::new(128).expect("nonzero"),
            cache_bytes: NonZeroU64::new(2 << 30).expect("nonzero"),
            resident_bytes: NonZeroU64::new(12 << 30).expect("nonzero"),
        }
    }
}

impl V41ModelOptions {
    fn validate(&self) -> Result<(), V41GenerateError> {
        if self.backend != V41Backend::Scalar {
            return Err(V41GenerateError::UnsupportedBackend);
        }
        if !(2..=32_768).contains(&self.context_tokens.get()) {
            return Err(V41GenerateError::Options(
                "context_tokens must be 2..=32768",
            ));
        }
        if !(2..=128).contains(&self.prefill_step_tokens.get()) {
            return Err(V41GenerateError::Options(
                "prefill_step_tokens must be 2..=128",
            ));
        }
        Ok(())
    }
}

/// Metadata admitted before reading any checkpoint weight payload.
#[derive(Clone, Debug)]
pub struct V41ModelMetadata {
    vocabulary: usize,
    revision: String,
    resident_weight_bytes: u64,
}

impl V41ModelMetadata {
    /// Number of valid token IDs, including padded vocabulary entries.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocabulary
    }

    /// Checkpoint revision verified against the pinned Engram artifact.
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Conservative retained checkpoint-weight bound, not a process-footprint cap.
    #[must_use]
    pub const fn resident_weight_bytes(&self) -> u64 {
        self.resident_weight_bytes
    }
}

struct AdmittedStore {
    metadata: V41ModelMetadata,
    config: V41InferenceConfig,
    inputs: EngramHashInputs,
    cache: LocalCache,
}

/// Resident scalar weights and a shared, bounded local range cache.
pub struct V41Model {
    weights: V41CheckpointWeights,
    admitted: V41ModelMetadata,
    config: V41InferenceConfig,
    inputs: EngramHashInputs,
    cache: Mutex<LocalCache>,
    options: V41ModelOptions,
}

impl V41Model {
    /// Checks options, all text/draft tensor metadata and pinned Engram inputs.
    /// No tensor or row payload is opened. Callers can validate prompt IDs using
    /// the returned vocabulary before [`Self::open`] loads resident weights.
    ///
    /// # Errors
    /// Returns the same metadata, options and admission errors as [`Self::open`].
    pub fn inspect_store(
        root: &Path,
        options: &V41ModelOptions,
    ) -> Result<V41ModelMetadata, V41GenerateError> {
        Ok(admit_store(root, options)?.metadata)
    }

    /// Admits the documented store layout, then loads non-expert weights and head.
    /// Admission is repeated even after [`Self::inspect_store`]; metadata is not
    /// a lease against external modifications. The source verifies each payload.
    ///
    /// # Errors
    /// Refuses unsupported options/backend, invalid metadata, a resident-weight
    /// bound above the cap, missing/corrupt local payloads, or invalid operators.
    pub fn open(root: &Path, options: V41ModelOptions) -> Result<Self, V41GenerateError> {
        let AdmittedStore {
            metadata,
            config,
            inputs,
            mut cache,
        } = admit_store(root, &options)?;
        let mut weights = V41CheckpointWeights::load(
            &mut cache,
            &config,
            0..config.layers(),
            position_limit(options.context_tokens)?,
        )?;
        weights.load_head(&mut cache)?;
        // Validate assembly before returning a loadable model; no forward runs.
        weights.request_model(V41Engrams::Definitions(
            weights.engram_definitions(&inputs)?,
        ))?;
        Ok(Self {
            weights,
            admitted: metadata,
            config,
            inputs,
            cache: Mutex::new(cache),
            options,
        })
    }

    /// Number of valid token IDs.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.admitted.vocabulary
    }

    /// Constructs fresh request state borrowing this model's resident tensors.
    ///
    /// # Errors
    /// Reports invalid component construction without panicking or running tokens.
    pub fn session(&self) -> Result<V41Session<'_>, V41GenerateError> {
        let model = self
            .weights
            .request_model(V41Engrams::Definitions(
                self.weights.engram_definitions(&self.inputs)?,
            ))?
            .with_max_step_tokens(position_limit(self.options.prefill_step_tokens)?);
        Ok(V41Session {
            model: self,
            request: RequestSession::from_owned(model)?,
            poisoned: false,
            positions: 0,
        })
    }
}

/// One append-only sequence; any append error requires a successful reset.
pub struct V41Session<'m> {
    model: &'m V41Model,
    request: RequestSession<'m>,
    poisoned: bool,
    positions: u32,
}

impl V41Session<'_> {
    /// Appends IDs and returns the last position's logits, without token selection.
    /// Initial input currently needs two tokens: the ratio-two partial-group
    /// schedule cannot start without prior ratio-one keys. Consecutive chunks
    /// contain at most `prefill_step_tokens` positions, including after prefill.
    ///
    /// # Errors
    /// Empty/invalid input, context overflow, missing/corrupt ranges and request
    /// failures poison the entire session. No error triggers a network request.
    pub fn append(&mut self, tokens: &[TokenId]) -> Result<Vec<f32>, V41GenerateError> {
        let logits = self.append_validated(tokens)?;
        self.positions = u32::try_from(self.request.next_start()).map_err(|_| {
            self.poisoned = true;
            V41GenerateError::PositionOverflow
        })?;
        Ok(logits)
    }

    fn append_validated(&mut self, tokens: &[TokenId]) -> Result<Vec<f32>, V41GenerateError> {
        let config = &self.model.config;
        let cache = &self.model.cache;
        let experts: Vec<_> = (0..config.layers())
            .map(|layer| {
                V41CachedRoutedExperts::new(
                    cache,
                    layer,
                    config.width(),
                    config.intermediate_width(),
                )
            })
            .collect();
        let expert_refs: Vec<_> = experts
            .iter()
            .map(|source| Some(source as &dyn RoutedExpertSource))
            .collect();
        let rows: Vec<_> = config
            .engram_layers()
            .iter()
            .map(|&layer| V41CachedEngramRows::new(cache, layer, config.engram_width()))
            .collect();
        let row_refs: Vec<_> = rows
            .iter()
            .map(|source| Some(source as &dyn EngramRowSource))
            .collect();
        let embedding = V41CachedEmbeddingRows::new(cache, config.width());
        append_with_sources(
            &mut self.request,
            &mut self.poisoned,
            tokens,
            self.model.vocab_size(),
            self.model.options.context_tokens.get(),
            StepSources {
                experts: &expert_refs,
                engram_rows: &row_refs,
                embedding_rows: Some(&embedding),
            },
        )
    }

    /// Successfully committed positions; after failure, reset before reuse.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.positions
    }

    /// Whether no positions have committed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reconstructs all request state; shared immutable weights/cache remain usable.
    ///
    /// # Errors
    /// Returns a component construction error; a failed reset leaves the session poisoned.
    pub fn reset(&mut self) -> Result<(), V41GenerateError> {
        self.poisoned = true;
        self.request.restart()?;
        self.positions = 0;
        self.poisoned = false;
        Ok(())
    }
}

// Shared by the facade and fixture-backed lifecycle tests; sources are supplied
// only for this call, while the request retains all sequence state.
pub(crate) fn append_with_sources(
    request: &mut RequestSession<'_>,
    poisoned: &mut bool,
    tokens: &[TokenId],
    vocabulary: usize,
    context_tokens: u32,
    sources: StepSources<'_>,
) -> Result<Vec<f32>, V41GenerateError> {
    if *poisoned {
        return Err(V41GenerateError::Poisoned);
    }
    *poisoned = true;
    if tokens.is_empty() {
        return Err(V41GenerateError::EmptyInput);
    }
    if request.next_start() == 0 && tokens.len() < 2 {
        return Err(V41GenerateError::InitialPromptTooShort);
    }
    let requested = (request.next_start() as u64)
        .checked_add(tokens.len() as u64)
        .ok_or(V41GenerateError::PositionOverflow)?;
    if requested > u64::from(context_tokens) {
        return Err(V41GenerateError::ContextFull {
            context_tokens,
            requested,
        });
    }
    if let Some(&token) = tokens.iter().find(|id| id.get() as usize >= vocabulary) {
        return Err(V41GenerateError::TokenOutOfVocabulary(token));
    }
    let ids: Vec<_> = tokens.iter().map(|id| i64::from(id.get())).collect();
    let output = request.prefill_with_sources(&ids, sources)?;
    let logits = output
        .heads()
        .last()
        .ok_or(V41GenerateError::MissingLogits)?
        .logits()
        .to_vec();
    *poisoned = false;
    Ok(logits)
}

#[derive(Deserialize)]
struct Manifest {
    revision: String,
}

fn position_limit(tokens: NonZeroU32) -> Result<NonZeroUsize, V41GenerateError> {
    usize::try_from(tokens.get())
        .ok()
        .and_then(NonZeroUsize::new)
        .ok_or(V41GenerateError::Options(
            "token count exceeds platform size",
        ))
}

fn admit_store(root: &Path, options: &V41ModelOptions) -> Result<AdmittedStore, V41GenerateError> {
    options.validate()?;
    let native = read_metadata(&root.join("config.json"), 4 << 20)?;
    let native = std::str::from_utf8(&native).map_err(V41GenerateError::Utf8)?;
    let manifest: Manifest =
        serde_json::from_slice(&read_metadata(&root.join("headers.json"), 4 << 20)?)?;
    let identity = V41_ENGRAM_INPUTS_IDENTITIES
        .iter()
        .find(|identity| identity.revision == manifest.revision)
        .ok_or(V41GenerateError::Revision)?;
    let weights = root.join("weights");
    let shards = root.join("shards");
    let source = if shards.is_dir() {
        V41LocalStoreSource::with_shards(weights, manifest.revision.clone(), shards)
    } else {
        V41LocalStoreSource::new(weights, manifest.revision.clone())
    };
    let cache = V41RangeCache::load(
        source,
        &root.join("model.safetensors.index.json"),
        root,
        &manifest.revision,
        options.cache_bytes.get(),
    )?;
    let resident_weight_bytes = cache.admit_native(native)?;
    if resident_weight_bytes > options.resident_bytes.get() {
        return Err(V41GenerateError::ResidentLimit {
            required: resident_weight_bytes,
            limit: options.resident_bytes.get(),
        });
    }
    let config = V41InferenceConfig::from_native(native)?;
    let inference = read_metadata(&root.join("inference-config.json"), 4 << 20)?;
    if format!("{:x}", Sha256::digest(&inference)) != identity.inference_config_sha256 {
        return Err(V41GenerateError::ConfigurationMismatch);
    }
    let pinned = V41InferenceConfig::parse(
        std::str::from_utf8(&inference).map_err(V41GenerateError::Utf8)?,
    )?;
    if config != pinned {
        return Err(V41GenerateError::ConfigurationMismatch);
    }
    let inputs = EngramHashInputs::parse(
        &read_metadata(&root.join("v41-engram-inputs.bin"), 8 << 20)?,
        identity,
    )?;
    let metadata = V41ModelMetadata {
        vocabulary: config.vocabulary(),
        revision: manifest.revision,
        resident_weight_bytes,
    };
    Ok(AdmittedStore {
        metadata,
        config,
        inputs,
        cache,
    })
}

/// Failure to admit, load or advance a local scalar generation session.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41GenerateError {
    /// Only the scalar generation backend is currently admitted.
    #[error("local generation currently supports only the scalar backend")]
    UnsupportedBackend,
    /// Options exceed the current runtime's supported geometry.
    #[error("invalid generation options: {0}")]
    Options(&'static str),
    /// A previous append/reset failed.
    #[error("session is poisoned; reset before reuse")]
    Poisoned,
    /// No tokens were supplied.
    #[error("append requires at least one token")]
    EmptyInput,
    /// The partial ratio-two startup needs a completed initial pair.
    #[error("initial prompt requires at least two tokens")]
    InitialPromptTooShort,
    /// A token is outside the admitted vocabulary.
    #[error("token {0:?} is outside the model vocabulary")]
    TokenOutOfVocabulary(TokenId),
    /// The append would exceed the complete request capacity.
    #[error("requested {requested} positions exceeds context limit {context_tokens}")]
    ContextFull {
        /// Configured position capacity.
        context_tokens: u32,
        /// Total positions including the proposed append.
        requested: u64,
    },
    /// Position arithmetic could not represent the request.
    #[error("request position overflow")]
    PositionOverflow,
    /// The conservative retained-weight bound exceeds admission.
    #[error("resident weights require at most {required} bytes, exceeding limit {limit}")]
    ResidentLimit {
        /// Derived bound.
        required: u64,
        /// Caller limit.
        limit: u64,
    },
    /// Store revision is incompatible with pinned Engram inputs.
    #[error("store revision does not match pinned Engram inputs")]
    Revision,
    /// Native execution fields disagree with the pinned source configuration.
    #[error("native and pinned inference configurations disagree")]
    ConfigurationMismatch,
    /// Metadata was not UTF-8.
    #[error("invalid metadata UTF-8: {0}")]
    Utf8(#[source] std::str::Utf8Error),
    /// Metadata JSON was malformed.
    #[error("invalid store metadata: {0}")]
    Json(#[from] serde_json::Error),
    /// Native tensor metadata failed admission.
    #[error(transparent)]
    Admission(#[from] V41ArtifactInspectionError),
    /// A range or its receipt failed local verification.
    #[error(transparent)]
    Source(#[from] V41RangeCacheError),
    /// Resident weights could not form their model operators.
    #[error(transparent)]
    Model(#[from] V41CheckpointModelError),
    /// Pinned Engram inputs were missing or corrupt.
    #[error(transparent)]
    Engram(#[from] EngramInputsError),
    /// An admitted numerical request failed.
    #[error(transparent)]
    Request(#[from] RequestError),
    /// The request unexpectedly returned no final head.
    #[error("request produced no final logits")]
    MissingLogits,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_options_are_refused_before_any_store_read() {
        let root = Path::new("nonexistent-generation-store");
        let mut options = V41ModelOptions {
            prefill_step_tokens: NonZeroU32::new(129).unwrap(),
            ..V41ModelOptions::default()
        };
        assert!(matches!(
            V41Model::inspect_store(root, &options),
            Err(V41GenerateError::Options(_))
        ));
        options.prefill_step_tokens = NonZeroU32::new(128).unwrap();
        options.context_tokens = NonZeroU32::new(32_769).unwrap();
        assert!(matches!(
            V41Model::open(root, options),
            Err(V41GenerateError::Options(_))
        ));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn unsupported_backend_is_refused_before_io_or_device_creation() {
        let options = V41ModelOptions {
            backend: V41Backend::Metal,
            ..V41ModelOptions::default()
        };
        assert!(matches!(
            V41Model::open(Path::new("nonexistent-generation-store"), options),
            Err(V41GenerateError::UnsupportedBackend)
        ));
    }
}
