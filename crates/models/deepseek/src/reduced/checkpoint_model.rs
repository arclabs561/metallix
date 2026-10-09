//! The 40-layer DeepSeek-V4.1 Flash request model over real checkpoint tensors.
//!
//! [`V41InferenceConfig`] reads the pinned `inference-config.json` and derives
//! each layer's attention role from `compress_ratios`, `kv_source_layers` and
//! `index_source_layers`. [`V41CheckpointWeights`] reads the non-expert tensors
//! for a span of layers through a [`V41RangeCache`] and lends them to
//! [`RequestModel::from_schedule`].
//!
//! Storage: FP8 codes and E8M0 scales stay in the cache's `Arc<[u8]>` payloads,
//! so the cache can evict its own reference without invalidating the model.
//! BF16 and FP32 tensors are converted to owned `Vec<u16>`/`Vec<f32>`, because
//! reinterpreting a byte payload as wider words needs alignment this crate
//! cannot prove without `unsafe`. `wo_a` is dequantized to BF16 as the pinned
//! `convert.py` does, and the FP8 source of it is dropped.
//!
//! Three tables are read per request step from caller sources rather than
//! held here: routed experts (tails are built over an empty sparse table),
//! Engram embedding rows (hundreds of GB), and token-embedding rows (startup
//! is built [`super::StartupDefinition::with_row_source`]). The output head
//! is held as BF16 and computes the last position's logits only.

use std::{collections::BTreeMap, num::NonZeroUsize, ops::Range, sync::Arc};

use serde::Deserialize;
use thiserror::Error;

use crate::{
    RotaryFrequency, RotaryFrequencyParameters,
    attention::layer::{
        CompressedAttentionPublication, Fp8Projection, LayerAttentionLayout, LayerAttentionState,
        LayerAttentionWeights,
    },
    checkpoint::{
        V41StorageDtype,
        range_cache::{V41RangeCache, V41RangeCacheError, V41RangeSource},
    },
    engram::inputs::EngramHashInputs,
    ffn::FfnSublayerReference,
    indexer::{
        key::{IndexKeyLayout, IndexKeyWeights},
        owner::RatioOneOwnerWeights,
        query::{CandidateQueryLayout, CandidateQueryWeights, IndexQueryLayout, IndexQueryWeights},
    },
    moe::{Fp4ExpertWeights, Fp8ExpertWeights, MoEConfig, MoEReference, RoutedExpertSource},
    precision::{decode_e4m3fn, decode_e8m0, f32_to_bf16_rne},
};

use super::{
    AttentionInput, BlockDefinition, BlockTailReference, CandidateProjector, EngramDefinition,
    EngramSessionConfig, EngramSessionWeights, FinalHead, HeadPositions, HeadWeights,
    LayerFourCall, LayerFourConfig, LayerFourDefinition, LayerFourSession, LayerKind, LayerOneCall,
    LayerOneConfig, LayerOneDefinition, LayerOneSession, LayerOneStepOutput, LayerThreeCall,
    LayerThreeConfig, LayerThreeDefinition, LayerThreePublication, LayerThreeSession,
    LayerThreeStepOutput, RatioTwoOwnerLayout, RatioTwoOwnerWeights, RequestError, RequestModel,
    ReusedAttentionDefinition, ScheduledLayer, StartupDefinition,
};

#[cfg(feature = "metal")]
use super::MetalBf16Head;
#[cfg(feature = "metal")]
use crate::precision::{DeviceCounts, DeviceLinears, Fp8Buffers};

/// Largest routed-expert count the static empty table covers.
const MAX_ROUTED_EXPERTS: usize = 384;

/// Lower bound on the Engram gate's |score| before its signed square root,
/// as in the pinned source and the private native harness.
const ENGRAM_GATE_CLAMP: f32 = 1e-6;

// Every tail is built over this empty sparse table: a request step reads routed
// experts from its `StepSources::experts` (for example
// `checkpoint::range_cache::V41CachedRoutedExperts`), and a tail run without
// one fails with `MoEError::MissingRoutedExpert` instead of reading weights.
static NO_ROUTED_EXPERTS: [Option<Fp4ExpertWeights<'static>>; MAX_ROUTED_EXPERTS] =
    [None; MAX_ROUTED_EXPERTS];

/// The text-model fields of the pinned `inference-config.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct V41InferenceConfig {
    vocab_size: usize,
    dim: usize,
    moe_inter_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_routed_experts: usize,
    n_shared_experts: usize,
    n_activated_experts: usize,
    score_func: String,
    route_scale: f32,
    swiglu_limit: f32,
    q_lora_rank: usize,
    head_dim: usize,
    rope_head_dim: usize,
    norm_eps: f32,
    o_groups: usize,
    o_lora_rank: usize,
    window_size: usize,
    kv_source_layers: Vec<usize>,
    index_source_layers: Vec<usize>,
    original_seq_len: usize,
    rope_theta: f32,
    rope_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    index_n_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    candidate_source_layer: usize,
    candidate_topk_blocks: usize,
    candidate_block_size: usize,
    hc_mult: usize,
    hc_sinkhorn_iters: usize,
    hc_eps: f32,
    engram_layer_ids: Vec<usize>,
    engram_max_ngram_size: usize,
    engram_n_heads: usize,
    engram_head_dim: usize,
    compress_rope_theta: f32,
    compress_ratios: Vec<usize>,
}

/// The attention role of one model layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum V41LayerRole {
    /// Layer 0: token startup and window-only attention.
    Startup,
    /// Window-only attention after startup.
    WindowOnly,
    /// Ratio-two compressed-KV owner with its own indexer.
    RatioTwoOwner,
    /// Ratio-two attention over `source`'s KV and indices.
    RatioTwoConsumer {
        /// Owning layer.
        source: usize,
    },
    /// Ratio-one compressed-KV owner and candidate-block source.
    RatioOneOwner,
    /// Ratio-one indexer scoring inside `source`'s candidate blocks.
    RatioOneIndexer {
        /// Owning layer.
        source: usize,
    },
    /// Ratio-one attention over `source`'s KV and the latest indices.
    RatioOneConsumer {
        /// Owning layer.
        source: usize,
    },
}

impl V41InferenceConfig {
    /// Parses and checks the fields this native schedule supports.
    ///
    /// # Errors
    ///
    /// Returns [`V41CheckpointModelError::Config`] when the configuration does
    /// not parse or describes a layout this path cannot run.
    pub fn parse(json: &str) -> Result<Self, V41CheckpointModelError> {
        let config: Self = serde_json::from_str(json)
            .map_err(|error| V41CheckpointModelError::Config(error.to_string()))?;
        let fail = |reason: String| Err(V41CheckpointModelError::Config(reason));
        if config.score_func != "sqrtsoftplus" {
            return fail(format!("unsupported score_func {:?}", config.score_func));
        }
        if config.n_shared_experts != 1 {
            return fail("exactly one shared expert is supported".to_owned());
        }
        if config.n_routed_experts > MAX_ROUTED_EXPERTS {
            return fail(format!("{} routed experts", config.n_routed_experts));
        }
        if config.n_layers == 0 || config.compress_ratios.len() < config.n_layers {
            return fail("compress_ratios must cover every layer".to_owned());
        }
        if config.compress_ratios[0] != 0 {
            return fail("layer 0 must be window-only startup".to_owned());
        }
        for layer in 0..config.n_layers {
            config.role(layer)?;
        }
        if config.role(config.candidate_source_layer)? != V41LayerRole::RatioOneOwner {
            return fail("candidate_source_layer must be a ratio-one owner".to_owned());
        }
        Ok(config)
    }

    /// Number of transformer layers.
    #[must_use]
    pub const fn layers(&self) -> usize {
        self.n_layers
    }

    /// Hidden width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.dim
    }

    /// Routed-expert FFN width.
    #[must_use]
    pub const fn intermediate_width(&self) -> usize {
        self.moe_inter_dim
    }

    /// Returns `layer`'s role, derived from the compression schedule.
    ///
    /// # Errors
    ///
    /// Returns [`V41CheckpointModelError::Config`] for a layer past the
    /// configuration, or one whose compression ratio, owner and indexer roles
    /// form no supported combination.
    pub fn role(&self, layer: usize) -> Result<V41LayerRole, V41CheckpointModelError> {
        let fail = |reason: &str| {
            Err(V41CheckpointModelError::Config(format!(
                "layer {layer}: {reason}"
            )))
        };
        let Some(&ratio) = self
            .compress_ratios
            .get(layer)
            .filter(|_| layer < self.n_layers)
        else {
            return fail("outside the model");
        };
        let owner = self.kv_source_layers.contains(&layer);
        let indexed = self.index_source_layers.contains(&layer);
        let source = self
            .kv_source_layers
            .iter()
            .copied()
            .filter(|&source| source <= layer && self.compress_ratios[source] == ratio)
            .max();
        Ok(match (ratio, owner, indexed, source) {
            (0, false, false, _) if layer == 0 => V41LayerRole::Startup,
            (0, false, false, _) => V41LayerRole::WindowOnly,
            (2, true, true, _) => V41LayerRole::RatioTwoOwner,
            (2, false, false, Some(source)) => V41LayerRole::RatioTwoConsumer { source },
            (1, true, true, _) => V41LayerRole::RatioOneOwner,
            (1, false, true, Some(source)) => V41LayerRole::RatioOneIndexer { source },
            (1, false, false, Some(source)) => V41LayerRole::RatioOneConsumer { source },
            _ => return fail("unsupported ratio, owner or indexer combination"),
        })
    }

    fn attention_layout(
        &self,
        layer: usize,
    ) -> Result<LayerAttentionLayout, V41CheckpointModelError> {
        let source = match self.role(layer)? {
            V41LayerRole::Startup | V41LayerRole::WindowOnly => None,
            V41LayerRole::RatioTwoOwner | V41LayerRole::RatioOneOwner => Some(layer),
            V41LayerRole::RatioTwoConsumer { source }
            | V41LayerRole::RatioOneIndexer { source }
            | V41LayerRole::RatioOneConsumer { source } => Some(source),
        };
        let common = (
            NonZeroUsize::MIN,
            nonzero(self.dim)?,
            nonzero(self.n_heads)?,
            nonzero(self.head_dim)?,
            nonzero(self.rope_head_dim / 2)?,
            nonzero(self.q_lora_rank)?,
            nonzero(self.window_size)?,
            nonzero(self.o_groups)?,
            nonzero(self.o_lora_rank)?,
        );
        let scale = f32::from(u16::try_from(self.head_dim).map_err(component)?)
            .sqrt()
            .recip();
        match source {
            None => LayerAttentionLayout::new_window_only(
                common.0,
                common.1,
                common.2,
                common.3,
                common.4,
                common.5,
                common.6,
                common.7,
                common.8,
                self.norm_eps,
                scale,
            ),
            Some(source) => LayerAttentionLayout::new(
                common.0,
                common.1,
                common.2,
                common.3,
                common.4,
                common.5,
                common.6,
                common.7,
                common.8,
                u16::try_from(source).map_err(component)?,
                nonzero(self.compress_ratios[layer])?,
                self.norm_eps,
                scale,
            ),
        }
        .map_err(component)
    }

    fn query_layout(&self) -> Result<CandidateQueryLayout, V41CheckpointModelError> {
        CandidateQueryLayout::new(
            IndexQueryLayout::new(
                NonZeroUsize::MIN,
                nonzero(self.dim)?,
                nonzero(self.q_lora_rank)?,
                nonzero(self.index_n_heads)?,
                nonzero(self.index_head_dim)?,
                nonzero(self.rope_head_dim / 2)?,
            )
            .map_err(component)?,
            self.norm_eps,
        )
        .map_err(component)
    }

    fn hc_rows(&self) -> usize {
        (self.hc_mult + 2) * self.hc_mult
    }
}

/// Compressed-KV publications a teacher-forced walk carries between layers.
#[derive(Debug, Default)]
pub struct V41TeacherState {
    ratio_two: Option<LayerOneStepOutput>,
    ratio_one: Option<LayerThreeStepOutput>,
    indexer_indices: Option<Vec<i32>>,
}

/// One teacher-forced layer's attention and tail results.
#[derive(Debug)]
pub struct V41LayerTrace {
    /// Normalized attention input `[tokens, dim]`.
    pub attention_input: Vec<u16>,
    /// Attention output `[tokens, dim]`.
    pub attention_output: Vec<u16>,
    /// The tail, or why it did not run (for example an unavailable expert).
    pub tail: Result<V41TailTrace, String>,
}

/// One teacher-forced layer's FFN results.
#[derive(Debug)]
pub struct V41TailTrace {
    /// Normalized FFN input `[tokens, dim]`.
    pub ffn_input: Vec<u16>,
    /// `MoE` output `[tokens, dim]`.
    pub ffn_output: Vec<u16>,
    /// Block output stream `[tokens, hc_mult, dim]`.
    pub output: Vec<u16>,
    /// Outgoing pre-mix `[tokens, hc_mult]`.
    pub next_pre: Vec<f32>,
}

/// HOOK(engram): how Engram enters the schedule.
#[derive(Debug)]
pub enum V41Engrams {
    /// One definition per `engram_layer_ids` entry, in order.
    Definitions(Vec<EngramDefinition>),
    /// No Engram runs. This is not the source forward; it is only meaningful
    /// when the caller substitutes Engram outputs, as teacher-forced
    /// comparison against captured layer inputs does.
    Omitted,
}

enum Stored {
    Bytes(Arc<[u8]>),
    Bf16(Vec<u16>),
    F32(Vec<f32>),
}

/// Non-expert checkpoint tensors for a contiguous span of layers.
pub struct V41CheckpointWeights {
    config: V41InferenceConfig,
    layers: Range<usize>,
    tensors: BTreeMap<String, Stored>,
    window_frequencies: Vec<RotaryFrequency>,
    compressed_frequencies: Vec<RotaryFrequency>,
    max_tokens: NonZeroUsize,
    backend: V41Backend,
    #[cfg(feature = "metal")]
    metal_head: Option<MetalBf16Head>,
    #[cfg(feature = "metal")]
    device: Option<Arc<DeviceLinears>>,
}

/// Which implementation runs the request paths that have a Metal variant: the
/// FP8 attention and shared-expert linears, the routed FP4 experts, the Engram
/// WKV projection and the final-head projection. Everything else (norms,
/// routing, attention, the HC mixes) stays on the scalar CPU path either way.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum V41Backend {
    /// The scalar CPU reference.
    #[default]
    Scalar,
    /// Metal kernels where they exist.
    #[cfg(feature = "metal")]
    Metal,
}

impl std::fmt::Debug for V41CheckpointWeights {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("V41CheckpointWeights")
            .field("layers", &self.layers)
            .field("tensors", &self.tensors.len())
            .field("max_tokens", &self.max_tokens)
            .finish_non_exhaustive()
    }
}

impl V41CheckpointWeights {
    /// Reads every non-expert, non-Engram-table tensor of `layers` plus the
    /// final norm, checking each header dtype and shape before reading it.
    /// Rotary tables cover `max_tokens` positions.
    ///
    /// # Errors
    ///
    /// Returns [`V41CheckpointModelError::Config`] for a layer range or token
    /// budget the configuration cannot serve,
    /// [`V41CheckpointModelError::Cache`] when a tensor cannot be read, and
    /// [`V41CheckpointModelError::Tensor`],
    /// [`V41CheckpointModelError::Missing`],
    /// [`V41CheckpointModelError::Component`] when a tensor has the wrong
    /// shape, is absent, or cannot form its component.
    pub fn load<S: V41RangeSource>(
        cache: &mut V41RangeCache<S>,
        config: &V41InferenceConfig,
        layers: Range<usize>,
        max_tokens: NonZeroUsize,
    ) -> Result<Self, V41CheckpointModelError> {
        if layers.is_empty() || layers.end > config.n_layers {
            return Err(V41CheckpointModelError::Config(format!(
                "layer span {layers:?} is outside 0..{}",
                config.n_layers
            )));
        }
        let pairs = NonZeroUsize::new(config.rope_head_dim).ok_or_else(|| {
            V41CheckpointModelError::Config("rope_head_dim must be nonzero".to_owned())
        })?;
        let window_frequencies = RotaryFrequencyParameters::new(
            pairs,
            0,
            config.rope_theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        .and_then(|parameters| parameters.frequencies(0, max_tokens))
        .map_err(component)?;
        let compressed_frequencies = RotaryFrequencyParameters::new(
            pairs,
            config.original_seq_len,
            config.compress_rope_theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        .and_then(|parameters| parameters.frequencies(0, max_tokens))
        .map_err(component)?;
        let mut weights = Self {
            config: config.clone(),
            layers: layers.clone(),
            tensors: BTreeMap::new(),
            window_frequencies,
            compressed_frequencies,
            max_tokens,
            backend: V41Backend::Scalar,
            #[cfg(feature = "metal")]
            metal_head: None,
            #[cfg(feature = "metal")]
            device: None,
        };
        weights.read(cache, "norm.weight", Read::Bf16, &[config.dim])?;
        for layer in layers {
            weights.read_layer(cache, layer)?;
        }
        Ok(weights)
    }

    /// Builds one Engram definition per `engram_layer_ids` entry. Embedding
    /// rows are not loaded: request steps read them from an
    /// [`crate::engram::embedding::EngramRowSource`] (e.g.
    /// `checkpoint::engram_rows::V41CachedEngramRows`).
    ///
    /// # Errors
    ///
    /// Returns [`V41CheckpointModelError::Component`] or
    /// [`V41CheckpointModelError::Missing`] when the Engram tensors or `inputs`
    /// cannot form a definition.
    pub fn engram_definitions(
        &self,
        inputs: &EngramHashInputs,
    ) -> Result<Vec<EngramDefinition>, V41CheckpointModelError> {
        (0..self.config.engram_layer_ids.len())
            .map(|index| {
                let (config, weights) = self.engram_parts(index, inputs)?;
                let definition = EngramDefinition::new(config, weights);
                #[cfg(feature = "metal")]
                let definition = definition.with_metal_wkv(self.backend == V41Backend::Metal);
                Ok(definition)
            })
            .collect()
    }

    fn engram_parts(
        &self,
        index: usize,
        inputs: &EngramHashInputs,
    ) -> Result<(EngramSessionConfig, EngramSessionWeights), V41CheckpointModelError> {
        let c = &self.config;
        if inputs.layer_ids() != c.engram_layer_ids {
            return Err(V41CheckpointModelError::Config(format!(
                "Engram inputs cover layers {:?}, the config {:?}",
                inputs.layer_ids(),
                c.engram_layer_ids
            )));
        }
        let layer = c.engram_layer_ids[index];
        let rows = usize::try_from(inputs.num_embeddings()[index]).map_err(component)?;
        let config = EngramSessionConfig::new(
            inputs.hash_layout().map_err(component)?,
            inputs.token_map().iter().map(|&id| i64::from(id)).collect(),
            index,
            self.max_tokens.get(),
            c.hc_mult,
            c.dim,
            rows,
            c.engram_head_dim,
            c.norm_eps,
            ENGRAM_GATE_CLAMP,
        )
        .map_err(component)?;
        let p = |name: &str| format!("layers.{layer}.engram.{name}");
        let weights = EngramSessionWeights::without_embedding_table(
            self.bytes(&p("wkv.weight"))?.to_vec(),
            self.bytes(&p("wkv.scale"))?.to_vec(),
            self.bf16(&p("q_weight"))?.to_vec(),
            self.bf16(&p("k_weight"))?.to_vec(),
        );
        Ok((config, weights))
    }

    /// Reads the BF16 output head `[vocab_size, dim]` (1.3 GB). The token
    /// embedding is not loaded: startup reads its rows per step.
    ///
    /// # Errors
    ///
    /// Returns [`V41CheckpointModelError::Cache`] when a head tensor cannot be
    /// read, and [`V41CheckpointModelError::Tensor`],
    /// [`V41CheckpointModelError::Component`] when the head cannot be built
    /// from it.
    pub fn load_head<S: V41RangeSource>(
        &mut self,
        cache: &mut V41RangeCache<S>,
    ) -> Result<(), V41CheckpointModelError> {
        let shape = [self.config.vocab_size, self.config.dim];
        self.read(cache, "head.weight", Read::Bf16, &shape)
    }

    /// Selects the backend for [`Self::engram_definitions`] and
    /// [`Self::request_model`]. `V41Backend::Metal` needs the head loaded:
    /// it uploads a 2.6 GB FP32 copy of it to the GPU once. Each FP8 linear's
    /// weights (attention and shared expert) are uploaded on first use and
    /// stay resident while these weights live.
    ///
    /// # Errors
    ///
    /// Returns [`V41CheckpointModelError::Component`] when a loaded weight
    /// cannot be prepared for `backend`.
    pub fn set_backend(&mut self, backend: V41Backend) -> Result<(), V41CheckpointModelError> {
        #[cfg(feature = "metal")]
        {
            (self.metal_head, self.device) = match backend {
                V41Backend::Metal => {
                    let head = MetalBf16Head::new(
                        self.bf16("head.weight")?,
                        self.config.vocab_size,
                        self.config.dim,
                    )
                    .map_err(component)?;
                    (
                        Some(head),
                        Some(Arc::new(DeviceLinears::new(self.fp8_tensors()))),
                    )
                }
                V41Backend::Scalar => (None, None),
            };
        }
        self.backend = backend;
        Ok(())
    }

    /// Every `(codes, scales)` pair stored as `<name>.weight` and `<name>.scale` bytes.
    #[cfg(feature = "metal")]
    fn fp8_tensors(&self) -> Vec<Fp8Buffers> {
        self.tensors
            .iter()
            .filter_map(|(name, stored)| {
                let base = name.strip_suffix(".weight")?;
                match (stored, self.tensors.get(&format!("{base}.scale"))?) {
                    (Stored::Bytes(codes), Stored::Bytes(scales)) => {
                        Some((Arc::clone(codes), Arc::clone(scales)))
                    }
                    _ => None,
                }
            })
            .collect()
    }

    /// What the Metal backend's FP8 linears and routed experts have run so
    /// far, or `None` on the scalar backend.
    #[cfg(feature = "metal")]
    #[must_use]
    pub fn device_counts(&self) -> Option<DeviceCounts> {
        self.device.as_ref().map(|device| device.counts())
    }

    /// Returns the selected backend.
    #[must_use]
    pub const fn backend(&self) -> V41Backend {
        self.backend
    }

    /// Assembles the request model. Requires every layer and the head. Steps
    /// supply routed experts, Engram rows and embedding rows through
    /// [`super::StepSources`].
    ///
    /// # Errors
    ///
    /// Returns [`V41CheckpointModelError::PartialModel`] when the model was
    /// loaded for a layer range rather than all layers,
    /// [`V41CheckpointModelError::Config`] when the configuration cannot form a
    /// request, and [`V41CheckpointModelError::Component`],
    /// [`V41CheckpointModelError::Request`] when a layer cannot be built.
    pub fn request_model(
        &self,
        engrams: V41Engrams,
    ) -> Result<RequestModel<'_>, V41CheckpointModelError> {
        let config = &self.config;
        if self.layers != (0..config.n_layers) {
            return Err(V41CheckpointModelError::PartialModel {
                layers: self.layers.clone(),
            });
        }
        let engrams = match engrams {
            V41Engrams::Definitions(engrams) if engrams.len() == config.engram_layer_ids.len() => {
                Some(engrams)
            }
            V41Engrams::Definitions(engrams) => {
                return Err(V41CheckpointModelError::Config(format!(
                    "{} Engram definitions for {} Engram layers",
                    engrams.len(),
                    config.engram_layer_ids.len()
                )));
            }
            V41Engrams::Omitted => None,
        };
        let startup = StartupDefinition::with_row_source(
            config.vocab_size,
            self.bf16("layers.0.attn_norm.weight")?,
            config.norm_eps,
            config.attention_layout(0)?,
            self.attention_weights(0)?,
            self.tail(0)?,
            &self.window_frequencies,
        );
        let mut layers = Vec::with_capacity(config.n_layers - 1);
        for layer in 1..config.n_layers {
            let mut scheduled = ScheduledLayer::new(self.layer_kind(layer)?, self.block(layer)?);
            if engrams.is_some() {
                if let Some(index) = config.engram_layer_ids.iter().position(|&id| id == layer) {
                    scheduled = scheduled.with_engram(index);
                }
            }
            layers.push(scheduled);
        }
        let head = FinalHead::with_weights(
            self.bf16("norm.weight")?,
            HeadWeights::Bf16(self.bf16("head.weight")?),
            config.vocab_size,
            config.hc_mult,
            config.norm_eps,
        )
        .map_err(component)?;
        // The source computes logits for the last position only.
        let model = RequestModel::from_schedule(
            startup,
            layers,
            engrams.unwrap_or_default(),
            head,
            &self.compressed_frequencies,
            self.max_tokens,
        )?
        .with_head_positions(HeadPositions::Last)
        // The startup and Engram per-step token bounds.
        .with_max_step_tokens(nonzero(128)?);
        #[cfg(feature = "metal")]
        let model = match &self.metal_head {
            Some(head) => model.with_metal_head(head),
            None => model,
        };
        #[cfg(feature = "metal")]
        let model = match &self.device {
            Some(device) => model.with_device_linears(Arc::clone(device)),
            None => model,
        };
        Ok(model)
    }

    /// Runs one layer of a start-zero prefill on a caller-supplied input
    /// stream and pre-mix (teacher forcing), with the same components and
    /// call order as a request step. Layers must run in order within one
    /// `state`, because consumers read the latest owner's publication.
    /// Engram is not applied: a caller forcing layer inputs supplies its output.
    ///
    /// # Errors
    ///
    /// Returns [`V41CheckpointModelError::Config`] for a layer this model did
    /// not load, and [`V41CheckpointModelError::Component`],
    /// [`V41CheckpointModelError::Request`] when a layer stage fails.
    #[allow(
        clippy::too_many_lines,
        reason = "each attention role's call stays visible in one teacher-forced step"
    )]
    pub fn teacher_forced_layer(
        &self,
        layer: usize,
        residual: &[u16],
        pre: &[f32],
        state: &mut V41TeacherState,
        experts: &dyn RoutedExpertSource,
    ) -> Result<V41LayerTrace, V41CheckpointModelError> {
        let c = &self.config;
        let (width, copies) = (c.dim, c.hc_mult);
        let tokens = residual.len() / (copies * width);
        let positions = NonZeroUsize::new(tokens).ok_or_else(|| {
            V41CheckpointModelError::Config("teacher forcing needs at least one token".to_owned())
        })?;
        let span = tokens * (c.rope_head_dim / 2);
        if residual.len() != tokens * copies * width || pre.len() != tokens * copies {
            return Err(V41CheckpointModelError::Config(
                "residual and pre-mix lengths disagree".to_owned(),
            ));
        }
        if span > self.window_frequencies.len() {
            return Err(V41CheckpointModelError::Config(format!(
                "{tokens} tokens exceed the {}-position rotary table",
                self.max_tokens
            )));
        }
        let (window, compressed) = (
            &self.window_frequencies[..span],
            &self.compressed_frequencies[..span],
        );
        let input = self.attention_input(layer)?;
        let mut attention_input = Vec::with_capacity(tokens * width);
        for (row, pre) in residual
            .chunks_exact(copies * width)
            .zip(pre.chunks_exact(copies))
        {
            attention_input.extend_from_slice(
                input
                    .forward(row, pre)
                    .map_err(component)?
                    .normalized_bf16(),
            );
        }
        let weights = self.attention_weights(layer)?;
        let layout = c.attention_layout(layer)?;
        let missing =
            || V41CheckpointModelError::Config(format!("layer {layer}: no preceding owner"));
        let attention_output = match c.role(layer)? {
            V41LayerRole::Startup | V41LayerRole::WindowOnly => {
                LayerAttentionState::new(layout)
                    .forward_window_only(&attention_input, 0, window, weights)
                    .map_err(component)?
                    .final_output
            }
            V41LayerRole::RatioTwoOwner => {
                let output = LayerOneSession::new(
                    self.ratio_two_config(layer)?,
                    self.bf16(&format!("layers.{layer}.attn.compressor.norm.weight"))?,
                )
                .map_err(component)?
                .step(LayerOneCall::new(
                    &attention_input,
                    positions,
                    &self.compressed_frequencies,
                    self.ratio_two_weights(layer)?,
                    self.query_weights(layer)?,
                    c.query_layout()?,
                    weights,
                    None,
                ))
                .map_err(component)?;
                let attended = output.attention().final_output.clone();
                state.ratio_two = Some(output);
                attended
            }
            V41LayerRole::RatioTwoConsumer { .. } => {
                let owner = state.ratio_two.as_ref().ok_or_else(missing)?;
                let publication = owner.publication();
                LayerAttentionState::new(layout)
                    .forward(
                        &attention_input,
                        0,
                        compressed,
                        weights,
                        CompressedAttentionPublication {
                            source_layer: publication.source_layer(),
                            epoch: publication.epoch(),
                            call_id: publication.call_id(),
                            numerical_bf16: owner.kv_prefix(),
                            indices: owner.selected_indices(),
                        },
                    )
                    .map_err(component)?
                    .final_output
            }
            V41LayerRole::RatioOneOwner => {
                let output = LayerThreeSession::new(
                    self.ratio_one_config(layer)?,
                    u16::try_from(layer).map_err(component)?,
                    self.bf16(&format!("layers.{layer}.attn.compressor.norm.weight"))?,
                    c.norm_eps,
                )
                .map_err(component)?
                .step(LayerThreeCall::new(
                    &attention_input,
                    positions,
                    compressed,
                    self.ratio_one_weights(layer)?,
                    self.candidate_projector(layer)?,
                    weights,
                ))
                .map_err(component)?;
                let attended = output.attention().final_output.clone();
                state.ratio_one = Some(output);
                state.indexer_indices = None;
                attended
            }
            V41LayerRole::RatioOneIndexer { .. } => {
                let owner = state.ratio_one.as_ref().ok_or_else(missing)?;
                let output = LayerFourSession::new(self.ratio_one_indexer_config(layer)?)
                    .step(LayerFourCall::new(
                        &attention_input,
                        compressed,
                        self.query_weights(layer)?,
                        weights,
                        LayerThreePublication::new(
                            owner.publication(),
                            owner.key_prefix(),
                            owner.kv_prefix(),
                            owner.candidate().candidates(),
                        ),
                    ))
                    .map_err(component)?;
                state.indexer_indices = Some(output.selection().indices.clone());
                output.attention().final_output.clone()
            }
            V41LayerRole::RatioOneConsumer { .. } => {
                let owner = state.ratio_one.as_ref().ok_or_else(missing)?;
                let publication = owner.publication();
                LayerAttentionState::new(layout)
                    .forward(
                        &attention_input,
                        0,
                        compressed,
                        weights,
                        CompressedAttentionPublication {
                            source_layer: publication.source_layer(),
                            epoch: publication.epoch(),
                            call_id: publication.call_id(),
                            numerical_bf16: owner.kv_prefix(),
                            indices: state
                                .indexer_indices
                                .as_deref()
                                .unwrap_or_else(|| owner.selected_indices()),
                        },
                    )
                    .map_err(component)?
                    .final_output
            }
        };
        let tail = self.tail(layer)?;
        let mut trace = V41TailTrace {
            ffn_input: Vec::with_capacity(tokens * width),
            ffn_output: Vec::with_capacity(tokens * width),
            output: Vec::with_capacity(tokens * copies * width),
            next_pre: Vec::with_capacity(tokens * copies),
        };
        let mut tail_result = Ok(());
        for (row, attended) in residual
            .chunks_exact(copies * width)
            .zip(attention_output.chunks_exact(width))
        {
            match tail.forward_token_with(row, attended, experts) {
                Ok(diagnostic) => {
                    let ffn = diagnostic.ffn();
                    trace.ffn_input.extend_from_slice(ffn.normalized_bf16());
                    trace.ffn_output.extend_from_slice(ffn.moe().output_bf16());
                    trace.output.extend_from_slice(ffn.output_bf16());
                    trace.next_pre.extend_from_slice(ffn.coefficients().pre());
                }
                Err(error) => {
                    tail_result = Err(error.to_string());
                    break;
                }
            }
        }
        Ok(V41LayerTrace {
            attention_input,
            attention_output,
            tail: tail_result.map(|()| trace),
        })
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the per-layer tensor inventory stays one visible list"
    )]
    fn read_layer<S: V41RangeSource>(
        &mut self,
        cache: &mut V41RangeCache<S>,
        layer: usize,
    ) -> Result<(), V41CheckpointModelError> {
        let c = &self.config;
        let (width, heads, head_dim, q_rank) = (c.dim, c.n_heads, c.head_dim, c.q_lora_rank);
        let (index_head_dim, intermediate, hc_rows) =
            (c.index_head_dim, c.moe_inter_dim, c.hc_rows());
        let output_rows = c.o_groups * c.o_lora_rank;
        let role = c.role(layer)?;
        let owner = matches!(
            role,
            V41LayerRole::RatioTwoOwner | V41LayerRole::RatioOneOwner
        );
        let indexed = owner || matches!(role, V41LayerRole::RatioOneIndexer { .. });
        // FP8 projections as (name, rows, columns); each has an E8M0 `.scale`.
        let mut fp8 = vec![
            ("attn.wq_a", q_rank, width),
            ("attn.wq_b", heads * head_dim, q_rank),
            ("attn.wkv", head_dim, width),
            ("attn.wo_b", width, output_rows),
            ("ffn.shared_experts.w1", intermediate, width),
            ("ffn.shared_experts.w2", width, intermediate),
            ("ffn.shared_experts.w3", intermediate, width),
        ];
        let mut other = vec![
            (
                "attn.wo_a.weight",
                Read::WoA,
                vec![output_rows, heads / c.o_groups * head_dim],
            ),
            ("attn.q_norm.weight", Read::Bf16, vec![q_rank]),
            ("attn.kv_norm.weight", Read::Bf16, vec![head_dim]),
            ("attn.attn_sink", Read::F32, vec![heads]),
            ("attn_norm.weight", Read::Bf16, vec![width]),
            ("ffn_norm.weight", Read::Bf16, vec![width]),
            ("hc_attn_fn", Read::F32, vec![hc_rows, c.hc_mult * width]),
            ("hc_attn_scale", Read::F32, vec![3]),
            ("hc_attn_base", Read::F32, vec![hc_rows]),
            ("hc_ffn_fn", Read::F32, vec![hc_rows, c.hc_mult * width]),
            ("hc_ffn_scale", Read::F32, vec![3]),
            ("hc_ffn_base", Read::F32, vec![hc_rows]),
            (
                "ffn.gate.weight",
                Read::Bf16,
                vec![c.n_routed_experts, width],
            ),
            ("ffn.gate.bias", Read::F32, vec![c.n_routed_experts]),
        ];
        if owner {
            other.extend([
                ("attn.compressor.norm.weight", Read::Bf16, vec![head_dim]),
                (
                    "attn.indexer.wk.weight",
                    Read::Bf16,
                    vec![index_head_dim, head_dim],
                ),
                (
                    "attn.indexer.k_norm.weight",
                    Read::Bf16,
                    vec![index_head_dim],
                ),
            ]);
        }
        // The ratio-two owner's compressor takes FP32 projections; ratio-one takes BF16.
        match role {
            V41LayerRole::RatioTwoOwner => other.extend([
                (
                    "attn.compressor.wkv.weight",
                    Read::Bf16AsF32,
                    vec![head_dim, width],
                ),
                (
                    "attn.compressor.wgate.weight",
                    Read::Bf16AsF32,
                    vec![head_dim, width],
                ),
            ]),
            V41LayerRole::RatioOneOwner => {
                other.push((
                    "attn.compressor.wkv.weight",
                    Read::Bf16,
                    vec![head_dim, width],
                ));
            }
            _ => {}
        }
        if c.engram_layer_ids.contains(&layer) {
            let reduction = (c.engram_max_ngram_size - 1) * c.engram_n_heads * c.engram_head_dim;
            fp8.push(("engram.wkv", (c.hc_mult + 1) * width, reduction));
            other.extend([
                ("engram.q_weight", Read::Bf16, vec![c.hc_mult, width]),
                ("engram.k_weight", Read::Bf16, vec![c.hc_mult, width]),
            ]);
        }
        if indexed {
            fp8.push((
                "attn.indexer.wq_b",
                c.index_n_heads * index_head_dim,
                q_rank,
            ));
            other.push((
                "attn.indexer.weights_proj.weight",
                Read::Bf16,
                vec![c.index_n_heads, width],
            ));
        }
        for (name, rows, columns) in fp8 {
            let name = format!("layers.{layer}.{name}");
            self.read(
                cache,
                &format!("{name}.weight"),
                Read::Fp8,
                &[rows, columns],
            )?;
            self.read(
                cache,
                &format!("{name}.scale"),
                Read::Scale,
                &[rows.div_ceil(32), columns / 32],
            )?;
        }
        for (name, read, shape) in other {
            self.read(cache, &format!("layers.{layer}.{name}"), read, &shape)?;
        }
        Ok(())
    }

    fn read<S: V41RangeSource>(
        &mut self,
        cache: &mut V41RangeCache<S>,
        name: &str,
        read: Read,
        shape: &[usize],
    ) -> Result<(), V41CheckpointModelError> {
        let range = cache.tensor_range(name)?;
        let dtype = match read {
            Read::Fp8 | Read::WoA => V41StorageDtype::F8E4M3Fn,
            Read::Scale => V41StorageDtype::F8E8M0Fnu,
            Read::Bf16 | Read::Bf16AsF32 => V41StorageDtype::Bf16,
            Read::F32 => V41StorageDtype::F32,
        };
        let expected: Vec<u64> = shape.iter().map(|&dimension| dimension as u64).collect();
        if range.dtype() != dtype || range.shape() != expected {
            return Err(V41CheckpointModelError::Tensor {
                tensor: name.to_owned(),
                reason: format!(
                    "header has {:?} {:?}, expected {dtype:?} {expected:?}",
                    range.dtype(),
                    range.shape()
                ),
            });
        }
        let bytes = cache.get_tensor(name)?;
        let stored = match read {
            Read::Fp8 | Read::Scale => Stored::Bytes(bytes),
            Read::Bf16 => Stored::Bf16(le_u16(&bytes)),
            Read::Bf16AsF32 => Stored::F32(
                le_u16(&bytes)
                    .into_iter()
                    .map(|bits| f32::from_bits(u32::from(bits) << 16))
                    .collect(),
            ),
            Read::F32 => Stored::F32(
                bytes
                    .chunks_exact(4)
                    .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                    .collect(),
            ),
            Read::WoA => {
                let scale_name = name.replace(".weight", ".scale");
                let [rows, columns] = [shape[0], shape[1]];
                let scale_shape = [rows.div_ceil(32), columns / 32];
                let scale_range = cache.tensor_range(&scale_name)?;
                if scale_range.dtype() != V41StorageDtype::F8E8M0Fnu
                    || scale_range.shape() != scale_shape.map(|d| d as u64)
                {
                    return Err(V41CheckpointModelError::Tensor {
                        tensor: scale_name,
                        reason: "unexpected wo_a scale dtype or shape".to_owned(),
                    });
                }
                let scales = cache.get_tensor(&scale_name)?;
                // The pinned convert.py dequantizes wo_a to BF16 over 32x32 E8M0 blocks.
                Stored::Bf16(
                    bytes
                        .iter()
                        .enumerate()
                        .map(|(index, &code)| {
                            let (row, column) = (index / columns, index % columns);
                            let scale = scales[(row / 32) * (columns / 32) + column / 32];
                            f32_to_bf16_rne(decode_e4m3fn(code) * decode_e8m0(scale))
                        })
                        .collect(),
                )
            }
        };
        self.tensors.insert(name.to_owned(), stored);
        Ok(())
    }

    fn stored(&self, name: &str) -> Result<&Stored, V41CheckpointModelError> {
        self.tensors
            .get(name)
            .ok_or_else(|| V41CheckpointModelError::Missing(name.to_owned()))
    }

    fn bytes(&self, name: &str) -> Result<&[u8], V41CheckpointModelError> {
        match self.stored(name)? {
            Stored::Bytes(bytes) => Ok(bytes),
            _ => Err(V41CheckpointModelError::Missing(name.to_owned())),
        }
    }

    fn bf16(&self, name: &str) -> Result<&[u16], V41CheckpointModelError> {
        match self.stored(name)? {
            Stored::Bf16(values) => Ok(values),
            _ => Err(V41CheckpointModelError::Missing(name.to_owned())),
        }
    }

    fn f32(&self, name: &str) -> Result<&[f32], V41CheckpointModelError> {
        match self.stored(name)? {
            Stored::F32(values) => Ok(values),
            _ => Err(V41CheckpointModelError::Missing(name.to_owned())),
        }
    }

    fn fp8(&self, name: &str) -> Result<Fp8Projection<'_>, V41CheckpointModelError> {
        Ok(Fp8Projection {
            codes: self.bytes(&format!("{name}.weight"))?,
            scales: self.bytes(&format!("{name}.scale"))?,
        })
    }

    fn attention_weights(
        &self,
        layer: usize,
    ) -> Result<LayerAttentionWeights<'_>, V41CheckpointModelError> {
        let p = |name: &str| format!("layers.{layer}.attn.{name}");
        Ok(LayerAttentionWeights {
            wq_a: self.fp8(&p("wq_a"))?,
            q_norm: self.bf16(&p("q_norm.weight"))?,
            wq_b: self.fp8(&p("wq_b"))?,
            wkv: self.fp8(&p("wkv"))?,
            kv_norm: self.bf16(&p("kv_norm.weight"))?,
            attn_sink: self.f32(&p("attn_sink"))?,
            wo_a: self.bf16(&p("wo_a.weight"))?,
            wo_b: self.fp8(&p("wo_b"))?,
        })
    }

    fn attention_input(&self, layer: usize) -> Result<AttentionInput<'_>, V41CheckpointModelError> {
        AttentionInput::new(
            self.bf16(&format!("layers.{layer}.attn_norm.weight"))?,
            self.config.hc_mult,
            self.config.norm_eps,
        )
        .map_err(component)
    }

    fn scale3(&self, name: &str) -> Result<&[f32; 3], V41CheckpointModelError> {
        self.f32(name)?
            .try_into()
            .map_err(|_| V41CheckpointModelError::Missing(name.to_owned()))
    }

    fn tail(&self, layer: usize) -> Result<BlockTailReference<'_>, V41CheckpointModelError> {
        let c = &self.config;
        let p = |name: &str| format!("layers.{layer}.{name}");
        let shared = |projection: &str, kind: &str| {
            self.bytes(&p(&format!("ffn.shared_experts.{projection}.{kind}")))
        };
        let moe = MoEReference::new_sparse(
            MoEConfig::new(
                c.dim,
                c.moe_inter_dim,
                c.swiglu_limit,
                c.n_activated_experts,
                1.0,
                true,
                c.route_scale,
            )
            .map_err(component)?,
            self.bf16(&p("ffn.gate.weight"))?,
            self.f32(&p("ffn.gate.bias"))?,
            &NO_ROUTED_EXPERTS[..c.n_routed_experts],
            Fp8ExpertWeights::new(
                c.dim,
                c.moe_inter_dim,
                shared("w1", "weight")?,
                shared("w1", "scale")?,
                shared("w2", "weight")?,
                shared("w2", "scale")?,
                shared("w3", "weight")?,
                shared("w3", "scale")?,
            )
            .map_err(component)?,
        )
        .map_err(component)?;
        let ffn = FfnSublayerReference::new(
            moe,
            self.bf16(&p("ffn_norm.weight"))?,
            self.f32(&p("hc_ffn_fn"))?,
            self.scale3(&p("hc_ffn_scale"))?,
            self.f32(&p("hc_ffn_base"))?,
            c.hc_mult,
            c.norm_eps,
            c.hc_sinkhorn_iters,
            c.hc_eps,
        )
        .map_err(component)?;
        BlockTailReference::new(
            ffn,
            self.f32(&p("hc_attn_fn"))?,
            self.scale3(&p("hc_attn_scale"))?,
            self.f32(&p("hc_attn_base"))?,
            c.hc_mult,
            c.norm_eps,
            c.hc_sinkhorn_iters,
            c.hc_eps,
        )
        .map_err(component)
    }

    fn block(&self, layer: usize) -> Result<BlockDefinition<'_>, V41CheckpointModelError> {
        Ok(BlockDefinition::new(
            self.attention_input(layer)?,
            self.tail(layer)?,
        ))
    }

    fn query_weights(
        &self,
        layer: usize,
    ) -> Result<CandidateQueryWeights<'_>, V41CheckpointModelError> {
        let p = |name: &str| format!("layers.{layer}.attn.{name}");
        let index = self.fp8(&p("indexer.wq_b"))?;
        Ok(CandidateQueryWeights {
            wq_a: self.fp8(&p("wq_a"))?,
            q_norm: self.bf16(&p("q_norm.weight"))?,
            index: IndexQueryWeights {
                wq_b_codes: index.codes,
                wq_b_scales: index.scales,
                weights_proj: self.bf16(&p("indexer.weights_proj.weight"))?,
            },
        })
    }

    fn index_key(&self, layer: usize) -> Result<IndexKeyWeights<'_>, V41CheckpointModelError> {
        Ok(IndexKeyWeights::new(
            self.bf16(&format!("layers.{layer}.attn.indexer.wk.weight"))?,
            self.bf16(&format!("layers.{layer}.attn.indexer.k_norm.weight"))?,
        ))
    }

    fn ratio_two_owner(
        &self,
        layer: usize,
    ) -> Result<LayerOneDefinition<'_>, V41CheckpointModelError> {
        Ok(LayerOneDefinition::new(
            self.ratio_two_config(layer)?,
            self.bf16(&format!("layers.{layer}.attn.compressor.norm.weight"))?,
            self.ratio_two_weights(layer)?,
            self.query_weights(layer)?,
            self.config.query_layout()?,
            self.attention_weights(layer)?,
        ))
    }

    fn ratio_two_config(&self, layer: usize) -> Result<LayerOneConfig, V41CheckpointModelError> {
        let c = &self.config;
        let owner_layout = RatioTwoOwnerLayout::new(
            NonZeroUsize::MIN,
            nonzero(c.dim)?,
            nonzero(c.head_dim)?,
            nonzero(c.index_head_dim)?,
            nonzero(c.rope_head_dim / 2)?,
            nonzero(self.max_tokens.get().div_ceil(2))?,
            c.norm_eps,
        )
        .map_err(component)?;
        LayerOneConfig::new(
            owner_layout,
            c.attention_layout(layer)?,
            nonzero(c.index_topk)?,
        )
        .map_err(component)
    }

    fn ratio_two_weights(
        &self,
        layer: usize,
    ) -> Result<RatioTwoOwnerWeights<'_>, V41CheckpointModelError> {
        Ok(RatioTwoOwnerWeights::new(
            self.f32(&format!("layers.{layer}.attn.compressor.wkv.weight"))?,
            self.f32(&format!("layers.{layer}.attn.compressor.wgate.weight"))?,
            self.index_key(layer)?,
        ))
    }

    fn ratio_one_owner(
        &self,
        layer: usize,
    ) -> Result<LayerThreeDefinition<'_>, V41CheckpointModelError> {
        Ok(LayerThreeDefinition::new(
            self.ratio_one_config(layer)?,
            self.bf16(&format!("layers.{layer}.attn.compressor.norm.weight"))?,
            self.config.norm_eps,
            self.ratio_one_weights(layer)?,
            self.candidate_projector(layer)?,
            self.attention_weights(layer)?,
        ))
    }

    fn ratio_one_config(&self, layer: usize) -> Result<LayerThreeConfig, V41CheckpointModelError> {
        let c = &self.config;
        let key_layout = IndexKeyLayout::new(
            NonZeroUsize::MIN,
            nonzero(c.head_dim)?,
            nonzero(c.index_head_dim)?,
            nonzero(c.rope_head_dim / 2)?,
            c.norm_eps,
        )
        .map_err(component)?;
        LayerThreeConfig::new(
            key_layout,
            nonzero(c.dim)?,
            self.max_tokens,
            c.attention_layout(layer)?,
            nonzero(c.window_size)?,
            nonzero(c.index_topk)?,
        )
        .map_err(component)
    }

    fn ratio_one_weights(
        &self,
        layer: usize,
    ) -> Result<RatioOneOwnerWeights<'_>, V41CheckpointModelError> {
        Ok(RatioOneOwnerWeights::new(
            self.bf16(&format!("layers.{layer}.attn.compressor.wkv.weight"))?,
            self.index_key(layer)?,
        ))
    }

    fn candidate_projector(
        &self,
        layer: usize,
    ) -> Result<CandidateProjector<'_>, V41CheckpointModelError> {
        let c = &self.config;
        Ok(CandidateProjector::new(
            self.query_weights(layer)?,
            c.query_layout()?,
            nonzero(c.index_head_dim)?,
            nonzero(c.candidate_topk_blocks)?,
            nonzero(c.candidate_block_size)?,
        ))
    }

    fn ratio_one_indexer_config(
        &self,
        layer: usize,
    ) -> Result<LayerFourConfig, V41CheckpointModelError> {
        let c = &self.config;
        LayerFourConfig::new(
            c.query_layout()?,
            c.attention_layout(layer)?,
            nonzero(c.index_topk)?,
        )
        .map_err(component)
    }

    fn layer_kind(&self, layer: usize) -> Result<LayerKind<'_>, V41CheckpointModelError> {
        let c = &self.config;
        let reused = || -> Result<ReusedAttentionDefinition<'_>, V41CheckpointModelError> {
            Ok(ReusedAttentionDefinition::new(
                c.attention_layout(layer)?,
                self.attention_weights(layer)?,
            ))
        };
        Ok(match c.role(layer)? {
            V41LayerRole::Startup => {
                return Err(V41CheckpointModelError::Config(
                    "layer 0 is the startup block, not a scheduled layer".to_owned(),
                ));
            }
            V41LayerRole::WindowOnly => LayerKind::WindowOnly(reused()?),
            V41LayerRole::RatioTwoOwner => LayerKind::RatioTwoOwner(self.ratio_two_owner(layer)?),
            V41LayerRole::RatioTwoConsumer { .. } => LayerKind::RatioTwoConsumer(reused()?),
            V41LayerRole::RatioOneOwner => LayerKind::RatioOneOwner(self.ratio_one_owner(layer)?),
            V41LayerRole::RatioOneIndexer { .. } => {
                LayerKind::RatioOneIndexer(LayerFourDefinition::new(
                    self.ratio_one_indexer_config(layer)?,
                    self.query_weights(layer)?,
                    self.attention_weights(layer)?,
                ))
            }
            V41LayerRole::RatioOneConsumer { .. } => LayerKind::RatioOneConsumer(reused()?),
        })
    }
}

#[derive(Clone, Copy)]
enum Read {
    Fp8,
    Scale,
    Bf16,
    Bf16AsF32,
    F32,
    WoA,
}

fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect()
}

fn nonzero(value: usize) -> Result<NonZeroUsize, V41CheckpointModelError> {
    NonZeroUsize::new(value)
        .ok_or_else(|| V41CheckpointModelError::Config("model geometry must be nonzero".to_owned()))
}

fn component(error: impl std::fmt::Display) -> V41CheckpointModelError {
    V41CheckpointModelError::Component(error.to_string())
}

/// The checkpoint model could not be read or assembled.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum V41CheckpointModelError {
    /// `inference-config.json` is invalid or outside the supported schedule.
    #[error("inference config: {0}")]
    Config(String),
    /// A tensor's header dtype or shape is not what the schedule expects.
    #[error("tensor {tensor:?}: {reason}")]
    Tensor {
        /// Tensor name.
        tensor: String,
        /// Disagreement.
        reason: String,
    },
    /// A tensor was not loaded for this span.
    #[error("tensor {0:?} was not loaded")]
    Missing(String),
    /// A runtime component rejected the checkpoint operands.
    #[error("component rejected checkpoint operands: {0}")]
    Component(String),
    /// A request model needs every layer.
    #[error("request model needs every layer; loaded {layers:?}")]
    PartialModel {
        /// Loaded span.
        layers: Range<usize>,
    },
    /// Checkpoint bytes were unavailable or failed verification.
    #[error(transparent)]
    Cache(#[from] V41RangeCacheError),
    /// The request schedule was rejected.
    #[error(transparent)]
    Request(#[from] RequestError),
}

#[cfg(test)]
mod tests {
    use std::{num::NonZeroUsize, path::Path, sync::Mutex};

    use super::{V41CheckpointWeights, V41InferenceConfig, V41LayerRole, V41TeacherState, le_u16};
    use crate::{
        checkpoint::range_cache::{V41CachedRoutedExperts, V41LocalWeightsSource, V41RangeCache},
        indexer::key::IndexKeyLayout,
        reduced::{
            FinalHead, FinalHeadError, HeadWeights, LayerFourConfig, LayerOneConfig,
            LayerThreeConfig, RatioTwoOwnerLayout,
        },
    };

    /// The schedule fields of the pinned V4.1 Flash `inference-config.json`.
    fn pinned_schedule() -> String {
        let ratios: Vec<usize> = (0..43)
            .map(|layer| match layer {
                0 | 1 | 40.. => 0,
                2..=19 => 2,
                _ => 1,
            })
            .collect();
        serde_json::json!({
            "vocab_size": 129_280, "dim": 5120, "moe_inter_dim": 2304, "n_layers": 40,
            "n_heads": 64, "n_routed_experts": 384, "n_shared_experts": 1,
            "n_activated_experts": 6, "score_func": "sqrtsoftplus", "route_scale": 1.5,
            "swiglu_limit": 10.0, "q_lora_rank": 1280, "head_dim": 512, "rope_head_dim": 64,
            "norm_eps": 1e-20, "o_groups": 8, "o_lora_rank": 1024, "window_size": 128,
            "kv_source_layers": [2, 8, 14, 20], "index_source_layers": [2, 8, 14, 20, 24, 28, 32, 36],
            "original_seq_len": 65536, "rope_theta": 10000, "rope_factor": 16, "beta_fast": 32,
            "beta_slow": 1, "index_n_heads": 32, "index_head_dim": 128, "index_topk": 512,
            "candidate_source_layer": 20, "candidate_topk_blocks": 2048, "candidate_block_size": 8,
            "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "engram_layer_ids": [1, 14], "engram_max_ngram_size": 4, "engram_n_heads": 8,
            "engram_head_dim": 256,
            "compress_rope_theta": 160_000, "compress_ratios": ratios,
        })
        .to_string()
    }

    #[test]
    fn derives_the_forty_layer_schedule_from_compression_and_source_layers() {
        let config = V41InferenceConfig::parse(&pinned_schedule()).expect("pinned schedule");
        let roles: Vec<_> = (0..40)
            .map(|layer| config.role(layer).expect("role"))
            .collect();
        let consumer2 = |source| V41LayerRole::RatioTwoConsumer { source };
        let consumer1 = V41LayerRole::RatioOneConsumer { source: 20 };
        let indexer = V41LayerRole::RatioOneIndexer { source: 20 };
        let mut expected = vec![V41LayerRole::Startup, V41LayerRole::WindowOnly];
        for owner in [2, 8, 14] {
            expected.push(V41LayerRole::RatioTwoOwner);
            expected.extend([consumer2(owner); 5]);
        }
        expected.push(V41LayerRole::RatioOneOwner);
        expected.extend([consumer1; 3]);
        for _ in [24, 28, 32, 36] {
            expected.push(indexer);
            expected.extend([consumer1; 3]);
        }
        assert_eq!(roles, expected);
        assert!(
            config.role(40).is_err(),
            "MTP layers are outside the text schedule"
        );
    }

    #[test]
    fn rejects_schedules_the_native_kinds_cannot_express() {
        let mut json: serde_json::Value =
            serde_json::from_str(&pinned_schedule()).expect("pinned schedule");
        json["kv_source_layers"] = serde_json::json!([8, 14, 20]);
        assert!(
            V41InferenceConfig::parse(&json.to_string()).is_err(),
            "consumer before owner"
        );
        let mut json: serde_json::Value =
            serde_json::from_str(&pinned_schedule()).expect("pinned schedule");
        json["score_func"] = serde_json::json!("softmax");
        assert!(V41InferenceConfig::parse(&json.to_string()).is_err());
    }

    /// V4.1 index keys are `index_head_dim` (128) wide while attention heads
    /// are `head_dim` (512); every compressed-layer config must accept that.
    #[test]
    fn compressed_layer_configs_accept_index_keys_narrower_than_attention_heads() {
        let config = V41InferenceConfig::parse(&pinned_schedule()).expect("pinned schedule");
        let query = config.query_layout().expect("query layout");
        assert_eq!(query.key_dimension().get(), 128);
        let nz = |value| NonZeroUsize::new(value).expect("nonzero");
        let owner =
            RatioTwoOwnerLayout::new(nz(1), nz(5120), nz(512), nz(128), nz(32), nz(64), 1e-20)
                .expect("ratio-two owner layout");
        LayerOneConfig::new(owner, config.attention_layout(2).expect("layout"), nz(512))
            .expect("ratio-two owner with 128-wide keys");
        let keys = IndexKeyLayout::new(nz(1), nz(512), nz(128), nz(32), 1e-20).expect("key layout");
        LayerThreeConfig::new(
            keys,
            nz(5120),
            nz(128),
            config.attention_layout(20).expect("layout"),
            nz(128),
            nz(512),
        )
        .expect("ratio-one owner with 128-wide keys");
        LayerFourConfig::new(query, config.attention_layout(24).expect("layout"), nz(512))
            .expect("ratio-one indexer");
    }

    /// The real head passes every BF16 cap and reaches the length check; the
    /// FP32 head stays bounded.
    #[test]
    fn real_vocabulary_fits_the_bf16_head_only() {
        assert!(matches!(
            FinalHead::with_weights(&[0x3f80; 5120], HeadWeights::Bf16(&[]), 129_280, 4, 1e-20),
            Err(FinalHeadError::Length {
                field: "head_weight",
                actual: 0,
                expected: 661_913_600,
            })
        ));
        assert!(matches!(
            FinalHead::new(&[0x3f80; 5120], &[], 129_280, 4, 1e-20),
            Err(FinalHeadError::ElementLimit {
                field: "head_weight",
                ..
            })
        ));
    }

    fn bf16(bits: u16) -> f64 {
        f64::from(f32::from_bits(u32::from(bits) << 16))
    }

    /// (mismatching elements, max |difference|, cosine), as the private harness reports.
    fn agreement(native: &[u16], source: &[u16]) -> (usize, f64, f64) {
        assert_eq!(native.len(), source.len());
        let (mut mismatch, mut max_abs, mut dot, mut nn, mut ss) = (0, 0_f64, 0_f64, 0_f64, 0_f64);
        for (&a, &b) in native.iter().zip(source) {
            let (x, y) = (bf16(a), bf16(b));
            mismatch += usize::from(a != b);
            max_abs = max_abs.max((x - y).abs());
            dot += x * y;
            nn += x * x;
            ss += y * y;
        }
        (mismatch, max_abs, dot / (nn.sqrt() * ss.sqrt()))
    }

    #[test]
    #[ignore = "requires .agents/receipts/route-trace weights, metadata and capture-parity3 (about 8 GB of non-expert layers)"]
    fn teacher_forced_layers_match_the_source_capture() {
        const TOKENS: usize = 3;
        const REVISION: &str = "dba1be0a40aa45a94ad051997016db3960a90277";
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.agents/receipts");
        let trace = root.join("route-trace");
        let config = V41InferenceConfig::parse(
            &std::fs::read_to_string(trace.join("inference-config.json")).expect("config"),
        )
        .expect("pinned inference config");
        let layers = std::env::var("METALLIX_V41_TEACHER_LAYERS")
            .map_or(config.layers(), |value| value.parse().expect("layer count"));
        let mut cache = V41RangeCache::load(
            V41LocalWeightsSource::new(trace.join("weights"), REVISION),
            &root.join(
                "control/receipts/candidate-control/real-expert/model.safetensors.index.json",
            ),
            &trace,
            REVISION,
            256 << 20,
        )
        .expect("pinned cache");
        let weights = V41CheckpointWeights::load(
            &mut cache,
            &config,
            0..layers,
            NonZeroUsize::new(128).expect("nonzero"),
        )
        .expect("non-expert layers");

        // The real layer-1 Engram config and weights build a session.
        if layers > 1 {
            let inputs = crate::engram::inputs::EngramHashInputs::parse(
                &std::fs::read(root.join("engram-hash/v41-engram-inputs.bin"))
                    .expect("Engram inputs"),
                &crate::engram::inputs::V41_ENGRAM_INPUTS_IDENTITY,
            )
            .expect("pinned Engram inputs");
            let (engram_config, engram_weights) = weights
                .engram_parts(0, &inputs)
                .expect("layer-1 Engram parts");
            crate::reduced::EngramSession::new(engram_config, engram_weights)
                .expect("layer-1 Engram session");
        }

        let cache = Mutex::new(cache);
        let capture = trace.join("capture-parity3");
        let read = |layer: usize, kind: &str, ty: &str| {
            std::fs::read(capture.join(format!("layer{layer:02}.{kind}.torch.{ty}.bin")))
                .expect("captured tensor")
        };
        let mut state = V41TeacherState::default();
        for layer in 0..layers {
            let residual = le_u16(&read(layer, "in", "bfloat16"));
            let pre: Vec<f32> = read(layer, "premix_in", "float32")
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
                .collect();
            assert_eq!(residual.len(), TOKENS * config.hc_mult * config.dim);
            let experts =
                V41CachedRoutedExperts::new(&cache, layer, config.dim, config.moe_inter_dim);
            let layer_trace = weights
                .teacher_forced_layer(layer, &residual, &pre, &mut state, &experts)
                .unwrap_or_else(|error| panic!("layer {layer}: {error}"));
            let attn_in = agreement(
                &layer_trace.attention_input,
                &le_u16(&read(layer, "attn_in", "bfloat16")),
            );
            let attn_out = agreement(
                &layer_trace.attention_output,
                &le_u16(&read(layer, "attn_out", "bfloat16")),
            );
            eprintln!("layer {layer:02} attn_in {attn_in:?} attn_out {attn_out:?}");
            assert_eq!(
                attn_in.0, 0,
                "layer {layer}: attention input must match exactly"
            );
            match &layer_trace.tail {
                Err(reason) => eprintln!("layer {layer:02} tail not run: {reason}"),
                Ok(tail) => eprintln!(
                    "layer {layer:02} ffn_in {:?} ffn_out {:?} out {:?}",
                    agreement(&tail.ffn_input, &le_u16(&read(layer, "ffn_in", "bfloat16"))),
                    agreement(
                        &tail.ffn_output,
                        &le_u16(&read(layer, "ffn_out", "bfloat16"))
                    ),
                    agreement(&tail.output, &le_u16(&read(layer, "out", "bfloat16"))),
                ),
            }
        }
    }
}
