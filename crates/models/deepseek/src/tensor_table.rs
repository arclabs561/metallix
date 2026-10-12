//! The tensors a V4.1 checkpoint must hold, derived from its configuration alone.
//!
//! [`V41TensorTable::from_config`] reads only `config.json` and lists every
//! text and draft tensor with the dtype and shape the pinned source would allocate
//! for it. [`V41TensorTable::admit`] then compares that list with the index and
//! the shard headers before any payload is read. Taking shapes from the
//! configuration rather than from the headers is the point: a header that
//! disagrees with the model it claims to be is refused here, not halfway
//! through a forward pass.
//!
//! Draft (`mtp.*`) tensors are checked too; vision tensors are classified only.
//! Geometry follows the pinned reference `Linear`, `Attention`, `Block`,
//! `Engram` and `DSparkBlock` constructors (source hashes in
//! `scripts/v41_source_loader.py`). Checkpoint storage uses BF16 for the
//! compressor and output/confidence heads, which the reference promotes to FP32.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use thiserror::Error;

use crate::{V41SafetensorsHeader, V41StorageDtype, manifest::V41SafetensorsIndex};

/// Elements per E8M0 scale along a routed expert's input dimension. The pinned
/// `kernel.py` fixes this (`fp4_block_size`); the configuration does not carry it.
const FP4_SCALE_BLOCK: u64 = 32;

/// Codes per byte in the routed experts' packed FP4 storage.
const FP4_CODES_PER_BYTE: u64 = 2;

/// Mixing coefficients per Hyper-Connection projection: pre, post and the
/// `hc_mult x hc_mult` residual matrix.
const fn hc_mix(hc_mult: u64) -> u64 {
    (2 + hc_mult) * hc_mult
}

/// The dtype and shape a checkpoint tensor must have.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct V41ExpectedTensor {
    dtype: V41StorageDtype,
    shape: Vec<u64>,
}

impl V41ExpectedTensor {
    fn new(dtype: V41StorageDtype, shape: &[u64]) -> Self {
        Self {
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn bytes(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(self.dtype.bytes_per_element(), |n, &dim| n.checked_mul(dim))
    }
}

/// Which part of the model a checkpoint tensor belongs to.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum V41TensorRole {
    /// Backbone layers, embedding, final norm and head.
    Text,
    /// Next-token-prediction draft layers (`mtp.N.*`).
    Draft,
    /// The vision tower, its aligner and the image marker embeddings.
    Vision,
}

/// What [`V41TensorTable::admit`] accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct V41Admission {
    text_tensors: usize,
    draft_tensors: usize,
    vision_tensors: usize,
    text_bytes: u64,
    total_bytes: u64,
}

/// Every text and draft tensor of one V4.1 configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct V41TensorTable {
    text: BTreeMap<String, V41ExpectedTensor>,
    layers: u64,
    draft_layers: u64,
    routed_expert_bytes: u64,
}

impl V41TensorTable {
    /// Builds the table from a V4.1 `config.json` (top level or `text_config`).
    ///
    /// # Errors
    ///
    /// [`V41TensorTableError`] when the document is not JSON, is not a
    /// `deepseek_v41_text` configuration with FP8/FP4 quantization, or has a
    /// width or layer list the table cannot be derived from.
    pub(super) fn from_config(json: &str) -> Result<Self, V41TensorTableError> {
        crate::V41ExecutionShape::parse(json).map_err(V41TensorTableError::Execution)?;
        let root: RawRoot = serde_json::from_str(json).map_err(V41TensorTableError::Json)?;
        let c = root.text_config.unwrap_or(root.text);
        if c.model_type != "deepseek_v41_text" {
            return Err(V41TensorTableError::ModelType(c.model_type));
        }
        let q = root
            .quantization_config
            .ok_or(V41TensorTableError::Quantization)?;
        if q.quant_method != "fp8" || q.scale_fmt != "ue8m0" || q.expert_dtype != "fp4" {
            return Err(V41TensorTableError::Quantization);
        }
        let [block_rows, block_columns] = q.weight_block_size.as_slice() else {
            return Err(V41TensorTableError::Quantization);
        };
        Builder::new(
            &c,
            *block_rows,
            *block_columns,
            root.vision_config
                .as_ref()
                .is_some_and(|vision| vision.num_hidden_layers > 0),
        )?
        .build()
    }

    /// Which part of the model `name` belongs to, or `None` for a name no
    /// part of a V4.1 checkpoint should have.
    #[must_use]
    pub(super) fn role(&self, name: &str) -> Option<V41TensorRole> {
        if let Some(rest) = name.strip_prefix("layers.") {
            let layer = rest.split('.').next()?.parse::<u64>().ok()?;
            return (layer < self.layers).then_some(V41TensorRole::Text);
        }
        if let Some(rest) = name.strip_prefix("mtp.") {
            let layer = rest.split('.').next()?.parse::<u64>().ok()?;
            return (layer < self.draft_layers).then_some(V41TensorRole::Draft);
        }
        match name {
            "embed.weight" | "head.weight" | "norm.weight" => Some(V41TensorRole::Text),
            "image_start" | "image_end" | "image_newline" => Some(V41TensorRole::Vision),
            _ if name.starts_with("vision.") || name.starts_with("aligner.") => {
                Some(V41TensorRole::Vision)
            }
            _ => None,
        }
    }

    /// Checks an index and its shard headers against the table, reading no
    /// payload. `headers` must hold one parsed header per index shard.
    ///
    /// Checked, in order: every index shard has exactly one header, whose
    /// names equal the index's names for that shard; every name has a role;
    /// the text and draft names equal the table, with the table's dtypes and
    /// shapes; every routed expert's six tensors total
    /// the configured routed-expert payload size; and all tensors
    /// together total the index's declared size. Offsets inside each shard
    /// (sorted, gap-free, ending at the file size) were already enforced when
    /// the header was parsed.
    ///
    /// # Errors
    ///
    /// The first failed check, as a [`V41AdmissionError`].
    pub(super) fn admit<'a, I>(
        &self,
        index: &V41SafetensorsIndex,
        headers: I,
    ) -> Result<V41Admission, V41AdmissionError>
    where
        I: IntoIterator<Item = (&'a str, &'a V41SafetensorsHeader)>,
    {
        let mut by_shard = BTreeMap::new();
        for (shard, header) in headers {
            if by_shard.insert(shard, header).is_some() {
                return Err(V41AdmissionError::DuplicateShard(shard.to_owned()));
            }
        }
        for shard in index.shard_paths() {
            let header = by_shard
                .remove(shard.as_str())
                .ok_or_else(|| V41AdmissionError::MissingShard(shard.clone()))?;
            header
                .validate_index_shard(index, shard)
                .map_err(|source| V41AdmissionError::ShardNames {
                    shard: shard.clone(),
                    source,
                })?;
            by_shard.insert(shard.as_str(), header);
        }
        if by_shard.len() != index.shard_paths().len() {
            let extra = by_shard
                .keys()
                .find(|shard| !index.shard_paths().iter().any(|path| path == *shard))
                .map_or_else(String::new, |shard| (*shard).to_owned());
            return Err(V41AdmissionError::UnindexedShard(extra));
        }

        let mut admission = V41Admission {
            text_tensors: 0,
            draft_tensors: 0,
            vision_tensors: 0,
            text_bytes: 0,
            total_bytes: 0,
        };
        let mut seen = BTreeSet::new();
        let mut experts: BTreeMap<(bool, u64, u64), u64> = BTreeMap::new();
        for header in by_shard.values() {
            for (name, range) in header.tensors() {
                let bytes = range.file_range().end - range.file_range().start;
                admission.total_bytes = admission
                    .total_bytes
                    .checked_add(bytes)
                    .ok_or(V41AdmissionError::Overflow)?;
                match self.role(name) {
                    None => return Err(V41AdmissionError::Unclassified(name.to_owned())),
                    Some(V41TensorRole::Vision) => admission.vision_tensors += 1,
                    Some(role @ (V41TensorRole::Text | V41TensorRole::Draft)) => {
                        let expected = self
                            .text
                            .get(name)
                            .ok_or_else(|| V41AdmissionError::Unexpected(name.to_owned()))?;
                        if expected.dtype != range.dtype() || expected.shape != range.shape() {
                            return Err(V41AdmissionError::Mismatch {
                                tensor: name.to_owned(),
                                expected: expected.clone(),
                                actual: V41ExpectedTensor::new(range.dtype(), range.shape()),
                            });
                        }
                        seen.insert(name);
                        if role == V41TensorRole::Text {
                            admission.text_tensors += 1;
                            admission.text_bytes = admission
                                .text_bytes
                                .checked_add(bytes)
                                .ok_or(V41AdmissionError::Overflow)?;
                        } else {
                            admission.draft_tensors += 1;
                        }
                        if let Some(key) = routed_expert_key(name) {
                            let total = experts.entry(key).or_default();
                            *total = total
                                .checked_add(bytes)
                                .ok_or(V41AdmissionError::Overflow)?;
                        }
                    }
                }
            }
        }
        if let Some(missing) = self.text.keys().find(|name| !seen.contains(name.as_str())) {
            return Err(V41AdmissionError::Missing(missing.clone()));
        }
        if let Some((&(draft, layer, expert), &bytes)) = experts
            .iter()
            .find(|(_, bytes)| **bytes != self.routed_expert_bytes)
        {
            return Err(V41AdmissionError::ExpertBytes {
                draft,
                layer,
                expert,
                expected: self.routed_expert_bytes,
                actual: bytes,
            });
        }
        let declared = index.total_bytes().get();
        if admission.total_bytes != declared {
            return Err(V41AdmissionError::TotalBytes {
                declared,
                headers: admission.total_bytes,
            });
        }
        Ok(admission)
    }
}

/// Namespace, layer and expert of a routed-expert tensor.
fn routed_expert_key(name: &str) -> Option<(bool, u64, u64)> {
    let draft = name.starts_with("mtp.");
    let rest = name
        .strip_prefix("layers.")
        .or_else(|| name.strip_prefix("mtp."))?;
    let (layer, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("ffn.experts.")?;
    let (expert, _) = rest.split_once('.')?;
    Some((draft, layer.parse().ok()?, expert.parse().ok()?))
}

/// A configuration the table cannot be derived from.
#[derive(Debug, Error)]
#[non_exhaustive]
pub(super) enum V41TensorTableError {
    /// The document is not JSON or misses a field.
    #[error("invalid V4.1 config JSON: {0}")]
    Json(serde_json::Error),
    #[error("invalid execution contract: {0}")]
    Execution(crate::V41ConfigError),
    #[error("configured tensor table exceeds bounded metadata admission or arithmetic capacity")]
    Capacity,
    /// The text model is not `deepseek_v41_text`.
    #[error("config model_type is {0:?}, not deepseek_v41_text")]
    ModelType(String),
    /// Quantization is not the published FP8/UE8M0/FP4 scheme with a 2-D block.
    #[error("config quantization is not FP8 weights, UE8M0 scales and FP4 experts")]
    Quantization,
    /// A width is zero or does not divide as the tensor layout needs.
    #[error("config field {field} = {value} does not fit the V4.1 tensor layout")]
    Width {
        /// The configuration field.
        field: &'static str,
        /// Its value.
        value: u64,
    },
    /// A layer list names a layer outside the backbone, or has the wrong length.
    #[error("config field {field} does not fit {layers} backbone layers")]
    LayerList {
        /// The configuration field.
        field: &'static str,
        /// Backbone layers.
        layers: u64,
    },
}

/// A checkpoint that does not match its configuration.
#[derive(Debug, Error)]
#[non_exhaustive]
pub(super) enum V41AdmissionError {
    #[error("checkpoint byte total overflow")]
    Overflow,
    /// The index names a shard with no supplied header.
    #[error("no header for indexed shard {0}")]
    MissingShard(String),
    /// A header was supplied twice for one shard.
    #[error("two headers for shard {0}")]
    DuplicateShard(String),
    /// A header was supplied for a shard the index does not name.
    #[error("header for unindexed shard {0}")]
    UnindexedShard(String),
    /// A shard's header names differ from the index's names for it.
    #[error("shard {shard} disagrees with the index: {source}")]
    ShardNames {
        /// The shard.
        shard: String,
        /// The header/index disagreement.
        source: crate::V41SafetensorsHeaderError,
    },
    /// A tensor name belongs to no part of a V4.1 checkpoint.
    #[error("tensor {0} belongs to no part of the model")]
    Unclassified(String),
    /// A text or draft tensor the configuration does not produce.
    #[error("text tensor {0} is not in the configured table")]
    Unexpected(String),
    /// A configured text or draft tensor is absent.
    #[error("configured text tensor {0} is absent")]
    Missing(String),
    /// A text or draft tensor's dtype or shape differs from the configuration.
    #[error("tensor {tensor} is {actual:?}, the config needs {expected:?}")]
    Mismatch {
        /// The tensor.
        tensor: String,
        /// What the configuration requires.
        expected: V41ExpectedTensor,
        /// What the header declares.
        actual: V41ExpectedTensor,
    },
    /// A routed expert's six tensors do not total one expert's bytes.
    #[error("draft={draft} layer {layer} expert {expert} holds {actual} bytes, not {expected}")]
    ExpertBytes {
        /// Whether this is a draft layer.
        draft: bool,
        /// Backbone layer.
        layer: u64,
        /// Expert index.
        expert: u64,
        /// Bytes one expert must hold.
        expected: u64,
        /// Bytes found.
        actual: u64,
    },
    /// The headers' tensors do not total the index's declared size.
    #[error("headers hold {headers} payload bytes; the index declares {declared}")]
    TotalBytes {
        /// The index `total_size`.
        declared: u64,
        /// The sum over every header tensor.
        headers: u64,
    },
}

#[derive(Deserialize)]
struct RawRoot {
    #[serde(default)]
    text_config: Option<RawText>,
    #[serde(flatten)]
    text: RawText,
    #[serde(default)]
    quantization_config: Option<RawQuantization>,
    #[serde(default)]
    vision_config: Option<RawVision>,
}

#[derive(Deserialize)]
struct RawVision {
    num_hidden_layers: u16,
}

#[derive(Deserialize)]
struct RawQuantization {
    quant_method: String,
    scale_fmt: String,
    expert_dtype: String,
    weight_block_size: Vec<u64>,
}

#[derive(Deserialize)]
struct RawText {
    #[serde(default)]
    model_type: String,
    #[serde(default)]
    vocab_size: u64,
    #[serde(default)]
    hidden_size: u64,
    #[serde(default)]
    num_hidden_layers: u64,
    #[serde(default)]
    num_nextn_predict_layers: u64,
    #[serde(default)]
    num_attention_heads: u64,
    #[serde(default)]
    head_dim: u64,
    #[serde(default)]
    q_lora_rank: u64,
    #[serde(default)]
    o_lora_rank: u64,
    #[serde(default)]
    o_groups: u64,
    #[serde(default)]
    n_routed_experts: u64,
    #[serde(default)]
    n_shared_experts: u64,
    #[serde(default)]
    moe_intermediate_size: u64,
    #[serde(default)]
    hc_mult: u64,
    #[serde(default)]
    compress_ratios: Vec<u64>,
    #[serde(default)]
    kv_source_layer_ids: Vec<u64>,
    #[serde(default)]
    index_source_layer_ids: Vec<u64>,
    #[serde(default)]
    index_n_heads: u64,
    #[serde(default)]
    index_head_dim: u64,
    #[serde(default)]
    engram_layer_ids: Vec<u64>,
    #[serde(default)]
    engram_num_embeddings: Vec<u64>,
    #[serde(default)]
    engram_max_ngram_size: u64,
    #[serde(default)]
    engram_n_heads: u64,
    #[serde(default)]
    engram_head_dim: u64,
    #[serde(default)]
    dspark_target_layer_ids: Vec<u64>,
    #[serde(default)]
    dspark_markov_rank: u64,
    #[serde(default)]
    dspark_n_routed_experts: u64,
}

fn validate_expansion(c: &RawText) -> Result<(), V41TensorTableError> {
    // Every generated tensor consumes at least 32 JSON bytes in an index.
    // Bound expansion by the inspector's existing 16 MiB metadata budget.
    let entries = u128::from(c.num_hidden_layers) * (96 + 6 * u128::from(c.n_routed_experts))
        + u128::from(c.num_nextn_predict_layers)
            * (96 + 6 * u128::from(c.dspark_n_routed_experts.max(c.n_routed_experts)));
    if entries * 32 > 16 * 1024 * 1024 {
        return Err(V41TensorTableError::Capacity);
    }
    // ExecutionShape already bounds the core dimensions. Bound the extra
    // multiplicative dimensions to the same u16 representation before math.
    for value in [
        c.hc_mult,
        c.engram_max_ngram_size,
        c.engram_n_heads,
        c.engram_head_dim,
        c.dspark_markov_rank,
        c.dspark_n_routed_experts,
    ] {
        if value > u64::from(u16::MAX) {
            return Err(V41TensorTableError::Capacity);
        }
    }
    if c.num_nextn_predict_layers > 0
        && (c.dspark_markov_rank == 0
            || c.dspark_target_layer_ids.is_empty()
            || c.dspark_target_layer_ids
                .iter()
                .any(|&id| id >= c.num_hidden_layers)
            || c.compress_ratios
                [usize::try_from(c.num_hidden_layers).expect("validated u16 layer count")..]
                .iter()
                .any(|&ratio| ratio != 0))
    {
        return Err(V41TensorTableError::LayerList {
            field: "dspark",
            layers: c.num_hidden_layers,
        });
    }
    Ok(())
}

struct Builder<'c> {
    c: &'c RawText,
    block_rows: u64,
    block_columns: u64,
    text: BTreeMap<String, V41ExpectedTensor>,
    vision: bool,
    invalid_bytes: bool,
}

impl<'c> Builder<'c> {
    fn new(
        c: &'c RawText,
        block_rows: u64,
        block_columns: u64,
        vision: bool,
    ) -> Result<Self, V41TensorTableError> {
        validate_expansion(c)?;
        let positive = [
            ("vocab_size", c.vocab_size),
            ("hidden_size", c.hidden_size),
            ("num_hidden_layers", c.num_hidden_layers),
            ("num_attention_heads", c.num_attention_heads),
            ("head_dim", c.head_dim),
            ("q_lora_rank", c.q_lora_rank),
            ("o_lora_rank", c.o_lora_rank),
            ("o_groups", c.o_groups),
            ("n_routed_experts", c.n_routed_experts),
            ("moe_intermediate_size", c.moe_intermediate_size),
            ("hc_mult", c.hc_mult),
            ("weight_block_size[0]", block_rows),
            ("weight_block_size[1]", block_columns),
        ];
        if let Some(&(field, value)) = positive.iter().find(|(_, value)| *value == 0) {
            return Err(V41TensorTableError::Width { field, value });
        }
        if !(c.num_attention_heads * c.head_dim).is_multiple_of(c.o_groups) {
            return Err(V41TensorTableError::Width {
                field: "o_groups",
                value: c.o_groups,
            });
        }
        for (field, value) in [
            ("hidden_size", c.hidden_size),
            ("moe_intermediate_size", c.moe_intermediate_size),
        ] {
            if !value.is_multiple_of(FP4_SCALE_BLOCK) || !value.is_multiple_of(FP4_CODES_PER_BYTE) {
                return Err(V41TensorTableError::Width { field, value });
            }
        }
        let layers = c.num_hidden_layers;
        let in_backbone = |ids: &[u64]| ids.iter().all(|&layer| layer < layers);
        if c.compress_ratios.len() as u64 != layers + c.num_nextn_predict_layers {
            return Err(V41TensorTableError::LayerList {
                field: "compress_ratios",
                layers,
            });
        }
        for (field, ids) in [
            ("kv_source_layer_ids", &c.kv_source_layer_ids),
            ("index_source_layer_ids", &c.index_source_layer_ids),
            ("engram_layer_ids", &c.engram_layer_ids),
        ] {
            if !in_backbone(ids) || ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
                return Err(V41TensorTableError::LayerList { field, layers });
            }
        }
        if c.engram_num_embeddings.len() != c.engram_layer_ids.len()
            || c.engram_num_embeddings.contains(&0)
        {
            return Err(V41TensorTableError::LayerList {
                field: "engram_num_embeddings",
                layers,
            });
        }
        if !c.engram_layer_ids.is_empty()
            && (c.engram_max_ngram_size < 2
                || c.engram_n_heads == 0
                || c.engram_head_dim == 0
                || !c.engram_head_dim.is_multiple_of(32))
        {
            return Err(V41TensorTableError::Width {
                field: "engram_max_ngram_size",
                value: c.engram_max_ngram_size,
            });
        }
        Ok(Self {
            c,
            block_rows,
            block_columns,
            text: BTreeMap::new(),
            vision,
            invalid_bytes: false,
        })
    }

    fn put(&mut self, name: String, dtype: V41StorageDtype, shape: &[u64]) {
        let value = V41ExpectedTensor::new(dtype, shape);
        if value.bytes().is_none() {
            self.invalid_bytes = true;
        }
        self.text.insert(name, value);
    }

    /// An FP8 linear `[out, in]` and its E8M0 block scales.
    fn fp8(&mut self, prefix: &str, out: u64, input: u64) {
        self.put(
            format!("{prefix}.weight"),
            V41StorageDtype::F8E4M3Fn,
            &[out, input],
        );
        self.put(
            format!("{prefix}.scale"),
            V41StorageDtype::F8E8M0Fnu,
            &[
                out.div_ceil(self.block_rows),
                input.div_ceil(self.block_columns),
            ],
        );
    }

    /// A packed FP4 expert projection `[out, in]`: two codes per byte and one
    /// E8M0 scale per [`FP4_SCALE_BLOCK`] inputs.
    fn fp4(&mut self, prefix: &str, out: u64, input: u64) {
        self.put(
            format!("{prefix}.weight"),
            V41StorageDtype::I8,
            &[out, input / FP4_CODES_PER_BYTE],
        );
        self.put(
            format!("{prefix}.scale"),
            V41StorageDtype::F8E8M0Fnu,
            &[out, input / FP4_SCALE_BLOCK],
        );
    }

    fn attention(&mut self, layer: u64, prefix: &str) {
        use V41StorageDtype::{Bf16, F32};
        let config = self.c;
        let (hidden, heads, head_dim) = (
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim,
        );
        let attention = format!("{prefix}.attn");
        self.put(format!("{attention}.attn_sink"), F32, &[heads]);
        self.put(
            format!("{attention}.q_norm.weight"),
            Bf16,
            &[config.q_lora_rank],
        );
        self.put(format!("{attention}.kv_norm.weight"), Bf16, &[head_dim]);
        self.fp8(&format!("{attention}.wq_a"), config.q_lora_rank, hidden);
        self.fp8(
            &format!("{attention}.wq_b"),
            heads * head_dim,
            config.q_lora_rank,
        );
        self.fp8(&format!("{attention}.wkv"), head_dim, hidden);
        self.fp8(
            &format!("{attention}.wo_a"),
            config.o_groups * config.o_lora_rank,
            heads * head_dim / config.o_groups,
        );
        self.fp8(
            &format!("{attention}.wo_b"),
            hidden,
            config.o_groups * config.o_lora_rank,
        );

        let kv_source = config.kv_source_layer_ids.contains(&layer);
        if kv_source {
            // Only ratios above one compute a separate score (gate) projection.
            self.put(
                format!("{attention}.compressor.wkv.weight"),
                Bf16,
                &[head_dim, hidden],
            );
            self.put(
                format!("{attention}.compressor.norm.weight"),
                Bf16,
                &[head_dim],
            );
            if config.compress_ratios[usize::try_from(layer).expect("validated u16 layer index")]
                > 1
            {
                self.put(
                    format!("{attention}.compressor.wgate.weight"),
                    Bf16,
                    &[head_dim, hidden],
                );
            }
        }
        if config.index_source_layer_ids.contains(&layer) {
            let indexer = format!("{attention}.indexer");
            self.fp8(
                &format!("{indexer}.wq_b"),
                config.index_n_heads * config.index_head_dim,
                config.q_lora_rank,
            );
            self.put(
                format!("{indexer}.weights_proj.weight"),
                Bf16,
                &[config.index_n_heads, hidden],
            );
            // Index keys are owned (projected and cached) only at KV sources.
            if kv_source {
                self.put(
                    format!("{indexer}.wk.weight"),
                    Bf16,
                    &[config.index_head_dim, head_dim],
                );
                self.put(
                    format!("{indexer}.k_norm.weight"),
                    Bf16,
                    &[config.index_head_dim],
                );
            }
        }
    }

    fn engram(&mut self, layer: u64, prefix: &str) {
        use V41StorageDtype::{Bf16, F8E4M3Fn, F8E8M0Fnu};
        let config = self.c;
        let hidden = config.hidden_size;
        if let Some(slot) = config.engram_layer_ids.iter().position(|&id| id == layer) {
            let engram = format!("{prefix}.engram");
            let rows = config.engram_num_embeddings[slot];
            self.put(
                format!("{engram}.embed.weight"),
                F8E4M3Fn,
                &[rows, config.engram_head_dim],
            );
            self.put(
                format!("{engram}.embed.scale"),
                F8E8M0Fnu,
                &[rows, config.engram_head_dim.div_ceil(self.block_columns)],
            );
            let columns =
                (config.engram_max_ngram_size - 1) * config.engram_n_heads * config.engram_head_dim;
            self.fp8(
                &format!("{engram}.wkv"),
                (config.hc_mult + 1) * hidden,
                columns,
            );
            self.put(
                format!("{engram}.q_weight"),
                Bf16,
                &[config.hc_mult, hidden],
            );
            self.put(
                format!("{engram}.k_weight"),
                Bf16,
                &[config.hc_mult, hidden],
            );
        }
    }

    fn ffn(&mut self, prefix: &str, experts: u64) {
        use V41StorageDtype::{Bf16, F32};
        let config = self.c;
        let (hidden, inter) = (config.hidden_size, config.moe_intermediate_size);
        let ffn = format!("{prefix}.ffn");
        self.put(format!("{ffn}.gate.weight"), Bf16, &[experts, hidden]);
        self.put(format!("{ffn}.gate.bias"), F32, &[experts]);
        if self.vision {
            self.put(format!("{ffn}.gate.bias_vl"), F32, &[experts]);
        }
        if config.n_shared_experts > 0 {
            let shared = inter * config.n_shared_experts;
            self.fp8(&format!("{ffn}.shared_experts.w1"), shared, hidden);
            self.fp8(&format!("{ffn}.shared_experts.w2"), hidden, shared);
            self.fp8(&format!("{ffn}.shared_experts.w3"), shared, hidden);
        }
        for expert in 0..experts {
            let expert_prefix = format!("{ffn}.experts.{expert}");
            self.fp4(&format!("{expert_prefix}.w1"), inter, hidden);
            self.fp4(&format!("{expert_prefix}.w2"), hidden, inter);
            self.fp4(&format!("{expert_prefix}.w3"), inter, hidden);
        }
    }

    fn build(mut self) -> Result<V41TensorTable, V41TensorTableError> {
        use V41StorageDtype::{Bf16, F32};
        let config = self.c;
        let hidden = config.hidden_size;
        self.put("embed.weight".into(), Bf16, &[config.vocab_size, hidden]);
        self.put("head.weight".into(), Bf16, &[config.vocab_size, hidden]);
        self.put("norm.weight".into(), Bf16, &[hidden]);

        let mix = hc_mix(config.hc_mult);
        for layer in 0..config.num_hidden_layers + config.num_nextn_predict_layers {
            let draft = layer >= config.num_hidden_layers;
            let stage = layer.saturating_sub(config.num_hidden_layers);
            let prefix = if draft {
                format!("mtp.{stage}")
            } else {
                format!("layers.{layer}")
            };
            let experts = if draft && config.dspark_n_routed_experts != 0 {
                config.dspark_n_routed_experts
            } else {
                config.n_routed_experts
            };
            if draft && stage == 0 {
                self.fp8(
                    &format!("{prefix}.main_proj"),
                    hidden,
                    hidden * config.dspark_target_layer_ids.len() as u64,
                );
                self.put(format!("{prefix}.main_norm.weight"), Bf16, &[hidden]);
            }
            if draft && stage + 1 == config.num_nextn_predict_layers {
                self.put(format!("{prefix}.norm.weight"), Bf16, &[hidden]);
                for part in ["embed", "head"] {
                    self.put(
                        format!("{prefix}.markov_head.{part}.weight"),
                        Bf16,
                        &[config.vocab_size, config.dspark_markov_rank],
                    );
                }
                self.put(
                    format!("{prefix}.confidence_head.proj.weight"),
                    Bf16,
                    &[1, hidden + config.dspark_markov_rank],
                );
            }
            self.put(format!("{prefix}.attn_norm.weight"), Bf16, &[hidden]);
            self.put(format!("{prefix}.ffn_norm.weight"), Bf16, &[hidden]);
            for part in ["attn", "ffn"] {
                self.put(
                    format!("{prefix}.hc_{part}_fn"),
                    F32,
                    &[mix, config.hc_mult * hidden],
                );
                self.put(format!("{prefix}.hc_{part}_base"), F32, &[mix]);
                self.put(format!("{prefix}.hc_{part}_scale"), F32, &[3]);
            }

            self.attention(layer, &prefix);
            self.engram(layer, &prefix);

            self.ffn(&prefix, experts);
        }

        if self.invalid_bytes {
            return Err(V41TensorTableError::Capacity);
        }
        let routed_expert_bytes = ["w1", "w2", "w3"]
            .iter()
            .flat_map(|w| {
                ["weight", "scale"].map(move |k| format!("layers.0.ffn.experts.0.{w}.{k}"))
            })
            .map(|name| self.text[&name].bytes())
            .try_fold(0_u64, |sum, bytes| sum.checked_add(bytes?))
            .ok_or(V41TensorTableError::Capacity)?;
        Ok(V41TensorTable {
            text: self.text,
            layers: config.num_hidden_layers,
            draft_layers: config.num_nextn_predict_layers,
            routed_expert_bytes,
        })
    }
}

#[cfg(test)]
mod tests;
