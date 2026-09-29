//! Stateful bounded Engram composition over runtime token IDs and residuals.
//!
//! This module owns request-local hash history and immutable numerical operands.
//! It has no fixture, checkpoint, or expected-output dependency.

use thiserror::Error;

use crate::{
    engram::{
        CompressedToken, EngramHashError, EngramHashLayout, EngramHashState,
        embedding::{EngramEmbeddingError, EngramEmbeddingLayout, engram_embedding_bf16_reference},
        gate::{
            EngramGateError, EngramGateInputs, EngramGateLayout, EngramGateParams,
            engram_residual_gate_bf16_reference,
        },
    },
    precision::{
        ActivationGroup, ActivationQuantError, Fp8LinearError, bf16_to_f32, decode_e4m3fn,
        decode_e8m0, f32_to_bf16_rne, fp8_linear_runtime_f32, quantize_bf16_activations_e4m3fn,
    },
};

const MAX_SESSION_ELEMENTS: usize = 1 << 20;

/// Immutable layout and token-compression operands for one batch-one Engram request.
#[derive(Clone, Debug)]
pub struct EngramSessionConfig {
    hash_layout: EngramHashLayout,
    token_map: Vec<i64>,
    hash_layer: usize,
    capacity: usize,
    copies: usize,
    width: usize,
    embedding_rows: usize,
    embedding_width: usize,
    norm_epsilon: f32,
    gate_clamp: f32,
}

impl EngramSessionConfig {
    /// Creates bounded batch-one Engram request configuration.
    #[allow(
        clippy::too_many_arguments,
        reason = "each source-shaped geometry is explicit"
    )]
    pub fn new(
        hash_layout: EngramHashLayout,
        token_map: Vec<i64>,
        hash_layer: usize,
        capacity: usize,
        copies: usize,
        width: usize,
        embedding_rows: usize,
        embedding_width: usize,
        norm_epsilon: f32,
        gate_clamp: f32,
    ) -> Result<Self, EngramSessionError> {
        if token_map.is_empty() {
            return Err(EngramSessionError::EmptyTokenMap);
        }
        if token_map.len() > MAX_SESSION_ELEMENTS {
            return Err(EngramSessionError::ElementLimit {
                field: "token map",
                elements: token_map.len(),
            });
        }
        if let Some((index, &id)) = token_map.iter().enumerate().find(|&(_, &id)| id < 0) {
            return Err(EngramSessionError::NegativeTokenMap { index, id });
        }
        if hash_layer >= hash_layout.layers() {
            return Err(EngramSessionError::HashLayerOutOfRange {
                layer: hash_layer,
                layers: hash_layout.layers(),
            });
        }
        if [capacity, copies, width, embedding_rows, embedding_width].contains(&0) {
            return Err(EngramSessionError::EmptyDimension);
        }
        if !norm_epsilon.is_finite() || norm_epsilon <= 0.0 {
            return Err(EngramSessionError::InvalidEpsilon);
        }
        if !gate_clamp.is_finite() || gate_clamp <= 0.0 {
            return Err(EngramSessionError::InvalidGateClamp);
        }
        if !embedding_width.is_multiple_of(32) {
            return Err(EngramSessionError::EmbeddingWidthNotGrouped { embedding_width });
        }
        let columns = hash_columns(&hash_layout)?;
        let reduction = checked_product(columns, embedding_width, "WKV reduction")?;
        if !reduction.is_multiple_of(32) {
            return Err(EngramSessionError::ReductionNotGrouped { reduction });
        }
        let wkv_width = wkv_width(copies, width)?;
        for (field, elements) in [
            ("history", capacity),
            (
                "embedding table",
                checked_product(embedding_rows, embedding_width, "embedding table")?,
            ),
            (
                "WKV weight",
                checked_product(wkv_width, reduction, "WKV weight")?,
            ),
        ] {
            if elements > MAX_SESSION_ELEMENTS {
                return Err(EngramSessionError::ElementLimit { field, elements });
            }
        }
        Ok(Self {
            hash_layout,
            token_map,
            hash_layer,
            capacity,
            copies,
            width,
            embedding_rows,
            embedding_width,
            norm_epsilon,
            gate_clamp,
        })
    }
}

/// Owned immutable FP8/BF16 numerical operands for one Engram session.
#[derive(Clone, Debug)]
pub struct EngramSessionWeights {
    embedding_codes: Vec<u8>,
    embedding_scales: Vec<u8>,
    wkv_codes: Vec<u8>,
    wkv_scales: Vec<u8>,
    q_weight: Vec<u16>,
    k_weight: Vec<u16>,
}

impl EngramSessionWeights {
    /// Takes ownership of source-shaped Engram embedding, WKV, and gate weights.
    #[must_use]
    pub fn new(
        embedding_codes: Vec<u8>,
        embedding_scales: Vec<u8>,
        wkv_codes: Vec<u8>,
        wkv_scales: Vec<u8>,
        q_weight: Vec<u16>,
        k_weight: Vec<u16>,
    ) -> Self {
        Self {
            embedding_codes,
            embedding_scales,
            wkv_codes,
            wkv_scales,
            q_weight,
            k_weight,
        }
    }
}

/// Diagnostics retained from one completed Engram operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngramStepOutput {
    start: usize,
    hash_ids: Vec<i64>,
    embedding: Vec<u16>,
    wkv: Vec<u16>,
    key: Vec<u16>,
    value: Vec<u16>,
    output: Vec<u16>,
}

impl EngramStepOutput {
    #[must_use]
    pub const fn start(&self) -> usize {
        self.start
    }
    #[must_use]
    pub fn hash_ids(&self) -> &[i64] {
        &self.hash_ids
    }
    #[must_use]
    pub fn embedding(&self) -> &[u16] {
        &self.embedding
    }
    #[must_use]
    pub fn wkv(&self) -> &[u16] {
        &self.wkv
    }
    #[must_use]
    pub fn key(&self) -> &[u16] {
        &self.key
    }
    #[must_use]
    pub fn value(&self) -> &[u16] {
        &self.value
    }
    #[must_use]
    pub fn output(&self) -> &[u16] {
        &self.output
    }
}

/// Batch-one request-local Engram history and immutable runtime operands.
#[derive(Clone, Debug)]
pub struct EngramSession {
    config: EngramSessionConfig,
    weights: EngramSessionWeights,
    query_weights: Vec<f32>,
    key_weights: Vec<f32>,
    hashes: EngramHashState,
    next_start: usize,
}

impl EngramSession {
    /// Validates static operands and starts at absolute token position zero.
    pub fn new(
        config: EngramSessionConfig,
        weights: EngramSessionWeights,
    ) -> Result<Self, EngramSessionError> {
        validate_weights(&config, &weights)?;
        let query_weights = bf16_weights_to_f32(&weights.q_weight, "q weight")?;
        let key_weights = bf16_weights_to_f32(&weights.k_weight, "k weight")?;
        let hashes = EngramHashState::new(config.hash_layout.try_clone()?, 1, config.capacity)?;
        Ok(Self {
            config,
            weights,
            query_weights,
            key_weights,
            hashes,
            next_start: 0,
        })
    }

    /// Returns the only admissible start position for the next call.
    #[must_use]
    pub const fn next_start(&self) -> usize {
        self.next_start
    }

    /// Reconstructs pristine request-local history after a successfully allocated reset.
    pub fn reset(&mut self) -> Result<(), EngramSessionError> {
        let hashes = EngramHashState::new(
            self.config.hash_layout.try_clone()?,
            1,
            self.config.capacity,
        )?;
        self.hashes = hashes;
        self.next_start = 0;
        Ok(())
    }

    /// Computes one contiguous Engram chunk and commits hash history only on success.
    pub fn step(
        &mut self,
        start: usize,
        token_ids: &[i64],
        residual: &[u16],
    ) -> Result<EngramStepOutput, EngramSessionError> {
        if start != self.next_start {
            return Err(EngramSessionError::UnexpectedStart {
                actual: start,
                expected: self.next_start,
            });
        }
        let positions = token_ids.len();
        if positions == 0 {
            return Err(EngramSessionError::EmptyChunk);
        }
        let next_start = start
            .checked_add(positions)
            .ok_or(EngramSessionError::ShapeOverflow {
                field: "next start",
            })?;
        if next_start > self.config.capacity {
            return Err(EngramSessionError::ChunkExceedsCapacity {
                end: next_start,
                capacity: self.config.capacity,
            });
        }
        let expected_residual = checked_product(
            checked_product(positions, self.config.copies, "residual rows")?,
            self.config.width,
            "residual",
        )?;
        if residual.len() != expected_residual {
            return Err(EngramSessionError::Length {
                field: "residual",
                actual: residual.len(),
                expected: expected_residual,
            });
        }
        validate_bf16(residual, "residual")?;
        let mut tokens = reserved_vec(positions, "compressed tokens")?;
        for (index, &id) in token_ids.iter().enumerate() {
            tokens.push(self.compressed_token(index, id)?);
        }

        // All later work uses a fallibly allocated staged history, so any
        // error leaves history and the request cursor unchanged.
        let mut hashes = self.hashes.try_clone()?;
        let all_hashes = hashes.write_and_hash(&tokens, positions, start)?;
        let hash_ids = select_hash_column(
            &all_hashes,
            positions,
            self.config.hash_layout.layers(),
            hash_columns(&self.config.hash_layout)?,
            self.config.hash_layer,
        )?;
        let embedding = self.embedding(&hash_ids)?;
        let wkv = self.project_wkv(&embedding, positions)?;
        let (key, value) = self.split_wkv(&wkv, positions)?;
        let output = self.gate(residual, &key, &value, positions)?;

        self.hashes = hashes;
        self.next_start = next_start;
        Ok(EngramStepOutput {
            start,
            hash_ids,
            embedding,
            wkv,
            key,
            value,
            output,
        })
    }

    fn compressed_token(
        &self,
        index: usize,
        id: i64,
    ) -> Result<CompressedToken, EngramSessionError> {
        let position =
            usize::try_from(id).map_err(|_| EngramSessionError::TokenIdOutOfRange { index, id })?;
        let compressed = *self
            .config
            .token_map
            .get(position)
            .ok_or(EngramSessionError::TokenIdOutOfRange { index, id })?;
        if compressed < 0 {
            return Err(EngramSessionError::NegativeCompressedToken {
                index,
                id: compressed,
            });
        }
        Ok(CompressedToken::Live(compressed))
    }

    fn embedding(&self, hash_ids: &[i64]) -> Result<Vec<u16>, EngramSessionError> {
        let layout = EngramEmbeddingLayout::new(
            self.config.embedding_rows,
            self.config.embedding_width,
            32,
        )?;
        let elements = checked_product(
            hash_ids.len(),
            self.config.embedding_width,
            "embedding output",
        )?;
        let mut output = reserved_vec(elements, "embedding output")?;
        output.resize(elements, 0);
        engram_embedding_bf16_reference(
            hash_ids,
            &self.weights.embedding_codes,
            &self.weights.embedding_scales,
            layout,
            &mut output,
        )?;
        Ok(output)
    }

    fn project_wkv(
        &self,
        embedding: &[u16],
        positions: usize,
    ) -> Result<Vec<u16>, EngramSessionError> {
        let columns = hash_columns(&self.config.hash_layout)?;
        let reduction = checked_product(columns, self.config.embedding_width, "WKV reduction")?;
        let outputs = wkv_width(self.config.copies, self.config.width)?;
        let activation_elements = checked_product(positions, reduction, "WKV activations")?;
        if embedding.len() != activation_elements {
            return Err(EngramSessionError::Length {
                field: "embedding",
                actual: embedding.len(),
                expected: activation_elements,
            });
        }
        let mut codes = reserved_vec(activation_elements, "WKV activation codes")?;
        codes.resize(activation_elements, 0);
        let scale_elements = checked_product(positions, reduction / 32, "WKV activation scales")?;
        let mut scales = reserved_vec(scale_elements, "WKV activation scales")?;
        scales.resize(scale_elements, 0);
        quantize_bf16_activations_e4m3fn(
            embedding,
            positions,
            reduction,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )?;
        let output_elements = checked_product(positions, outputs, "WKV output")?;
        let mut projected = reserved_vec(output_elements, "WKV FP32 output")?;
        projected.resize(output_elements, 0.0);
        fp8_linear_runtime_f32(
            &codes,
            &scales,
            &self.weights.wkv_codes,
            &self.weights.wkv_scales,
            positions,
            reduction,
            outputs,
            ActivationGroup::Elements32,
            &mut projected,
        )?;
        let mut wkv = reserved_vec(output_elements, "WKV BF16 output")?;
        for value in projected {
            wkv.push(f32_to_bf16_rne(value));
        }
        Ok(wkv)
    }

    fn split_wkv(
        &self,
        wkv: &[u16],
        positions: usize,
    ) -> Result<(Vec<u16>, Vec<u16>), EngramSessionError> {
        let key_width = checked_product(self.config.copies, self.config.width, "key width")?;
        let row_width = wkv_width(self.config.copies, self.config.width)?;
        let expected = checked_product(positions, row_width, "WKV output")?;
        if wkv.len() != expected {
            return Err(EngramSessionError::Length {
                field: "WKV output",
                actual: wkv.len(),
                expected,
            });
        }
        let mut key = reserved_vec(checked_product(positions, key_width, "key")?, "key")?;
        let mut value = reserved_vec(
            checked_product(positions, self.config.width, "value")?,
            "value",
        )?;
        for row in wkv.chunks_exact(row_width) {
            key.extend_from_slice(&row[..key_width]);
            value.extend_from_slice(&row[key_width..]);
        }
        Ok((key, value))
    }

    fn gate(
        &self,
        residual: &[u16],
        key: &[u16],
        value: &[u16],
        positions: usize,
    ) -> Result<Vec<u16>, EngramSessionError> {
        let layout = EngramGateLayout::new(1, positions, self.config.copies, self.config.width)?;
        let params = EngramGateParams::new(self.config.norm_epsilon, self.config.gate_clamp)?;
        let elements = checked_product(
            checked_product(positions, self.config.copies, "gate rows")?,
            self.config.width,
            "gate output",
        )?;
        let mut output = reserved_vec(elements, "gate output")?;
        output.resize(elements, 0);
        engram_residual_gate_bf16_reference(
            EngramGateInputs {
                stream: residual,
                key,
                value,
                q_weight: &self.query_weights,
                k_weight: &self.key_weights,
                mask: None,
            },
            layout,
            params,
            &mut output,
        )?;
        Ok(output)
    }
}

fn validate_weights(
    config: &EngramSessionConfig,
    weights: &EngramSessionWeights,
) -> Result<(), EngramSessionError> {
    let columns = hash_columns(&config.hash_layout)?;
    let embedding_elements = checked_product(
        config.embedding_rows,
        config.embedding_width,
        "embedding weights",
    )?;
    let reduction = checked_product(columns, config.embedding_width, "WKV reduction")?;
    let outputs = wkv_width(config.copies, config.width)?;
    let wkv_elements = checked_product(outputs, reduction, "WKV weights")?;
    let wkv_scale_elements = checked_product(outputs.div_ceil(32), reduction / 32, "WKV scales")?;
    let gate_elements = checked_product(config.copies, config.width, "gate weights")?;
    for (field, actual, expected) in [
        (
            "embedding codes",
            weights.embedding_codes.len(),
            embedding_elements,
        ),
        (
            "embedding scales",
            weights.embedding_scales.len(),
            embedding_elements / 32,
        ),
        ("WKV codes", weights.wkv_codes.len(), wkv_elements),
        ("WKV scales", weights.wkv_scales.len(), wkv_scale_elements),
        ("q weight", weights.q_weight.len(), gate_elements),
        ("k weight", weights.k_weight.len(), gate_elements),
    ] {
        if actual != expected {
            return Err(EngramSessionError::Length {
                field,
                actual,
                expected,
            });
        }
    }
    for (field, codes) in [
        ("embedding codes", &weights.embedding_codes),
        ("WKV codes", &weights.wkv_codes),
    ] {
        if let Some((index, _)) = codes
            .iter()
            .enumerate()
            .find(|&(_, &code)| !decode_e4m3fn(code).is_finite())
        {
            return Err(EngramSessionError::NonFiniteFp8 { field, index });
        }
    }
    for (field, scales) in [
        ("embedding scales", &weights.embedding_scales),
        ("WKV scales", &weights.wkv_scales),
    ] {
        if let Some((index, _)) = scales
            .iter()
            .enumerate()
            .find(|&(_, &scale)| !decode_e8m0(scale).is_finite())
        {
            return Err(EngramSessionError::NonFiniteScale { field, index });
        }
    }
    validate_bf16(&weights.q_weight, "q weight")?;
    validate_bf16(&weights.k_weight, "k weight")
}

fn validate_bf16(values: &[u16], field: &'static str) -> Result<(), EngramSessionError> {
    if let Some((index, _)) = values
        .iter()
        .enumerate()
        .find(|&(_, &bits)| !bf16_to_f32(bits).is_finite())
    {
        return Err(EngramSessionError::NonFiniteBf16 { field, index });
    }
    Ok(())
}

fn bf16_weights_to_f32(
    values: &[u16],
    field: &'static str,
) -> Result<Vec<f32>, EngramSessionError> {
    validate_bf16(values, field)?;
    let mut converted = reserved_vec(values.len(), field)?;
    converted.extend(values.iter().map(|&bits| bf16_to_f32(bits)));
    Ok(converted)
}

fn hash_columns(layout: &EngramHashLayout) -> Result<usize, EngramSessionError> {
    checked_product(layout.max_ngram_size() - 1, layout.heads(), "hash columns")
}

fn wkv_width(copies: usize, width: usize) -> Result<usize, EngramSessionError> {
    let rows = copies
        .checked_add(1)
        .ok_or(EngramSessionError::ShapeOverflow { field: "WKV rows" })?;
    checked_product(rows, width, "WKV width")
}

fn checked_product(
    left: usize,
    right: usize,
    field: &'static str,
) -> Result<usize, EngramSessionError> {
    match left.checked_mul(right) {
        Some(elements) if elements <= MAX_SESSION_ELEMENTS => Ok(elements),
        Some(elements) => Err(EngramSessionError::ElementLimit { field, elements }),
        None => Err(EngramSessionError::ShapeOverflow { field }),
    }
}

fn reserved_vec<T>(elements: usize, field: &'static str) -> Result<Vec<T>, EngramSessionError> {
    if elements > MAX_SESSION_ELEMENTS {
        return Err(EngramSessionError::ElementLimit { field, elements });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| EngramSessionError::AllocationFailed { field, elements })?;
    Ok(values)
}

/// Errors from session construction, request ordering, or Engram primitives.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum EngramSessionError {
    #[error("Engram token map must not be empty")]
    EmptyTokenMap,
    #[error("Engram token map entry {index} is negative: {id}")]
    NegativeTokenMap { index: usize, id: i64 },
    #[error("Engram hash layer {layer} is outside {layers} configured layers")]
    HashLayerOutOfRange { layer: usize, layers: usize },
    #[error("Engram session dimensions must be nonzero")]
    EmptyDimension,
    #[error("Engram session epsilon must be finite and positive")]
    InvalidEpsilon,
    #[error("Engram session gate clamp must be finite and positive")]
    InvalidGateClamp,
    #[error("Engram embedding width {embedding_width} is not grouped by 32")]
    EmbeddingWidthNotGrouped { embedding_width: usize },
    #[error("Engram WKV reduction {reduction} is not grouped by 32")]
    ReductionNotGrouped { reduction: usize },
    #[error("Engram {field} exceeds session bound with {elements} elements")]
    ElementLimit {
        field: &'static str,
        elements: usize,
    },
    #[error("Engram {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("Engram BF16 {field} is nonfinite at {index}")]
    NonFiniteBf16 { field: &'static str, index: usize },
    #[error("Engram FP8 {field} is nonfinite at {index}")]
    NonFiniteFp8 { field: &'static str, index: usize },
    #[error("Engram E8M0 {field} is nonfinite at {index}")]
    NonFiniteScale { field: &'static str, index: usize },
    #[error("Engram token ID {id} at {index} is outside the token map")]
    TokenIdOutOfRange { index: usize, id: i64 },
    #[error("Engram compressed token {id} at {index} is negative")]
    NegativeCompressedToken { index: usize, id: i64 },
    #[error("Engram call starts at {actual}, expected {expected}")]
    UnexpectedStart { actual: usize, expected: usize },
    #[error("Engram calls require at least one token")]
    EmptyChunk,
    #[error("Engram chunk ends at {end}, beyond capacity {capacity}")]
    ChunkExceedsCapacity { end: usize, capacity: usize },
    #[error("Engram shape overflowed for {field}")]
    ShapeOverflow { field: &'static str },
    #[error("could not allocate {elements} Engram {field} elements")]
    AllocationFailed {
        field: &'static str,
        elements: usize,
    },
    #[error(transparent)]
    Hash(#[from] EngramHashError),
    #[error(transparent)]
    Embedding(#[from] EngramEmbeddingError),
    #[error(transparent)]
    Activation(#[from] ActivationQuantError),
    #[error(transparent)]
    Wkv(#[from] Fp8LinearError),
    #[error(transparent)]
    Gate(#[from] EngramGateError),
}

fn select_hash_column(
    all: &[i64],
    positions: usize,
    layers: usize,
    columns: usize,
    hash_layer: usize,
) -> Result<Vec<i64>, EngramSessionError> {
    let row_width = checked_product(layers, columns, "hash row")?;
    let expected = checked_product(positions, row_width, "hash output")?;
    if all.len() != expected {
        return Err(EngramSessionError::Length {
            field: "hash output",
            actual: all.len(),
            expected,
        });
    }
    let start = checked_product(hash_layer, columns, "hash column start")?;
    let mut selected = reserved_vec(checked_product(positions, columns, "hash IDs")?, "hash IDs")?;
    for row in all.chunks_exact(row_width) {
        selected.extend_from_slice(&row[start..start + columns]);
    }
    Ok(selected)
}
