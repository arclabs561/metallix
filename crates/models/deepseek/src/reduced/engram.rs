//! Stateful bounded Engram composition over runtime token IDs and residuals.
//!
//! This module owns request-local hash history and immutable numerical operands.
//! It has no fixture, checkpoint, or expected-output dependency.

use std::sync::{Arc, OnceLock};

use thiserror::Error;

use crate::{
    engram::{
        CompressedToken, EngramHashError, EngramHashLayout, EngramHashState,
        embedding::{
            EngramEmbeddingError, EngramEmbeddingLayout, EngramRowSource,
            engram_embedding_bf16_from_source, engram_embedding_bf16_reference,
        },
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
/// Most token positions one step may process. Prefill longer prompts in
/// chunks. 128 keeps every later per-step stage within its own bound at V4.1
/// width (the embedding lookup's 2^20 elements admit 170 positions).
const MAX_ENGRAM_STEP_TOKENS: usize = 128;
/// Per-step buffer bound: [`MAX_ENGRAM_STEP_TOKENS`] rows of the widest
/// per-token buffer at V4.1 Flash, the WKV output `(4 + 1) * 5120`.
const MAX_STEP_ELEMENTS: usize = MAX_ENGRAM_STEP_TOKENS * 25_600;

/// Immutable layout and token-compression operands for one batch-one Engram request.
///
/// Cloning shares the token map rather than copying it.
#[derive(Clone, Debug)]
pub struct EngramSessionConfig {
    hash_layout: EngramHashLayout,
    token_map: Arc<[i64]>,
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
        // The embedding table is bounded where it is owned (see
        // `validate_weights`); a row-source table may be any size.
        if capacity > MAX_SESSION_ELEMENTS {
            return Err(EngramSessionError::ElementLimit {
                field: "history",
                elements: capacity,
            });
        }
        wkv_weight_elements(wkv_width, reduction, "WKV weight")?;
        Ok(Self {
            hash_layout,
            token_map: token_map.into(),
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

/// Immutable FP8/BF16 numerical operands for Engram sessions.
///
/// Cloning shares the operands: definitions build every session from one copy,
/// and the full-value finiteness scan runs once per set of operands.
#[derive(Clone, Debug)]
pub struct EngramSessionWeights(Arc<WeightsInner>);

#[derive(Debug)]
struct WeightsInner {
    /// Owned `(codes, scales)` table, or `None` when rows come from an
    /// [`EngramRowSource`] at each step.
    embedding: Option<(Vec<u8>, Vec<u8>)>,
    wkv_codes: Vec<u8>,
    wkv_scales: Vec<u8>,
    q_weight: Vec<u16>,
    k_weight: Vec<u16>,
    scan: OnceLock<Result<(), ScanFault>>,
}

/// The first nonfinite value found by the shared operand scan.
#[derive(Clone, Copy, Debug)]
enum ScanFault {
    Fp8(&'static str, usize),
    Scale(&'static str, usize),
    Bf16(&'static str, usize),
}

impl From<ScanFault> for EngramSessionError {
    fn from(fault: ScanFault) -> Self {
        match fault {
            ScanFault::Fp8(field, index) => Self::NonFiniteFp8 { field, index },
            ScanFault::Scale(field, index) => Self::NonFiniteScale { field, index },
            ScanFault::Bf16(field, index) => Self::NonFiniteBf16 { field, index },
        }
    }
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
        Self::build(
            Some((embedding_codes, embedding_scales)),
            wkv_codes,
            wkv_scales,
            q_weight,
            k_weight,
        )
    }

    /// Takes WKV and gate weights for sessions whose embedding rows come from
    /// an [`EngramRowSource`] passed to [`EngramSession::step_with`].
    #[must_use]
    pub fn without_embedding_table(
        wkv_codes: Vec<u8>,
        wkv_scales: Vec<u8>,
        q_weight: Vec<u16>,
        k_weight: Vec<u16>,
    ) -> Self {
        Self::build(None, wkv_codes, wkv_scales, q_weight, k_weight)
    }

    fn build(
        embedding: Option<(Vec<u8>, Vec<u8>)>,
        wkv_codes: Vec<u8>,
        wkv_scales: Vec<u8>,
        q_weight: Vec<u16>,
        k_weight: Vec<u16>,
    ) -> Self {
        Self(Arc::new(WeightsInner {
            embedding,
            wkv_codes,
            wkv_scales,
            q_weight,
            k_weight,
            scan: OnceLock::new(),
        }))
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
        let query_weights = bf16_weights_to_f32(&weights.0.q_weight, "q weight")?;
        let key_weights = bf16_weights_to_f32(&weights.0.k_weight, "k weight")?;
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
    ///
    /// Embedding rows come from the session's owned table; weights built
    /// [`EngramSessionWeights::without_embedding_table`] need [`Self::step_with`].
    pub fn step(
        &mut self,
        start: usize,
        token_ids: &[i64],
        residual: &[u16],
    ) -> Result<EngramStepOutput, EngramSessionError> {
        self.step_from(start, token_ids, residual, None)
    }

    /// Like [`Self::step`], with embedding rows read from `rows` after hashing
    /// selects them. Any owned table is not consulted. A source failure leaves
    /// history and the request cursor unchanged.
    pub fn step_with(
        &mut self,
        start: usize,
        token_ids: &[i64],
        residual: &[u16],
        rows: &dyn EngramRowSource,
    ) -> Result<EngramStepOutput, EngramSessionError> {
        self.step_from(start, token_ids, residual, Some(rows))
    }

    fn step_from(
        &mut self,
        start: usize,
        token_ids: &[i64],
        residual: &[u16],
        rows: Option<&dyn EngramRowSource>,
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
        if positions > MAX_ENGRAM_STEP_TOKENS {
            return Err(EngramSessionError::StepTooLong {
                positions,
                maximum: MAX_ENGRAM_STEP_TOKENS,
            });
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
        let expected_residual = step_product(
            step_product(positions, self.config.copies, "residual rows")?,
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
        let mut tokens = step_vec(positions, "compressed tokens")?;
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
        let embedding = self.embedding(&hash_ids, rows)?;
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

    fn embedding(
        &self,
        hash_ids: &[i64],
        rows: Option<&dyn EngramRowSource>,
    ) -> Result<Vec<u16>, EngramSessionError> {
        let elements = step_product(
            hash_ids.len(),
            self.config.embedding_width,
            "embedding output",
        )?;
        let mut output = step_vec(elements, "embedding output")?;
        output.resize(elements, 0);
        match (rows, &self.weights.0.embedding) {
            (Some(rows), _) => engram_embedding_bf16_from_source(
                hash_ids,
                self.config.embedding_rows,
                self.config.embedding_width,
                rows,
                &mut output,
            )?,
            (None, Some((codes, scales))) => engram_embedding_bf16_reference(
                hash_ids,
                codes,
                scales,
                EngramEmbeddingLayout::new(
                    self.config.embedding_rows,
                    self.config.embedding_width,
                    32,
                )?,
                &mut output,
            )?,
            (None, None) => return Err(EngramSessionError::MissingEmbeddingRows),
        }
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
        let activation_elements = step_product(positions, reduction, "WKV activations")?;
        if embedding.len() != activation_elements {
            return Err(EngramSessionError::Length {
                field: "embedding",
                actual: embedding.len(),
                expected: activation_elements,
            });
        }
        let mut codes = step_vec(activation_elements, "WKV activation codes")?;
        codes.resize(activation_elements, 0);
        let scale_elements = step_product(positions, reduction / 32, "WKV activation scales")?;
        let mut scales = step_vec(scale_elements, "WKV activation scales")?;
        scales.resize(scale_elements, 0);
        quantize_bf16_activations_e4m3fn(
            embedding,
            positions,
            reduction,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )?;
        let output_elements = step_product(positions, outputs, "WKV output")?;
        let mut projected = step_vec(output_elements, "WKV FP32 output")?;
        projected.resize(output_elements, 0.0);
        fp8_linear_runtime_f32(
            &codes,
            &scales,
            &self.weights.0.wkv_codes,
            &self.weights.0.wkv_scales,
            positions,
            reduction,
            outputs,
            ActivationGroup::Elements32,
            &mut projected,
        )?;
        let mut wkv = step_vec(output_elements, "WKV BF16 output")?;
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
        let expected = step_product(positions, row_width, "WKV output")?;
        if wkv.len() != expected {
            return Err(EngramSessionError::Length {
                field: "WKV output",
                actual: wkv.len(),
                expected,
            });
        }
        let mut key = step_vec(step_product(positions, key_width, "key")?, "key")?;
        let mut value = step_vec(
            step_product(positions, self.config.width, "value")?,
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
        let elements = step_product(
            step_product(positions, self.config.copies, "gate rows")?,
            self.config.width,
            "gate output",
        )?;
        let mut output = step_vec(elements, "gate output")?;
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
    let weights = &*weights.0;
    let columns = hash_columns(&config.hash_layout)?;
    let reduction = checked_product(columns, config.embedding_width, "WKV reduction")?;
    let outputs = wkv_width(config.copies, config.width)?;
    let wkv_elements = wkv_weight_elements(outputs, reduction, "WKV weights")?;
    let wkv_scale_elements = checked_product(outputs.div_ceil(32), reduction / 32, "WKV scales")?;
    let gate_elements = checked_product(config.copies, config.width, "gate weights")?;
    if let Some((codes, scales)) = &weights.embedding {
        let embedding_elements = checked_product(
            config.embedding_rows,
            config.embedding_width,
            "embedding weights",
        )?;
        check_length("embedding codes", codes.len(), embedding_elements)?;
        check_length("embedding scales", scales.len(), embedding_elements / 32)?;
    }
    check_length("WKV codes", weights.wkv_codes.len(), wkv_elements)?;
    check_length("WKV scales", weights.wkv_scales.len(), wkv_scale_elements)?;
    check_length("q weight", weights.q_weight.len(), gate_elements)?;
    check_length("k weight", weights.k_weight.len(), gate_elements)?;
    // Lengths depend on the config; values do not, so shared operands are
    // scanned once however many sessions they back.
    (*weights.scan.get_or_init(|| scan_values(weights)))?;
    Ok(())
}

fn check_length(
    field: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), EngramSessionError> {
    if actual == expected {
        Ok(())
    } else {
        Err(EngramSessionError::Length {
            field,
            actual,
            expected,
        })
    }
}

fn scan_values(weights: &WeightsInner) -> Result<(), ScanFault> {
    let embedding = weights.embedding.as_ref();
    let empty: &[u8] = &[];
    for (field, codes) in [
        (
            "embedding codes",
            embedding.map_or(empty, |(codes, _)| codes.as_slice()),
        ),
        ("WKV codes", weights.wkv_codes.as_slice()),
    ] {
        if let Some(index) = codes
            .iter()
            .position(|&code| !decode_e4m3fn(code).is_finite())
        {
            return Err(ScanFault::Fp8(field, index));
        }
    }
    for (field, scales) in [
        (
            "embedding scales",
            embedding.map_or(empty, |(_, scales)| scales.as_slice()),
        ),
        ("WKV scales", weights.wkv_scales.as_slice()),
    ] {
        if let Some(index) = scales
            .iter()
            .position(|&scale| !decode_e8m0(scale).is_finite())
        {
            return Err(ScanFault::Scale(field, index));
        }
    }
    for (field, values) in [
        ("q weight", &weights.q_weight),
        ("k weight", &weights.k_weight),
    ] {
        if let Some(index) = values
            .iter()
            .position(|&bits| !bf16_to_f32(bits).is_finite())
        {
            return Err(ScanFault::Bf16(field, index));
        }
    }
    Ok(())
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

/// The static WKV weight is caller-owned and only length-checked, so its size
/// is not capped; every per-step buffer and the work a step does are bounded
/// through [`step_product`] (the projection output is `positions x outputs`).
fn wkv_weight_elements(
    outputs: usize,
    reduction: usize,
    field: &'static str,
) -> Result<usize, EngramSessionError> {
    outputs
        .checked_mul(reduction)
        .ok_or(EngramSessionError::ShapeOverflow { field })
}

/// Like [`checked_product`] for a buffer sized by one step's positions.
fn step_product(
    positions_or_rows: usize,
    width: usize,
    field: &'static str,
) -> Result<usize, EngramSessionError> {
    match positions_or_rows.checked_mul(width) {
        Some(elements) if elements <= MAX_STEP_ELEMENTS => Ok(elements),
        Some(elements) => Err(EngramSessionError::ElementLimit { field, elements }),
        None => Err(EngramSessionError::ShapeOverflow { field }),
    }
}

/// Like [`reserved_vec`] for a buffer sized by one step's positions.
fn step_vec<T>(elements: usize, field: &'static str) -> Result<Vec<T>, EngramSessionError> {
    if elements > MAX_STEP_ELEMENTS {
        return Err(EngramSessionError::ElementLimit { field, elements });
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| EngramSessionError::AllocationFailed { field, elements })?;
    Ok(values)
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
    #[error("Engram session has no owned embedding table; step with a row source")]
    MissingEmbeddingRows,
    #[error("Engram chunk ends at {end}, beyond capacity {capacity}")]
    ChunkExceedsCapacity { end: usize, capacity: usize },
    #[error("Engram step of {positions} positions exceeds {maximum}; prefill in chunks")]
    StepTooLong { positions: usize, maximum: usize },
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
    let expected = step_product(positions, row_width, "hash output")?;
    if all.len() != expected {
        return Err(EngramSessionError::Length {
            field: "hash output",
            actual: all.len(),
            expected,
        });
    }
    let start = checked_product(hash_layer, columns, "hash column start")?;
    let mut selected = step_vec(step_product(positions, columns, "hash IDs")?, "hash IDs")?;
    for row in all.chunks_exact(row_width) {
        selected.extend_from_slice(&row[start..start + columns]);
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{EngramSession, EngramSessionConfig, EngramSessionError, EngramSessionWeights};
    use crate::engram::{
        EngramHashLayout,
        embedding::{EngramEmbeddingError, EngramRowSource},
    };

    /// A config with V4.1 Flash Engram geometry (4-grams, 8 heads, 256-wide
    /// rows, 4 HC copies) at hidden width `width`.
    fn v41_config(width: usize) -> Result<EngramSessionConfig, EngramSessionError> {
        let hash_layout = EngramHashLayout::new(4, 8, 1, 2, vec![3; 24], vec![0; 24], vec![1; 4])
            .expect("hash layout");
        EngramSessionConfig::new(
            hash_layout,
            vec![0, 1],
            0,
            128,
            4,
            width,
            384_006_168,
            256,
            1e-20,
            1e-6,
        )
    }

    /// Every row is zero codes with unit scales.
    struct ZeroRows;

    impl EngramRowSource for ZeroRows {
        fn read_rows(
            &self,
            _: &[usize],
            codes: &mut [u8],
            scales: &mut [u8],
        ) -> Result<(), EngramEmbeddingError> {
            codes.fill(0);
            scales.fill(127);
            Ok(())
        }
    }

    #[test]
    fn real_v41_wkv_is_admitted_and_each_step_is_bounded_by_its_token_count() {
        // (4 + 1) * 5120 = 25600 outputs over 24 * 256 = 6144 reductions.
        let config = v41_config(5120).expect("real V4.1 WKV weight");
        let weights = EngramSessionWeights::without_embedding_table(
            vec![0; 25_600 * 6_144],
            vec![127; 800 * 192],
            vec![0x3f80; 4 * 5120],
            vec![0x3f80; 4 * 5120],
        );
        let mut session = EngramSession::new(config, weights).expect("real-size session");
        let step = |session: &mut EngramSession, positions: usize| {
            session.step_with(
                0,
                &vec![1; positions],
                &vec![0x3f80; positions * 4 * 5120],
                &ZeroRows,
            )
        };
        // 129 positions exceed the per-step token bound before any work.
        assert!(matches!(
            step(&mut session, 129),
            Err(EngramSessionError::StepTooLong {
                positions: 129,
                maximum: 128,
            })
        ));
        assert_eq!(session.next_start(), 0, "a rejected step commits nothing");
        // The bound admits the widest per-step buffer at that token count.
        assert_eq!(
            super::step_product(128, 25_600, "WKV output").expect("128 positions"),
            3_276_800
        );
        assert!(matches!(
            super::step_product(129, 25_600, "WKV output"),
            Err(EngramSessionError::ElementLimit {
                field: "WKV output",
                elements: 3_302_400,
            })
        ));
    }

    #[test]
    fn sessions_share_definition_operands_instead_of_copying_them() {
        let hash_layout =
            EngramHashLayout::new(2, 1, 1, 1, vec![3], vec![0], vec![1, 1]).expect("hash layout");
        let config =
            EngramSessionConfig::new(hash_layout, vec![0, 1], 0, 4, 1, 32, 2, 32, 1.0e-6, 1.0e-6)
                .expect("config");
        let weights = EngramSessionWeights::new(
            vec![0x38; 64],
            vec![127; 2],
            vec![0; 64 * 32],
            vec![127; 2],
            vec![0x3f80; 32],
            vec![0x3f80; 32],
        );
        let first = EngramSession::new(config.clone(), weights.clone()).expect("first session");
        let second = EngramSession::new(config.clone(), weights.clone()).expect("second session");
        for session in [&first, &second] {
            assert!(Arc::ptr_eq(&session.weights.0, &weights.0));
            assert!(Arc::ptr_eq(&session.config.token_map, &config.token_map));
        }
        assert!(matches!(weights.0.scan.get(), Some(Ok(()))));
    }
}
