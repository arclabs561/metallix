//! Bounded F32 `ModernBERT` blocks and 22-layer prefill for the pinned Julia-1 encoder.
//!
//! The full prefill accepts caller-owned selected embedding rows and all layer
//! weights. It does not load a checkpoint or serve ordinary typed requests.

use crate::head::{ATTENTION_HEADS, HEAD_WIDTH, WIDTH};
use thiserror::Error;

pub const ENCODER_FF_WIDTH: usize = 1_152;
const EPSILON: f32 = 1e-5;
const ROPE_THETA: f32 = 160_000.0;
const WIDTH_F32: f32 = 384.0;
const HEAD_WIDTH_F32: f32 = 64.0;
const MAX_POSITIONS: usize = 126;
const MAX_WORK: usize = 256_000_000;
const FULL_ENCODER_LAYERS: usize = 22;
const MAX_PREFILL_POSITIONS: usize = 8;
const MAX_SELECTED_ROWS: usize = MAX_PREFILL_POSITIONS;
const MAX_FULL_ENCODER_WORK: usize = 384_000_000;
const PUBLISHED_VOCAB_SIZE: u64 = 256_000;

/// Raw parameters in the same row-major layout as `PyTorch` `nn.Linear` weights.
#[derive(Clone, Debug)]
pub struct EncoderBlockWeights {
    pub wqkv_weight: Vec<f32>,
    pub wo_weight: Vec<f32>,
    pub wi_weight: Vec<f32>,
    pub wo_mlp_weight: Vec<f32>,
    pub attn_norm_weight: Vec<f32>,
    pub mlp_norm_weight: Vec<f32>,
}

/// Caller-supplied selected embedding rows and all weights for the 22-layer encoder.
#[derive(Clone, Debug)]
pub struct FullEncoderWeights {
    /// Strictly increasing token IDs corresponding to `token_rows`.
    pub token_ids: Vec<u64>,
    /// Row-major F32 embedding rows, one row for every `token_ids` entry.
    pub token_rows: Vec<f32>,
    pub embedding_norm_weight: Vec<f32>,
    pub layers: Vec<EncoderBlockWeights>,
    pub final_norm_weight: Vec<f32>,
}

/// Token IDs and a padding mask for one bounded, already serialized sequence.
#[derive(Clone, Debug)]
pub struct EncoderInput {
    pub input_ids: Vec<u64>,
    pub attention_mask: Vec<bool>,
}

/// One unbatched encoder sequence.  `layer` selects the pinned global/local regime.
#[derive(Clone, Debug)]
pub struct EncoderBlockInput {
    pub hidden: Vec<f32>,
    pub positions: usize,
    pub attention_mask: Vec<bool>,
    pub layer: usize,
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum JuliaEncoderError {
    #[error("Julia encoder positions must be 1..={MAX_POSITIONS}, got {0}")]
    Positions(usize),
    #[error("Julia encoder layer must be below 22, got {0}")]
    Layer(usize),
    #[error("Julia encoder {field} length is {actual}, expected {expected}")]
    Length {
        field: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("Julia encoder {field} contains a non-finite value at {index}")]
    NonFinite { field: &'static str, index: usize },
    #[error("Julia encoder has no unmasked key positions")]
    NoKeys,
    #[error("Julia encoder affine/attention MAC count exceeds {MAX_WORK}")]
    Work,
    #[error("Julia full encoder requires exactly {FULL_ENCODER_LAYERS} layers, got {0}")]
    FullLayers(usize),
    #[error("Julia full encoder selected rows must be 1..={MAX_SELECTED_ROWS}, got {0}")]
    SelectedRows(usize),
    #[error("Julia full encoder token IDs must be strictly increasing")]
    TokenIds,
    #[error("Julia full encoder has no selected row for token ID {0}")]
    TokenId(u64),
    #[error("Julia full encoder positions must be 1..={MAX_PREFILL_POSITIONS}, got {0}")]
    PrefillPositions(usize),
    #[error("Julia full encoder affine/attention MAC count exceeds {MAX_FULL_ENCODER_WORK}")]
    FullWork,
    #[error("Julia full encoder token ID {0} is outside the published vocabulary")]
    VocabularyId(u64),
}

/// Validated weights for one `ModernBERT` `ModernBertEncoderLayer`.
pub struct EncoderBlock {
    weights: EncoderBlockWeights,
}

struct AttentionTrace {
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}

pub(crate) struct Layer0Trace {
    pub qkv: Vec<f32>,
    pub rotated_query: Vec<f32>,
    pub rotated_key: Vec<f32>,
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub attended: Vec<f32>,
    pub post_wo_residual: Vec<f32>,
}

impl EncoderBlock {
    pub fn new(weights: EncoderBlockWeights) -> Result<Self, JuliaEncoderError> {
        validate_weights(&weights)?;
        Ok(Self { weights })
    }

    /// Executes `attn_norm -> attention + residual -> mlp_norm -> GEGLU + residual`.
    ///
    /// Layer zero has the source `Identity` attention norm. Layers divisible by
    /// three use full attention; the rest use the source's symmetric +/-64 window.
    pub fn forward(&self, input: &EncoderBlockInput) -> Result<Vec<f32>, JuliaEncoderError> {
        self.forward_inner(input, None)
    }

    #[cfg(test)]
    pub(crate) fn forward_layer0_trace(
        &self,
        input: &EncoderBlockInput,
    ) -> Result<Layer0Trace, JuliaEncoderError> {
        if input.layer != 0 {
            return Err(JuliaEncoderError::Layer(input.layer));
        }
        let mut trace = Layer0Trace {
            qkv: Vec::new(),
            rotated_query: Vec::new(),
            rotated_key: Vec::new(),
            logits: Vec::new(),
            probabilities: Vec::new(),
            attended: Vec::new(),
            post_wo_residual: Vec::new(),
        };
        self.forward_inner(input, Some(&mut trace))?;
        Ok(trace)
    }

    fn forward_inner(
        &self,
        input: &EncoderBlockInput,
        mut trace: Option<&mut Layer0Trace>,
    ) -> Result<Vec<f32>, JuliaEncoderError> {
        validate_input(input)?;
        if !input.attention_mask.iter().any(|&x| x) {
            return Err(JuliaEncoderError::NoKeys);
        }
        let affine = WIDTH
            .checked_mul(4 * WIDTH + 3 * ENCODER_FF_WIDTH)
            .ok_or(JuliaEncoderError::Work)?;
        let attention_work = input
            .positions
            .checked_mul(input.positions)
            .and_then(|x| x.checked_mul(2 * WIDTH))
            .ok_or(JuliaEncoderError::Work)?;
        if input
            .positions
            .checked_mul(affine)
            .and_then(|x| x.checked_add(attention_work))
            .ok_or(JuliaEncoderError::Work)?
            > MAX_WORK
        {
            return Err(JuliaEncoderError::Work);
        }
        let attn_input = if input.layer == 0 {
            input.hidden.clone()
        } else {
            norm_rows(
                &input.hidden,
                &self.weights.attn_norm_weight,
                "attention norm",
            )?
        };
        let qkv = linear(
            &attn_input,
            input.positions,
            WIDTH,
            3 * WIDTH,
            &self.weights.wqkv_weight,
            "Wqkv",
        )?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.qkv.clone_from(&qkv);
            trace.rotated_query = rotated_part(&qkv, input.positions, 0);
            trace.rotated_key = rotated_part(&qkv, input.positions, 1);
        }
        let mut attention_trace = trace.as_deref_mut().map(|_| AttentionTrace {
            logits: Vec::new(),
            probabilities: Vec::new(),
        });
        let attended = attention(
            &qkv,
            input.positions,
            &input.attention_mask,
            input.layer,
            attention_trace.as_mut(),
        )?;
        if let (Some(trace), Some(attention_trace)) = (trace.as_deref_mut(), attention_trace) {
            trace.logits = attention_trace.logits;
            trace.probabilities = attention_trace.probabilities;
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.attended.clone_from(&attended);
        }
        let attended = linear(
            &attended,
            input.positions,
            WIDTH,
            WIDTH,
            &self.weights.wo_weight,
            "attention Wo",
        )?;
        let mut residual = input.hidden.clone();
        add_assign(&mut residual, &attended, "attention residual")?;
        if let Some(trace) = trace {
            trace.post_wo_residual.clone_from(&residual);
        }
        let normalized = norm_rows(&residual, &self.weights.mlp_norm_weight, "MLP norm")?;
        let wi = linear(
            &normalized,
            input.positions,
            WIDTH,
            2 * ENCODER_FF_WIDTH,
            &self.weights.wi_weight,
            "MLP Wi",
        )?;
        let mut gated = vec![0.0; input.positions * ENCODER_FF_WIDTH];
        for row in 0..input.positions {
            for col in 0..ENCODER_FF_WIDTH {
                let base = row * 2 * ENCODER_FF_WIDTH + col;
                gated[row * ENCODER_FF_WIDTH + col] = gelu(wi[base]) * wi[base + ENCODER_FF_WIDTH];
            }
        }
        let output = linear(
            &gated,
            input.positions,
            ENCODER_FF_WIDTH,
            WIDTH,
            &self.weights.wo_mlp_weight,
            "MLP Wo",
        )?;
        add_assign(&mut residual, &output, "MLP residual")?;
        Ok(residual)
    }
}

/// A bounded CPU `ModernBERT` encoder with selected token rows, not a checkpoint loader.
pub struct JuliaEncoder {
    token_ids: Vec<u64>,
    token_rows: Vec<f32>,
    embedding_norm_weight: Vec<f32>,
    layers: Vec<EncoderBlock>,
    final_norm_weight: Vec<f32>,
}

impl JuliaEncoder {
    pub fn new(weights: FullEncoderWeights) -> Result<Self, JuliaEncoderError> {
        if weights.token_ids.is_empty() || weights.token_ids.len() > MAX_SELECTED_ROWS {
            return Err(JuliaEncoderError::SelectedRows(weights.token_ids.len()));
        }
        if weights.token_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(JuliaEncoderError::TokenIds);
        }
        if let Some(&token_id) = weights
            .token_ids
            .iter()
            .find(|&&token_id| token_id >= PUBLISHED_VOCAB_SIZE)
        {
            return Err(JuliaEncoderError::VocabularyId(token_id));
        }
        length(
            "selected token rows",
            &weights.token_rows,
            weights.token_ids.len() * WIDTH,
        )?;
        length(
            "embedding norm weight",
            &weights.embedding_norm_weight,
            WIDTH,
        )?;
        length("final norm weight", &weights.final_norm_weight, WIDTH)?;
        if weights.layers.len() != FULL_ENCODER_LAYERS {
            return Err(JuliaEncoderError::FullLayers(weights.layers.len()));
        }
        let mut layers = Vec::with_capacity(FULL_ENCODER_LAYERS);
        for layer in weights.layers {
            layers.push(EncoderBlock::new(layer)?);
        }
        Ok(Self {
            token_ids: weights.token_ids,
            token_rows: weights.token_rows,
            embedding_norm_weight: weights.embedding_norm_weight,
            layers,
            final_norm_weight: weights.final_norm_weight,
        })
    }

    /// Looks up supplied rows, applies embedding/final normalization, and runs layers 0 through 21.
    ///
    /// The MAC bound counts affine projections and both attention reductions;
    /// row lookup and normalization reductions are deliberately outside that count.
    pub fn forward(&self, input: &EncoderInput) -> Result<Vec<f32>, JuliaEncoderError> {
        self.forward_inner(input, None)
    }

    #[cfg(test)]
    pub(crate) fn forward_boundaries(
        &self,
        input: &EncoderInput,
    ) -> Result<Vec<Vec<f32>>, JuliaEncoderError> {
        let mut boundaries = Vec::with_capacity(FULL_ENCODER_LAYERS + 2);
        self.forward_inner(input, Some(&mut boundaries))?;
        Ok(boundaries)
    }

    fn forward_inner(
        &self,
        input: &EncoderInput,
        boundaries: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<f32>, JuliaEncoderError> {
        let positions = Self::validate_prefill_input(input)?;
        let hidden = norm_rows(
            &self.lookup_rows(input)?,
            &self.embedding_norm_weight,
            "embedding norm",
        )?;
        self.forward_from_embedding_inner(input, positions, hidden, boundaries)
    }

    #[cfg(test)]
    pub(crate) fn forward_boundaries_from_embedding(
        &self,
        input: &EncoderInput,
        embedding: Vec<f32>,
    ) -> Result<Vec<Vec<f32>>, JuliaEncoderError> {
        let positions = Self::validate_prefill_input(input)?;
        // Keep test-only injection subject to the same selected-token admission
        // as the public prefill path, even though it supplies the post-lookup rows.
        self.lookup_rows(input)?;
        let mut boundaries = Vec::with_capacity(FULL_ENCODER_LAYERS + 2);
        self.forward_from_embedding_inner(input, positions, embedding, Some(&mut boundaries))?;
        Ok(boundaries)
    }

    fn validate_prefill_input(input: &EncoderInput) -> Result<usize, JuliaEncoderError> {
        let positions = input.input_ids.len();
        if positions == 0 || positions > MAX_PREFILL_POSITIONS {
            return Err(JuliaEncoderError::PrefillPositions(positions));
        }
        if input.attention_mask.len() != positions {
            return Err(JuliaEncoderError::Length {
                field: "prefill attention mask",
                actual: input.attention_mask.len(),
                expected: positions,
            });
        }
        if !input.attention_mask.iter().any(|&value| value) {
            return Err(JuliaEncoderError::NoKeys);
        }
        let per_layer = positions
            .checked_mul(WIDTH * (4 * WIDTH + 3 * ENCODER_FF_WIDTH))
            .and_then(|value| value.checked_add(positions * positions * 2 * WIDTH))
            .ok_or(JuliaEncoderError::FullWork)?;
        if per_layer
            .checked_mul(FULL_ENCODER_LAYERS)
            .ok_or(JuliaEncoderError::FullWork)?
            > MAX_FULL_ENCODER_WORK
        {
            return Err(JuliaEncoderError::FullWork);
        }
        Ok(positions)
    }

    fn forward_from_embedding_inner(
        &self,
        input: &EncoderInput,
        positions: usize,
        mut hidden: Vec<f32>,
        mut boundaries: Option<&mut Vec<Vec<f32>>>,
    ) -> Result<Vec<f32>, JuliaEncoderError> {
        length("prefill embedding", &hidden, positions * WIDTH)?;
        record_boundary(&mut boundaries, &hidden);
        for (layer, block) in self.layers.iter().enumerate() {
            hidden = block.forward(&EncoderBlockInput {
                hidden,
                positions,
                attention_mask: input.attention_mask.clone(),
                layer,
            })?;
            record_boundary(&mut boundaries, &hidden);
        }
        hidden = norm_rows(&hidden, &self.final_norm_weight, "final norm")?;
        record_boundary(&mut boundaries, &hidden);
        Ok(hidden)
    }

    fn lookup_rows(&self, input: &EncoderInput) -> Result<Vec<f32>, JuliaEncoderError> {
        let mut hidden = Vec::with_capacity(input.input_ids.len() * WIDTH);
        for &token_id in &input.input_ids {
            let row = self
                .token_ids
                .binary_search(&token_id)
                .map_err(|_| JuliaEncoderError::TokenId(token_id))?;
            hidden.extend_from_slice(&self.token_rows[row * WIDTH..(row + 1) * WIDTH]);
        }
        Ok(hidden)
    }

    #[cfg(test)]
    pub(crate) fn lookup_rows_for_trace(
        &self,
        input: &EncoderInput,
    ) -> Result<Vec<f32>, JuliaEncoderError> {
        self.lookup_rows(input)
    }
}

fn record_boundary(boundaries: &mut Option<&mut Vec<Vec<f32>>>, hidden: &[f32]) {
    if let Some(boundaries) = boundaries.as_deref_mut() {
        boundaries.push(hidden.to_vec());
    }
}

fn attention(
    qkv: &[f32],
    positions: usize,
    mask: &[bool],
    layer: usize,
    mut trace: Option<&mut AttentionTrace>,
) -> Result<Vec<f32>, JuliaEncoderError> {
    let mut out = vec![0.0; positions * WIDTH];
    let scale = HEAD_WIDTH_F32.sqrt().recip();
    let local = !layer.is_multiple_of(3);
    for query in 0..positions {
        for head in 0..ATTENTION_HEADS {
            // Transformers builds its additive mask with `torch.finfo(f32).min`,
            // rather than negative infinity.  In the unusual all-masked local
            // query case this deliberately produces a finite uniform softmax.
            let mut logits = vec![f32::MIN; positions];
            let mut maximum = f32::MIN;
            for key in 0..positions {
                if !mask[key] || (local && query.abs_diff(key) > 64) {
                    continue;
                }
                let mut dot = 0.0;
                for dim in 0..HEAD_WIDTH {
                    let q = rope(qkv[qkv_index(query, 0, head, dim)], dim, query);
                    let k = rope(qkv[qkv_index(key, 1, head, dim)], dim, key);
                    // rotate_half needs the matching other half; replace values below.
                    let q = q + rope_rotation(qkv, query, 0, head, dim);
                    let k = k + rope_rotation(qkv, key, 1, head, dim);
                    dot += q * k;
                }
                logits[key] = dot * scale;
                maximum = maximum.max(logits[key]);
            }
            let denominator: f32 = logits.iter().map(|x| (*x - maximum).exp()).sum();
            if !denominator.is_finite() || denominator == 0.0 {
                return Err(JuliaEncoderError::NonFinite {
                    field: "attention softmax",
                    index: query,
                });
            }
            let probabilities: Vec<f32> = logits
                .iter()
                .map(|logit| (*logit - maximum).exp() / denominator)
                .collect();
            if let Some(trace) = trace.as_deref_mut() {
                trace.logits.extend_from_slice(&logits);
                trace.probabilities.extend_from_slice(&probabilities);
            }
            for dim in 0..HEAD_WIDTH {
                let mut value = 0.0;
                for key in 0..positions {
                    value += probabilities[key] * qkv[qkv_index(key, 2, head, dim)];
                }
                out[query * WIDTH + head * HEAD_WIDTH + dim] = value;
            }
        }
    }
    Ok(out)
}

fn qkv_index(position: usize, part: usize, head: usize, dim: usize) -> usize {
    position * 3 * WIDTH + part * WIDTH + head * HEAD_WIDTH + dim
}

fn rotated_part(qkv: &[f32], positions: usize, part: usize) -> Vec<f32> {
    let mut rotated = vec![0.0; positions * WIDTH];
    for position in 0..positions {
        for head in 0..ATTENTION_HEADS {
            for dim in 0..HEAD_WIDTH {
                let value = rope(qkv[qkv_index(position, part, head, dim)], dim, position)
                    + rope_rotation(qkv, position, part, head, dim);
                rotated[position * WIDTH + head * HEAD_WIDTH + dim] = value;
            }
        }
    }
    rotated
}

// The source creates cos/sin in F32 and applies split-half rotation to Q/K.
fn rope(value: f32, dim: usize, position: usize) -> f32 {
    let half_dim = dim % (HEAD_WIDTH / 2);
    let frequency = ROPE_THETA.powf(-(2.0 * small_index(half_dim)) / HEAD_WIDTH_F32);
    value * (small_index(position) * frequency).cos()
}
fn rope_rotation(qkv: &[f32], position: usize, part: usize, head: usize, dim: usize) -> f32 {
    let half = HEAD_WIDTH / 2;
    let source = if dim < half {
        -qkv[qkv_index(position, part, head, dim + half)]
    } else {
        qkv[qkv_index(position, part, head, dim - half)]
    };
    let half_dim = dim % half;
    let frequency = ROPE_THETA.powf(-(2.0 * small_index(half_dim)) / HEAD_WIDTH_F32);
    source * (small_index(position) * frequency).sin()
}

fn small_index(value: usize) -> f32 {
    // Calls use dimensions <=64 and validated positions <=126.
    f32::from(u8::try_from(value).expect("bounded encoder index"))
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + libm::erff(x / 2.0_f32.sqrt()))
}
fn add_assign(a: &mut [f32], b: &[f32], field: &'static str) -> Result<(), JuliaEncoderError> {
    for (index, (left, right)) in a.iter_mut().zip(b).enumerate() {
        *left += right;
        if !left.is_finite() {
            return Err(JuliaEncoderError::NonFinite { field, index });
        }
    }
    Ok(())
}
fn norm_rows(
    value: &[f32],
    weight: &[f32],
    field: &'static str,
) -> Result<Vec<f32>, JuliaEncoderError> {
    let mut out = Vec::with_capacity(value.len());
    for (row_index, row) in value.chunks_exact(WIDTH).enumerate() {
        let mean = row.iter().sum::<f32>() / WIDTH_F32;
        let variance = row
            .iter()
            .map(|x| {
                let d = *x - mean;
                d * d
            })
            .sum::<f32>()
            / WIDTH_F32;
        let inverse = (variance + EPSILON).sqrt().recip();
        if !mean.is_finite() || !variance.is_finite() || !inverse.is_finite() {
            return Err(JuliaEncoderError::NonFinite {
                field,
                index: row_index,
            });
        }
        for (column, (x, w)) in row.iter().zip(weight).enumerate() {
            let normalized = (*x - mean) * inverse * *w;
            if !normalized.is_finite() {
                return Err(JuliaEncoderError::NonFinite {
                    field,
                    index: row_index * WIDTH + column,
                });
            }
            out.push(normalized);
        }
    }
    Ok(out)
}
fn linear(
    input: &[f32],
    rows: usize,
    reduction: usize,
    outputs: usize,
    weight: &[f32],
    field: &'static str,
) -> Result<Vec<f32>, JuliaEncoderError> {
    let mut out = vec![0.0; rows * outputs];
    for row in 0..rows {
        for output in 0..outputs {
            let mut sum = 0.0;
            for col in 0..reduction {
                sum += input[row * reduction + col] * weight[output * reduction + col];
            }
            if !sum.is_finite() {
                return Err(JuliaEncoderError::NonFinite {
                    field,
                    index: row * outputs + output,
                });
            }
            out[row * outputs + output] = sum;
        }
    }
    Ok(out)
}
fn check(field: &'static str, value: &[f32]) -> Result<(), JuliaEncoderError> {
    for (index, x) in value.iter().enumerate() {
        if !x.is_finite() {
            return Err(JuliaEncoderError::NonFinite { field, index });
        }
    }
    Ok(())
}
fn length(field: &'static str, value: &[f32], expected: usize) -> Result<(), JuliaEncoderError> {
    if value.len() != expected {
        return Err(JuliaEncoderError::Length {
            field,
            actual: value.len(),
            expected,
        });
    }
    check(field, value)
}
fn validate_weights(w: &EncoderBlockWeights) -> Result<(), JuliaEncoderError> {
    length("Wqkv weight", &w.wqkv_weight, 3 * WIDTH * WIDTH)?;
    length("attention Wo weight", &w.wo_weight, WIDTH * WIDTH)?;
    length("MLP Wi weight", &w.wi_weight, 2 * ENCODER_FF_WIDTH * WIDTH)?;
    length("MLP Wo weight", &w.wo_mlp_weight, WIDTH * ENCODER_FF_WIDTH)?;
    length("attention norm weight", &w.attn_norm_weight, WIDTH)?;
    length("MLP norm weight", &w.mlp_norm_weight, WIDTH)
}
fn validate_input(i: &EncoderBlockInput) -> Result<(), JuliaEncoderError> {
    if i.positions == 0 || i.positions > MAX_POSITIONS {
        return Err(JuliaEncoderError::Positions(i.positions));
    }
    if i.layer >= 22 {
        return Err(JuliaEncoderError::Layer(i.layer));
    }
    length("hidden", &i.hidden, i.positions * WIDTH)?;
    if i.attention_mask.len() != i.positions {
        return Err(JuliaEncoderError::Length {
            field: "attention mask",
            actual: i.attention_mask.len(),
            expected: i.positions,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn values(length: usize, ordinal: usize) -> Vec<f32> {
        (0..length)
            .map(|index| {
                (f32::from(u8::try_from((index + ordinal * 17) % 97).unwrap()) - 48.0) / 1000.0
            })
            .collect()
    }
    fn block() -> EncoderBlock {
        EncoderBlock::new(EncoderBlockWeights {
            wqkv_weight: values(3 * WIDTH * WIDTH, 0),
            wo_weight: values(WIDTH * WIDTH, 1),
            wi_weight: values(2 * ENCODER_FF_WIDTH * WIDTH, 2),
            wo_mlp_weight: values(WIDTH * ENCODER_FF_WIDTH, 3),
            attn_norm_weight: values(WIDTH, 4),
            mlp_norm_weight: values(WIDTH, 5),
        })
        .unwrap()
    }
    fn input(positions: usize, layer: usize) -> EncoderBlockInput {
        EncoderBlockInput {
            hidden: (0..positions * WIDTH)
                .map(|index| (f32::from(u8::try_from((index * 7) % 29).unwrap()) - 14.0) / 20.0)
                .collect(),
            positions,
            attention_mask: vec![true; positions],
            layer,
        }
    }

    fn frozen_case(name: &str) -> Value {
        serde_json::from_str::<Value>(include_str!(
            "../../../../fixtures/julia-1/encoder-reference.json"
        ))
        .unwrap()["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("missing fixture case {name}"))
            .clone()
    }

    #[test]
    fn matches_all_frozen_pinned_source_outputs() {
        for name in [
            "global_padding",
            "global_padding_perturbed",
            "global_unmasked_control",
            "local_window_crossing",
            "global_window_control",
            "local_distant_perturbation",
            "global_distant_perturbation",
            "local_all_masked_query",
        ] {
            let case = frozen_case(name);
            let positions = usize::try_from(case["positions"].as_u64().unwrap()).unwrap();
            let layer = usize::try_from(case["layer"].as_u64().unwrap()).unwrap();
            let mut source_input = input(positions, layer);
            source_input.attention_mask = case["attention_mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_bool().unwrap())
                .collect();
            if case["perturb_padding"].as_bool() == Some(true) {
                let start = (positions - 1) * WIDTH;
                for (index, value) in source_input.hidden[start..].iter_mut().enumerate() {
                    *value += (f32::from(u8::try_from((index * 11) % 31).unwrap()) - 15.0) / 3.0;
                }
            }
            let actual = block().forward(&source_input).unwrap();
            let expected = case["expected_output"].as_array().unwrap();
            assert!(actual.len() >= expected.len() * WIDTH);
            for (index, (value, expected)) in actual
                .iter()
                .take(expected.len() * WIDTH)
                .zip(expected.iter().flat_map(|row| row.as_array().unwrap()))
                .enumerate()
            {
                let expected = serde_json::from_value::<f32>(expected.clone()).unwrap();
                assert!(
                    (value - expected).abs() <= 1e-5,
                    "{name}[{index}]: {value} != {expected}"
                );
            }
        }
    }

    #[test]
    fn masked_padding_does_not_change_real_rows() {
        let model = block();
        let mut base = input(8, 0);
        base.attention_mask[6..].fill(false);
        let expected = model.forward(&base).unwrap();
        let mut perturbed = base.clone();
        for (index, value) in perturbed.hidden[6 * WIDTH..].iter_mut().enumerate() {
            *value += (f32::from(u8::try_from((index * 11) % 31).unwrap()) - 15.0) / 3.0;
        }
        let actual = model.forward(&perturbed).unwrap();
        assert_eq!(&expected[..6 * WIDTH], &actual[..6 * WIDTH]);
        perturbed.attention_mask.fill(true);
        let control = model.forward(&perturbed).unwrap();
        assert_ne!(&expected[..6 * WIDTH], &control[..6 * WIDTH]);
    }

    #[test]
    fn local_layer_handles_a_window_crossing_sequence() {
        let model = block();
        let local = model.forward(&input(66, 1)).unwrap();
        let global = model.forward(&input(66, 3)).unwrap();
        assert!(local.iter().all(|value| value.is_finite()));
        assert_ne!(
            local, global,
            "local +/-64 mask must affect a 66-token sequence"
        );
    }

    #[test]
    fn rejects_bad_shapes_and_absent_keys() {
        let model = block();
        let mut bad = input(1, 22);
        assert_eq!(model.forward(&bad), Err(JuliaEncoderError::Layer(22)));
        bad.layer = 0;
        bad.attention_mask[0] = false;
        assert_eq!(model.forward(&bad), Err(JuliaEncoderError::NoKeys));
    }
}
