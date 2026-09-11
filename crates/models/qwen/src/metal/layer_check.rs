//! Synchronous one-block loading experiment; the resident decoder is the control.

use std::{collections::HashMap, path::Path, time::Instant};

use mlx_rs::Array;

use super::{Qwen3MetalLoadError, Qwen3MlxWeights, compare_tensor_values, decode_bf16};
use crate::{checkpoint::Qwen3CheckpointInspection, forward::Qwen3ForwardConfig};

const LAYER_SUFFIXES: [&str; 11] = [
    "input_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

/// Whether to run a resident reference after the candidate completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LayerCheckMode {
    /// Compare every output with resident weights; process peak includes that reference.
    CompareResident,
    /// Skip the reference so an external process-memory measurement isolates the candidate.
    CandidateOnly,
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Verification {
    NotRun,
    BitExact { compared_values: usize },
}

/// A real layer evaluated from selected checkpoint ranges, with a resident control.
#[derive(Debug, serde::Serialize)]
pub struct Qwen3LayerCheck {
    schema_version: u32,
    operation: &'static str,
    layer: usize,
    input_shape: [usize; 3],
    input_recipe: &'static str,
    read_bytes: u64,
    logical_weight_bytes: u64,
    planned_peak_weight_and_staging_bytes: u64,
    max_weight_bytes: u64,
    load_ms: f64,
    execute_ms: f64,
    verification: Verification,
    scope: &'static str,
}

/// Loads and evaluates one Qwen block, then compares it with resident MLX weights.
///
/// Inputs are deterministic hidden states, not tokenizer output or an embedding
/// from preceding layers. The budget covers logical weights and read/conversion
/// staging only; it excludes attention/MLP scratch, headers, allocator retention,
/// hidden states, and the resident reference. This is not a whole-process limit.
pub fn qualify_layer(
    model: &Path,
    layer: usize,
    tokens: usize,
    max_weight_bytes: u64,
    mode: LayerCheckMode,
) -> Result<Qwen3LayerCheck, Qwen3MetalLoadError> {
    let inspection = Qwen3CheckpointInspection::inspect(model)?;
    let config_path = model.join("config.json");
    let config_json = std::fs::read_to_string(&config_path).map_err(|source| {
        crate::checkpoint::Qwen3CheckpointError::ReadConfig {
            path: config_path,
            source,
        }
    })?;
    let config = Qwen3ForwardConfig::parse(&config_json)?;
    if layer >= config.hidden_layers() || tokens == 0 || tokens > 32 {
        return Err(Qwen3MetalLoadError::DimensionOutOfRange(
            "layer or token count",
        ));
    }
    let names: Vec<_> = LAYER_SUFFIXES
        .iter()
        .map(|suffix| format!("model.layers.{layer}.{suffix}"))
        .collect();
    let lengths = names
        .iter()
        .map(|name| inspection.bf16_tensor_bytes(name))
        .collect::<Result<Vec<_>, _>>()?;
    let (read_bytes, weight_bytes, planned_peak) = plan_weight_bytes(&lengths)?;
    if planned_peak > max_weight_bytes {
        return Err(Qwen3MetalLoadError::LayerWeightBudget {
            required: planned_peak,
            maximum: max_weight_bytes,
        });
    }
    let elements = tokens
        .checked_mul(config.hidden_size())
        .ok_or(Qwen3MetalLoadError::DimensionOutOfRange("input elements"))?;
    // Bound synthetic host input independently of the weight budget.
    if elements > 1_048_576 {
        return Err(Qwen3MetalLoadError::DimensionOutOfRange("input elements"));
    }
    let input_values: Vec<_> = (0_i16..101)
        .cycle()
        .take(elements)
        .map(|bounded| f32::from(bounded - 50) * 0.001)
        .collect();
    let shape = [
        1,
        i32::try_from(tokens).map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("tokens"))?,
        i32::try_from(config.hidden_size())
            .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("hidden"))?,
    ];
    let input = Array::from_slice(&input_values, &shape);
    input.eval()?;

    let started = Instant::now();
    let candidate_weights = load_layer(&inspection, &names, &lengths)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let candidate = crate::forward::forward_layer(&config, &candidate_weights, layer, &input)?;
    candidate.eval()?;
    let execute_ms = started.elapsed().as_secs_f64() * 1000.0;
    let candidate_values = candidate.as_slice::<f32>().to_vec();
    // Evaluation and readback finish before dropping any weight dependencies.
    drop(candidate);
    drop(candidate_weights);

    let verification = if mode == LayerCheckMode::CompareResident {
        let mut reference_weights = Qwen3MlxWeights::load(model)?;
        reference_weights.prepare_float32()?;
        let reference =
            crate::forward::forward_layer(&config, &reference_weights.tensors, layer, &input)?;
        reference.eval()?;
        compare_tensor_values(&candidate_values, reference.as_slice::<f32>())?;
        Verification::BitExact {
            compared_values: candidate_values.len(),
        }
    } else {
        Verification::NotRun
    };
    Ok(Qwen3LayerCheck {
        schema_version: 1,
        operation: "qwen3_selected_layer_check",
        layer,
        input_shape: [1, tokens, config.hidden_size()],
        input_recipe: "f32((flat_index % 101) - 50) * 0.001",
        read_bytes,
        logical_weight_bytes: weight_bytes,
        planned_peak_weight_and_staging_bytes: planned_peak,
        max_weight_bytes,
        load_ms,
        execute_ms,
        verification,
        scope: "one real transformer block on synthetic hidden states; synchronous reads; weight/staging budget excludes execution scratch, allocator retention and resident reference; no full-model streaming or physical SSD measurement",
    })
}

fn load_layer(
    inspection: &Qwen3CheckpointInspection,
    names: &[String],
    lengths: &[u64],
) -> Result<HashMap<String, Array>, Qwen3MetalLoadError> {
    let mut weights = HashMap::new();
    for (name, &length) in names.iter().zip(lengths) {
        let payload = inspection.read_tensor(name, length)?;
        let values = decode_bf16(payload.bytes())?;
        let shape = payload
            .shape()
            .iter()
            .map(|&dimension| {
                i32::try_from(dimension)
                    .map_err(|_| Qwen3MetalLoadError::DimensionOutOfRange("tensor shape"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let array = Array::from_slice(&values, &shape);
        array.eval()?;
        weights.insert(name.clone(), array);
        // Raw and widened host staging leave scope before the next read.
    }
    Ok(weights)
}

fn plan_weight_bytes(lengths: &[u64]) -> Result<(u64, u64, u64), Qwen3MetalLoadError> {
    let mut raw_total = 0_u64;
    let mut resident = 0_u64;
    let mut peak = 0_u64;
    for &raw in lengths {
        let next_peak = raw
            .checked_mul(5)
            .and_then(|staging| resident.checked_add(staging))
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange(
                "weight/staging bytes",
            ))?;
        peak = peak.max(next_peak);
        raw_total = raw_total
            .checked_add(raw)
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange("raw bytes"))?;
        resident = raw_total
            .checked_mul(2)
            .ok_or(Qwen3MetalLoadError::DimensionOutOfRange("FP32 bytes"))?;
    }
    Ok((raw_total, resident, peak))
}

#[cfg(test)]
mod tests {
    #[test]
    fn staging_plan_counts_prior_arrays_and_both_current_fp32_copies() {
        // First tensor: 10 raw + 20 host FP32 + 20 array. Second:
        // 20 retained + 4 raw + 8 host FP32 + 8 array = 40.
        assert_eq!(super::plan_weight_bytes(&[10, 4]).unwrap(), (14, 28, 50));
        assert!(super::plan_weight_bytes(&[u64::MAX]).is_err());
    }
}
