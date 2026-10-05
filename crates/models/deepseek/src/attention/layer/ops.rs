//! Numerical stage helpers shared by the QR prefix and the forward paths.

use std::num::NonZeroUsize;

use crate::{
    RotaryDirection, RotaryFrequency, RotaryTailLayout,
    precision::{
        ActivationGroup, f32_to_bf16_rne, fp8_linear_f32, quantize_bf16_activations_e4m3fn,
    },
    rms_norm_bf16_reference, rotate_tail,
};

use super::{
    Fp8Projection, LayerAttentionError, LayerAttentionLayoutError, MAX_LAYER_ATTENTION_ELEMENTS,
    state::TailShape,
};

pub(super) fn fp8_project_bf16(
    input: &[u16],
    rows: usize,
    reduction: usize,
    outputs: usize,
    projection: Fp8Projection<'_>,
) -> Result<Vec<u16>, LayerAttentionError> {
    if input.len() > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "FP8 projection input",
            elements: input.len(),
        });
    }
    let output_elements = checked_product(&[rows, outputs], "projection output")?;
    if output_elements > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "FP8 projection output",
            elements: output_elements,
        });
    }
    let mut codes = vec![0; input.len()];
    let mut scales = vec![0; rows * (reduction / 32)];
    quantize_bf16_activations_e4m3fn(
        input,
        rows,
        reduction,
        ActivationGroup::Elements32,
        &mut codes,
        &mut scales,
    )?;
    let mut fp32 = vec![0.0; output_elements];
    fp8_linear_f32(
        &codes,
        &scales,
        projection.codes,
        projection.scales,
        rows,
        reduction,
        outputs,
        ActivationGroup::Elements32,
        &mut fp32,
    )?;
    fp32.into_iter()
        .enumerate()
        .map(|(element, value)| {
            let bits = f32_to_bf16_rne(value);
            if bf16_to_f32(bits).is_finite() {
                Ok(bits)
            } else {
                Err(LayerAttentionError::NonFiniteProjection { element })
            }
        })
        .collect()
}

pub(super) fn rms_norm_rows(
    input: &[u16],
    rows: usize,
    width: usize,
    weight: &[u16],
    epsilon: f32,
) -> Result<Vec<u16>, LayerAttentionError> {
    let expected = checked_product(&[rows, width], "RMS norm input")?;
    if input.len() != expected {
        return Err(LayerAttentionError::ShapeOverflow {
            field: "RMS norm input",
        });
    }
    let mut output = vec![0; input.len()];
    for row in 0..rows {
        let start = row * width;
        rms_norm_bf16_reference(
            &input[start..start + width],
            weight,
            epsilon,
            &mut output[start..start + width],
        )?;
    }
    Ok(output)
}

pub(super) fn rotate_bf16_tail(
    values: &[u16],
    shape: TailShape,
    frequencies: &[RotaryFrequency],
    direction: RotaryDirection,
) -> Result<Vec<u16>, LayerAttentionError> {
    let prefix = shape.head_dimension.get() - shape.rope_pairs.get() * 2;
    let mut tail =
        Vec::with_capacity(values.len() / shape.head_dimension.get() * shape.rope_pairs.get() * 2);
    for head in values.chunks_exact(shape.head_dimension.get()) {
        tail.extend(head[prefix..].iter().map(|&bits| bf16_to_f32(bits)));
    }
    let layout = RotaryTailLayout::new(
        shape.batches,
        NonZeroUsize::new(shape.positions).expect("positions validated"),
        shape.heads,
        shape.rope_pairs,
    )?;
    rotate_tail(&mut tail, layout, frequencies, direction)?;
    let mut result = values.to_vec();
    for (head, rotated) in result
        .chunks_exact_mut(shape.head_dimension.get())
        .zip(tail.chunks_exact(shape.rope_pairs.get() * 2))
    {
        for (offset, &value) in rotated.iter().enumerate() {
            let bits = f32_to_bf16_rne(value);
            if !bf16_to_f32(bits).is_finite() {
                return Err(LayerAttentionError::NonFiniteRotary { element: offset });
            }
            head[prefix + offset] = bits;
        }
    }
    Ok(result)
}

pub(super) fn concatenate_kv(
    window: &[u16],
    window_keys: usize,
    compressed: &[u16],
    compressed_keys: usize,
    batches: usize,
    width: usize,
) -> Result<Vec<u16>, LayerAttentionError> {
    let keys =
        window_keys
            .checked_add(compressed_keys)
            .ok_or(LayerAttentionError::ShapeOverflow {
                field: "shared KV keys",
            })?;
    let per_batch = checked_product(&[keys, width], "shared KV per batch")?;
    let total = checked_product(&[batches, per_batch], "shared KV")?;
    if total > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "shared KV",
            elements: total,
        });
    }
    let mut result = Vec::with_capacity(total);
    for batch in 0..batches {
        result.extend_from_slice(
            &window[batch * window_keys * width..(batch + 1) * window_keys * width],
        );
        result.extend_from_slice(
            &compressed[batch * compressed_keys * width..(batch + 1) * compressed_keys * width],
        );
    }
    Ok(result)
}

pub(super) fn concatenate_indices(
    window: &[i32],
    compressed: &[i32],
    batches: usize,
    positions: usize,
) -> Result<Vec<i32>, LayerAttentionError> {
    let rows = checked_product(&[batches, positions], "index rows")?;
    if !window.len().is_multiple_of(rows) || !compressed.len().is_multiple_of(rows) {
        return Err(LayerAttentionError::ShapeOverflow {
            field: "index rows",
        });
    }
    let window_slots = window.len() / rows;
    let compressed_slots = compressed.len() / rows;
    let total =
        window
            .len()
            .checked_add(compressed.len())
            .ok_or(LayerAttentionError::ShapeOverflow {
                field: "concatenated indices",
            })?;
    if total > MAX_LAYER_ATTENTION_ELEMENTS {
        return Err(LayerAttentionError::ElementLimit {
            field: "concatenated indices",
            elements: total,
        });
    }
    let mut result = Vec::with_capacity(total);
    for row in 0..rows {
        result.extend_from_slice(&window[row * window_slots..(row + 1) * window_slots]);
        result.extend_from_slice(&compressed[row * compressed_slots..(row + 1) * compressed_slots]);
    }
    Ok(result)
}

pub(super) fn checked_product(
    values: &[usize],
    field: &'static str,
) -> Result<usize, LayerAttentionLayoutError> {
    values.iter().try_fold(1_usize, |total, &value| {
        total
            .checked_mul(value)
            .ok_or(LayerAttentionLayoutError::ShapeOverflow { field })
    })
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}
