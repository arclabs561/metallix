//! Narrow floating-point references, re-exported from the `blockfloat` crate.
//!
//! The Metal FP8/FP4 kernels stay here because they depend on this crate's
//! device features.

pub use blockfloat::{
    ActivationGroup, ActivationQuantError, ActivationRoundtripError, Bf16LinearError,
    BlockDecodeError, Fp4ActivationError, Fp4ActivationMode, Fp4LinearError, Fp8LinearError,
    Fp32LinearError, MAX_ACTIVATION_ROUNDTRIP_ELEMENTS, MAX_BF16_LINEAR_ELEMENTS,
    MAX_FP4_ACTIVATION_ELEMENTS, MAX_FP32_LINEAR_ELEMENTS, bf16_linear_reference, decode_e2m1,
    decode_e2m1x2, decode_e4m3fn, decode_e8m0, expand_e2m1x2_blocks32, fp4_linear_runtime_f32,
    fp8_linear_runtime_f32, fp32_linear_reference, quantize_bf16_activations_e4m3fn,
    requantize_bf16_activations_e2m1, requantize_bf16_activations_e4m3fn,
};
pub(crate) use blockfloat::{bf16_to_f32, f32_to_bf16_rne, fp4_linear_runtime_f32_owned};
use thiserror::Error;

#[cfg(feature = "metal")]
mod fp8_metal;
#[cfg(feature = "metal")]
pub use fp8_metal::{
    Fp4MetalExpert, Fp4MetalKernel, Fp4MetalWeights, Fp8MetalError, Fp8MetalKernel, Fp8MetalWeights,
};

#[cfg(feature = "metal")]
mod device;
#[cfg(feature = "metal")]
pub use device::DeviceCounts;
#[cfg(feature = "metal")]
pub(crate) use device::{DeviceLinears, Fp8Buffers, ResidentFp8, active_device};

/// An FP8 linear failure on the forward path that ran it.
#[derive(Clone, Debug, Error, PartialEq)]
pub(crate) enum Fp8ForwardError {
    /// The scalar reference rejected its inputs or arithmetic.
    #[error(transparent)]
    Scalar(#[from] Fp8LinearError),
    /// Inside a device scope, the device rejected the call or failed. The
    /// scalar path is not rerun: that would hide the fault and change numerics.
    #[cfg(feature = "metal")]
    #[error("Metal FP8 linear failed: {0}")]
    Device(#[from] Fp8MetalError),
}

/// FP8 linear for the model's forward paths.
///
/// Inside a `DeviceLinears` scope (with the `metal` feature) and with G32
/// activations this runs the device kernel, whose only difference from
/// [`fp8_linear_runtime_f32`] is the order of the 32 products inside each
/// block, and returns any device error. Outside a scope, or with another
/// activation group, it is the scalar reference.
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors fp8_linear_runtime_f32 so call sites swap one name"
)]
pub(crate) fn fp8_linear_f32(
    activation_codes: &[u8],
    activation_scales: &[u8],
    weight_codes: &[u8],
    weight_scales: &[u8],
    rows: usize,
    reduction: usize,
    outputs: usize,
    activation_group: ActivationGroup,
    output: &mut [f32],
) -> Result<(), Fp8ForwardError> {
    #[cfg(feature = "metal")]
    if activation_group == ActivationGroup::Elements32
        && let Some(device) = active_device()
    {
        let values = device.fp8_linear(
            (activation_codes, activation_scales),
            (weight_codes, weight_scales),
            rows,
            reduction,
            outputs,
        )?;
        if values.len() != output.len() {
            return Err(Fp8MetalError::Length {
                field: "output",
                expected: output.len(),
                actual: values.len(),
            }
            .into());
        }
        output.copy_from_slice(&values);
        return Ok(());
    }
    Ok(fp8_linear_runtime_f32(
        activation_codes,
        activation_scales,
        weight_codes,
        weight_scales,
        rows,
        reduction,
        outputs,
        activation_group,
        output,
    )?)
}

/// One device FP8 linear with weights uploaded for this call only.
#[cfg(feature = "metal")]
pub(crate) fn metal_fp8_linear(
    activation_codes: &[u8],
    activation_scales: &[u8],
    weight_codes: &[u8],
    weight_scales: &[u8],
    rows: usize,
    reduction: usize,
    outputs: usize,
) -> Result<Vec<f32>, Fp8MetalError> {
    let _device = crate::device_lock();
    let weights = Fp8MetalWeights::new(weight_codes, weight_scales, outputs, reduction)?;
    // Built per call under the lock: MLX caches the compiled kernel, and a
    // thread-local kernel would be freed at thread exit outside the lock.
    Fp8MetalKernel::new()?.forward(activation_codes, activation_scales, rows, &weights)
}

/// One routed FP4 expert on the device, as `Fp4ExpertWeights::forward_token`
/// computes it from the token's G32 E4M3FN activation `(codes, scales)`, with
/// weights uploaded for this call only.
///
/// Returns `Ok(None)` when the kernel cannot express the request: a `SwiGLU`
/// limit that is not an integer in `0..=255`. That is the one case in which
/// the caller runs the scalar expert instead (and counts it).
#[cfg(feature = "metal")]
#[allow(
    clippy::too_many_arguments,
    reason = "each encoded projection owns distinct code and scale storage"
)]
pub(crate) fn metal_fp4_expert(
    activations: (&[u8], &[u8]),
    hidden_width: usize,
    intermediate_width: usize,
    w1: (&[u8], &[u8]),
    w2: (&[u8], &[u8]),
    w3: (&[u8], &[u8]),
    swiglu_limit: f32,
    route_weight: Option<f32>,
) -> Result<Option<Vec<u16>>, Fp8MetalError> {
    let Some(limit) =
        (0..=u8::MAX).find(|&limit| f32::from(limit).to_bits() == swiglu_limit.to_bits())
    else {
        return Ok(None);
    };
    let _device = crate::device_lock();
    let expert = Fp4MetalExpert {
        w1: Fp4MetalWeights::new(w1.0, w1.1, intermediate_width, hidden_width)?,
        w3: Fp4MetalWeights::new(w3.0, w3.1, intermediate_width, hidden_width)?,
        w2: Fp4MetalWeights::new(w2.0, w2.1, hidden_width, intermediate_width)?,
    };
    // An absent route weight is the scalar path's skipped multiply; `1.0 * v` is exact.
    let weight = [route_weight.unwrap_or(1.0)];
    let outputs = Fp4MetalKernel::new()?.experts_forward(
        activations.0,
        activations.1,
        1,
        &[&expert],
        &[&weight],
        limit,
    )?;
    let output = outputs.first().ok_or(Fp8MetalError::Length {
        field: "expert outputs",
        expected: 1,
        actual: 0,
    })?;
    output
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let bits = f32_to_bf16_rne(value);
            if bf16_to_f32(bits).is_finite() {
                Ok(bits)
            } else {
                Err(Fp8MetalError::NonFiniteOutput { index })
            }
        })
        .collect::<Result<_, _>>()
        .map(Some)
}

#[cfg(all(test, feature = "metal"))]
mod device_parity_tests;
#[cfg(test)]
mod expert_composition_tests;
