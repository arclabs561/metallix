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

#[cfg(feature = "metal")]
mod fp8_metal;
#[cfg(feature = "metal")]
pub use fp8_metal::{
    Fp4MetalExpert, Fp4MetalKernel, Fp4MetalWeights, Fp8MetalError, Fp8MetalKernel, Fp8MetalWeights,
};

#[cfg(test)]
mod expert_composition_tests;
