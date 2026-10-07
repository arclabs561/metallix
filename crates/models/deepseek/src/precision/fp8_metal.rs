//! Fused FP8 linear on Metal through MLX's custom-kernel API.
//!
//! Weights stay resident as one-byte E4M3FN codes with E8M0 block scales. One
//! SIMD group computes one output row: lane `j` handles element `j` of each
//! 32-element block, the block's dot is a SIMD sum, and blocks are accumulated
//! in ascending order after applying the activation and weight scales. That
//! is the order of [`super::fp8_linear_runtime_f32`], so outputs match it
//! up to the order of the 32 products inside each block.
//!
//! The pinned `mlx-rs` does not wrap MLX-C's `mlx_fast_metal_kernel_*` API, so
//! this module calls it directly. It is the only module allowed `unsafe`; each
//! call site validates every input before crossing the FFI boundary, and all
//! MLX-C handles are owned by RAII guards.
#![allow(unsafe_code)]

use std::ffi::CString;
use std::sync::OnceLock;

use mlx_rs::{Array, Dtype, Stream, StreamOrDevice};
use mlx_sys as sys;
use thiserror::Error;

use super::{decode_e2m1x2, decode_e4m3fn, decode_e8m0};

const GROUP: usize = 32;

const SOURCE: &str = r"
    uint column = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    float accumulated = 0.0f;
    uint weight_base = column * K;
    uint activation_base = row * K;
    uint groups = K / 32;
    for (uint group = 0; group < groups; ++group) {
        uint offset = group * 32 + lane;
        float product = lut[activation_codes[activation_base + offset]]
            * lut[weight_codes[weight_base + offset]];
        float dot = simd_sum(product);
        accumulated += dot * activation_scales[row * groups + group]
            * weight_scales[(column / 32) * groups + group];
    }
    if (lane == 0) {
        out[row * N + column] = accumulated;
    }
";

const FP4_SOURCE: &str = r"
    uint column = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    float accumulated = 0.0f;
    uint groups = K / 32;
    uint weight_base = column * (K / 2);
    uint activation_base = row * K;
    for (uint group = 0; group < groups; ++group) {
        uchar packed = weight_codes[weight_base + group * 16 + lane / 2];
        uint nibble = (lane & 1) ? (packed >> 4) : (packed & 15);
        float product = lut[activation_codes[activation_base + group * 32 + lane]] * fp4_lut[nibble];
        float dot = simd_sum(product);
        accumulated += dot * activation_scales[row * groups + group]
            * weight_scales[column * groups + group];
    }
    if (lane == 0) {
        out[row * N + column] = accumulated;
    }
";

/// `SwiGLU`, route weighting and the source's FP8 activation quantization, on device.
///
/// Inputs are the gate and up projections in FP32. Each is rounded to BF16
/// (the source's matrix output dtype), clamped, combined as `silu(gate) * up`
/// in FP32, scaled by the row's route weight and rounded to BF16, then
/// quantized per 32 elements with a power-of-two E8M0 scale and
/// round-to-nearest-even E4M3FN, as `quantize_bf16_activations_e4m3fn` does.
/// The output packs the decoded E4M3FN values `[rows, I]` followed by the
/// scales `[rows, I / 32]`.
const SWIGLU_QUANT_SOURCE: &str = r"
    uint group = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    uint index = row * I + group * 32 + lane;
    uint gate_bits = as_type<uint>(gate[index]);
    uint up_bits = as_type<uint>(up[index]);
    float g = as_type<float>((gate_bits + 0x7fffu + ((gate_bits >> 16) & 1u)) & 0xffff0000u);
    float u = as_type<float>((up_bits + 0x7fffu + ((up_bits >> 16) & 1u)) & 0xffff0000u);
    if (LIMIT > 0) {
        u = clamp(u, -float(LIMIT), float(LIMIT));
        g = min(g, float(LIMIT));
    }
    float v = (g / (1.0f + metal::precise::exp(-g))) * u;
    v = route_weights[row] * v;
    uint v_bits = as_type<uint>(v);
    v = as_type<float>((v_bits + 0x7fffu + ((v_bits >> 16) & 1u)) & 0xffff0000u);
    float amax = max(simd_max(fabs(v)), 1e-4f);
    uint scaled = as_type<uint>(amax * (1.0f / 448.0f));
    int exponent = int((scaled >> 23) & 0xffu) - 127 + ((scaled & 0x7fffffu) != 0u ? 1 : 0);
    float scale = as_type<float>(uint(exponent + 127) << 23);
    float normalized = clamp(v / scale, -448.0f, 448.0f);
    float magnitude = fabs(normalized);
    float quantized;
    if (magnitude < 0.015625f) {
        quantized = rint(magnitude * 512.0f) / 512.0f;
    } else {
        uint bits = as_type<uint>(magnitude);
        quantized = as_type<float>((bits + 0x7ffffu + ((bits >> 20) & 1u)) & 0xfff00000u);
    }
    out[index] = copysign(quantized, normalized);
    if (lane == 0) {
        out[ROWS * I + row * (I / 32) + group] = scale;
    }
";

/// The FP4 linear with activations already decoded to FP32 values plus scales.
const FP4_DECODED_SOURCE: &str = r"
    uint column = threadgroup_position_in_grid.x;
    uint row = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    float accumulated = 0.0f;
    uint groups = K / 32;
    uint weight_base = column * (K / 2);
    uint activation_base = row * K;
    for (uint group = 0; group < groups; ++group) {
        uchar packed = weight_codes[weight_base + group * 16 + lane / 2];
        uint nibble = (lane & 1) ? (packed >> 4) : (packed & 15);
        float product = activations[activation_base + group * 32 + lane] * fp4_lut[nibble];
        float dot = simd_sum(product);
        accumulated += dot * activations[ROWS * K + row * groups + group]
            * weight_scales[column * groups + group];
    }
    if (lane == 0) {
        out[row * N + column] = accumulated;
    }
";

/// An invalid fused FP8 or FP4 linear request or MLX failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum Fp8MetalError {
    /// A dimension is zero or the reduction is not a multiple of 32.
    #[error("invalid FP8 Metal geometry: rows {rows}, reduction {reduction}, outputs {outputs}")]
    Geometry {
        /// Activation rows.
        rows: usize,
        /// Reduction width.
        reduction: usize,
        /// Output width.
        outputs: usize,
    },
    /// A buffer has the wrong length for the requested geometry.
    #[error("FP8 Metal {field} length is {actual}, expected {expected}")]
    Length {
        /// Buffer role.
        field: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// A dimension does not fit MLX's 32-bit shape and grid arguments.
    #[error("FP8 Metal {field} {value} exceeds the 32-bit MLX limit")]
    TooLarge {
        /// Dimension role.
        field: &'static str,
        /// Supplied value.
        value: usize,
    },
    /// An E4M3FN code is NaN (`0x7f` or `0xff`) or an E8M0 scale is NaN (`0xff`).
    #[error("nonfinite FP8 Metal {field} code at index {index}")]
    NonFinite {
        /// Buffer role.
        field: &'static str,
        /// Flat element index.
        index: usize,
    },
    /// MLX rejected kernel construction or application.
    #[error("MLX custom kernel failed: {0}")]
    Mlx(String),
    /// The device produced a nonfinite output.
    #[error("FP8 Metal output is nonfinite at index {index}")]
    NonFiniteOutput {
        /// Flat output index.
        index: usize,
    },
}

/// Resident FP8 weights `[outputs, reduction]` with E8M0 scales per 32x32 block.
pub struct Fp8MetalWeights {
    codes: Array,
    scales: Array,
    outputs: usize,
    reduction: usize,
}

impl Fp8MetalWeights {
    /// Validates and uploads checkpoint-layout FP8 codes and block scales.
    ///
    /// `codes` is row-major `[outputs, reduction]`; `scales` is row-major
    /// `[ceil(outputs / 32), reduction / 32]`, matching the checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`Fp8MetalError`] for invalid geometry, lengths, sizes beyond
    /// MLX's 32-bit limits, or NaN codes or scales.
    pub fn new(
        codes: &[u8],
        scales: &[u8],
        outputs: usize,
        reduction: usize,
    ) -> Result<Self, Fp8MetalError> {
        if outputs == 0 || reduction == 0 || !reduction.is_multiple_of(GROUP) {
            return Err(Fp8MetalError::Geometry {
                rows: 1,
                reduction,
                outputs,
            });
        }
        let code_count = checked(outputs, reduction, "weight codes")?;
        let scale_count = checked(outputs.div_ceil(GROUP), reduction / GROUP, "weight scales")?;
        exact_length("weight codes", code_count, codes.len())?;
        exact_length("weight scales", scale_count, scales.len())?;
        finite_e4m3("weight codes", codes)?;
        let decoded_scales = decode_scales("weight scales", scales)?;
        Ok(Self {
            codes: Array::from_slice(codes, &[dimension("weight codes", code_count)?]),
            scales: Array::from_slice(&decoded_scales, &[dimension("weight scales", scale_count)?]),
            outputs,
            reduction,
        })
    }

    /// Returns the output width.
    #[must_use]
    pub const fn outputs(&self) -> usize {
        self.outputs
    }

    /// Returns the reduction width.
    #[must_use]
    pub const fn reduction(&self) -> usize {
        self.reduction
    }
}

/// Resident packed E2M1 weights `[outputs, reduction]` with one E8M0 scale per 32 elements.
pub struct Fp4MetalWeights {
    codes: Array,
    scales: Array,
    outputs: usize,
    reduction: usize,
}

impl Fp4MetalWeights {
    /// Validates and uploads checkpoint-layout packed FP4 codes and scales.
    ///
    /// `codes` is row-major `[outputs, reduction / 2]` low-nibble-first
    /// `E2M1x2` bytes; `scales` is row-major `[outputs, reduction / 32]`.
    ///
    /// # Errors
    ///
    /// Returns [`Fp8MetalError`] for invalid geometry, lengths, sizes beyond
    /// MLX's 32-bit limits, or NaN scales. Every E2M1 code is finite.
    pub fn new(
        codes: &[u8],
        scales: &[u8],
        outputs: usize,
        reduction: usize,
    ) -> Result<Self, Fp8MetalError> {
        if outputs == 0 || reduction == 0 || !reduction.is_multiple_of(GROUP) {
            return Err(Fp8MetalError::Geometry {
                rows: 1,
                reduction,
                outputs,
            });
        }
        let code_count = checked(outputs, reduction / 2, "weight codes")?;
        let scale_count = checked(outputs, reduction / GROUP, "weight scales")?;
        exact_length("weight codes", code_count, codes.len())?;
        exact_length("weight scales", scale_count, scales.len())?;
        let decoded_scales = decode_scales("weight scales", scales)?;
        Ok(Self {
            codes: Array::from_slice(codes, &[dimension("weight codes", code_count)?]),
            scales: Array::from_slice(&decoded_scales, &[dimension("weight scales", scale_count)?]),
            outputs,
            reduction,
        })
    }

    /// Returns the output width.
    #[must_use]
    pub const fn outputs(&self) -> usize {
        self.outputs
    }

    /// Returns the reduction width.
    #[must_use]
    pub const fn reduction(&self) -> usize {
        self.reduction
    }
}

/// One routed expert's resident packed FP4 projections.
pub struct Fp4MetalExpert {
    /// Gate projection `[inter, dim]`.
    pub w1: Fp4MetalWeights,
    /// Up projection `[inter, dim]`.
    pub w3: Fp4MetalWeights,
    /// Down projection `[dim, inter]`.
    pub w2: Fp4MetalWeights,
}

/// A compiled fused FP8-activation by FP4-weight linear kernel for routed experts.
pub struct Fp4MetalKernel {
    kernel: KernelHandle,
    swiglu: KernelHandle,
    decoded: KernelHandle,
    lut: Array,
    fp4_lut: Array,
}

impl Fp4MetalKernel {
    /// Constructs the kernel. MLX compiles and caches it on first application.
    ///
    /// # Errors
    ///
    /// Returns [`Fp8MetalError::Mlx`] if MLX cannot create the kernel object.
    pub fn new() -> Result<Self, Fp8MetalError> {
        let fp4: Vec<f32> = (0..16_u8).map(|code| decode_e2m1x2(code)[0]).collect();
        Ok(Self {
            kernel: KernelHandle::new(
                "metallix_fp4_linear_g32",
                &[
                    "activation_codes",
                    "activation_scales",
                    "weight_codes",
                    "weight_scales",
                    "lut",
                    "fp4_lut",
                ],
                "out",
                FP4_SOURCE,
            )?,
            swiglu: KernelHandle::new(
                "metallix_swiglu_quant_g32",
                &["gate", "up", "route_weights"],
                "out",
                SWIGLU_QUANT_SOURCE,
            )?,
            decoded: KernelHandle::new(
                "metallix_fp4_decoded_linear_g32",
                &["activations", "weight_codes", "weight_scales", "fp4_lut"],
                "out",
                FP4_DECODED_SOURCE,
            )?,
            lut: e4m3_lut(),
            fp4_lut: Array::from_slice(&fp4, &[16]),
        })
    }

    /// Computes `activation x weightsᵀ` for E4M3FN activations with G32 E8M0 scales.
    ///
    /// Inputs are as for [`Fp8MetalKernel::forward`]; the weights are packed FP4.
    /// This is the source `fp4_gemm` with `act_block_size = 32`.
    ///
    /// # Errors
    ///
    /// Returns [`Fp8MetalError`] for invalid lengths or codes, an MLX failure,
    /// or a nonfinite output.
    pub fn forward(
        &self,
        activation_codes: &[u8],
        activation_scales: &[u8],
        rows: usize,
        weights: &Fp4MetalWeights,
    ) -> Result<Vec<f32>, Fp8MetalError> {
        let (codes, scales) = upload_activations(
            activation_codes,
            activation_scales,
            rows,
            weights.reduction,
            weights.outputs,
        )?;
        run(
            &self.kernel,
            &[
                &codes,
                &scales,
                &weights.codes,
                &weights.scales,
                &self.lut,
                &self.fp4_lut,
            ],
            rows,
            weights.reduction,
            weights.outputs,
        )
    }
}

impl Fp4MetalKernel {
    /// Applies the same activations to several weight matrices in one device submission.
    ///
    /// Each output equals [`Self::forward`] on that weight bit for bit; only the
    /// host round trips are shared. Use it for the gate and up projections of a
    /// layer's routed experts, which all read the same normalized input.
    ///
    /// # Errors
    ///
    /// As for [`Self::forward`]; every weight must share one reduction width.
    pub fn forward_many(
        &self,
        activation_codes: &[u8],
        activation_scales: &[u8],
        rows: usize,
        weights: &[&Fp4MetalWeights],
    ) -> Result<Vec<Vec<f32>>, Fp8MetalError> {
        let Some(first) = weights.first() else {
            return Ok(Vec::new());
        };
        let reduction = first.reduction;
        let (codes, scales) = upload_activations(
            activation_codes,
            activation_scales,
            rows,
            reduction,
            first.outputs,
        )?;
        let mut pending = Vec::with_capacity(weights.len());
        for weight in weights {
            if weight.reduction != reduction {
                return Err(Fp8MetalError::Geometry {
                    rows,
                    reduction: weight.reduction,
                    outputs: weight.outputs,
                });
            }
            pending.push(launch(
                &self.kernel,
                &[
                    &codes,
                    &scales,
                    &weight.codes,
                    &weight.scales,
                    &self.lut,
                    &self.fp4_lut,
                ],
                rows,
                reduction,
                weight.outputs,
            )?);
        }
        mlx_rs::transforms::eval(pending.iter())
            .map_err(|error| Fp8MetalError::Mlx(error.to_string()))?;
        pending.iter().map(read_finite).collect()
    }
}

impl Fp4MetalKernel {
    /// Runs whole routed experts on device and waits once for all of them.
    ///
    /// For each expert this computes the source `Expert.forward`: gate and up
    /// projections from the shared quantized input, `SwiGLU` with `swiglu_limit`
    /// clamps, the row's route weight, BF16 rounding and FP8 requantization,
    /// then the down projection. `route_weights[e]` holds one weight per row
    /// for expert `e`. Returns each expert's FP32 down-projection output
    /// `[rows, dim]`, before the source's final BF16 rounding.
    ///
    /// # Errors
    ///
    /// Returns [`Fp8MetalError`] for mismatched geometry or lengths, an MLX
    /// failure, or a nonfinite output.
    pub fn experts_forward(
        &self,
        activation_codes: &[u8],
        activation_scales: &[u8],
        rows: usize,
        experts: &[&Fp4MetalExpert],
        route_weights: &[&[f32]],
        swiglu_limit: u8,
    ) -> Result<Vec<Vec<f32>>, Fp8MetalError> {
        exact_length("route weight sets", experts.len(), route_weights.len())?;
        let Some(first) = experts.first() else {
            return Ok(Vec::new());
        };
        let dim = first.w1.reduction;
        let (codes, scales) = upload_activations(
            activation_codes,
            activation_scales,
            rows,
            dim,
            first.w1.outputs,
        )?;
        let mut pending = Vec::with_capacity(experts.len());
        for (expert, weights) in experts.iter().zip(route_weights) {
            let inter = expert.w1.outputs;
            if expert.w1.reduction != dim
                || expert.w3.reduction != dim
                || expert.w3.outputs != inter
                || expert.w2.reduction != inter
                || expert.w2.outputs != dim
            {
                return Err(Fp8MetalError::Geometry {
                    rows,
                    reduction: expert.w2.reduction,
                    outputs: expert.w2.outputs,
                });
            }
            exact_length("route weights", rows, weights.len())?;
            if let Some(index) = weights.iter().position(|value| !value.is_finite()) {
                return Err(Fp8MetalError::NonFinite {
                    field: "route weights",
                    index,
                });
            }
            let gate = launch(
                &self.kernel,
                &[
                    &codes,
                    &scales,
                    &expert.w1.codes,
                    &expert.w1.scales,
                    &self.lut,
                    &self.fp4_lut,
                ],
                rows,
                dim,
                inter,
            )?;
            let up = launch(
                &self.kernel,
                &[
                    &codes,
                    &scales,
                    &expert.w3.codes,
                    &expert.w3.scales,
                    &self.lut,
                    &self.fp4_lut,
                ],
                rows,
                dim,
                inter,
            )?;
            let weights = Array::from_slice(weights, &[dimension("rows", rows)?]);
            let packed_len = checked(rows, inter + inter / GROUP, "packed activations")?;
            let activations = self.swiglu.apply(
                &[&gate, &up, &weights],
                &[dimension("packed activations", packed_len)?],
                &[
                    ("I", dimension("inter", inter)?),
                    ("ROWS", dimension("rows", rows)?),
                    ("LIMIT", i32::from(swiglu_limit)),
                ],
                [dimension("inter", inter)?, dimension("rows", rows)?, 1],
                [dimension("simd width", GROUP)?, 1, 1],
            )?;
            let grid_x = dimension("output columns", checked(dim, GROUP, "grid")?)?;
            pending.push(self.decoded.apply(
                &[
                    &activations,
                    &expert.w2.codes,
                    &expert.w2.scales,
                    &self.fp4_lut,
                ],
                &[dimension("output", checked(rows, dim, "output")?)?],
                &[
                    ("N", dimension("outputs", dim)?),
                    ("K", dimension("inter", inter)?),
                    ("ROWS", dimension("rows", rows)?),
                ],
                [grid_x, dimension("rows", rows)?, 1],
                [dimension("simd width", GROUP)?, 1, 1],
            )?);
        }
        mlx_rs::transforms::eval(pending.iter())
            .map_err(|error| Fp8MetalError::Mlx(error.to_string()))?;
        pending.iter().map(read_finite).collect()
    }
}

/// A compiled fused FP8 linear kernel. Build once and reuse across calls.
pub struct Fp8MetalKernel {
    kernel: KernelHandle,
    lut: Array,
}

impl Fp8MetalKernel {
    /// Constructs the kernel. MLX compiles and caches it on first application.
    ///
    /// # Errors
    ///
    /// Returns [`Fp8MetalError::Mlx`] if MLX cannot create the kernel object.
    pub fn new() -> Result<Self, Fp8MetalError> {
        Ok(Self {
            kernel: KernelHandle::new(
                "metallix_fp8_linear_g32",
                &[
                    "activation_codes",
                    "activation_scales",
                    "weight_codes",
                    "weight_scales",
                    "lut",
                ],
                "out",
                SOURCE,
            )?,
            lut: e4m3_lut(),
        })
    }

    /// Computes `activation x weightsᵀ` for E4M3FN activations with G32 E8M0 scales.
    ///
    /// `activation_codes` is `[rows, reduction]` and `activation_scales` is
    /// `[rows, reduction / 32]`, as produced by
    /// [`super::quantize_bf16_activations_e4m3fn`]. Returns FP32 `[rows, outputs]`.
    ///
    /// # Errors
    ///
    /// Returns [`Fp8MetalError`] for invalid lengths or codes, an MLX failure,
    /// or a nonfinite output.
    pub fn forward(
        &self,
        activation_codes: &[u8],
        activation_scales: &[u8],
        rows: usize,
        weights: &Fp8MetalWeights,
    ) -> Result<Vec<f32>, Fp8MetalError> {
        let (codes, scales) = upload_activations(
            activation_codes,
            activation_scales,
            rows,
            weights.reduction,
            weights.outputs,
        )?;
        run(
            &self.kernel,
            &[&codes, &scales, &weights.codes, &weights.scales, &self.lut],
            rows,
            weights.reduction,
            weights.outputs,
        )
    }
}

fn e4m3_lut() -> Array {
    let lut: Vec<f32> = (0..=u8::MAX)
        .map(|code| {
            let value = decode_e4m3fn(code);
            // NaN codes are rejected before upload; keep the table finite.
            if value.is_finite() { value } else { 0.0 }
        })
        .collect();
    Array::from_slice(&lut, &[256])
}

fn upload_activations(
    activation_codes: &[u8],
    activation_scales: &[u8],
    rows: usize,
    reduction: usize,
    outputs: usize,
) -> Result<(Array, Array), Fp8MetalError> {
    if rows == 0 {
        return Err(Fp8MetalError::Geometry {
            rows,
            reduction,
            outputs,
        });
    }
    let code_count = checked(rows, reduction, "activation codes")?;
    let scale_count = checked(rows, reduction / GROUP, "activation scales")?;
    exact_length("activation codes", code_count, activation_codes.len())?;
    exact_length("activation scales", scale_count, activation_scales.len())?;
    finite_e4m3("activation codes", activation_codes)?;
    let decoded_scales = decode_scales("activation scales", activation_scales)?;
    Ok((
        Array::from_slice(
            activation_codes,
            &[dimension("activation codes", code_count)?],
        ),
        Array::from_slice(
            &decoded_scales,
            &[dimension("activation scales", scale_count)?],
        ),
    ))
}

fn launch(
    kernel: &KernelHandle,
    inputs: &[&Array],
    rows: usize,
    reduction: usize,
    outputs: usize,
) -> Result<Array, Fp8MetalError> {
    let output_count = checked(rows, outputs, "output")?;
    let grid_x = dimension("output columns", checked(outputs, GROUP, "grid")?)?;
    kernel.apply(
        inputs,
        &[dimension("output", output_count)?],
        &[
            ("N", dimension("outputs", outputs)?),
            ("K", dimension("reduction", reduction)?),
        ],
        [grid_x, dimension("rows", rows)?, 1],
        [dimension("simd width", GROUP)?, 1, 1],
    )
}

fn run(
    kernel: &KernelHandle,
    inputs: &[&Array],
    rows: usize,
    reduction: usize,
    outputs: usize,
) -> Result<Vec<f32>, Fp8MetalError> {
    let output = launch(kernel, inputs, rows, reduction, outputs)?;
    output
        .eval()
        .map_err(|error| Fp8MetalError::Mlx(error.to_string()))?;
    read_finite(&output)
}

fn read_finite(output: &Array) -> Result<Vec<f32>, Fp8MetalError> {
    let values = output.as_slice::<f32>().to_vec();
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(Fp8MetalError::NonFiniteOutput { index });
    }
    Ok(values)
}

fn checked(left: usize, right: usize, field: &'static str) -> Result<usize, Fp8MetalError> {
    left.checked_mul(right).ok_or(Fp8MetalError::TooLarge {
        field,
        value: usize::MAX,
    })
}

fn dimension(field: &'static str, value: usize) -> Result<i32, Fp8MetalError> {
    i32::try_from(value).map_err(|_| Fp8MetalError::TooLarge { field, value })
}

fn exact_length(field: &'static str, expected: usize, actual: usize) -> Result<(), Fp8MetalError> {
    if expected == actual {
        Ok(())
    } else {
        Err(Fp8MetalError::Length {
            field,
            expected,
            actual,
        })
    }
}

fn finite_e4m3(field: &'static str, codes: &[u8]) -> Result<(), Fp8MetalError> {
    match codes.iter().position(|&code| code & 0x7f == 0x7f) {
        Some(index) => Err(Fp8MetalError::NonFinite { field, index }),
        None => Ok(()),
    }
}

fn decode_scales(field: &'static str, scales: &[u8]) -> Result<Vec<f32>, Fp8MetalError> {
    scales
        .iter()
        .enumerate()
        .map(|(index, &code)| {
            let value = decode_e8m0(code);
            if value.is_finite() {
                Ok(value)
            } else {
                Err(Fp8MetalError::NonFinite { field, index })
            }
        })
        .collect()
}

/// Owns an `mlx_fast_metal_kernel` and frees it on drop.
struct KernelHandle(sys::mlx_fast_metal_kernel);

/// Owns an `mlx_vector_string` and frees it on drop.
struct StringVector(sys::mlx_vector_string);

impl StringVector {
    fn new(items: &[&str]) -> Result<Self, Fp8MetalError> {
        // SAFETY: `mlx_vector_string_new` has no preconditions and returns an owned handle.
        let vector = Self(unsafe { sys::mlx_vector_string_new() });
        for item in items {
            let text =
                CString::new(*item).map_err(|error| Fp8MetalError::Mlx(error.to_string()))?;
            // SAFETY: `vector.0` is a live handle and `text` outlives the call; MLX copies the string.
            check(unsafe { sys::mlx_vector_string_append_value(vector.0, text.as_ptr()) })?;
        }
        Ok(vector)
    }
}

impl Drop for StringVector {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `mlx_vector_string_new` and is freed once.
        unsafe {
            sys::mlx_vector_string_free(self.0);
        }
    }
}

/// Owns an `mlx_vector_array` and frees it on drop.
struct ArrayVector(sys::mlx_vector_array);

impl Drop for ArrayVector {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `mlx_vector_array_new` and is freed once.
        unsafe {
            sys::mlx_vector_array_free(self.0);
        }
    }
}

/// Owns an `mlx_fast_metal_kernel_config` and frees it on drop.
struct ConfigHandle(sys::mlx_fast_metal_kernel_config);

impl Drop for ConfigHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `mlx_fast_metal_kernel_config_new` and is freed once.
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.0) }
    }
}

/// Installs mlx-rs's MLX error handler before any raw MLX-C call.
///
/// MLX-C's default handler prints the message and calls `exit(-1)`, so a
/// rejected kernel launch would end the process instead of returning a status.
/// mlx-rs replaces that handler the first time it runs a guarded op, which
/// this module's raw calls never do; one tiny guarded op forces the install.
fn ensure_mlx_error_handler() -> Result<(), Fp8MetalError> {
    static INSTALLED: OnceLock<Result<(), String>> = OnceLock::new();
    INSTALLED
        .get_or_init(|| {
            Array::zeros::<f32>(&[1])
                .map(drop)
                .map_err(|error| error.to_string())
        })
        .clone()
        .map_err(Fp8MetalError::Mlx)
}

fn check(status: std::os::raw::c_int) -> Result<(), Fp8MetalError> {
    if status == 0 {
        Ok(())
    } else {
        tracing::debug!(status, "MLX-C host call rejected");
        Err(Fp8MetalError::Mlx(format!(
            "MLX-C returned status {status}"
        )))
    }
}

impl KernelHandle {
    // Host construction only: creating a kernel handle is not GPU execution.
    #[tracing::instrument(
        name = "deepseek.metal.kernel_new_host",
        level = "debug",
        skip_all,
        fields(kernel = name, input_count = inputs.len(), output_count = 1, completed = false)
    )]
    fn new(name: &str, inputs: &[&str], output: &str, source: &str) -> Result<Self, Fp8MetalError> {
        ensure_mlx_error_handler()?;
        let name = CString::new(name).map_err(|error| Fp8MetalError::Mlx(error.to_string()))?;
        let source = CString::new(source).map_err(|error| Fp8MetalError::Mlx(error.to_string()))?;
        let header = CString::default();
        let input_names = StringVector::new(inputs)?;
        let output_names = StringVector::new(&[output])?;
        // SAFETY: all pointers are valid NUL-terminated strings and live handles for the call;
        // MLX copies the names and source. The returned handle is owned by `KernelHandle`.
        let kernel = unsafe {
            sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                input_names.0,
                output_names.0,
                source.as_ptr(),
                header.as_ptr(),
                true,
                false,
            )
        };
        if kernel.ctx.is_null() {
            tracing::debug!(null_handle = true, "MLX-C kernel construction rejected");
            return Err(Fp8MetalError::Mlx(
                "kernel construction returned null".to_owned(),
            ));
        }
        tracing::Span::current().record("completed", true);
        Ok(Self(kernel))
    }

    // These spans measure host setup and C graph construction, not GPU duration.
    #[tracing::instrument(
        name = "deepseek.metal.kernel_apply_host",
        level = "debug",
        skip_all,
        fields(
            input_count = inputs.len(),
            output_shape = ?output_shape,
            template_count = template_ints.len(),
            grid = ?grid,
            threadgroup = ?threadgroup,
            completed = false
        )
    )]
    fn apply(
        &self,
        inputs: &[&Array],
        output_shape: &[i32],
        template_ints: &[(&str, i32)],
        grid: [i32; 3],
        threadgroup: [i32; 3],
    ) -> Result<Array, Fp8MetalError> {
        ensure_mlx_error_handler()?;
        // SAFETY: `mlx_fast_metal_kernel_config_new` has no preconditions.
        let config = ConfigHandle(unsafe { sys::mlx_fast_metal_kernel_config_new() });
        // SAFETY: `config.0` is live; the shape slice outlives the call and MLX copies it.
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_add_output_arg(
                config.0,
                output_shape.as_ptr(),
                output_shape.len(),
                Dtype::Float32.into(),
            )
        })?;
        for (name, value) in template_ints {
            let name =
                CString::new(*name).map_err(|error| Fp8MetalError::Mlx(error.to_string()))?;
            // SAFETY: `config.0` is live and `name` outlives the call; MLX copies it.
            check(unsafe {
                sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                    config.0,
                    name.as_ptr(),
                    *value,
                )
            })?;
        }
        // SAFETY: `config.0` is live; grid and threadgroup sizes were validated as positive i32.
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_set_grid(config.0, grid[0], grid[1], grid[2])
        })?;
        // SAFETY: as above.
        check(unsafe {
            sys::mlx_fast_metal_kernel_config_set_thread_group(
                config.0,
                threadgroup[0],
                threadgroup[1],
                threadgroup[2],
            )
        })?;
        // SAFETY: `mlx_vector_array_new` has no preconditions.
        let input_vector = ArrayVector(unsafe { sys::mlx_vector_array_new() });
        for input in inputs {
            // SAFETY: both handles are live; MLX retains its own reference to the array.
            check(unsafe { sys::mlx_vector_array_append_value(input_vector.0, input.as_ptr()) })?;
        }
        let stream = StreamOrDevice::gpu();
        let stream: &Stream = stream.as_ref();
        // SAFETY: `mlx_vector_array_new` has no preconditions; ownership moves into the guard.
        let mut outputs = ArrayVector(unsafe { sys::mlx_vector_array_new() });
        {
            let span = tracing::debug_span!(
                "deepseek.metal.kernel_apply_c_graph",
                status = tracing::field::Empty
            );
            let _entered = span.enter();
            // SAFETY: every handle is live; `outputs.0` receives a new owned vector.
            let status = unsafe {
                sys::mlx_fast_metal_kernel_apply(
                    &raw mut outputs.0,
                    self.0,
                    input_vector.0,
                    config.0,
                    stream.as_ptr(),
                )
            };
            span.record("status", status);
            check(status)?;
        }
        // SAFETY: `mlx_array_new` returns an empty owned array that `vector_array_get` fills.
        let mut result = unsafe { sys::mlx_array_new() };
        // SAFETY: `outputs.0` is live and holds exactly one output array.
        let status = unsafe { sys::mlx_vector_array_get(&raw mut result, outputs.0, 0) };
        // SAFETY: `result` is an owned MLX array handle; `Array` takes ownership and frees it.
        let array = unsafe { Array::from_ptr(result) };
        check(status)?;
        tracing::Span::current().record("completed", true);
        Ok(array)
    }
}

impl Drop for KernelHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `mlx_fast_metal_kernel_new` and is freed once.
        unsafe { sys::mlx_fast_metal_kernel_free(self.0) }
    }
}

#[cfg(test)]
mod tests {
    use mlx_rs::Array;

    use super::{
        Fp4MetalExpert, Fp4MetalKernel, Fp4MetalWeights, Fp8MetalError, Fp8MetalKernel,
        Fp8MetalWeights, KernelHandle,
    };
    use crate::precision::{
        ActivationGroup, fp4_linear_runtime_f32, fp8_linear_runtime_f32,
        quantize_bf16_activations_e4m3fn,
    };

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(*seed >> 33).expect("31-bit value")
    }

    #[test]
    fn matches_scalar_fp8_linear_on_random_finite_codes() {
        let _gpu = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (rows, reduction, outputs): (usize, usize, usize) = (3, 256, 96);
        let mut seed = 17_u64;
        let mut finite_code = || loop {
            let code = u8::try_from(lcg(&mut seed) & 0xff).expect("byte");
            if code & 0x7f != 0x7f {
                break code;
            }
        };
        let weight_codes: Vec<u8> = (0..outputs * reduction).map(|_| finite_code()).collect();
        let mut seed = 99_u64;
        let weight_scales: Vec<u8> = (0..outputs.div_ceil(32) * (reduction / 32))
            .map(|_| 120 + u8::try_from(lcg(&mut seed) % 14).expect("byte"))
            .collect();
        let activation: Vec<u16> = (0..rows * reduction)
            .map(|index| {
                let value = (f32::from(u16::try_from(index % 97).expect("small")) - 48.0) / 7.0;
                u16::try_from(value.to_bits() >> 16).expect("bf16")
            })
            .collect();
        let mut codes = vec![0; rows * reduction];
        let mut scales = vec![0; rows * reduction / 32];
        quantize_bf16_activations_e4m3fn(
            &activation,
            rows,
            reduction,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )
        .expect("activation quantization");
        let mut expected = vec![0.0; rows * outputs];
        fp8_linear_runtime_f32(
            &codes,
            &scales,
            &weight_codes,
            &weight_scales,
            rows,
            reduction,
            outputs,
            ActivationGroup::Elements32,
            &mut expected,
        )
        .expect("scalar reference");
        let weights = Fp8MetalWeights::new(&weight_codes, &weight_scales, outputs, reduction)
            .expect("resident weights");
        let actual = Fp8MetalKernel::new()
            .expect("kernel")
            .forward(&codes, &scales, rows, &weights)
            .expect("metal forward");
        // Only the order of the 32 products inside a block may differ from the
        // serial reference; allow a few FP32 ulps relative to the row scale.
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = want.abs().max(1.0) * 4.0 * f32::EPSILON * 32.0;
            assert!(
                (got - want).abs() <= tolerance,
                "output {index}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn fp4_matches_scalar_fp4_linear_on_every_code() {
        let _gpu = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (rows, reduction, outputs): (usize, usize, usize) = (3, 256, 40);
        let mut seed = 5_u64;
        // Every byte value appears, so every nibble pair is exercised in both positions.
        let weight_codes: Vec<u8> = (0..outputs * reduction / 2)
            .map(|index| u8::try_from((index + lcg(&mut seed) as usize) % 256).expect("byte"))
            .collect();
        let weight_scales: Vec<u8> = (0..outputs * reduction / 32)
            .map(|_| 118 + u8::try_from(lcg(&mut seed) % 18).expect("byte"))
            .collect();
        let activation: Vec<u16> = (0..rows * reduction)
            .map(|index| {
                let value = (f32::from(u16::try_from(index % 89).expect("small")) - 44.0) / 5.0;
                u16::try_from(value.to_bits() >> 16).expect("bf16")
            })
            .collect();
        let mut codes = vec![0; rows * reduction];
        let mut scales = vec![0; rows * reduction / 32];
        quantize_bf16_activations_e4m3fn(
            &activation,
            rows,
            reduction,
            ActivationGroup::Elements32,
            &mut codes,
            &mut scales,
        )
        .expect("activation quantization");
        let mut expected = vec![0.0; rows * outputs];
        fp4_linear_runtime_f32(
            &codes,
            &scales,
            &weight_codes,
            &weight_scales,
            rows,
            reduction,
            outputs,
            ActivationGroup::Elements32,
            &mut expected,
        )
        .expect("scalar reference");
        let weights = Fp4MetalWeights::new(&weight_codes, &weight_scales, outputs, reduction)
            .expect("resident weights");
        let actual = Fp4MetalKernel::new()
            .expect("kernel")
            .forward(&codes, &scales, rows, &weights)
            .expect("metal forward");
        let kernel = Fp4MetalKernel::new().expect("kernel");
        let half = Fp4MetalWeights::new(
            &weight_codes[..outputs / 2 * reduction / 2],
            &weight_scales[..outputs / 2 * reduction / 32],
            outputs / 2,
            reduction,
        )
        .expect("half weights");
        let batched = kernel
            .forward_many(&codes, &scales, rows, &[&weights, &half])
            .expect("batched forward");
        assert_eq!(batched.len(), 2);
        assert_eq!(
            batched[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "one submission must equal separate calls bit for bit"
        );
        let separate = kernel
            .forward(&codes, &scales, rows, &half)
            .expect("separate forward");
        assert_eq!(
            batched[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            separate.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // Block order and scale placement match the scalar reference; only the
        // order of the 32 products inside a block may differ.
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            let tolerance = want.abs().max(1.0) * 4.0 * f32::EPSILON * 32.0;
            assert!(
                (got - want).abs() <= tolerance,
                "output {index}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn whole_experts_match_the_scalar_expert_on_bf16_outputs() {
        let _gpu = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let (dim, inter) = (128_usize, 96_usize);
        let mut seed = 23_u64;
        let mut bytes = |count: usize| -> Vec<u8> {
            (0..count)
                .map(|_| u8::try_from(lcg(&mut seed) & 0xff).expect("byte"))
                .collect()
        };
        let mut expert_codes = Vec::new();
        for _ in 0..2 {
            expert_codes.push((
                bytes(inter * dim / 2),
                bytes(dim * inter / 2),
                bytes(inter * dim / 2),
            ));
        }
        let mut seed = 31_u64;
        let mut scales = |count: usize| -> Vec<u8> {
            (0..count)
                .map(|_| 120 + u8::try_from(lcg(&mut seed) % 10).expect("byte"))
                .collect()
        };
        let expert_scales: Vec<_> = (0..2)
            .map(|_| {
                (
                    scales(inter * dim / 32),
                    scales(dim * inter / 32),
                    scales(inter * dim / 32),
                )
            })
            .collect();
        let input: Vec<u16> = (0..dim)
            .map(|index| {
                let value = (f32::from(u16::try_from(index % 61).expect("small")) - 30.0) / 3.0;
                u16::try_from(value.to_bits() >> 16).expect("bf16")
            })
            .collect();
        let (limit, route_weights) = (2_u8, [0.75_f32, 1.25]);
        let kernel = Fp4MetalKernel::new().expect("kernel");
        let experts: Vec<Fp4MetalExpert> = expert_codes
            .iter()
            .zip(&expert_scales)
            .map(|((w1, w2, w3), (s1, s2, s3))| Fp4MetalExpert {
                w1: Fp4MetalWeights::new(w1, s1, inter, dim).expect("w1"),
                w3: Fp4MetalWeights::new(w3, s3, inter, dim).expect("w3"),
                w2: Fp4MetalWeights::new(w2, s2, dim, inter).expect("w2"),
            })
            .collect();
        let mut codes = vec![0; dim];
        let mut activation_scales = vec![0; dim / 32];
        quantize_bf16_activations_e4m3fn(
            &input,
            1,
            dim,
            ActivationGroup::Elements32,
            &mut codes,
            &mut activation_scales,
        )
        .expect("activation quantization");
        let weights: Vec<[f32; 1]> = route_weights.iter().map(|&w| [w]).collect();
        let actual = kernel
            .experts_forward(
                &codes,
                &activation_scales,
                1,
                &experts.iter().collect::<Vec<_>>(),
                &weights.iter().map(<[f32; 1]>::as_slice).collect::<Vec<_>>(),
                limit,
            )
            .expect("device experts");
        for (index, (((w1, w2, w3), (s1, s2, s3)), route)) in expert_codes
            .iter()
            .zip(&expert_scales)
            .zip(route_weights)
            .enumerate()
        {
            let expected = crate::moe::Fp4ExpertWeights::new(dim, inter, w1, s1, w2, s2, w3, s3)
                .expect("scalar expert")
                .forward_token(&input, f32::from(limit), Some(route))
                .expect("scalar forward");
            let got: Vec<u16> = actual[index]
                .iter()
                .map(|&value| crate::precision::f32_to_bf16_rne(value))
                .collect();
            assert_eq!(got, expected, "expert {index} BF16 outputs");
        }
    }

    #[test]
    fn rejects_invalid_inputs_before_device_work() {
        assert!(matches!(
            Fp4MetalWeights::new(&[0; 32], &[127], 1, 32),
            Err(Fp8MetalError::Length {
                field: "weight codes",
                ..
            })
        ));
        assert!(matches!(
            Fp8MetalWeights::new(&[0; 31], &[127], 1, 31),
            Err(Fp8MetalError::Geometry { .. })
        ));
        assert!(matches!(
            Fp8MetalWeights::new(&[0; 64], &[127], 1, 32),
            Err(Fp8MetalError::Length {
                field: "weight codes",
                ..
            })
        ));
        assert!(matches!(
            Fp8MetalWeights::new(&[0x7f; 32], &[127], 1, 32),
            Err(Fp8MetalError::NonFinite {
                field: "weight codes",
                index: 0
            })
        ));
        assert!(matches!(
            Fp8MetalWeights::new(&[0; 32], &[0xff], 1, 32),
            Err(Fp8MetalError::NonFinite {
                field: "weight scales",
                index: 0
            })
        ));
    }

    // Run alone with `--exact` to exercise a process where nothing else has
    // installed the handler yet; without the install this exits with status 255.
    #[test]
    fn rejected_kernel_launch_returns_an_error_instead_of_exiting() {
        let _gpu = crate::GPU_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let kernel = KernelHandle::new(
            "metallix_two_input_probe",
            &["left", "right"],
            "out",
            "out[0] = left[0] + right[0];",
        )
        .expect("kernel object");
        let left = Array::from_slice(&[1.0_f32], &[1]);
        let launched = kernel.apply(&[&left], &[1], &[], [1, 1, 1], [1, 1, 1]);
        assert!(
            matches!(launched, Err(Fp8MetalError::Mlx(_))),
            "{launched:?}"
        );
    }
}
