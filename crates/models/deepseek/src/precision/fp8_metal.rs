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

use mlx_rs::{Array, Dtype, Stream, StreamOrDevice};
use mlx_sys as sys;
use thiserror::Error;

use super::{decode_e4m3fn, decode_e8m0};

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

/// An invalid fused FP8 linear request or MLX failure.
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
        let lut: Vec<f32> = (0..=u8::MAX)
            .map(|code| {
                let value = decode_e4m3fn(code);
                // NaN codes are rejected before upload; keep the table finite.
                if value.is_finite() { value } else { 0.0 }
            })
            .collect();
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
            lut: Array::from_slice(&lut, &[256]),
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
        let reduction = weights.reduction;
        let outputs = weights.outputs;
        if rows == 0 {
            return Err(Fp8MetalError::Geometry {
                rows,
                reduction,
                outputs,
            });
        }
        let code_count = checked(rows, reduction, "activation codes")?;
        let scale_count = checked(rows, reduction / GROUP, "activation scales")?;
        let output_count = checked(rows, outputs, "output")?;
        exact_length("activation codes", code_count, activation_codes.len())?;
        exact_length("activation scales", scale_count, activation_scales.len())?;
        finite_e4m3("activation codes", activation_codes)?;
        let decoded_scales = decode_scales("activation scales", activation_scales)?;
        let codes = Array::from_slice(
            activation_codes,
            &[dimension("activation codes", code_count)?],
        );
        let scales = Array::from_slice(
            &decoded_scales,
            &[dimension("activation scales", scale_count)?],
        );
        let grid_x = dimension("output columns", checked(outputs, GROUP, "grid")?)?;
        let output = self.kernel.apply(
            &[&codes, &scales, &weights.codes, &weights.scales, &self.lut],
            &[dimension("output", output_count)?],
            &[
                ("N", dimension("outputs", outputs)?),
                ("K", dimension("reduction", reduction)?),
            ],
            [grid_x, dimension("rows", rows)?, 1],
            [dimension("simd width", GROUP)?, 1, 1],
        )?;
        output
            .eval()
            .map_err(|error| Fp8MetalError::Mlx(error.to_string()))?;
        let values = output.as_slice::<f32>().to_vec();
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(Fp8MetalError::NonFiniteOutput { index });
        }
        Ok(values)
    }
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

fn check(status: std::os::raw::c_int) -> Result<(), Fp8MetalError> {
    if status == 0 {
        Ok(())
    } else {
        Err(Fp8MetalError::Mlx(format!(
            "MLX-C returned status {status}"
        )))
    }
}

impl KernelHandle {
    fn new(name: &str, inputs: &[&str], output: &str, source: &str) -> Result<Self, Fp8MetalError> {
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
            return Err(Fp8MetalError::Mlx(
                "kernel construction returned null".to_owned(),
            ));
        }
        Ok(Self(kernel))
    }

    fn apply(
        &self,
        inputs: &[&Array],
        output_shape: &[i32],
        template_ints: &[(&str, i32)],
        grid: [i32; 3],
        threadgroup: [i32; 3],
    ) -> Result<Array, Fp8MetalError> {
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
        // SAFETY: every handle is live for the call; `outputs.0` receives a new owned vector.
        check(unsafe {
            sys::mlx_fast_metal_kernel_apply(
                &raw mut outputs.0,
                self.0,
                input_vector.0,
                config.0,
                stream.as_ptr(),
            )
        })?;
        // SAFETY: `mlx_array_new` returns an empty owned array that `vector_array_get` fills.
        let mut result = unsafe { sys::mlx_array_new() };
        // SAFETY: `outputs.0` is live and holds exactly one output array.
        let status = unsafe { sys::mlx_vector_array_get(&raw mut result, outputs.0, 0) };
        // SAFETY: `result` is an owned MLX array handle; `Array` takes ownership and frees it.
        let array = unsafe { Array::from_ptr(result) };
        check(status)?;
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
    use super::{Fp8MetalError, Fp8MetalKernel, Fp8MetalWeights};
    use crate::precision::{
        ActivationGroup, fp8_linear_runtime_f32, quantize_bf16_activations_e4m3fn,
    };

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(*seed >> 33).expect("31-bit value")
    }

    #[test]
    fn matches_scalar_fp8_linear_on_random_finite_codes() {
        let _gpu = crate::GPU_TEST_LOCK.lock().expect("GPU test lock");
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
    fn rejects_invalid_inputs_before_device_work() {
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
}
