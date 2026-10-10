//! A few-row affine 4-bit matmul: 2..=16 activation rows share every weight
//! load.
//!
//! MLX's `quantized_matmul` runs few-row inputs as matrix-vector products,
//! one activation row per grid slice, so each extra row (a speculative draft
//! token, a batched decode row) re-reads the whole weight matrix. This kernel
//! (`small_m/qmv.metal`) reads each packed weight once for up to 16 rows, on
//! 8x8 simdgroup matrices, following llama.cpp's few-row MMA mat-mul.
//!
//! One row stays on MLX (bit-identical to the unrouted path), as do 6- and
//! 8-bit weights and more than 16 rows. Routing is off by default
//! ([`set_small_m_qmv`]) until timing on an idle machine confirms the speed
//! bars; results match MLX to FP32 rounding, not bit for bit.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the with_stream migration is a separate change"
)]

use std::{
    cell::OnceCell,
    sync::atomic::{AtomicBool, Ordering},
};

use mlx_kernels::{Grid, KernelSpec, Launch, MetalKernel, Output, TemplateArg, ThreadGroup};
use mlx_rs::{Array, Dtype, StreamOrDevice};

use super::{Qwen3ForwardError, as_i32};
use crate::quantization::{Qwen3AffineBits, Qwen3AffineQuantization};

const SOURCE: &str = include_str!("small_m/qmv.metal");

/// Most activation rows one launch takes.
pub(super) const MAX_ROWS: usize = 16;
/// Values per K chunk a simdgroup takes at a time.
const CHUNK: usize = 32;
/// Output tiles (8 columns each) per threadgroup.
const TILES: usize = 2;
/// Simdgroups per threadgroup, splitting K.
const SIMDGROUPS: usize = 4;

static ENABLED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static KERNEL: OnceCell<MetalKernel> = const { OnceCell::new() };
}

/// Sends 4-bit projections of 2..=16 rows to the few-row kernel instead of
/// MLX's `quantized_matmul` (off by default). Process-wide.
pub fn set_small_m_qmv(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether `rows` rows of `input` against `quantization` weights go to the
/// kernel: routing is on, the weights are 4-bit, there are 2..=16 rows, and
/// K is a whole number of 32-value chunks.
pub(super) fn applies(input: &Array, rows: usize, quantization: Qwen3AffineQuantization) -> bool {
    ENABLED.load(Ordering::Relaxed)
        && quantization.bits == Qwen3AffineBits::Four
        && (2..=MAX_ROWS).contains(&rows)
        && input
            .shape()
            .last()
            .is_some_and(|&k| usize::try_from(k).is_ok_and(|k| k > 0 && k.is_multiple_of(CHUNK)))
        && matches!(
            input.dtype(),
            Dtype::Bfloat16 | Dtype::Float16 | Dtype::Float32
        )
}

/// `input @ dequantize(packed, scales, biases).T` for `input` `[.., K]` with
/// 2..=16 rows in total, `packed` `[N, K / 8]` u32 and `scales`/`biases`
/// `[N, K / group_size]` in `input`'s dtype. Returns `[.., N]`.
pub(super) fn matmul(
    input: &Array,
    packed: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Qwen3ForwardError> {
    KERNEL.with(|kernel| {
        let kernel = if let Some(kernel) = kernel.get() {
            kernel
        } else {
            let built = build(SOURCE)?;
            kernel.get_or_init(|| built)
        };
        launch(kernel, input, packed, scales, biases, group_size)
    })
}

/// [`matmul`] with another kernel body; tests use it to check that a broken
/// kernel is caught.
#[cfg(test)]
pub(super) fn matmul_with_source(
    source: &str,
    input: &Array,
    packed: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Qwen3ForwardError> {
    launch(&build(source)?, input, packed, scales, biases, group_size)
}

fn build(source: &str) -> Result<MetalKernel, Qwen3ForwardError> {
    Ok(MetalKernel::new(&KernelSpec::new(
        "metallix_small_m_qmv",
        &["x", "w", "scales", "biases"],
        &["out"],
        source,
    ))?)
}

fn launch(
    kernel: &MetalKernel,
    input: &Array,
    packed: &Array,
    scales: &Array,
    biases: &Array,
    group_size: i32,
) -> Result<Array, Qwen3ForwardError> {
    let stream = StreamOrDevice::gpu();
    let shape = input.shape().to_vec();
    let k = *shape.last().ok_or(Qwen3ForwardError::CacheInconsistent)?;
    let rows = input.size() / usize::try_from(k).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    let n = usize::try_from(packed.shape()[0]).map_err(|_| Qwen3ForwardError::ShapeOverflow)?;
    if !(2..=MAX_ROWS).contains(&rows) {
        return Err(Qwen3ForwardError::CacheInconsistent);
    }
    let x = input.reshape_device(&[as_i32(rows)?, k], &stream)?;
    let output = kernel
        .apply(
            &[&x, packed, scales, biases],
            &Launch {
                outputs: &[Output::new(&[rows, n], input.dtype())?],
                template: &[
                    ("T", TemplateArg::Dtype(input.dtype())),
                    ("M", TemplateArg::Int(as_i32(rows)?)),
                    ("RT", TemplateArg::Int(as_i32(rows.div_ceil(8))?)),
                    ("NT", TemplateArg::Int(as_i32(TILES)?)),
                    ("NSG", TemplateArg::Int(as_i32(SIMDGROUPS)?)),
                    ("GroupSize", TemplateArg::Int(group_size)),
                    ("NOut", TemplateArg::Int(as_i32(n)?)),
                ],
                grid: Grid::new(n.div_ceil(8 * TILES) * 32, SIMDGROUPS, 1)?,
                threadgroup: ThreadGroup::new(32, SIMDGROUPS, 1)?,
                init_value: None,
            },
        )?
        .swap_remove(0);
    let mut out_shape = shape;
    *out_shape
        .last_mut()
        .ok_or(Qwen3ForwardError::CacheInconsistent)? = as_i32(n)?;
    Ok(output.reshape_device(&out_shape, &stream)?)
}
