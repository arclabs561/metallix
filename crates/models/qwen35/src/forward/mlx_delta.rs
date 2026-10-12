//! Scalar, unmasked mlx-lm 0.32.0 recurrence. The adjacent Metal bodies retain
//! Apple's original operation order; see third_party/LICENSE-MLX-LM.txt.
//! Native state stays [Hv, Dk, Dv]; this boundary uses [Hv, Dv, Dk],
//! the source's single-batch [1, Hv, Dv, Dk] storage order without a batch axis.

use std::cell::{OnceCell, RefCell};

use mlx_kernels::{Grid, KernelSpec, Launch, MetalKernel, Output, TemplateArg, ThreadGroup};
use mlx_rs::{Array, Dtype, StreamOrDevice, ops, ops::indexing::IndexOp};

use super::{CompiledGraph, Qwen35Error};

thread_local! {
    static PACKED: OnceCell<MetalKernel> = const { OnceCell::new() };
    static SCALAR: OnceCell<MetalKernel> = const { OnceCell::new() };
    static STEP: RefCell<Option<CompiledGraph>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Packed,
    Scalar,
    Ops,
}

fn path(dk: i32, dv: i32) -> Path {
    if dk < 32 || dk % 32 != 0 {
        Path::Ops
    } else if dk == 128 && dv % 8 == 0 {
        Path::Packed
    } else {
        Path::Scalar
    }
}

/// Matches the pinned reference's default packed-enabled policy. No ambient
/// environment switch changes a checkpoint's numerical policy in this adapter.
pub(super) fn update(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array), Qwen35Error> {
    let [seq, hk, dk, hv, dv] = admit(q, k, v, g, beta, state)?;
    let gpu = StreamOrDevice::gpu();
    let state = state.transpose_axes_device(&[0, 2, 1], &gpu)?;
    let (output, state) = match path(dk, dv) {
        Path::Ops => ops_update(q, k, v, g, beta, state)?,
        selected => {
            let cell = if selected == Path::Packed {
                &PACKED
            } else {
                &SCALAR
            };
            cell.with(|cell| {
                let kernel = if let Some(kernel) = cell.get() {
                    kernel
                } else {
                    let (name, source) = if selected == Path::Packed {
                        (
                            "metallix_qwen35_delta_packed",
                            include_str!("mlx_delta/packed.metal"),
                        )
                    } else {
                        (
                            "metallix_qwen35_delta_scalar",
                            include_str!("mlx_delta/scalar.metal"),
                        )
                    };
                    let kernel = MetalKernel::new(&KernelSpec::new(
                        name,
                        &["q", "k", "v", "g", "beta", "state_in", "T"],
                        &["y", "state_out"],
                        source,
                    ))?;
                    cell.get_or_init(|| kernel)
                };
                let mut outputs = kernel.apply(
                    &[q, k, v, g, beta, &state, &Array::from_int(seq)],
                    &Launch {
                        outputs: &[
                            Output::new(
                                &[
                                    usize::try_from(seq).map_err(|_| Qwen35Error::ShapeOverflow)?,
                                    usize::try_from(hv).map_err(|_| Qwen35Error::ShapeOverflow)?,
                                    usize::try_from(dv).map_err(|_| Qwen35Error::ShapeOverflow)?,
                                ],
                                q.dtype(),
                            )?,
                            Output::new(
                                &[
                                    usize::try_from(hv).map_err(|_| Qwen35Error::ShapeOverflow)?,
                                    usize::try_from(dv).map_err(|_| Qwen35Error::ShapeOverflow)?,
                                    usize::try_from(dk).map_err(|_| Qwen35Error::ShapeOverflow)?,
                                ],
                                Dtype::Float32,
                            )?,
                        ],
                        template: &[
                            ("InT", TemplateArg::Dtype(q.dtype())),
                            ("StT", TemplateArg::Dtype(Dtype::Float32)),
                            ("Dk", TemplateArg::Int(dk)),
                            ("Dv", TemplateArg::Int(dv)),
                            ("Hk", TemplateArg::Int(hk)),
                            ("Hv", TemplateArg::Int(hv)),
                        ],
                        grid: Grid::new(
                            32,
                            if selected == Path::Packed {
                                usize::try_from(dv).map_err(|_| Qwen35Error::ShapeOverflow)? / 8
                            } else {
                                usize::try_from(dv).map_err(|_| Qwen35Error::ShapeOverflow)?
                            },
                            usize::try_from(hv).map_err(|_| Qwen35Error::ShapeOverflow)?,
                        )?,
                        threadgroup: ThreadGroup::new(
                            32,
                            if selected == Path::Packed { 2 } else { 4 },
                            1,
                        )?,
                        init_value: None,
                    },
                )?;
                let state = outputs.pop().ok_or(Qwen35Error::ShapeOverflow)?;
                let output = outputs.pop().ok_or(Qwen35Error::ShapeOverflow)?;
                Ok::<_, Qwen35Error>((output, state))
            })?
        }
    };
    Ok((output, state.transpose_axes_device(&[0, 2, 1], &gpu)?))
}

/// Reject incompatible buffers before constructing a kernel or grid. All kernel
/// offsets use signed 32-bit arithmetic, so bound products as well as extents.
fn admit(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<[i32; 5], Qwen35Error> {
    if q.ndim() != 3 || v.ndim() != 3 {
        return Err(Qwen35Error::ShapeOverflow);
    }
    let [seq, hk, dk] = <[i32; 3]>::try_from(q.shape()).map_err(|_| Qwen35Error::ShapeOverflow)?;
    let hv = v.shape()[1];
    let dv = v.shape()[2];
    if seq == 0 {
        return Err(Qwen35Error::EmptyInput);
    }
    if [seq, hk, dk, hv, dv].iter().any(|&n| n <= 0) || hv % hk != 0 {
        return Err(Qwen35Error::ShapeOverflow);
    }
    for (name, array, expected) in [
        ("delta.k", k, vec![seq, hk, dk]),
        ("delta.v", v, vec![seq, hv, dv]),
        ("delta.g", g, vec![seq, hv]),
        ("delta.beta", beta, vec![seq, hv]),
        ("delta.state", state, vec![hv, dk, dv]),
    ] {
        if array.shape() != expected {
            return Err(Qwen35Error::TensorShape {
                name: name.into(),
                expected,
                actual: array.shape().to_vec(),
            });
        }
    }
    if !matches!(q.dtype(), Dtype::Float32 | Dtype::Float16 | Dtype::Bfloat16)
        || [k, v, beta].iter().any(|a| a.dtype() != q.dtype())
        || g.dtype() != Dtype::Float32
        || state.dtype() != Dtype::Float32
    {
        return Err(mlx_kernels::KernelError::Mlx("recurrence input dtype mismatch".into()).into());
    }
    for dims in [[seq, hk, dk], [seq, hv, dv], [hv, dk, dv], [seq, hv, dk]] {
        dims.into_iter()
            .try_fold(1_i32, i32::checked_mul)
            .ok_or(Qwen35Error::ShapeOverflow)?;
    }
    Ok([seq, hk, dk, hv, dv])
}

/// Source fallback: compiled per-token elementwise reductions, never chunked.
fn ops_update(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    mut state: Array,
) -> Result<(Array, Array), Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let repeat = v.shape()[1] / q.shape()[1];
    let q = ops::repeat_axis_device::<f32>(q.clone(), repeat, 1, &gpu)?;
    let k = ops::repeat_axis_device::<f32>(k.clone(), repeat, 1, &gpu)?;
    let mut rows = Vec::new();
    for t in 0..q.shape()[0] {
        let mut outputs = STEP.with(|cell| {
            let mut cell = cell.borrow_mut();
            let compiled = cell.get_or_insert_with(|| {
                Box::new(mlx_rs::transforms::compile::compile(
                    |args: &[Array]| {
                        let [q, k, v, g, beta, state] = args else {
                            unreachable!()
                        };
                        let state = state.multiply(g.reshape(&[-1, 1, 1])?)?;
                        let key = k.expand_dims(-2)?;
                        let predicted = state.multiply(&key)?.sum_axis(-1, false)?;
                        let delta = v.subtract(predicted)?.multiply(beta.expand_dims(-1)?)?;
                        let state = state.add(key.multiply(delta.expand_dims(-1)?)?)?;
                        let output = state
                            .multiply(q.expand_dims(-2)?)?
                            .sum_axis(-1, false)?
                            .as_dtype(q.dtype())?;
                        Ok(vec![output, state])
                    },
                    false,
                ))
            });
            compiled(&[
                q.index(t),
                k.index(t),
                v.index(t),
                g.index(t),
                beta.index(t),
                state.clone(),
            ])
        })?;
        state = outputs.pop().ok_or(Qwen35Error::ShapeOverflow)?;
        rows.push(outputs.pop().ok_or(Qwen35Error::ShapeOverflow)?);
    }
    Ok((ops::stack_axis_device(&rows, 0, &gpu)?, state))
}

#[cfg(test)]
mod tests;
