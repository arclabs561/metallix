//! Routed experts read on demand: the forward pass's side of
//! [`checkpoint::expert`].
//!
//! Per `MoE` call the router's choices are read back to the host, the
//! call's unique experts are made resident (cache hits, or reads through the
//! pager). The execution stack retains the original expert axis and router IDs,
//! with zeros in unselected positions: MLX uses the stored expert count for
//! kernel and tile selection, so compacting that axis can change arithmetic.
//! Bit-identical outputs remain the qualification target. The cache holds only
//! fetched experts; temporary execution stacks have a full layer's shape and
//! are outside the cache byte budget.

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Instant,
};

use checkpoint::{
    DType, ReadCache, TensorEncoding, TensorInfo,
    expert::{ExpertCache, ExpertKey, ExpertLocator, ExpertPager},
    safetensors::SafetensorsCheckpoint,
};
use mlx_rs::{Array, Dtype, StreamOrDevice, ops};

use super::{AffineWeight, Matrix, Qwen35Error, Routing};
use crate::experts::Qwen35ExpertLocator;

/// How a paged load reads routed experts.
#[derive(Clone, Copy, Debug)]
pub struct Qwen35Paging {
    /// Logical bytes of routed experts retained in the cache. Must hold at
    /// least one layer's full set, since one prefill call can route to every
    /// expert. Excludes temporary stacks with a full layer's shape, in-flight
    /// graphs, other model weights, allocator retention and the OS file cache;
    /// this is not a process cap.
    pub budget_bytes: u64,
    /// Reader threads (requests in flight at the SSD).
    pub readers: usize,
    /// Whether reads request an OS file-cache bypass. [`ReadCache::Bypass`]
    /// does not establish physical SSD traffic or a process-memory cap.
    pub cache: ReadCache,
}

/// Counters since load.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Qwen35PagerStats {
    /// `MoE` calls served.
    pub calls: u64,
    /// Unique experts those calls needed.
    pub experts: u64,
    /// Of those, already resident.
    pub hits: u64,
    /// Of those, read from the checkpoint.
    pub misses: u64,
    /// Bytes read.
    pub bytes_read: u64,
    /// Wall time inside reads.
    pub read_seconds: f64,
    /// Wall time waiting for the router's choices to reach the host.
    pub route_seconds: f64,
    /// Resident expert bytes now.
    pub resident_bytes: u64,
}

/// One part's layout: its per-expert shape `[1, ..]` and element type.
#[derive(Clone, Debug)]
struct PartLayout {
    shape: Vec<i32>,
    dtype: DType,
}

/// One resident expert: [`crate::experts::PARTS`] of each projection.
/// Arrays are reference-counted handles, so cloning one shares its buffer.
type Resident = [Array; 9];

struct State {
    cache: ExpertCache<Resident>,
    stats: Qwen35PagerStats,
}

/// Routed experts behind a byte budget.
pub(super) struct PagedExperts {
    locator: Qwen35ExpertLocator,
    pager: ExpertPager<SafetensorsCheckpoint>,
    /// Per layer, the nine parts' layouts.
    layouts: Vec<[PartLayout; 9]>,
    /// Per layer, each projection's bits and group size.
    affine: Vec<[(i32, i32); 3]>,
    state: Mutex<State>,
}

/// A layer's routed projections with the original logical expert axis.
pub(super) struct ExpertStack {
    projections: [AffineWeight; 3],
}

impl ExpertStack {
    pub(super) fn matrices(&self) -> [Matrix<'_>; 3] {
        let [gate, up, down] = &self.projections;
        [
            Matrix::Affine(gate),
            Matrix::Affine(up),
            Matrix::Affine(down),
        ]
    }
}

impl PagedExperts {
    /// Opens `model_dir`'s shards for paging the routed experts of `layers`
    /// layers with `experts` each.
    pub(super) fn open(
        model_dir: &std::path::Path,
        layers: usize,
        experts: usize,
        paging: Qwen35Paging,
    ) -> Result<Self, Qwen35Error> {
        let source = Arc::new(SafetensorsCheckpoint::open(model_dir, paging.cache)?);
        let locator = Qwen35ExpertLocator::mlx_affine(&*source, layers, experts)?;
        let mut layouts = Vec::with_capacity(layers);
        let mut affine = Vec::with_capacity(layers);
        let mut needed = 0;
        for layer in 0..layers {
            let mut parts = Vec::with_capacity(9);
            let mut settings = [(0, 0); 3];
            for index in 0..9 {
                let info = locator
                    .part(layer, index)
                    .ok_or(Qwen35Error::ShapeOverflow)?;
                if let TensorEncoding::Affine { bits, group_size } = info.encoding() {
                    settings[index / 3] = (i32::from(bits), i32::from(group_size));
                }
                parts.push(part_layout(info)?);
            }
            layouts.push(
                parts
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("nine parts")),
            );
            affine.push(settings);
            needed = needed
                .max(locator.expert_bytes(layer) * u64::try_from(experts).unwrap_or(u64::MAX));
        }
        if paging.budget_bytes < needed {
            return Err(Qwen35Error::PagerBudget {
                budget: paging.budget_bytes,
                needed,
            });
        }
        Ok(Self {
            locator,
            pager: ExpertPager::new(source, paging.readers),
            layouts,
            affine,
            state: Mutex::new(State {
                cache: ExpertCache::new(paging.budget_bytes),
                stats: Qwen35PagerStats::default(),
            }),
        })
    }

    pub(super) fn stats(&self) -> Qwen35PagerStats {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        Qwen35PagerStats {
            resident_bytes: state.cache.used_bytes(),
            ..state.stats
        }
    }

    /// The chosen experts at their original positions in the logical expert
    /// axis, and unchanged routing. Unselected positions contain zeros.
    pub(super) fn stack(
        &self,
        layer: usize,
        routing: &Routing,
    ) -> Result<(ExpertStack, Routing), Qwen35Error> {
        let started = Instant::now();
        // `chosen` is a slice of the partitioned indices, not contiguous.
        let host = routing.chosen.contiguous()?;
        host.eval()?;
        let chosen: Vec<u32> = host.as_slice::<u32>().to_vec();
        let route_seconds = started.elapsed().as_secs_f64();

        let layer_key = u16::try_from(layer).map_err(|_| Qwen35Error::ShapeOverflow)?;
        let mut unique = chosen.clone();
        unique.sort_unstable();
        unique.dedup();
        let keys = unique
            .iter()
            .map(|&expert| {
                Ok(ExpertKey {
                    layer: layer_key,
                    expert: u16::try_from(expert).map_err(|_| Qwen35Error::ShapeOverflow)?,
                })
            })
            .collect::<Result<Vec<_>, Qwen35Error>>()?;

        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let wanted = keys
            .iter()
            .filter(|&&key| !state.cache.contains(key))
            .map(|&key| Ok((key, self.locator.ranges(key)?)))
            .collect::<Result<Vec<_>, Qwen35Error>>()?;
        let misses = wanted.len();
        let read_started = Instant::now();
        let fetched = self.pager.fetch(wanted)?;
        let read_seconds = read_started.elapsed().as_secs_f64();
        let mut bytes_read = 0;
        for expert in fetched {
            let bytes: u64 = expert.parts.iter().map(|part| part.len() as u64).sum();
            bytes_read += bytes;
            let arrays = self.materialize(layer, &expert.parts)?;
            state
                .cache
                .insert(expert.key, arrays, bytes, &keys)
                .map_err(|error| Qwen35Error::Pager(error.to_string()))?;
        }
        let resident = keys
            .iter()
            .map(|&key| {
                state
                    .cache
                    .get(key)
                    .cloned()
                    .ok_or_else(|| Qwen35Error::Pager(format!("{key:?} not resident")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let counters = &mut state.stats;
        counters.calls += 1;
        counters.experts += keys.len() as u64;
        counters.misses += misses as u64;
        counters.hits += (keys.len() - misses) as u64;
        counters.bytes_read += bytes_read;
        counters.read_seconds += read_seconds;
        counters.route_seconds += route_seconds;
        drop(state);

        let part = |index: usize| -> Result<Array, Qwen35Error> {
            let arrays: Vec<&Array> = resident.iter().map(|expert| &expert[index]).collect();
            logical_expert_stack(&unique, &arrays, self.locator.experts())
        };
        let mut projections = Vec::with_capacity(3);
        for (projection, &(bits, group_size)) in self.affine[layer].iter().enumerate() {
            projections.push(AffineWeight {
                codes: part(3 * projection)?,
                scales: part(3 * projection + 1)?,
                biases: part(3 * projection + 2)?,
                bits,
                group_size,
            });
        }
        Ok((
            ExpertStack {
                projections: projections
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("three projections")),
            },
            Routing {
                chosen: routing.chosen.clone(),
                weights: routing.weights.clone(),
            },
        ))
    }

    /// One expert's nine parts as evaluated device arrays `[1, ..]`.
    fn materialize(&self, layer: usize, parts: &[Vec<u8>]) -> Result<Resident, Qwen35Error> {
        let gpu = StreamOrDevice::gpu();
        let mut arrays = Vec::with_capacity(9);
        for (bytes, layout) in parts.iter().zip(&self.layouts[layer]) {
            let array = match layout.dtype {
                DType::U32 => {
                    Array::from_slice(&words::<4, u32>(bytes, u32::from_le_bytes), &layout.shape)
                }
                DType::F32 => {
                    Array::from_slice(&words::<4, f32>(bytes, f32::from_le_bytes), &layout.shape)
                }
                DType::BF16 | DType::F16 => {
                    let dtype = if layout.dtype == DType::BF16 {
                        Dtype::Bfloat16
                    } else {
                        Dtype::Float16
                    };
                    Array::from_slice(&words::<2, u16>(bytes, u16::from_le_bytes), &layout.shape)
                        .view_dtype_device(dtype, &gpu)?
                }
                other => {
                    return Err(Qwen35Error::Encoding {
                        tensor: format!("routed expert part of layer {layer}"),
                        encoding: TensorEncoding::Plain(other),
                    });
                }
            };
            arrays.push(array);
        }
        mlx_rs::transforms::eval(arrays.iter())?;
        Ok(arrays
            .try_into()
            .unwrap_or_else(|_| unreachable!("nine parts")))
    }
}

/// Little-endian words of `bytes`.
fn words<const N: usize, T>(bytes: &[u8], from: fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|chunk| from(chunk.try_into().unwrap_or_else(|_| unreachable!())))
        .collect()
}

/// A stored part's per-expert shape `[1, ..]` and its element type (`u32`
/// for affine codes).
fn part_layout(info: &TensorInfo) -> Result<PartLayout, Qwen35Error> {
    let dtype = match info.encoding() {
        TensorEncoding::Affine { .. } => DType::U32,
        TensorEncoding::Plain(dtype) => dtype,
        other @ TensorEncoding::Gguf(_) => {
            return Err(Qwen35Error::Encoding {
                tensor: "routed expert part".to_owned(),
                encoding: other,
            });
        }
    };
    let mut shape = vec![1];
    for &dim in &info.shape()[1..] {
        shape.push(i32::try_from(dim).map_err(|_| Qwen35Error::ShapeOverflow)?);
    }
    Ok(PartLayout { shape, dtype })
}

/// Preserve backend dispatch dimensions while materializing only selected
/// experts. Inputs are ascending unique IDs and matching single-expert arrays.
/// Zero runs use the part's dtype, including integer affine codes.
pub(super) fn logical_expert_stack(
    selected: &[u32],
    arrays: &[&Array],
    experts: usize,
) -> Result<Array, Qwen35Error> {
    let gpu = StreamOrDevice::gpu();
    let first = arrays
        .first()
        .ok_or_else(|| Qwen35Error::Pager("empty expert selection".into()))?;
    let mut parts = Vec::with_capacity(2 * arrays.len() + 1);
    let mut next = 0;
    let zeros = |count: usize| -> Result<Array, Qwen35Error> {
        let mut shape = first.shape().to_vec();
        shape[0] = i32::try_from(count).map_err(|_| Qwen35Error::ShapeOverflow)?;
        Ok(ops::zeros_dtype_device(&shape, first.dtype(), &gpu)?)
    };
    for (&index, &array) in selected.iter().zip(arrays) {
        let index = usize::try_from(index).map_err(|_| Qwen35Error::ShapeOverflow)?;
        if index > next {
            parts.push(zeros(index - next)?);
        }
        parts.push(array.clone());
        next = index + 1;
    }
    if next < experts {
        parts.push(zeros(experts - next)?);
    }
    Ok(ops::concatenate_axis_device(
        &parts.iter().collect::<Vec<_>>(),
        0,
        &gpu,
    )?)
}
