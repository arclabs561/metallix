//! Device execution of the forward's FP8 linears and routed FP4 experts.
//!
//! A request runs its linears on Metal only while a [`DeviceLinears`] scope is
//! entered on its thread; everywhere else they are scalar. The scope also
//! carries FP8 weights kept resident for the life of the model weights that
//! own them, and counts what ran where. See the crate's `device_lock` docs
//! for the threading assumption.

use std::{
    cell::RefCell,
    collections::HashMap,
    marker::PhantomData,
    sync::{
        Arc, Mutex, PoisonError,
        atomic::{AtomicU64, Ordering},
    },
};

use super::{Fp8MetalError, Fp8MetalKernel, Fp8MetalWeights, metal_fp8_linear};

/// Host FP8 `(codes, scales)` buffers in checkpoint layout.
pub(crate) type Fp8Buffers = (Arc<[u8]>, Arc<[u8]>);

/// FP8 weights uploaded to the device on first use, then kept.
///
/// Owned by long-lived weights, never by per-request state. MLX arrays are
/// `Send` but not `Sync`, so the upload sits behind a lock.
#[derive(Default)]
pub(crate) struct ResidentFp8(Mutex<Option<Fp8MetalWeights>>);

impl std::fmt::Debug for ResidentFp8 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResidentFp8")
            .field("resident", &self.is_resident())
            .finish()
    }
}

impl ResidentFp8 {
    /// Projects activations through the resident copy of `weights`
    /// (`(codes, scales)` in checkpoint layout), uploading them on the first
    /// call. Every call must pass the same weights; only the geometry of a
    /// later call is checked against the upload.
    pub(crate) fn forward(
        &self,
        activations: (&[u8], &[u8]),
        weights: (&[u8], &[u8]),
        rows: usize,
        reduction: usize,
        outputs: usize,
    ) -> Result<Vec<f32>, Fp8MetalError> {
        // The device lock comes before the slot lock, as at every MLX entry point.
        let _device = crate::device_lock();
        // A panic while holding the lock leaves either no upload or a complete one.
        let mut slot = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        let resident = match &mut *slot {
            Some(resident) => resident,
            empty => empty.insert(Fp8MetalWeights::new(
                weights.0, weights.1, outputs, reduction,
            )?),
        };
        if resident.outputs() != outputs || resident.reduction() != reduction {
            return Err(Fp8MetalError::Geometry {
                rows,
                reduction,
                outputs,
            });
        }
        Fp8MetalKernel::new()?.forward(activations.0, activations.1, rows, resident)
    }

    /// Whether the weights have been uploaded.
    pub(crate) fn is_resident(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }
}

/// What a [`DeviceLinears`] scope ran, summed over its lifetime.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct DeviceCounts {
    /// FP8 linears run over resident weights.
    pub resident_fp8: u64,
    /// FP8 linears whose weights were uploaded for that call only.
    pub uploaded_fp8: u64,
    /// Routed FP4 experts run on the device (weights uploaded per call).
    pub fp4_experts: u64,
    /// Routed experts run on the scalar path inside the scope because the
    /// kernel cannot encode their `SwiGLU` limit (not an integer in `0..=255`).
    /// The only fallback: device errors are returned, never rerun.
    pub scalar_fallbacks: u64,
    /// Bytes of resident FP8 weights on the device: one byte per code and
    /// four per decoded block scale.
    pub resident_bytes: u64,
}

/// One FP8 tensor eligible for residency. Holding the host buffers keeps
/// their addresses from being reused while the entry is keyed by them.
struct ResidentEntry {
    codes: Arc<[u8]>,
    scales: Arc<[u8]>,
    weights: ResidentFp8,
}

/// The device execution for one model: resident FP8 weights keyed by the
/// address of their host codes, plus counters. Enter it with [`Self::enter`].
#[derive(Default)]
pub(crate) struct DeviceLinears {
    resident: HashMap<usize, ResidentEntry>,
    resident_fp8: AtomicU64,
    uploaded_fp8: AtomicU64,
    fp4_experts: AtomicU64,
    scalar_fallbacks: AtomicU64,
}

impl std::fmt::Debug for DeviceLinears {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceLinears")
            .field("eligible", &self.resident.len())
            .field("counts", &self.counts())
            .finish_non_exhaustive()
    }
}

impl DeviceLinears {
    /// Makes each `(codes, scales)` pair eligible for residency. A linear
    /// over the same buffers (same address and length) then uploads them once;
    /// any other FP8 linear in the scope uploads its weights per call.
    pub(crate) fn new(fp8: impl IntoIterator<Item = Fp8Buffers>) -> Self {
        let resident = fp8
            .into_iter()
            .map(|(codes, scales)| {
                let entry = ResidentEntry {
                    codes,
                    scales,
                    weights: ResidentFp8::default(),
                };
                (entry.codes.as_ptr().addr(), entry)
            })
            .collect();
        Self {
            resident,
            ..Self::default()
        }
    }

    /// Runs device FP8 and FP4 calls on this thread until the guard drops.
    /// Scopes do not nest (a debug assertion checks), and the guard cannot
    /// leave the thread that entered it.
    pub(crate) fn enter(self: &Arc<Self>) -> DeviceScope {
        let previous = ACTIVE.replace(Some(Arc::clone(self)));
        debug_assert!(
            previous.is_none(),
            "a device scope is already entered on this thread"
        );
        DeviceScope {
            linears: Arc::clone(self),
            thread: std::thread::current().id(),
            not_send: PhantomData,
        }
    }

    /// The counts so far.
    pub(crate) fn counts(&self) -> DeviceCounts {
        let resident_bytes = self
            .resident
            .values()
            .filter(|entry| entry.weights.is_resident())
            .map(|entry| entry.codes.len() + 4 * entry.scales.len())
            .sum::<usize>();
        DeviceCounts {
            resident_fp8: self.resident_fp8.load(Ordering::Relaxed),
            uploaded_fp8: self.uploaded_fp8.load(Ordering::Relaxed),
            fp4_experts: self.fp4_experts.load(Ordering::Relaxed),
            scalar_fallbacks: self.scalar_fallbacks.load(Ordering::Relaxed),
            resident_bytes: u64::try_from(resident_bytes).unwrap_or(u64::MAX),
        }
    }

    /// One G32 FP8 linear on the device, over resident weights when these are
    /// an eligible tensor.
    pub(crate) fn fp8_linear(
        &self,
        activations: (&[u8], &[u8]),
        weights: (&[u8], &[u8]),
        rows: usize,
        reduction: usize,
        outputs: usize,
    ) -> Result<Vec<f32>, Fp8MetalError> {
        let entry = self
            .resident
            .get(&weights.0.as_ptr().addr())
            .filter(|entry| {
                entry.codes.len() == weights.0.len()
                    && entry.scales.as_ptr() == weights.1.as_ptr()
                    && entry.scales.len() == weights.1.len()
            });
        let values = if let Some(entry) = entry {
            let values = entry
                .weights
                .forward(activations, weights, rows, reduction, outputs)?;
            self.resident_fp8.fetch_add(1, Ordering::Relaxed);
            values
        } else {
            let values = metal_fp8_linear(
                activations.0,
                activations.1,
                weights.0,
                weights.1,
                rows,
                reduction,
                outputs,
            )?;
            self.uploaded_fp8.fetch_add(1, Ordering::Relaxed);
            values
        };
        Ok(values)
    }

    pub(crate) fn count_fp4_expert(&self) {
        self.fp4_experts.fetch_add(1, Ordering::Relaxed);
    }

    /// Counts a routed expert the kernel cannot encode, run scalar instead.
    pub(crate) fn count_fallback(&self) {
        self.scalar_fallbacks.fetch_add(1, Ordering::Relaxed);
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Arc<DeviceLinears>>> = const { RefCell::new(None) };
}

/// The scope entered on this thread, if any.
pub(crate) fn active_device() -> Option<Arc<DeviceLinears>> {
    ACTIVE.with_borrow(Clone::clone)
}

/// Ends the scope when dropped. Not `Send`: the scope lives in the entering
/// thread's local state.
#[must_use = "the scope ends when the guard drops"]
pub(crate) struct DeviceScope {
    linears: Arc<DeviceLinears>,
    thread: std::thread::ThreadId,
    not_send: PhantomData<*const ()>,
}

impl Drop for DeviceScope {
    fn drop(&mut self) {
        let active = ACTIVE.take();
        debug_assert_eq!(
            self.thread,
            std::thread::current().id(),
            "a device scope ended on another thread"
        );
        debug_assert!(
            active.is_some_and(|active| Arc::ptr_eq(&active, &self.linears)),
            "the active device scope is not the one this guard entered"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{DeviceLinears, active_device};

    #[test]
    fn resident_entries_own_their_buffers() {
        let codes: Arc<[u8]> = vec![0x38; 64].into();
        let scales: Arc<[u8]> = vec![127; 2].into();
        let address = codes.as_ptr().addr();
        let linears = DeviceLinears::new([(Arc::clone(&codes), Arc::clone(&scales))]);
        // The entry holds its own reference, so the keyed bytes outlive any
        // caller's copy and the address cannot be reused while it exists.
        assert_eq!(
            (Arc::strong_count(&codes), Arc::strong_count(&scales)),
            (2, 2)
        );
        drop((codes, scales));
        let entry = &linears.resident[&address];
        assert_eq!(entry.codes.as_ptr().addr(), address);
        assert_eq!((entry.codes.len(), entry.scales.len()), (64, 2));
    }

    #[test]
    fn a_scope_is_active_only_on_its_thread_until_dropped() {
        let linears = Arc::new(DeviceLinears::default());
        assert!(active_device().is_none());
        {
            let _scope = linears.enter();
            assert!(active_device().is_some_and(|active| Arc::ptr_eq(&active, &linears)));
            std::thread::scope(|threads| {
                threads.spawn(|| assert!(active_device().is_none()));
            });
        }
        assert!(active_device().is_none());
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "already entered")]
    fn entering_twice_is_caught() {
        let linears = Arc::new(DeviceLinears::default());
        let _outer = linears.enter();
        let _inner = linears.enter();
    }
}
