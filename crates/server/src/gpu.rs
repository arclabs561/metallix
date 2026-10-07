//! MLX memory figures, Metal frame capture, and keeping a served model's GPU
//! memory resident between requests.
//!
//! The pinned `mlx-rs` does not wrap MLX-C's memory, capture and device-info
//! functions, so this module calls them directly. It is the server's only
//! module allowed `unsafe`; each call passes either no arguments, a pointer to
//! a local, a checked C string, or a handle created and freed in the same
//! block.
#![allow(unsafe_code)]

use std::{
    ffi::CString,
    path::{Path, PathBuf},
    sync::{
        OnceLock,
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, RecvTimeoutError},
    },
    time::{Duration, Instant},
};

use mlx_sys as sys;

/// Seconds after its last request that a model process keeps its GPU memory
/// wired and resident; `0` turns off both the wiring and the keepalive.
const KEEPALIVE_ENV: &str = "METALLIX_GPU_KEEPALIVE_S";
/// Long enough to cover the pauses of an agent between turns, short enough
/// that an idle server stops issuing GPU work (llama.cpp's default too).
const DEFAULT_KEEPALIVE: Duration = Duration::from_secs(180);
/// macOS stops treating a process's GPU memory as resident about 2 s after
/// its last GPU command, and the next command that reads it pays roughly
/// 20-30 ms per GiB to make it resident again. A command every 0.5 s keeps
/// wired memory resident; unwired memory is released either way.
const KEEPALIVE_INTERVAL: Duration = Duration::from_millis(500);

/// Set once [`wire_resident`] wires memory; the keepalive only helps then.
static WIRED: AtomicBool = AtomicBool::new(false);

/// MLX's active, cached and peak allocation in bytes, or `None` when MLX
/// reports an error.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Memory {
    pub(crate) active: u64,
    pub(crate) cache: u64,
    pub(crate) peak: u64,
}

impl Memory {
    pub(crate) fn read() -> Option<Self> {
        let mut active = 0_usize;
        let mut cache = 0_usize;
        let mut peak = 0_usize;
        // SAFETY: each function only writes one `size_t` through the pointer.
        let status = unsafe {
            sys::mlx_get_active_memory(&raw mut active)
                | sys::mlx_get_cache_memory(&raw mut cache)
                | sys::mlx_get_peak_memory(&raw mut peak)
        };
        (status == 0).then_some(Self {
            active: active as u64,
            cache: cache as u64,
            peak: peak as u64,
        })
    }

    /// Records the figures as `mlx.active_bytes`, `mlx.cache_bytes` and
    /// `mlx.peak_bytes` on `span`.
    pub(crate) fn record_on(span: &tracing::Span) {
        if let Some(memory) = Self::read() {
            span.record("mlx.active_bytes", memory.active);
            span.record("mlx.cache_bytes", memory.cache);
            span.record("mlx.peak_bytes", memory.peak);
        }
    }
}

/// Default for [`cap_cache`]: enough to reuse a step's buffers, far below
/// MLX's own default, which is its whole memory limit.
pub(crate) const DEFAULT_CACHE_LIMIT_MIB: usize = 2048;

/// Environment variable overriding [`DEFAULT_CACHE_LIMIT_MIB`], in MiB.
pub(crate) const CACHE_LIMIT_ENV: &str = "METALLIX_MLX_CACHE_MIB";

/// The allocator cache limit in bytes: [`CACHE_LIMIT_ENV`] when it is a whole
/// number of MiB, else the default.
pub(crate) fn cache_limit_bytes() -> usize {
    let mib = match std::env::var(CACHE_LIMIT_ENV) {
        Ok(value) => value.trim().parse().unwrap_or_else(|_| {
            tracing::warn!("{CACHE_LIMIT_ENV}={value:?} is not a whole number of MiB; using {DEFAULT_CACHE_LIMIT_MIB}");
            DEFAULT_CACHE_LIMIT_MIB
        }),
        Err(_) => DEFAULT_CACHE_LIMIT_MIB,
    };
    mib.saturating_mul(1 << 20)
}

/// Caps MLX's process-wide allocator cache at [`cache_limit_bytes`]. MLX keeps
/// freed buffers for reuse up to this limit, and reuses one only for a request
/// of nearly the same size, so a process whose array shapes keep changing
/// (prompt lengths, batch sizes) otherwise holds every past shape's buffers.
pub(crate) fn cap_cache() -> Result<usize, String> {
    let limit = cache_limit_bytes();
    let mut previous = 0_usize;
    // SAFETY: writes one `size_t` through the pointer.
    let status = unsafe { sys::mlx_set_cache_limit(&raw mut previous, limit) };
    if status == 0 {
        Ok(limit)
    } else {
        Err(String::from("MLX refused the allocator cache limit"))
    }
}

/// MLX's active plus cached bytes: cached buffers are freed by MLX but still
/// held from Metal, so both count toward the process's GPU footprint.
#[cfg(test)]
pub(crate) fn held_bytes() -> Option<u64> {
    let mut active = 0_usize;
    let mut cache = 0_usize;
    // SAFETY: both functions only write one `size_t` through the pointer.
    let status = unsafe {
        sys::mlx_get_active_memory(&raw mut active) | sys::mlx_get_cache_memory(&raw mut cache)
    };
    (status == 0).then_some(active as u64 + cache as u64)
}

/// Starts a new peak-memory window, so the next [`Memory::read`] reports the
/// peak since this call. The peak is process-wide; `mx serve` runs one model
/// per child process and one request per model, so a window is one request.
pub(crate) fn reset_peak_memory() {
    // SAFETY: takes no arguments.
    let _ = unsafe { sys::mlx_reset_peak_memory() };
}

/// The keepalive window from `METALLIX_GPU_KEEPALIVE_S`, read once.
fn keepalive_window() -> Duration {
    static WINDOW: OnceLock<Duration> = OnceLock::new();
    *WINDOW.get_or_init(|| match std::env::var(KEEPALIVE_ENV) {
        Err(_) => DEFAULT_KEEPALIVE,
        Ok(text) => text.parse().map_or_else(
            |_| {
                tracing::warn!(
                    value = %text,
                    "{KEEPALIVE_ENV} is not a whole number of seconds; using {}",
                    DEFAULT_KEEPALIVE.as_secs()
                );
                DEFAULT_KEEPALIVE
            },
            Duration::from_secs,
        ),
    })
}

/// Metal's recommended working-set size for the default GPU, which MLX
/// refuses to wire beyond.
fn recommended_working_set() -> Option<u64> {
    let mut size = 0_usize;
    // SAFETY: the device and info handles are created, read and freed inside
    // this block; `size` is a local and the key is a NUL-terminated literal.
    // The caller has already run an `mlx-rs` operation, so an MLX-C failure
    // returns a status instead of exiting.
    let status = unsafe {
        let device = sys::mlx_device_new_type(sys::mlx_device_type__MLX_GPU, 0);
        let mut info = sys::mlx_device_info_new();
        let mut status = sys::mlx_device_info_get(&raw mut info, device);
        if status == 0 {
            status = sys::mlx_device_info_get_size(
                &raw mut size,
                info,
                c"max_recommended_working_set_size".as_ptr(),
            );
        }
        sys::mlx_device_info_free(info);
        sys::mlx_device_free(device);
        status
    };
    (status == 0 && size > 0).then_some(size as u64)
}

/// Wires the loaded model's memory so it can stay resident between requests:
/// MLX's active bytes now (the weights) plus `planned_bytes` (K/V and prefix
/// cache budgets) and the allocator cache cap ([`cache_limit_bytes`], whose
/// buffers the next request reuses), capped at Metal's recommended working
/// set. Returns the wired limit, or `None` when the keepalive is off or MLX
/// refuses. Call once, after load and before any request is queued, since
/// MLX must not change the limit while an asynchronous evaluation runs.
#[tracing::instrument(
    name = "model.wire",
    level = "info",
    skip_all,
    fields(planned_bytes = planned_bytes, limit = tracing::field::Empty)
)]
pub(crate) fn wire_resident(planned_bytes: u64) -> Option<u64> {
    if keepalive_window().is_zero() {
        return None;
    }
    let active = mlx_rs::memory::active_memory().ok()? as u64;
    let working_set = tracing::info_span!("wire.working_set").in_scope(recommended_working_set)?;
    let limit = active
        .saturating_add(planned_bytes)
        .saturating_add(cache_limit_bytes() as u64)
        .min(working_set);
    tracing::Span::current().record("limit", limit);
    let limit_bytes = usize::try_from(limit).ok()?;
    let applied = tracing::info_span!("wire.set_limit")
        .in_scope(|| mlx_rs::memory::set_wired_limit(limit_bytes));
    match applied {
        Ok(_previous) => {
            WIRED.store(true, Ordering::Release);
            Some(limit)
        }
        Err(error) => {
            tracing::warn!(%error, limit, "MLX refused the wired-memory limit");
            None
        }
    }
}

/// Iterates over `jobs` like the receiver itself, but while memory is wired
/// and the keepalive window after the previous job is open, it issues a
/// trivial GPU command every [`KEEPALIVE_INTERVAL`] so the next request does
/// not pay to make the model resident again; after the window it blocks
/// without GPU work. Ends once every sender is gone. Create it after
/// [`wire_resident`].
pub(crate) fn keep_resident<T>(jobs: Receiver<T>) -> ResidentJobs<T> {
    let window = if WIRED.load(Ordering::Acquire) {
        keepalive_window()
    } else {
        Duration::ZERO
    };
    ResidentJobs { jobs, window }
}

/// See [`keep_resident`].
pub(crate) struct ResidentJobs<T> {
    jobs: Receiver<T>,
    window: Duration,
}

impl<T> Iterator for ResidentJobs<T> {
    type Item = T;

    /// The caller asks for the next job when the previous one has finished,
    /// so the window starts now.
    fn next(&mut self) -> Option<T> {
        next_job_within(&self.jobs, Instant::now(), self.window)
    }
}

fn next_job_within<T>(jobs: &Receiver<T>, last_job: Instant, window: Duration) -> Option<T> {
    while last_job.elapsed() < window {
        match jobs.recv_timeout(KEEPALIVE_INTERVAL) {
            Ok(job) => return Some(job),
            Err(RecvTimeoutError::Timeout) => touch_gpu(),
            Err(RecvTimeoutError::Disconnected) => return None,
        }
    }
    jobs.recv().ok()
}

/// Evaluates a one-element sum on the GPU stream of the calling thread.
fn touch_gpu() {
    let one = mlx_rs::Array::from_int(1);
    if let Err(error) = mlx_rs::ops::add(&one, &one).and_then(|sum| sum.eval()) {
        tracing::warn!(%error, "GPU keepalive command failed");
    }
}

/// Checks what Metal and MLX need before a capture can start, so a bad
/// `--gpu-capture` is rejected before any model loads.
pub(crate) fn check_capture_path(path: &Path) -> Result<(), String> {
    if std::env::var_os("MTL_CAPTURE_ENABLED").is_none_or(|value| value != "1") {
        return Err(String::from(
            "--gpu-capture requires MTL_CAPTURE_ENABLED=1 in the environment; Metal refuses to capture without it",
        ));
    }
    if path
        .extension()
        .is_none_or(|extension| extension != "gputrace")
    {
        return Err(format!(
            "--gpu-capture path {} must end in .gputrace",
            path.display()
        ));
    }
    if path.exists() {
        return Err(format!(
            "--gpu-capture path {} already exists; Metal does not overwrite captures",
            path.display()
        ));
    }
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        if !parent.is_dir() {
            return Err(format!(
                "--gpu-capture directory {} does not exist",
                parent.display()
            ));
        }
    }
    Ok(())
}

/// A running Metal capture, stopped when dropped.
pub(crate) struct Capture {
    path: PathBuf,
}

impl Capture {
    pub(crate) fn start(path: &Path) -> Result<Self, String> {
        check_capture_path(path)?;
        let text = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| format!("--gpu-capture path {} contains NUL", path.display()))?;
        // An MLX-C failure calls the process-wide error handler, whose MLX-C
        // default exits the process. Any `mlx-rs` operation installs the
        // `mlx-rs` handler for this thread first, so a failure returns here.
        let zero = mlx_rs::Array::from_int(0);
        let _ = mlx_rs::ops::add(&zero, &zero);
        // SAFETY: `text` is a NUL-terminated string that outlives the call.
        let status = unsafe { sys::mlx_metal_start_capture(text.as_ptr()) };
        if status != 0 {
            return Err(format!(
                "MLX could not start a Metal capture to {} (status {status})",
                path.display()
            ));
        }
        tracing::info!(path = %path.display(), "started Metal capture");
        Ok(Self {
            path: path.to_owned(),
        })
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // SAFETY: takes no arguments; a capture is running.
        let status = unsafe { sys::mlx_metal_stop_capture() };
        if status == 0 {
            tracing::info!(path = %self.path.display(), "wrote Metal capture");
        } else {
            tracing::error!(path = %self.path.display(), status, "MLX could not stop the Metal capture");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_figures_are_readable() {
        let memory = Memory::read().expect("MLX memory figures");
        assert!(memory.peak >= memory.active);
    }

    #[test]
    fn keepalive_wait_ends_promptly_when_the_senders_are_gone() {
        // A sender that hangs up mid-window, after at least one keepalive
        // tick, must end the wait at the next tick rather than at the end of
        // a 60 s window. The first GPU command of a process can take seconds
        // on a loaded machine (Metal setup, kernel builds), so it runs before
        // the clock starts, and the bound stays far below the window.
        touch_gpu();
        let (sender, jobs) = std::sync::mpsc::channel::<u8>();
        let hang_up = std::thread::spawn(move || {
            std::thread::sleep(KEEPALIVE_INTERVAL + Duration::from_millis(100));
            drop(sender);
        });
        let started = Instant::now();
        assert_eq!(
            next_job_within(&jobs, Instant::now(), Duration::from_secs(60)),
            None
        );
        assert!(started.elapsed() < Duration::from_secs(10));
        hang_up.join().expect("sender thread");
    }

    #[test]
    fn keepalive_wait_hands_over_jobs_inside_and_after_the_window() {
        let (sender, jobs) = std::sync::mpsc::channel();
        sender.send(1).expect("queued");
        assert_eq!(
            next_job_within(&jobs, Instant::now(), Duration::from_secs(60)),
            Some(1)
        );
        // Past the window the wait is a plain blocking receive.
        sender.send(2).expect("queued");
        assert_eq!(
            next_job_within(&jobs, Instant::now(), Duration::ZERO),
            Some(2)
        );
        drop(sender);
        assert_eq!(next_job_within(&jobs, Instant::now(), Duration::ZERO), None);
    }

    #[test]
    fn capture_paths_are_checked_before_mlx_is_called() {
        // The environment check runs first, so these only reach the path
        // checks when the test process has MTL_CAPTURE_ENABLED=1.
        let error = check_capture_path(Path::new("trace.json")).unwrap_err();
        assert!(error.contains("MTL_CAPTURE_ENABLED") || error.contains(".gputrace"));
        let error = check_capture_path(Path::new("/no/such/dir/x.gputrace")).unwrap_err();
        assert!(error.contains("MTL_CAPTURE_ENABLED") || error.contains("does not exist"));
    }
}
