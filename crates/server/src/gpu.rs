//! MLX memory figures and Metal frame capture for diagnostics.
//!
//! The pinned `mlx-rs` does not wrap MLX-C's memory and capture functions, so
//! this module calls them directly. It is the server's only module allowed
//! `unsafe`; each call passes either no arguments, a pointer to a local, or a
//! checked C string.
#![allow(unsafe_code)]

use std::{
    ffi::CString,
    path::{Path, PathBuf},
};

use mlx_sys as sys;

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
    fn capture_paths_are_checked_before_mlx_is_called() {
        // The environment check runs first, so these only reach the path
        // checks when the test process has MTL_CAPTURE_ENABLED=1.
        let error = check_capture_path(Path::new("trace.json")).unwrap_err();
        assert!(error.contains("MTL_CAPTURE_ENABLED") || error.contains(".gputrace"));
        let error = check_capture_path(Path::new("/no/such/dir/x.gputrace")).unwrap_err();
        assert!(error.contains("MTL_CAPTURE_ENABLED") || error.contains("does not exist"));
    }
}
