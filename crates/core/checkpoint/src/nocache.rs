//! `F_NOCACHE` on an open file. The one call that needs `unsafe`.
#![allow(unsafe_code)]

use std::{fs::File, io};

/// Asks macOS not to keep this file's read pages in the unified buffer cache.
#[cfg(target_os = "macos")]
pub(crate) fn set(file: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: `fcntl(F_NOCACHE, 1)` reads no memory and only sets a flag on
    // a descriptor `file` keeps open for the duration of the call.
    let result = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if result == -1 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// No-op where `F_NOCACHE` does not exist.
#[cfg(not(target_os = "macos"))]
pub(crate) fn set(_file: &File) -> io::Result<()> {
    Ok(())
}
