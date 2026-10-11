# Vendored mlx-sys

This is mlx-sys 0.6.0 from crates.io
(checksum `1848b6f1686114039e6e0adcf1578911e2e4d7707b9768ad61da074132319e1c`).
It was published from https://github.com/oxiglade/mlx-rs, path `mlx-sys`,
at commit `9c2fd72e172babb1e3719327a723717d98ccbe97`. Its bundled mlx-c
builds MLX core v0.32.2, pinned by `GIT_TAG` in `src/mlx-c/CMakeLists.txt`.
The workspace uses this copy through `[patch.crates-io]` in the root
`Cargo.toml`.

Removed from the published copy: `.cargo-ok` and `Cargo.lock`, which are
registry artifacts. Everything else is unchanged except the edits below.
`LICENSE-MIT` and `LICENSE-APACHE` are added from the same mlx-rs commit,
because the published crate declares `MIT OR Apache-2.0` but ships no
license text.
`patches/0001-deployment-target-and-nax-query.diff` is the exact diff
against the published crate.

## Changes and why

1. `build.rs` passes an explicit `CMAKE_OSX_DEPLOYMENT_TARGET`:
   `MACOSX_DEPLOYMENT_TARGET`, or 26.2 when unset. It also reruns when that
   variable changes.
   - Unset, MLX targets the build host's macOS. A binary built on a newer
     Mac then refuses to run on an older one.
   - MLX compiles its Neural Accelerator (NAX) kernels only for a target of
     26.2 or later, and drops them without an error below that.
   - Passing the value on every configure also overrides a stale empty
     value in an existing CMake cache.
2. `build.rs` keys the shared metallib directory by deployment target,
   giving `~/.mlx/lib/<mlx-c key>-macos<target>/`. It also refuses to
   install a metallib built below 26.2 there unless `MLX_RS_METAL_PATH`
   names another directory.
   - Every binary built from these sources on a machine loads the shared
     metallib. A low-target build would otherwise replace it with one
     lacking the NAX kernels, for all of them.
3. mlx-c gains `mlx_metal_is_nax_available(bool*)`, in `mlx/c/metal.h` and
   `mlx/c/metal.cpp`. A CMake definition limits it to Metal builds.
   - It wraps MLX's internal `metal::is_nax_available()`, so callers can
     tell whether MLX will run NAX kernels. The answer depends on the OS
     (26.2+), the GPU generation (17+, 18+ for phone GPUs) and the build.
   - `mlx/c/metal.cpp` is generated upstream from MLX's public headers. The
     upstream route is to declare `is_nax_available` in MLX's
     `mlx/backend/metal/metal.h` and regenerate mlx-c.

4. `build.rs` and `src/lib.rs` export the metallib path compiled into MLX as
   `mlx_sys::DEFAULT_METALLIB_PATH`. MLX's own `get_metallib_path` returns
   only an override set through the API, so without this a caller cannot
   tell which metallib MLX loaded. `mx doctor` reads it to count the NAX
   kernels in the file actually in use.

## Updating

To move to a newer mlx-sys, copy the published crate here again, then apply
the diff in `patches/` and check that each change is still needed.
