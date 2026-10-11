#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// The metallib this build installed and compiled in as MLX's last-resort
/// default (`MLX_METAL_PATH`); empty without the `metal` feature. MLX
/// prefers a path set with `mlx_metal_set_metallib_path`, then an
/// `mlx.metallib` next to the executable, before this one.
pub const DEFAULT_METALLIB_PATH: &str = env!("MLX_SYS_DEFAULT_METALLIB");
