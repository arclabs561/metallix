//! Model-free building blocks shared by the model crates.
//!
//! A block here owns its numerics and its validation, and nothing about
//! checkpoint tensor names: each model crate finds its tensors and hands
//! them in. Graph code needs the `metal` feature; configuration types build
//! and test without MLX.

#![deny(missing_docs)]

pub mod moe;

// MLX's native test operations share process-global device initialization.
#[cfg(all(test, feature = "metal"))]
pub(crate) static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
