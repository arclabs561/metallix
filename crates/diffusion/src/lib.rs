//! Model-free pieces of latent diffusion image generation.
//!
//! A diffusion pipeline is not a token decoder: it encodes the prompt once,
//! runs a fixed number of denoiser passes over a latent tensor, each followed
//! by a scheduler update, and decodes the final latent to pixels. This crate
//! holds the parts that need no tensor runtime, so they are testable without
//! Metal: the flow-matching noise schedule.

mod schedule;
pub use schedule::{
    FlowMatchConfig, FlowMatchStep, ScheduleError, Sigmas, TimeShift, flux2_empirical_mu, linear_mu,
};
