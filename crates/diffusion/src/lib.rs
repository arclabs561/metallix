//! Model-free pieces of latent diffusion image generation.
//!
//! A diffusion pipeline is not a token decoder: it encodes the prompt once,
//! runs a fixed number of denoiser passes over a latent tensor, each followed
//! by a scheduler update, and decodes the final latent to pixels. This crate
//! holds the parts that need no tensor runtime, so they are testable without
//! Metal: the flow-matching noise schedule.
//!
//! # Overview
//!
//! [`FlowMatchConfig::from_json`] validates the scheduler configuration.
//! [`flux2_empirical_mu`] and [`linear_mu`] compute pipeline-specific dynamic
//! shifts; [`Sigmas::new`] constructs the schedule and [`Sigmas::steps`] yields
//! each [`FlowMatchStep`]. [`FlowMatchStep::dt`] supplies the Euler step size.
//! These are scalar schedule calculations, not a denoiser or image decoder.
//!
//! # Example
//!
//! ```
//! use diffusion::{FlowMatchConfig, Sigmas};
//!
//! let config = FlowMatchConfig::from_json(
//!     r#"{"_class_name":"FlowMatchEulerDiscreteScheduler","num_train_timesteps":1000}"#,
//! )?;
//! let schedule = Sigmas::new(&config, 2, None)?;
//! assert_eq!(schedule.as_slice(), &[1.0, 0.5, 0.0]);
//! assert_eq!(schedule.steps().next().unwrap().dt(), -0.5);
//! # Ok::<(), diffusion::ScheduleError>(())
//! ```

#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc)]

pub mod metrics;

mod schedule;
pub use schedule::{
    FlowMatchConfig, FlowMatchStep, ScheduleError, Sigmas, TimeShift, flux2_empirical_mu, linear_mu,
};
