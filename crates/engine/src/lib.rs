//! Shared serving contracts independent of a model architecture or GPU backend.
//!
//! Everything here is host-side bookkeeping and arithmetic over values a model
//! has already produced: token counts, KV-cache pages, logits and weights. No
//! module loads a model, runs a graph or touches MLX, so the crate builds and
//! tests on any platform, and each model adapter or server path composes the
//! parts it needs.
//!
//! # Overview
//!
//! * Limits: [`OutputTokenLimit`] and [`RequestLimits`] are validated,
//!   nonzero request bounds.
//! * KV memory: [`kv`] plans a fixed logical page pool, [`admission`] reserves
//!   pages for admitted requests, and [`blocks`] does the full paged-pool
//!   accounting with block tables, reference counts and prefix caching.
//! * Token selection: [`sampling`] draws one token from a masked,
//!   temperature-scaled distribution using caller-supplied randomness, and
//!   [`speculative`] verifies a drafter's proposed tokens against the target
//!   model. With the `structured-output` feature, `constraint` restricts
//!   selection to output matching a JSON Schema.
//! * Search: [`smc`] keeps weighted particles for sequential Monte Carlo.
//! * Operations: [`lifecycle`] checks a model's load, warm and ready phases,
//!   and [`benchmark`] describes a workload and its measured samples.
//!
//! Randomness always arrives as a caller-supplied variate, never from an RNG
//! inside the crate, so a seeded caller can replay any decision.
//!
//! # Example
//!
//! ```
//! use engine::{RequestLimits, sampling::sample_categorical};
//!
//! let limits = RequestLimits::new(32_768, 1_024)?;
//! assert_eq!(limits.max_output_tokens().get(), 1_024);
//!
//! // Only token 2 is legal, so every uniform variate selects it.
//! let sample = sample_categorical(&[1.0, 5.0, 2.0], &[false, false, true], 1.0, 0.5)
//!     .expect("one legal token");
//! assert_eq!(sample.token_id, 2);
//! # Ok::<(), engine::LimitError>(())
//! ```

#![deny(missing_docs)]
// The workspace allows this lint; crates opt in once their docs are complete.
#![warn(clippy::missing_errors_doc)]

pub mod admission;
pub mod benchmark;
pub mod blocks;
#[cfg(feature = "structured-output")]
pub mod constraint;
pub mod kv;
pub mod lifecycle;
pub mod sampling;
pub mod smc;
pub mod speculative;

use std::num::NonZeroU32;

use thiserror::Error;

/// A validated upper bound for tokens produced by one request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OutputTokenLimit(NonZeroU32);

impl OutputTokenLimit {
    /// Creates a request output limit.
    ///
    /// # Errors
    ///
    /// Returns [`LimitError::Zero`] when `tokens` is zero.
    pub fn new(tokens: u32) -> Result<Self, LimitError> {
        NonZeroU32::new(tokens).map(Self).ok_or(LimitError::Zero)
    }

    /// Returns the validated token count.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// Limits that make admission decisions finite and observable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestLimits {
    max_context_tokens: NonZeroU32,
    max_output_tokens: OutputTokenLimit,
}

impl RequestLimits {
    /// Creates request limits.
    ///
    /// # Errors
    ///
    /// Returns [`LimitError::Zero`] when either limit is zero.
    pub fn new(max_context_tokens: u32, max_output_tokens: u32) -> Result<Self, LimitError> {
        let max_context_tokens = NonZeroU32::new(max_context_tokens).ok_or(LimitError::Zero)?;
        let max_output_tokens = OutputTokenLimit::new(max_output_tokens)?;
        Ok(Self {
            max_context_tokens,
            max_output_tokens,
        })
    }

    /// Returns the maximum admitted context size.
    #[must_use]
    pub const fn max_context_tokens(self) -> u32 {
        self.max_context_tokens.get()
    }

    /// Returns the maximum output size.
    #[must_use]
    pub const fn max_output_tokens(self) -> OutputTokenLimit {
        self.max_output_tokens
    }
}

/// Invalid serving limits.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum LimitError {
    /// A limit that must be positive was zero.
    #[error("limits must be greater than zero")]
    Zero,
}

#[cfg(test)]
mod tests {
    use super::{LimitError, RequestLimits};

    #[test]
    fn limits_reject_zero_values() {
        assert_eq!(RequestLimits::new(0, 1), Err(LimitError::Zero));
        assert_eq!(RequestLimits::new(1, 0), Err(LimitError::Zero));
    }

    #[test]
    fn limits_preserve_validated_values() {
        let limits = RequestLimits::new(32_768, 1_024).expect("valid limits");
        assert_eq!(limits.max_context_tokens(), 32_768);
        assert_eq!(limits.max_output_tokens().get(), 1_024);
    }
}
