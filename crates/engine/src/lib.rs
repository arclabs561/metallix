//! Shared serving contracts independent of a model architecture or GPU backend.

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
    /// Returns an error when either limit is zero.
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
