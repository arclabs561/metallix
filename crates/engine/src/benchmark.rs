use std::{num::NonZeroU16, time::Duration};

use thiserror::Error;

use crate::OutputTokenLimit;

/// Whether a benchmark begins without or with an eligible shared prefix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheCondition {
    /// The prompt has no reusable prefix state.
    Cold,
    /// The prompt shares an eligible prefix with a prior request.
    SharedPrefix,
}

/// A validated benchmark workload independent of model architecture.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkProfile {
    context_tokens: u32,
    output_tokens: OutputTokenLimit,
    concurrency: NonZeroU16,
    cache_condition: CacheCondition,
}

impl BenchmarkProfile {
    /// Creates a finite benchmark workload.
    ///
    /// # Errors
    ///
    /// Returns [`BenchmarkError::ZeroContext`] or
    /// [`BenchmarkError::ZeroConcurrency`] when a required dimension is zero.
    pub fn new(
        context_tokens: u32,
        output_tokens: OutputTokenLimit,
        concurrency: u16,
        cache_condition: CacheCondition,
    ) -> Result<Self, BenchmarkError> {
        if context_tokens == 0 {
            return Err(BenchmarkError::ZeroContext);
        }
        let concurrency = NonZeroU16::new(concurrency).ok_or(BenchmarkError::ZeroConcurrency)?;
        Ok(Self {
            context_tokens,
            output_tokens,
            concurrency,
            cache_condition,
        })
    }

    /// Returns the prompt context length.
    #[must_use]
    pub const fn context_tokens(self) -> u32 {
        self.context_tokens
    }

    /// Returns the planned output-token count.
    #[must_use]
    pub const fn output_tokens(self) -> OutputTokenLimit {
        self.output_tokens
    }

    /// Returns the number of concurrent requests.
    #[must_use]
    pub const fn concurrency(self) -> u16 {
        self.concurrency.get()
    }

    /// Returns the cache condition this profile requires.
    #[must_use]
    pub const fn cache_condition(self) -> CacheCondition {
        self.cache_condition
    }
}

/// One measured request outcome for a benchmark workload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BenchmarkSample {
    time_to_first_token: Duration,
    inter_token_latency: Option<Duration>,
    completion_tokens: u32,
    peak_resident_bytes: u64,
    ssd_read_bytes: u64,
}

impl BenchmarkSample {
    /// Records a request outcome without logging prompt contents.
    #[must_use]
    pub const fn new(
        time_to_first_token: Duration,
        inter_token_latency: Option<Duration>,
        completion_tokens: u32,
        peak_resident_bytes: u64,
        ssd_read_bytes: u64,
    ) -> Self {
        Self {
            time_to_first_token,
            inter_token_latency,
            completion_tokens,
            peak_resident_bytes,
            ssd_read_bytes,
        }
    }

    /// Returns time to the first streamed token.
    #[must_use]
    pub const fn time_to_first_token(self) -> Duration {
        self.time_to_first_token
    }

    /// Returns the mean decode spacing when at least two tokens were emitted.
    #[must_use]
    pub const fn inter_token_latency(self) -> Option<Duration> {
        self.inter_token_latency
    }

    /// Returns emitted completion tokens.
    #[must_use]
    pub const fn completion_tokens(self) -> u32 {
        self.completion_tokens
    }

    /// Returns peak resident unified-memory bytes.
    #[must_use]
    pub const fn peak_resident_bytes(self) -> u64 {
        self.peak_resident_bytes
    }

    /// Returns SSD bytes read during the request.
    #[must_use]
    pub const fn ssd_read_bytes(self) -> u64 {
        self.ssd_read_bytes
    }
}

/// Invalid benchmark workload dimensions.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum BenchmarkError {
    /// Prompt context must contain at least one token.
    #[error("benchmark context tokens must be greater than zero")]
    ZeroContext,
    /// At least one concurrent request is required.
    #[error("benchmark concurrency must be greater than zero")]
    ZeroConcurrency,
}

#[cfg(test)]
mod tests {
    use super::{BenchmarkError, BenchmarkProfile, BenchmarkSample, CacheCondition};
    use crate::OutputTokenLimit;
    use std::time::Duration;

    #[test]
    fn workload_preserves_cache_and_concurrency_dimensions() {
        let profile = BenchmarkProfile::new(
            8_192,
            OutputTokenLimit::new(128).expect("positive output limit"),
            4,
            CacheCondition::SharedPrefix,
        )
        .expect("valid profile");
        assert_eq!(profile.context_tokens(), 8_192);
        assert_eq!(profile.concurrency(), 4);
        assert_eq!(profile.cache_condition(), CacheCondition::SharedPrefix);
    }

    #[test]
    fn workload_rejects_zero_dimensions() {
        let output = OutputTokenLimit::new(1).expect("positive output limit");
        assert_eq!(
            BenchmarkProfile::new(0, output, 1, CacheCondition::Cold),
            Err(BenchmarkError::ZeroContext)
        );
        assert_eq!(
            BenchmarkProfile::new(1, output, 0, CacheCondition::Cold),
            Err(BenchmarkError::ZeroConcurrency)
        );
    }

    #[test]
    fn sample_keeps_ssd_traffic_separate_from_memory() {
        let sample = BenchmarkSample::new(
            Duration::from_millis(12),
            Some(Duration::from_millis(3)),
            4,
            1_024,
            2_048,
        );
        assert_eq!(sample.peak_resident_bytes(), 1_024);
        assert_eq!(sample.ssd_read_bytes(), 2_048);
    }
}
