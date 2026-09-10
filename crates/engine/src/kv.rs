use std::num::NonZeroU32;

use thiserror::Error;

/// A validated fixed width for one logical KV-cache page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPageTokens(NonZeroU32);

impl KvPageTokens {
    /// Creates a page width.
    ///
    /// # Errors
    ///
    /// Returns [`KvPlanError::ZeroPageWidth`] when `tokens` is zero.
    pub fn new(tokens: u32) -> Result<Self, KvPlanError> {
        NonZeroU32::new(tokens)
            .map(Self)
            .ok_or(KvPlanError::ZeroPageWidth)
    }

    /// Returns the number of tokens held by a full page.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

/// The model-independent logical capacity of a paged KV cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvPagePlan {
    page_tokens: KvPageTokens,
    page_count: NonZeroU32,
}

impl KvPagePlan {
    /// Creates a finite logical KV page pool.
    ///
    /// # Errors
    ///
    /// Returns [`KvPlanError::ZeroPageCount`] when `page_count` is zero.
    pub fn new(page_tokens: KvPageTokens, page_count: u32) -> Result<Self, KvPlanError> {
        let page_count = NonZeroU32::new(page_count).ok_or(KvPlanError::ZeroPageCount)?;
        Ok(Self {
            page_tokens,
            page_count,
        })
    }

    /// Returns the logical page demand for `tokens`, rounding up a partial page.
    #[must_use]
    pub const fn pages_for(self, tokens: u32) -> u32 {
        tokens.div_ceil(self.page_tokens.get())
    }

    /// Returns whether one sequence of `tokens` fits in the complete pool.
    #[must_use]
    pub const fn can_admit(self, tokens: u32) -> bool {
        self.pages_for(tokens) <= self.page_count.get()
    }

    /// Returns the maximum logical token capacity.
    #[must_use]
    pub const fn token_capacity(self) -> u32 {
        self.page_tokens.get().saturating_mul(self.page_count.get())
    }
}

/// Invalid logical KV page-pool configuration.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum KvPlanError {
    /// KV pages must contain at least one token.
    #[error("KV page width must be greater than zero")]
    ZeroPageWidth,
    /// The pool must contain at least one page.
    #[error("KV page count must be greater than zero")]
    ZeroPageCount,
}

#[cfg(test)]
mod tests {
    use super::{KvPagePlan, KvPageTokens, KvPlanError};

    #[test]
    fn rejects_zero_dimensions() {
        assert_eq!(KvPageTokens::new(0), Err(KvPlanError::ZeroPageWidth));
        let width = KvPageTokens::new(16).expect("positive page width");
        assert_eq!(KvPagePlan::new(width, 0), Err(KvPlanError::ZeroPageCount));
    }

    #[test]
    fn rounds_partial_sequences_to_a_full_page() {
        let width = KvPageTokens::new(16).expect("positive page width");
        let plan = KvPagePlan::new(width, 4).expect("positive page count");
        assert_eq!(plan.pages_for(0), 0);
        assert_eq!(plan.pages_for(1), 1);
        assert_eq!(plan.pages_for(16), 1);
        assert_eq!(plan.pages_for(17), 2);
        assert!(plan.can_admit(64));
        assert!(!plan.can_admit(65));
    }
}
