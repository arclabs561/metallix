//! The model adapters `mx inspect supports` can name.
//!
//! The enum lives outside the `supports` module, which needs the `metal`
//! feature, so that tests covering every adapter also run without it.

#![cfg_attr(
    not(feature = "metal"),
    allow(
        dead_code,
        reason = "outside tests only `supports`, which needs the metal feature, names adapters"
    )
)]

/// A model adapter whose configuration gate recognized a document.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Adapter {
    /// Causal `qwen3` through the Qwen3 decoder.
    Qwen3,
    /// `llama` (`MiniCPM5`) through the Qwen3 decoder without Q/K norms.
    Llama,
    /// `qwen2` (Qwen2.5) through the Qwen3 decoder with Q/K/V biases.
    Qwen2,
    /// Bidirectional `bidirectional_pplx_qwen3` (pplx-embed).
    PplxQwen3,
    /// Julia-1 decisions on the CPU.
    Julia,
    /// `qwen3_5` hybrid linear/full attention.
    Qwen35,
    /// Experimental dense Qwen3-ASR transcription.
    Qwen3Asr,
    /// Gemma 4 dense text.
    Gemma4,
    /// `DeepSeek` V4.1.
    DeepseekV41,
}

impl Adapter {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::PplxQwen3 => "pplx_qwen3",
            Self::Julia => "julia",
            Self::Qwen35 => "qwen35",
            Self::Qwen3Asr => "qwen3_asr",
            Self::Gemma4 => "gemma4",
            Self::DeepseekV41 => "deepseek_v41",
        }
    }
}

#[cfg(test)]
mod tests;
