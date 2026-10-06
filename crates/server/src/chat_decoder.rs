//! Resident chat generation over any decoder that returns a full logit row.
//!
//! The turn rules (render, budget, token picks, stop tokens, streamed text)
//! are the ones the Qwen session uses, from [`super::turn`]; a
//! [`FullRowDecoder`] supplies only the logits. Every step reads the full
//! logit row: the GPU pick path and the prompt-prefix cache are built on
//! Qwen's executor types, so each turn here prefills its whole prompt in a
//! fresh sequence.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use chat_format::{ChatFormat, TurnDelta};
use tracing::field::Empty;

use super::{
    ChatBackend, ChatFinishReason, ChatGeneration, ChatGenerationError, ChatGenerationMetrics,
    ChatRequest, GenerationDeadline, ResidentChatLimits, SamplingDefaults, elapsed_ms, timed,
    turn::{TurnModel, TurnStart, TurnStep},
};

/// What a decoder learns from the checkpoint's configuration alone.
pub(crate) struct DecoderPlan {
    pub(crate) vocabulary_size: usize,
    /// Every per-sequence byte at the context limit: K/V plus any fixed
    /// recurrent state.
    pub(crate) planned_state_bytes: u64,
}

/// Why a decoder cannot plan or load a checkpoint.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum DecoderError {
    /// The configuration could not be read or is not one this decoder runs.
    UnsupportedConfig(String),
    /// The context is zero or past the checkpoint's positions.
    ContextOutOfRange { requested: usize, maximum: usize },
    /// The sequence state at the context limit exceeds the state budget.
    StateOverBudget {
        context: usize,
        planned_bytes: u64,
        budget_bytes: u64,
    },
    /// Reading or converting the weights failed.
    Load(String),
}

impl std::fmt::Display for DecoderError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedConfig(reason) => {
                write!(formatter, "unsupported checkpoint: {reason}")
            }
            Self::ContextOutOfRange { requested, maximum } => write!(
                formatter,
                "context must be 1 through {maximum} tokens; requested {requested}"
            ),
            Self::StateOverBudget {
                context,
                planned_bytes,
                budget_bytes,
            } => write!(
                formatter,
                "sequence state for {context} tokens needs {planned_bytes} bytes, more than the {budget_bytes}-byte K/V budget"
            ),
            Self::Load(reason) => write!(
                formatter,
                "checkpoint weights could not be loaded: {reason}"
            ),
        }
    }
}

/// A loaded checkpoint that decodes one sequence at a time and returns the
/// whole logit row after each call.
pub(crate) trait FullRowDecoder: Sized + 'static {
    /// One turn's sequence, borrowing the weights.
    type Sequence<'a>: FullRowSequence
    where
        Self: 'a;

    /// Reads the configuration only, refusing a context or state budget the
    /// checkpoint cannot meet before any weights load.
    fn plan(model: &Path, limits: ResidentChatLimits) -> Result<DecoderPlan, DecoderError>;

    fn load(model: &Path, plan: &DecoderPlan) -> Result<Self, DecoderError>;

    /// A fresh sequence; dropping it drops all of its state.
    fn sequence(&self, context_limit: usize) -> Result<Self::Sequence<'_>, String>;
}

/// One sequence of a [`FullRowDecoder`].
pub(crate) trait FullRowSequence {
    /// Consumes the prompt and returns the logits after its last token.
    fn prefill(&mut self, tokens: &[i32]) -> Result<Vec<f32>, String>;

    /// Appends one token and returns its logits.
    fn step(&mut self, token: i32) -> Result<Vec<f32>, String>;
}

/// A resident checkpoint for serial chat turns.
pub(crate) struct ChatDecoderSession<D> {
    decoder: D,
    format: ChatFormat,
    vocabulary_size: usize,
    context_limit: usize,
    planned_state_bytes: u64,
    load_ms: f64,
    /// Checkpoint directory, read again only to compile a JSON-schema grammar.
    model: PathBuf,
    sampling_defaults: SamplingDefaults,
}

impl<D: FullRowDecoder> ChatDecoderSession<D> {
    /// Plans the sequence state and loads the tokenizer and template before
    /// the weights, so a bad context or chat format fails without a weight load.
    pub(crate) fn load(model: &Path, limits: ResidentChatLimits) -> Result<Self, String> {
        let started = Instant::now();
        let plan = D::plan(model, limits).map_err(|error| error.to_string())?;
        let format = ChatFormat::load(model, plan.vocabulary_size)?;
        let sampling_defaults = SamplingDefaults::load(model)?;
        let decoder = D::load(model, &plan).map_err(|error| error.to_string())?;
        Ok(Self {
            decoder,
            format,
            vocabulary_size: plan.vocabulary_size,
            context_limit: limits.context_tokens(),
            planned_state_bytes: plan.planned_state_bytes,
            load_ms: elapsed_ms(started.elapsed()),
            model: model.to_path_buf(),
            sampling_defaults,
        })
    }

    fn turn_model(&self) -> TurnModel<'_> {
        TurnModel {
            format: &self.format,
            sampling_defaults: self.sampling_defaults,
            model: &self.model,
            vocabulary_size: self.vocabulary_size,
            context_limit: self.context_limit,
        }
    }

    fn generate_with_deadline(
        &mut self,
        request: ChatRequest<'_>,
        deadline: GenerationDeadline,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        let start = TurnStart::prepare(self.turn_model(), request, deadline)?;
        let (render_ms, max_tokens) = (start.render_ms, start.max_tokens);
        let (input_ids, mut turn, mut text) = start.into_parts();
        let mut sequence = self.decoder.sequence(self.context_limit)?;

        deadline.check()?;
        // Ends with the full-vocabulary logit readback, so it times the GPU work.
        let prefill = tracing::info_span!(
            "chat.prefill",
            prompt_tokens = input_ids.len(),
            cached_tokens = 0,
            prefill_ms = Empty
        );
        let (mut logits, prefill_ms) =
            timed(&prefill, "prefill_ms", || sequence.prefill(&input_ids))?;
        let mut decode_ms = Vec::new();
        let mut finish_reason = ChatFinishReason::Length;
        for step in 0..max_tokens {
            deadline.check()?;
            let accepted = turn.pick(&self.format, &mut logits)?;
            if accepted.visible {
                text.push(&self.format, accepted.token, on_token)?;
            }
            if let TurnStep::Stop(reason) = accepted.step {
                finish_reason = reason;
                break;
            }
            deadline.check()?;
            // One span per generated token, ending with its logit readback.
            let span = tracing::info_span!("chat.decode_step", step, decode_ms = Empty);
            let (next, step_ms) = timed(&span, "decode_ms", || sequence.step(accepted.token))?;
            logits = next;
            decode_ms.push(step_ms);
        }

        deadline.check()?;
        let end = text.finish(
            &self.format,
            turn.generated(),
            request.tools,
            finish_reason == ChatFinishReason::Eos,
            on_token,
        )?;
        let output = turn.finish()?;
        let decode_total_ms = decode_ms.iter().sum();
        let generated_tokens = output.generated.len();
        Ok(ChatGeneration {
            text: end.text,
            turn: end.turn,
            generated_token_ids: output.generated,
            finish_reason,
            logprobs: output.logprobs,
            sampling: Some(output.sampling),
            metrics: ChatGenerationMetrics {
                context_tokens: self.context_limit,
                planned_kv_bytes: self.planned_state_bytes,
                session_load_ms: self.load_ms,
                render_ms,
                prefill_ms,
                time_to_first_token_ms: end.time_to_first_token_ms,
                decode_ms,
                decode_total_ms,
                prompt_tokens: input_ids.len(),
                cached_prompt_tokens: 0,
                generated_tokens,
                // Prompt-lookup speculation runs on the Qwen executor only;
                // it never changes output, so a request asking for it is
                // decoded one token per step here.
                speculation: None,
            },
        })
    }
}

impl<D: FullRowDecoder> ChatBackend for ChatDecoderSession<D> {
    fn load_ms(&self) -> f64 {
        self.load_ms
    }

    fn generate_with_timeout(
        &mut self,
        request: ChatRequest<'_>,
        timeout: Duration,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        // Start before validation and rendering so the caller budgets the full turn.
        self.generate_with_deadline(request, GenerationDeadline::after(timeout), on_token)
    }
}

/// Helpers for opt-in checkpoint tests of decoder sessions.
#[cfg(test)]
pub(crate) mod test_support {
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::Duration,
    };

    use serde_json::Value;

    use crate::{chat_generation::ChatBackend, sse::test_support::exchange};

    /// Aborts the test process when MLX's active plus cached memory passes
    /// `cap_gib` (`METALLIX_GPU_CAP_GIB` overrides it), checked every 200 ms;
    /// stops checking when dropped. MLX Metal buffers can grow far past the
    /// process's resident size, so RSS alone does not show a leak.
    pub(crate) struct GpuCap(Arc<AtomicBool>);

    impl GpuCap {
        pub(crate) fn start(cap_gib: u64) -> Self {
            let cap_gib: u64 = std::env::var("METALLIX_GPU_CAP_GIB")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(cap_gib);
            let running = Arc::new(AtomicBool::new(true));
            let watching = Arc::clone(&running);
            thread::spawn(move || {
                while watching.load(Ordering::Acquire) {
                    if let Some(used) = crate::gpu::held_bytes() {
                        if used > cap_gib << 30 {
                            eprintln!(
                                "MLX memory {used} bytes passed the {cap_gib} GiB cap; aborting"
                            );
                            std::process::abort();
                        }
                    }
                    thread::sleep(Duration::from_millis(200));
                }
            });
            Self(running)
        }
    }

    impl Drop for GpuCap {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }

    /// One `/v1/chat/completions` exchange against `session`.
    pub(crate) fn chat_completion(body: &Value, session: &mut dyn ChatBackend) -> String {
        exchange(
            "/v1/chat/completions",
            &body.to_string(),
            |connection, body| {
                let (_, prepared) = crate::chat_completions::prepare(body).unwrap();
                crate::chat_completions::respond(
                    connection,
                    &prepared,
                    session,
                    "m",
                    Duration::from_secs(600),
                )
                .unwrap();
            },
        )
    }
}
