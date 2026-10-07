//! Resident chat generation over any decoder that returns a full logit row.
//!
//! The turn rules (render, budget, token picks, stop tokens, streamed text)
//! are the ones the Qwen session uses, from [`super::turn`]; a
//! [`FullRowDecoder`] supplies only the logits. Every step reads the full
//! logit row: the GPU pick path is built on Qwen's executor types.
//!
//! A decoder that can save its sequence state ([`FullRowDecoder::SAVES_STATE`])
//! gets a prompt-prefix cache: before prefilling, the session finds the
//! prompt's reusable boundaries (the shared rule in
//! [`super::prefix_cache::boundaries`]), prefills up to each one, saves the
//! state there and keys it by the exact token prefix and the request's cache
//! salt. A later prompt that starts with a saved boundary resumes from it.
//! State is saved only at those boundaries, never cut back afterwards, so a
//! decoder whose state cannot be truncated (recurrent layers) can take part.
//! Other decoders prefill every prompt whole in a fresh sequence.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use chat_format::{ChatFormat, TurnDelta};
use tracing::field::Empty;

use sha2::{Digest, Sha256};

use super::{
    ChatBackend, ChatFinishReason, ChatGeneration, ChatGenerationError, ChatGenerationMetrics,
    ChatRequest, GenerationDeadline, ResidentChatLimits, SamplingDefaults, elapsed_ms,
    prefix_cache::{self, PrefixCache},
    timed,
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
    /// A state-budget refusal with an exact capacity under the adapter's cost model.
    StateBudgetCapacity {
        context: usize,
        planned_bytes: u64,
        budget_bytes: u64,
        largest_fitting: Option<std::num::NonZeroUsize>,
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
            Self::StateBudgetCapacity {
                context,
                planned_bytes,
                budget_bytes,
                largest_fitting,
            } => {
                write!(
                    formatter,
                    "sequence state for {context} tokens needs {planned_bytes} bytes, more than the {budget_bytes}-byte K/V budget; "
                )?;
                match largest_fitting {
                    Some(tokens) => write!(
                        formatter,
                        "largest fitting logical context is {tokens} tokens"
                    ),
                    None => formatter.write_str("no positive logical context fits"),
                }
            }
            Self::Load(reason) => write!(
                formatter,
                "checkpoint weights could not be loaded: {reason}"
            ),
        }
    }
}

/// The snapshot type of a decoder that cannot save sequence state. It has no
/// values, so code that would restore one cannot run.
pub(crate) enum NoSnapshot {}

/// A loaded checkpoint that decodes one sequence at a time and returns the
/// whole logit row after each call.
pub(crate) trait FullRowDecoder: Sized + 'static {
    /// One turn's sequence, borrowing the weights.
    type Sequence<'a>: FullRowSequence<Snapshot = Self::Snapshot>
    where
        Self: 'a;

    /// A sequence's state saved after some prompt tokens; [`NoSnapshot`] for
    /// a decoder that cannot save state.
    type Snapshot;

    /// Whether [`FullRowSequence::snapshot`] saves state, which turns on the
    /// session's prompt-prefix cache.
    const SAVES_STATE: bool = false;

    /// Reads the configuration only, refusing a context or state budget the
    /// checkpoint cannot meet before any weights load.
    fn plan(model: &Path, limits: ResidentChatLimits) -> Result<DecoderPlan, DecoderError>;

    fn load(model: &Path, plan: &DecoderPlan) -> Result<Self, DecoderError>;

    /// A fresh sequence; dropping it drops all of its state.
    fn sequence(&self, context_limit: usize) -> Result<Self::Sequence<'_>, String>;

    /// A sequence resumed from `snapshot`, holding its tokens.
    fn sequence_from(
        &self,
        snapshot: &Self::Snapshot,
        context_limit: usize,
    ) -> Result<Self::Sequence<'_>, String>;

    /// Bytes `snapshot` keeps alive, charged to the prefix-cache budget.
    fn snapshot_bytes(snapshot: &Self::Snapshot) -> usize;
}

/// One sequence of a [`FullRowDecoder`].
pub(crate) trait FullRowSequence {
    /// The decoder's [`FullRowDecoder::Snapshot`].
    type Snapshot;

    /// Appends prompt tokens and returns the logits after the last one. A
    /// session may call it more than once per prompt, to save state at
    /// boundaries in between.
    fn prefill(&mut self, tokens: &[i32]) -> Result<Vec<f32>, String>;

    /// Appends one token and returns its logits.
    fn step(&mut self, token: i32) -> Result<Vec<f32>, String>;

    /// The state after every token consumed so far. Called only when the
    /// decoder's [`FullRowDecoder::SAVES_STATE`] is true.
    fn snapshot(&self) -> Result<Self::Snapshot, String>;
}

/// A resident checkpoint for serial chat turns.
pub(crate) struct ChatDecoderSession<D: FullRowDecoder> {
    decoder: D,
    format: ChatFormat,
    vocabulary_size: usize,
    context_limit: usize,
    planned_state_bytes: u64,
    load_ms: f64,
    /// Checkpoint directory, read again only to compile a JSON-schema grammar.
    model: PathBuf,
    sampling_defaults: SamplingDefaults,
    /// Saved prompt-prefix state, bounded by `--prefix-cache-mib`; never
    /// filled unless [`FullRowDecoder::SAVES_STATE`].
    prefix_cache: PrefixCache<D::Snapshot>,
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
        // The weights, configuration, tokenizer and template fix how tokens
        // become state, so they key the cache.
        let config = std::fs::read(model.join("config.json"))
            .map_err(|_| String::from("local model config.json could not be read"))?;
        let identity = format!(
            "model={}\0config={:x}\0tokenizer={}\0template={}",
            model.display(),
            Sha256::digest(&config),
            format.tokenizer().source_sha256(),
            format.template().sha256()
        );
        let prefix_budget = usize::try_from(limits.prefix_cache_bytes()).unwrap_or(usize::MAX);
        Ok(Self {
            decoder,
            format,
            vocabulary_size: plan.vocabulary_size,
            context_limit: limits.context_tokens(),
            planned_state_bytes: plan.planned_state_bytes,
            load_ms: elapsed_ms(started.elapsed()),
            model: model.to_path_buf(),
            sampling_defaults,
            prefix_cache: PrefixCache::new(identity, prefix_budget),
        })
    }

    /// Starts this turn's sequence and prefills `input_ids`, resuming from
    /// the longest saved boundary and saving state at each new boundary when
    /// the decoder can. Returns the sequence, the last prompt logits and the
    /// prompt tokens resumed rather than prefilled.
    fn prefill<'s>(
        decoder: &'s D,
        prefix_cache: &mut PrefixCache<D::Snapshot>,
        format: &ChatFormat,
        context_limit: usize,
        request: ChatRequest<'_>,
        input_ids: &[i32],
    ) -> Result<(D::Sequence<'s>, Vec<f32>, usize, usize), String> {
        if !D::SAVES_STATE {
            let mut sequence = decoder.sequence(context_limit)?;
            let logits = sequence.prefill(input_ids)?;
            return Ok((sequence, logits, 0, 0));
        }
        let (mut sequence, cached) = match prefix_cache.lookup(request.cache_salt, input_ids) {
            Some(hit) => (decoder.sequence_from(hit.value, context_limit)?, hit.tokens),
            None => (decoder.sequence(context_limit)?, 0),
        };
        let mut consumed = cached;
        let mut written = 0;
        for boundary in prefix_cache::boundaries(format, request, input_ids) {
            // A boundary must leave a token to prefill after it, and one at
            // or before the resumed prefix is already cached.
            if boundary.len() <= consumed
                || boundary.len() >= input_ids.len()
                || prefix_cache.contains(request.cache_salt, &boundary)
            {
                continue;
            }
            sequence.prefill(&input_ids[consumed..boundary.len()])?;
            consumed = boundary.len();
            let snapshot = sequence.snapshot()?;
            let bytes = D::snapshot_bytes(&snapshot);
            let extent = boundary.len();
            if prefix_cache.insert(request.cache_salt, boundary, snapshot, bytes) {
                written = written.max(extent);
            }
        }
        let logits = sequence.prefill(&input_ids[consumed..])?;
        Ok((
            sequence,
            logits,
            cached,
            prefix_cache::AcceptedPrefixTokens::from_accepted_extent(written)
                .new_tokens_after(cached),
        ))
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

        deadline.check()?;
        // Ends with the full-vocabulary logit readback, so it times the GPU work.
        let prefill = tracing::info_span!(
            "chat.prefill",
            prompt_tokens = input_ids.len(),
            cached_tokens = Empty,
            prefill_ms = Empty
        );
        let ((mut sequence, mut logits, cached_prompt_tokens, cache_write_tokens), prefill_ms) =
            timed(&prefill, "prefill_ms", || {
                Self::prefill(
                    &self.decoder,
                    &mut self.prefix_cache,
                    &self.format,
                    self.context_limit,
                    request,
                    &input_ids,
                )
            })?;
        prefill.record("cached_tokens", cached_prompt_tokens);
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
                cached_prompt_tokens,
                cache_write_tokens,
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

    /// One `/v1/messages` exchange against `session`.
    pub(crate) fn messages(body: &Value, session: &mut dyn ChatBackend) -> String {
        exchange("/v1/messages", &body.to_string(), |connection, body| {
            let (_, prepared) = crate::anthropic_messages::prepare(body).unwrap();
            crate::anthropic_messages::respond(
                connection,
                &prepared,
                session,
                "m",
                Duration::from_secs(600),
            )
            .unwrap();
        })
    }

    /// One `/v1/responses` exchange against `session`.
    pub(crate) fn responses(body: &Value, session: &mut dyn ChatBackend) -> String {
        use crate::responses::{Request, messages, respond, tools};
        exchange("/v1/responses", &body.to_string(), |connection, body| {
            let request: Request = serde_json::from_slice(body).unwrap();
            let messages = messages(&request).unwrap();
            let tools = tools(&request).unwrap();
            respond(
                connection,
                &request,
                &messages,
                &tools,
                session,
                "m",
                Duration::from_secs(600),
            )
            .unwrap();
        })
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

#[cfg(test)]
mod prefix_cache_tests {
    use std::{cell::RefCell, path::Path, rc::Rc, time::Duration};

    use chat_format::test_model::{ModelDir, VOCABULARY_SIZE};
    use serde_json::json;

    use super::{ChatDecoderSession, DecoderError, DecoderPlan, FullRowDecoder, FullRowSequence};
    use crate::chat_generation::{
        ChatBackend, ChatMessage, ChatRequest, ChatRole, ResidentChatLimits,
    };

    /// A decoder whose logits are a fixed function of the whole token
    /// history, and whose snapshot is that history: resuming from a snapshot
    /// gives exactly the logits of a full prefill only if the session resumes
    /// at the right token. Counts every token it is asked to prefill.
    struct HistoryDecoder {
        prefilled: Rc<RefCell<usize>>,
    }

    thread_local! {
        static PREFILLED: Rc<RefCell<usize>> = Rc::new(RefCell::new(0));
    }

    struct HistorySequence {
        history: Vec<i32>,
        prefilled: Rc<RefCell<usize>>,
    }

    impl HistorySequence {
        /// Never end-of-turn (3); otherwise a hash of the history picks one
        /// of the two words, so a resumed sequence that lost or doubled a
        /// token answers differently.
        fn logits(&self) -> Vec<f32> {
            let hash = self.history.iter().fold(17_u64, |hash, &token| {
                hash.wrapping_mul(31)
                    .wrapping_add(u64::from(token.unsigned_abs()))
            });
            let mut logits = vec![0.0; VOCABULARY_SIZE];
            logits[3] = -100.0;
            logits[4 + usize::from(hash % 2 == 1)] = 10.0;
            logits
        }
    }

    impl FullRowDecoder for HistoryDecoder {
        type Sequence<'a> = HistorySequence;
        type Snapshot = Vec<i32>;
        const SAVES_STATE: bool = true;

        fn plan(_: &Path, _: ResidentChatLimits) -> Result<DecoderPlan, DecoderError> {
            Ok(DecoderPlan {
                vocabulary_size: VOCABULARY_SIZE,
                planned_state_bytes: 0,
            })
        }

        fn load(_: &Path, _: &DecoderPlan) -> Result<Self, DecoderError> {
            Ok(Self {
                prefilled: PREFILLED.with(Rc::clone),
            })
        }

        fn sequence(&self, _: usize) -> Result<HistorySequence, String> {
            self.sequence_from(&Vec::new(), 0)
        }

        fn sequence_from(&self, snapshot: &Vec<i32>, _: usize) -> Result<HistorySequence, String> {
            Ok(HistorySequence {
                history: snapshot.clone(),
                prefilled: Rc::clone(&self.prefilled),
            })
        }

        fn snapshot_bytes(snapshot: &Vec<i32>) -> usize {
            snapshot.len() * size_of::<i32>()
        }
    }

    impl FullRowSequence for HistorySequence {
        type Snapshot = Vec<i32>;

        fn prefill(&mut self, tokens: &[i32]) -> Result<Vec<f32>, String> {
            *self.prefilled.borrow_mut() += tokens.len();
            self.history.extend_from_slice(tokens);
            Ok(self.logits())
        }

        fn step(&mut self, token: i32) -> Result<Vec<f32>, String> {
            self.history.push(token);
            Ok(self.logits())
        }

        fn snapshot(&self) -> Result<Vec<i32>, String> {
            Ok(self.history.clone())
        }
    }

    /// Every message's content, then `Hello` as the generation prompt.
    fn model() -> ModelDir {
        ModelDir::new(
            &json!({"chat_template": "{% for m in messages %}{{ m.content }} {% endfor %}{% if add_generation_prompt %}Hello{% endif %}"}),
            &json!({"eos_token_id": 3, "vocab_size": VOCABULARY_SIZE}),
            None,
        )
    }

    /// Runs two turns of one conversation; returns each turn's generated
    /// tokens, resumed prompt tokens, and prefilled tokens.
    fn two_turns(
        prefix_cache_mib: u32,
        second_salt: Option<&str>,
    ) -> Vec<(Vec<i32>, usize, usize, usize)> {
        let model = model();
        PREFILLED.with(|count| *count.borrow_mut() = 0);
        let mut session = ChatDecoderSession::<HistoryDecoder>::load(
            model.path(),
            ResidentChatLimits::from_mib(64, 1).with_prefix_cache_mib(prefix_cache_mib),
        )
        .expect("load");
        let first = [
            ChatMessage::text(ChatRole::System, "hi hi hi hi"),
            ChatMessage::text(ChatRole::User, "hi"),
        ];
        let second = [
            first[0].clone(),
            first[1].clone(),
            ChatMessage::text(ChatRole::Assistant, "Hello hi"),
            ChatMessage::text(ChatRole::User, "hi hi"),
        ];
        let mut turns = Vec::new();
        for (messages, salt) in [(&first[..], None), (&second[..], second_salt)] {
            let before = PREFILLED.with(|count| *count.borrow());
            let mut request = ChatRequest::new(messages, 3);
            request.cache_salt = salt;
            let generation = session
                .generate_with_timeout(request, Duration::from_secs(10), &mut |_| Ok(()))
                .expect("turn");
            let prefilled = PREFILLED.with(|count| *count.borrow()) - before;
            turns.push((
                generation.generated_token_ids,
                generation.metrics.cached_prompt_tokens,
                prefilled,
                generation.metrics.cache_write_tokens,
            ));
        }
        turns
    }

    #[test]
    fn a_saved_boundary_resumes_the_next_turn_with_identical_output() {
        let cached = two_turns(1, None);
        let uncached = two_turns(0, None);
        // Greedy output does not depend on the cache.
        assert_eq!(cached[0].0, uncached[0].0);
        assert_eq!(cached[1].0, uncached[1].0);
        // Turn 1 prompt: "hi hi hi hi hi Hello" (6 tokens), nothing cached.
        assert_eq!((cached[0].1, cached[0].2), (0, 6));
        // Turn 2 prompt: 4 + 1 + 2 + 2 words + "Hello" = 10 tokens. It
        // resumes turn 1's history boundary ("hi" x 5, 5 tokens) and
        // prefills the other 5; without the cache it prefills all 10.
        assert_eq!((cached[1].1, cached[1].2), (5, 5));
        assert_eq!((uncached[1].1, uncached[1].2), (0, 10));
        assert_eq!(cached[0].3, 5, "first accepted history has five new tokens");
        assert_eq!(
            cached[1].3, 4,
            "nine-token history extends five restored tokens"
        );
        assert_eq!((uncached[0].3, uncached[1].3), (0, 0));
    }

    #[test]
    fn another_salt_resumes_nothing() {
        let salted = two_turns(1, Some("tenant-b"));
        assert_eq!((salted[1].1, salted[1].2), (0, 10));
    }
}
