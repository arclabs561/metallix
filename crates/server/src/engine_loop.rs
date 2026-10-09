//! Continuous batching for one paged text decoder.
//!
//! One engine thread owns the decoder and every MLX array. Each generation
//! request runs its protocol handler on its own writer thread, which prepares
//! the turn (render, encode, budget, picker) and submits it here. Every
//! iteration the engine admits waiting sequences in arrival order while the
//! running cap and the pool allow, prefills them, then decodes one token for
//! every running sequence in a single batched forward. Each row's token goes
//! through the row's own [`TurnLoop`], the same stop, sampling and receipt
//! rules the serial session uses, and is sent back to its writer, which owns
//! the text.
//!
//! When the pool cannot take a step, the most recently admitted sequence is
//! preempted by recompute: its blocks are freed and it waits at the front of
//! the queue with its tokens so far, so its blocks usually come back as a
//! prefix-cache hit. A writer that disconnects, or a turn past its deadline,
//! frees its sequence at the next iteration.
//!
//! Work that is not a batched generation (decisions, embeddings) runs on this
//! thread between steps.

use std::{
    collections::VecDeque,
    sync::{
        Arc,
        mpsc::{Receiver, SyncSender, TrySendError, sync_channel},
    },
    time::{Duration, Instant},
};

use chat_format::{ChatFormat, TurnDelta};
use engine::blocks::{HashKeys, SequenceId};
use qwen::forward::{
    BatchDecoded, BatchReadback, PagedChunk, QueuedDecode, Qwen3ForwardError, StepInput,
};
use tracing::field::Empty;

use crate::chat_generation::{
    Accepted, ChatBackend, ChatFinishReason, ChatGeneration, ChatGenerationError,
    ChatGenerationMetrics, ChatRequest, GenerationDeadline, SamplingDefaults, TurnLoop, TurnModel,
    TurnOutput, TurnStart, TurnStep,
};

/// A decoder that keeps many sequences in one paged pool and steps them
/// together: the step-level seam a model needs for continuous batching.
///
/// Decoders whose state cannot be paged or truncated (recurrent hybrids) do
/// not implement it and stay on the serial path.
pub(crate) trait TextDecoder {
    /// Free blocks admitting `input_ids` under `keys` would consume now.
    fn prefill_cost(&self, input_ids: &[i32], keys: &HashKeys) -> Result<usize, DecodeError>;
    /// Free blocks in the pool, cached ones included.
    fn free_blocks(&self) -> usize;
    /// Prefills the next at most `budget` tokens of `seq`'s prompt
    /// `input_ids`, starting `seq` on the first call; see
    /// [`PagedQwen3Session::prefill_chunk`]. On [`DecodeError::OutOfBlocks`]
    /// `seq` is as it was.
    fn prefill_chunk(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
        keys: &HashKeys,
        budget: usize,
    ) -> Result<PagedChunk, DecodeError>;
    /// Modeled milliseconds to prefill `tokens` prompt tokens after `start`
    /// computed ones; see [`PagedQwen3Session::prefill_ms`].
    fn prefill_ms(&self, start: usize, tokens: usize) -> f64;
    /// Tokens per pool block; prefill chunks are whole blocks.
    fn block_tokens(&self) -> usize;
    /// Appends one token to each row's sequence. On an error other than
    /// [`DecodeError::OutOfBlocks`] the rows' sequences are gone.
    fn decode(
        &mut self,
        rows: &[(SequenceId, i32)],
        readback: BatchReadback,
    ) -> Result<BatchDecoded, DecodeError>;
    /// Releases `seq`; unknown sequences are ignored. The owner must first
    /// finish every queued step referencing it, including after errors.
    fn free(&mut self, seq: SequenceId);
    /// A greedy step queued and not yet read back.
    type Queued;
    /// Queues one greedy step whose inputs may be `previous`'s device picks;
    /// see [`PagedQwen3Session::queue_decode`]. On
    /// [`DecodeError::OutOfBlocks`] nothing changed. Other errors retain the
    /// allocations: the owner drains outstanding steps before freeing rows.
    fn queue(
        &mut self,
        rows: &[(SequenceId, StepInput)],
        previous: Option<&Self::Queued>,
    ) -> Result<Self::Queued, DecodeError>;
    /// Waits for `step` and returns each row's pick, in row order. Errors
    /// retain rows for the owner to retire after other queued steps settle.
    fn finish(&mut self, step: &Self::Queued) -> Result<Vec<i32>, DecodeError>;
    /// `step`'s rows, in row order.
    fn queued_rows(step: &Self::Queued) -> &[SequenceId];
    /// Whether greedy steps may be queued ahead of the previous readback.
    fn pipelines(&self) -> bool {
        true
    }
    /// The device allocator's active, cached and peak bytes, for the step
    /// span; `None` when the decoder has no device.
    fn memory(&self) -> Option<DeviceMemory> {
        None
    }
}

/// Device allocator figures in bytes, recorded on each `engine.step` span.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeviceMemory {
    pub(crate) active: usize,
    pub(crate) cache: usize,
    pub(crate) peak: usize,
}

/// Why a decoder step did not run.
#[derive(Debug)]
pub(crate) enum DecodeError {
    /// The pool has no room; nothing changed.
    OutOfBlocks,
    Failed(String),
}

impl From<Qwen3ForwardError> for DecodeError {
    fn from(error: Qwen3ForwardError) -> Self {
        match error {
            Qwen3ForwardError::KvBlocks(engine::blocks::BlockError::OutOfBlocks { .. }) => {
                Self::OutOfBlocks
            }
            other => Self::Failed(other.to_string()),
        }
    }
}

impl<S: std::hash::BuildHasher> TextDecoder for qwen::forward::PagedQwen3Session<'_, S> {
    fn prefill_cost(&self, input_ids: &[i32], keys: &HashKeys) -> Result<usize, DecodeError> {
        Self::prefill_cost(self, input_ids, keys.clone()).map_err(DecodeError::from)
    }

    fn free_blocks(&self) -> usize {
        self.blocks().free_blocks()
    }

    fn prefill_chunk(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
        keys: &HashKeys,
        budget: usize,
    ) -> Result<PagedChunk, DecodeError> {
        Ok(Self::prefill_chunk(
            self,
            seq,
            input_ids,
            keys.clone(),
            budget,
        )?)
    }

    fn prefill_ms(&self, start: usize, tokens: usize) -> f64 {
        Self::prefill_ms(self, start, tokens)
    }

    fn block_tokens(&self) -> usize {
        self.blocks().block_tokens()
    }

    fn decode(
        &mut self,
        rows: &[(SequenceId, i32)],
        readback: BatchReadback,
    ) -> Result<BatchDecoded, DecodeError> {
        Ok(self.decode_batch(rows, readback)?)
    }

    fn free(&mut self, seq: SequenceId) {
        let _ = Self::free(self, seq);
    }

    type Queued = QueuedDecode;

    fn queue(
        &mut self,
        rows: &[(SequenceId, StepInput)],
        previous: Option<&QueuedDecode>,
    ) -> Result<QueuedDecode, DecodeError> {
        Ok(self.queue_decode(rows, previous)?)
    }

    fn finish(&mut self, step: &QueuedDecode) -> Result<Vec<i32>, DecodeError> {
        Ok(self.finish_decode(step)?)
    }

    fn queued_rows(step: &QueuedDecode) -> &[SequenceId] {
        step.rows()
    }

    fn memory(&self) -> Option<DeviceMemory> {
        // Process-wide MLX figures, so a leak shows as a trend across steps.
        Some(DeviceMemory {
            active: mlx_rs::memory::active_memory().ok()?,
            cache: mlx_rs::memory::cache_memory().ok()?,
            peak: mlx_rs::memory::peak_memory().ok()?,
        })
    }
}

impl<T: TextDecoder> TextDecoder for &mut T {
    fn prefill_cost(&self, input_ids: &[i32], keys: &HashKeys) -> Result<usize, DecodeError> {
        T::prefill_cost(self, input_ids, keys)
    }

    fn free_blocks(&self) -> usize {
        T::free_blocks(self)
    }

    fn prefill_chunk(
        &mut self,
        seq: SequenceId,
        input_ids: &[i32],
        keys: &HashKeys,
        budget: usize,
    ) -> Result<PagedChunk, DecodeError> {
        T::prefill_chunk(self, seq, input_ids, keys, budget)
    }

    fn prefill_ms(&self, start: usize, tokens: usize) -> f64 {
        T::prefill_ms(self, start, tokens)
    }

    fn block_tokens(&self) -> usize {
        T::block_tokens(self)
    }

    fn decode(
        &mut self,
        rows: &[(SequenceId, i32)],
        readback: BatchReadback,
    ) -> Result<BatchDecoded, DecodeError> {
        T::decode(self, rows, readback)
    }

    fn free(&mut self, seq: SequenceId) {
        T::free(self, seq);
    }

    type Queued = T::Queued;

    fn queue(
        &mut self,
        rows: &[(SequenceId, StepInput)],
        previous: Option<&T::Queued>,
    ) -> Result<T::Queued, DecodeError> {
        T::queue(self, rows, previous)
    }

    fn finish(&mut self, step: &T::Queued) -> Result<Vec<i32>, DecodeError> {
        T::finish(self, step)
    }

    fn queued_rows(step: &T::Queued) -> &[SequenceId] {
        T::queued_rows(step)
    }

    fn memory(&self) -> Option<DeviceMemory> {
        T::memory(self)
    }

    fn pipelines(&self) -> bool {
        T::pipelines(self)
    }
}

/// What the engine reports to a request's writer.
#[derive(Debug)]
pub(crate) enum Event {
    Prefilled {
        cached_prompt_tokens: usize,
        cache_write_tokens: usize,
        /// Wall time from the prompt's first prefill chunk to its last. With
        /// chunks between decode steps this includes those steps, so it is
        /// the wait before the first token, not prefill compute.
        prefill_ms: f64,
    },
    Token(Accepted),
    Done {
        output: TurnOutput,
        finish_reason: ChatFinishReason,
        decode_ms: Vec<f64>,
    },
    Failed(ChatGenerationError),
}

/// A prepared turn handed to the engine.
pub(crate) struct Submission {
    input_ids: Vec<i32>,
    turn: TurnLoop,
    keys: HashKeys,
    deadline: GenerationDeadline,
    events: SyncSender<Event>,
    cleanup_tx: SyncSender<()>,
}

/// One message to the engine thread: a turn to batch, or exclusive work to
/// run between steps.
pub(crate) enum EngineMessage<J> {
    Submit(Box<Submission>),
    Exclusive(J),
}

/// The checkpoint facts writers prepare turns from.
pub(crate) struct EngineModel {
    pub(crate) format: Arc<ChatFormat>,
    pub(crate) sampling_defaults: SamplingDefaults,
    pub(crate) model: std::path::PathBuf,
    pub(crate) vocabulary_size: usize,
    pub(crate) context_limit: usize,
    pub(crate) pool_bytes: u64,
    pub(crate) load_ms: f64,
}

/// The engine's admission limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EngineLimits {
    /// Sequences decoded together at most.
    pub(crate) max_num_seqs: usize,
    /// Most prompt tokens prefilled in one chunk while sequences are
    /// decoding, a multiple of the block size; a longer prompt is prefilled
    /// in chunks between decode steps instead of stalling them for its whole
    /// length.
    pub(crate) prefill_chunk_tokens: usize,
    /// Target milliseconds per decoded token while prompts are prefilled:
    /// each iteration's chunks get what the last decode step left of it, by
    /// [`TextDecoder::prefill_ms`], at least one block.
    pub(crate) prefill_target_ms: u32,
}

impl EngineLimits {
    /// Generations a child admits at once: a full batch running and as many
    /// again waiting in the engine. Beyond it the child answers busy, and the
    /// front process's queue holds the request.
    #[must_use]
    pub(crate) const fn capacity(self) -> usize {
        self.max_num_seqs * 2
    }

    /// Whether these limits ask for the batching engine at all. With one
    /// sequence at a time a model keeps the serial path.
    #[must_use]
    pub(crate) const fn batches(self) -> bool {
        self.max_num_seqs > 1
    }
}

/// The least share of a decode step's time that prefill chunks get when the
/// step alone exceeds `prefill_target_ms`: TPOT then grows at most this much,
/// and a long prompt still advances by more than one block per step.
const PREFILL_FLOOR_SHARE: f64 = 0.25;

/// The decode step's share of an iteration that took `iteration_ms` and ran
/// chunks modeled at `prefill_ms`. At least half the iteration: the cost model
/// uses fixed rates, so a chunk it overprices (mostly cached, or a faster
/// machine) would otherwise drive the estimate to zero and hand the next
/// chunk the whole target.
fn decode_estimate_ms(iteration_ms: f64, prefill_ms: f64) -> f64 {
    (iteration_ms - prefill_ms).max(iteration_ms * 0.5)
}

/// Milliseconds of prefill an iteration may add beside a `decode_step_ms`
/// decode step: what the step leaves of `target_ms`, but at least
/// [`PREFILL_FLOOR_SHARE`] of the step.
fn chunk_budget_ms(target_ms: f64, decode_step_ms: f64) -> f64 {
    (target_ms - decode_step_ms).max(decode_step_ms * PREFILL_FLOOR_SHARE)
}

/// The largest whole number of `block`-token blocks, at most `ceiling`
/// tokens, whose modeled `cost` fits `budget_ms`; at least one block, so a
/// prompt always progresses.
fn chunk_tokens(
    cost: impl Fn(usize) -> f64,
    budget_ms: f64,
    ceiling: usize,
    block: usize,
) -> usize {
    let mut tokens = ceiling / block * block;
    while tokens > block && cost(tokens) > budget_ms {
        tokens -= block;
    }
    tokens.max(block)
}

struct Sequence {
    id: SequenceId,
    turn: TurnLoop,
    /// Prompt plus every accepted token; a preempted sequence re-prefills
    /// all but the last, which is its next decode input.
    context: Vec<i32>,
    prompt_tokens: usize,
    keys: HashKeys,
    deadline: GenerationDeadline,
    events: SyncSender<Event>,
    cleanup_tx: SyncSender<()>,
    decode_ms: Vec<f64>,
}

impl Sequence {
    fn next_input(&self) -> i32 {
        self.context[self.context.len() - 1]
    }

    fn resumed(&self) -> bool {
        self.context.len() > self.prompt_tokens
    }

    /// The tokens to prefill: the prompt, or for a preempted sequence
    /// everything before its next input.
    fn prefill_ids(&self) -> &[i32] {
        if self.resumed() {
            &self.context[..self.context.len() - 1]
        } else {
            &self.context
        }
    }

    /// Sends `event`; false when the writer is gone. The channel holds a
    /// whole turn's events, so a slow writer never blocks the engine.
    fn send(&self, event: Event) -> bool {
        match self.events.try_send(event) {
            Ok(()) => true,
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => false,
        }
    }

    fn fail(self, error: ChatGenerationError) {
        let _ = self.send(Event::Failed(error));
    }
}

/// A sequence whose prompt is partly prefilled.
struct Prefilling {
    sequence: Sequence,
    /// Prompt tokens in the pool so far, cached ones included.
    done: usize,
    cached_tokens: usize,
    cache_write_tokens: usize,
    started: Instant,
}

/// The engine thread's state between iterations.
struct Engine<'a, D: TextDecoder> {
    decoder: D,
    format: &'a ChatFormat,
    limits: EngineLimits,
    waiting: VecDeque<Sequence>,
    /// Running sequences in admission order.
    running: Vec<Sequence>,
    /// The sequence whose prompt is being prefilled in chunks; its next
    /// chunk runs before any new sequence is admitted. It is never in a
    /// queued step, so its blocks can be freed at once.
    prefilling: Option<Prefilling>,
    next_id: u64,
    steps: u64,
    /// The greedy step queued ahead of the next readback, when every running
    /// sequence picks greedily on the GPU.
    queued: Option<D::Queued>,
    /// Retired rows still owned by a queued step. Their blocks and writer
    /// cleanup lifetimes cannot be released until that step is read back.
    retired: Vec<(SequenceId, SyncSender<()>)>,
    /// When the last step was read back, for per-token timings.
    last_readback: Instant,
    /// The last iteration's decode time: its wall time less the modeled cost
    /// of the prefill chunks it ran. A chunk's budget is measured against it.
    decode_step_ms: f64,
}

/// Runs the engine until every sender is gone. `exclusive` runs other work
/// between steps.
pub(crate) fn run<D: TextDecoder, J>(
    decoder: D,
    format: &ChatFormat,
    limits: EngineLimits,
    messages: &Receiver<EngineMessage<J>>,
    mut exclusive: impl FnMut(J),
) {
    let mut engine = Engine {
        decoder,
        format,
        limits,
        waiting: VecDeque::new(),
        running: Vec::new(),
        prefilling: None,
        next_id: 1,
        steps: 0,
        queued: None,
        retired: Vec::new(),
        last_readback: Instant::now(),
        decode_step_ms: 0.0,
    };
    loop {
        if engine.running.is_empty()
            && engine.waiting.is_empty()
            && engine.prefilling.is_none()
            && engine.queued.is_none()
        {
            match messages.recv() {
                Ok(message) => engine.accept(message, &mut exclusive),
                Err(_) => return,
            }
        }
        while let Ok(message) = messages.try_recv() {
            engine.accept(message, &mut exclusive);
        }
        engine.step();
    }
}

impl<D: TextDecoder> Engine<'_, D> {
    /// A queued row retains both physical pool ownership and its writer's
    /// cleanup lifetime. Unrelated running rows do not delay release.
    fn release_sequence(&mut self, sequence: &Sequence) {
        if self
            .queued
            .as_ref()
            .is_some_and(|step| D::queued_rows(step).contains(&sequence.id))
        {
            self.retired
                .push((sequence.id, sequence.cleanup_tx.clone()));
        } else {
            self.decoder.free(sequence.id);
        }
    }

    /// Call only after local queued steps have finished or moved to `queued`.
    fn release_retired(&mut self) {
        let mut retained = Vec::new();
        for (id, cleanup) in self.retired.drain(..) {
            if self
                .queued
                .as_ref()
                .is_some_and(|step| D::queued_rows(step).contains(&id))
            {
                retained.push((id, cleanup));
            } else {
                self.decoder.free(id);
                drop(cleanup);
            }
        }
        self.retired = retained;
    }

    fn accept<J>(&mut self, message: EngineMessage<J>, exclusive: &mut impl FnMut(J)) {
        match message {
            EngineMessage::Exclusive(job) => exclusive(job),
            EngineMessage::Submit(submission) => {
                let Submission {
                    input_ids,
                    turn,
                    keys,
                    deadline,
                    events,
                    cleanup_tx,
                } = *submission;
                let id = SequenceId(self.next_id);
                self.next_id += 1;
                self.waiting.push_back(Sequence {
                    id,
                    turn,
                    prompt_tokens: input_ids.len(),
                    context: input_ids,
                    keys,
                    deadline,
                    events,
                    cleanup_tx,
                    decode_ms: Vec::new(),
                });
            }
        }
    }

    /// One iteration: drop expired waiters, admit, then decode one token for
    /// every running sequence.
    fn step(&mut self) {
        let span = tracing::info_span!(
            "engine.step",
            step = self.steps,
            running = Empty,
            waiting = Empty,
            free_blocks = Empty,
            mlx.active_bytes = Empty,
            mlx.cache_bytes = Empty,
            mlx.peak_bytes = Empty,
        );
        let _entered = span.enter();
        let now = Instant::now();
        let mut waiting = VecDeque::with_capacity(self.waiting.len());
        for sequence in self.waiting.drain(..) {
            if sequence.deadline.check_at(now).is_err() {
                sequence.fail(ChatGenerationError::DeadlineExceeded);
            } else {
                waiting.push_back(sequence);
            }
        }
        self.waiting = waiting;
        if let Some(prefilling) = self.prefilling.take() {
            if prefilling.sequence.deadline.check_at(now).is_err() {
                self.release_sequence(&prefilling.sequence);
                prefilling
                    .sequence
                    .fail(ChatGenerationError::DeadlineExceeded);
            } else {
                self.prefilling = Some(prefilling);
            }
        }
        // Drain retired writes before admitting work into an otherwise idle
        // pool; those rows are not reusable merely because their turns ended.
        if self.running.is_empty() && self.queued.is_some() {
            self.decode();
        }
        let started = Instant::now();
        let steps = self.steps;
        let prefill_ms = self.admit();
        self.decode();
        if self.steps > steps {
            // The iteration's wall time less the chunks' modeled cost, so the
            // estimate follows the batch size even while prompts stream in.
            let iteration_ms = started.elapsed().as_secs_f64() * 1000.0;
            self.decode_step_ms = decode_estimate_ms(iteration_ms, prefill_ms);
        }
        span.record("running", self.running.len());
        span.record("waiting", self.waiting.len());
        span.record("free_blocks", self.decoder.free_blocks());
        if let Some(memory) = self.decoder.memory() {
            span.record("mlx.active_bytes", memory.active);
            span.record("mlx.cache_bytes", memory.cache);
            span.record("mlx.peak_bytes", memory.peak);
        }
    }

    /// Spends this iteration's prefill budget: the partly prefilled
    /// sequence's next chunk first, then new sequences in arrival order.
    /// With nothing decoding no step can stall, so prompts go in whole;
    /// otherwise chunks get what the last decode step left of
    /// [`EngineLimits::prefill_target_ms`], by the decoder's cost model, at
    /// most `prefill_chunk_tokens` and at least one block each iteration.
    /// When the decode step alone already exceeds the target, chunks still
    /// get [`PREFILL_FLOOR_SHARE`] of the step's time, so prompts keep
    /// moving at a bounded cost to TPOT. Returns the modeled milliseconds of
    /// the chunks run beside decodes.
    fn admit(&mut self) -> f64 {
        let mut budget_ms = chunk_budget_ms(
            f64::from(self.limits.prefill_target_ms),
            self.decode_step_ms,
        );
        let mut spent_ms = 0.0;
        let mut chunked = false;
        loop {
            let Some(prefilling) = self.prefilling.take().or_else(|| self.next_admission()) else {
                return spent_ms;
            };
            // Decided per chunk: once a prompt queued at idle finishes and
            // starts decoding, the next prompt is budgeted like any other.
            let unlimited = self.running.is_empty();
            let Some(budget) = self.chunk_budget(&prefilling, unlimited, chunked, budget_ms) else {
                self.prefilling = Some(prefilling);
                return spent_ms;
            };
            let Prefilling {
                sequence,
                done,
                mut cached_tokens,
                mut cache_write_tokens,
                started,
            } = prefilling;
            let span = tracing::info_span!(
                "engine.prefill",
                tokens = sequence.prefill_ids().len(),
                start = done,
                budget = budget.min(sequence.prefill_ids().len()),
                resumed = sequence.resumed(),
                computed = Empty,
                cached_tokens = Empty,
                chunk_ms = Empty,
            );
            let _entered = span.enter();
            let chunk_started = Instant::now();
            match self.decoder.prefill_chunk(
                sequence.id,
                sequence.prefill_ids(),
                &sequence.keys,
                budget,
            ) {
                Ok(chunk) => {
                    span.record("computed", chunk.computed);
                    span.record("cached_tokens", chunk.cached_tokens);
                    span.record("chunk_ms", chunk_started.elapsed().as_secs_f64() * 1000.0);
                    let start = done + chunk.cached_tokens;
                    let chunk_ms = self.decoder.prefill_ms(start, chunk.computed);
                    budget_ms -= chunk_ms;
                    if !unlimited && chunk.computed > 0 {
                        chunked = true;
                        spent_ms += chunk_ms;
                    }
                    cached_tokens += chunk.cached_tokens;
                    cache_write_tokens += chunk.cache_write_tokens;
                    let done = start + chunk.computed;
                    if let Some(logits) = chunk.logits {
                        self.prefilled(
                            sequence,
                            logits,
                            (cached_tokens, cache_write_tokens),
                            started.elapsed(),
                        );
                    } else {
                        // The budget ran out, or is short of a block.
                        self.prefilling = Some(Prefilling {
                            sequence,
                            done,
                            cached_tokens,
                            cache_write_tokens,
                            started,
                        });
                        return spent_ms;
                    }
                }
                Err(DecodeError::OutOfBlocks) if !self.running.is_empty() => {
                    // Wait for decodes to free blocks. The chunks already
                    // computed are published, so they come back as a prefix
                    // hit; a fresh ID keeps the old one's state apart.
                    self.release_sequence(&sequence);
                    let mut sequence = sequence;
                    sequence.id = SequenceId(self.next_id);
                    self.next_id += 1;
                    self.waiting.push_front(sequence);
                    return spent_ms;
                }
                Err(DecodeError::OutOfBlocks) => {
                    self.release_sequence(&sequence);
                    sequence.fail(ChatGenerationError::message(
                        "the prompt does not fit the KV pool",
                    ));
                }
                Err(DecodeError::Failed(error)) => {
                    self.release_sequence(&sequence);
                    sequence.fail(ChatGenerationError::Message(error));
                }
            }
        }
    }

    /// Tokens `prefilling`'s next chunk may take, or `None` to wait for the
    /// next iteration. Unlimited with nothing decoding. Beside decodes, a
    /// chunk waits while the pool cannot keep a free block per running
    /// sequence, as admission keeps (so it cannot force a preemption that
    /// discards it), and a chunk after the iteration's first needs room for
    /// at least a block in what is left of `budget_ms`.
    fn chunk_budget(
        &self,
        prefilling: &Prefilling,
        unlimited: bool,
        chunked: bool,
        budget_ms: f64,
    ) -> Option<usize> {
        if unlimited {
            return Some(usize::MAX);
        }
        let block = self.decoder.block_tokens();
        if self.decoder.free_blocks() <= self.running.len()
            || (chunked && self.decoder.prefill_ms(prefilling.done, block) > budget_ms)
        {
            return None;
        }
        Some(chunk_tokens(
            |tokens| self.decoder.prefill_ms(prefilling.done, tokens),
            budget_ms,
            self.limits.prefill_chunk_tokens,
            block,
        ))
    }

    /// The first waiting sequence the batch and the pool can take now, or
    /// `None`. A sequence whose prompt cannot be costed fails and the next is
    /// tried.
    fn next_admission(&mut self) -> Option<Prefilling> {
        loop {
            if self.running.len() >= self.limits.max_num_seqs {
                return None;
            }
            let sequence = self.waiting.front()?;
            // Keep one block of headroom per running sequence, so admitting
            // does not immediately force a preemption.
            let cost = match self
                .decoder
                .prefill_cost(sequence.prefill_ids(), &sequence.keys)
            {
                Ok(cost) => cost,
                Err(error) => {
                    if let Some(sequence) = self.waiting.pop_front() {
                        self.release_sequence(&sequence);
                        let message = match error {
                            DecodeError::OutOfBlocks => {
                                "the prompt does not fit the KV pool".to_owned()
                            }
                            DecodeError::Failed(message) => message,
                        };
                        sequence.fail(ChatGenerationError::Message(message));
                    }
                    continue;
                }
            };
            if cost + self.running.len() > self.decoder.free_blocks() && !self.running.is_empty() {
                return None;
            }
            return self.waiting.pop_front().map(|sequence| Prefilling {
                sequence,
                done: 0,
                cached_tokens: 0,
                cache_write_tokens: 0,
                started: Instant::now(),
            });
        }
    }

    /// `prefix` is the prompt tokens served from the prefix cache and those
    /// newly published to it.
    fn prefilled(
        &mut self,
        mut sequence: Sequence,
        mut logits: Vec<f32>,
        prefix: (usize, usize),
        elapsed: Duration,
    ) {
        if sequence.resumed() {
            // The token after this prefix was already chosen and sent.
            self.running.push(sequence);
            return;
        }
        let event = Event::Prefilled {
            cached_prompt_tokens: prefix.0,
            cache_write_tokens: prefix.1,
            prefill_ms: elapsed.as_secs_f64() * 1000.0,
        };
        if !sequence.send(event) {
            self.release_sequence(&sequence);
            return;
        }
        match sequence.turn.pick(self.format, &mut logits) {
            Ok(accepted) => {
                if let Some(sequence) = self.accepted(sequence, accepted) {
                    self.running.push(sequence);
                }
            }
            Err(error) => {
                self.release_sequence(&sequence);
                sequence.fail(ChatGenerationError::Message(error));
            }
        }
    }

    /// Records `accepted` for `sequence` and returns it when it keeps
    /// running; a finished or abandoned sequence is freed.
    fn accepted(&mut self, mut sequence: Sequence, accepted: Accepted) -> Option<Sequence> {
        sequence.context.push(accepted.token);
        if !sequence.send(Event::Token(accepted)) {
            self.release_sequence(&sequence);
            return None;
        }
        match accepted.step {
            TurnStep::Continue => Some(sequence),
            TurnStep::Stop(finish_reason) => {
                self.release_sequence(&sequence);
                let Sequence {
                    turn,
                    events,
                    decode_ms,
                    ..
                } = sequence;
                let event = match turn.finish() {
                    Ok(output) => Event::Done {
                        output,
                        finish_reason,
                        decode_ms,
                    },
                    Err(error) => Event::Failed(ChatGenerationError::Message(error)),
                };
                let _ = events.try_send(event);
                None
            }
        }
    }

    fn decode(&mut self) {
        let now = Instant::now();
        let running = std::mem::take(&mut self.running);
        for sequence in running {
            if sequence.deadline.check_at(now).is_err() {
                self.release_sequence(&sequence);
                sequence.fail(ChatGenerationError::DeadlineExceeded);
            } else {
                self.running.push(sequence);
            }
        }
        if self.running.is_empty() {
            if let Some(queued) = self.queued.take() {
                // Every row left; drain the step so its pool writes land.
                let _ = self.decoder.finish(&queued);
            }
            self.release_retired();
            return;
        }
        let pipelined = self.decoder.pipelines()
            && self
                .running
                .iter()
                .all(|sequence| sequence.turn.verifies_with_gpu_greedy(self.format));
        if pipelined {
            self.decode_pipelined();
        } else if let Some(queued) = self.queued.take() {
            // A row that needs its logits joined: settle the queued step and
            // decode synchronously from the next iteration on.
            self.settle(&queued);
            self.release_retired();
        } else {
            self.decode_now();
        }
    }

    /// Queues the next greedy step, with every carried row's input still on
    /// the device, before reading back the queued one. A row that stops at
    /// the queued step leaves a row in the new step whose pick is discarded.
    fn decode_pipelined(&mut self) {
        let previous = self.queued.take();
        let rows = self
            .running
            .iter()
            .map(|sequence| {
                let carried = previous.as_ref().and_then(|queued| {
                    D::queued_rows(queued)
                        .iter()
                        .position(|&row| row == sequence.id)
                });
                let input = carried.map_or_else(
                    || StepInput::Host(sequence.next_input()),
                    StepInput::Previous,
                );
                (sequence.id, input)
            })
            .collect::<Vec<_>>();
        let span = tracing::info_span!(
            "engine.queue",
            rows = rows.len(),
            carried = rows
                .iter()
                .filter(|(_, input)| matches!(input, StepInput::Previous(_)))
                .count(),
        );
        let next = match span.in_scope(|| self.decoder.queue(&rows, previous.as_ref())) {
            Ok(next) => Some(next),
            Err(DecodeError::OutOfBlocks) => None,
            Err(DecodeError::Failed(error)) => {
                if let Some(previous) = &previous {
                    let _ = self.decoder.finish(previous);
                }
                self.release_retired();
                for sequence in std::mem::take(&mut self.running) {
                    self.release_sequence(&sequence);
                    sequence.fail(ChatGenerationError::Message(error.clone()));
                }
                return;
            }
        };
        // Publish the outstanding step before settling the previous one:
        // a turn that ends there may still own a row in this next step.
        self.queued = next;
        if let Some(previous) = &previous {
            self.settle(previous);
        } else if self.queued.is_none() {
            self.preempt();
        }
        self.release_retired();
    }

    /// Reads back `step` and accepts each still-running row's pick.
    fn settle(&mut self, step: &D::Queued) {
        let span = tracing::info_span!(
            "engine.decode",
            rows = D::queued_rows(step).len(),
            greedy_readback = true,
            step_ms = Empty,
        );
        let _entered = span.enter();
        let tokens = self.decoder.finish(step);
        let step_ms = self.last_readback.elapsed().as_secs_f64() * 1000.0;
        self.last_readback = Instant::now();
        span.record("step_ms", step_ms);
        self.steps += 1;
        let rows = D::queued_rows(step).to_vec();
        let tokens = match tokens {
            Ok(tokens) => tokens,
            Err(error) => {
                let error = match error {
                    DecodeError::Failed(error) => error,
                    DecodeError::OutOfBlocks => String::from("a step readback ran out of blocks"),
                };
                for sequence in std::mem::take(&mut self.running) {
                    if rows.contains(&sequence.id) {
                        self.release_sequence(&sequence);
                        sequence.fail(ChatGenerationError::Message(error.clone()));
                    } else {
                        self.running.push(sequence);
                    }
                }
                return;
            }
        };
        let running = std::mem::take(&mut self.running);
        for mut sequence in running {
            let Some(index) = rows.iter().position(|&row| row == sequence.id) else {
                // Joined after the step was queued; its input is on the host.
                self.running.push(sequence);
                continue;
            };
            sequence.decode_ms.push(step_ms);
            match sequence.turn.accept_gpu_greedy(self.format, tokens[index]) {
                Ok(accepted) => {
                    if let Some(sequence) = self.accepted(sequence, accepted) {
                        self.running.push(sequence);
                    }
                }
                Err(error) => {
                    self.release_sequence(&sequence);
                    sequence.fail(ChatGenerationError::Message(error));
                }
            }
        }
    }

    /// One synchronous step, for batches with a row that needs its logits.
    fn decode_now(&mut self) {
        let rows = self
            .running
            .iter()
            .map(|sequence| (sequence.id, sequence.next_input()))
            .collect::<Vec<_>>();
        let span = tracing::info_span!(
            "engine.decode",
            rows = rows.len(),
            greedy_readback = false,
            step_ms = Empty,
        );
        let _entered = span.enter();
        let started = Instant::now();
        let mut decoded = match self.decoder.decode(&rows, BatchReadback::Logits) {
            Ok(decoded) => decoded,
            Err(DecodeError::OutOfBlocks) => {
                self.preempt();
                return;
            }
            Err(DecodeError::Failed(error)) => {
                for sequence in std::mem::take(&mut self.running) {
                    self.release_sequence(&sequence);
                    sequence.fail(ChatGenerationError::Message(error.clone()));
                }
                return;
            }
        };
        let step_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.last_readback = Instant::now();
        span.record("step_ms", step_ms);
        self.steps += 1;
        let running = std::mem::take(&mut self.running);
        for (index, mut sequence) in running.into_iter().enumerate() {
            sequence.decode_ms.push(step_ms);
            let accepted = match &mut decoded {
                BatchDecoded::Greedy(tokens) => {
                    sequence.turn.accept_gpu_greedy(self.format, tokens[index])
                }
                // Each row is read once, so pick works on it in place.
                BatchDecoded::Logits(rows) => sequence.turn.pick(self.format, &mut rows[index]),
            };
            match accepted {
                Ok(accepted) => {
                    if let Some(sequence) = self.accepted(sequence, accepted) {
                        self.running.push(sequence);
                    }
                }
                Err(error) => {
                    self.release_sequence(&sequence);
                    sequence.fail(ChatGenerationError::Message(error));
                }
            }
        }
    }

    /// Frees the most recently admitted sequence and queues it first, to
    /// recompute when blocks are free. A lone sequence that cannot grow fails.
    fn preempt(&mut self) {
        // A partly prefilled prompt is the newest holder of blocks, and its
        // computed chunks come back as a prefix hit.
        if let Some(Prefilling { sequence, .. }) = self.prefilling.take() {
            tracing::info!(sequence = sequence.id.0, "prefill preempted for decodes");
            self.release_sequence(&sequence);
            let mut sequence = sequence;
            sequence.id = SequenceId(self.next_id);
            self.next_id += 1;
            self.waiting.push_front(sequence);
            return;
        }
        let Some(sequence) = self.running.pop() else {
            return;
        };
        self.release_sequence(&sequence);
        if self.running.is_empty() {
            sequence.fail(ChatGenerationError::message(
                "the sequence outgrew the KV pool",
            ));
        } else {
            tracing::info!(sequence = sequence.id.0, "preempted for recompute");
            // A fresh ID, so a step still queued with the old one cannot be
            // read back into the re-admitted sequence.
            let mut sequence = sequence;
            sequence.id = SequenceId(self.next_id);
            self.next_id += 1;
            self.waiting.push_front(sequence);
        }
    }
}

/// The event receiver cancels a writer's sequence when dropped. Its cleanup
/// channel stays owned by that sequence until the engine releases its state.
/// Wait here on every return (including callback errors) so the HTTP writer
/// keeps its admission and socket until the worker is actually ready again.
struct WriterEvents {
    events: Option<Receiver<Event>>,
    cleanup: Receiver<()>,
}

impl WriterEvents {
    fn recv(&self) -> Result<Event, std::sync::mpsc::RecvError> {
        self.events
            .as_ref()
            .expect("receiver held until drop")
            .recv()
    }
}

impl Drop for WriterEvents {
    fn drop(&mut self) {
        drop(self.events.take());
        // There are no messages: closing the sequence's sole sender is the
        // acknowledgment. Engine teardown also releases this lifetime.
        let _ = self.cleanup.recv();
    }
}

/// The writer-side [`ChatBackend`] for an engine-served model: prepares each
/// turn on the calling thread, submits it, and turns the engine's events into
/// streamed text and a [`ChatGeneration`].
pub(crate) struct EngineClient<J> {
    model: Arc<EngineModel>,
    messages: SyncSender<EngineMessage<J>>,
}

impl<J> Clone for EngineClient<J> {
    fn clone(&self) -> Self {
        Self {
            model: Arc::clone(&self.model),
            messages: self.messages.clone(),
        }
    }
}

impl<J> EngineClient<J> {
    pub(crate) const fn new(
        model: Arc<EngineModel>,
        messages: SyncSender<EngineMessage<J>>,
    ) -> Self {
        Self { model, messages }
    }

    /// The sender exclusive work goes through.
    pub(crate) fn messages(&self) -> &SyncSender<EngineMessage<J>> {
        &self.messages
    }

    fn turn_model(&self) -> TurnModel<'_> {
        TurnModel {
            format: &self.model.format,
            sampling_defaults: self.model.sampling_defaults,
            model: &self.model.model,
            vocabulary_size: self.model.vocabulary_size,
            context_limit: self.model.context_limit,
        }
    }
}

impl<J> ChatBackend for EngineClient<J> {
    fn load_ms(&self) -> f64 {
        self.model.load_ms
    }

    fn generate_with_timeout(
        &mut self,
        request: ChatRequest<'_>,
        timeout: Duration,
        on_token: &mut dyn FnMut(TurnDelta) -> Result<(), String>,
    ) -> Result<ChatGeneration, ChatGenerationError> {
        let deadline = GenerationDeadline::after(timeout);
        let start = TurnStart::prepare(self.turn_model(), request, deadline)?;
        let (render_ms, prompt_tokens) = (start.render_ms, start.input_ids.len());
        let capacity = start.max_tokens as usize + 3;
        let (input_ids, turn, mut text) = start.into_parts();
        let keys = request
            .cache_salt
            .map_or_else(HashKeys::new, |salt| HashKeys::new().with_salt(salt));
        let (events, receiver) = sync_channel(capacity);
        let (cleanup_tx, cleanup) = sync_channel(0);
        let receiver = WriterEvents {
            events: Some(receiver),
            cleanup,
        };
        self.messages
            .send(EngineMessage::Submit(Box::new(Submission {
                input_ids,
                turn,
                keys,
                deadline,
                events,
                cleanup_tx,
            })))
            .map_err(|_| ChatGenerationError::message("the model engine is unavailable"))?;
        let format = Arc::clone(&self.model.format);
        let mut prefill = (0, 0, 0.0);
        loop {
            match receiver.recv() {
                Ok(Event::Prefilled {
                    cached_prompt_tokens,
                    cache_write_tokens,
                    prefill_ms,
                }) => prefill = (cached_prompt_tokens, cache_write_tokens, prefill_ms),
                Ok(Event::Token(accepted)) => {
                    if accepted.visible {
                        // An error cancels the sequence; WriterEvents waits
                        // for its cleanup before this call can return.
                        text.push(&format, accepted.token, on_token)?;
                    }
                }
                Ok(Event::Done {
                    output,
                    finish_reason,
                    decode_ms,
                }) => {
                    let end = text.finish(
                        &format,
                        &output.generated,
                        request.tools,
                        finish_reason == ChatFinishReason::Eos,
                        on_token,
                    )?;
                    let (cached_prompt_tokens, cache_write_tokens, prefill_ms) = prefill;
                    let generated_tokens = output.generated.len();
                    let time_to_first_token_ms = end.time_to_first_token_ms;
                    return Ok(ChatGeneration::finished(
                        end,
                        output,
                        finish_reason,
                        ChatGenerationMetrics {
                            context_tokens: self.model.context_limit,
                            planned_kv_bytes: self.model.pool_bytes,
                            session_load_ms: self.model.load_ms,
                            render_ms,
                            prefill_ms,
                            time_to_first_token_ms,
                            decode_total_ms: decode_ms.iter().sum(),
                            decode_ms,
                            prompt_tokens,
                            cached_prompt_tokens,
                            cache_write_tokens,
                            generated_tokens,
                            // The batched engine does not speculate.
                            speculation: None,
                        },
                    ));
                }
                Ok(Event::Failed(error)) => return Err(error),
                Err(_) => return Err(ChatGenerationError::message("the model engine stopped")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        sync::{
            Arc, Mutex,
            atomic::{AtomicBool, Ordering},
            mpsc::{SyncSender, sync_channel},
        },
        thread,
        time::{Duration, Instant},
    };

    use chat_format::{
        ChatFormat, TurnDelta,
        test_model::{ModelDir, VOCABULARY_SIZE},
    };
    use engine::{
        blocks::{BlockManager, BlockTokens, HashKeys, PoolConfig, SequenceId},
        speculative::SpeculationRequest,
    };
    use qwen::forward::{BatchDecoded, BatchReadback, PagedChunk, StepInput};
    use serde_json::json;

    use super::{
        DecodeError, EngineClient, EngineLimits, EngineMessage, EngineModel, TextDecoder, run,
    };
    use crate::chat_generation::{
        ChatBackend, ChatFinishReason, ChatGenerationError, ChatMessage, ChatRequest, ChatRole,
        SamplingDefaults, SamplingRequest,
    };

    /// Test vocabulary: `Hello` (4) and `hi` (5) are text; `<|im_end|>` (3)
    /// ends the turn.
    const END: i32 = 3;

    /// What the fake decoder observed, for assertions.
    #[derive(Default)]
    struct Observed {
        fail_first_finish: bool,
        fail_next_cost: bool,
        rejected_pool: Option<String>,
        rejected_cleanup_unchanged: Option<bool>,
        live_rows_at_finish_gate: Option<bool>,
        preemptions: usize,
        readbacks: Vec<BatchReadback>,
        largest_batch: usize,
        /// Steps queued before the previous one was read back.
        queued_ahead: usize,
        /// Device work in order: prefill chunks (tokens computed) and decode
        /// steps (rows).
        events: Vec<Step>,
        /// Wall time each decode step takes, to keep turns running while a
        /// test submits more.
        step_delay: Duration,
        /// Engine limits for this harness, read when the engine starts;
        /// `None` keeps the harness default.
        limits: Option<EngineLimits>,
        /// Modeled prefill milliseconds per token; `None` means one.
        prefill_ms_per_token: Option<f64>,
        /// Caps the free blocks the decoder reports, to simulate a pool
        /// with no headroom without changing what it can allocate.
        free_blocks_cap: Option<usize>,
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Step {
        Chunk(usize),
        Decode(usize),
    }

    /// A queued fake step: its rows and where each input comes from.
    struct FakeQueued {
        rows: Vec<SequenceId>,
        inputs: Vec<StepInput>,
    }

    /// A decoder whose next token depends only on the sequence's tokens, with
    /// the real block manager for pool accounting, so batching and
    /// preemption cannot change any sequence's output.
    struct FakeDecoder {
        blocks: BlockManager,
        tokens: HashMap<SequenceId, Vec<i32>>,
        /// Each sequence's last pick, the device value a carried input reads.
        last_pick: HashMap<SequenceId, i32>,
        /// Whether a queued step has not been read back yet.
        in_flight: bool,
        /// False for the reference run: every step synchronous.
        pipelines: bool,
        observed: Arc<Mutex<Observed>>,
        cleanup_gate: Option<(SyncSender<()>, std::sync::mpsc::Receiver<()>)>,
        retired_step_gate: Option<(SyncSender<()>, std::sync::mpsc::Receiver<()>)>,
        finish_calls: usize,
    }

    impl FakeDecoder {
        fn new(blocks: u32, pipelines: bool, observed: Arc<Mutex<Observed>>) -> Self {
            let tokens = BlockTokens::new(4).expect("power of two");
            let slabs = blocks.div_ceil(engine::blocks::SLAB_BLOCKS);
            let mut manager = BlockManager::new(PoolConfig::new(tokens, slabs).expect("pool"));
            // Pin the blocks past `blocks` so the pool holds exactly that many.
            let spare = (slabs * engine::blocks::SLAB_BLOCKS - blocks) as usize;
            if spare > 0 {
                let filler = vec![7_u32; spare * 4];
                let hit = manager.lookup_prefix(&filler, HashKeys::new().with_salt("filler"));
                manager
                    .admit(SequenceId(u64::MAX), hit, &filler)
                    .expect("filler fits");
            }
            Self {
                blocks: manager,
                tokens: HashMap::new(),
                last_pick: HashMap::new(),
                in_flight: false,
                pipelines,
                observed,
                cleanup_gate: None,
                retired_step_gate: None,
                finish_calls: 0,
            }
        }

        fn logits(tokens: &[i32]) -> Vec<f32> {
            // Ends after the sequence reaches a length that is a multiple
            // of 7 past 10 tokens; otherwise alternates by token sum.
            let next = if tokens.len() >= 10 && tokens.len().is_multiple_of(7) {
                END
            } else if tokens.iter().sum::<i32>() % 2 == 0 {
                4
            } else {
                5
            };
            let mut logits = vec![0.0; VOCABULARY_SIZE];
            logits[usize::try_from(next).expect("small")] = 10.0;
            logits
        }

        fn ids(tokens: &[i32]) -> Vec<u32> {
            tokens
                .iter()
                .map(|&id| u32::try_from(id).expect("token"))
                .collect()
        }
    }

    impl TextDecoder for FakeDecoder {
        fn prefill_cost(&self, input_ids: &[i32], keys: &HashKeys) -> Result<usize, DecodeError> {
            let mut observed = self.observed.lock().expect("observed");
            if std::mem::take(&mut observed.fail_next_cost) {
                observed.rejected_pool = Some(format!("{:?}", self.blocks));
                return Err(DecodeError::Failed(
                    "injected prefix validation failure".into(),
                ));
            }
            drop(observed);
            let tokens = Self::ids(input_ids);
            let hit = self.blocks.lookup_prefix(&tokens, keys.clone());
            let cached = hit.cached_tokens();
            self.blocks
                .admit_cost(&hit, tokens.len() - cached)
                .map_err(|error| DecodeError::Failed(error.to_string()))
        }

        fn free_blocks(&self) -> usize {
            let cap = self.observed.lock().expect("observed").free_blocks_cap;
            cap.map_or(self.blocks.free_blocks(), |cap| {
                cap.min(self.blocks.free_blocks())
            })
        }

        fn prefill_chunk(
            &mut self,
            seq: SequenceId,
            input_ids: &[i32],
            keys: &HashKeys,
            budget: usize,
        ) -> Result<PagedChunk, DecodeError> {
            let nothing = PagedChunk {
                computed: 0,
                cached_tokens: 0,
                cache_write_tokens: 0,
                logits: None,
            };
            let tokens = Self::ids(input_ids);
            let (start, cached) = if let Ok(done) = self.blocks.num_computed(seq) {
                let end = self.blocks.chunk_end(done, tokens.len(), budget);
                if end == done {
                    return Ok(nothing);
                }
                self.blocks
                    .allocate(seq, &tokens[done..end])
                    .map_err(|_| DecodeError::OutOfBlocks)?;
                (done, 0)
            } else {
                let hit = self.blocks.lookup_prefix(&tokens, keys.clone());
                let cached = hit.cached_tokens();
                let end = self.blocks.chunk_end(cached, tokens.len(), budget);
                if end == cached {
                    return Ok(nothing);
                }
                self.blocks
                    .admit(seq, hit, &tokens[cached..end])
                    .map_err(|_| DecodeError::OutOfBlocks)?;
                (cached, cached)
            };
            let published_before = self.blocks.published_tokens();
            self.blocks.commit(seq).expect("live");
            let end = self.blocks.num_computed(seq).expect("live");
            {
                let mut observed = self.observed.lock().expect("observed");
                observed.events.push(Step::Chunk(end - start));
            }
            let done = end == tokens.len();
            if done {
                self.tokens.insert(seq, input_ids.to_vec());
            }
            Ok(PagedChunk {
                computed: end - start,
                cached_tokens: cached,
                cache_write_tokens: self.blocks.published_tokens() - published_before,
                logits: done.then(|| Self::logits(input_ids)),
            })
        }

        /// A flat modeled cost per token, wherever it starts.
        #[allow(clippy::cast_precision_loss, reason = "test token counts are small")]
        fn prefill_ms(&self, _start: usize, tokens: usize) -> f64 {
            let per_token = self
                .observed
                .lock()
                .expect("observed")
                .prefill_ms_per_token
                .unwrap_or(1.0);
            tokens as f64 * per_token
        }

        fn block_tokens(&self) -> usize {
            self.blocks.block_tokens()
        }

        fn decode(
            &mut self,
            rows: &[(SequenceId, i32)],
            readback: BatchReadback,
        ) -> Result<BatchDecoded, DecodeError> {
            let needed: usize = rows
                .iter()
                .map(|&(seq, _)| self.blocks.append_cost(seq, 1).expect("live"))
                .sum();
            {
                let mut observed = self.observed.lock().expect("observed");
                if needed > self.blocks.free_blocks() {
                    observed.preemptions += 1;
                    return Err(DecodeError::OutOfBlocks);
                }
                observed.readbacks.push(readback);
                observed.largest_batch = observed.largest_batch.max(rows.len());
                observed.events.push(Step::Decode(rows.len()));
                thread::sleep(observed.step_delay);
            }
            let mut logits = Vec::new();
            for &(seq, token) in rows {
                self.blocks
                    .allocate(seq, &[u32::try_from(token).expect("token")])
                    .expect("checked above");
                self.blocks.commit(seq).expect("live");
                let tokens = self.tokens.get_mut(&seq).expect("live");
                tokens.push(token);
                logits.push(Self::logits(tokens));
            }
            Ok(match readback {
                BatchReadback::Logits => BatchDecoded::Logits(logits),
                BatchReadback::Greedy => BatchDecoded::Greedy(
                    logits
                        .iter()
                        .map(|row| {
                            let best = row
                                .iter()
                                .enumerate()
                                .max_by(|left, right| left.1.total_cmp(right.1))
                                .expect("row")
                                .0;
                            i32::try_from(best).expect("small")
                        })
                        .collect(),
                ),
            })
        }

        fn free(&mut self, seq: SequenceId) {
            if let Some((entered, release)) = self.cleanup_gate.take() {
                entered.send(()).expect("report cleanup");
                release
                    .recv_timeout(Duration::from_secs(10))
                    .expect("release cleanup");
            }
            let _ = self.blocks.free(seq);
            let mut observed = self.observed.lock().expect("observed");
            if let Some(before) = observed.rejected_pool.take() {
                observed.rejected_cleanup_unchanged = Some(before == format!("{:?}", self.blocks));
            }
            drop(observed);
            self.tokens.remove(&seq);
            self.last_pick.remove(&seq);
        }

        type Queued = FakeQueued;

        fn queue(
            &mut self,
            rows: &[(SequenceId, StepInput)],
            _previous: Option<&FakeQueued>,
        ) -> Result<FakeQueued, DecodeError> {
            let needed: usize = rows
                .iter()
                .map(|&(seq, _)| self.blocks.append_cost(seq, 1).expect("live"))
                .sum();
            {
                let mut observed = self.observed.lock().expect("observed");
                if needed > self.blocks.free_blocks() {
                    observed.preemptions += 1;
                    return Err(DecodeError::OutOfBlocks);
                }
                observed.readbacks.push(BatchReadback::Greedy);
                observed.largest_batch = observed.largest_batch.max(rows.len());
                observed.events.push(Step::Decode(rows.len()));
                thread::sleep(observed.step_delay);
                if self.in_flight {
                    observed.queued_ahead += 1;
                }
            }
            for &(seq, input) in rows {
                match input {
                    StepInput::Host(token) => {
                        let token = u32::try_from(token).expect("token");
                        self.blocks.allocate(seq, &[token]).expect("checked above");
                    }
                    StepInput::Previous(_) => {
                        self.blocks
                            .allocate_unresolved(seq, 1)
                            .expect("checked above");
                    }
                }
            }
            self.in_flight = true;
            Ok(FakeQueued {
                rows: rows.iter().map(|&(seq, _)| seq).collect(),
                inputs: rows.iter().map(|&(_, input)| input).collect(),
            })
        }

        fn finish(&mut self, step: &FakeQueued) -> Result<Vec<i32>, DecodeError> {
            self.finish_calls += 1;
            if self.finish_calls == 2 {
                if let Some((entered, release)) = self.retired_step_gate.take() {
                    self.observed
                        .lock()
                        .expect("observed")
                        .live_rows_at_finish_gate = Some(
                        step.rows
                            .iter()
                            .all(|id| self.blocks.num_tokens(*id).is_ok()),
                    );
                    entered.send(()).expect("report queued cleanup");
                    release
                        .recv_timeout(Duration::from_secs(10))
                        .expect("release queued cleanup");
                }
            }
            if self.finish_calls == 1 && self.observed.lock().expect("observed").fail_first_finish {
                // The real paged decoder retains failed rows until its
                // scheduling owner drains every outstanding reference.
                return Err(DecodeError::Failed(String::from(
                    "injected readback failure",
                )));
            }
            self.in_flight = false;
            let mut picks = Vec::with_capacity(step.rows.len());
            for (&seq, &input) in step.rows.iter().zip(&step.inputs) {
                let input = match input {
                    StepInput::Host(token) => token,
                    StepInput::Previous(_) => self.last_pick.get(&seq).copied().unwrap_or(0),
                };
                let Some(tokens) = self.tokens.get_mut(&seq) else {
                    // Freed while queued: the pick is discarded.
                    picks.push(0);
                    continue;
                };
                tokens.push(input);
                let length = tokens.len();
                let pick = argmax(&Self::logits(tokens));
                self.blocks.commit_through(seq, length).expect("live");
                if self.blocks.num_unresolved(seq).expect("live") > 0 {
                    self.blocks
                        .resolve(seq, &[u32::try_from(pick).expect("token")])
                        .expect("one unresolved");
                }
                self.last_pick.insert(seq, pick);
                picks.push(pick);
            }
            Ok(picks)
        }

        fn queued_rows(step: &FakeQueued) -> &[SequenceId] {
            &step.rows
        }

        fn pipelines(&self) -> bool {
            self.pipelines
        }
    }

    fn argmax(row: &[f32]) -> i32 {
        let best = row
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .expect("row")
            .0;
        i32::try_from(best).expect("small")
    }

    type Job = SyncSender<usize>;

    struct Harness {
        _model: ModelDir,
        client: EngineClient<Job>,
        engine: Option<thread::JoinHandle<FakeDecoder>>,
        observed: Arc<Mutex<Observed>>,
        gate: Option<SyncSender<()>>,
    }

    impl Harness {
        fn new(pool_blocks: u32, max_num_seqs: usize) -> Self {
            Self::with_pipelining(pool_blocks, max_num_seqs, true)
        }

        fn with_pipelining(pool_blocks: u32, max_num_seqs: usize, pipelines: bool) -> Self {
            let mut harness = Self::paused(pool_blocks, max_num_seqs, pipelines);
            harness.release();
            harness
        }

        /// The engine thread waits for [`Self::release`], so turns submitted
        /// before it are all waiting when the engine first looks.
        fn paused(pool_blocks: u32, max_num_seqs: usize, pipelines: bool) -> Self {
            Self::paused_with_cleanup(pool_blocks, max_num_seqs, pipelines, None, None)
        }

        fn paused_with_cleanup(
            pool_blocks: u32,
            max_num_seqs: usize,
            pipelines: bool,
            cleanup_gate: Option<(SyncSender<()>, std::sync::mpsc::Receiver<()>)>,
            retired_step_gate: Option<(SyncSender<()>, std::sync::mpsc::Receiver<()>)>,
        ) -> Self {
            let model = ModelDir::new(
                &json!({"chat_template": "{{ messages[0].content }}", "eos_token": "<|im_end|>"}),
                &json!({"eos_token_id": END, "vocab_size": VOCABULARY_SIZE}),
                None,
            );
            let format = Arc::new(ChatFormat::load(model.path(), VOCABULARY_SIZE).expect("format"));
            let observed = Arc::new(Mutex::new(Observed::default()));
            let mut decoder = FakeDecoder::new(pool_blocks, pipelines, Arc::clone(&observed));
            decoder.cleanup_gate = cleanup_gate;
            decoder.retired_step_gate = retired_step_gate;
            let (messages, receiver) = sync_channel(64);
            let engine_format = Arc::clone(&format);
            let (gate, start) = sync_channel::<()>(1);
            let engine = thread::spawn(move || {
                let mut decoder = decoder;
                let _ = start.recv();
                let limits =
                    decoder
                        .observed
                        .lock()
                        .expect("observed")
                        .limits
                        .unwrap_or(EngineLimits {
                            max_num_seqs,
                            prefill_chunk_tokens: TEST_CHUNK,
                            // Generous, so the token ceiling sets every chunk.
                            prefill_target_ms: 10_000,
                        });
                run(
                    &mut decoder,
                    &engine_format,
                    limits,
                    &receiver,
                    |job: Job| {
                        let _ = job.send(7);
                    },
                );
                // Returned for block accounting checks.
                decoder
            });
            let client = EngineClient::new(
                Arc::new(EngineModel {
                    format,
                    sampling_defaults: SamplingDefaults::default(),
                    model: model.path().to_path_buf(),
                    vocabulary_size: VOCABULARY_SIZE,
                    context_limit: 256,
                    pool_bytes: 0,
                    load_ms: 0.0,
                }),
                messages,
            );
            Self {
                _model: model,
                client,
                engine: Some(engine),
                observed,
                gate: Some(gate),
            }
        }

        fn release(&mut self) {
            if let Some(gate) = self.gate.take() {
                let _ = gate.send(());
            }
        }

        /// Stops the engine and returns its decoder.
        fn stop(mut self) -> FakeDecoder {
            let engine = self.engine.take().expect("engine");
            drop(self.client);
            engine.join().expect("engine thread")
        }
    }

    #[test]
    fn prefill_cost_failure_releases_admission_and_next_waiter_progresses() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut harness = Harness::paused(64, 2, false);
        harness.observed.lock().unwrap().fail_next_cost = true;
        let admitted = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
        let (intercept, submissions) = sync_channel(2);
        let mut writers = Vec::new();
        for (index, counter) in admitted.iter().enumerate() {
            let mut client = harness.client.clone();
            client.messages = intercept.clone();
            let admission = crate::serving::test_engine_admission(counter);
            writers.push(thread::spawn(move || {
                let _admission = admission;
                let messages = [ChatMessage::text(ChatRole::User, prompt(index))];
                let mut request = ChatRequest::new(&messages, 40);
                request.sampling = SamplingRequest::GREEDY;
                client.generate_with_timeout(request, Duration::from_secs(10), &mut |_| Ok(()))
            }));
            // Forward each prepared submission in order while the engine is paused.
            let submission = submissions.recv_timeout(Duration::from_secs(5)).unwrap();
            harness.client.messages().send(submission).unwrap();
        }
        assert!(
            admitted
                .iter()
                .all(|counter| counter.load(Ordering::Acquire) == 1)
        );
        harness.release();
        let results = writers
            .into_iter()
            .map(|writer| writer.join().unwrap())
            .collect::<Vec<_>>();
        let observed = Arc::clone(&harness.observed);
        let decoder = harness.stop();
        assert!(
            matches!(&results[0], Err(ChatGenerationError::Message(message))
                if message == "injected prefix validation failure"),
            "rejected waiter did not receive its failure"
        );
        let generation = results[1].as_ref().expect("next waiter completes");
        assert!(!generation.generated_token_ids.is_empty());
        assert_eq!(generation.finish_reason, ChatFinishReason::Eos);
        assert!(
            admitted
                .iter()
                .all(|counter| counter.load(Ordering::Acquire) == 0)
        );
        assert_eq!(
            observed.lock().unwrap().rejected_cleanup_unchanged,
            Some(true)
        );
        assert_eq!(decoder.blocks.free_blocks(), 64);
        assert_eq!(decoder.blocks.sequences(), 0);
        assert!(decoder.tokens.is_empty());
    }

    #[test]
    fn callback_failure_keeps_writer_admission_and_socket_until_engine_cleanup() {
        use std::{
            io::Read,
            net::{TcpListener, TcpStream},
            sync::atomic::{AtomicUsize, Ordering},
        };
        let (entered, cleanup) = sync_channel(1);
        let (release, released) = sync_channel(1);
        let mut harness =
            Harness::paused_with_cleanup(64, 2, false, Some((entered, released)), None);
        let mut client = harness.client.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        peer.set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let (socket, _) = listener.accept().unwrap();
        let admitted = Arc::new(AtomicUsize::new(0));
        let admission = crate::serving::test_engine_admission(&admitted);
        let (callback_tx, callback) = sync_channel(1);
        let writer = thread::spawn(move || {
            let _socket = socket;
            let _admission = admission;
            let messages = [ChatMessage::text(ChatRole::User, prompt(1))];
            let mut request = ChatRequest::new(&messages, 40);
            request.sampling = SamplingRequest::GREEDY;
            client.generate_with_timeout(request, Duration::from_secs(10), &mut |_| {
                callback_tx.send(()).unwrap();
                Err(String::from("client disconnected"))
            })
        });
        harness.release();
        callback.recv_timeout(Duration::from_secs(5)).unwrap();
        cleanup.recv_timeout(Duration::from_secs(5)).unwrap();
        let read = peer.read(&mut [0_u8; 1]);
        let held = admitted.load(Ordering::Acquire);
        // Release the engine before assertions so a red run cleans up threads.
        release.send(()).unwrap();
        assert!(writer.join().unwrap().is_err());
        let decoder = harness.stop();
        assert_eq!(decoder.blocks.sequences(), 0);
        assert_eq!(held, 1, "writer released admission before engine cleanup");
        assert!(
            matches!(read, Err(ref error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut)),
            "socket closed before engine cleanup: {read:?}"
        );
        assert_eq!(admitted.load(Ordering::Acquire), 0);
        assert_eq!(peer.read(&mut [0_u8; 1]).unwrap(), 0);
    }

    #[test]
    fn pipelined_completion_holds_writer_until_retired_step_finishes() {
        check_pipelined_retirement(false);
    }

    #[test]
    fn failed_pipelined_readback_holds_rows_and_writer_until_pending_step_finishes() {
        check_pipelined_retirement(true);
    }

    fn check_pipelined_retirement(fail_first_finish: bool) {
        use std::{
            io::Read,
            net::{TcpListener, TcpStream},
            sync::atomic::{AtomicUsize, Ordering},
        };
        let (entered, settling) = sync_channel(1);
        let (release, released) = sync_channel(1);
        let mut harness =
            Harness::paused_with_cleanup(64, 2, true, None, Some((entered, released)));
        harness.observed.lock().expect("observed").fail_first_finish = fail_first_finish;
        let mut client = harness.client.clone();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        peer.set_read_timeout(Some(Duration::from_millis(200)))
            .unwrap();
        let (socket, _) = listener.accept().unwrap();
        let admitted = Arc::new(AtomicUsize::new(0));
        let admission = crate::serving::test_engine_admission(&admitted);
        let writer = thread::spawn(move || {
            let _socket = socket;
            let _admission = admission;
            let messages = [ChatMessage::text(ChatRole::User, prompt(1))];
            // The previous step ends this turn with a speculative next step
            // still queued against its pool rows.
            let mut request = ChatRequest::new(&messages, 2);
            request.sampling = SamplingRequest::GREEDY;
            client.generate_with_timeout(request, Duration::from_secs(10), &mut |_| Ok(()))
        });
        harness.release();
        let settled = settling.recv_timeout(Duration::from_secs(5));
        let read = peer.read(&mut [0_u8; 1]);
        let held = admitted.load(Ordering::Acquire);
        let _ = release.send(());
        if settled.is_err() {
            // Wake the old idle path so a red run can drain and join instead
            // of abandoning an engine thread with a queued step.
            let (wake, _received) = sync_channel(1);
            let _ = harness
                .client
                .messages()
                .send(EngineMessage::Exclusive(wake));
        }
        assert_eq!(writer.join().unwrap().is_err(), fail_first_finish);
        let decoder = harness.stop();
        assert!(
            settled.is_ok(),
            "queued retirement never drained: {settled:?}"
        );
        assert_eq!(
            decoder
                .observed
                .lock()
                .expect("observed")
                .live_rows_at_finish_gate,
            Some(true),
            "physical rows freed before queued writes settled"
        );
        assert_eq!(
            decoder.blocks.free_blocks(),
            64,
            "all pool rows reusable after cleanup"
        );
        assert_eq!(decoder.blocks.sequences(), 0);
        assert_eq!(
            held, 1,
            "writer released admission before queued GPU cleanup"
        );
        assert!(
            matches!(read, Err(ref error) if matches!(error.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut)),
            "socket closed before queued GPU cleanup: {read:?}"
        );
        assert_eq!(admitted.load(Ordering::Acquire), 0);
        assert_eq!(peer.read(&mut [0_u8; 1]).unwrap(), 0);
    }

    fn prompt(index: usize) -> String {
        // Distinct prompts of different lengths from the two text words.
        (0..=index % 5 + 2)
            .map(|word| {
                if (word + index).is_multiple_of(3) {
                    "hi"
                } else {
                    "Hello"
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Runs `prompts` concurrently, one writer thread each, and returns each
    /// one's streamed text, final text and finish reason.
    fn generate_all(
        client: &EngineClient<Job>,
        prompts: &[String],
        sampling: SamplingRequest,
    ) -> Vec<(String, String, ChatFinishReason, Vec<i32>)> {
        let writers = prompts
            .iter()
            .map(|prompt| {
                let mut client = client.clone();
                let prompt = prompt.clone();
                thread::spawn(move || {
                    let messages = [ChatMessage::text(ChatRole::User, prompt)];
                    let mut request = ChatRequest::new(&messages, 40);
                    request.sampling = sampling;
                    let mut streamed = String::new();
                    let generation = client
                        .generate_with_timeout(request, Duration::from_secs(30), &mut |delta| {
                            if let TurnDelta::Text(text) = delta {
                                streamed.push_str(&text);
                            }
                            Ok(())
                        })
                        .expect("generation");
                    (
                        streamed,
                        generation
                            .turn
                            .as_ref()
                            .map(|turn| turn.text.clone())
                            .expect("plain text parses"),
                        generation.finish_reason,
                        generation.generated_token_ids,
                    )
                })
            })
            .collect::<Vec<_>>();
        writers
            .into_iter()
            .map(|writer| writer.join().expect("writer"))
            .collect()
    }

    fn reference(prompts: &[String]) -> Vec<(String, String, ChatFinishReason, Vec<i32>)> {
        // One sequence at a time in a roomy pool, every step synchronous: no
        // batching, no preemption, no pipelining.
        let harness = Harness::with_pipelining(256, 1, false);
        let outputs = prompts
            .iter()
            .flat_map(|prompt| {
                generate_all(
                    &harness.client,
                    std::slice::from_ref(prompt),
                    SamplingRequest::GREEDY,
                )
            })
            .collect();
        let decoder = harness.stop();
        assert_eq!(decoder.blocks.free_blocks(), 256);
        outputs
    }

    #[test]
    fn sixteen_concurrent_turns_complete_with_their_serial_outputs() {
        let prompts = (0..16).map(prompt).collect::<Vec<_>>();
        let expected = reference(&prompts);
        let harness = Harness::new(256, 16);
        let outputs = generate_all(&harness.client, &prompts, SamplingRequest::GREEDY);
        let observed = Arc::clone(&harness.observed);
        let decoder = harness.stop();
        assert_eq!(outputs, expected);
        for (streamed, text, finish, _) in &outputs {
            assert_eq!(streamed, text);
            assert_eq!(*finish, ChatFinishReason::Eos);
        }
        let observed = observed.lock().expect("observed");
        assert!(observed.largest_batch > 1, "turns were batched");
        assert!(
            observed.queued_ahead > 0,
            "greedy steps were queued before the previous readback"
        );
        assert!(
            observed
                .readbacks
                .iter()
                .all(|readback| *readback == BatchReadback::Greedy),
            "all-greedy batches read back only token IDs"
        );
        assert_eq!(decoder.blocks.free_blocks(), 256, "every block came back");
    }

    #[test]
    fn a_tiny_pool_preempts_and_still_gives_the_serial_outputs() {
        let prompts = (0..8).map(prompt).collect::<Vec<_>>();
        let expected = reference(&prompts);
        // 4-token blocks: each turn needs 3 to 5 blocks, 8 turns need ~36.
        // All eight are waiting before the engine starts, so they are
        // admitted together and outgrow the pool.
        let mut harness = Harness::paused(14, 8, true);
        let writers = {
            let client = harness.client.clone();
            let prompts = prompts.clone();
            thread::spawn(move || generate_all(&client, &prompts, SamplingRequest::GREEDY))
        };
        // Generous: each writer only prepares a short prompt and submits it.
        thread::sleep(Duration::from_millis(500));
        harness.release();
        let outputs = writers.join().expect("writers");
        let observed = Arc::clone(&harness.observed);
        let decoder = harness.stop();
        assert_eq!(outputs, expected);
        assert!(
            observed.lock().expect("observed").preemptions > 0,
            "the pool forced preemption"
        );
        assert_eq!(decoder.blocks.free_blocks(), 14);
    }

    #[test]
    fn sampled_rows_read_back_logits() {
        let prompts = (0..4).map(prompt).collect::<Vec<_>>();
        let harness = Harness::new(256, 4);
        let sampled = SamplingRequest {
            temperature: Some(0.7),
            top_p: Some(1.0),
            top_k: None,
            seed: Some(3),
        };
        let outputs = generate_all(&harness.client, &prompts, sampled);
        let observed = Arc::clone(&harness.observed);
        harness.stop();
        assert_eq!(outputs.len(), 4);
        assert!(
            observed
                .lock()
                .expect("observed")
                .readbacks
                .iter()
                .all(|readback| *readback == BatchReadback::Logits)
        );
    }

    #[test]
    fn a_writer_that_leaves_frees_its_blocks() {
        let harness = Harness::new(64, 4);
        let mut client = harness.client.clone();
        let messages = [ChatMessage::text(ChatRole::User, prompt(3))];
        let request = ChatRequest::new(&messages, 40);
        let error = client
            .generate_with_timeout(request, Duration::from_secs(30), &mut |_| {
                Err(String::from("client went away"))
            })
            .expect_err("the writer fails on its first delta");
        assert!(error.to_string().contains("client went away"));
        // A later turn runs after the abandoned one is freed.
        let outputs = generate_all(&harness.client, &[prompt(1)], SamplingRequest::GREEDY);
        assert_eq!(outputs.len(), 1);
        drop(client);
        let decoder = harness.stop();
        assert_eq!(decoder.blocks.free_blocks(), 64);
    }

    #[test]
    fn an_expired_turn_fails_and_frees_its_blocks() {
        let harness = Harness::new(64, 4);
        let mut client = harness.client.clone();
        let messages = [ChatMessage::text(ChatRole::User, prompt(2))];
        let result = client.generate_with_timeout(
            ChatRequest::new(&messages, 40),
            Duration::ZERO,
            &mut |_| Ok(()),
        );
        assert!(matches!(result, Err(ChatGenerationError::DeadlineExceeded)));
        drop(client);
        let decoder = harness.stop();
        assert_eq!(decoder.blocks.free_blocks(), 64);
    }

    #[test]
    fn exclusive_work_runs_on_the_engine_thread() {
        let harness = Harness::new(64, 4);
        let (reply, answer) = sync_channel(1);
        harness
            .client
            .messages()
            .send(EngineMessage::Exclusive(reply))
            .expect("engine running");
        assert_eq!(answer.recv().expect("job ran"), 7);
        harness.stop();
    }

    #[test]
    fn batched_turns_never_speculate_even_when_asked() {
        // Speculation "on" in a batch behaves as automatic speculation that
        // found no draft: plain decode steps, no verify receipt.
        let prompts = (0..2).map(prompt).collect::<Vec<_>>();
        let expected = reference(&prompts);
        let mut harness = Harness::paused(64, 2, true);
        let writers = [SpeculationRequest::Enabled, SpeculationRequest::Automatic]
            .into_iter()
            .zip(&prompts)
            .map(|(speculation, prompt)| {
                let mut client = harness.client.clone();
                let prompt = prompt.clone();
                thread::spawn(move || {
                    let messages = [ChatMessage::text(ChatRole::User, prompt)];
                    let mut request = ChatRequest::new(&messages, 40);
                    request.sampling = SamplingRequest::GREEDY;
                    request.speculation = speculation;
                    client
                        .generate_with_timeout(request, Duration::from_secs(30), &mut |_| Ok(()))
                        .expect("generation")
                })
            })
            .collect::<Vec<_>>();
        thread::sleep(Duration::from_millis(500));
        harness.release();
        let generations = writers
            .into_iter()
            .map(|writer| writer.join().expect("writer"))
            .collect::<Vec<_>>();
        let observed = Arc::clone(&harness.observed);
        harness.stop();
        for (generation, expected) in generations.iter().zip(&expected) {
            assert!(generation.metrics.speculation.is_none());
            assert_eq!(generation.generated_token_ids, expected.3);
        }
        assert_eq!(observed.lock().expect("observed").largest_batch, 2);
    }

    #[test]
    fn an_overpriced_chunk_leaves_the_decode_at_least_half_the_iteration() {
        // A chunk-free iteration is all decode.
        assert!((super::decode_estimate_ms(12.0, 0.0) - 12.0).abs() < 1e-9);
        assert!((super::decode_estimate_ms(40.0, 15.0) - 25.0).abs() < 1e-9);
        // A 30 ms iteration whose chunk the model priced at 45 ms: the
        // estimate stays at half the iteration instead of zero, so the next
        // budget is 50 - 15 = 35 ms, not the whole 50.
        let estimate = super::decode_estimate_ms(30.0, 45.0);
        assert!((estimate - 15.0).abs() < 1e-9);
        assert!((super::chunk_budget_ms(50.0, estimate) - 35.0).abs() < 1e-9);
    }

    #[test]
    fn a_decode_step_over_the_target_still_leaves_prefill_more_than_a_block() {
        assert!((super::chunk_budget_ms(50.0, 12.0) - 38.0).abs() < 1e-9);
        // A 60 ms step is over the 50 ms target: prefill still gets a quarter
        // of it, which is about ten blocks of a Qwen3-0.6B prompt's start
        // rather than the one-block floor.
        let budget = super::chunk_budget_ms(50.0, 60.0);
        assert!((budget - 15.0).abs() < 1e-9);
        #[allow(clippy::cast_precision_loss, reason = "small counts")]
        let cost = |tokens: usize| {
            let tokens = tokens as f64;
            0.084 * tokens + 2.87e-5 * tokens * (tokens / 2.0)
        };
        assert!(super::chunk_tokens(cost, budget, 1024, 16) >= 160);
    }

    #[test]
    fn chunks_shrink_as_their_start_moves_later() {
        // Qwen3-0.6B's modeled cost: 0.084 ms per token plus attention over
        // the prefix and the chunk's causal half.
        let cost = |start: usize| {
            move |tokens: usize| {
                #[allow(clippy::cast_precision_loss, reason = "small counts")]
                let (tokens, start) = (tokens as f64, start as f64);
                0.084 * tokens + 2.87e-5 * tokens * (start + tokens / 2.0)
            }
        };
        let early = super::chunk_tokens(cost(0), 40.0, 1024, 16);
        let late = super::chunk_tokens(cost(8192), 40.0, 1024, 16);
        assert!(early > late && late >= 16, "early {early}, late {late}");
        assert!(cost(0)(early) <= 40.0 && cost(0)(early + 16) > 40.0);
        assert_eq!(
            super::chunk_tokens(cost(0), 1e9, 1024, 16),
            1024,
            "the ceiling"
        );
        assert_eq!(
            super::chunk_tokens(cost(0), -5.0, 1024, 16),
            16,
            "one block at least"
        );
    }

    /// Prompt tokens per prefill chunk in the harness: two 4-token blocks.
    const TEST_CHUNK: usize = 8;

    /// A turn that decodes until `stop` is set, to keep a batch running
    /// while a test submits prompts beside it.
    fn keeper(client: &EngineClient<Job>, stop: &Arc<AtomicBool>) -> thread::JoinHandle<()> {
        let mut client = client.clone();
        let stop = Arc::clone(stop);
        thread::spawn(move || {
            let messages = [ChatMessage::text(ChatRole::User, prompt(0))];
            // The harness context is 256 tokens.
            let mut request = ChatRequest::new(&messages, 200);
            request.ignore_eos = true;
            let _ = client.generate_with_timeout(request, Duration::from_secs(60), &mut |_| {
                if stop.load(Ordering::Acquire) {
                    Err(String::from("stopped"))
                } else {
                    Ok(())
                }
            });
        })
    }

    fn words(count: usize, index: usize) -> String {
        (0..count)
            .map(|word| {
                if (word + index).is_multiple_of(3) {
                    "hi"
                } else {
                    "Hello"
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// A decode step above the target (60 ms against 50) still leaves prefill
    /// a quarter of the step: a 192-token prompt at 0.01 modeled ms per token
    /// fits that 15 ms in one chunk. With the budget at target minus step
    /// (negative here) it gets one 4-token block per step, 48 steps of 60 ms.
    #[test]
    fn a_long_prompt_beside_a_slow_decode_step_finishes_in_large_chunks() {
        let mut harness = Harness::paused(4096, 4, true);
        {
            let mut observed = harness.observed.lock().expect("observed");
            observed.limits = Some(EngineLimits {
                max_num_seqs: 4,
                prefill_chunk_tokens: 1024,
                prefill_target_ms: 50,
            });
            observed.step_delay = Duration::from_millis(60);
            observed.prefill_ms_per_token = Some(0.01);
        }
        let stop = Arc::new(AtomicBool::new(false));
        let keeper = keeper(&harness.client, &stop);
        harness.release();
        // Several decode steps, so the engine's step estimate is ~60 ms.
        thread::sleep(Duration::from_millis(400));
        let started = Instant::now();
        let outputs = generate_all(&harness.client, &[words(192, 1)], SamplingRequest::GREEDY);
        let elapsed = started.elapsed();
        stop.store(true, Ordering::Release);
        keeper.join().expect("keeper");
        let observed = Arc::clone(&harness.observed);
        harness.stop();
        assert_eq!(outputs.len(), 1);
        let events = observed.lock().expect("observed").events.clone();
        let first_decode = events
            .iter()
            .position(|event| matches!(event, Step::Decode(_)))
            .expect("a decode step ran");
        let chunks = events[first_decode..]
            .iter()
            .filter_map(|event| match event {
                Step::Chunk(tokens) => Some(*tokens),
                Step::Decode(_) => None,
            })
            .collect::<Vec<_>>();
        // One chunk, or a few if the step estimate is still settling; one
        // block per step would be 48.
        assert!(
            chunks.len() <= 4,
            "{} chunks beside a slow decode step: {chunks:?}",
            chunks.len()
        );
        // Its 40 output tokens take ~2.4 s of 60 ms steps on their own; 48
        // prefill steps would add another ~2.9 s.
        assert!(
            elapsed < Duration::from_secs(4),
            "the prompt took {elapsed:?} beside a 60 ms decode step"
        );
    }

    /// A chunk beside decodes waits while free blocks are at or below the
    /// running rows, so it cannot take the block a decode needs, and resumes
    /// when blocks free.
    #[test]
    fn a_chunk_waits_while_the_pool_lacks_headroom_and_resumes_after() {
        let long = words(40, 2);
        let expected = reference(std::slice::from_ref(&long));
        let mut harness = Harness::paused(256, 4, true);
        {
            let mut observed = harness.observed.lock().expect("observed");
            observed.step_delay = Duration::from_millis(10);
            // At one modeled ms per token, a 12 ms target beside a ~10 ms
            // step leaves one 4-token block per step, so the prompt is still
            // prefilling when the pool runs out of headroom.
            observed.limits = Some(EngineLimits {
                max_num_seqs: 4,
                prefill_chunk_tokens: TEST_CHUNK,
                prefill_target_ms: 12,
            });
        }
        let stop = Arc::new(AtomicBool::new(false));
        let keeper = keeper(&harness.client, &stop);
        harness.release();
        thread::sleep(Duration::from_millis(100));
        let writer = {
            let client = harness.client.clone();
            let long = long.clone();
            thread::spawn(move || generate_all(&client, &[long], SamplingRequest::GREEDY))
        };
        let chunk_beside_decode = |events: &[Step]| {
            events
                .iter()
                .position(|event| matches!(event, Step::Decode(_)))
                .is_some_and(|first| {
                    events[first..]
                        .iter()
                        .any(|event| matches!(event, Step::Chunk(_)))
                })
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while !chunk_beside_decode(&harness.observed.lock().expect("observed").events) {
            assert!(Instant::now() < deadline, "no chunk ran beside a decode");
            thread::sleep(Duration::from_millis(1));
        }
        // One running row: report one free block, which leaves no headroom.
        let before = {
            let mut observed = harness.observed.lock().expect("observed");
            observed.free_blocks_cap = Some(1);
            observed.events.len()
        };
        thread::sleep(Duration::from_millis(200));
        let during = {
            let mut observed = harness.observed.lock().expect("observed");
            observed.free_blocks_cap = None;
            observed.events[before..].to_vec()
        };
        let outputs = writer.join().expect("writer");
        stop.store(true, Ordering::Release);
        keeper.join().expect("keeper");
        harness.stop();
        assert!(
            during.iter().any(|event| matches!(event, Step::Decode(_))),
            "decodes kept running: {during:?}"
        );
        assert!(
            !during.iter().any(|event| matches!(event, Step::Chunk(_))),
            "a chunk ran without headroom: {during:?}"
        );
        assert_eq!(outputs, expected, "the prompt resumed and finished");
    }

    #[test]
    fn prompts_queued_at_idle_chunk_once_the_first_is_decoding() {
        let long = |index: usize| {
            (0..40)
                .map(|word| {
                    if (word + index).is_multiple_of(3) {
                        "hi"
                    } else {
                        "Hello"
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let prompts = (0..3).map(long).collect::<Vec<_>>();
        let expected = reference(&prompts);
        // All three are waiting before the engine looks, with nothing
        // decoding: only the first may be prefilled whole.
        let mut harness = Harness::paused(256, 4, true);
        let writers = {
            let client = harness.client.clone();
            let prompts = prompts.clone();
            thread::spawn(move || generate_all(&client, &prompts, SamplingRequest::GREEDY))
        };
        thread::sleep(Duration::from_millis(500));
        harness.release();
        let outputs = writers.join().expect("writers");
        let observed = Arc::clone(&harness.observed);
        harness.stop();
        assert_eq!(outputs, expected, "chunked prefill changed an output");
        let events = observed.lock().expect("observed").events.clone();
        // Until the first decode step, the first prompt is the only one that
        // started with nothing decoding; later prompts may go whole only
        // after every running sequence has finished.
        let first_decode = events
            .iter()
            .position(|event| matches!(event, Step::Decode(_)))
            .expect("a decode step ran");
        let whole = events[..first_decode]
            .iter()
            .filter(|event| matches!(event, Step::Chunk(tokens) if *tokens > TEST_CHUNK))
            .count();
        assert_eq!(
            whole, 1,
            "a prompt queued at idle was prefilled whole after another started decoding: {events:?}"
        );
    }

    #[test]
    fn long_prompts_prefill_in_chunks_between_decode_steps() {
        let long = |index: usize| {
            (0..40)
                .map(|word| {
                    if (word + index).is_multiple_of(3) {
                        "hi"
                    } else {
                        "Hello"
                    }
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        let prompts = (0..3).map(long).collect::<Vec<_>>();
        let expected = reference(&prompts);
        let mut harness = Harness::paused(256, 4, true);
        harness.observed.lock().expect("observed").step_delay = Duration::from_millis(2);
        // A short turn decodes first, so the long prompts arrive while a
        // decode is running and must not stall it for their whole length.
        let keeper = {
            let mut client = harness.client.clone();
            thread::spawn(move || {
                let messages = [ChatMessage::text(ChatRole::User, prompt(0))];
                let mut request = ChatRequest::new(&messages, 40);
                request.ignore_eos = true;
                client
                    .generate_with_timeout(request, Duration::from_secs(30), &mut |_| Ok(()))
                    .expect("keeper");
            })
        };
        harness.release();
        thread::sleep(Duration::from_millis(10));
        let outputs = generate_all(&harness.client, &prompts, SamplingRequest::GREEDY);
        keeper.join().expect("keeper");
        let observed = Arc::clone(&harness.observed);
        let decoder = harness.stop();
        assert_eq!(outputs, expected, "chunked prefill changed an output");
        let events = observed.lock().expect("observed").events.clone();
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, Step::Chunk(tokens) if *tokens > TEST_CHUNK)),
            "a chunk exceeded the budget while decoding: {events:?}"
        );
        assert!(
            events.windows(3).any(|window| matches!(
                window,
                [Step::Chunk(TEST_CHUNK), Step::Decode(_), Step::Chunk(_)]
            )),
            "no decode step ran between two chunks of a prompt: {events:?}"
        );
        assert_eq!(decoder.blocks.free_blocks(), 256, "every block came back");
    }

    #[test]
    fn salts_separate_tenants_and_unsalted_requests_share_one_namespace() {
        let harness = Harness::new(64, 2);
        let messages = [ChatMessage::text(ChatRole::User, prompt(4))];
        let mut cached = Vec::new();
        let mut written = Vec::new();
        for salt in [Some("a"), Some("a"), Some("b"), None, Some(""), None] {
            let mut request = ChatRequest::new(&messages, 40);
            request.cache_salt = salt;
            let generation = harness
                .client
                .clone()
                .generate_with_timeout(request, Duration::from_secs(30), &mut |_| Ok(()))
                .expect("generation");
            cached.push(generation.metrics.cached_prompt_tokens);
            written.push(generation.metrics.cache_write_tokens);
            assert!(
                generation.metrics.cache_write_tokens + generation.metrics.cached_prompt_tokens
                    <= generation.metrics.prompt_tokens
            );
        }
        harness.stop();
        // The repeat of salt "a" hits, as does the second unsalted request
        // (one default namespace, like the serial prefix cache). Different
        // salts, salted against unsalted, and an empty salt never share.
        assert_eq!(cached[0], 0);
        assert!(cached[1] > 0, "{cached:?}");
        assert_eq!(&cached[2..5], [0, 0, 0], "{cached:?}");
        assert!(cached[5] > 0, "{cached:?}");
        assert!(written[0] > 0, "{written:?}");
        assert_eq!(written[1], 0);
        assert_eq!(&written[2..5], [written[0]; 3]);
        assert_eq!(written[5], 0);
    }

    /// Bytes MLX may hold above the post-warmup level once a round of turns
    /// has finished: in-flight step buffers are gone by then.
    const ACTIVE_GROWTH_BOUND: usize = 256 << 20;

    /// MLX trims its cache to the limit as buffers return to it, so the cache
    /// can sit slightly above the limit between trims (0.7 MB measured).
    const CACHE_OVERSHOOT_BOUND: usize = 64 << 20;

    /// `ioreg`'s "In use system memory" of the GPU, in bytes: every process's
    /// Metal allocations, so other jobs add noise. Reported, not asserted.
    fn gpu_in_use_bytes() -> Option<u64> {
        let output = std::process::Command::new("ioreg")
            .args(["-r", "-c", "AGXAccelerator", "-d", "1"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&output.stdout);
        let field = "\"In use system memory\"=";
        let start = text.find(field)? + field.len();
        text[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .ok()
    }

    /// The real engine on Qwen3-0.6B through rounds of concurrent turns whose
    /// batch sizes, prompt lengths and output lengths keep changing, as under
    /// serving load. After each round MLX's active bytes must return to the
    /// post-warmup level and its allocator cache must stay under the cap.
    ///
    /// `METALLIX_ENGINE_TEST_UNCAPPED=1` runs without the cap, to show the
    /// check fails on the build that grew without bound.
    #[test]
    #[ignore = "requires METALLIX_QWEN_MODEL pointing to Qwen3-0.6B on Apple-Silicon Metal"]
    #[allow(
        clippy::too_many_lines,
        reason = "one load, serve, measure and assert sequence"
    )]
    fn real_engine_memory_stays_bounded_across_changing_shapes() {
        let model_dir = std::path::PathBuf::from(
            std::env::var_os("METALLIX_QWEN_MODEL").expect("METALLIX_QWEN_MODEL is required"),
        );
        let uncapped = std::env::var_os("METALLIX_ENGINE_TEST_UNCAPPED").is_some();
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("config.json")).expect("config"),
        )
        .expect("config JSON");
        let vocabulary_size =
            usize::try_from(config["vocab_size"].as_u64().expect("vocab_size")).expect("fits");
        let format = Arc::new(ChatFormat::load(&model_dir, vocabulary_size).expect("format"));
        let (messages, receiver) = sync_channel::<EngineMessage<()>>(64);
        let (ready, started) = sync_channel(1);
        let engine_format = Arc::clone(&format);
        let engine_dir = model_dir.clone();
        // MLX state stays on the thread that builds it.
        let engine = thread::spawn(move || {
            let mut weights =
                qwen::metal::Qwen3MlxWeights::load(&engine_dir).expect("checkpoint load");
            weights
                .prepare_precision(qwen::metal::Qwen3FloatPrecision::BFloat16)
                .expect("bf16 weights");
            let paged = weights.paged_weights();
            let pool = paged
                .pool_for_budget(2048 << 20, BlockTokens::DEFAULT)
                .expect("pool");
            let session = paged.session(pool).expect("session");
            if !uncapped {
                crate::gpu::cap_cache().expect("cache cap");
            }
            ready.send(session.pool_bytes()).expect("test waiting");
            run(
                session,
                &engine_format,
                EngineLimits {
                    max_num_seqs: 16,
                    prefill_chunk_tokens: 1024,
                    prefill_target_ms: 50,
                },
                &receiver,
                |()| {},
            );
        });
        let pool_bytes = started.recv().expect("engine started");
        let client = EngineClient::new(
            Arc::new(EngineModel {
                format,
                sampling_defaults: SamplingDefaults::default(),
                model: model_dir,
                vocabulary_size,
                context_limit: 4096,
                pool_bytes: pool_bytes as u64,
                load_ms: 0.0,
            }),
            messages,
        );
        let words = [
            "river", "stone", "lamp", "orbit", "cedar", "quartz", "harbor",
        ];
        let round = |round: usize| {
            // 4 to 16 turns, 20 to ~900 words, 8 to 64 output tokens.
            let turns = 4 + (round * 5) % 13;
            let writers = (0..turns)
                .map(|turn| {
                    let mut client = client.clone();
                    let length = 20 + (round * 131 + turn * 197) % 880;
                    let max_tokens =
                        8 + u32::try_from((round * 7 + turn * 11) % 57).expect("small");
                    let prompt = (0..length)
                        .map(|word| words[(word + turn + round) % words.len()])
                        .collect::<Vec<_>>()
                        .join(" ");
                    thread::spawn(move || {
                        let messages = [ChatMessage::text(ChatRole::User, prompt)];
                        client
                            .generate_with_timeout(
                                ChatRequest::new(&messages, max_tokens),
                                Duration::from_secs(120),
                                &mut |_| Ok(()),
                            )
                            .expect("generation");
                    })
                })
                .collect::<Vec<_>>();
            for writer in writers {
                writer.join().expect("writer");
            }
        };
        let read = || {
            (
                mlx_rs::memory::active_memory().expect("active"),
                mlx_rs::memory::cache_memory().expect("cache"),
                gpu_in_use_bytes(),
            )
        };
        round(0);
        let (base_active, base_cache, base_gpu) = read();
        let mut worst = (0_usize, 0_usize);
        for index in 1..=24 {
            round(index);
            let (active, cache, gpu) = read();
            println!(
                "engine_memory round={index} uncapped={uncapped} active_bytes={active} \
                 cache_bytes={cache} gpu_in_use_bytes={gpu:?} base_active={base_active} \
                 base_cache={base_cache} base_gpu={base_gpu:?}"
            );
            worst = (worst.0.max(active), worst.1.max(cache));
        }
        drop(client);
        engine.join().expect("engine thread");
        assert!(
            worst.0 <= base_active + ACTIVE_GROWTH_BOUND,
            "active bytes grew from {base_active} to {}",
            worst.0
        );
        assert!(
            worst.1 <= crate::gpu::cache_limit_bytes() + CACHE_OVERSHOOT_BOUND,
            "allocator cache reached {} bytes, above the {} cap plus overshoot",
            worst.1,
            crate::gpu::cache_limit_bytes()
        );
    }
}
