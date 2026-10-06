//! One chat turn's host-side semantics, independent of how its logits are
//! computed.
//!
//! A turn has three parts, so that the serial session and the batched engine
//! decode with one copy of these rules:
//!
//! - [`TurnStart`]: validate, render, encode, budget the output and build the
//!   token picker, before any model work.
//! - [`TurnLoop`]: per step, pick a token from logits (or accept a token the
//!   GPU picked), classify it against the checkpoint's stop tokens, record its
//!   log probability, and decide whether the turn continues.
//! - [`TurnText`]: turn accepted tokens into streamed text deltas and the
//!   final text.
//!
//! The loop and the text can run on different threads: the engine keeps the
//! loop next to the logits and sends accepted tokens to the request's writer,
//! which owns the text.

use std::{path::Path, time::Instant};

use chat_format::{ChatFormat, QwenIncrementalDecode, TokenClass, TokenId};
use qwen::forward::{Qwen3PickRule, Qwen3RowCandidates, Qwen3TokenPicks};

use super::{
    AppliedSampling, ChatFinishReason, ChatGenerationError, ChatRequest, GenerationDeadline,
    SamplingDefaults, TokenLogprob, TokenPicker, TopLogprob, elapsed_ms, full_row, timed,
    validate_request,
};

/// A validated, rendered and encoded turn, ready for prefill.
pub(crate) struct TurnStart {
    pub(crate) input_ids: Vec<i32>,
    /// The resolved output limit; at least one token.
    pub(crate) max_tokens: u32,
    pub(crate) render_ms: f64,
    /// When the turn began, for time to first token.
    pub(crate) started: Instant,
    picker: TokenPicker,
    top_logprobs: Option<u8>,
    vocabulary_size: usize,
}

/// What a checkpoint contributes to preparing a turn.
#[derive(Clone, Copy)]
pub(crate) struct TurnModel<'a> {
    pub(crate) format: &'a ChatFormat,
    pub(crate) sampling_defaults: SamplingDefaults,
    /// Checkpoint directory, read only to compile a JSON-schema grammar.
    pub(crate) model: &'a Path,
    pub(crate) vocabulary_size: usize,
    pub(crate) context_limit: usize,
}

impl TurnStart {
    /// Prepares `request`, checking `deadline` between host phases. A bad
    /// schema fails here, before any model work.
    pub(crate) fn prepare(
        model: TurnModel<'_>,
        request: ChatRequest<'_>,
        deadline: GenerationDeadline,
    ) -> Result<Self, ChatGenerationError> {
        deadline.check()?;
        validate_request(request)?;
        let started = Instant::now();
        let render = tracing::info_span!("chat.render", render_ms = tracing::field::Empty);
        // Renders and encodes; the render span covers both.
        let (prompt, render_ms) = timed(&render, "render_ms", || {
            model.format.prompt(request.conversation(), true)
        })?;
        let input_ids = prompt.ids;
        deadline.check()?;
        let max_tokens = output_budget(model.context_limit, request.max_tokens, input_ids.len())?;
        let picker = TokenPicker::new(
            request,
            model.sampling_defaults,
            model.model,
            model.vocabulary_size,
        )?;
        Ok(Self {
            input_ids,
            max_tokens,
            render_ms,
            started,
            picker,
            top_logprobs: request.top_logprobs,
            vocabulary_size: model.vocabulary_size,
        })
    }

    /// Splits the turn into its token loop and its text.
    pub(crate) fn into_parts(self) -> (Vec<i32>, TurnLoop, TurnText) {
        let turn_loop = TurnLoop {
            picker: self.picker,
            top_logprobs: self.top_logprobs,
            vocabulary_size: self.vocabulary_size,
            max_tokens: self.max_tokens,
            generated: Vec::with_capacity(self.max_tokens.min(4096) as usize),
            logprobs: Vec::new(),
        };
        (self.input_ids, turn_loop, TurnText::new(self.started))
    }
}

/// Resolves the output limit (`None` fills the remaining context) and rejects
/// one that cannot fit after the prompt.
fn output_budget(
    context_limit: usize,
    requested: Option<u32>,
    prompt_tokens: usize,
) -> Result<u32, ChatGenerationError> {
    let max_tokens = match requested {
        Some(max_tokens) => max_tokens,
        None => u32::try_from(context_limit.saturating_sub(prompt_tokens)).unwrap_or(u32::MAX),
    };
    if max_tokens == 0 {
        return Err(ChatGenerationError::message(format!(
            "chat prompt of {prompt_tokens} tokens leaves no room for output in the {context_limit}-token context",
        )));
    }
    let total_tokens = prompt_tokens
        .checked_add(max_tokens as usize)
        .ok_or_else(|| {
            ChatGenerationError::message("chat prompt plus generation budget overflows")
        })?;
    if total_tokens > context_limit {
        return Err(ChatGenerationError::message(format!(
            "chat requires prompt_tokens + max_tokens <= {context_limit}; received {prompt_tokens} + {max_tokens} = {total_tokens}",
        )));
    }
    Ok(max_tokens)
}

/// Whether a turn continues after a token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TurnStep {
    Continue,
    Stop(ChatFinishReason),
}

/// One accepted token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Accepted {
    pub(crate) token: i32,
    /// Ordinary text, as opposed to a stop token, which is never shown.
    pub(crate) visible: bool,
    pub(crate) step: TurnStep,
}

/// A turn's token selection and stop rules.
pub(crate) struct TurnLoop {
    picker: TokenPicker,
    top_logprobs: Option<u8>,
    vocabulary_size: usize,
    max_tokens: u32,
    generated: Vec<i32>,
    logprobs: Vec<TokenLogprob>,
}

/// What a finished token loop hands to the response.
pub(crate) struct TurnOutput {
    pub(crate) generated: Vec<i32>,
    pub(crate) logprobs: Vec<TokenLogprob>,
    pub(crate) sampling: AppliedSampling,
}

impl TurnLoop {
    /// How the GPU picks this turn's tokens, with the variate `ahead` draws
    /// past the next uncommitted one, or `None` when every step must read the
    /// whole logit row: under a grammar mask, sampling without `top_k`, or a
    /// checkpoint that suppresses tokens (the GPU pick has no suppression
    /// mask).
    pub(crate) fn gpu_rule(&self, format: &ChatFormat, ahead: usize) -> Option<Qwen3PickRule> {
        if format.suppresses_tokens() {
            return None;
        }
        self.picker
            .gpu_rule(ahead, self.top_logprobs, self.vocabulary_size)
    }

    /// Picks the next token from the whole logit row, after the checkpoint's
    /// suppression.
    pub(crate) fn pick(
        &mut self,
        format: &ChatFormat,
        logits: &mut [f32],
    ) -> Result<Accepted, String> {
        format.suppress(logits);
        let (token, grammar_complete) = self.picker.pick(logits)?;
        let receipt = match self.receipt_wanted(format, token)? {
            Some(top) => Some(token_logprob(format, logits, token, top)?),
            None => None,
        };
        self.accept(format, token, grammar_complete, receipt)
    }

    /// Reads back a GPU-picked step and settles its token, and its logprob
    /// receipt when asked, as the full-row path would: from the step's top
    /// candidates when they decide it, else from the whole row. Returns the
    /// accepted token and the GPU's own pick; when they differ, a step queued
    /// on the GPU's pick was built on the wrong token.
    pub(crate) fn settle(
        &mut self,
        format: &ChatFormat,
        picks: &Qwen3TokenPicks,
    ) -> Result<(Accepted, i32), String> {
        let (tokens, candidates) = picks
            .wait_with_candidates()
            .map_err(|error| error.to_string())?;
        let [gpu_token] = tokens[..] else {
            return Err("a decode step picks one token".into());
        };
        let candidates = candidates.as_ref().and_then(|rows| rows.first());
        let mut row = None;
        let token = if let Some(token) = self.picker.pick_from_candidates(gpu_token, candidates)? {
            token
        } else {
            let full = full_row(picks)?;
            let (token, _) = self.picker.pick(&full)?;
            row = Some(full);
            token
        };
        let receipt = match self.receipt_wanted(format, token)? {
            None => None,
            Some(top) => {
                let settled = match candidates {
                    Some(candidates) => candidate_logprob(format, candidates, token, top)?,
                    None => None,
                };
                Some(if let Some(receipt) = settled {
                    receipt
                } else {
                    let full = match row {
                        Some(full) => full,
                        None => full_row(picks)?,
                    };
                    token_logprob(format, &full, token, top)?
                })
            }
        };
        Ok((self.accept(format, token, false, receipt)?, gpu_token))
    }

    /// The `top_logprobs` to record for `token`: only ordinary text tokens
    /// carry receipts.
    fn receipt_wanted(&self, format: &ChatFormat, token: i32) -> Result<Option<u8>, String> {
        let class = format.stops().classify(TokenId::from_model(token)?);
        Ok(self.top_logprobs.filter(|_| class == TokenClass::Normal))
    }

    fn accept(
        &mut self,
        format: &ChatFormat,
        token: i32,
        grammar_complete: bool,
        receipt: Option<TokenLogprob>,
    ) -> Result<Accepted, String> {
        let class = format.stops().classify(TokenId::from_model(token)?);
        if class == TokenClass::Normal && self.top_logprobs.is_some() {
            self.logprobs
                .push(receipt.ok_or("a log probability was requested but not computed")?);
        }
        self.generated.push(token);
        let step = match class {
            // Stop tokens are never visible text; a tool turn's end is
            // complete like any other.
            TokenClass::EndTurn | TokenClass::ToolEnd => TurnStep::Stop(ChatFinishReason::Eos),
            // A schema can close on an ordinary token, such as a final `}`.
            TokenClass::Normal if grammar_complete => TurnStep::Stop(ChatFinishReason::Eos),
            TokenClass::Normal if self.generated.len() >= self.max_tokens as usize => {
                TurnStep::Stop(ChatFinishReason::Length)
            }
            TokenClass::Normal => TurnStep::Continue,
        };
        Ok(Accepted {
            token,
            visible: class == TokenClass::Normal,
            step,
        })
    }

    /// Tokens accepted so far.
    pub(crate) fn generated(&self) -> &[i32] {
        &self.generated
    }

    /// Ends the loop, validating a completed schema output.
    pub(crate) fn finish(self) -> Result<TurnOutput, String> {
        self.picker.check_complete_output()?;
        Ok(TurnOutput {
            generated: self.generated,
            logprobs: self.logprobs,
            sampling: self.picker.applied,
        })
    }
}

/// [`token_logprob`] from a row's top candidates, or `None` when they cannot
/// settle it: `token` is not among them, or the `top + 8` most likely tokens
/// the full-row path would rank are not certain to be. Only the softmax
/// normalizer differs from the full row's, by its f32 summation.
fn candidate_logprob(
    format: &ChatFormat,
    candidates: &Qwen3RowCandidates,
    token: i32,
    top: u8,
) -> Result<Option<TokenLogprob>, String> {
    let head = usize::from(top) + 8;
    let selected = candidates.top.iter().find(|&&(id, _)| id == token);
    let Some(&(_, selected_logit)) = selected.filter(|_| candidates.settles_head(head)) else {
        return Ok(None);
    };
    let top_logprobs = candidates.top[..head]
        .iter()
        .filter_map(|&(id, logit)| {
            let bytes = format.tokenizer().token_bytes(id).ok()?;
            Some(TopLogprob {
                token: String::from_utf8_lossy(&bytes).into_owned(),
                bytes,
                logprob: candidates.logprob(logit),
            })
        })
        .take(usize::from(top))
        .collect();
    let bytes = format.tokenizer().token_bytes(token)?;
    Ok(Some(TokenLogprob {
        token: String::from_utf8_lossy(&bytes).into_owned(),
        bytes,
        logprob: candidates.logprob(selected_logit),
        top_logprobs,
    }))
}

/// Scores `token` and the `top` most likely tokens under the raw logits.
fn token_logprob(
    format: &ChatFormat,
    logits: &[f32],
    token: i32,
    top: u8,
) -> Result<TokenLogprob, String> {
    let maximum = logits
        .iter()
        .map(|&value| f64::from(value))
        .fold(f64::NEG_INFINITY, f64::max);
    let log_normalizer = logits
        .iter()
        .map(|&value| (f64::from(value) - maximum).exp())
        .sum::<f64>()
        .ln();
    let logprob = |index: usize| f64::from(logits[index]) - maximum - log_normalizer;
    let selected = usize::try_from(token)
        .ok()
        .filter(|&index| index < logits.len())
        .ok_or_else(|| String::from("selected token is outside model vocabulary"))?;
    // Padded logit rows past the tokenizer vocabulary have no spelling;
    // over-select a little so skipping them still leaves `top` entries.
    let wanted = usize::from(top);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    let head = (wanted + 8).min(order.len());
    let by_logit = |left: &usize, right: &usize| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then(left.cmp(right))
    };
    if head < order.len() {
        order.select_nth_unstable_by(head, by_logit);
        order.truncate(head);
    }
    order.sort_unstable_by(by_logit);
    let top_logprobs = order
        .into_iter()
        .filter_map(|index| {
            let bytes = format
                .tokenizer()
                .token_bytes(i32::try_from(index).ok()?)
                .ok()?;
            Some(TopLogprob {
                token: String::from_utf8_lossy(&bytes).into_owned(),
                bytes,
                logprob: logprob(index),
            })
        })
        .take(wanted)
        .collect();
    let bytes = format.tokenizer().token_bytes(token)?;
    Ok(TokenLogprob {
        token: String::from_utf8_lossy(&bytes).into_owned(),
        bytes,
        logprob: logprob(selected),
        top_logprobs,
    })
}

/// A turn's streamed and final text.
pub(crate) struct TurnText {
    decoder: QwenIncrementalDecode,
    emitted: String,
    time_to_first_token_ms: Option<f64>,
    started: Instant,
}

impl TurnText {
    pub(crate) fn new(started: Instant) -> Self {
        Self {
            decoder: chat_format::QwenTokenizer::generated_decoder(),
            emitted: String::new(),
            time_to_first_token_ms: None,
            started,
        }
    }

    /// Streams the text a visible token completes, if any.
    pub(crate) fn push(
        &mut self,
        format: &ChatFormat,
        token: i32,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<(), String> {
        if let Some(delta) = format
            .tokenizer()
            .decode_generated_token(&mut self.decoder, token)?
        {
            self.emit(&delta, on_token)?;
        }
        Ok(())
    }

    /// Decodes the whole output (without a final stop token), streams what
    /// the incremental decoder held back, and returns the text and the time
    /// to first token.
    pub(crate) fn finish(
        mut self,
        format: &ChatFormat,
        generated: &[i32],
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<(String, Option<f64>), ChatGenerationError> {
        let visible = match generated.split_last() {
            Some((&last, visible))
                if format.stops().classify(TokenId::from_model(last)?) != TokenClass::Normal =>
            {
                visible
            }
            _ => generated,
        };
        let text = format.tokenizer().decode_generated(visible)?;
        let remaining = text.strip_prefix(&self.emitted).ok_or_else(|| {
            ChatGenerationError::message(
                "incremental tokenizer decoder diverged from complete generated text",
            )
        })?;
        let remaining = remaining.to_owned();
        self.emit(&remaining, on_token)?;
        Ok((text, self.time_to_first_token_ms))
    }

    fn emit(
        &mut self,
        delta: &str,
        on_token: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<(), String> {
        if !delta.is_empty() {
            on_token(delta)?;
            self.emitted.push_str(delta);
            if self.time_to_first_token_ms.is_none() {
                self.time_to_first_token_ms = Some(elapsed_ms(self.started.elapsed()));
            }
        }
        Ok(())
    }
}
