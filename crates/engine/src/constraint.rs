//! Bounded JSON-Schema constrained decoding for one tokenizer vocabulary.
//!
//! This module intentionally owns no model, cache, or tokenizer
//! encoder. It turns one already-produced logit vector into an allowed token
//! and keeps the grammar state for that one sequence.

use std::{collections::BTreeSet, sync::Arc};

use crate::sampling::{SamplingError, sample_categorical};
use jsonschema::{Draft, Validator};
use llguidance::{
    Matcher, ParserFactory,
    api::{StopReason, TopLevelGrammar},
    token_bytes_from_tokenizer_json,
    toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie},
};
use serde_json::Value;
use thiserror::Error;

/// Bounds untrusted schema, tokenizer, and generated-byte inputs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstraintLimits {
    /// Largest serialized JSON Schema accepted by the constructor.
    pub max_schema_bytes: usize,
    /// Largest serialized tokenizer JSON accepted by the constructor.
    pub max_tokenizer_bytes: usize,
    /// Largest decoded output retained by one session.
    pub max_output_bytes: usize,
}

impl ConstraintLimits {
    /// Validates all positive bounds.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::ZeroLimit`] when any bound is zero.
    pub fn validate(self) -> Result<Self, ConstraintError> {
        if self.max_schema_bytes == 0 || self.max_tokenizer_bytes == 0 || self.max_output_bytes == 0
        {
            return Err(ConstraintError::ZeroLimit);
        }
        Ok(self)
    }
}

/// A consumed constrained token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintStep {
    /// The grammar remains open after consuming `token_id`.
    Token {
        /// Token selected from the supplied logits.
        token_id: u32,
    },
    /// Consuming `token_id` completed the grammar.
    Complete {
        /// Final token selected from the supplied logits.
        token_id: u32,
    },
}

/// Natural-log probabilities for the token selected by constrained argmax.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenLogProbs {
    /// Log probability under the unmasked model distribution, including padded rows.
    pub model_logprob: f64,
    /// Log probability after conditioning on grammar-allowed tokenizer tokens.
    pub constrained_logprob: f64,
    /// Natural log of the model probability mass permitted by the grammar.
    pub allowed_log_mass: f64,
}

/// Natural-log probabilities for a caller-variate grammar-masked categorical draw.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingTokenLogProbs {
    /// Log probability under the unmasked temperature-one model, including padded rows.
    pub model_logprob: f64,
    /// Log probability under the temperature-one model conditioned on the grammar mask.
    pub constrained_logprob: f64,
    /// Natural log of the temperature-one model mass permitted by the grammar.
    pub allowed_log_mass: f64,
    /// Log probability under the deployed temperature-conditioned grammar-masked policy.
    pub sampling_logprob: f64,
}

/// How a request ended from this session's perspective.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintFinish {
    /// The grammar reached an accepting terminal state.
    GrammarComplete,
    /// The caller ended generation before grammar completion.
    Truncated,
}

/// A same-session grammar and output checkpoint.
///
/// This does not snapshot model logits, RNG state, or model/cache state. A
/// caller that branches model execution must checkpoint those layers too.
pub struct JsonConstraintCheckpoint {
    owner: Arc<()>,
    matcher: Matcher,
    output: Vec<u8>,
    state: SessionState,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SessionState {
    Active,
    Complete,
    Truncated,
}

/// Content-safe failures for constrained generation.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConstraintError {
    /// At least one public bound was zero.
    #[error("constraint limits must be greater than zero")]
    ZeroLimit,
    /// Serialized schema input exceeded the configured bound.
    #[error("JSON Schema exceeds the configured size limit")]
    SchemaTooLarge,
    /// Serialized tokenizer input exceeded the configured bound.
    #[error("tokenizer JSON exceeds the configured size limit")]
    TokenizerTooLarge,
    /// The model width cannot represent the tokenizer vocabulary.
    #[error("model vocabulary width is smaller than tokenizer vocabulary")]
    ModelVocabularyTooSmall,
    /// EOS was outside the tokenizer vocabulary.
    #[error("EOS token is outside the tokenizer vocabulary")]
    EosOutsideVocabulary,
    /// The schema includes a reference outside its own document.
    #[error("external JSON Schema references are not supported")]
    ExternalReference,
    /// The schema selected a different JSON Schema dialect.
    #[error("only JSON Schema draft 2020-12 is supported")]
    UnsupportedDialect,
    /// Grammar compilation reported an unsupported construct warning.
    #[error("JSON Schema grammar compilation reported unsupported constructs")]
    GrammarWarning,
    /// Grammar or independent-schema compilation failed.
    #[error("JSON Schema compilation failed")]
    SchemaCompilation,
    /// A checkpoint belongs to a different constraint session.
    #[error("constraint checkpoint belongs to a different session")]
    CheckpointMismatch,
    /// Tokenizer metadata could not be converted into a trie.
    #[error("tokenizer metadata could not be converted into a token trie")]
    TokenizerCompilation,
    /// Tokenizer token IDs would require an unreasonable sparse allocation.
    #[error("tokenizer vocabulary is too large")]
    TokenizerVocabularyTooLarge,
    /// The logits did not have exactly the configured model width.
    #[error("logit width does not match the configured model vocabulary")]
    LogitWidth,
    /// At least one logit was NaN or infinite.
    #[error("logits must all be finite")]
    NonFiniteLogit,
    /// The caller-supplied categorical temperature or variate was invalid.
    #[error("sampling temperature or uniform variate is invalid")]
    InvalidSamplingParameter,
    /// No tokenizer token was both grammar-allowed and selectable.
    #[error("grammar left no selectable token")]
    NoAllowedToken,
    /// The grammar stopped in a non-accepting state.
    #[error("grammar stopped before accepting completion")]
    NonAcceptingStop,
    /// A caller attempted to select after the terminal state.
    #[error("constraint session is already terminal")]
    TerminalSession,
    /// The next decoded token would exceed the output bound.
    #[error("constrained output exceeds the configured size limit")]
    OutputTooLarge,
    /// Completion validation was requested before grammar completion.
    #[error("cannot validate a non-complete constrained output")]
    NotComplete,
    /// Grammar-complete output was not valid JSON.
    #[error("grammar-complete output was not valid JSON")]
    InvalidJson,
    /// Independent JSON Schema validation rejected grammar-complete output.
    #[error("independent JSON Schema validation rejected the output")]
    SchemaValidation,
}

/// How many of the highest logits greedy selection checks one token at a
/// time before it computes the full vocabulary mask.
pub const LAZY_ARGMAX_CANDIDATES: usize = 8;

/// The tokenizer half of constrained decoding: the token trie and llguidance's
/// parser factory for one tokenizer, EOS ID and model logit width.
///
/// Building it walks the whole vocabulary several times (hundreds of
/// milliseconds for a 151k-token tokenizer), while each schema then compiles
/// in under a millisecond, so callers build one per model and share it
/// across requests. Sessions from one compiler use the same trie and parser
/// factory, so their masks are the ones a freshly built session would compute.
pub struct JsonConstraintCompiler {
    env: TokEnv,
    factory: ParserFactory,
    model_vocab_size: usize,
    tokenizer_vocab_size: usize,
    eos_token_id: u32,
}

impl JsonConstraintCompiler {
    /// Builds the trie and parser factory for one bounded tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::ZeroLimit`] for any zero limit and
    /// [`ConstraintError::TokenizerTooLarge`] when serialized input exceeds its bound.
    /// Invalid or oversized vocabulary metadata produces
    /// [`ConstraintError::TokenizerCompilation`],
    /// [`ConstraintError::TokenizerVocabularyTooLarge`],
    /// [`ConstraintError::ModelVocabularyTooSmall`] or
    /// [`ConstraintError::EosOutsideVocabulary`]. Parser-factory creation can
    /// return [`ConstraintError::SchemaCompilation`].
    pub fn new(
        tokenizer_json: &Value,
        eos_token_id: u32,
        model_vocab_size: usize,
        limits: ConstraintLimits,
    ) -> Result<Self, ConstraintError> {
        let limits = limits.validate()?;
        check_serialized_bound(tokenizer_json, limits.max_tokenizer_bytes)
            .map_err(|()| ConstraintError::TokenizerTooLarge)?;
        Self::compile(tokenizer_json, eos_token_id, model_vocab_size)
    }

    fn compile(
        tokenizer_json: &Value,
        eos_token_id: u32,
        model_vocab_size: usize,
    ) -> Result<Self, ConstraintError> {
        let declared_tokenizer_vocab = declared_tokenizer_vocab_size(tokenizer_json)?;
        if declared_tokenizer_vocab > MAX_TOKENIZER_VOCABULARY {
            return Err(ConstraintError::TokenizerVocabularyTooLarge);
        }
        if declared_tokenizer_vocab > model_vocab_size {
            return Err(ConstraintError::ModelVocabularyTooSmall);
        }
        let token_bytes = token_bytes_from_tokenizer_json(tokenizer_json)
            .map_err(|_| ConstraintError::TokenizerCompilation)?;
        let tokenizer_vocab_size = token_bytes.len();
        if model_vocab_size < tokenizer_vocab_size {
            return Err(ConstraintError::ModelVocabularyTooSmall);
        }
        let eos_index =
            usize::try_from(eos_token_id).map_err(|_| ConstraintError::EosOutsideVocabulary)?;
        if eos_index >= tokenizer_vocab_size {
            return Err(ConstraintError::EosOutsideVocabulary);
        }
        let vocab_size = u32::try_from(tokenizer_vocab_size)
            .map_err(|_| ConstraintError::TokenizerCompilation)?;
        let info = TokRxInfo {
            vocab_size,
            tok_eos: eos_token_id,
            tok_bos: None,
            tok_pad: None,
            tok_unk: None,
            tok_end_of_turn: None,
        };
        let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &token_bytes)));
        let factory =
            ParserFactory::new_simple(&env).map_err(|_| ConstraintError::SchemaCompilation)?;
        Ok(Self {
            env,
            factory,
            model_vocab_size,
            tokenizer_vocab_size,
            eos_token_id,
        })
    }

    /// The EOS ID the grammar ends output on.
    #[must_use]
    pub const fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// The model logit width every session's logits must have.
    #[must_use]
    pub const fn model_vocab_size(&self) -> usize {
        self.model_vocab_size
    }

    /// Compiles one bounded JSON Schema into a fresh session.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::ZeroLimit`] for any zero limit,
    /// [`ConstraintError::SchemaTooLarge`] when the schema exceeds its bound,
    /// [`ConstraintError::ExternalReference`] for a nonlocal reference, and
    /// [`ConstraintError::UnsupportedDialect`] for an unsupported dialect.
    /// Schema compilation can return [`ConstraintError::SchemaCompilation`]
    /// or [`ConstraintError::GrammarWarning`]. Existing sessions are unchanged.
    pub fn session(
        &self,
        schema: Value,
        limits: ConstraintLimits,
    ) -> Result<JsonConstraintSession, ConstraintError> {
        let limits = limits.validate()?;
        check_schema(&schema, limits)?;
        self.compile_session(schema, limits)
    }

    fn compile_session(
        &self,
        schema: Value,
        limits: ConstraintLimits,
    ) -> Result<JsonConstraintSession, ConstraintError> {
        let validator = jsonschema::options()
            .with_draft(Draft::Draft202012)
            .offline()
            .build(&schema)
            .map_err(|_| ConstraintError::SchemaCompilation)?;
        let mut matcher = Matcher::new(
            self.factory
                .create_parser(TopLevelGrammar::from_json_schema(schema)),
        );
        if matcher.is_error() || !matcher.grammar_warnings().is_empty() {
            return Err(ConstraintError::GrammarWarning);
        }

        Ok(JsonConstraintSession {
            env: Arc::clone(&self.env),
            owner: Arc::new(()),
            matcher,
            validator,
            model_vocab_size: self.model_vocab_size,
            tokenizer_vocab_size: self.tokenizer_vocab_size,
            eos_token_id: self.eos_token_id,
            output: Vec::new(),
            legal_mask: vec![false; self.model_vocab_size],
            limits,
            state: SessionState::Active,
        })
    }
}

/// One JSON-Schema grammar session bound to a tokenizer and model logit width.
pub struct JsonConstraintSession {
    env: TokEnv,
    /// Identifies this session's checkpoints; the trie may be shared.
    owner: Arc<()>,
    matcher: Matcher,
    validator: Validator,
    model_vocab_size: usize,
    tokenizer_vocab_size: usize,
    eos_token_id: u32,
    output: Vec<u8>,
    legal_mask: Vec<bool>,
    limits: ConstraintLimits,
    state: SessionState,
}

impl JsonConstraintSession {
    /// Builds a bounded session for one tokenizer vocabulary and JSON Schema.
    ///
    /// This compiles the tokenizer too; a caller serving many requests for one
    /// model builds a [`JsonConstraintCompiler`] once and calls
    /// [`JsonConstraintCompiler::session`] instead.
    ///
    /// `model_vocab_size` is the width of the logit rows passed to the
    /// `select_*` methods; it may exceed the tokenizer's vocabulary when the
    /// model pads its output rows.
    ///
    /// # Errors
    ///
    /// * [`ConstraintError::ZeroLimit`] for a zero bound in `limits`.
    /// * [`ConstraintError::TokenizerTooLarge`] and
    ///   [`ConstraintError::SchemaTooLarge`] when the serialized input passes
    ///   its bound.
    /// * [`ConstraintError::ExternalReference`] for a `$ref` outside the
    ///   schema, and [`ConstraintError::UnsupportedDialect`] for a dialect
    ///   other than draft 2020-12.
    /// * [`ConstraintError::TokenizerVocabularyTooLarge`],
    ///   [`ConstraintError::ModelVocabularyTooSmall`],
    ///   [`ConstraintError::EosOutsideVocabulary`] and
    ///   [`ConstraintError::TokenizerCompilation`] for a tokenizer that does
    ///   not fit the model or cannot be read.
    /// * [`ConstraintError::SchemaCompilation`] and
    ///   [`ConstraintError::GrammarWarning`] for a schema the grammar compiler
    ///   rejects or only partly supports.
    pub fn new(
        tokenizer_json: &Value,
        eos_token_id: u32,
        model_vocab_size: usize,
        schema: Value,
        limits: ConstraintLimits,
    ) -> Result<Self, ConstraintError> {
        let limits = limits.validate()?;
        check_serialized_bound(tokenizer_json, limits.max_tokenizer_bytes)
            .map_err(|()| ConstraintError::TokenizerTooLarge)?;
        check_schema(&schema, limits)?;
        JsonConstraintCompiler::compile(tokenizer_json, eos_token_id, model_vocab_size)?
            .compile_session(schema, limits)
    }

    /// Selects and consumes the finite maximum grammar-allowed tokenizer logit.
    ///
    /// The model's own top choices are usually legal, so the few highest
    /// logits are checked one token at a time first, in the order the full
    /// mask would rank them (higher logit, then lower ID); the first legal
    /// one is exactly the constrained maximum. Only after
    /// [`LAZY_ARGMAX_CANDIDATES`] illegal candidates does this compute the
    /// mask over the whole vocabulary.
    ///
    /// # Errors
    ///
    /// Decoded bytes advance only on success. An unexpected matcher failure
    /// while consuming a validated token can leave the matcher in its error state.
    ///
    /// * [`ConstraintError::TerminalSession`] after completion or
    ///   [`JsonConstraintSession::finish`].
    /// * [`ConstraintError::LogitWidth`] unless `logits` has the model's
    ///   vocabulary width, and [`ConstraintError::NonFiniteLogit`] for a NaN
    ///   or infinite logit.
    /// * [`ConstraintError::NoAllowedToken`] when the grammar allows no token.
    /// * [`ConstraintError::OutputTooLarge`] when the token's bytes would pass
    ///   [`ConstraintLimits::max_output_bytes`].
    /// * [`ConstraintError::NonAcceptingStop`] when the grammar stops, or would
    ///   stop after this token, in a non-accepting state.
    pub fn select_argmax(&mut self, logits: &[f32]) -> Result<ConstraintStep, ConstraintError> {
        if let Some(index) = self.first_legal_by_logit(logits)? {
            return self.commit_in_place(index);
        }
        self.select_argmax_by_mask(logits)
    }

    /// [`Self::select_argmax`] through the full vocabulary mask.
    fn select_argmax_by_mask(&mut self, logits: &[f32]) -> Result<ConstraintStep, ConstraintError> {
        self.select(logits, false).map(|(step, _)| step)
    }

    /// The highest-ranked legal token among the [`LAZY_ARGMAX_CANDIDATES`]
    /// best logits, or `None` to fall back to the mask (including every case
    /// the mask path reports as an error, so errors stay identical).
    fn first_legal_by_logit(&mut self, logits: &[f32]) -> Result<Option<usize>, ConstraintError> {
        if self.state != SessionState::Active
            || logits.len() != self.model_vocab_size
            || !all_finite(logits)
            || self.matcher.is_stopped()
            // With a canonical tokenizer, llguidance's mask allows only the
            // canonical tokenization of forced bytes, while validate_tokens
            // accepts any split of them. The approximate environment built
            // here is not canonical, so both allow the same tokens.
            || self.env.tokenize_is_canonical()
        {
            return Ok(None);
        }
        // Padded model rows past the tokenizer are never selectable.
        let ranked = |left: &usize, right: &usize| {
            logits[*right]
                .total_cmp(&logits[*left])
                .then(left.cmp(right))
        };
        let mut candidates: Vec<usize> = Vec::with_capacity(LAZY_ARGMAX_CANDIDATES + 1);
        for (index, &logit) in logits[..self.tokenizer_vocab_size].iter().enumerate() {
            // Indices rise, so a logit that does not beat the current
            // last-ranked one (ties included) ranks below it: the common
            // case costs one comparison.
            if candidates.len() == LAZY_ARGMAX_CANDIDATES
                && logit <= logits[candidates[LAZY_ARGMAX_CANDIDATES - 1]]
            {
                continue;
            }
            let position = candidates
                .binary_search_by(|candidate| ranked(candidate, &index))
                .unwrap_or_else(|position| position);
            candidates.insert(position, index);
            candidates.truncate(LAZY_ARGMAX_CANDIDATES);
        }
        for index in candidates {
            let token_id =
                u32::try_from(index).map_err(|_| ConstraintError::TokenizerCompilation)?;
            let legal = self
                .matcher
                .validate_tokens(&[token_id])
                .map_err(|_| ConstraintError::NonAcceptingStop)?;
            if legal == 1 {
                return Ok(Some(index));
            }
        }
        Ok(None)
    }

    /// Consumes a token already validated against the live matcher, without
    /// the clone the mask path commits from. A completion that does not
    /// accept is rolled back. Validation rules out a failed consumption; if
    /// llguidance failed anyway, the matcher would keep its error state.
    fn commit_in_place(
        &mut self,
        selected_index: usize,
    ) -> Result<ConstraintStep, ConstraintError> {
        let token_id =
            u32::try_from(selected_index).map_err(|_| ConstraintError::TokenizerCompilation)?;
        let token_bytes = if token_id == self.eos_token_id {
            Vec::new()
        } else {
            self.env.tok_trie().decode(&[token_id])
        };
        if self.output.len().saturating_add(token_bytes.len()) > self.limits.max_output_bytes {
            return Err(ConstraintError::OutputTooLarge);
        }
        self.matcher
            .consume_token(token_id)
            .map_err(|_| ConstraintError::NonAcceptingStop)?;
        let complete = if self.matcher.is_stopped() {
            if !is_accepting_terminal(&mut self.matcher)? {
                self.matcher
                    .rollback(1)
                    .map_err(|_| ConstraintError::NonAcceptingStop)?;
                return Err(ConstraintError::NonAcceptingStop);
            }
            true
        } else {
            false
        };
        self.output.extend_from_slice(&token_bytes);
        if complete {
            self.state = SessionState::Complete;
            Ok(ConstraintStep::Complete { token_id })
        } else {
            Ok(ConstraintStep::Token { token_id })
        }
    }

    /// Captures grammar state and decoded bytes for a reversible branch.
    #[must_use]
    pub fn checkpoint(&self) -> JsonConstraintCheckpoint {
        JsonConstraintCheckpoint {
            owner: Arc::clone(&self.owner),
            matcher: self.matcher.deep_clone(),
            output: self.output.clone(),
            state: self.state,
        }
    }

    /// Restores a checkpoint created by this session.
    ///
    /// The checkpoint is consumed so a caller cannot accidentally reuse a
    /// stale branch after restoring it once.
    ///
    /// # Errors
    ///
    /// Returns [`ConstraintError::CheckpointMismatch`], leaving the session
    /// unchanged, when the checkpoint came from another session.
    pub fn restore(&mut self, checkpoint: JsonConstraintCheckpoint) -> Result<(), ConstraintError> {
        if !Arc::ptr_eq(&self.owner, &checkpoint.owner) {
            return Err(ConstraintError::CheckpointMismatch);
        }
        self.matcher = checkpoint.matcher;
        self.output = checkpoint.output;
        self.state = checkpoint.state;
        self.legal_mask.fill(false);
        Ok(())
    }

    /// Selects and consumes the constrained maximum, returning pre-consumption log probabilities.
    ///
    /// # Errors
    ///
    /// The grammar state and decoded bytes advance only on success. The mask
    /// is computed on the live matcher, so an unexpected matcher failure
    /// there, or while consuming a selected token, can leave the matcher in
    /// its error state.
    ///
    /// * [`ConstraintError::TerminalSession`] after completion or
    ///   [`JsonConstraintSession::finish`].
    /// * [`ConstraintError::LogitWidth`] unless `logits` has the model's
    ///   vocabulary width, and [`ConstraintError::NonFiniteLogit`] for a NaN
    ///   or infinite logit.
    /// * [`ConstraintError::NoAllowedToken`] when the grammar allows no token.
    /// * [`ConstraintError::OutputTooLarge`] when the token's bytes would pass
    ///   [`ConstraintLimits::max_output_bytes`].
    /// * [`ConstraintError::NonAcceptingStop`] when the grammar stops, or would
    ///   stop after this token, in a non-accepting state.
    pub fn select_argmax_with_logprobs(
        &mut self,
        logits: &[f32],
    ) -> Result<(ConstraintStep, TokenLogProbs), ConstraintError> {
        let (step, logprobs) = self.select(logits, true)?;
        Ok((step, logprobs.ok_or(ConstraintError::NoAllowedToken)?))
    }

    /// Samples and consumes one grammar-allowed token from caller-provided entropy.
    ///
    /// The caller owns entropy and commits any RNG advance only after this method
    /// succeeds. On error, the grammar's parse and decoded bytes are not
    /// advanced (see the error-state caveat under Errors).
    ///
    /// The returned receipt distinguishes raw temperature-one model probability,
    /// its current grammar-conditioned form, and the deployed temperature-conditioned
    /// categorical policy. Padded model rows contribute to the raw probability but
    /// are never grammar-selectable.
    ///
    /// # Errors
    ///
    /// The grammar state and decoded bytes advance only on success. The mask
    /// is computed on the live matcher, so an unexpected matcher failure
    /// there, or while consuming a selected token, can leave the matcher in
    /// its error state.
    ///
    /// * [`ConstraintError::TerminalSession`] after completion or
    ///   [`JsonConstraintSession::finish`].
    /// * [`ConstraintError::LogitWidth`] unless `logits` has the model's
    ///   vocabulary width, and [`ConstraintError::NonFiniteLogit`] for a NaN
    ///   or infinite logit.
    /// * [`ConstraintError::NoAllowedToken`] when the grammar allows no token.
    /// * [`ConstraintError::OutputTooLarge`] when the token's bytes would pass
    ///   [`ConstraintLimits::max_output_bytes`].
    /// * [`ConstraintError::NonAcceptingStop`] when the grammar stops, or would
    ///   stop after this token, in a non-accepting state.
    /// * [`ConstraintError::InvalidSamplingParameter`] for an invalid
    ///   `temperature` or `uniform`.
    pub fn select_categorical_with_logprobs(
        &mut self,
        logits: &[f32],
        temperature: f64,
        uniform: f64,
    ) -> Result<(ConstraintStep, SamplingTokenLogProbs), ConstraintError> {
        let mask = self.prepare_selection(logits)?;
        self.fill_legal_mask(&mask);
        let selected = sample_categorical(logits, &self.legal_mask, temperature, uniform)
            .map_err(map_sampling_error)?;
        let selected_index = usize::try_from(selected.token_id)
            .map_err(|_| ConstraintError::TokenizerCompilation)?;
        let probabilities = self.logprobs_for(logits, selected_index, selected.sampling_logprob);
        let step = self.commit_in_place(selected_index)?;
        Ok((step, probabilities))
    }

    fn select(
        &mut self,
        logits: &[f32],
        collect_logprobs: bool,
    ) -> Result<(ConstraintStep, Option<TokenLogProbs>), ConstraintError> {
        let mask = self.prepare_selection(logits)?;
        let selected_index = masked_argmax(&mask, logits, self.tokenizer_vocab_size)
            .ok_or(ConstraintError::NoAllowedToken)?;
        let logprobs = collect_logprobs.then(|| {
            self.fill_legal_mask(&mask);
            self.argmax_logprobs_for(logits, selected_index)
        });
        let step = self.commit_in_place(selected_index)?;
        Ok((step, logprobs))
    }

    /// Checks the logits and computes the grammar mask on the live matcher.
    ///
    /// Computing a mask does not advance llguidance's parse; the selected
    /// token is committed separately by [`Self::commit_in_place`].
    fn prepare_selection(&mut self, logits: &[f32]) -> Result<SimpleVob, ConstraintError> {
        if self.state != SessionState::Active {
            return Err(ConstraintError::TerminalSession);
        }
        if logits.len() != self.model_vocab_size {
            return Err(ConstraintError::LogitWidth);
        }
        if !all_finite(logits) {
            return Err(ConstraintError::NonFiniteLogit);
        }
        if self.matcher.is_stopped() {
            return Err(ConstraintError::NonAcceptingStop);
        }
        let mask = self
            .matcher
            .compute_mask()
            .map_err(|_| ConstraintError::NonAcceptingStop)?;
        mask.first_bit_set()
            .is_some_and(|index| index < self.tokenizer_vocab_size)
            .then_some(mask)
            .ok_or(ConstraintError::NoAllowedToken)
    }

    /// Expands the mask into [`Self::legal_mask`] for the receipt and sampling
    /// helpers, which take one flag per model row.
    fn fill_legal_mask(&mut self, mask: &SimpleVob) {
        self.legal_mask.fill(false);
        for (word_index, &word) in mask.as_slice().iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let index = word_index * 32 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if index >= self.tokenizer_vocab_size {
                    return;
                }
                self.legal_mask[index] = true;
            }
        }
    }

    fn argmax_logprobs_for(&self, logits: &[f32], selected_index: usize) -> TokenLogProbs {
        let (model_maximum, model_log_sum) = logsumexp_parts(logits);
        let (allowed_maximum, allowed_log_sum) = masked_logsumexp_parts(logits, &self.legal_mask);
        let selected = f64::from(logits[selected_index]);
        let model_logprob = (selected - model_maximum) - model_log_sum;
        TokenLogProbs {
            model_logprob,
            constrained_logprob: (selected - allowed_maximum) - allowed_log_sum,
            allowed_log_mass: (allowed_maximum - model_maximum) + (allowed_log_sum - model_log_sum),
        }
    }

    fn logprobs_for(
        &self,
        logits: &[f32],
        selected_index: usize,
        sampling_logprob: f64,
    ) -> SamplingTokenLogProbs {
        let argmax = self.argmax_logprobs_for(logits, selected_index);
        SamplingTokenLogProbs {
            model_logprob: argmax.model_logprob,
            constrained_logprob: argmax.constrained_logprob,
            allowed_log_mass: argmax.allowed_log_mass,
            sampling_logprob,
        }
    }

    /// Returns whether a consumed token completed an accepting grammar state.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        matches!(self.state, SessionState::Complete)
    }

    /// Marks an active request as truncated, preserving its decoded bytes.
    pub fn finish(&mut self) -> ConstraintFinish {
        if self.is_complete() {
            ConstraintFinish::GrammarComplete
        } else {
            self.state = SessionState::Truncated;
            ConstraintFinish::Truncated
        }
    }

    /// Returns the decoded bytes accumulated from consumed tokens.
    #[must_use]
    pub fn decoded_bytes(&self) -> &[u8] {
        &self.output
    }

    /// Parses and independently validates accepting grammar output.
    ///
    /// # Errors
    ///
    /// * [`ConstraintError::NotComplete`] before the grammar completes.
    /// * [`ConstraintError::InvalidJson`] when the decoded bytes do not parse.
    /// * [`ConstraintError::SchemaValidation`] when the independent validator
    ///   rejects the value the grammar accepted.
    pub fn validate_complete(&self) -> Result<Value, ConstraintError> {
        if !self.is_complete() {
            return Err(ConstraintError::NotComplete);
        }
        let value =
            serde_json::from_slice(&self.output).map_err(|_| ConstraintError::InvalidJson)?;
        self.validator
            .validate(&value)
            .map_err(|_| ConstraintError::SchemaValidation)?;
        Ok(value)
    }
}

/// Whether every logit is finite. A fold without early exit compiles to
/// vector compares; `any` stops at the first hit and stays scalar.
fn all_finite(logits: &[f32]) -> bool {
    logits
        .iter()
        .fold(true, |finite, logit| finite & logit.is_finite())
}

/// The highest logit among the mask's tokens below `width`, ties to the lower
/// ID, read one 32-token mask word at a time. A fully legal word (string
/// interiors allow most of the vocabulary) takes a vectorizable maximum of
/// its 32 logits first, and is searched only when it beats the best so far.
fn masked_argmax(mask: &SimpleVob, logits: &[f32], width: usize) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (word_index, &word) in mask.as_slice().iter().enumerate() {
        let base = word_index * 32;
        if base >= width {
            break;
        }
        if word == 0 {
            continue;
        }
        if word == u32::MAX && base + 32 <= width {
            let chunk = &logits[base..base + 32];
            let maximum = chunk
                .iter()
                .fold(f32::NEG_INFINITY, |left, &right| left.max(right));
            if best.is_none_or(|(_, current)| maximum > current) {
                // Exact equality on purpose: `maximum` is one of these logits,
                // and `==` treats -0.0 and 0.0 as tied, as the row scan's `>` does.
                #[allow(clippy::float_cmp)]
                let offset = chunk.iter().position(|&logit| logit == maximum)?;
                best = Some((base + offset, maximum));
            }
            continue;
        }
        let mut bits = word;
        while bits != 0 {
            let index = base + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            if index >= width {
                break;
            }
            if best.is_none_or(|(_, current)| logits[index] > current) {
                best = Some((index, logits[index]));
            }
        }
    }
    best.map(|(index, _)| index)
}

fn logsumexp_parts(values: &[f32]) -> (f64, f64) {
    let maximum = values
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let log_sum = values
        .iter()
        .map(|&value| (f64::from(value) - maximum).exp())
        .sum::<f64>()
        .ln();
    (maximum, log_sum)
}

fn masked_logsumexp_parts(values: &[f32], mask: &[bool]) -> (f64, f64) {
    let mut maximum = f64::NEG_INFINITY;
    for (&value, &allowed) in values.iter().zip(mask) {
        if allowed {
            maximum = maximum.max(f64::from(value));
        }
    }
    let mut sum = 0.0;
    for (&value, &allowed) in values.iter().zip(mask) {
        if allowed {
            sum += (f64::from(value) - maximum).exp();
        }
    }
    (maximum, sum.ln())
}

fn map_sampling_error(error: SamplingError) -> ConstraintError {
    match error {
        SamplingError::EmptySupport => ConstraintError::NoAllowedToken,
        SamplingError::NonFiniteLogit => ConstraintError::NonFiniteLogit,
        SamplingError::MaskLengthMismatch | SamplingError::VocabularyTooLarge => {
            ConstraintError::TokenizerCompilation
        }
        SamplingError::EmptyLogits
        | SamplingError::InvalidTemperature
        | SamplingError::InvalidUniform => ConstraintError::InvalidSamplingParameter,
    }
}

const MAX_TOKENIZER_VOCABULARY: usize = 1_000_000;

fn is_accepting_terminal(matcher: &mut Matcher) -> Result<bool, ConstraintError> {
    if !matches!(
        matcher.stop_reason(),
        StopReason::EndOfSentence | StopReason::NoExtension | StopReason::NoExtensionBias
    ) {
        return Ok(false);
    }
    matcher
        .is_accepting()
        .map_err(|_| ConstraintError::NonAcceptingStop)
}

fn declared_tokenizer_vocab_size(tokenizer_json: &Value) -> Result<usize, ConstraintError> {
    let mut largest_id = None;
    let mut ids = BTreeSet::new();
    let mut observe = |value: &Value| -> Result<(), ConstraintError> {
        let id = value
            .as_u64()
            .and_then(|id| usize::try_from(id).ok())
            .ok_or(ConstraintError::TokenizerCompilation)?;
        largest_id = Some(largest_id.map_or(id, |current: usize| current.max(id)));
        ids.insert(id);
        Ok(())
    };
    let vocab = tokenizer_json
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .ok_or(ConstraintError::TokenizerCompilation)?;
    for id in vocab.values() {
        observe(id)?;
    }
    if let Some(added_tokens) = tokenizer_json.get("added_tokens").and_then(Value::as_array) {
        for token in added_tokens {
            observe(
                token
                    .get("id")
                    .ok_or(ConstraintError::TokenizerCompilation)?,
            )?;
        }
    }
    let vocabulary_size = largest_id
        .and_then(|id| id.checked_add(1))
        .ok_or(ConstraintError::TokenizerCompilation)?;
    if vocabulary_size > MAX_TOKENIZER_VOCABULARY {
        return Err(ConstraintError::TokenizerVocabularyTooLarge);
    }
    if ids.len() != vocabulary_size {
        return Err(ConstraintError::TokenizerCompilation);
    }
    Ok(vocabulary_size)
}

/// The schema checks that need no tokenizer, in the order sessions apply them.
fn check_schema(schema: &Value, limits: ConstraintLimits) -> Result<(), ConstraintError> {
    check_serialized_bound(schema, limits.max_schema_bytes)
        .map_err(|()| ConstraintError::SchemaTooLarge)?;
    reject_non_local_references(schema)?;
    reject_non_202012_dialect(schema)
}

fn check_serialized_bound(value: &Value, limit: usize) -> Result<(), ()> {
    (serde_json::to_vec(value).map_err(|_| ())?.len() <= limit)
        .then_some(())
        .ok_or(())
}

fn reject_non_local_references(value: &Value) -> Result<(), ConstraintError> {
    reject_non_local_references_at_schema(value)
}

fn reject_non_local_references_at_schema(value: &Value) -> Result<(), ConstraintError> {
    if let Value::Object(object) = value {
        for (key, child) in object {
            if matches!(key.as_str(), "$ref" | "$dynamicRef" | "$recursiveRef")
                && child
                    .as_str()
                    .is_some_and(|reference| !reference.starts_with('#'))
            {
                return Err(ConstraintError::ExternalReference);
            }
            if !matches!(key.as_str(), "const" | "default" | "enum" | "examples") {
                reject_schema_children(key, child)?;
            }
        }
    }
    Ok(())
}

fn reject_schema_children(keyword: &str, value: &Value) -> Result<(), ConstraintError> {
    match keyword {
        "properties" | "patternProperties" | "dependentSchemas" | "$defs" | "definitions" => {
            if let Value::Object(children) = value {
                for child in children.values() {
                    reject_non_local_references_at_schema(child)?;
                }
            }
        }
        "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
            if let Value::Array(children) = value {
                for child in children {
                    reject_non_local_references_at_schema(child)?;
                }
            }
        }
        _ => reject_non_local_references_at_schema(value)?,
    }
    Ok(())
}

fn reject_non_202012_dialect(schema: &Value) -> Result<(), ConstraintError> {
    match schema.get("$schema").and_then(Value::as_str) {
        Some("https://json-schema.org/draft/2020-12/schema") | None => Ok(()),
        Some(_) => Err(ConstraintError::UnsupportedDialect),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ConstraintError, ConstraintFinish, ConstraintLimits, ConstraintStep,
        JsonConstraintCompiler, JsonConstraintSession, SessionState, is_accepting_terminal,
    };
    use serde_json::{Value, json};

    const EOS: u32 = 14;
    const TOKENIZER_VOCAB: usize = 15;
    const MODEL_VOCAB: usize = 18;

    fn limits() -> ConstraintLimits {
        ConstraintLimits {
            max_schema_bytes: 4_096,
            max_tokenizer_bytes: 4_096,
            max_output_bytes: 1_024,
        }
    }

    fn tokenizer() -> Value {
        json!({
            "decoder": {"type": "ByteLevel"},
            "added_tokens": [{"id": EOS, "content": "<eos>", "special": true}],
            "model": {"vocab": {
                "{": 0, "}": 1, "\"": 2, "o": 3, "k": 4, ":": 5,
                "t": 6, "r": 7, "u": 8, "e": 9, "f": 10, "a": 11,
                "l": 12, "s": 13
            }}
        })
    }

    fn schema() -> Value {
        json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {"ok": {"const": true}},
            "required": ["ok"],
            "additionalProperties": false,
        })
    }

    fn session() -> JsonConstraintSession {
        JsonConstraintSession::new(&tokenizer(), EOS, MODEL_VOCAB, schema(), limits())
            .expect("bounded test session")
    }

    fn boolean_session() -> JsonConstraintSession {
        boolean_session_with_output_limit(limits().max_output_bytes)
    }

    fn boolean_session_with_output_limit(max_output_bytes: usize) -> JsonConstraintSession {
        let mut configured_limits = limits();
        configured_limits.max_output_bytes = max_output_bytes;
        JsonConstraintSession::new(
            &tokenizer(),
            EOS,
            MODEL_VOCAB,
            json!({"type": "boolean"}),
            configured_limits,
        )
        .expect("bounded boolean session")
    }

    fn number_session() -> JsonConstraintSession {
        let tokenizer = json!({
            "decoder": {"type": "ByteLevel"},
            "added_tokens": [{"id": 15, "content": "<eos>", "special": true}],
            "model": {"vocab": {
                "{": 0, "}": 1, "\"": 2, "o": 3, "k": 4, ":": 5,
                "t": 6, "r": 7, "u": 8, "e": 9, "f": 10, "a": 11,
                "l": 12, "s": 13, "1": 14
            }}
        });
        JsonConstraintSession::new(
            &tokenizer,
            15,
            MODEL_VOCAB,
            json!({"type": "number"}),
            limits(),
        )
        .expect("bounded number session")
    }

    fn logits(token: u32) -> Vec<f32> {
        let mut logits = vec![-10.0; MODEL_VOCAB];
        logits[token as usize] = 1.0;
        logits[TOKENIZER_VOCAB] = 100.0;
        logits
    }

    fn token_id(byte: u8) -> u32 {
        match byte {
            b'{' => 0,
            b'}' => 1,
            b'"' => 2,
            b'o' => 3,
            b'k' => 4,
            b':' => 5,
            b't' => 6,
            b'r' => 7,
            b'u' => 8,
            b'e' => 9,
            b'f' => 10,
            b'a' => 11,
            b'l' => 12,
            b's' => 13,
            _ => panic!("test tokenizer has no ID for this byte"),
        }
    }

    /// The `{"ok": true}` schema with an `ok` token beside `o` and `k`. After
    /// `{"` the grammar forces the bytes `ok":true}`, and the mask allows only
    /// the forced tokenization while llguidance's `validate_tokens` accepts
    /// any token those bytes start with.
    fn forced_session() -> JsonConstraintSession {
        let tokenizer = json!({
            "decoder": {"type": "ByteLevel"},
            "added_tokens": [{"id": 15, "content": "<eos>", "special": true}],
            "model": {"vocab": {
                "{": 0, "}": 1, "\"": 2, "o": 3, "k": 4, ":": 5,
                "t": 6, "r": 7, "u": 8, "e": 9, "f": 10, "a": 11,
                "l": 12, "s": 13, "ok": 14
            }}
        });
        JsonConstraintSession::new(&tokenizer, 15, MODEL_VOCAB, schema(), limits())
            .expect("bounded forced-bytes session")
    }

    /// A greedy tokenizer that llguidance may treat as canonical, which
    /// turns on its forced-token mask.
    struct CanonicalEnv(llguidance::toktrie::TokTrie);

    impl llguidance::toktrie::TokenizerEnv for CanonicalEnv {
        fn tok_trie(&self) -> &llguidance::toktrie::TokTrie {
            &self.0
        }

        fn tokenize_bytes(&self, s: &[u8]) -> Vec<u32> {
            self.0.greedy_tokenize(s)
        }

        fn tokenize_is_canonical(&self) -> bool {
            true
        }
    }

    /// The forced-bytes session rebuilt over [`CanonicalEnv`].
    fn canonical_forced_session() -> JsonConstraintSession {
        let mut session = forced_session();
        let env: llguidance::toktrie::TokEnv =
            std::sync::Arc::new(CanonicalEnv(session.env.tok_trie().clone()));
        let factory = llguidance::ParserFactory::new_simple(&env).expect("factory");
        session.matcher = llguidance::Matcher::new(
            factory.create_parser(llguidance::api::TopLevelGrammar::from_json_schema(schema())),
        );
        session.env = env;
        session
    }

    /// Under a canonical tokenizer the mask forces `ok` after `{"` while
    /// validation also accepts `o`, so the first-legal walk must not run.
    #[test]
    fn canonical_forcing_falls_back_to_the_mask() {
        let mut probe = canonical_forced_session();
        let mut values = vec![0.0; MODEL_VOCAB];
        for favored in [0, 2] {
            values.fill(0.0);
            values[favored] = 5.0;
            probe.select_argmax_by_mask(&values).expect("prefix");
        }
        let mask = probe.matcher.compute_mask().expect("mask");
        assert!(mask.is_allowed(14) && !mask.is_allowed(3));
        assert_eq!(probe.matcher.validate_tokens(&[3]).ok(), Some(1));

        let (mut lazy, mut masked) = (canonical_forced_session(), canonical_forced_session());
        for favored in [0, 2, 3, 14, 2, 5] {
            values.fill(0.0);
            values[favored] = 5.0;
            values[3] = 4.0;
            let expected = masked.select_argmax_by_mask(&values);
            assert_eq!(lazy.select_argmax(&values), expected, "favored {favored}");
            assert_eq!(lazy.decoded_bytes(), masked.decoded_bytes());
        }
    }

    #[test]
    fn forced_bytes_pick_the_masks_token() {
        let (mut lazy, mut masked) = (forced_session(), forced_session());
        for favored in [0, 2, 14, 3, 4] {
            let mut values = vec![0.0; MODEL_VOCAB];
            values[favored] = 5.0;
            values[14] = 4.0;
            let expected = masked.select_argmax_by_mask(&values);
            assert_eq!(lazy.select_argmax(&values), expected, "favored {favored}");
            assert_eq!(lazy.decoded_bytes(), masked.decoded_bytes());
        }
    }

    /// Selection as it was before the mask path read words and stopped
    /// cloning, kept as an oracle independent of both selection paths: mask
    /// on a clone of the matcher, a scan over every tokenizer row (higher
    /// logit, then lower ID), and the commit on that clone. The session is
    /// never advanced, so repeated calls show what the old path would do
    /// after an error.
    fn reference_select(
        session: &JsonConstraintSession,
        logits: &[f32],
    ) -> Result<(ConstraintStep, Vec<u8>), ConstraintError> {
        if session.state != SessionState::Active {
            return Err(ConstraintError::TerminalSession);
        }
        if logits.len() != session.model_vocab_size {
            return Err(ConstraintError::LogitWidth);
        }
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(ConstraintError::NonFiniteLogit);
        }
        let mut matcher = session.matcher.deep_clone();
        if matcher.is_stopped() {
            return Err(ConstraintError::NonAcceptingStop);
        }
        let mask = matcher
            .compute_mask()
            .map_err(|_| ConstraintError::NonAcceptingStop)?;
        let mut selected: Option<usize> = None;
        for index in 0..session.tokenizer_vocab_size {
            let token = u32::try_from(index).expect("small vocabulary");
            if mask.is_allowed(token)
                && selected.is_none_or(|current| logits[index] > logits[current])
            {
                selected = Some(index);
            }
        }
        let token_id = u32::try_from(selected.ok_or(ConstraintError::NoAllowedToken)?)
            .expect("small vocabulary");
        let bytes = if token_id == session.eos_token_id {
            Vec::new()
        } else {
            session.env.tok_trie().decode(&[token_id])
        };
        if session.output.len() + bytes.len() > session.limits.max_output_bytes {
            return Err(ConstraintError::OutputTooLarge);
        }
        matcher
            .consume_token(token_id)
            .map_err(|_| ConstraintError::NonAcceptingStop)?;
        let step = if matcher.is_stopped() {
            if !is_accepting_terminal(&mut matcher)? {
                return Err(ConstraintError::NonAcceptingStop);
            }
            ConstraintStep::Complete { token_id }
        } else {
            ConstraintStep::Token { token_id }
        };
        let mut output = session.output.clone();
        output.extend_from_slice(&bytes);
        Ok((step, output))
    }

    /// A grammar that dead-ends: after the opening quote it needs `z`, which
    /// the tokenizer cannot produce, so no token is legal.
    fn dead_end_session() -> JsonConstraintSession {
        JsonConstraintSession::new(
            &tokenizer(),
            EOS,
            MODEL_VOCAB,
            json!({"const": "zz"}),
            limits(),
        )
        .expect("bounded dead-end session")
    }

    /// Once no token is legal, both selection paths report what the old
    /// cloned path reported, and keep reporting it on the next call: the
    /// in-place mask leaves the session where the clone left it.
    #[test]
    fn no_legal_token_errors_match_the_cloned_path_and_repeat() {
        let mut values = vec![0.0; MODEL_VOCAB];
        values[2] = 5.0;
        for by_mask in [false, true] {
            let mut session = dead_end_session();
            let first = session.select_argmax(&values).expect("opening quote");
            assert_eq!(first, ConstraintStep::Token { token_id: 2 });
            for _ in 0..3 {
                let expected = reference_select(&session, &values).map(|(step, _)| step);
                assert!(expected.is_err(), "the dead end must leave no legal token");
                let actual = if by_mask {
                    session.select_argmax_by_mask(&values)
                } else {
                    session.select_argmax(&values)
                };
                assert_eq!(actual, expected, "by_mask {by_mask}");
                assert_eq!(session.decoded_bytes(), b"\"");
            }
            let mut sampled = dead_end_session();
            sampled.select_argmax(&values).expect("opening quote");
            for _ in 0..2 {
                let expected = reference_select(&sampled, &values).map(|(step, _)| step);
                let actual = sampled
                    .select_categorical_with_logprobs(&values, 1.0, 0.5)
                    .map(|(step, _)| step);
                assert_eq!(actual, expected);
            }
        }
    }

    /// EOS ranked first before the boolean is complete: every path refuses it
    /// as the oracle does, picks the next legal token, and once `true`
    /// completes, a further call is a terminal error with the bytes kept.
    #[test]
    fn eos_before_completion_matches_the_oracle_on_every_path() {
        let eos = usize::try_from(EOS).expect("small");
        for path in 0..3 {
            let mut session = boolean_session();
            for favored in [6, 7, 8, 9, 9] {
                let mut values = vec![0.0; MODEL_VOCAB];
                values[eos] = 9.0;
                values[favored] = 5.0;
                let before = session.decoded_bytes().to_vec();
                let expected = reference_select(&session, &values);
                let actual = match path {
                    0 => session.select_argmax(&values),
                    1 => session.select_argmax_by_mask(&values),
                    _ => session
                        .select_argmax_with_logprobs(&values)
                        .map(|(step, _)| step),
                };
                assert_eq!(
                    actual,
                    expected
                        .as_ref()
                        .map(|(step, _)| *step)
                        .map_err(|error| *error)
                );
                let bytes = expected
                    .as_ref()
                    .map_or(before.as_slice(), |(_, bytes)| bytes.as_slice());
                assert_eq!(
                    session.decoded_bytes(),
                    bytes,
                    "path {path} favored {favored}"
                );
            }
            assert_eq!(session.decoded_bytes(), b"true");
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(48))]

        /// Both selection paths pick what the pre-change oracle picks, step
        /// for step, errors included and repeated after an error, over five
        /// grammars (one dead-ends), with ties and padded model rows.
        #[test]
        fn selection_matches_the_reference_oracle(
            kind in 0_usize..5,
            steps in proptest::collection::vec(
                proptest::collection::vec(-3_i8..=3, MODEL_VOCAB),
                1..40,
            ),
        ) {
            let make = || match kind {
                0 => session(),
                1 => boolean_session(),
                2 => number_session(),
                3 => forced_session(),
                _ => dead_end_session(),
            };
            let (mut lazy, mut masked) = (make(), make());
            for step in steps {
                let values: Vec<f32> = step.into_iter().map(f32::from).collect();
                let before = masked.decoded_bytes().to_vec();
                let expected = reference_select(&masked, &values);
                let expected_step = expected.as_ref().map(|(step, _)| *step).map_err(|error| *error);
                proptest::prop_assert_eq!(lazy.select_argmax(&values), expected_step);
                proptest::prop_assert_eq!(masked.select_argmax_by_mask(&values), expected_step);
                // An error, a rolled-back non-accepting stop included, leaves
                // the bytes where they were.
                let bytes = expected.as_ref().map_or(before.as_slice(), |(_, bytes)| bytes.as_slice());
                proptest::prop_assert_eq!(lazy.decoded_bytes(), bytes);
                proptest::prop_assert_eq!(masked.decoded_bytes(), bytes);
                if masked.is_complete() {
                    break;
                }
            }
        }
    }

    proptest::proptest! {
        /// The word-at-a-time argmax picks what a scan over every row picks:
        /// the highest legal logit below the width, ties to the lower ID,
        /// across sparse words, fully legal words and a ragged last word.
        #[test]
        fn masked_argmax_matches_a_row_scan(
            width in 1_usize..200,
            dense_words in proptest::collection::vec(proptest::bool::ANY, 7),
            sparse in proptest::collection::vec(proptest::bool::ANY, 224),
            values in proptest::collection::vec(-3_i8..=3, 224),
        ) {
            let mut mask = llguidance::toktrie::SimpleVob::alloc(224);
            for (index, &allowed) in sparse.iter().enumerate() {
                if allowed || dense_words[index / 32] {
                    mask.allow_token(u32::try_from(index).expect("small"));
                }
            }
            // Signed zeros tie under `>`; alternate them so a tie between
            // -0.0 and 0.0 must still go to the lower ID.
            let logits: Vec<f32> = values
                .into_iter()
                .enumerate()
                .map(|(index, value)| if value == 0 && index % 2 == 0 { -0.0 } else { f32::from(value) })
                .collect();
            let mut expected: Option<usize> = None;
            for index in 0..width {
                if mask.is_allowed(u32::try_from(index).expect("small"))
                    && expected.is_none_or(|current| logits[index] > logits[current])
                {
                    expected = Some(index);
                }
            }
            proptest::prop_assert_eq!(super::masked_argmax(&mask, &logits, width), expected);
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(48))]

        /// Greedy selection through first-legal candidates picks the same
        /// token as the full mask at every step, ties and padded rows
        /// included, until both complete or fail the same way.
        #[test]
        fn first_legal_argmax_matches_the_mask(
            kind in 0_usize..4,
            steps in proptest::collection::vec(
                proptest::collection::vec(-3_i8..=3, MODEL_VOCAB),
                1..40,
            ),
        ) {
            let make = || match kind {
                0 => session(),
                1 => boolean_session(),
                2 => number_session(),
                _ => forced_session(),
            };
            let (mut lazy, mut masked) = (make(), make());
            for step in steps {
                let values: Vec<f32> = step.into_iter().map(f32::from).collect();
                // The walk relies on validation allowing exactly the mask's
                // tokens, forced-bytes states included.
                let mut probe = lazy.matcher.deep_clone();
                if let Ok(mask) = probe.compute_mask() {
                    for token in 0..lazy.tokenizer_vocab_size {
                        let token = u32::try_from(token).expect("small vocabulary");
                        let valid = probe.validate_tokens(&[token]).ok() == Some(1);
                        proptest::prop_assert_eq!(mask.is_allowed(token), valid, "token {}", token);
                    }
                }
                let expected = masked.select_argmax_by_mask(&values);
                proptest::prop_assert_eq!(lazy.select_argmax(&values), expected);
                proptest::prop_assert_eq!(lazy.decoded_bytes(), masked.decoded_bytes());
                if expected.is_err() || masked.is_complete() {
                    break;
                }
            }
        }
    }

    #[test]
    fn padded_rows_and_eos_cannot_win_before_completion() {
        let mut session = session();
        let mut values = logits(EOS);
        values[0] = 2.0;
        assert_eq!(
            session.select_argmax(&values),
            Ok(ConstraintStep::Token { token_id: 0 })
        );
        assert_eq!(session.decoded_bytes(), b"{");
    }

    #[test]
    fn logprobs_include_padded_model_rows_but_not_allowed_mass() {
        let mut baseline = session();
        let mut measured = session();
        let values = logits(0);
        let baseline_step = baseline.select_argmax(&values).expect("baseline selection");
        let (measured_step, probabilities) = measured
            .select_argmax_with_logprobs(&values)
            .expect("measured selection");
        assert_eq!(measured_step, baseline_step);
        assert_eq!(measured_step, ConstraintStep::Token { token_id: 0 });
        assert!(
            (probabilities.constrained_logprob
                - (probabilities.model_logprob - probabilities.allowed_log_mass))
                .abs()
                < 1e-12
        );
        assert!(probabilities.model_logprob < -90.0);
        assert!(probabilities.allowed_log_mass < -90.0);
        assert!(probabilities.constrained_logprob.abs() < 1e-12);
    }

    #[test]
    fn logprobs_are_analytic_for_equal_logits_and_stable_for_extremes() {
        let mut equal = session();
        let values = vec![0.0; MODEL_VOCAB];
        let (step, probabilities) = equal
            .select_argmax_with_logprobs(&values)
            .expect("equal-logit selection");
        assert_eq!(step, ConstraintStep::Token { token_id: 0 });
        let expected = -f64::from(u32::try_from(MODEL_VOCAB).expect("test width fits u32")).ln();
        assert!((probabilities.model_logprob - expected).abs() < 1e-12);
        assert!((probabilities.allowed_log_mass - expected).abs() < 1e-12);
        assert!(probabilities.constrained_logprob.abs() < 1e-12);

        let mut translated = session();
        let values = vec![1.0e30; MODEL_VOCAB];
        let (step, probabilities) = translated
            .select_argmax_with_logprobs(&values)
            .expect("translated equal-logit selection");
        assert_eq!(step, ConstraintStep::Token { token_id: 0 });
        assert!((probabilities.model_logprob - expected).abs() < 1e-12);
        assert!((probabilities.allowed_log_mass - expected).abs() < 1e-12);
        assert!(probabilities.constrained_logprob.abs() < 1e-12);

        let mut extreme = session();
        let mut values = vec![-1.0e30; MODEL_VOCAB];
        values[0] = -1.0e20;
        values[TOKENIZER_VOCAB] = 1.0e30;
        let (step, probabilities) = extreme
            .select_argmax_with_logprobs(&values)
            .expect("extreme finite selection");
        assert_eq!(step, ConstraintStep::Token { token_id: 0 });
        assert!(probabilities.model_logprob.is_finite());
        assert!(probabilities.constrained_logprob.is_finite());
        assert!(probabilities.allowed_log_mass.is_finite());
    }

    #[test]
    fn validates_complete_json_after_replaying_masked_argmax_tokens() {
        let mut session = session();
        for &token in b"{\"ok\":true}" {
            let token_id = token_id(token);
            let step = session
                .select_argmax(&logits(token_id))
                .expect("allowed token");
            if token == b'}' {
                assert_eq!(step, ConstraintStep::Complete { token_id });
            } else {
                assert_eq!(step, ConstraintStep::Token { token_id });
            }
        }
        assert!(session.is_complete());
        assert_eq!(session.decoded_bytes(), b"{\"ok\":true}");
        assert_eq!(
            session.validate_complete().expect("independent validation")["ok"],
            true
        );
    }

    #[test]
    fn rejects_bad_width_and_non_finite_logits_before_selecting() {
        let mut session = session();
        assert_eq!(
            session.select_argmax(&[0.0; MODEL_VOCAB - 1]),
            Err(ConstraintError::LogitWidth)
        );
        let mut values = vec![0.0; MODEL_VOCAB];
        values[MODEL_VOCAB - 1] = f32::NAN;
        assert_eq!(
            session.select_argmax(&values),
            Err(ConstraintError::NonFiniteLogit)
        );
    }

    #[test]
    fn checkpoint_restores_grammar_and_output_for_a_branch() {
        let mut session = session();
        session
            .select_argmax(&logits(token_id(b'{')))
            .expect("opening brace is allowed");
        let checkpoint = session.checkpoint();

        session
            .select_argmax(&logits(token_id(b'"')))
            .expect("quote is allowed after opening brace");
        assert_eq!(session.decoded_bytes(), b"{\"");

        session.restore(checkpoint).expect("same-session restore");
        assert_eq!(session.decoded_bytes(), b"{");
        session
            .select_argmax(&logits(token_id(b'"')))
            .expect("restored branch remains usable");
        assert_eq!(session.decoded_bytes(), b"{\"");
    }

    /// The vocabulary of [`number_session`], which covers all three test schemas.
    fn wide_tokenizer() -> Value {
        json!({
            "decoder": {"type": "ByteLevel"},
            "added_tokens": [{"id": 15, "content": "<eos>", "special": true}],
            "model": {"vocab": {
                "{": 0, "}": 1, "\"": 2, "o": 3, "k": 4, ":": 5,
                "t": 6, "r": 7, "u": 8, "e": 9, "f": 10, "a": 11,
                "l": 12, "s": 13, "1": 14
            }}
        })
    }

    fn mask_words(session: &JsonConstraintSession) -> Option<Vec<u32>> {
        let mut probe = session.matcher.deep_clone();
        probe
            .compute_mask()
            .ok()
            .map(|mask| mask.as_slice().to_vec())
    }

    /// Consecutive requests on one shared compiler see the masks, tokens and
    /// bytes a freshly built session sees, step for step, including after an
    /// earlier session from the same compiler has run to completion.
    #[test]
    fn shared_compiler_sessions_match_fresh_sessions() {
        let compiler = JsonConstraintCompiler::new(&wide_tokenizer(), 15, MODEL_VOCAB, limits())
            .expect("bounded compiler");
        let schemas = [
            schema(),
            json!({"type": "boolean"}),
            json!({"type": "number"}),
        ];
        let mut seed = 0x2545_f491_u32;
        for round in 0..4 {
            for schema in &schemas {
                let mut shared = compiler.session(schema.clone(), limits()).expect("shared");
                let mut fresh = JsonConstraintSession::new(
                    &wide_tokenizer(),
                    15,
                    MODEL_VOCAB,
                    schema.clone(),
                    limits(),
                )
                .expect("fresh");
                for _ in 0..24 {
                    assert_eq!(mask_words(&shared), mask_words(&fresh), "round {round}");
                    let values: Vec<f32> = (0..MODEL_VOCAB)
                        .map(|_| {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            f32::from(i8::try_from(seed % 7).expect("small") - 3)
                        })
                        .collect();
                    let expected = fresh.select_argmax(&values);
                    assert_eq!(shared.select_argmax(&values), expected, "round {round}");
                    assert_eq!(shared.decoded_bytes(), fresh.decoded_bytes());
                    if expected.is_err() || fresh.is_complete() {
                        break;
                    }
                }
            }
        }
    }

    #[test]
    fn checkpoint_cannot_cross_sessions_of_one_compiler() {
        let compiler = JsonConstraintCompiler::new(&tokenizer(), EOS, MODEL_VOCAB, limits())
            .expect("bounded compiler");
        let source = compiler.session(schema(), limits()).expect("source");
        let mut destination = compiler.session(schema(), limits()).expect("destination");
        assert_eq!(
            destination.restore(source.checkpoint()),
            Err(ConstraintError::CheckpointMismatch)
        );
        let mut same = compiler.session(schema(), limits()).expect("same");
        assert_eq!(same.restore(same.checkpoint()), Ok(()));
    }

    #[test]
    fn compiler_can_be_shared_across_threads() {
        fn shareable<T: Send + Sync>() {}
        shareable::<JsonConstraintCompiler>();
    }

    #[test]
    fn checkpoint_cannot_cross_constraint_sessions() {
        let source = session();
        let mut destination = session();

        assert_eq!(
            destination.restore(source.checkpoint()),
            Err(ConstraintError::CheckpointMismatch)
        );
    }

    #[test]
    fn truncation_never_validates_as_completion() {
        let mut session = session();
        session
            .select_argmax(&logits(token_id(b'{')))
            .expect("first token");
        assert_eq!(session.finish(), ConstraintFinish::Truncated);
        assert!(!session.is_complete());
        assert_eq!(
            session.validate_complete(),
            Err(ConstraintError::NotComplete)
        );
        assert_eq!(
            session.select_argmax(&logits(token_id(b'\"'))),
            Err(ConstraintError::TerminalSession)
        );
    }

    #[test]
    fn rejects_external_references_before_validator_construction() {
        let external = json!({"$ref": "https://example.invalid/schema.json"});
        assert!(matches!(
            JsonConstraintSession::new(&tokenizer(), EOS, MODEL_VOCAB, external, limits()),
            Err(ConstraintError::ExternalReference)
        ));
    }

    #[test]
    fn rejects_sparse_token_ids_before_constructing_the_trie() {
        let mut sparse = tokenizer();
        sparse["added_tokens"][0]["id"] = json!(1_000_000);
        assert!(matches!(
            JsonConstraintSession::new(&sparse, EOS, 2_000_000, schema(), limits()),
            Err(ConstraintError::TokenizerVocabularyTooLarge)
        ));
    }

    #[test]
    fn does_not_treat_json_literals_as_schema_references() {
        let literal = json!({
            "const": {"$ref": "https://example.invalid/not-a-schema-reference"}
        });
        assert!(
            JsonConstraintSession::new(&tokenizer(), EOS, MODEL_VOCAB, literal, limits()).is_ok()
        );
    }

    #[test]
    fn categorical_receipt_excludes_padded_and_eos_rows_from_temperature_policy() {
        let mut session = boolean_session();
        let mut values = vec![-100.0; MODEL_VOCAB];
        values[token_id(b't') as usize] = 0.0;
        values[token_id(b'f') as usize] = 1.0;
        values[EOS as usize] = 3.0;
        values[TOKENIZER_VOCAB] = 4.0;

        let (step, probabilities) = session
            .select_categorical_with_logprobs(&values, 0.5, 0.0)
            .expect("first categorical boolean token");
        assert_eq!(
            step,
            ConstraintStep::Token {
                token_id: token_id(b't')
            }
        );
        assert_eq!(session.decoded_bytes(), b"t");
        let allowed_log_sum = (1.0_f64 + (-1.0_f64).exp()).ln();
        let model_log_sum = (1.0_f64
            + (-1.0_f64).exp()
            + (-3.0_f64).exp()
            + (-4.0_f64).exp()
            + 14.0 * (-104.0_f64).exp())
        .ln();
        assert!((probabilities.model_logprob - (-4.0 - model_log_sum)).abs() < 1e-12);
        assert!((probabilities.constrained_logprob - (-1.0 - allowed_log_sum)).abs() < 1e-12);
        assert!(
            (probabilities.allowed_log_mass - (-3.0 + allowed_log_sum - model_log_sum)).abs()
                < 1e-12
        );
        assert!((probabilities.sampling_logprob + (1.0_f64 + 2.0_f64.exp()).ln()).abs() < 1e-12);
    }

    #[test]
    fn categorical_errors_do_not_advance_the_grammar_session() {
        let mut after_failure = boolean_session();
        let mut baseline = boolean_session();
        let mut invalid = vec![-100.0; MODEL_VOCAB];
        invalid[token_id(b't') as usize] = 1.0;
        assert_eq!(
            after_failure.select_categorical_with_logprobs(&invalid, 1.0, 1.0),
            Err(ConstraintError::InvalidSamplingParameter)
        );

        let after = after_failure
            .select_categorical_with_logprobs(&invalid, 1.0, 0.0)
            .expect("selection after rejected logits");
        let expected = baseline
            .select_categorical_with_logprobs(&invalid, 1.0, 0.0)
            .expect("fresh selection");
        assert_eq!(after, expected);
        assert_eq!(after_failure.decoded_bytes(), baseline.decoded_bytes());
    }

    #[test]
    fn output_bound_failure_does_not_commit_the_candidate_grammar_state() {
        let mut session = boolean_session_with_output_limit(3);
        for byte in b"tru" {
            let mut values = vec![-100.0; MODEL_VOCAB];
            values[token_id(*byte) as usize] = 1.0;
            session
                .select_categorical_with_logprobs(&values, 1.0, 0.0)
                .expect("bounded boolean prefix token");
        }
        let mut final_token = vec![-100.0; MODEL_VOCAB];
        final_token[token_id(b'e') as usize] = 1.0;
        assert_eq!(
            session.select_categorical_with_logprobs(&final_token, 1.0, 0.0),
            Err(ConstraintError::OutputTooLarge)
        );
        assert_eq!(session.decoded_bytes(), b"tru");
        assert!(!session.is_complete());
        assert_eq!(
            session.select_categorical_with_logprobs(&final_token, 1.0, 0.0),
            Err(ConstraintError::OutputTooLarge)
        );
        assert_eq!(session.decoded_bytes(), b"tru");
    }

    #[test]
    fn categorical_boolean_completion_returns_the_final_token_receipt() {
        let mut session = boolean_session();
        for byte in b"true" {
            let mut values = vec![-100.0; MODEL_VOCAB];
            values[token_id(*byte) as usize] = 1.0;
            let (step, probabilities) = session
                .select_categorical_with_logprobs(&values, 0.7, 0.0)
                .expect("categorical boolean token");
            if *byte == b'e' {
                assert_eq!(
                    step,
                    ConstraintStep::Complete {
                        token_id: token_id(b'e')
                    }
                );
                assert!(probabilities.sampling_logprob.abs() < 1e-12);
            } else {
                assert_eq!(
                    step,
                    ConstraintStep::Token {
                        token_id: token_id(*byte)
                    }
                );
            }
        }
        assert!(session.is_complete());
        assert_eq!(session.decoded_bytes(), b"true");
    }

    #[test]
    fn categorical_eos_completes_an_accepting_number_without_appending_special_bytes() {
        let mut session = number_session();
        let mut one = vec![-100.0; MODEL_VOCAB];
        one[14] = 1.0;
        assert_eq!(
            session
                .select_categorical_with_logprobs(&one, 1.0, 0.0)
                .expect("categorical number prefix")
                .0,
            ConstraintStep::Token { token_id: 14 }
        );
        assert_eq!(session.decoded_bytes(), b"1");

        let mut eos = vec![-100.0; MODEL_VOCAB];
        eos[15] = 1.0;
        let (step, probabilities) = session
            .select_categorical_with_logprobs(&eos, 1.0, 0.5)
            .expect("categorical accepting EOS");
        assert_eq!(step, ConstraintStep::Complete { token_id: 15 });
        assert!(probabilities.sampling_logprob.is_finite());
        assert!(probabilities.sampling_logprob > -1e-12);
        assert!(session.is_complete());
        assert_eq!(session.decoded_bytes(), b"1");
        assert_eq!(session.validate_complete(), Ok(json!(1)));
    }
}
