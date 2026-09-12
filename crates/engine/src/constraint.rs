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
    toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie},
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

/// One JSON-Schema grammar session bound to a tokenizer and model logit width.
pub struct JsonConstraintSession {
    env: TokEnv,
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
        check_serialized_bound(&schema, limits.max_schema_bytes)
            .map_err(|()| ConstraintError::SchemaTooLarge)?;
        reject_non_local_references(&schema)?;
        reject_non_202012_dialect(&schema)?;

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
        let validator = jsonschema::options()
            .with_draft(Draft::Draft202012)
            .offline()
            .build(&schema)
            .map_err(|_| ConstraintError::SchemaCompilation)?;
        let mut matcher =
            Matcher::new(factory.create_parser(TopLevelGrammar::from_json_schema(schema)));
        if matcher.is_error() || !matcher.grammar_warnings().is_empty() {
            return Err(ConstraintError::GrammarWarning);
        }

        Ok(Self {
            env,
            matcher,
            validator,
            model_vocab_size,
            tokenizer_vocab_size,
            eos_token_id,
            output: Vec::new(),
            legal_mask: vec![false; model_vocab_size],
            limits,
            state: SessionState::Active,
        })
    }

    /// Selects and consumes the finite maximum grammar-allowed tokenizer logit.
    pub fn select_argmax(&mut self, logits: &[f32]) -> Result<ConstraintStep, ConstraintError> {
        self.select(logits, false).map(|(step, _)| step)
    }

    /// Selects and consumes the constrained maximum, returning pre-consumption log probabilities.
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
    /// succeeds. On error, the grammar's semantic state and decoded bytes are not
    /// advanced.
    ///
    /// The returned receipt distinguishes raw temperature-one model probability,
    /// its current grammar-conditioned form, and the deployed temperature-conditioned
    /// categorical policy. Padded model rows contribute to the raw probability but
    /// are never grammar-selectable.
    pub fn select_categorical_with_logprobs(
        &mut self,
        logits: &[f32],
        temperature: f64,
        uniform: f64,
    ) -> Result<(ConstraintStep, SamplingTokenLogProbs), ConstraintError> {
        let matcher = self.prepare_selection(logits)?;
        let selected = sample_categorical(logits, &self.legal_mask, temperature, uniform)
            .map_err(map_sampling_error)?;
        let selected_index = usize::try_from(selected.token_id)
            .map_err(|_| ConstraintError::TokenizerCompilation)?;
        let probabilities = self.logprobs_for(logits, selected_index, selected.sampling_logprob);
        let step = self.commit_selection(matcher, selected_index)?;
        Ok((step, probabilities))
    }

    fn select(
        &mut self,
        logits: &[f32],
        collect_logprobs: bool,
    ) -> Result<(ConstraintStep, Option<TokenLogProbs>), ConstraintError> {
        let matcher = self.prepare_selection(logits)?;
        let mut selected = None;
        let vocabulary_width = u32::try_from(self.tokenizer_vocab_size)
            .map_err(|_| ConstraintError::TokenizerCompilation)?;
        for token_id in 0..vocabulary_width {
            let index =
                usize::try_from(token_id).map_err(|_| ConstraintError::TokenizerCompilation)?;
            if self.legal_mask[index]
                && selected.is_none_or(|current| logits[index] > logits[current])
            {
                selected = Some(index);
            }
        }
        let selected_index = selected.ok_or(ConstraintError::NoAllowedToken)?;
        let logprobs = collect_logprobs.then(|| self.argmax_logprobs_for(logits, selected_index));
        let step = self.commit_selection(matcher, selected_index)?;
        Ok((step, logprobs))
    }

    fn prepare_selection(&mut self, logits: &[f32]) -> Result<Matcher, ConstraintError> {
        if self.state != SessionState::Active {
            return Err(ConstraintError::TerminalSession);
        }
        if logits.len() != self.model_vocab_size {
            return Err(ConstraintError::LogitWidth);
        }
        if logits.iter().any(|logit| !logit.is_finite()) {
            return Err(ConstraintError::NonFiniteLogit);
        }
        let mut matcher = self.matcher.deep_clone();
        if matcher.is_stopped() {
            return Err(ConstraintError::NonAcceptingStop);
        }
        let mask = matcher
            .compute_mask()
            .map_err(|_| ConstraintError::NonAcceptingStop)?;
        self.legal_mask.fill(false);
        let vocabulary_width = u32::try_from(self.tokenizer_vocab_size)
            .map_err(|_| ConstraintError::TokenizerCompilation)?;
        for token_id in 0..vocabulary_width {
            if mask.is_allowed(token_id) {
                let index =
                    usize::try_from(token_id).map_err(|_| ConstraintError::TokenizerCompilation)?;
                self.legal_mask[index] = true;
            }
        }
        self.legal_mask
            .iter()
            .any(|allowed| *allowed)
            .then_some(matcher)
            .ok_or(ConstraintError::NoAllowedToken)
    }

    fn commit_selection(
        &mut self,
        mut matcher: Matcher,
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
        matcher
            .consume_token(token_id)
            .map_err(|_| ConstraintError::NonAcceptingStop)?;
        let complete = if matcher.is_stopped() {
            if !is_accepting_terminal(&mut matcher)? {
                return Err(ConstraintError::NonAcceptingStop);
            }
            true
        } else {
            false
        };
        self.matcher = matcher;
        self.output.extend_from_slice(&token_bytes);
        if complete {
            self.state = SessionState::Complete;
            Ok(ConstraintStep::Complete { token_id })
        } else {
            Ok(ConstraintStep::Token { token_id })
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
        ConstraintError, ConstraintFinish, ConstraintLimits, ConstraintStep, JsonConstraintSession,
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
