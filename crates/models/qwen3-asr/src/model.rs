//! A loaded Qwen3-ASR checkpoint and greedy transcription.

#![allow(
    deprecated,
    reason = "mlx-rs 0.32 deprecates the *_device ops; the qwen crate defers the same migration"
)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use audio::{LogMel, LogMelExtractor, LogMelSpec, MonoAudio};
use media::{EmbeddedPrompt, EmbeddedSpan, SpanError, SpanKind};
use mlx_rs::{Array, Dtype, StreamOrDevice};
use qwen::forward::{Qwen3ForwardConfig, Qwen3ForwardError, Qwen3ForwardExecutor};
use sha2::{Digest, Sha256};

use crate::config::Qwen3AsrConfig;
use crate::encoder::{AudioEncoder, EncoderError};
use crate::output::{Transcript, parse_output};
use crate::tokenizer::{AsrTokenizer, AsrTokenizerError, STOP_TOKENS};

const DECODER_PREFIX: &str = "thinker.model.";
/// The output projection; the decoder ties it to the token embedding.
const TIED_HEAD: &str = "thinker.lm_head.weight";
/// The reference refuses longer single inputs and splits them first.
pub const MAX_AUDIO_SECONDS: f64 = 1_200.0;

/// Options for one transcription.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscribeOptions {
    /// System-message context, such as names or terms to expect.
    pub context: String,
    /// Force this language (a name from the checkpoint's list, such as
    /// `English`), or detect it.
    pub language: Option<String>,
    /// Most tokens to generate.
    pub max_new_tokens: usize,
    /// Largest logical K/V allocation the request may plan.
    pub max_kv_bytes: u64,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            context: String::new(),
            language: None,
            max_new_tokens: 512,
            max_kv_bytes: 4 << 30,
        }
    }
}

/// What one transcription produced and how long its stages took.
#[derive(Clone, Debug, PartialEq)]
pub struct TranscribeResult {
    /// The parsed language and text.
    pub transcript: Transcript,
    /// Decoded assistant text before parsing.
    pub raw: String,
    /// Generated token ids, stop token excluded.
    pub generated: Vec<i32>,
    /// Whether generation stopped on an end token rather than the limit.
    pub stopped: bool,
    /// Audio duration in seconds.
    pub audio_seconds: f64,
    /// Prompt length, audio placeholders included.
    pub prompt_tokens: usize,
    /// Log-mel extraction time.
    pub features: Duration,
    /// Encoder time (evaluated before the decoder starts).
    pub encode: Duration,
    /// Decoder prefill time, through the first token's logits.
    pub prefill: Duration,
    /// Remaining decode time.
    pub decode: Duration,
}

/// A Qwen3-ASR checkpoint resident on the GPU.
pub struct Qwen3Asr {
    config: Qwen3AsrConfig,
    tokenizer: AsrTokenizer,
    features: LogMelExtractor,
    encoder: AudioEncoder,
    decoder_config: Qwen3ForwardConfig,
    decoder: HashMap<String, Array>,
}

impl std::fmt::Debug for Qwen3Asr {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Qwen3Asr")
            .field("encoder", &self.encoder)
            .finish_non_exhaustive()
    }
}

impl Qwen3Asr {
    /// Loads a dense checkpoint directory (`config.json`, the tokenizer files and
    /// `*.safetensors`).
    ///
    /// Declared quantized/packed checkpoints are rejected during configuration
    /// parsing, before tokenizer or tensor files are read. Packed ASR encoder
    /// and decoder adaptation is not implemented.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError`] for an unsupported configuration, a missing or
    /// misshapen tensor, or an unexpected tensor name.
    #[tracing::instrument(name = "qwen3_asr.load", level = "info", skip_all, fields(model_dir = %model_dir.display()))]
    pub fn load(model_dir: &Path) -> Result<Self, AsrError> {
        let config_json = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|error| AsrError::Read(model_dir.join("config.json"), error.to_string()))?;
        let config = Qwen3AsrConfig::parse(&config_json)?;
        let mut decoder_config = Qwen3ForwardConfig::parse(&config.text_config_json())?;
        let tokenizer = AsrTokenizer::load(model_dir, config.tokens())?;

        let mut tensors = HashMap::new();
        let mut shards: Vec<PathBuf> = std::fs::read_dir(model_dir)
            .map_err(|error| AsrError::Read(model_dir.to_path_buf(), error.to_string()))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "safetensors")
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(AsrError::Read(
                model_dir.to_path_buf(),
                "no .safetensors files".to_owned(),
            ));
        }
        for shard in shards {
            // MLX safetensors I/O runs on the CPU stream.
            tensors.extend(
                Array::load_safetensors_device(&shard, StreamOrDevice::cpu())
                    .map_err(|error| AsrError::Read(shard.clone(), error.to_string()))?,
            );
        }
        let encoder = AudioEncoder::from_tensors(config.encoder().clone(), &mut tensors)?;
        let mut decoder = HashMap::new();
        let mut head = None;
        for (name, tensor) in tensors {
            if let Some(rest) = name.strip_prefix(DECODER_PREFIX) {
                decoder.insert(format!("model.{rest}"), tensor);
            } else if name == TIED_HEAD {
                head = Some(tensor);
            } else {
                return Err(AsrError::UnexpectedTensor(name));
            }
        }
        // The released checkpoints store `lm_head` as a byte-identical copy
        // of the embedding, and the config ties them. Drop the copy only when
        // it really is one; otherwise keep it and project through it.
        if let Some(head) = head {
            let embedding = decoder.get("model.embed_tokens.weight").ok_or_else(|| {
                AsrError::UnexpectedTensor("model.embed_tokens.weight missing".into())
            })?;
            let identical = head.shape() == embedding.shape()
                && head.dtype() == embedding.dtype()
                && head.eq_values(embedding)?;
            if !identical || !decoder_config.tied_output_embedding() {
                tracing::warn!(
                    identical,
                    "keeping thinker.lm_head.weight as an untied output projection"
                );
                let mut text: serde_json::Value = serde_json::from_str(&config.text_config_json())
                    .map_err(crate::config::Qwen3AsrConfigError::Json)?;
                text["tie_word_embeddings"] = serde_json::Value::Bool(false);
                decoder_config = Qwen3ForwardConfig::parse(&text.to_string())?;
                decoder.insert("lm_head.weight".to_owned(), head);
            }
        } else if !decoder_config.tied_output_embedding() {
            return Err(AsrError::UnexpectedTensor(format!("{TIED_HEAD} missing")));
        }
        let embedding = decoder.get("model.embed_tokens.weight").ok_or_else(|| {
            AsrError::UnexpectedTensor("model.embed_tokens.weight missing".into())
        })?;
        if embedding.shape().len() != 2
            || usize::try_from(embedding.shape()[1]).ok() != Some(config.encoder().output_dim)
        {
            return Err(AsrError::UnexpectedTensor(format!(
                "model.embed_tokens.weight has shape {:?}",
                embedding.shape()
            )));
        }
        Ok(Self {
            config,
            tokenizer,
            features: LogMelExtractor::new(LogMelSpec::QWEN3_ASR),
            encoder,
            decoder_config,
            decoder,
        })
    }

    /// Converts encoder and decoder tensors to `dtype` (f32 for parity runs).
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::Mlx`] if MLX fails.
    pub fn convert(&mut self, dtype: Dtype) -> Result<(), AsrError> {
        self.encoder.convert(dtype)?;
        for tensor in self.decoder.values_mut() {
            let converted = tensor.as_dtype_device(dtype, StreamOrDevice::gpu())?;
            converted.eval()?;
            *tensor = converted;
        }
        Ok(())
    }

    /// The parsed configuration.
    #[must_use]
    pub const fn config(&self) -> &Qwen3AsrConfig {
        &self.config
    }

    /// The tokenizer.
    #[must_use]
    pub const fn tokenizer(&self) -> &AsrTokenizer {
        &self.tokenizer
    }

    /// The audio encoder.
    #[must_use]
    pub const fn encoder(&self) -> &AudioEncoder {
        &self.encoder
    }

    /// Log-mel features for 16 kHz mono audio.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError::Features`] for audio at another rate or too short.
    pub fn features(&self, audio: &MonoAudio) -> Result<LogMel, AsrError> {
        Ok(self.features.extract(audio)?)
    }

    /// Logits at the end of the prompt for `mel`, after the audio splice:
    /// what the first generated token is chosen from.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError`] if a stage fails.
    pub fn first_logits(
        &self,
        mel: &LogMel,
        options: &TranscribeOptions,
    ) -> Result<Vec<f32>, AsrError> {
        let digest = feature_digest(mel);
        let rows = self.encoder.encode(mel)?;
        let frames = usize::try_from(rows.shape()[0]).map_err(|_| AsrError::Shape)?;
        let prompt =
            self.tokenizer
                .prompt(&options.context, options.language.as_deref(), frames)?;
        let mut executor = Qwen3ForwardExecutor::resident(
            &self.decoder_config,
            &self.decoder,
            prompt.ids().len(),
            options.max_kv_bytes,
        )?;
        let embedded = EmbeddedPrompt::new(
            prompt.ids(),
            vec![EmbeddedSpan::new(
                prompt.audio_start(),
                frames,
                SpanKind::Audio,
                digest,
                rows,
            )],
        )?;
        Ok(executor.prefill_embedded_last_logits(&embedded)?)
    }

    /// Transcribes one clip with greedy decoding.
    ///
    /// # Errors
    ///
    /// Returns [`AsrError`] for unusable audio, an unsupported forced
    /// language, a context too large for the K/V budget, or a failed stage.
    #[tracing::instrument(name = "qwen3_asr.transcribe", level = "info", skip_all, fields(seconds = audio.duration_seconds()))]
    pub fn transcribe(
        &self,
        audio: &MonoAudio,
        options: &TranscribeOptions,
    ) -> Result<TranscribeResult, AsrError> {
        self.transcribe_with_control(audio, options, &mut || Ok(()))
    }

    /// Transcribes with cooperative checks around host/native stages and
    /// each decode step. The callback may return [`AsrError::Cancelled`] or
    /// [`AsrError::DeadlineExceeded`]. An in-flight native operation cannot
    /// be interrupted; outstanding decode work is drained before returning.
    ///
    /// # Errors
    ///
    /// As [`Self::transcribe`], or the error returned by `check`.
    pub fn transcribe_with_control(
        &self,
        audio: &MonoAudio,
        options: &TranscribeOptions,
        check: &mut dyn FnMut() -> Result<(), AsrError>,
    ) -> Result<TranscribeResult, AsrError> {
        check()?;
        if audio.duration_seconds() > MAX_AUDIO_SECONDS {
            return Err(AsrError::TooLong(audio.duration_seconds()));
        }
        if let Some(language) = &options.language {
            if !self
                .config
                .languages()
                .iter()
                .any(|known| known == language)
            {
                return Err(AsrError::Language(language.clone()));
            }
        }
        let started = Instant::now();
        let mel = self.features.extract(audio)?;
        let features = started.elapsed();
        check()?;
        let digest = feature_digest(&mel);
        check()?;

        let started = Instant::now();
        let rows = self.encoder.encode(&mel)?;
        rows.eval()?;
        let encode = started.elapsed();
        check()?;
        let frames = usize::try_from(rows.shape()[0]).map_err(|_| AsrError::Shape)?;

        let prompt =
            self.tokenizer
                .prompt(&options.context, options.language.as_deref(), frames)?;
        let context = prompt
            .ids()
            .len()
            .checked_add(options.max_new_tokens)
            .ok_or(AsrError::Shape)?;
        let mut executor = Qwen3ForwardExecutor::resident(
            &self.decoder_config,
            &self.decoder,
            context,
            options.max_kv_bytes,
        )?;
        let embedded = EmbeddedPrompt::new(
            prompt.ids(),
            vec![EmbeddedSpan::new(
                prompt.audio_start(),
                frames,
                SpanKind::Audio,
                digest,
                rows,
            )],
        )?;
        check()?;
        let started = Instant::now();
        let first = argmax(&executor.prefill_embedded_last_logits(&embedded)?)?;
        let prefill = started.elapsed();
        check()?;

        let started = Instant::now();
        let (generated, stopped) =
            decode_controlled(&mut executor, first, options.max_new_tokens, check)?;
        let decode = started.elapsed();

        check()?;
        let raw = self.tokenizer.decode(&generated)?;
        check()?;
        Ok(TranscribeResult {
            transcript: parse_output(&raw, options.language.as_deref()),
            raw,
            generated,
            stopped,
            audio_seconds: audio.duration_seconds(),
            prompt_tokens: prompt.ids().len(),
            features,
            encode,
            prefill,
            decode,
        })
    }
}

/// Owns the queued native step until it is read or drained on any exit.
struct PendingDecode(Option<qwen::forward::Qwen3TokenPicks>);

impl Drop for PendingDecode {
    fn drop(&mut self) {
        if let Some(pending) = &self.0 {
            // Preserve the original error/timeout while still completing
            // the native evaluation before cache and admission cleanup.
            let _ = pending.wait_one();
        }
    }
}

fn decode_controlled<S: std::hash::BuildHasher>(
    executor: &mut Qwen3ForwardExecutor<'_, S>,
    first: i32,
    maximum: usize,
    check: &mut dyn FnMut() -> Result<(), AsrError>,
) -> Result<(Vec<i32>, bool), AsrError> {
    let mut generated = Vec::new();
    let mut stopped = STOP_TOKENS.contains(&first);
    if !stopped && maximum > 0 {
        check()?;
        generated.push(first);
        let mut pending = PendingDecode(Some(executor.decode_greedy(first)?));
        while generated.len() < maximum {
            check()?;
            // Preserve one-step lookahead, but give the guard the new native
            // step before waiting on the preceding result.
            let next = executor.decode_greedy_after(pending.0.as_ref().expect("queued step"))?;
            let current = pending.0.replace(next).expect("queued step");
            let token = current.wait_one()?;
            check()?;
            if STOP_TOKENS.contains(&token) {
                stopped = true;
                break;
            }
            generated.push(token);
        }
        if let Some(last) = pending.0.take() {
            last.wait_one()?;
        }
    }
    check()?;
    Ok((generated, stopped))
}

/// Identifies the audio span for anything keyed on the prompt: SHA-256 of
/// the log-mel features, which fix both the samples and the front end.
fn feature_digest(mel: &LogMel) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"qwen3-asr log-mel f32 bin-major\0");
    hasher.update(u64::try_from(mel.bins()).unwrap_or(u64::MAX).to_le_bytes());
    hasher.update(
        u64::try_from(mel.frames())
            .unwrap_or(u64::MAX)
            .to_le_bytes(),
    );
    for value in mel.values() {
        hasher.update(value.to_le_bytes());
    }
    hasher.finalize().into()
}

/// The first index of the largest logit, as `torch.argmax` picks.
fn argmax(logits: &[f32]) -> Result<i32, AsrError> {
    let mut best = 0;
    for (index, &value) in logits.iter().enumerate() {
        if !value.is_finite() {
            return Err(AsrError::NonFiniteLogits);
        }
        if value > logits[best] {
            best = index;
        }
    }
    i32::try_from(best).map_err(|_| AsrError::Shape)
}

/// Errors while loading or running Qwen3-ASR.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AsrError {
    /// The caller cancelled the transcription.
    #[error("transcription cancelled")]
    Cancelled,
    /// The caller's cooperative time budget expired.
    #[error("generation time budget exceeded")]
    DeadlineExceeded,
    /// A checkpoint file could not be read.
    #[error("cannot read {0}: {1}")]
    Read(PathBuf, String),
    /// The configuration is unsupported.
    #[error(transparent)]
    Config(#[from] crate::config::Qwen3AsrConfigError),
    /// The tokenizer could not be built or used.
    #[error(transparent)]
    Tokenizer(#[from] AsrTokenizerError),
    /// The encoder failed.
    #[error(transparent)]
    Encoder(#[from] EncoderError),
    /// The decoder failed.
    #[error(transparent)]
    Decoder(#[from] Qwen3ForwardError),
    /// The audio span does not fit the prompt.
    #[error(transparent)]
    Span(#[from] SpanError),
    /// Feature extraction refused the audio.
    #[error(transparent)]
    Features(#[from] audio::LogMelError),
    /// A checkpoint tensor is not part of this model.
    #[error("unexpected checkpoint tensor {0}")]
    UnexpectedTensor(String),
    /// The audio is longer than one pass accepts.
    #[error("audio is {0:.1} s; at most {MAX_AUDIO_SECONDS} s is transcribed in one pass")]
    TooLong(f64),
    /// The forced language is not one the checkpoint lists.
    #[error("unsupported language {0:?}")]
    Language(String),
    /// The decoder produced a non-finite logit.
    #[error("non-finite logits")]
    NonFiniteLogits,
    /// A size does not fit.
    #[error("shape overflow")]
    Shape,
    /// MLX reported an error.
    #[error(transparent)]
    Mlx(#[from] mlx_rs::error::Exception),
}

#[cfg(test)]
mod loader_boundary_tests {
    use super::*;

    #[test]
    fn packed_checkpoint_is_refused_before_tokenizer_or_weight_files_are_needed() {
        // This directory intentionally has only metadata: reaching tokenizer or
        // payload loading would produce a different error and fail this gate.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "metallix-asr-packed-boundary-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let config = directory.join("config.json");
        std::fs::write(
            &config,
            r#"{"quantization":{"bits":8,"group_size":64,"mode":"affine"}}"#,
        )
        .unwrap();
        let result = Qwen3Asr::load(&directory);
        std::fs::remove_file(config).unwrap();
        std::fs::remove_dir(directory).unwrap();
        assert!(matches!(result, Err(AsrError::Config(
            crate::config::Qwen3AsrConfigError::UnsupportedQuantization { field }
        )) if field == "/quantization"));
    }
}
