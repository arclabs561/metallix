//! Qwen3-ASR `config.json`: the audio encoder, the Qwen3 text decoder and the
//! audio placeholder tokens.
//!
//! Layout reference: `configuration_qwen3_asr.py` and `modeling_qwen3_asr.py`
//! at <https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff>.

use serde::Deserialize;
use serde_json::Value;

/// Mel frames per encoder chunk. The reference output-length formula
/// (`_get_feat_extract_output_lengths`) hard-codes 100, so only checkpoints
/// with `n_window = 50` are accepted.
pub const CHUNK_FRAMES: usize = 100;

/// Encoder frames produced by one full chunk: three stride-2 convolutions,
/// `ceil(100 / 8)`.
pub const FRAMES_PER_CHUNK: usize = 13;

/// Mel bins the encoder's first convolution expects.
pub const MEL_BINS: usize = 128;

/// The audio encoder's shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioEncoderConfig {
    /// Transformer width.
    pub d_model: usize,
    /// Encoder layers.
    pub layers: usize,
    /// Attention heads per layer.
    pub heads: usize,
    /// MLP width.
    pub ffn_dim: usize,
    /// Channels of the three downsampling convolutions.
    pub conv_channels: usize,
    /// Sinusoid table length; must cover one chunk's frames.
    pub max_source_positions: usize,
    /// Attention window in chunks (`n_window_infer / (2 n_window)`).
    pub window_chunks: usize,
    /// Width of the projected output: the decoder's hidden size.
    pub output_dim: usize,
}

impl AudioEncoderConfig {
    /// Frequency rows left after the three stride-2 convolutions over
    /// [`MEL_BINS`]: `ceil(ceil(ceil(128 / 2) / 2) / 2) = 16`.
    #[must_use]
    pub const fn conv_frequency_rows() -> usize {
        MEL_BINS.div_ceil(2).div_ceil(2).div_ceil(2)
    }

    /// Encoder frames in one attention window (104 for the released models).
    #[must_use]
    pub const fn window_frames(&self) -> usize {
        self.window_chunks * FRAMES_PER_CHUNK
    }
}

/// Token ids that frame and stand in for audio in the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioTokens {
    /// `<|audio_start|>`.
    pub start: i32,
    /// `<|audio_end|>`.
    pub end: i32,
    /// `<|audio_pad|>`, repeated once per encoder frame.
    pub pad: i32,
}

/// A validated Qwen3-ASR configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct Qwen3AsrConfig {
    encoder: AudioEncoderConfig,
    tokens: AudioTokens,
    text_config: Value,
    languages: Vec<String>,
}

impl Qwen3AsrConfig {
    /// Parses and validates a dense checkpoint's `config.json`.
    ///
    /// Packed/quantized checkpoints are not implemented by the ASR loader,
    /// even when the standalone Qwen decoder supports their format. Non-null
    /// `quantization` or `quantization_config` declarations in the root,
    /// thinker, audio or text configuration are rejected before weight I/O.
    ///
    /// # Errors
    ///
    /// Returns [`Qwen3AsrConfigError`] for another model type, an encoder
    /// layout the reference code would not run the same way, or a decoder
    /// rope that is not equivalent to plain 1-D `RoPE` for audio prompts.
    pub fn parse(json: &str) -> Result<Self, Qwen3AsrConfigError> {
        let value: Value = serde_json::from_str(json)?;
        reject_quantization(&value)?;
        let raw: RawConfig = serde_json::from_value(value)?;
        if raw.model_type != "qwen3_asr" {
            return Err(Qwen3AsrConfigError::ModelType(raw.model_type));
        }
        let thinker = raw.thinker_config;
        let audio = thinker.audio_config;
        if audio.activation_function != "gelu" {
            return Err(Qwen3AsrConfigError::Activation(audio.activation_function));
        }
        if audio.num_mel_bins != MEL_BINS {
            return Err(Qwen3AsrConfigError::MelBins(audio.num_mel_bins));
        }
        if audio.n_window * 2 != CHUNK_FRAMES
            || audio.n_window_infer == 0
            || !audio.n_window_infer.is_multiple_of(CHUNK_FRAMES)
        {
            return Err(Qwen3AsrConfigError::Window {
                n_window: audio.n_window,
                n_window_infer: audio.n_window_infer,
            });
        }
        if audio.encoder_attention_heads == 0
            || !audio.d_model.is_multiple_of(audio.encoder_attention_heads)
            || audio.encoder_layers == 0
            || audio.encoder_ffn_dim == 0
            || audio.downsample_hidden_size == 0
        {
            return Err(Qwen3AsrConfigError::EncoderShape);
        }
        if audio.scale_embedding {
            return Err(Qwen3AsrConfigError::ScaledEmbedding);
        }
        if audio.max_source_positions < FRAMES_PER_CHUNK {
            return Err(Qwen3AsrConfigError::EncoderShape);
        }

        let mut text_config = thinker.text_config;
        let text = text_config
            .as_object_mut()
            .ok_or(Qwen3AsrConfigError::TextConfig)?;
        let hidden_size = text
            .get("hidden_size")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(Qwen3AsrConfigError::TextConfig)?;
        let head_dim = text
            .get("head_dim")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(Qwen3AsrConfigError::TextConfig)?;
        if audio.output_dim != hidden_size {
            return Err(Qwen3AsrConfigError::ProjectionWidth {
                output_dim: audio.output_dim,
                hidden_size,
            });
        }
        // The decoder declares interleaved MRoPE, but `get_rope_index` gives
        // every token one position repeated on all three axes. Each frequency
        // then rotates by the same position whichever axis it is assigned to,
        // so this is plain rotate-half `RoPE`, which the Qwen3 decoder runs
        // once the field is removed.
        if let Some(rope) = text.remove("rope_scaling") {
            check_degenerate_mrope(&rope, head_dim)?;
        }

        Ok(Self {
            encoder: AudioEncoderConfig {
                d_model: audio.d_model,
                layers: audio.encoder_layers,
                heads: audio.encoder_attention_heads,
                ffn_dim: audio.encoder_ffn_dim,
                conv_channels: audio.downsample_hidden_size,
                max_source_positions: audio.max_source_positions,
                window_chunks: audio.n_window_infer / CHUNK_FRAMES,
                output_dim: audio.output_dim,
            },
            tokens: AudioTokens {
                start: thinker.audio_start_token_id,
                end: thinker.audio_end_token_id,
                pad: thinker.audio_token_id,
            },
            text_config,
            languages: raw.support_languages,
        })
    }

    /// The audio encoder's shape.
    #[must_use]
    pub const fn encoder(&self) -> &AudioEncoderConfig {
        &self.encoder
    }

    /// Prompt token ids for audio.
    #[must_use]
    pub const fn tokens(&self) -> AudioTokens {
        self.tokens
    }

    /// The decoder's `text_config` with the degenerate `rope_scaling`
    /// removed, ready for the Qwen3 decoder's configuration parser.
    #[must_use]
    pub fn text_config_json(&self) -> String {
        self.text_config.to_string()
    }

    /// Language names the checkpoint supports (`support_languages`), as
    /// written after `language` in its output.
    #[must_use]
    pub fn languages(&self) -> &[String] {
        &self.languages
    }
}

fn check_degenerate_mrope(rope: &Value, head_dim: usize) -> Result<(), Qwen3AsrConfigError> {
    let rope_type = rope
        .get("rope_type")
        .or_else(|| rope.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("default");
    if rope_type != "default" {
        return Err(Qwen3AsrConfigError::Rope(rope.to_string()));
    }
    let sections: Option<Vec<u64>> = rope
        .get("mrope_section")
        .and_then(Value::as_array)
        .and_then(|values| values.iter().map(Value::as_u64).collect());
    match sections {
        Some(sections)
            if sections.len() == 3
                && usize::try_from(sections.iter().sum::<u64>()).ok() == Some(head_dim / 2) =>
        {
            Ok(())
        }
        _ => Err(Qwen3AsrConfigError::Rope(rope.to_string())),
    }
}

/// Errors while reading a Qwen3-ASR configuration.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Qwen3AsrConfigError {
    /// A declared packed format that the dense ASR loader cannot execute.
    #[error("unsupported ASR quantization at {field}; only dense checkpoints are implemented")]
    UnsupportedQuantization {
        /// JSON pointer to the unsupported declaration.
        field: String,
    },
    /// The JSON does not have the expected structure.
    #[error("invalid Qwen3-ASR config: {0}")]
    Json(#[from] serde_json::Error),
    /// `model_type` is not `qwen3_asr`.
    #[error("model_type {0:?} is not qwen3_asr")]
    ModelType(String),
    /// The encoder activation is not GELU.
    #[error("unsupported encoder activation {0:?}")]
    Activation(String),
    /// The encoder expects a mel bin count the front end does not produce.
    #[error("encoder expects {0} mel bins; the front end produces 128")]
    MelBins(usize),
    /// Chunk or attention-window sizes differ from the reference's fixed
    /// 100-frame chunks.
    #[error("unsupported encoder windows: n_window {n_window}, n_window_infer {n_window_infer}")]
    Window {
        /// Half the chunk length in mel frames.
        n_window: usize,
        /// Attention window in mel frames.
        n_window_infer: usize,
    },
    /// A zero or inconsistent encoder dimension.
    #[error("inconsistent encoder dimensions")]
    EncoderShape,
    /// `scale_embedding` is set, which the released models do not use.
    #[error("scaled encoder embeddings are not supported")]
    ScaledEmbedding,
    /// `text_config` lacks `hidden_size` or `head_dim`.
    #[error("text_config is missing hidden_size or head_dim")]
    TextConfig,
    /// The encoder projects to a width other than the decoder's.
    #[error("encoder output_dim {output_dim} differs from decoder hidden_size {hidden_size}")]
    ProjectionWidth {
        /// Encoder projection width.
        output_dim: usize,
        /// Decoder hidden size.
        hidden_size: usize,
    },
    /// A rope configuration that is not plain `RoPE` for audio prompts.
    #[error("unsupported decoder rope_scaling {0}")]
    Rope(String),
}

fn reject_quantization(config: &Value) -> Result<(), Qwen3AsrConfigError> {
    // Only known configuration scopes: unrelated extension metadata remains
    // forward-compatible, and null is the ordinary unset value.
    for scope in [
        "",
        "/thinker_config",
        "/audio_config",
        "/text_config",
        "/thinker_config/audio_config",
        "/thinker_config/text_config",
    ] {
        for key in ["quantization", "quantization_config"] {
            let field = format!("{scope}/{key}");
            if config.pointer(&field).is_some_and(|value| !value.is_null()) {
                return Err(Qwen3AsrConfigError::UnsupportedQuantization { field });
            }
        }
    }
    Ok(())
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    #[serde(default)]
    support_languages: Vec<String>,
    thinker_config: RawThinker,
}

#[derive(Deserialize)]
struct RawThinker {
    audio_config: RawAudio,
    text_config: Value,
    audio_start_token_id: i32,
    audio_end_token_id: i32,
    audio_token_id: i32,
}

#[derive(Deserialize)]
struct RawAudio {
    activation_function: String,
    d_model: usize,
    encoder_layers: usize,
    encoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    num_mel_bins: usize,
    downsample_hidden_size: usize,
    max_source_positions: usize,
    n_window: usize,
    n_window_infer: usize,
    output_dim: usize,
    #[serde(default)]
    scale_embedding: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(audio: &str, rope: &str) -> String {
        format!(
            r#"{{"model_type": "qwen3_asr", "support_languages": ["English", "Chinese"],
            "thinker_config": {{
              "audio_start_token_id": 151669, "audio_end_token_id": 151670,
              "audio_token_id": 151676,
              "audio_config": {{"activation_function": "gelu", "d_model": 896,
                "encoder_layers": 18, "encoder_attention_heads": 14,
                "encoder_ffn_dim": 3584, "num_mel_bins": 128,
                "downsample_hidden_size": 480, "max_source_positions": 1500,
                "n_window": 50, "n_window_infer": 800, "output_dim": 1024,
                "scale_embedding": false {audio}}},
              "text_config": {{"model_type": "qwen3", "hidden_size": 1024,
                "head_dim": 128, "rope_theta": 1000000 {rope}}}}}}}"#
        )
    }

    const MROPE: &str = r#", "rope_scaling": {"interleaved": true,
        "mrope_interleaved": true, "mrope_section": [24, 20, 20],
        "rope_type": "default", "type": "default"}"#;

    #[test]
    fn reads_the_released_0_6b_layout() {
        let parsed = Qwen3AsrConfig::parse(&config("", MROPE)).unwrap();
        assert_eq!(parsed.encoder().window_frames(), 104);
        assert_eq!(AudioEncoderConfig::conv_frequency_rows(), 16);
        assert_eq!(
            parsed.tokens(),
            AudioTokens {
                start: 151_669,
                end: 151_670,
                pad: 151_676
            }
        );
        assert_eq!(parsed.languages(), ["English", "Chinese"]);
        let text: Value = serde_json::from_str(&parsed.text_config_json()).unwrap();
        assert!(text.get("rope_scaling").is_none());
        assert_eq!(text["rope_theta"], 1_000_000);
    }

    #[test]
    fn refuses_layouts_the_reference_would_run_differently() {
        let scaled = MROPE.replace(
            r#""rope_type": "default", "type": "default""#,
            r#""rope_type": "yarn""#,
        );
        assert!(matches!(
            Qwen3AsrConfig::parse(&config("", &scaled)),
            Err(Qwen3AsrConfigError::Rope(_))
        ));
        let uneven = MROPE.replace("[24, 20, 20]", "[24, 20, 16]");
        assert!(matches!(
            Qwen3AsrConfig::parse(&config("", &uneven)),
            Err(Qwen3AsrConfigError::Rope(_))
        ));
        let windows = config("", MROPE).replace(r#""n_window": 50"#, r#""n_window": 25"#);
        assert!(matches!(
            Qwen3AsrConfig::parse(&windows),
            Err(Qwen3AsrConfigError::Window { .. })
        ));
        let width = config("", MROPE).replace(r#""output_dim": 1024"#, r#""output_dim": 2048"#);
        assert!(matches!(
            Qwen3AsrConfig::parse(&width),
            Err(Qwen3AsrConfigError::ProjectionWidth { .. })
        ));
    }

    #[test]
    fn rejects_packed_declarations_in_each_checkpoint_config_scope() {
        for scope in [
            "",
            "/thinker_config",
            "/audio_config",
            "/text_config",
            "/thinker_config/audio_config",
            "/thinker_config/text_config",
        ] {
            for key in ["quantization", "quantization_config"] {
                let mut value: Value = serde_json::from_str(&config("", MROPE)).unwrap();
                if matches!(scope, "/audio_config" | "/text_config") {
                    value[&scope[1..]] = serde_json::json!({});
                }
                value.pointer_mut(scope).unwrap()[key] = serde_json::json!({
                    "group_size": 64, "bits": 8, "mode": "affine"
                });
                let error = Qwen3AsrConfig::parse(&value.to_string()).unwrap_err();
                assert!(
                    matches!(error, Qwen3AsrConfigError::UnsupportedQuantization { field }
                    if field == format!("{scope}/{key}"))
                );
            }
        }
    }

    #[test]
    fn unset_quantization_and_unrelated_metadata_keep_dense_config_valid() {
        let mut value: Value = serde_json::from_str(&config("", MROPE)).unwrap();
        value["quantization"] = Value::Null;
        value["thinker_config"]["text_config"]["quantization_config"] = Value::Null;
        value["extension"] = serde_json::json!({"quantization": "unrelated metadata"});
        assert!(Qwen3AsrConfig::parse(&value.to_string()).is_ok());
    }

    #[test]
    fn packed_declaration_is_rejected_before_dense_layout_deserialization() {
        let error = Qwen3AsrConfig::parse(r#"{"quantization_config":{"bits":8}}"#).unwrap_err();
        assert!(matches!(
            error,
            Qwen3AsrConfigError::UnsupportedQuantization { .. }
        ));
    }
}
