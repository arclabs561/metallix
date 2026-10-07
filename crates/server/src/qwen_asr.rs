//! Qwen3-ASR's model-local adaptation to the transcription worker contract.

use std::path::Path;

use crate::{
    chat_generation::ResidentChatLimits,
    transcriptions::{self, Error, Request, Response},
};

pub(crate) struct QwenAsr {
    model: qwen3_asr::Qwen3Asr,
    kv_bytes: u64,
}

impl QwenAsr {
    pub(crate) fn load(path: &Path, limits: ResidentChatLimits) -> Result<Self, String> {
        Ok(Self {
            model: qwen3_asr::Qwen3Asr::load(path).map_err(|e| e.to_string())?,
            kv_bytes: limits.kv_budget_bytes(),
        })
    }

    #[tracing::instrument(name = "asr.request", skip_all, fields(model = %request.model))]
    pub(crate) fn transcribe(
        &self,
        request: &Request,
        control: &mut transcriptions::Control<'_>,
    ) -> Result<Response, Error> {
        control.check().map_err(Error::from)?;
        let audio = waveform(request)?;
        let language = request
            .language
            .as_deref()
            .map(|language| {
                qwen3_asr::languages::resolve(language, self.model.config().languages())
                    .map(str::to_owned)
                    .ok_or_else(|| Error::invalid("language is not supported by this checkpoint"))
            })
            .transpose()?;
        let options = qwen3_asr::TranscribeOptions {
            context: request.prompt.clone(),
            language,
            max_kv_bytes: self.kv_bytes,
            ..Default::default()
        };
        let generated = self
            .model
            .transcribe_with_control(&audio, &options, &mut || {
                control.check().map_err(|stop| match stop {
                    transcriptions::Stop::Cancelled => qwen3_asr::AsrError::Cancelled,
                    transcriptions::Stop::DeadlineExceeded => qwen3_asr::AsrError::DeadlineExceeded,
                })
            })
            .map_err(|error| match error {
                qwen3_asr::AsrError::Cancelled => Error::from(transcriptions::Stop::Cancelled),
                qwen3_asr::AsrError::DeadlineExceeded => {
                    Error::from(transcriptions::Stop::DeadlineExceeded)
                }
                other => Error::invalid(other.to_string()),
            })?;
        Ok(transcriptions::result(
            request.format,
            generated.transcript.text,
            &generated.transcript.language,
            generated.audio_seconds,
        ))
    }
}

fn waveform(request: &Request) -> Result<audio::MonoAudio, Error> {
    let audio = audio::decode_wav_limited(request.file(), qwen3_asr::model::MAX_AUDIO_SECONDS)
        .map_err(|error| match error {
            audio::WavError::NotWave
            | audio::WavError::UnsupportedFormat(_)
            | audio::WavError::UnsupportedBitDepth { .. } => Error::unsupported(error.to_string()),
            _ => Error::invalid(error.to_string()),
        })?;
    if audio.sample_rate() != 16_000 {
        return Err(Error::unsupported(
            "this experimental ASR adapter accepts 16 kHz WAV only; resampling is not implemented",
        ));
    }
    Ok(audio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcription_adapter_checks_container_and_rate_before_model_work() {
        let wav = include_bytes!("../../audio/tests/fixtures/pcm16_stereo.wav");
        let request = |file: &[u8]| {
            Request::from_body(
                crate::transcriptions::tests::form(&[("model", b"asr"), ("file", file)]),
                Some("multipart/form-data; boundary=boundary"),
            )
            .unwrap()
        };
        assert_eq!(waveform(&request(wav)).unwrap().samples().len(), 6);
        assert_eq!(waveform(&request(b"not a WAV")).unwrap_err().status, 415);
        let mut other_rate = wav.to_vec();
        other_rate[24..28].copy_from_slice(&8_000_u32.to_le_bytes());
        other_rate[28..32].copy_from_slice(&32_000_u32.to_le_bytes());
        assert_eq!(waveform(&request(&other_rate)).unwrap_err().status, 415);
    }
}
