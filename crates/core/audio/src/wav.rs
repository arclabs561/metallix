//! A bounded RIFF/WAVE reader for uploaded audio.
//!
//! It reads integer PCM (8, 16, 24 and 32 bit) and IEEE float (32 and 64 bit)
//! samples, including `WAVE_FORMAT_EXTENSIBLE` files, and averages channels to
//! mono. Integer samples are scaled the way libsndfile (and therefore
//! soundfile and librosa) scales them: by `2^(bits - 1)`, with 8-bit samples
//! offset by 128 first.

/// Mono samples at a known sample rate.
///
/// The rate travels with the samples so a feature extractor can refuse audio
/// recorded at a rate it was not built for, instead of silently computing
/// features on the wrong time scale.
#[derive(Clone, Debug, PartialEq)]
pub struct MonoAudio {
    sample_rate: u32,
    samples: Vec<f32>,
}

impl MonoAudio {
    /// Wraps finite samples at `sample_rate` Hz.
    ///
    /// # Errors
    ///
    /// Returns [`WavError::SampleRate`] for a zero rate and
    /// [`WavError::NonFiniteSample`] for a NaN or infinite sample.
    pub fn new(sample_rate: u32, samples: Vec<f32>) -> Result<Self, WavError> {
        if sample_rate == 0 {
            return Err(WavError::SampleRate(sample_rate));
        }
        if let Some(index) = samples.iter().position(|sample| !sample.is_finite()) {
            return Err(WavError::NonFiniteSample { index });
        }
        Ok(Self {
            sample_rate,
            samples,
        })
    }

    /// The sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The samples, nominally in `[-1, 1]`.
    #[must_use]
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }

    /// Duration in seconds.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "durations are reported, not used for indexing"
    )]
    pub fn duration_seconds(&self) -> f64 {
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }
}

/// Errors while reading a WAV file.
#[derive(Debug, thiserror::Error, PartialEq)]
#[non_exhaustive]
pub enum WavError {
    /// The bytes do not start with a RIFF/WAVE header.
    #[error("not a RIFF/WAVE file")]
    NotWave,
    /// A chunk header or the `fmt ` chunk is cut short.
    #[error("WAV file is truncated")]
    Truncated,
    /// No `fmt ` chunk came before the `data` chunk.
    #[error("WAV file has no fmt chunk before its data")]
    MissingFormat,
    /// No `data` chunk was found.
    #[error("WAV file has no data chunk")]
    MissingData,
    /// The sample encoding is not integer PCM or IEEE float.
    #[error("unsupported WAV sample format tag {0:#06x}")]
    UnsupportedFormat(u16),
    /// The sample width is not supported for this encoding.
    #[error("unsupported WAV sample width: {bits} bits for format tag {format:#06x}")]
    UnsupportedBitDepth {
        /// Format tag after resolving `WAVE_FORMAT_EXTENSIBLE`.
        format: u16,
        /// Declared bits per sample.
        bits: u16,
    },
    /// The channel count is zero or above `MAX_CHANNELS`.
    #[error("unsupported WAV channel count {0}")]
    Channels(u16),
    /// The sample rate is zero or above `MAX_SAMPLE_RATE`.
    #[error("unsupported sample rate {0} Hz")]
    SampleRate(u32),
    /// `block_align` disagrees with the channel count and sample width.
    #[error("WAV block_align {actual} does not match {expected}")]
    BlockAlign {
        /// Declared `block_align`.
        actual: u16,
        /// `channels * bits / 8`.
        expected: u32,
    },
    /// The audio is longer than the caller accepts.
    #[error("audio is {seconds:.1} s long; the limit is {limit:.0} s")]
    TooLong {
        /// Declared duration.
        seconds: f64,
        /// The caller's limit.
        limit: f64,
    },
    /// A float sample is NaN or infinite.
    #[error("sample {index} is not finite")]
    NonFiniteSample {
        /// Index of the first non-finite mono sample.
        index: usize,
    },
}

/// Largest channel count accepted.
pub const MAX_CHANNELS: u16 = 32;
/// Largest sample rate accepted, in Hz.
pub const MAX_SAMPLE_RATE: u32 = 768_000;

const FORMAT_PCM: u16 = 0x0001;
const FORMAT_FLOAT: u16 = 0x0003;
const FORMAT_EXTENSIBLE: u16 = 0xfffe;

#[derive(Clone, Copy)]
struct Format {
    tag: u16,
    channels: u16,
    sample_rate: u32,
    bits: u16,
}

/// What a WAV header declares, read without decoding samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WavInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Interleaved channels.
    pub channels: u16,
    /// Whole frames present in the input.
    pub frames: usize,
}

impl WavInfo {
    /// Duration in seconds.
    #[must_use]
    #[allow(
        clippy::cast_precision_loss,
        reason = "durations are compared to limits, not used for indexing"
    )]
    pub fn duration_seconds(&self) -> f64 {
        self.frames as f64 / f64::from(self.sample_rate)
    }
}

/// Reads a WAV file's format and length without decoding its samples, so a
/// caller can refuse long audio before allocating for it.
///
/// # Errors
///
/// Returns [`WavError`] for malformed headers or unsupported encodings.
pub fn probe_wav(bytes: &[u8]) -> Result<WavInfo, WavError> {
    let (format, body, _) = locate(bytes)?;
    Ok(WavInfo {
        sample_rate: format.sample_rate,
        channels: format.channels,
        frames: body.len() / frame_bytes(format),
    })
}

/// [`decode_wav`] after checking, from the header alone, that the audio is
/// at most `max_seconds` long.
///
/// # Errors
///
/// Returns [`WavError::TooLong`] for longer audio, and otherwise as
/// [`decode_wav`].
pub fn decode_wav_limited(bytes: &[u8], max_seconds: f64) -> Result<MonoAudio, WavError> {
    let seconds = probe_wav(bytes)?.duration_seconds();
    if seconds > max_seconds {
        return Err(WavError::TooLong {
            seconds,
            limit: max_seconds,
        });
    }
    decode_wav(bytes)
}

/// Decodes a complete WAV file and averages its channels to mono.
///
/// A `data` chunk whose declared size runs past the end of the input (as
/// written by encoders streaming to a pipe) is read up to the last whole
/// frame present; every other inconsistency is an error.
///
/// # Errors
///
/// Returns [`WavError`] for malformed headers, unsupported encodings, or
/// non-finite float samples.
pub fn decode_wav(bytes: &[u8]) -> Result<MonoAudio, WavError> {
    let (format, body, complete) = locate(bytes)?;
    decode_samples(format, body, complete)
}

/// The format and the `data` chunk body, and whether that body is as long as
/// its header declares.
fn locate(bytes: &[u8]) -> Result<(Format, &[u8], bool), WavError> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(WavError::NotWave);
    }
    let mut cursor = 12_usize;
    let mut format = None;
    while cursor < bytes.len() {
        let header = bytes.get(cursor..cursor + 8).ok_or(WavError::Truncated)?;
        let id = &header[0..4];
        let declared = usize::try_from(u32::from_le_bytes([
            header[4], header[5], header[6], header[7],
        ]))
        .map_err(|_| WavError::Truncated)?;
        let body_start = cursor + 8;
        let remaining = bytes.len() - body_start;
        if id == b"data" {
            let format = format.ok_or(WavError::MissingFormat)?;
            let body = &bytes[body_start..body_start + declared.min(remaining)];
            return Ok((format, body, declared <= remaining));
        }
        if declared > remaining {
            return Err(WavError::Truncated);
        }
        let body = &bytes[body_start..body_start + declared];
        if id == b"fmt " {
            format = Some(parse_format(body)?);
        }
        // Chunks are padded to an even length.
        cursor = body_start + declared + (declared & 1);
    }
    Err(WavError::MissingData)
}

fn frame_bytes(format: Format) -> usize {
    usize::from(format.bits / 8) * usize::from(format.channels)
}

fn parse_format(body: &[u8]) -> Result<Format, WavError> {
    let field = |offset: usize| -> Result<u16, WavError> {
        body.get(offset..offset + 2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .ok_or(WavError::Truncated)
    };
    let mut tag = field(0)?;
    let channels = field(2)?;
    let sample_rate = body
        .get(4..8)
        .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .ok_or(WavError::Truncated)?;
    let block_align = field(12)?;
    let bits = field(14)?;
    if tag == FORMAT_EXTENSIBLE {
        // The sub-format GUID starts at byte 24; its first two bytes are the
        // ordinary format tag.
        tag = field(24)?;
    }
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(WavError::Channels(channels));
    }
    if sample_rate == 0 || sample_rate > MAX_SAMPLE_RATE {
        return Err(WavError::SampleRate(sample_rate));
    }
    match (tag, bits) {
        (FORMAT_PCM, 8 | 16 | 24 | 32) | (FORMAT_FLOAT, 32 | 64) => {}
        (FORMAT_PCM | FORMAT_FLOAT, _) => {
            return Err(WavError::UnsupportedBitDepth { format: tag, bits });
        }
        _ => return Err(WavError::UnsupportedFormat(tag)),
    }
    let expected = u32::from(channels) * u32::from(bits / 8);
    if u32::from(block_align) != expected {
        return Err(WavError::BlockAlign {
            actual: block_align,
            expected,
        });
    }
    Ok(Format {
        tag,
        channels,
        sample_rate,
        bits,
    })
}

fn decode_samples(format: Format, body: &[u8], complete: bool) -> Result<MonoAudio, WavError> {
    let width = usize::from(format.bits / 8);
    let channels = usize::from(format.channels);
    let frame_bytes = frame_bytes(format);
    if complete && !body.len().is_multiple_of(frame_bytes) {
        return Err(WavError::Truncated);
    }
    let frames = body.len() / frame_bytes;
    let mut mono = Vec::with_capacity(frames);
    #[allow(
        clippy::cast_precision_loss,
        reason = "channel counts are at most MAX_CHANNELS"
    )]
    let channel_count = channels as f32;
    for frame in body.chunks_exact(frame_bytes) {
        let mut sum = 0.0_f32;
        for sample in frame.chunks_exact(width) {
            sum += sample_value(format.tag, sample);
        }
        mono.push(if channels == 1 {
            sum
        } else {
            sum / channel_count
        });
    }
    MonoAudio::new(format.sample_rate, mono)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "this is the libsndfile int and f64 to f32 conversion"
)]
fn sample_value(tag: u16, bytes: &[u8]) -> f32 {
    match (tag, bytes.len()) {
        (FORMAT_FLOAT, 4) => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        (FORMAT_FLOAT, 8) => f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f32,
        (_, 1) => (f32::from(bytes[0]) - 128.0) / 128.0,
        (_, 2) => f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32_768.0,
        // Sign-extend by placing the 24 bits in the high bytes of an i32.
        (_, 3) => (i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 8) as f32 / 8_388_608.0,
        (_, _) => {
            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32 / 2_147_483_648.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn wav(tag: u16, channels: u16, rate: u32, bits: u16, data: &[u8]) -> Vec<u8> {
        let block_align = channels * (bits / 8);
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        fmt.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        fmt.extend_from_slice(&block_align.to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());
        let mut out = b"RIFF\0\0\0\0WAVE".to_vec();
        for (id, body) in [(b"fmt ", fmt.as_slice()), (b"data", data)] {
            out.extend_from_slice(id);
            out.extend_from_slice(&u32::try_from(body.len()).unwrap().to_le_bytes());
            out.extend_from_slice(body);
            if body.len() % 2 == 1 {
                out.push(0);
            }
        }
        let riff = u32::try_from(out.len() - 8).unwrap();
        out[4..8].copy_from_slice(&riff.to_le_bytes());
        out
    }

    #[test]
    fn scales_integer_pcm_like_libsndfile() {
        let pcm16: Vec<u8> = [i16::MIN, -1, 0, 16_384, i16::MAX]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let audio = decode_wav(&wav(1, 1, 16_000, 16, &pcm16)).unwrap();
        assert_eq!(audio.sample_rate(), 16_000);
        assert_eq!(
            audio.samples(),
            [-1.0, -1.0 / 32_768.0, 0.0, 0.5, 32_767.0 / 32_768.0]
        );

        // 24-bit: -2^23, -1 and 2^22.
        let pcm24 = [0x00, 0x00, 0x80, 0xff, 0xff, 0xff, 0x00, 0x00, 0x40];
        let audio = decode_wav(&wav(1, 1, 8_000, 24, &pcm24)).unwrap();
        assert_eq!(audio.samples(), [-1.0, -1.0 / 8_388_608.0, 0.5]);

        let pcm8 = [0, 128, 192];
        let audio = decode_wav(&wav(1, 1, 8_000, 8, &pcm8)).unwrap();
        assert_eq!(audio.samples(), [-1.0, 0.0, 0.5]);
    }

    #[test]
    fn averages_channels_to_mono() {
        let stereo: Vec<u8> = [0.5_f32, -0.25, 1.0, 1.0]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let audio = decode_wav(&wav(3, 2, 48_000, 32, &stereo)).unwrap();
        assert_eq!(audio.samples(), [0.125, 1.0]);
    }

    #[test]
    fn reads_extensible_float_and_skips_unknown_chunks() {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&FORMAT_EXTENSIBLE.to_le_bytes());
        fmt.extend_from_slice(&1_u16.to_le_bytes());
        fmt.extend_from_slice(&16_000_u32.to_le_bytes());
        fmt.extend_from_slice(&64_000_u32.to_le_bytes());
        fmt.extend_from_slice(&4_u16.to_le_bytes());
        fmt.extend_from_slice(&32_u16.to_le_bytes());
        fmt.extend_from_slice(&22_u16.to_le_bytes());
        fmt.extend_from_slice(&32_u16.to_le_bytes());
        fmt.extend_from_slice(&0_u32.to_le_bytes());
        fmt.extend_from_slice(&FORMAT_FLOAT.to_le_bytes());
        fmt.extend_from_slice(&[0; 14]);
        let mut bytes = b"RIFF\0\0\0\0WAVE".to_vec();
        bytes.extend_from_slice(b"LIST");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(b"abc\0");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&u32::try_from(fmt.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&fmt);
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&0.75_f32.to_le_bytes());
        assert_eq!(decode_wav(&bytes).unwrap().samples(), [0.75]);
    }

    #[test]
    fn refuses_inconsistent_or_unsupported_headers() {
        assert_eq!(decode_wav(b"RIFX\0\0\0\0WAVE"), Err(WavError::NotWave));
        assert_eq!(
            decode_wav(&wav(2, 1, 16_000, 16, &[0, 0])),
            Err(WavError::UnsupportedFormat(2))
        );
        assert_eq!(
            decode_wav(&wav(1, 1, 16_000, 12, &[0, 0])),
            Err(WavError::UnsupportedBitDepth {
                format: 1,
                bits: 12
            })
        );
        assert_eq!(
            decode_wav(&wav(1, 0, 16_000, 16, &[])),
            Err(WavError::Channels(0))
        );
        assert_eq!(
            decode_wav(&wav(1, 1, 0, 16, &[])),
            Err(WavError::SampleRate(0))
        );
        // Three bytes of 16-bit data is half a frame.
        assert_eq!(
            decode_wav(&wav(1, 1, 16_000, 16, &[0, 0, 0])),
            Err(WavError::Truncated)
        );
        let nan: Vec<u8> = f32::NAN.to_le_bytes().to_vec();
        assert_eq!(
            decode_wav(&wav(3, 1, 16_000, 32, &nan)),
            Err(WavError::NonFiniteSample { index: 0 })
        );
        let mut no_fmt = b"RIFF\0\0\0\0WAVEdata\x02\0\0\0\0\0".to_vec();
        no_fmt[4] = 14;
        assert_eq!(decode_wav(&no_fmt), Err(WavError::MissingFormat));
    }

    #[test]
    fn reads_whole_frames_of_an_overlong_streamed_data_chunk() {
        let mut bytes = wav(1, 1, 16_000, 16, &[0, 64, 0, 0xc0]);
        // Declare far more data than present, as a pipe writer does, and end
        // the input with half a frame.
        let data_size_at = bytes.len() - 4 - 4;
        bytes[data_size_at..data_size_at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        bytes.push(1);
        assert_eq!(decode_wav(&bytes).unwrap().samples(), [0.5, -0.5]);
    }

    #[test]
    fn probes_duration_and_refuses_long_audio_before_decoding() {
        let stereo_second = vec![0_u8; 16_000 * 2 * 2];
        let bytes = wav(1, 2, 16_000, 16, &stereo_second);
        let info = probe_wav(&bytes).unwrap();
        assert_eq!(
            (info.sample_rate, info.channels, info.frames),
            (16_000, 2, 16_000)
        );
        assert!((info.duration_seconds() - 1.0).abs() < 1e-12);
        assert_eq!(
            decode_wav_limited(&bytes, 0.5),
            Err(WavError::TooLong {
                seconds: 1.0,
                limit: 0.5
            })
        );
        assert_eq!(
            decode_wav_limited(&bytes, 1.0).unwrap().samples().len(),
            16_000
        );
    }

    proptest! {
        // Arbitrary bytes, or a valid file cut at any point, must return a
        // value or an error, never panic.
        #[test]
        fn never_panics_on_arbitrary_or_truncated_input(
            noise in proptest::collection::vec(any::<u8>(), 0..256),
            cut in 0_usize..64,
        ) {
            let _ = decode_wav(&noise);
            let valid = wav(1, 2, 16_000, 16, &[1, 2, 3, 4, 5, 6, 7, 8]);
            let _ = decode_wav(&valid[..valid.len().saturating_sub(cut)]);
            let mut prefixed = b"RIFF\0\0\0\0WAVE".to_vec();
            prefixed.extend_from_slice(&noise);
            let _ = decode_wav(&prefixed);
        }
    }
}
