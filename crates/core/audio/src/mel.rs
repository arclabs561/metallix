//! Whisper-style log-mel features, as computed by transformers'
//! `WhisperFeatureExtractor` (its torch path) and `Qwen3ASRFeatureExtractor`.
//!
//! Steps, matching
//! <https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/whisper/feature_extraction_whisper.py>
//! and `mel_filter_bank` in the same release's `audio_utils.py`:
//! a centered STFT with reflect padding and a periodic Hann window; the power
//! spectrum with its last frame dropped; Slaney-scale, Slaney-normalized
//! triangular mel filters computed in f64 and used in f32;
//! `log10(max(x, 1e-10))`; a floor at the clip's maximum minus 8; then
//! `(x + 4) / 4`.
//!
//! The floor depends on the whole clip, so features of a clip's parts differ
//! from the matching parts of the whole clip's features.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

use crate::MonoAudio;

/// Parameters of a Whisper-style log-mel front end.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogMelSpec {
    sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    mel_bins: usize,
    max_frequency: f64,
}

impl LogMelSpec {
    /// Qwen3-ASR's `preprocessor_config.json`: 16 kHz, 128 bins, 25 ms window,
    /// 10 ms hop, filters up to 8 kHz.
    pub const QWEN3_ASR: Self = Self {
        sample_rate: 16_000,
        n_fft: 400,
        hop_length: 160,
        mel_bins: 128,
        max_frequency: 8_000.0,
    };

    /// Input sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of mel bins per frame.
    #[must_use]
    pub const fn mel_bins(&self) -> usize {
        self.mel_bins
    }

    /// Samples between frames.
    #[must_use]
    pub const fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Frames produced for `samples` input samples: one per hop, after the
    /// centered STFT's final frame is dropped.
    #[must_use]
    pub const fn frames_for(&self, samples: usize) -> usize {
        samples / self.hop_length
    }
}

/// Log-mel features, `[mel_bins, frames]` in row-major order (the layout of
/// the reference `input_features` for one clip).
#[derive(Clone, Debug, PartialEq)]
pub struct LogMel {
    bins: usize,
    frames: usize,
    values: Vec<f32>,
}

impl LogMel {
    /// Wraps bin-major `[bins, frames]` values computed elsewhere, such as a
    /// reference implementation's features for a parity check.
    ///
    /// # Errors
    ///
    /// Returns [`LogMelError::Shape`] when `values` is not `bins * frames`
    /// long.
    pub fn from_values(bins: usize, frames: usize, values: Vec<f32>) -> Result<Self, LogMelError> {
        if bins.checked_mul(frames) != Some(values.len()) {
            return Err(LogMelError::Shape {
                bins,
                frames,
                values: values.len(),
            });
        }
        Ok(Self {
            bins,
            frames,
            values,
        })
    }

    /// Number of mel bins.
    #[must_use]
    pub const fn bins(&self) -> usize {
        self.bins
    }

    /// Number of frames.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// All values, bin-major: `values()[bin * frames() + frame]`.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Errors while extracting features.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogMelError {
    /// The audio is at a different rate than the extractor expects.
    #[error("audio is at {actual} Hz; this front end expects {expected} Hz")]
    SampleRate {
        /// Rate of the supplied audio.
        actual: u32,
        /// Rate the spec was built for.
        expected: u32,
    },
    /// Supplied values do not fill `bins * frames`.
    #[error("{values} values do not fill {bins} bins by {frames} frames")]
    Shape {
        /// Declared bins.
        bins: usize,
        /// Declared frames.
        frames: usize,
        /// Supplied value count.
        values: usize,
    },
    /// Reflect padding needs more samples than half a window, and at least
    /// one frame must remain.
    #[error("{actual} samples is too short; at least {minimum} are needed")]
    TooShort {
        /// Supplied sample count.
        actual: usize,
        /// Smallest accepted sample count.
        minimum: usize,
    },
}

/// A reusable extractor holding the window, mel filters and FFT plan.
pub struct LogMelExtractor {
    spec: LogMelSpec,
    window: Vec<f32>,
    /// `[frequency_bins, mel_bins]`, as transformers stores `mel_filters`.
    filters: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl std::fmt::Debug for LogMelExtractor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LogMelExtractor")
            .field("spec", &self.spec)
            .finish_non_exhaustive()
    }
}

impl LogMelExtractor {
    /// Builds the window, filters and FFT plan for `spec`.
    #[must_use]
    pub fn new(spec: LogMelSpec) -> Self {
        Self {
            spec,
            window: periodic_hann(spec.n_fft),
            filters: slaney_mel_filters(spec),
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(spec.n_fft),
        }
    }

    /// The spec this extractor implements.
    #[must_use]
    pub const fn spec(&self) -> LogMelSpec {
        self.spec
    }

    /// Computes features for one clip.
    ///
    /// # Errors
    ///
    /// Returns [`LogMelError`] when the rate differs from the spec or the clip
    /// is shorter than one hop or half a window.
    ///
    /// # Panics
    ///
    /// Never: the FFT buffers are made by the same plan, so their lengths
    /// always match.
    pub fn extract(&self, audio: &MonoAudio) -> Result<LogMel, LogMelError> {
        let spec = self.spec;
        if audio.sample_rate() != spec.sample_rate {
            return Err(LogMelError::SampleRate {
                actual: audio.sample_rate(),
                expected: spec.sample_rate,
            });
        }
        let samples = audio.samples();
        let pad = spec.n_fft / 2;
        let minimum = (pad + 1).max(spec.hop_length);
        if samples.len() < minimum {
            return Err(LogMelError::TooShort {
                actual: samples.len(),
                minimum,
            });
        }
        let padded = reflect_pad(samples, pad);
        let frames = spec.frames_for(samples.len());
        let frequency_bins = spec.n_fft / 2 + 1;

        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let mut power = vec![0.0_f32; frequency_bins];
        let mut values = vec![0.0_f32; spec.mel_bins * frames];
        for frame in 0..frames {
            let start = frame * spec.hop_length;
            for ((slot, sample), weight) in input
                .iter_mut()
                .zip(&padded[start..start + spec.n_fft])
                .zip(&self.window)
            {
                *slot = sample * weight;
            }
            self.fft
                .process_with_scratch(&mut input, &mut spectrum, &mut scratch)
                .expect("buffers come from this plan");
            for (bin, value) in power.iter_mut().zip(&spectrum) {
                // torch computes `abs()` (a hypot) and then squares it.
                let magnitude = value.re.hypot(value.im);
                *bin = magnitude * magnitude;
            }
            for mel in 0..spec.mel_bins {
                let energy: f32 = power
                    .iter()
                    .enumerate()
                    .map(|(bin, power)| self.filters[bin * spec.mel_bins + mel] * power)
                    .sum();
                values[mel * frames + frame] = energy.max(1e-10).log10();
            }
        }
        let floor = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 8.0;
        for value in &mut values {
            *value = (value.max(floor) + 4.0) / 4.0;
        }
        Ok(LogMel {
            bins: spec.mel_bins,
            frames,
            values,
        })
    }
}

/// `torch.hann_window(n)`, which is periodic: `0.5 - 0.5 cos(2 pi k / n)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "window lengths are small"
)]
fn periodic_hann(length: usize) -> Vec<f32> {
    (0..length)
        .map(|index| {
            (0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / length as f64).cos()) as f32
        })
        .collect()
}

/// `torch.stft(center=True)`'s reflect padding: `pad` mirrored samples on each
/// side, excluding the edge sample itself.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let last = samples.len() - 1;
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|offset| samples[offset]));
    padded.extend_from_slice(samples);
    padded.extend((1..=pad).map(|offset| samples[last - offset]));
    padded
}

fn hertz_to_slaney_mel(hertz: f64) -> f64 {
    let log_step = 27.0 / 6.4_f64.ln();
    if hertz >= 1_000.0 {
        15.0 + (hertz / 1_000.0).ln() * log_step
    } else {
        3.0 * hertz / 200.0
    }
}

fn slaney_mel_to_hertz(mel: f64) -> f64 {
    let log_step = 6.4_f64.ln() / 27.0;
    if mel >= 15.0 {
        1_000.0 * (log_step * (mel - 15.0)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// `numpy.linspace(start, stop, count)`: `start + index * step`, with the last
/// value set to `stop` exactly.
#[allow(clippy::cast_precision_loss, reason = "counts are small")]
fn linspace(start: f64, stop: f64, count: usize) -> Vec<f64> {
    let increment = (stop - start) / (count - 1) as f64;
    let mut values: Vec<f64> = (0..count)
        .map(|index| start + index as f64 * increment)
        .collect();
    values[count - 1] = stop;
    values
}

/// transformers `mel_filter_bank(num_frequency_bins = n_fft / 2 + 1,
/// min_frequency = 0, norm = "slaney", mel_scale = "slaney")`, flattened
/// `[frequency_bins, mel_bins]` and narrowed to f32.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the reference casts the f64 filters to f32 the same way"
)]
fn slaney_mel_filters(spec: LogMelSpec) -> Vec<f32> {
    let frequency_bins = spec.n_fft / 2 + 1;
    let mel_points = linspace(
        hertz_to_slaney_mel(0.0),
        hertz_to_slaney_mel(spec.max_frequency),
        spec.mel_bins + 2,
    );
    let filter_hertz: Vec<f64> = mel_points.into_iter().map(slaney_mel_to_hertz).collect();
    let fft_hertz = linspace(0.0, f64::from(spec.sample_rate / 2), frequency_bins);
    let mut filters = vec![0.0_f32; frequency_bins * spec.mel_bins];
    for (bin, frequency) in fft_hertz.iter().enumerate() {
        for mel in 0..spec.mel_bins {
            let down =
                -(filter_hertz[mel] - frequency) / (filter_hertz[mel + 1] - filter_hertz[mel]);
            let up = (filter_hertz[mel + 2] - frequency)
                / (filter_hertz[mel + 2] - filter_hertz[mel + 1]);
            let triangle = down.min(up).max(0.0);
            let area = 2.0 / (filter_hertz[mel + 2] - filter_hertz[mel]);
            filters[bin * spec.mel_bins + mel] = (triangle * area) as f32;
        }
    }
    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_scale_round_trips_and_switches_to_log_at_1_khz() {
        for hertz in [0.0, 200.0, 999.0, 1_000.0, 4_000.0, 8_000.0] {
            let back = slaney_mel_to_hertz(hertz_to_slaney_mel(hertz));
            assert!((back - hertz).abs() < 1e-9, "{hertz} -> {back}");
        }
        assert!((hertz_to_slaney_mel(1_000.0) - 15.0).abs() < 1e-12);
    }

    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        assert_eq!(
            reflect_pad(&[1.0, 2.0, 3.0, 4.0, 5.0], 2),
            [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]
        );
    }

    #[test]
    fn frame_count_drops_the_final_centered_frame() {
        let spec = LogMelSpec::QWEN3_ASR;
        assert_eq!(spec.frames_for(16_000), 100);
        assert_eq!(spec.frames_for(16_159), 100);
        assert_eq!(spec.frames_for(16_160), 101);
        let extractor = LogMelExtractor::new(spec);
        let audio = MonoAudio::new(16_000, vec![0.25; 16_159]).unwrap();
        assert_eq!(extractor.extract(&audio).unwrap().frames(), 100);
    }

    #[test]
    fn refuses_the_wrong_rate_and_too_short_clips() {
        let extractor = LogMelExtractor::new(LogMelSpec::QWEN3_ASR);
        let audio = MonoAudio::new(44_100, vec![0.0; 44_100]).unwrap();
        assert_eq!(
            extractor.extract(&audio),
            Err(LogMelError::SampleRate {
                actual: 44_100,
                expected: 16_000
            })
        );
        let audio = MonoAudio::new(16_000, vec![0.0; 200]).unwrap();
        assert_eq!(
            extractor.extract(&audio),
            Err(LogMelError::TooShort {
                actual: 200,
                minimum: 201
            })
        );
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "every step is exact for silence")]
    fn silence_sits_at_the_floor() {
        // Every bin is log10(1e-10) = -10, so the floor (max - 8) is below
        // every value and the output is (-10 + 4) / 4.
        let extractor = LogMelExtractor::new(LogMelSpec::QWEN3_ASR);
        let mel = extractor
            .extract(&MonoAudio::new(16_000, vec![0.0; 16_000]).unwrap())
            .unwrap();
        assert!(mel.values().iter().all(|&value| value == -1.5));
    }

    #[test]
    fn a_tone_peaks_in_the_mel_bin_containing_its_frequency() {
        // A 1 kHz tone's energy must land in the bin whose triangle peaks
        // nearest 1 kHz, the Slaney scale's linear/log knee.
        let spec = LogMelSpec::QWEN3_ASR;
        #[allow(clippy::cast_precision_loss, reason = "short test signal")]
        let tone: Vec<f32> = (0..16_000)
            .map(|index| (std::f32::consts::TAU * 1_000.0 * index as f32 / 16_000.0).sin())
            .collect();
        let mel = LogMelExtractor::new(spec)
            .extract(&MonoAudio::new(16_000, tone).unwrap())
            .unwrap();
        let frame = 50;
        let loudest = (0..mel.bins())
            .max_by(|&a, &b| {
                mel.values()[a * mel.frames() + frame]
                    .total_cmp(&mel.values()[b * mel.frames() + frame])
            })
            .unwrap();
        let peaks = linspace(
            hertz_to_slaney_mel(0.0),
            hertz_to_slaney_mel(8_000.0),
            spec.mel_bins() + 2,
        );
        let nearest = (0..spec.mel_bins())
            .min_by(|&a, &b| {
                (slaney_mel_to_hertz(peaks[a + 1]) - 1_000.0)
                    .abs()
                    .total_cmp(&(slaney_mel_to_hertz(peaks[b + 1]) - 1_000.0).abs())
            })
            .unwrap();
        assert_eq!(loudest, nearest);
    }
}
