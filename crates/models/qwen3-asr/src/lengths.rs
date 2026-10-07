//! How mel frames map to encoder chunks, frames and attention windows.

use crate::config::{CHUNK_FRAMES, FRAMES_PER_CHUNK};

/// The reference `_get_feat_extract_output_lengths`: 13 encoder frames per
/// full 100-frame chunk, and `ceil(rest / 8)` for a shorter final chunk.
#[must_use]
pub const fn encoder_frames(mel_frames: usize) -> usize {
    (mel_frames / CHUNK_FRAMES) * FRAMES_PER_CHUNK + (mel_frames % CHUNK_FRAMES).div_ceil(8)
}

/// Mel-frame lengths of the chunks one clip is split into: full
/// [`CHUNK_FRAMES`] chunks, then the remainder if there is one.
#[must_use]
pub fn chunk_lengths(mel_frames: usize) -> Vec<usize> {
    let mut lengths = vec![CHUNK_FRAMES; mel_frames / CHUNK_FRAMES];
    if !mel_frames.is_multiple_of(CHUNK_FRAMES) {
        lengths.push(mel_frames % CHUNK_FRAMES);
    }
    lengths
}

/// Encoder-frame lengths of the attention windows: full windows of
/// `window_frames`, then the remainder.
///
/// The reference sizes a window as the longest chunk's encoder frames times
/// the window's chunk count, so a clip shorter than one chunk gets one
/// window, which covers it whole either way.
#[must_use]
pub fn window_lengths(encoder_frames: usize, window_frames: usize) -> Vec<usize> {
    let mut lengths = vec![window_frames; encoder_frames / window_frames];
    if !encoder_frames.is_multiple_of(window_frames) {
        lengths.push(encoder_frames % window_frames);
    }
    lengths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_reference_length_formula() {
        // Values of the Python formula, floor division included.
        let expected = [
            (1, 1),
            (7, 1),
            (8, 1),
            (9, 2),
            (99, 13),
            (100, 13),
            (101, 14),
            (108, 14),
            (109, 15),
            (199, 26),
            (200, 26),
            (250, 33),
            (3000, 390),
            (3001, 391),
            (120_000, 15_600),
        ];
        for (mel, frames) in expected {
            assert_eq!(encoder_frames(mel), frames, "{mel} mel frames");
        }
    }

    #[test]
    fn chunks_and_windows_partition_the_clip() {
        assert_eq!(chunk_lengths(250), [100, 100, 50]);
        assert_eq!(chunk_lengths(200), [100, 100]);
        assert_eq!(chunk_lengths(42), [42]);
        let frames = encoder_frames(1_234);
        assert_eq!(
            chunk_lengths(1_234)
                .iter()
                .map(|&length| encoder_frames(length))
                .sum::<usize>(),
            frames
        );
        assert_eq!(window_lengths(frames, 104), [104, 57]);
        assert_eq!(window_lengths(208, 104), [104, 104]);
    }
}
