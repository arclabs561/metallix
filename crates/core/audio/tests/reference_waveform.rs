//! Independent RIFF container fixture made with Python 3.14's standard-library
//! `wave` writer, PCM16 stereo at 16 kHz. Frames are (-32768, 32767),
//! (-16384, 8192), (0, 0), (32767, 32767), (-32768, -32768), (4096, -4096).
//! Expectations are exact power-of-two normalization and channel means.

use audio::{decode_wav, probe_wav};

#[test]
fn standard_library_wave_fixture_decodes_exactly() {
    let bytes = include_bytes!("fixtures/pcm16_stereo.wav");
    let info = probe_wav(bytes).unwrap();
    assert_eq!(
        (info.sample_rate, info.channels, info.frames),
        (16_000, 2, 6)
    );
    let audio = decode_wav(bytes).unwrap();
    assert_eq!(audio.sample_rate(), 16_000);
    let expected: [f32; 6] = [-1.0 / 65_536.0, -0.125, 0.0, 32_767.0 / 32_768.0, -1.0, 0.0];
    assert_eq!(
        audio
            .samples()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}
