//! Transcribe a directory of WAV clips listed in a `manifest.json` (as
//! written by `scripts/qwen3-asr-reference.py prepare-clips`) and write one
//! JSON line per clip, for word-error-rate scoring against the reference.
//!
//! ```text
//! cargo run --release -p qwen3-asr --features metal --example transcribe_clips -- \
//!     <model dir> <clips dir> <out.jsonl> [--language English] [--float32] [--limit N]
//! ```
//!
//! With no clips directory present it prints a note and exits 0.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use qwen3_asr::{Qwen3Asr, TranscribeOptions};
use serde::Deserialize;

#[derive(Deserialize)]
struct Manifest {
    clips: Vec<Clip>,
}

#[derive(Deserialize)]
struct Clip {
    id: String,
    wav: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: transcribe_clips <model dir> <clips dir> <out.jsonl> \
                 [--language NAME] [--float32] [--limit N]";
    let (Some(model), Some(clips), Some(out)) = (args.next(), args.next(), args.next()) else {
        return Err(usage.into());
    };
    let (model, clips, out) = (
        PathBuf::from(model),
        PathBuf::from(clips),
        PathBuf::from(out),
    );
    let mut options = TranscribeOptions::default();
    let mut float32 = false;
    let mut limit = usize::MAX;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--language" => options.language = Some(args.next().ok_or(usage)?),
            "--float32" => float32 = true,
            "--limit" => limit = args.next().ok_or(usage)?.parse()?,
            _ => return Err(usage.into()),
        }
    }
    let manifest_path = clips.join("manifest.json");
    if !manifest_path.exists() {
        println!(
            "no {}; run scripts/qwen3-asr-reference.py prepare-clips first",
            manifest_path.display()
        );
        return Ok(());
    }
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(manifest_path)?)?;

    let started = Instant::now();
    let mut asr = Qwen3Asr::load(&model)?;
    if float32 {
        asr.convert(mlx_rs::Dtype::Float32)?;
    }
    println!("loaded in {:.2} s", started.elapsed().as_secs_f64());

    let mut writer = std::io::BufWriter::new(std::fs::File::create(&out)?);
    let (mut audio_seconds, mut wall_seconds) = (0.0_f64, 0.0_f64);
    for clip in manifest.clips.iter().take(limit) {
        let audio = audio::decode_wav(&std::fs::read(clips.join(&clip.wav))?)?;
        let started = Instant::now();
        let result = asr.transcribe(&audio, &options)?;
        let wall = started.elapsed().as_secs_f64();
        audio_seconds += result.audio_seconds;
        wall_seconds += wall;
        let record = serde_json::json!({
            "id": clip.id,
            "language": result.transcript.language,
            "text": result.transcript.text,
            "raw": result.raw,
            "generated_ids": result.generated,
            "stopped": result.stopped,
            "audio_seconds": result.audio_seconds,
            "prompt_tokens": result.prompt_tokens,
            "wall_seconds": wall,
            "features_seconds": result.features.as_secs_f64(),
            "encode_seconds": result.encode.as_secs_f64(),
            "prefill_seconds": result.prefill.as_secs_f64(),
            "decode_seconds": result.decode.as_secs_f64(),
        });
        writeln!(writer, "{record}")?;
    }
    writer.flush()?;
    println!(
        "{} clips, {audio_seconds:.1} s of audio in {wall_seconds:.2} s: real-time factor {:.4}",
        manifest.clips.len().min(limit),
        wall_seconds / audio_seconds
    );
    Ok(())
}
