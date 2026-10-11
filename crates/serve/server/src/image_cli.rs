//! `mx image generate`: one FLUX.2 [klein] text-to-image run to a PNG file.
//!
//! The text encoder runs first and is dropped before the transformer and VAE
//! load, so the two never hold memory together. The same prompt, size,
//! steps, seed, precision and checkpoint give the same PNG bytes.

use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use clap::{Args, ValueEnum};
use flux2::{
    Flux2Pipeline, Flux2Precision, ImageSize, InitialNoise, KleinTextEncoder, klein_prompt_text,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use tokenizers::Tokenizer;

/// Exit status when MLX memory passes `--abort-gpu-gib`.
const EXIT_MEMORY_CAP: u8 = 3;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Precision {
    Bf16,
    F32,
}

#[derive(Debug, Args)]
pub(crate) struct ImageGenerateArgs {
    /// Local diffusers `Flux2KleinPipeline` directory (step-distilled klein).
    #[arg(long)]
    model: PathBuf,
    /// Prompt text, rendered with the klein chat template.
    #[arg(long)]
    prompt: String,
    /// Output size as `WIDTHxHEIGHT`; each side a multiple of 16 in [256, 2048].
    #[arg(long, default_value = "1024x1024", value_parser = parse_size)]
    size: ImageSize,
    /// Denoising steps; the distilled checkpoints are trained for 4.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=64))]
    steps: u32,
    /// Noise seed. Reported in the receipt; drawn from the clock when absent.
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, value_enum, default_value_t = Precision::Bf16)]
    precision: Precision,
    /// PNG file to write.
    #[arg(long)]
    out: PathBuf,
    /// Stop (exit 3) when MLX active plus cached memory passes this many GiB.
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..=120))]
    abort_gpu_gib: u32,
    /// Print a JSON receipt (inputs, digests, timings, memory) to stdout.
    #[arg(long)]
    json: bool,
}

fn parse_size(value: &str) -> Result<ImageSize, String> {
    let (width, height) = value
        .split_once('x')
        .ok_or_else(|| format!("size {value:?} is not WIDTHxHEIGHT"))?;
    let side = |text: &str| {
        text.parse::<u32>()
            .map_err(|_| format!("size {value:?} is not WIDTHxHEIGHT"))
    };
    ImageSize::new(side(width)?, side(height)?).map_err(|error| error.to_string())
}

fn mlx_bytes() -> (usize, usize, usize) {
    (
        mlx_rs::memory::active_memory().unwrap_or(0),
        mlx_rs::memory::cache_memory().unwrap_or(0),
        mlx_rs::memory::peak_memory().unwrap_or(0),
    )
}

fn milliseconds(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

impl ImageGenerateArgs {
    pub(crate) fn run(self) -> ExitCode {
        match self.generate() {
            Ok(receipt) => {
                if self.json {
                    println!("{receipt}");
                } else {
                    eprintln!("wrote {}", self.out.display());
                }
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("mx image generate: {error}");
                ExitCode::FAILURE
            }
        }
    }

    fn generate(&self) -> Result<serde_json::Value, String> {
        let precision = match self.precision {
            Precision::Bf16 => Flux2Precision::Bf16,
            Precision::F32 => Flux2Precision::F32,
        };
        let seed = self.seed.unwrap_or_else(clock_seed);
        let cap_bytes = usize::try_from(u64::from(self.abort_gpu_gib) << 30).unwrap_or(usize::MAX);
        let check_cap = |phase: &str| {
            let (active, cache, _) = mlx_bytes();
            if active + cache > cap_bytes {
                eprintln!(
                    "mx image generate: MLX memory {} GiB after {phase} passed --abort-gpu-gib {}",
                    (active + cache) >> 30,
                    self.abort_gpu_gib
                );
                std::process::exit(i32::from(EXIT_MEMORY_CAP));
            }
        };

        let ids = tokenize(&self.model, &self.prompt)?;
        let started = Instant::now();
        let text = {
            let encoder = KleinTextEncoder::load(self.model.join("text_encoder"), precision)
                .map_err(|error| error.to_string())?;
            let text = encoder.encode(&ids).map_err(|error| error.to_string())?;
            text.eval().map_err(|error| error.to_string())?;
            text
        };
        // The encoder's weights are gone; release their cached buffers too.
        mlx_rs::memory::clear_cache().map_err(|error| error.to_string())?;
        let text_encode_ms = milliseconds(started);
        check_cap("text encoding");

        let pipeline =
            Flux2Pipeline::load(&self.model, precision).map_err(|error| error.to_string())?;
        let initial = pipeline
            .initial_latents(InitialNoise::Seed(seed), self.size)
            .map_err(|error| error.to_string())?;
        let mut step_ms = Vec::new();
        let mut step_started = Instant::now();
        let packed = pipeline
            .denoise(
                &text,
                &initial,
                self.size,
                self.steps as usize,
                |index, _| {
                    step_ms.push(milliseconds(step_started));
                    check_cap(&format!("step {index}"));
                    step_started = Instant::now();
                },
            )
            .map_err(|error| error.to_string())?;
        let started = Instant::now();
        let pixels = pipeline
            .decode_rgb8(&packed, self.size)
            .map_err(|error| error.to_string())?;
        let decode_ms = milliseconds(started);
        check_cap("decode");

        write_png(&self.out, self.size, &pixels)?;
        let png = std::fs::read(&self.out).map_err(|error| error.to_string())?;
        let (active, cache, peak) = mlx_bytes();
        Ok(json!({
            "model": self.model,
            "prompt_sha256": format!("{:x}", Sha256::digest(self.prompt.as_bytes())),
            "prompt_token_ids": ids,
            "width": self.size.width(),
            "height": self.size.height(),
            "steps": self.steps,
            "seed": seed,
            "precision": format!("{:?}", self.precision).to_lowercase(),
            "out": self.out,
            "png_sha256": format!("{:x}", Sha256::digest(&png)),
            "timings_ms": {
                "text_encode": text_encode_ms,
                "steps": step_ms,
                "decode": decode_ms,
            },
            "mlx_bytes": { "active": active, "cache": cache, "peak": peak },
        }))
    }
}

/// Token ids of the rendered prompt, as the pipeline's tokenizer call gives
/// them before padding (Qwen's tokenizer adds no special tokens of its own).
fn tokenize(model: &Path, prompt: &str) -> Result<Vec<i32>, String> {
    let path = model.join("tokenizer").join("tokenizer.json");
    let bytes = std::fs::read(&path)
        .map_err(|error| format!("{} could not be read: {error}", path.display()))?;
    let mut tokenizer = Tokenizer::from_bytes(&bytes)
        .map_err(|error| format!("{} could not be parsed: {error}", path.display()))?;
    tokenizer
        .with_truncation(None)
        .map_err(|_| String::from("tokenizer truncation could not be disabled"))?;
    tokenizer.with_padding(None);
    let encoding = tokenizer
        .encode(klein_prompt_text(prompt), false)
        .map_err(|error| format!("prompt could not be tokenized: {error}"))?;
    encoding
        .get_ids()
        .iter()
        .map(|&id| i32::try_from(id).map_err(|_| format!("token id {id} exceeds i32")))
        .collect()
}

fn clock_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| {
            u64::try_from(elapsed.as_nanos() & u128::from(u64::MAX)).unwrap_or(0)
        })
}

/// 8-bit RGB PNG with no timestamp or text chunks, so equal pixels give
/// equal bytes.
fn write_png(path: &Path, size: ImageSize, pixels: &[u8]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), size.width(), size.height());
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    writer
        .write_image_data(pixels)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    writer
        .finish()
        .map_err(|error| format!("{}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use crate::{Cli, Command, cli::ImageCommand};

    #[test]
    fn sizes_must_fit_the_latent_grid() {
        let parse = |size: &str| {
            Cli::try_parse_from([
                "mx", "image", "generate", "--model", "m", "--prompt", "p", "--out", "o.png",
                "--size", size,
            ])
        };
        let parsed = parse("768x512").expect("aligned size");
        let Command::Image {
            command: ImageCommand::Generate(args),
        } = parsed.command
        else {
            panic!("wrong command")
        };
        assert_eq!((args.size.width(), args.size.height()), (768, 512));
        assert_eq!(args.steps, 4);
        assert!(parse("1000x1000").is_err());
        assert!(parse("1024").is_err());
    }
}
