//! Local-only raw-token generation for the scalar V4.1 model facade.

use std::{
    num::{NonZeroU32, NonZeroU64},
    path::PathBuf,
    process::ExitCode,
};

use clap::Args;
use deepseek::{TokenId, V41Model, V41ModelOptions};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum Backend {
    /// CPU scalar reference execution. Metal generation is not supported here.
    #[default]
    Scalar,
}

#[derive(Debug, Args)]
pub(crate) struct DeepseekGenerateArgs {
    /// Portable local checkpoint directory; never downloads missing weights.
    #[arg(long, value_name = "DIR")]
    weights: PathBuf,
    /// At least two comma-separated vocabulary IDs; no tokenizer or template is applied.
    #[arg(long, value_delimiter = ',', required = true)]
    input_ids: Vec<u32>,
    /// Maximum number of greedy output IDs, including a selected EOS.
    #[arg(long, default_value_t = 32, value_parser = clap::value_parser!(u32).range(1..=256))]
    max_tokens: u32,
    /// Maximum evaluated positions: prompt plus generated IDs fed back to the model.
    /// The final predicted output ID does not consume another context position.
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..=32768))]
    context_tokens: u32,
    /// Admitted resident weight bytes in MiB; not a total process memory cap.
    #[arg(long, default_value_t = 12288, value_parser = clap::value_parser!(u32).range(1..))]
    resident_mib: u32,
    /// Local range-cache capacity in MiB.
    #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(1..))]
    cache_mib: u32,
    /// Stop when this ID is selected; no EOS is inferred from configuration.
    #[arg(long)]
    eos_token_id: Option<u32>,
    /// Execution backend; only scalar generation is currently supported.
    #[arg(long, value_enum, default_value_t = Backend::Scalar)]
    backend: Backend,
    /// Emit one structured raw-token result after successful generation.
    #[arg(long)]
    json: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "snake_case")]
enum StopReason {
    MaxTokens,
    Eos,
}

#[derive(Serialize)]
struct Generation {
    backend: &'static str,
    prompt_tokens: usize,
    token_ids: Vec<u32>,
    stop_reason: StopReason,
}

impl DeepseekGenerateArgs {
    pub(crate) fn run(self) -> ExitCode {
        match generate(&self) {
            Ok(result) => {
                if self.json {
                    match serde_json::to_string(&result) {
                        Ok(json) => println!("{json}"),
                        Err(error) => {
                            eprintln!("DeepSeek result serialization failed: {error}");
                            return ExitCode::FAILURE;
                        }
                    }
                } else {
                    println!(
                        "{}",
                        result
                            .token_ids
                            .iter()
                            .map(u32::to_string)
                            .collect::<Vec<_>>()
                            .join(",")
                    );
                }
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("DeepSeek generation failed: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn validate_request(args: &DeepseekGenerateArgs) -> Result<(), String> {
    if args.input_ids.len() < 2 {
        return Err("initial prompt requires at least two input IDs".to_owned());
    }
    let prompt = u32::try_from(args.input_ids.len()).map_err(|_| "too many input IDs")?;
    let decode_positions = args
        .max_tokens
        .checked_sub(1)
        .ok_or("max-tokens must be positive")?;
    if prompt
        .checked_add(decode_positions)
        .is_none_or(|total| total > args.context_tokens)
    {
        return Err("prompt plus evaluated decode positions exceeds context-tokens".to_owned());
    }
    Ok(())
}

fn generate(args: &DeepseekGenerateArgs) -> Result<Generation, String> {
    validate_request(args)?;
    let mut options = V41ModelOptions::default();
    options.context_tokens =
        NonZeroU32::new(args.context_tokens).ok_or("context-tokens must be positive")?;
    options.resident_bytes = NonZeroU64::new(u64::from(args.resident_mib) * (1 << 20))
        .ok_or("resident-mib must be positive")?;
    options.cache_bytes = NonZeroU64::new(u64::from(args.cache_mib) * (1 << 20))
        .ok_or("cache-mib must be positive")?;
    match args.backend {
        Backend::Scalar => options.backend = deepseek::V41Backend::Scalar,
    }
    let metadata = V41Model::inspect_store(&args.weights, &options)
        .map_err(|error| format!("local store metadata admission failed: {error}"))?;
    if args
        .input_ids
        .iter()
        .chain(args.eos_token_id.iter())
        .any(|id| u64::from(*id) >= metadata.vocab_size() as u64)
    {
        return Err("input or EOS token ID is outside the checkpoint vocabulary".to_owned());
    }
    let model = V41Model::open(&args.weights, options).map_err(|error| error.to_string())?;
    let mut session = model.session().map_err(|error| error.to_string())?;
    let prompt = args
        .input_ids
        .iter()
        .copied()
        .map(TokenId::new)
        .collect::<Vec<_>>();
    let mut logits = session.append(&prompt).map_err(|error| error.to_string())?;
    let mut token_ids = Vec::with_capacity(args.max_tokens as usize);
    let mut stop_reason = StopReason::MaxTokens;
    for index in 0..args.max_tokens {
        let next = greedy_token(&logits)?;
        token_ids.push(next);
        if Some(next) == args.eos_token_id {
            stop_reason = StopReason::Eos;
            break;
        }
        if index + 1 < args.max_tokens {
            logits = session
                .append(&[TokenId::new(next)])
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(Generation {
        backend: "scalar",
        prompt_tokens: prompt.len(),
        token_ids,
        stop_reason,
    })
}

fn greedy_token(logits: &[f32]) -> Result<u32, String> {
    let mut best = None;
    for (index, value) in logits.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err("model returned non-finite logits".to_owned());
        }
        if best.is_none_or(|(_, score)| value > score) {
            best = Some((index, value));
        }
    }
    let (index, _) = best.ok_or("model returned empty logits")?;
    u32::try_from(index).map_err(|_| "selected token ID exceeds u32".to_owned())
}

#[cfg(test)]
mod tests {
    use super::greedy_token;

    #[test]
    fn greedy_ties_choose_the_first_id_and_invalid_logits_fail() {
        assert_eq!(greedy_token(&[-2.0, 3.0, 3.0]), Ok(1));
        assert!(greedy_token(&[]).is_err());
        assert!(greedy_token(&[0.0, f32::NAN]).is_err());
        assert!(greedy_token(&[f32::INFINITY]).is_err());
    }
}
