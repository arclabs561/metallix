//! Direct, typed multiple-choice readout from one resident Qwen checkpoint.
//!
//! This intentionally performs no decode.  Each question is independently
//! rendered through the checkpoint chat template and scores only its fixed
//! answer-letter tokens from the final prefill logits.

use std::{collections::BTreeMap, fs::File, io::Read, path::Path, process::ExitCode};

use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::chat_generation::{ChatMessage, ChatRole, ChatSession, ResidentChatLimits};

const MAX_REQUEST_BYTES: usize = 1024 * 1024;
const MAX_QUESTIONS: usize = 16;
const MIN_OPTIONS: usize = 2;
const MAX_OPTIONS: usize = 16;
const LABELS: [&str; MAX_OPTIONS] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
];

/// Evaluates named, typed decisions using one loaded resident Qwen model.
pub(crate) fn decide(
    model: &Path,
    request: &Path,
    temperature: f64,
    limits: ResidentChatLimits,
) -> ExitCode {
    match decide_inner(model, request, temperature, limits) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mx decide: {error}");
            ExitCode::FAILURE
        }
    }
}

fn decide_inner(
    model: &Path,
    request_path: &Path,
    temperature: f64,
    limits: ResidentChatLimits,
) -> Result<(), String> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(String::from(
            "decision temperature must be finite and greater than zero",
        ));
    }
    let request_bytes = read_request(request_path)?;
    let request: DecisionRequest = serde_json::from_slice(&request_bytes)
        .map_err(|error| format!("decision request JSON could not be parsed: {error}"))?;
    validate_request(&request)?;

    let mut session = ChatSession::load(model, limits)?;
    let maximum_options = request
        .questions
        .values()
        .map(DecisionQuestion::options)
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .map(Vec::len)
        .max()
        .ok_or_else(|| String::from("decision request needs questions"))?;
    let label_tokens = answer_tokens(&session, maximum_options)?;
    let mut answers = serde_json::Map::new();
    let mut total_input_tokens = 0_usize;
    for (name, question) in &request.questions {
        let options = question.options()?;
        let prompt = prompt(&request.state, question, &options)?;
        let prefill = session.prefill_chat(&[ChatMessage::text(ChatRole::User, prompt)])?;
        total_input_tokens = total_input_tokens
            .checked_add(prefill.prompt_tokens)
            .ok_or_else(|| String::from("decision input token count overflows"))?;
        let probabilities =
            option_probabilities(&prefill.logits, &label_tokens[..options.len()], temperature)?;
        answers.insert(
            name.clone(),
            answer_receipt(question, &options, &label_tokens, &probabilities, &prefill)?,
        );
    }
    let provenance = provenance(&request_bytes, &session);
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema_version": 1,
            "operation": "qwen_typed_decision_prefill",
            "model": model.display().to_string(),
            "backend": "mlx-rs 0.25.3 Metal float32",
            "session_load_ms": session.load_ms(),
            "calibration": {
                "status": "uncalibrated",
                "method": "temperature_scaled_option_softmax",
                "temperature": temperature,
                "note": "These are normalized probabilities over supplied answer letters, not calibrated confidence. External fitting and held-out validation are required for calibration claims."
            },
            "answers": answers,
            "usage": {"input_tokens": total_input_tokens, "output_tokens": 0},
            "provenance": provenance,
        })).map_err(|error| error.to_string())?
    );
    Ok(())
}

#[derive(Debug, Deserialize)]
struct DecisionRequest {
    state: Value,
    questions: BTreeMap<String, DecisionQuestion>,
}

#[derive(Debug, Deserialize)]
struct DecisionQuestion {
    instructions: String,
    #[serde(flatten)]
    kind: DecisionKind,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DecisionKind {
    Choice {
        criteria: BTreeMap<String, String>,
    },
    Noul {
        #[serde(default)]
        criteria: BTreeMap<String, String>,
    },
    Score {
        criteria: Vec<String>,
    },
}

#[derive(Clone, Debug)]
struct OptionEntry {
    id: String,
    description: String,
}

impl DecisionQuestion {
    fn options(&self) -> Result<Vec<OptionEntry>, String> {
        let options = match &self.kind {
            DecisionKind::Choice { criteria } => {
                if criteria
                    .values()
                    .any(|description| description.trim().is_empty())
                {
                    return Err(String::from(
                        "choice decision criteria descriptions must be nonempty",
                    ));
                }
                criteria
                    .iter()
                    .map(|(id, description)| OptionEntry {
                        id: id.clone(),
                        description: description.clone(),
                    })
                    .collect()
            }
            DecisionKind::Noul { criteria } => vec![
                OptionEntry {
                    id: String::from("false"),
                    description: criteria.get("false").cloned().unwrap_or_default(),
                },
                OptionEntry {
                    id: String::from("true"),
                    description: criteria.get("true").cloned().unwrap_or_default(),
                },
            ],
            DecisionKind::Score { criteria } => {
                if criteria
                    .iter()
                    .any(|description| description.trim().is_empty())
                {
                    return Err(String::from(
                        "score decision criteria descriptions must be nonempty",
                    ));
                }
                criteria
                    .iter()
                    .enumerate()
                    .map(|(index, description)| OptionEntry {
                        id: index.to_string(),
                        description: description.clone(),
                    })
                    .collect()
            }
        };
        validate_options(&options)?;
        Ok(options)
    }
}

fn validate_request(request: &DecisionRequest) -> Result<(), String> {
    if !matches!(
        &request.state,
        Value::String(_) | Value::Array(_) | Value::Object(_)
    ) {
        return Err(String::from(
            "decision state must be a string, JSON array, or JSON object",
        ));
    }
    if request.questions.is_empty() || request.questions.len() > MAX_QUESTIONS {
        return Err(format!(
            "decision request requires 1 through {MAX_QUESTIONS} questions; received {}",
            request.questions.len()
        ));
    }
    for (name, question) in &request.questions {
        if name.is_empty() || name.len() > 256 {
            return Err(String::from(
                "decision question names must contain 1 through 256 bytes",
            ));
        }
        if question.instructions.trim().is_empty() {
            return Err(format!("decision question {name:?} has empty instructions"));
        }
        if let DecisionKind::Noul { criteria } = &question.kind {
            if criteria.keys().any(|key| key != "false" && key != "true") {
                return Err(format!(
                    "noul decision question {name:?} criteria may only name false and true"
                ));
            }
        }
        question.options()?;
    }
    Ok(())
}

fn validate_options(options: &[OptionEntry]) -> Result<(), String> {
    if !(MIN_OPTIONS..=MAX_OPTIONS).contains(&options.len()) {
        return Err(format!(
            "decision criteria requires {MIN_OPTIONS} through {MAX_OPTIONS} options; received {}",
            options.len()
        ));
    }
    if options
        .iter()
        .any(|option| option.id.is_empty() || option.description.len() > MAX_REQUEST_BYTES)
    {
        return Err(String::from(
            "decision option IDs must be nonempty and descriptions bounded",
        ));
    }
    Ok(())
}

fn prompt(
    state: &Value,
    question: &DecisionQuestion,
    options: &[OptionEntry],
) -> Result<String, String> {
    let state = serde_json::to_string(state)
        .map_err(|_| String::from("decision state could not be serialized"))?;
    let mut output = format!("{}\n\nState:\n{state}\n\n", question.instructions);
    output.push_str("Choose exactly one listed answer. Reply with its letter only.\n");
    for (index, option) in options.iter().enumerate() {
        output.push_str(LABELS[index]);
        output.push_str(". ");
        output.push_str(&option.id);
        if !option.description.is_empty() {
            output.push_str(": ");
            output.push_str(&option.description);
        }
        output.push('\n');
    }
    if output.len() > MAX_REQUEST_BYTES {
        return Err(String::from(
            "rendered decision instruction exceeds the 1048576-byte limit",
        ));
    }
    Ok(output)
}

fn answer_tokens(session: &ChatSession, count: usize) -> Result<Vec<i32>, String> {
    let tokens = LABELS[..count]
        .iter()
        .map(|label| session.answer_label_token(label))
        .collect::<Result<Vec<_>, _>>()?;
    for (left, token) in tokens.iter().enumerate() {
        if tokens[..left].contains(token) {
            return Err(String::from(
                "decision answer labels must map to distinct tokenizer tokens",
            ));
        }
    }
    Ok(tokens)
}

fn option_probabilities(
    logits: &[f32],
    tokens: &[i32],
    temperature: f64,
) -> Result<Vec<f64>, String> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(String::from(
            "decision prefill returned non-finite or empty logits",
        ));
    }
    let selected = tokens
        .iter()
        .map(|token| {
            let index = usize::try_from(*token)
                .map_err(|_| String::from("decision answer token ID is negative"))?;
            logits
                .get(index)
                .map(|value| f64::from(*value))
                .ok_or_else(|| String::from("decision answer token ID exceeds prefill vocabulary"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    softmax(&selected, temperature)
}

fn softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>, String> {
    if logits.is_empty() || !temperature.is_finite() || temperature <= 0.0 {
        return Err(String::from(
            "decision softmax requires finite logits and positive temperature",
        ));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(String::from("decision softmax received non-finite logits"));
    }
    let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // Center before temperature scaling. Tiny positive temperatures may
    // underflow nonmaximal mass to zero, but must not overflow the maximum.
    let weights = logits
        .iter()
        .map(|value| ((value - maximum) / temperature).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(String::from("decision softmax has no finite support"));
    }
    Ok(weights.into_iter().map(|weight| weight / sum).collect())
}

fn answer_receipt(
    question: &DecisionQuestion,
    options: &[OptionEntry],
    tokens: &[i32],
    probabilities: &[f64],
    prefill: &crate::chat_generation::ChatPrefill,
) -> Result<Value, String> {
    let option_order = options
        .iter()
        .map(|option| option.id.clone())
        .collect::<Vec<_>>();
    let option_tokens = options
        .iter()
        .zip(tokens)
        .enumerate()
        .map(|(index, (option, token_id))| {
            json!({"label": LABELS[index], "option": option.id, "token_id": token_id})
        })
        .collect::<Vec<_>>();
    let probability_map = options.iter().zip(probabilities).fold(
        serde_json::Map::new(),
        |mut map, (option, probability)| {
            map.insert(option.id.clone(), Value::from(*probability));
            map
        },
    );
    let mut result = serde_json::Map::new();
    result.insert(
        "type".into(),
        Value::String(
            match &question.kind {
                DecisionKind::Choice { .. } => "choice",
                DecisionKind::Noul { .. } => "noul",
                DecisionKind::Score { .. } => "score",
            }
            .into(),
        ),
    );
    result.insert("option_order".into(), json!(option_order));
    result.insert("option_tokens".into(), json!(option_tokens));
    result.insert("probabilities".into(), Value::Object(probability_map));
    result.insert("prompt_tokens".into(), json!(prefill.prompt_tokens));
    result.insert("render_ms".into(), json!(prefill.render_ms));
    result.insert("prefill_ms".into(), json!(prefill.prefill_ms));
    match &question.kind {
        DecisionKind::Choice { .. } => {
            let index = argmax(probabilities)?;
            let option = options
                .get(index)
                .ok_or("decision choice index is absent")?;
            result.insert("choice".into(), Value::String(option.id.clone()));
        }
        DecisionKind::Noul { .. } => {
            let value = probabilities
                .get(1)
                .copied()
                .filter(|value| value.is_finite())
                .ok_or("decision true probability is absent")?;
            result.insert("noul".into(), json!(value));
        }
        DecisionKind::Score { .. } => {
            let expected_index =
                probabilities
                    .iter()
                    .enumerate()
                    .try_fold(0.0, |total, (index, probability)| {
                        if probability.is_finite() {
                            let index = u32::try_from(index)
                                .map_err(|_| String::from("decision option index overflows"))?;
                            Ok(total + f64::from(index) * probability)
                        } else {
                            Err(String::from("decision probability is non-finite"))
                        }
                    })?;
            result.insert("score".into(), json!(expected_index));
        }
    }
    Ok(Value::Object(result))
}

fn argmax(items: &[f64]) -> Result<usize, String> {
    let mut best = None;
    for (index, &probability) in items.iter().enumerate() {
        if !probability.is_finite() {
            return Err(String::from("decision probability is non-finite"));
        }
        if best.is_none_or(|(_, current)| probability > current) {
            best = Some((index, probability));
        }
    }
    best.map(|(index, _)| index)
        .ok_or_else(|| String::from("decision choice needs probabilities"))
}

pub(crate) fn read_request(path: &Path) -> Result<Vec<u8>, String> {
    let file = File::open(path).map_err(|_| String::from("decision request must be readable"))?;
    let metadata = file
        .metadata()
        .map_err(|_| String::from("decision request must be readable"))?;
    if !metadata.is_file() || metadata.len() > MAX_REQUEST_BYTES as u64 {
        return Err(String::from(
            "decision request must be a regular file no larger than 1048576 bytes",
        ));
    }
    let capacity = usize::try_from(metadata.len()).unwrap_or(MAX_REQUEST_BYTES);
    let mut bytes = Vec::with_capacity(capacity);
    file.take((MAX_REQUEST_BYTES as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| String::from("decision request could not be read"))?;
    if bytes.len() > MAX_REQUEST_BYTES {
        return Err(String::from(
            "decision request exceeds the 1048576-byte limit",
        ));
    }
    Ok(bytes)
}

fn provenance(request_bytes: &[u8], session: &ChatSession) -> Value {
    json!({
        "request_sha256": format!("{:x}", Sha256::digest(request_bytes)),
        "config_json_sha256": session.config_sha256(),
        "tokenizer_json_sha256": session.tokenizer_sha256(),
        "template_sha256": session.template_sha256(),
        "model_identity_scope": "local model directory and configuration/tokenizer/template bytes; checkpoint weights are not fingerprinted",
        "template": {"add_generation_prompt": true, "enable_thinking": false},
    })
}

#[cfg(test)]
#[allow(
    clippy::float_cmp,
    reason = "exact boundary probabilities are intentional"
)]
mod tests {
    use std::collections::BTreeMap;

    use super::{DecisionKind, DecisionQuestion, DecisionRequest, softmax, validate_request};
    use serde_json::json;

    #[test]
    fn softmax_normalizes_extreme_logits_without_overflow() {
        let values = softmax(&[10_000.0, -10_000.0], 1.0).expect("finite softmax");
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], 0.0);
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn softmax_rejects_nonfinite_and_invalid_temperature() {
        assert!(softmax(&[f64::NAN, 0.0], 1.0).is_err());
        assert!(softmax(&[0.0], 0.0).is_err());
    }

    #[test]
    fn selected_logit_distribution_excludes_other_vocabulary_mass() {
        // Independent odds: selected logits ln(2), ln(6) imply [1/4,3/4],
        // even when an unrelated vocabulary token dominates the raw softmax.
        let logits = [2_f32.ln(), 6_f32.ln(), 100.0];
        let p = super::option_probabilities(&logits, &[0, 1], 1.0).unwrap();
        assert!((p[0] - 0.25).abs() < 1e-7);
        assert!((p[1] - 0.75).abs() < 1e-7);
        let cold = softmax(&[2.0, 6.0], f64::MIN_POSITIVE).unwrap();
        assert_eq!(cold, [0.0, 1.0]);
        assert!(super::option_probabilities(&logits, &[-1, 1], 1.0).is_err());
        assert!(super::option_probabilities(&logits, &[0, 3], 1.0).is_err());
    }

    #[test]
    fn ordinal_receipt_reports_expectation_not_argmax() {
        let question = DecisionQuestion {
            instructions: String::from("Rate severity"),
            kind: DecisionKind::Score {
                criteria: vec![String::from("low"), String::from("high")],
            },
        };
        let prefill = crate::chat_generation::ChatPrefill {
            logits: Vec::new(),
            prompt_tokens: 10,
            render_ms: 0.0,
            prefill_ms: 0.0,
        };
        let receipt = super::answer_receipt(
            &question,
            &question.options().unwrap(),
            &[32, 33],
            &[0.7, 0.3],
            &prefill,
        )
        .unwrap();
        assert!((receipt["score"].as_f64().unwrap() - 0.3).abs() < 1e-12);
        assert_eq!(receipt["option_order"], json!(["0", "1"]));
        assert_eq!(receipt["probabilities"], json!({"0":0.7,"1":0.3}));
    }

    #[test]
    fn choice_order_is_sorted_and_noul_is_false_then_true() {
        let request: DecisionRequest = serde_json::from_value(json!({
          "state": "fixture",
          "questions": {
            "z": {"type":"choice", "instructions":"x", "criteria":{"zeta":"z", "alpha":"a"}},
            "a": {"type":"noul", "instructions":"x", "criteria":{"true":"yes", "false":"no"}}
          }
        }))
        .expect("request parses");
        let names = request.questions.keys().cloned().collect::<Vec<_>>();
        assert_eq!(names, ["a", "z"]);
        assert_eq!(
            request.questions["z"]
                .options()
                .unwrap()
                .iter()
                .map(|x| x.id.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "zeta"]
        );
        assert_eq!(
            request.questions["a"]
                .options()
                .unwrap()
                .iter()
                .map(|x| x.id.as_str())
                .collect::<Vec<_>>(),
            ["false", "true"]
        );
    }

    #[test]
    fn request_rejects_empty_instructions_and_too_few_options() {
        let request = DecisionRequest {
            state: Value::Null,
            questions: BTreeMap::from([(
                "x".into(),
                DecisionQuestion {
                    instructions: String::new(),
                    kind: DecisionKind::Score {
                        criteria: vec![String::from("one")],
                    },
                },
            )]),
        };
        assert!(validate_request(&request).is_err());
    }

    use serde_json::Value;
}
