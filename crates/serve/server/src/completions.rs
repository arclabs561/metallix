//! `OpenAI` legacy Completions (`CreateCompletionRequest` and
//! `CreateCompletionResponse` in <https://github.com/openai/openai-openapi>)
//! for scoring a raw prompt: `echo` with `logprobs` returns the log
//! probability of every prompt token, and `max_tokens` 1 at temperature 0
//! appends the greedy next token. This is the request lm-eval's
//! `local-completions` model sends for log-likelihood tasks.
//!
//! The prompt is encoded without a chat template and without added special
//! tokens; a spelled added token becomes that token. Longer generation from
//! a raw prompt is not supported yet.
//!
//! One deviation from the published schema: `token_logprobs[0]` and
//! `top_logprobs[0]` of an echoed prompt are `null`, because the first token
//! has no prefix to be scored after. The schema types those items as
//! numbers and objects; `OpenAI`'s service and lm-eval use `null` there.

use serde::{Deserialize, de::IgnoredAny};
use serde_json::{Map, Value, json};

use crate::{
    chat_generation::{
        ChatBackend, ChatGenerationError, ScoreError, ScoreRequest, ScoreResult, ScoreText,
    },
    http_transport::Connection,
    responses::{echo_request_id, json_response},
    scoring::{record_score_usage, token_text},
};

/// Most alternatives per token, from the schema's `logprobs` maximum.
const MAX_LOGPROBS: u64 = 5;

/// The schema's `max_tokens` default.
const DEFAULT_MAX_TOKENS: u32 = 16;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) model: String,
    prompt: Prompt,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    echo: Option<bool>,
    #[serde(default)]
    logprobs: Option<u64>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    best_of: Option<u32>,
    #[serde(default)]
    stop: Option<Value>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    logit_bias: Option<Map<String, Value>>,
    // Accepted and ignored: only greedy decoding runs here, so `top_p` and
    // `seed` change nothing, and nothing streams.
    #[serde(default, rename = "top_p")]
    _top_p: Option<IgnoredAny>,
    #[serde(default, rename = "seed")]
    _seed: Option<IgnoredAny>,
    #[serde(default, rename = "stream_options")]
    _stream_options: Option<IgnoredAny>,
    #[serde(default, rename = "user")]
    _user: Option<IgnoredAny>,
}

/// The schema's four prompt shapes; a batch must hold one prompt.
#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Text(String),
    Texts(Vec<String>),
    Ids(Vec<i32>),
    IdBatch(Vec<Vec<i32>>),
}

/// A validated request, ready for a worker.
pub(crate) struct Prepared {
    request: Request,
    echo: bool,
    /// 0 to score the echoed prompt, 1 to add the greedy next token.
    max_tokens: u32,
    /// Alternatives per token, when log probabilities were requested.
    logprobs: Option<usize>,
}

/// An `OpenAI`-shaped `400` body naming the offending field.
fn invalid(param: &str, message: &str) -> Value {
    json!({"error":{"message":message,"type":"invalid_request_error","param":param,"code":null}})
}

/// Parses and validates one body; the error is a ready `400` body.
pub(crate) fn prepare(body: &[u8]) -> Result<(String, Prepared), Value> {
    let request: Request = serde_json::from_slice(body).map_err(|error| {
        json!({"error":{"message":error.to_string(),"type":"invalid_request_error","param":null,"code":null}})
    })?;
    let prepared = validate(request)?;
    Ok((prepared.request.model.clone(), prepared))
}

fn validate(request: Request) -> Result<Prepared, Value> {
    if request.n.is_some_and(|n| n != 1) {
        return Err(invalid("n", "only n=1 is supported"));
    }
    if request.best_of.is_some_and(|best_of| best_of != 1) {
        return Err(invalid("best_of", "best_of is unsupported"));
    }
    if request.logprobs.is_some_and(|top| top > MAX_LOGPROBS) {
        return Err(invalid(
            "logprobs",
            &format!("logprobs must be between 0 and {MAX_LOGPROBS}"),
        ));
    }
    if request.stream == Some(true) {
        return Err(invalid("stream", "streaming is unsupported on this route"));
    }
    if request
        .suffix
        .as_deref()
        .is_some_and(|suffix| !suffix.is_empty())
    {
        return Err(invalid("suffix", "suffix is unsupported"));
    }
    let has_stops = match &request.stop {
        None | Some(Value::Null) => false,
        Some(Value::String(stop)) => !stop.is_empty(),
        Some(Value::Array(stops)) => !stops.is_empty(),
        Some(_) => true,
    };
    if has_stops {
        return Err(invalid("stop", "stop sequences are unsupported"));
    }
    for (param, penalty) in [
        ("frequency_penalty", request.frequency_penalty),
        ("presence_penalty", request.presence_penalty),
    ] {
        if penalty.is_some_and(|penalty| penalty != 0.0) {
            return Err(invalid(
                param,
                "frequency and presence penalties are unsupported",
            ));
        }
    }
    if request
        .logit_bias
        .as_ref()
        .is_some_and(|bias| !bias.is_empty())
    {
        return Err(invalid("logit_bias", "logit_bias is unsupported"));
    }
    let single = match &request.prompt {
        Prompt::Text(_) | Prompt::Ids(_) => true,
        Prompt::Texts(prompts) => prompts.len() == 1,
        Prompt::IdBatch(prompts) => prompts.len() == 1,
    };
    if !single {
        return Err(invalid("prompt", "one prompt per request is supported"));
    }
    let echo = request.echo.unwrap_or(false);
    let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
    match max_tokens {
        0 if !echo => {
            return Err(invalid(
                "max_tokens",
                "max_tokens 0 returns nothing without echo",
            ));
        }
        1 if request.temperature != Some(0.0) => {
            return Err(invalid(
                "temperature",
                "a raw prompt generates only greedily; send temperature 0",
            ));
        }
        0 | 1 => {}
        _ => {
            return Err(invalid(
                "max_tokens",
                "generation from a raw prompt is not supported yet; send max_tokens 0 with echo to score, or use /v1/chat/completions",
            ));
        }
    }
    let logprobs = request
        .logprobs
        .map(|top| usize::try_from(top).unwrap_or(usize::MAX));
    Ok(Prepared {
        request,
        echo,
        max_tokens,
        logprobs,
    })
}

/// Scores and answers one prepared request.
pub(crate) fn respond(
    connection: Connection,
    prepared: &Prepared,
    session: &mut dyn ChatBackend,
    id: &str,
) {
    let prompt = match &prepared.request.prompt {
        Prompt::Text(text) => ScoreText::Text(text),
        Prompt::Texts(texts) => ScoreText::Text(&texts[0]),
        Prompt::Ids(ids) => ScoreText::Ids(ids),
        Prompt::IdBatch(prompts) => ScoreText::Ids(&prompts[0]),
    };
    let wanted = prepared.logprobs.unwrap_or(0);
    // The greedy next token is the most likely one after the prompt.
    let top_logprobs = if prepared.max_tokens == 1 {
        wanted.max(1)
    } else {
        wanted
    };
    let scored = session.score(ScoreRequest {
        prompt,
        continuation: None,
        score_prompt: prepared.echo,
        top_logprobs,
    });
    let request_id = connection.request_id().map(str::to_owned);
    match scored {
        Err(ScoreError::Unsupported) => {
            json_response(
                connection,
                501,
                &json!({"error":{"message":"this backend does not support scoring","type":"server_error","param":null,"code":"scoring_unsupported"}}),
            );
        }
        Ok(result) => {
            record_score_usage(&result);
            let mut value = completion_value(prepared, &result, id);
            echo_request_id(&mut value, request_id.as_deref());
            json_response(connection, 200, &value);
        }
        Err(ScoreError::Execution(ChatGenerationError::Message(message))) => {
            json_response(connection, 400, &invalid("prompt", &message));
        }
        Err(ScoreError::Execution(ChatGenerationError::DeadlineExceeded)) => {
            json_response(
                connection,
                408,
                &json!({"error":{"message":"generation time budget exceeded","type":"timeout","param":null,"code":"generation_timeout"}}),
            );
        }
    }
}

/// One position of the returned sequence: its token, the token's log
/// probability, and the most likely alternatives there.
struct Position<'a> {
    id: i32,
    scored: Option<(f32, &'a [(i32, f32)])>,
}

fn completion_value(prepared: &Prepared, result: &ScoreResult, id: &str) -> Value {
    let mut positions = Vec::new();
    if prepared.echo {
        positions.push(Position {
            id: result.ids[0],
            scored: None,
        });
        for (&token, scored) in result.ids[1..].iter().zip(&result.scores.tokens) {
            positions.push(Position {
                id: token,
                scored: Some((scored.logprob, scored.top.as_slice())),
            });
        }
    }
    let mut generated = None;
    if prepared.max_tokens == 1
        && let Some(&(token, logprob)) = result.scores.next.first()
    {
        generated = Some(token);
        positions.push(Position {
            id: token,
            scored: Some((logprob, result.scores.next.as_slice())),
        });
    }
    // The end-of-turn token is generated and scored but, as in `OpenAI`'s
    // service, not part of the text.
    let stopped = generated.is_some() && result.next_ends_turn;
    let piece = |token: i32| result.pieces.get(&token).map_or(&[][..], Vec::as_slice);
    let mut text_bytes = Vec::new();
    let mut tokens = Vec::new();
    let mut token_logprobs = Vec::new();
    let mut top_logprobs = Vec::new();
    let mut text_offset = Vec::new();
    let mut offset = 0;
    let shown = positions.len() - usize::from(stopped);
    for (index, position) in positions.iter().enumerate() {
        let bytes = piece(position.id);
        let text = token_text(bytes);
        text_offset.push(offset);
        if index < shown {
            text_bytes.extend_from_slice(bytes);
            offset += text.chars().count();
        }
        match position.scored {
            None => {
                token_logprobs.push(Value::Null);
                top_logprobs.push(Value::Null);
            }
            Some((logprob, top)) => {
                token_logprobs.push(json!(f64::from(logprob)));
                // The `logprobs` most likely tokens, and the chosen one, as
                // `OpenAI` returns up to `logprobs + 1`; a lossy spelling
                // shared by two tokens keeps the likelier one.
                let mut alternatives = Map::new();
                let wanted = prepared.logprobs.unwrap_or(0);
                for &(alternative, alternative_logprob) in top.iter().take(wanted) {
                    alternatives
                        .entry(token_text(piece(alternative)))
                        .or_insert_with(|| json!(f64::from(alternative_logprob)));
                }
                alternatives
                    .entry(text)
                    .or_insert_with(|| json!(f64::from(logprob)));
                top_logprobs.push(Value::Object(alternatives));
            }
        }
        tokens.push(json!(token_text(bytes)));
    }
    let logprobs = if prepared.logprobs.is_some() {
        json!({"tokens":tokens,"token_logprobs":token_logprobs,"top_logprobs":top_logprobs,"text_offset":text_offset})
    } else {
        Value::Null
    };
    let finish_reason = if stopped { "stop" } else { "length" };
    let completion_tokens = usize::from(generated.is_some());
    json!({
        "id": format!("cmpl-{id}"),
        "object": "text_completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs()),
        "model": prepared.request.model,
        "choices": [{
            "index": 0,
            "text": token_text(&text_bytes),
            "logprobs": logprobs,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": result.ids.len(),
            "completion_tokens": completion_tokens,
            "total_tokens": result.ids.len() + completion_tokens,
        },
        "metallix": {"score_ms": result.score_ms},
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use qwen::forward::{Qwen3ScoredToken, Qwen3Scores};

    use super::*;

    fn prepared(body: &Value) -> Prepared {
        prepare(body.to_string().as_bytes())
            .unwrap_or_else(|error| panic!("{body}: {error}"))
            .1
    }

    /// Prompt `[5, 6, 7]`: token 6 scored -1.0 (alternative 8 at -0.5),
    /// token 7 scored -2.0 and greedy, next token 9 at -0.25.
    fn result(next_ends_turn: bool) -> ScoreResult {
        ScoreResult {
            ids: vec![5, 6, 7],
            from: 1,
            scores: Qwen3Scores {
                tokens: vec![
                    Qwen3ScoredToken {
                        logprob: -1.0,
                        top: vec![(8, -0.5), (6, -1.0)],
                    },
                    Qwen3ScoredToken {
                        logprob: -2.0,
                        top: vec![(7, -2.0), (8, -3.0)],
                    },
                ],
                next: vec![(9, -0.25), (8, -4.0)],
            },
            pieces: HashMap::from([
                (5, b"Hi".to_vec()),
                (6, b" there".to_vec()),
                (7, b"!".to_vec()),
                (8, b" x".to_vec()),
                (9, b" ok".to_vec()),
            ]),
            next_ends_turn,
            score_ms: 1.0,
        }
    }

    /// lm-eval's log-likelihood request (`lm_eval/models/openai_completions.py`
    /// at d6de816): `echo`, `max_tokens` 1, `logprobs` 1, temperature 0, and
    /// the prompt as a batch of one token array. It sums
    /// `token_logprobs[ctxlen:-1]` and calls a span greedy when each
    /// logprob equals the largest value of its `top_logprobs` entry.
    #[test]
    fn the_lm_eval_request_returns_prompt_logprobs_and_one_greedy_token() {
        let request = prepared(
            &json!({"model":"m","prompt":[[5,6,7]],"temperature":0,"max_tokens":1,"logprobs":1,"seed":1234,"echo":true}),
        );
        let value = completion_value(&request, &result(false), "1");
        let choice = &value["choices"][0];
        assert_eq!(choice["text"], "Hi there! ok");
        assert_eq!(choice["finish_reason"], "length");
        let logprobs = &choice["logprobs"];
        assert_eq!(logprobs["tokens"], json!(["Hi", " there", "!", " ok"]));
        assert_eq!(logprobs["token_logprobs"], json!([null, -1.0, -2.0, -0.25]));
        assert_eq!(logprobs["text_offset"], json!([0, 2, 8, 9]));
        assert_eq!(
            logprobs["top_logprobs"],
            json!([null, {" x": -0.5, " there": -1.0}, {"!": -2.0}, {" ok": -0.25}])
        );
        // ctxlen 2: the continuation is token 7 alone.
        let tops = logprobs["top_logprobs"].as_array().expect("tops");
        let greedy = |index: usize| {
            let best = tops[index]
                .as_object()
                .expect("top")
                .values()
                .filter_map(Value::as_f64)
                .fold(f64::NEG_INFINITY, f64::max);
            logprobs["token_logprobs"][index].as_f64() == Some(best)
        };
        assert!(!greedy(1) && greedy(2));
        assert_eq!(
            value["usage"],
            json!({"prompt_tokens":3,"completion_tokens":1,"total_tokens":4})
        );
    }

    #[test]
    fn an_end_of_turn_next_token_is_scored_but_not_shown() {
        let request = prepared(
            &json!({"model":"m","prompt":"Hi there!","temperature":0,"max_tokens":1,"logprobs":0}),
        );
        let value = completion_value(&request, &result(true), "1");
        let choice = &value["choices"][0];
        assert_eq!(choice["text"], "");
        assert_eq!(choice["finish_reason"], "stop");
        assert_eq!(choice["logprobs"]["token_logprobs"], json!([-0.25]));
    }

    #[test]
    fn scoring_with_echo_and_no_logprobs_returns_the_prompt_text() {
        let request =
            prepared(&json!({"model":"m","prompt":"Hi there!","max_tokens":0,"echo":true}));
        let value = completion_value(&request, &result(false), "1");
        assert_eq!(value["choices"][0]["text"], "Hi there!");
        assert!(value["choices"][0]["logprobs"].is_null());
        assert_eq!(value["object"], "text_completion");
    }

    /// Unsupported fields are `400`s naming the field, in the schema's
    /// `ErrorResponse` shape.
    #[test]
    fn unsupported_fields_are_rejected_by_name() {
        for (body, param) in [
            (
                json!({"model":"m","prompt":"a","echo":true,"max_tokens":0,"logprobs":6}),
                "logprobs",
            ),
            (
                json!({"model":"m","prompt":"a","echo":true,"max_tokens":0,"n":2}),
                "n",
            ),
            (
                json!({"model":"m","prompt":"a","echo":true,"max_tokens":0,"best_of":2}),
                "best_of",
            ),
            (
                json!({"model":"m","prompt":"a","echo":true,"max_tokens":0,"stream":true}),
                "stream",
            ),
            (
                json!({"model":"m","prompt":"a","echo":true,"max_tokens":0,"stop":["x"]}),
                "stop",
            ),
            (
                json!({"model":"m","prompt":"a","echo":true,"max_tokens":0,"suffix":"z"}),
                "suffix",
            ),
            (
                json!({"model":"m","prompt":"a","echo":true,"max_tokens":0,"logit_bias":{"1":5}}),
                "logit_bias",
            ),
            (
                json!({"model":"m","prompt":["a","b"],"echo":true,"max_tokens":0}),
                "prompt",
            ),
            (json!({"model":"m","prompt":"a"}), "max_tokens"),
            (
                json!({"model":"m","prompt":"a","max_tokens":0}),
                "max_tokens",
            ),
            (
                json!({"model":"m","prompt":"a","max_tokens":1}),
                "temperature",
            ),
        ] {
            let Err(error) = prepare(body.to_string().as_bytes()) else {
                panic!("{body} was accepted");
            };
            assert_eq!(error["error"]["param"], param, "{body}");
            assert_eq!(error["error"]["type"], "invalid_request_error", "{body}");
            assert!(error["error"]["code"].is_null(), "{body}");
        }
    }
}
