//! `POST /v1/score`: the log probability the model gives each token of a
//! continuation after a prompt, computed teacher-forced in one prefill.
//!
//! The prompt is raw text, token IDs, or a conversation rendered by the chat
//! template with its generation prompt, as a generation request would see
//! it. A string continuation is encoded alone, without special tokens, and
//! its IDs are appended to the prompt's; `metallix.continuation_tokenization`
//! says so. Encoding prompt and continuation together, as lm-eval does, can
//! merge tokens across the boundary, so a caller who needs that split
//! encodes the text itself and passes token IDs.

use chat_format::Conversation;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::{
    chat_completions::{self, error_body},
    chat_generation::{
        ChatBackend, ChatGenerationError, ChatMessage, MAX_SCORE_TOP_LOGPROBS, ScoreError,
        ScoreRequest, ScoreResult, ScoreText,
    },
    http_transport::Connection,
    responses::{echo_request_id, json_response},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) model: String,
    prompt: Prompt,
    continuation: Continuation,
    #[serde(default)]
    top_logprobs: Option<u8>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Text(String),
    Ids(Vec<i32>),
    Chat(ChatPrompt),
}

/// A conversation in Chat Completions' message and tool shapes.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatPrompt {
    messages: Vec<Value>,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default)]
    reasoning_effort: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Continuation {
    Text(String),
    Ids(Vec<i32>),
}

/// A validated request with its rebuilt history, ready for a worker.
pub(crate) struct Prepared {
    request: Request,
    messages: Vec<ChatMessage>,
    tools: Vec<Value>,
    thinking: bool,
    effort: Option<String>,
}

/// Parses and validates one body; the error is a ready `400` body.
pub(crate) fn prepare(body: &[u8]) -> Result<(String, Prepared), Value> {
    let request: Request =
        serde_json::from_slice(body).map_err(|error| error_body(&error.to_string()))?;
    if request
        .top_logprobs
        .is_some_and(|top| usize::from(top) > MAX_SCORE_TOP_LOGPROBS)
    {
        return Err(error_body(&format!(
            "top_logprobs must be between 0 and {MAX_SCORE_TOP_LOGPROBS}"
        )));
    }
    let (messages, tools, thinking, effort) = match &request.prompt {
        Prompt::Chat(chat) => {
            let body = json!({
                "model": request.model,
                "messages": chat.messages,
                "tools": chat.tools,
                "reasoning_effort": chat.reasoning_effort,
            })
            .to_string();
            let (_, prepared) = chat_completions::prepare(body.as_bytes())?;
            prepared.into_prompt()
        }
        Prompt::Text(_) | Prompt::Ids(_) => (Vec::new(), Vec::new(), false, None),
    };
    Ok((
        request.model.clone(),
        Prepared {
            request,
            messages,
            tools,
            thinking,
            effort,
        },
    ))
}

/// Scores and answers one prepared request.
pub(crate) fn respond(
    connection: Connection,
    prepared: &Prepared,
    session: &mut dyn ChatBackend,
    id: &str,
) {
    let request = &prepared.request;
    let prompt = match &request.prompt {
        Prompt::Text(text) => ScoreText::Text(text),
        Prompt::Ids(ids) => ScoreText::Ids(ids),
        Prompt::Chat(_) => ScoreText::Chat(Conversation {
            messages: &prepared.messages,
            tools: &prepared.tools,
            enable_thinking: prepared.thinking,
            reasoning_effort: prepared.effort.as_deref(),
        }),
    };
    let (continuation, tokenization) = match &request.continuation {
        Continuation::Text(text) => (ScoreText::Text(text), "separate"),
        Continuation::Ids(ids) => (ScoreText::Ids(ids), "ids"),
    };
    let scored = session.score(ScoreRequest {
        prompt,
        continuation: Some(continuation),
        score_prompt: false,
        top_logprobs: usize::from(request.top_logprobs.unwrap_or(0)),
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
            let mut value = score_value(request, &result, id, tokenization);
            echo_request_id(&mut value, request_id.as_deref());
            json_response(connection, 200, &value);
        }
        Err(ScoreError::Execution(ChatGenerationError::Message(message))) => {
            json_response(connection, 400, &error_body(&message));
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

/// Records the `GenAI` usage attributes on the current request span; scoring
/// reads every token and writes none.
pub(crate) fn record_score_usage(result: &ScoreResult) {
    let span = tracing::Span::current();
    span.record("gen_ai.usage.input_tokens", result.ids.len());
    span.record("gen_ai.usage.output_tokens", 0);
}

/// A token's bytes as lossy UTF-8, as chat receipts show them.
pub(crate) fn token_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// One token as `/v1/score` reports it.
fn token_value(result: &ScoreResult, id: i32, logprob: f32) -> Map<String, Value> {
    let bytes = result.pieces.get(&id).map_or(&[][..], Vec::as_slice);
    let mut token = Map::new();
    token.insert("id".into(), json!(id));
    token.insert("token".into(), json!(token_text(bytes)));
    token.insert("bytes".into(), json!(bytes));
    token.insert("model_logprob".into(), json!(f64::from(logprob)));
    token
}

fn score_value(request: &Request, result: &ScoreResult, id: &str, tokenization: &str) -> Value {
    let continuation = &result.ids[result.from..];
    let tokens: Vec<Value> = continuation
        .iter()
        .zip(&result.scores.tokens)
        .map(|(&token_id, scored)| {
            let mut token = token_value(result, token_id, scored.logprob);
            if request.top_logprobs.is_some() {
                let top: Vec<Value> = scored
                    .top
                    .iter()
                    .map(|&(id, logprob)| Value::Object(token_value(result, id, logprob)))
                    .collect();
                token.insert("top_logprobs".into(), json!(top));
            }
            Value::Object(token)
        })
        .collect();
    let sum: f64 = result
        .scores
        .tokens
        .iter()
        .map(|scored| f64::from(scored.logprob))
        .sum();
    json!({
        "id": format!("score-{id}"),
        "object": "score",
        "model": request.model,
        "prompt_tokens": result.from,
        "tokens": tokens,
        "sum_logprob": sum,
        "metallix": {
            "continuation_tokenization": tokenization,
            "score_ms": result.score_ms,
        },
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use qwen::forward::{Qwen3ScoredToken, Qwen3Scores};

    use super::*;

    fn result() -> ScoreResult {
        ScoreResult {
            ids: vec![5, 6, 7],
            from: 2,
            scores: Qwen3Scores {
                tokens: vec![Qwen3ScoredToken {
                    logprob: -0.5,
                    top: vec![(9, -0.25), (7, -0.5)],
                }],
                next: Vec::new(),
            },
            pieces: HashMap::from([
                (5, b"a".to_vec()),
                (6, b"b".to_vec()),
                (7, b" c".to_vec()),
                (9, vec![0xE2]),
            ]),
            next_ends_turn: false,
            score_ms: 1.0,
        }
    }

    #[test]
    fn scoring_chat_prompt_uses_generation_history_validation() {
        let body = json!({"model":"m","prompt":{
            "messages":[
                {"role":"assistant","content":null,"tool_calls":[
                    {"id":"call1","type":"function","function":{"name":"read","arguments":"{\"path\":\"a\"}"}}
                ]},
                {"role":"tool","tool_call_id":"call1","content":"contents"}
            ],
            "tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],
            "reasoning_effort":"none"
        },"continuation":"answer"});
        let (_, prepared) = prepare(body.to_string().as_bytes()).expect("valid tool history");
        assert_eq!(
            prepared.messages[0].tool_calls[0].arguments,
            json!({"path":"a"})
        );
        assert_eq!(prepared.messages[1].name.as_deref(), Some("read"));
        assert_eq!(prepared.messages[1].tool_call_id.as_deref(), Some("call1"));
        assert_eq!(prepared.tools[0]["function"]["name"], "read");
        assert!(!prepared.thinking);
        let mut invalid = body;
        invalid["prompt"]["messages"][1]["tool_call_id"] = json!("missing");
        let Err(error) = prepare(invalid.to_string().as_bytes()) else {
            panic!("unmatched tool result was accepted");
        };
        assert_eq!(
            error["error"]["message"],
            "tool message has no matching pending call"
        );
    }

    #[test]
    fn requests_take_text_ids_or_a_conversation() {
        for body in [
            json!({"model":"m","prompt":"a b","continuation":" c"}),
            json!({"model":"m","prompt":[5,6],"continuation":[7]}),
            json!({"model":"m","prompt":{"messages":[{"role":"user","content":"hi"}]},"continuation":"ok","top_logprobs":2}),
        ] {
            prepare(body.to_string().as_bytes()).unwrap_or_else(|error| panic!("{body}: {error}"));
        }
        for (body, message) in [
            (
                json!({"model":"m","prompt":"a","continuation":"b","top_logprobs":21}),
                "top_logprobs must be between 0 and 20",
            ),
            (
                json!({"model":"m","prompt":{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"huge"},"continuation":"b"}),
                "unsupported reasoning_effort \"huge\"",
            ),
        ] {
            let Err(error) = prepare(body.to_string().as_bytes()) else {
                panic!("{body} was accepted");
            };
            assert_eq!(error["error"]["message"], message, "{body}");
        }
        assert!(prepare(br#"{"model":"m","prompt":"a"}"#).is_err());
    }

    /// Each continuation token carries its own log probability and bytes;
    /// a partial UTF-8 piece shows U+FFFD in `token` and exactly in `bytes`.
    #[test]
    fn the_response_lists_continuation_tokens_with_their_alternatives() {
        let body = json!({"model":"m","prompt":[5,6],"continuation":[7],"top_logprobs":2});
        let (_, prepared) = prepare(body.to_string().as_bytes()).expect("valid");
        let value = score_value(&prepared.request, &result(), "1", "ids");
        assert_eq!(value["prompt_tokens"], 2);
        assert_eq!(value["sum_logprob"], -0.5);
        assert_eq!(value["metallix"]["continuation_tokenization"], "ids");
        let token = &value["tokens"][0];
        assert_eq!(token["id"], 7);
        assert_eq!(token["token"], " c");
        assert_eq!(token["model_logprob"], -0.5);
        assert_eq!(token["top_logprobs"][0]["token"], "\u{FFFD}");
        assert_eq!(token["top_logprobs"][0]["bytes"], json!([0xE2]));
        assert_eq!(value["tokens"].as_array().map(Vec::len), Some(1));
    }
}
