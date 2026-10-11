//! The generation routes: each path's protocol, parsed and validated before
//! admission and answered on the model worker in that protocol's shape.

use std::time::Duration;

use serde_json::{Value, json};

use crate::{
    anthropic_messages, chat_completions,
    chat_generation::{ChatBackend, ChatMessage},
    completions,
    http_transport::Connection,
    responses::{self, json_response},
    scoring,
};

/// An error body in the shape of the protocol `path` speaks, so each client
/// SDK can parse it: Anthropic's for `/v1/messages`, `OpenAI`'s for every other
/// route and for a request whose path is unknown.
///
/// `OpenAI` (`ErrorResponse` in <https://github.com/openai/openai-openapi>):
/// `error` with `type`, `message`, `param` and `code`, the last two nullable.
/// Anthropic (<https://platform.claude.com/docs/en/api/errors>): top-level
/// `type: "error"`, `error` with a `type` named per status, and `request_id`.
pub(crate) fn error_body(
    path: Option<&str>,
    status: u16,
    code: Option<&str>,
    message: &str,
    request_id: Option<&str>,
) -> Value {
    if path == Some("/v1/messages") {
        let kind = match status {
            404 => "not_found_error",
            408 | 504 => "timeout_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            503 | 529 => "overloaded_error",
            500.. => "api_error",
            _ => "invalid_request_error",
        };
        return json!({"type":"error","error":{"type":kind,"message":message},"request_id":request_id});
    }
    let kind = if status >= 500 {
        "server_error"
    } else {
        "invalid_request_error"
    };
    json!({"error":{"message":message,"type":kind,"param":null,"code":code}})
}

/// Answers `status` with [`error_body`] for the connection's request path.
pub(crate) fn error_response(
    connection: Connection,
    status: u16,
    code: Option<&str>,
    message: &str,
) {
    let body = error_body(
        connection.path(),
        status,
        code,
        message,
        connection.request_id(),
    );
    json_response(connection, status, &body);
}

/// One validated generation request in its own protocol.
pub(crate) enum Generation {
    Responses {
        request: Box<responses::Request>,
        messages: Vec<ChatMessage>,
        tools: Vec<Value>,
    },
    ChatCompletions(Box<chat_completions::Prepared>),
    Messages(Box<anthropic_messages::Prepared>),
    /// Legacy Completions, scoring a raw prompt.
    Completions(Box<completions::Prepared>),
    /// Teacher-forced scoring of a continuation.
    Score(Box<scoring::Prepared>),
}

impl Generation {
    /// Whether a `POST` path generates.
    pub(crate) fn serves(path: &str) -> bool {
        matches!(
            path,
            "/v1/responses"
                | "/v1/chat/completions"
                | "/v1/messages"
                | "/v1/completions"
                | "/v1/score"
        )
    }

    /// Parses and validates one body for `path`, returning the requested
    /// model; the error is a ready `400` body in the protocol's shape.
    pub(crate) fn parse(path: &str, body: &[u8]) -> Result<(String, Self), Value> {
        debug_assert!(Self::serves(path), "{path} does not generate");
        if path == "/v1/messages" {
            return anthropic_messages::prepare(body)
                .map(|(model, prepared)| (model, Self::Messages(Box::new(prepared))));
        }
        if path == "/v1/chat/completions" {
            return chat_completions::prepare(body)
                .map(|(model, prepared)| (model, Self::ChatCompletions(Box::new(prepared))));
        }
        if path == "/v1/completions" {
            return completions::prepare(body)
                .map(|(model, prepared)| (model, Self::Completions(Box::new(prepared))));
        }
        if path == "/v1/score" {
            return scoring::prepare(body)
                .map(|(model, prepared)| (model, Self::Score(Box::new(prepared))));
        }
        let invalid = |message: &str| error_body(Some(path), 400, None, message, None);
        let request: responses::Request =
            serde_json::from_slice(body).map_err(|error| invalid(&error.to_string()))?;
        let (messages, tools) = responses::messages(&request)
            .and_then(|messages| responses::tools(&request).map(|tools| (messages, tools)))
            .map_err(|error| invalid(&error))?;
        Ok((
            request.model.clone(),
            Self::Responses {
                request: Box::new(request),
                messages,
                tools,
            },
        ))
    }

    /// The `GenAI` operation name the request span records.
    pub(crate) const fn operation(&self) -> &'static str {
        match self {
            Self::Completions(_) => "text_completion",
            Self::Score(_) => "score",
            Self::Responses { .. } | Self::ChatCompletions(_) | Self::Messages(_) => "chat",
        }
    }

    /// Generates and answers; `id` is the server-unique suffix each protocol
    /// prefixes with its own identifier style.
    pub(crate) fn respond(
        &self,
        connection: Connection,
        session: &mut dyn ChatBackend,
        id: &str,
        generation_timeout: Duration,
    ) -> Result<(), String> {
        match self {
            Self::Responses {
                request,
                messages,
                tools,
            } => responses::respond(
                connection,
                request,
                messages,
                tools,
                session,
                &format!("resp_{id}"),
                generation_timeout,
            ),
            Self::ChatCompletions(prepared) => {
                chat_completions::respond(connection, prepared, session, id, generation_timeout)
            }
            Self::Messages(prepared) => {
                anthropic_messages::respond(connection, prepared, session, id, generation_timeout)
            }
            // Scoring is one prefill with no decode loop, so it runs to the
            // end rather than against the generation time budget.
            Self::Completions(prepared) => {
                completions::respond(connection, prepared, session, id);
                Ok(())
            }
            Self::Score(prepared) => {
                scoring::respond(connection, prepared, session, id);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::sse::test_support::{exchange, json_body};

    /// Expected bodies come from the published shapes: `OpenAI`'s
    /// `ErrorResponse` requires `type`, `message`, `param` and `code`;
    /// Anthropic's errors page names the `error.type` for each status.
    #[test]
    fn error_bodies_follow_each_protocols_published_shape() {
        assert_eq!(
            error_body(
                Some("/v1/chat/completions"),
                404,
                None,
                "model is not loaded",
                Some("r1")
            ),
            json!({"error":{"message":"model is not loaded","type":"invalid_request_error","param":null,"code":null}})
        );
        assert_eq!(
            error_body(None, 503, Some("server_busy"), "busy", None),
            json!({"error":{"message":"busy","type":"server_error","param":null,"code":"server_busy"}})
        );
        assert_eq!(
            error_body(
                Some("/v1/messages"),
                404,
                None,
                "model is not loaded",
                Some("r1")
            ),
            json!({"type":"error","error":{"type":"not_found_error","message":"model is not loaded"},"request_id":"r1"})
        );
        for (status, kind) in [
            (400, "invalid_request_error"),
            (408, "timeout_error"),
            (413, "request_too_large"),
            (431, "invalid_request_error"),
            (500, "api_error"),
            (503, "overloaded_error"),
        ] {
            let body = error_body(Some("/v1/messages"), status, None, "x", None);
            assert_eq!(body["error"]["type"], kind, "{status}");
        }
    }

    #[test]
    fn error_responses_take_the_shape_of_the_request_path() {
        for (path, anthropic) in [
            ("/v1/messages", true),
            ("/v1/chat/completions", false),
            ("/v1/responses", false),
        ] {
            let wire = exchange(path, "{}", |connection, _| {
                error_response(connection, 404, None, "model is not loaded");
            });
            let (status, body) = json_body(&wire);
            assert_eq!(status, "HTTP/1.1 404 Not Found", "{path}");
            if anthropic {
                assert_eq!(body["type"], "error", "{path}: {body}");
                assert_eq!(body["error"]["type"], "not_found_error", "{path}: {body}");
            } else {
                assert_eq!(
                    body["error"]["type"], "invalid_request_error",
                    "{path}: {body}"
                );
                assert!(body["error"]["code"].is_null(), "{path}: {body}");
            }
        }
    }
}
