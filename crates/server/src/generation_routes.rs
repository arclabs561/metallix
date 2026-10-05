//! The generation routes: each path's protocol, parsed and validated before
//! admission and answered on the model worker in that protocol's shape.

use std::time::Duration;

use serde_json::{Value, json};

use crate::{
    chat_completions,
    chat_generation::{ChatBackend, ChatMessage},
    http_transport::Connection,
    responses,
};

/// One validated generation request in its own protocol.
pub(crate) enum Generation {
    Responses {
        request: Box<responses::Request>,
        messages: Vec<ChatMessage>,
        tools: Vec<Value>,
    },
    ChatCompletions(Box<chat_completions::Prepared>),
}

impl Generation {
    /// Whether a `POST` path generates.
    pub(crate) fn serves(path: &str) -> bool {
        matches!(path, "/v1/responses" | "/v1/chat/completions")
    }

    /// Parses and validates one body for `path`, returning the requested
    /// model; the error is a ready `400` body in the protocol's shape.
    pub(crate) fn parse(path: &str, body: &[u8]) -> Result<(String, Self), Value> {
        debug_assert!(Self::serves(path), "{path} does not generate");
        if path == "/v1/chat/completions" {
            return chat_completions::prepare(body)
                .map(|(model, prepared)| (model, Self::ChatCompletions(Box::new(prepared))));
        }
        let request: responses::Request = serde_json::from_slice(body)
            .map_err(|error| json!({"error":{"message":error.to_string()}}))?;
        let (messages, tools) = responses::messages(&request)
            .and_then(|messages| responses::tools(&request).map(|tools| (messages, tools)))
            .map_err(|error| json!({"error":{"message":error}}))?;
        Ok((
            request.model.clone(),
            Self::Responses {
                request: Box::new(request),
                messages,
                tools,
            },
        ))
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
        }
    }
}
