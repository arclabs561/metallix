//! Serial model continuation around the operator search command.
use std::time::{Duration, Instant};

use chat_format::{AssistantTurn, ChatMessage, ChatRole, ChatToolCall, TurnDelta};
use serde_json::{Value, json};

use super::{Backend, SearchRequest, SearchResult, fenced_results};
use crate::chat_generation::{ChatBackend, ChatGeneration, ChatRequest, GenerationControls};

#[derive(Clone, Debug)]
pub(crate) struct Policy {
    pub(crate) allowed: Vec<String>,
    pub(crate) blocked: Vec<String>,
    pub(crate) max_uses: usize,
}

impl Policy {
    pub(crate) fn parse(tool: &Value) -> Result<Self, String> {
        if tool
            .get("external_web_access")
            .is_some_and(|value| value != &Value::Bool(true))
        {
            return Err("hosted search requires external_web_access=true".into());
        }
        if tool
            .get("user_location")
            .is_some_and(|value| !value.is_null())
        {
            return Err("operator search commands do not support user_location".into());
        }

        let domains = |value: Option<&Value>| -> Result<Vec<String>, String> {
            match value {
                None => Ok(Vec::new()),
                Some(Value::Array(values)) if values.len() <= 64 => values
                    .iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| "search domains must be strings".into())
                    })
                    .collect(),
                _ => Err("search domains must be an array of at most 64 strings".into()),
            }
        };
        let allowed = domains(
            tool.get("allowed_domains")
                .or_else(|| tool.get("filters").and_then(|v| v.get("allowed_domains"))),
        )?;
        let blocked = domains(tool.get("blocked_domains"))?;
        let max_uses = match tool.get("max_uses") {
            None => 5,
            Some(value) => usize::try_from(
                value
                    .as_u64()
                    .ok_or("max_uses must be a nonnegative integer")?,
            )
            .map_err(|_| "max_uses is too large")?
            .min(5),
        };
        SearchRequest::new("validate".into(), &allowed, &blocked)
            .map_err(|_| "invalid search domain")?;
        Ok(Self {
            allowed,
            blocked,
            max_uses,
        })
    }
}

pub(crate) fn function_tool() -> Value {
    json!({"type":"function","function":{"name":"web_search","description":"Search the web for current information.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}})
}

pub(crate) enum Event<'a> {
    Delta(TurnDelta),
    Segment {
        generated: &'a ChatGeneration,
        turn: &'a AssistantTurn,
        searching: bool,
    },
    SearchStart {
        index: usize,
        query: &'a str,
    },
    SearchEnd {
        index: usize,
        query: &'a str,
        results: &'a Result<Vec<SearchResult>, super::SearchError>,
    },
}

pub(crate) struct Outcome {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
    pub(crate) searches: usize,
    pub(crate) web_content: bool,
    pub(crate) complete: bool,
}

fn execute_search(
    backend: &Backend,
    policy: &Policy,
    search: &ChatToolCall,
    deadline: Instant,
    outcome: &mut Outcome,
    emit: &mut dyn FnMut(Event<'_>) -> Result<(), String>,
) -> Result<String, String> {
    let query = search
        .arguments
        .get("query")
        .and_then(Value::as_str)
        .ok_or("web_search query must be a string")?;
    let request = SearchRequest::new(query.into(), &policy.allowed, &policy.blocked)
        .map_err(|_| "invalid search request")?;
    emit(Event::SearchStart {
        index: outcome.searches,
        query,
    })?;
    let results = backend.search(&request, deadline, &mut || {
        emit(Event::Delta(TurnDelta::Held))
    });
    emit(Event::SearchEnd {
        index: outcome.searches,
        query,
        results: &results,
    })?;
    let text = match results {
        Ok(results) => {
            outcome.web_content = true;
            fenced_results(query, &results)
        }
        Err(_) => "search failed".into(),
    };
    Ok(text)
}

/// Emits tokens immediately and emits hosted items only at segment boundaries.
/// Search availability is fixed by the caller once at prompt preparation.
#[allow(
    clippy::too_many_arguments,
    reason = "one serial turn and its explicit resource bounds"
)]
pub(crate) fn run(
    backend: &Backend,
    policy: &Policy,
    session: &mut dyn ChatBackend,
    controls: &GenerationControls,
    history: &[ChatMessage],
    client_tools: &[Value],
    cache_salt: Option<&str>,
    timeout: Duration,
    emit: &mut dyn FnMut(Event<'_>) -> Result<(), String>,
) -> Result<Outcome, String> {
    let deadline = Instant::now() + timeout;
    let mut messages = history.to_vec();
    let mut outcome = Outcome {
        input_tokens: 0,
        output_tokens: 0,
        searches: 0,
        web_content: history.iter().any(|message| {
            message.role == ChatRole::Tool && message.name.as_deref() == Some("web_search")
        }),
        complete: false,
    };
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or("generation time budget exceeded")?;
        let mut tools = client_tools.to_vec();
        let mut segment_controls = controls.clone();
        segment_controls
            .strict_tools
            .retain(|name| name != "web_search");
        if outcome.searches > 0
            && (matches!(
                segment_controls.tool_requirement,
                chat_format::ToolRequirement::Required
            ) || matches!(&segment_controls.tool_requirement, chat_format::ToolRequirement::Named(name) if name == "web_search"))
        {
            segment_controls.tool_requirement = chat_format::ToolRequirement::Auto;
        }

        if outcome.searches < policy.max_uses {
            tools.push(function_tool());
            segment_controls.strict_tools.push("web_search".into());
        }
        if let Some(limit) = controls.max_tokens {
            let used =
                u32::try_from(outcome.output_tokens).map_err(|_| "output token count overflow")?;
            segment_controls.max_tokens = Some(
                limit
                    .checked_sub(used)
                    .filter(|v| *v > 0)
                    .ok_or("generation output budget exhausted")?,
            );
        }
        segment_controls.validate(!tools.is_empty())?;
        let generated = session
            .generate_with_timeout(
                ChatRequest {
                    cache_salt,
                    ..segment_controls.request(&messages, &tools)
                },
                remaining,
                &mut |delta| emit(Event::Delta(delta)),
            )
            .map_err(|e| e.to_string())?;
        crate::responses::record_usage(&generated);
        outcome.input_tokens += generated.metrics.prompt_tokens;
        outcome.output_tokens += generated.generated_token_ids.len();
        let mut turn = crate::responses::assistant_turn(
            &tools,
            &segment_controls.strict_tools,
            controls.json_schema.as_ref(),
            &generated,
        )?;
        let search = turn
            .calls
            .iter()
            .find(|call| call.name == "web_search")
            .cloned();
        if search.is_some() && outcome.searches >= policy.max_uses {
            return Err("model called a withdrawn search tool".into());
        }
        // Client calls mixed with a search are regenerated after the results.
        if search.is_some() {
            turn.calls.retain(|call| call.name == "web_search");
        }
        emit(Event::Segment {
            generated: &generated,
            turn: &turn,
            searching: search.is_some(),
        })?;
        let Some(search) = search else {
            outcome.complete = turn.complete;
            return Ok(outcome);
        };
        let text = execute_search(backend, policy, &search, deadline, &mut outcome, emit)?;
        outcome.searches += 1;
        let mut assistant = ChatMessage::text(ChatRole::Assistant, turn.text);
        assistant.tool_calls.push(ChatToolCall {
            name: "web_search".into(),
            arguments: search.arguments,
        });
        messages.push(assistant);
        let mut result = ChatMessage::text(ChatRole::Tool, text);
        result.name = Some("web_search".into());
        messages.push(result);
        if controls
            .max_tokens
            .is_some_and(|limit| outcome.output_tokens >= limit as usize)
        {
            return Ok(outcome);
        }
    }
}
