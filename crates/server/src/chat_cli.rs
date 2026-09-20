//! Interactive chat and a bounded workspace-read agent using the native session.

use std::{
    io::{self, BufRead, Write},
    path::Path,
    process::ExitCode,
};

use serde_json::json;

use crate::{
    chat_generation::{
        ChatFinishReason, ChatMessage, ChatRequest, ChatRole, ChatSession, ChatToolCall,
        ChatToolResult,
    },
    chat_tools::{self, WorkspaceTools},
};

pub(crate) fn message(role: ChatRole, content: String) -> ChatMessage {
    ChatMessage::text(role, content)
}

pub(crate) fn chat(
    model: &Path,
    prompt: Option<String>,
    max_tokens: u32,
    json_output: bool,
) -> ExitCode {
    finish(chat_inner(model, prompt, max_tokens, json_output))
}

fn chat_inner(
    model: &Path,
    prompt: Option<String>,
    max_tokens: u32,
    json_output: bool,
) -> Result<(), String> {
    let mut session = ChatSession::load(model)?;
    let mut messages = Vec::new();
    if let Some(prompt) = prompt {
        messages.push(message(ChatRole::User, prompt));
        let result = session.generate(ChatRequest::new(&messages, max_tokens), &mut |delta| {
            if !json_output {
                print!("{delta}");
                io::stdout().flush().map_err(|e| e.to_string())?;
            }
            Ok(())
        })?;
        if json_output {
            println!(
                "{}",
                serde_json::to_string(&result).map_err(|e| e.to_string())?
            );
        } else {
            println!();
        }
        return Ok(());
    }
    eprintln!(
        "Native Qwen chat; /reset clears history, /quit exits. Context limit: 512 tokens including output budget."
    );
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!("you> ");
        io::stderr().flush().map_err(|e| e.to_string())?;
        let Some(line) = lines.next() else { break };
        let line = line.map_err(|e| e.to_string())?;
        match line.trim() {
            "/quit" => break,
            "/reset" => {
                messages.clear();
                continue;
            }
            "" => continue,
            _ => {}
        }
        messages.push(message(ChatRole::User, line));
        let result = session.generate(ChatRequest::new(&messages, max_tokens), &mut |delta| {
            print!("{delta}");
            io::stdout().flush().map_err(|e| e.to_string())
        });
        println!();
        match result {
            Ok(result) => messages.push(message(ChatRole::Assistant, result.text)),
            Err(error) => {
                messages.pop();
                eprintln!("chat: {error}; history unchanged, use /reset if full");
            }
        }
    }
    Ok(())
}

pub(crate) fn agent(
    model: &Path,
    workspace: &Path,
    prompt: String,
    max_tokens: u32,
    max_turns: u32,
) -> ExitCode {
    finish(agent_inner(model, workspace, prompt, max_tokens, max_turns))
}

fn agent_inner(
    model: &Path,
    workspace: &Path,
    prompt: String,
    max_tokens: u32,
    max_turns: u32,
) -> Result<(), String> {
    let workspace = WorkspaceTools::new(workspace)?;
    let mut session = ChatSession::load(model)?;
    let tools = chat_tools::definitions();
    let mut messages = vec![message(ChatRole::User, prompt)];
    for turn_index in 0..max_turns {
        let result = session.generate(
            ChatRequest {
                messages: &messages,
                tools: &tools,
                max_tokens,
                enable_thinking: false,
            },
            &mut |_| Ok(()),
        )?;
        let turn = chat_tools::parse_turn(&result.text)?;
        if turn.calls.is_empty() {
            if result.finish_reason != ChatFinishReason::Eos {
                return Err("agent reached output limit before finishing a response".into());
            }
            println!("{}", turn.text);
            return Ok(());
        }
        if result.finish_reason != ChatFinishReason::Eos {
            return Err("agent output was truncated; no tools executed".into());
        }
        let mut assistant = message(ChatRole::Assistant, turn.text);
        assistant.tool_calls = turn
            .calls
            .iter()
            .map(|call| ChatToolCall {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
            .collect();
        messages.push(assistant);
        for (index, call) in turn.calls.iter().enumerate() {
            eprintln!("tool: {}", call.name);
            let output = workspace
                .execute(call)
                .unwrap_or_else(|error| json!({"error":error}));
            messages.push(
                ChatToolResult {
                    tool_call_id: format!("call_{turn_index}_{index}"),
                    name: Some(call.name.clone()),
                    content: output.to_string(),
                }
                .into_message(),
            );
        }
    }
    Err(format!("agent stopped at the {max_turns}-turn limit"))
}

fn finish(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("mx: {error}");
            ExitCode::FAILURE
        }
    }
}
