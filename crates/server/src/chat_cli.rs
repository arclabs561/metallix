//! Interactive chat and a bounded workspace-read agent using the native session.

use std::{
    io::{self, BufRead, Write},
    path::Path,
    process::ExitCode,
};

use serde_json::json;

use crate::{
    agent_receipt::{AgentCallOutcome, AgentReceipt, AgentTurnReceipt},
    chat_generation::{
        ChatFinishReason, ChatMessage, ChatRequest, ChatRole, ChatSession, ChatToolCall,
        ChatToolResult, ResidentChatLimits,
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
    limits: ResidentChatLimits,
) -> ExitCode {
    finish(chat_inner(model, prompt, max_tokens, json_output, limits))
}

fn chat_inner(
    model: &Path,
    prompt: Option<String>,
    max_tokens: u32,
    json_output: bool,
    limits: ResidentChatLimits,
) -> Result<(), String> {
    let mut session = ChatSession::load(model, limits)?;
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
        "Native Qwen chat; /reset clears history, /quit exits. Context limit: {} tokens including output budget.",
        limits.context_tokens()
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

#[derive(Clone, Copy)]
struct AgentLimits {
    max_tokens: u32,
    max_turns: u32,
    resident: ResidentChatLimits,
}

pub(crate) fn agent(
    model: &Path,
    workspace: &Path,
    prompt: String,
    max_tokens: u32,
    max_turns: u32,
    resident: ResidentChatLimits,
    json_output: bool,
) -> ExitCode {
    let mut receipt = AgentReceipt::new();
    let result = agent_inner(
        model,
        workspace,
        prompt,
        AgentLimits {
            max_tokens,
            max_turns,
            resident,
        },
        &mut receipt,
        !json_output,
    );
    match result {
        Ok(final_text) => {
            receipt.complete(final_text.clone());
            if json_output {
                print_agent_receipt(&receipt)
            } else {
                println!("{final_text}");
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            receipt.fail();
            if json_output {
                let output_status = print_agent_receipt(&receipt);
                eprintln!("mx: {error}");
                if output_status == ExitCode::SUCCESS {
                    ExitCode::FAILURE
                } else {
                    output_status
                }
            } else {
                finish(Err(error))
            }
        }
    }
}

fn agent_inner(
    model: &Path,
    workspace: &Path,
    prompt: String,
    limits: AgentLimits,
    receipt: &mut AgentReceipt,
    report_tools: bool,
) -> Result<String, String> {
    let AgentLimits {
        max_tokens,
        max_turns,
        resident,
    } = limits;
    let workspace = WorkspaceTools::new(workspace)?;
    let mut session = ChatSession::load(model, resident)?;
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
        let mut turn_receipt = AgentTurnReceipt::from_generation(turn_index, &result);
        let turn = match chat_tools::parse_turn(&result.text) {
            Ok(turn) => turn,
            Err(error) => {
                receipt.push_turn(turn_receipt);
                return Err(error);
            }
        };
        if turn.calls.is_empty() {
            if result.finish_reason != ChatFinishReason::Eos {
                receipt.push_turn(turn_receipt);
                return Err("agent reached output limit before finishing a response".into());
            }
            receipt.push_turn(turn_receipt);
            return Ok(turn.text);
        }
        if result.finish_reason != ChatFinishReason::Eos {
            receipt.push_turn(turn_receipt);
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
        let outputs = execute_calls(
            &mut turn_receipt,
            result.finish_reason,
            &turn.calls,
            |call| {
                if report_tools {
                    eprintln!("tool: {}", call.name);
                }
                workspace.execute(call)
            },
        );
        for (index, (call, output)) in turn.calls.iter().zip(outputs).enumerate() {
            messages.push(
                ChatToolResult {
                    tool_call_id: format!("call_{turn_index}_{index}"),
                    name: Some(call.name.clone()),
                    content: output.to_string(),
                }
                .into_message(),
            );
        }
        receipt.push_turn(turn_receipt);
    }
    Err(format!("agent stopped at the {max_turns}-turn limit"))
}

/// Executes calls only after a complete model turn, and records the precise
/// execution order without retaining tool results in the receipt.
fn execute_calls<F>(
    receipt: &mut AgentTurnReceipt,
    finish_reason: ChatFinishReason,
    calls: &[chat_tools::ToolCall],
    mut execute: F,
) -> Vec<serde_json::Value>
where
    F: FnMut(&chat_tools::ToolCall) -> Result<serde_json::Value, String>,
{
    if finish_reason != ChatFinishReason::Eos {
        return Vec::new();
    }
    calls
        .iter()
        .map(|call| match execute(call) {
            Ok(output) => {
                receipt.record_call(call.name.clone(), &call.arguments, AgentCallOutcome::Ok);
                output
            }
            Err(error) => {
                receipt.record_call(call.name.clone(), &call.arguments, AgentCallOutcome::Error);
                json!({"error":error})
            }
        })
        .collect()
}

fn print_agent_receipt(receipt: &AgentReceipt) -> ExitCode {
    match serde_json::to_string(receipt) {
        Ok(serialized) => {
            println!("{serialized}");
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("mx: could not serialize agent receipt: {error}");
            ExitCode::FAILURE
        }
    }
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::json;

    use super::{AgentTurnReceipt, execute_calls};
    use crate::{
        agent_receipt::hash_arguments,
        chat_generation::{ChatFinishReason, ChatGeneration, ChatGenerationMetrics},
        chat_tools::ToolCall,
    };

    fn turn_receipt() -> AgentTurnReceipt {
        AgentTurnReceipt::from_generation(
            0,
            &ChatGeneration {
                text: String::new(),
                generated_token_ids: Vec::new(),
                finish_reason: ChatFinishReason::Eos,
                metrics: ChatGenerationMetrics {
                    context_tokens: 1,
                    planned_kv_bytes: 1,
                    session_load_ms: 0.0,
                    render_ms: 0.0,
                    prefill_ms: 0.0,
                    time_to_first_token_ms: None,
                    decode_ms: Vec::new(),
                    decode_total_ms: 0.0,
                    prompt_tokens: 0,
                    generated_tokens: 0,
                },
            },
        )
    }

    #[test]
    fn truncated_turn_executes_no_calls() {
        let calls = vec![ToolCall {
            name: "read_file".into(),
            arguments: json!({"path":"note.txt"}),
        }];
        let mut receipt = turn_receipt();
        let mut invoked = false;
        let output = execute_calls(&mut receipt, ChatFinishReason::Length, &calls, |_| {
            invoked = true;
            Ok(json!({"text":"private"}))
        });
        assert!(output.is_empty());
        assert!(!invoked);
        assert!(
            serde_json::to_value(receipt).unwrap()["calls"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn completed_turn_records_calls_in_execution_order() {
        let calls = vec![
            ToolCall {
                name: "read_file".into(),
                arguments: json!({"path":"first.txt"}),
            },
            ToolCall {
                name: "search_file".into(),
                arguments: json!({"path":"second.txt", "query":"private"}),
            },
        ];
        let mut receipt = turn_receipt();
        let output = execute_calls(&mut receipt, ChatFinishReason::Eos, &calls, |call| {
            Ok(json!({"name":call.name}))
        });
        assert_eq!(
            output,
            vec![json!({"name":"read_file"}), json!({"name":"search_file"})]
        );
        let calls = serde_json::to_value(receipt).unwrap()["calls"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(calls[0]["name"], "read_file");
        assert_eq!(calls[1]["name"], "search_file");
        assert!(calls[1].get("query").is_none());
    }

    proptest! {
        #[test]
        fn call_receipts_match_actual_execution_for_completed_turns(
            outcomes in prop::collection::vec(any::<bool>(), 0..9),
            complete in any::<bool>(),
        ) {
            let calls: Vec<_> = outcomes
                .iter()
                .enumerate()
                .map(|(index, _)| ToolCall {
                    name: format!("tool_{index}"),
                    arguments: json!({"path":format!("nested/{index}.txt"), "query":format!("q{index}")}),
                })
                .collect();
            let expected_names: Vec<_> = calls.iter().map(|call| call.name.clone()).collect();
            let expected_hashes: Vec<_> = calls
                .iter()
                .map(|call| hash_arguments(&call.arguments))
                .collect();
            let mut receipt = turn_receipt();
            let mut invoked = Vec::new();
            let mut outcome_index = 0;
            let finish_reason = if complete { ChatFinishReason::Eos } else { ChatFinishReason::Length };
            let output = execute_calls(&mut receipt, finish_reason, &calls, |call| {
                invoked.push(call.name.clone());
                let succeeded = outcomes[outcome_index];
                outcome_index += 1;
                if succeeded { Ok(json!({"private":"result"})) } else { Err("tool failed".into()) }
            });
            let recorded = serde_json::to_value(receipt).unwrap()["calls"]
                .as_array()
                .unwrap()
                .clone();
            if complete {
                prop_assert_eq!(invoked, expected_names);
                prop_assert_eq!(output.len(), outcomes.len());
                prop_assert_eq!(recorded.len(), outcomes.len());
                for (index, call) in recorded.iter().enumerate() {
                    let expected_name = format!("tool_{index}");
                    let expected_path = format!("nested/{index}.txt");
                    prop_assert_eq!(call["name"].as_str(), Some(expected_name.as_str()));
                    prop_assert_eq!(call["relative_path"].as_str(), Some(expected_path.as_str()));
                    prop_assert_eq!(call["arguments_sha256"].as_str(), Some(expected_hashes[index].as_str()));
                    prop_assert_eq!(call["outcome"].as_str(), Some(if outcomes[index] { "ok" } else { "error" }));
                    prop_assert!(call.get("private").is_none());
                }
            } else {
                prop_assert!(invoked.is_empty());
                prop_assert!(output.is_empty());
                prop_assert!(recorded.is_empty());
            }
        }
    }
}
