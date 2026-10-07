//! One finished assistant turn, parsed in the dialect its checkpoint's
//! template teaches: reasoning, visible text and tool calls.
//!
//! [`parse_turn`] is the only parser entry. The dialect is chosen once, from
//! the template source, and an unrecognized or ambiguous template selects
//! plain text, so a model's call syntax is never misread as another family's.

use serde::Serialize;
use serde_json::Value;

use crate::{
    messages::ChatToolCall,
    tools::{self, ParsedTurn, gemma_call, json_in_tags, minicpm_xml, xml_function_params},
};

/// How a template tells the model to write tool calls.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolDialect {
    /// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` (Qwen3).
    JsonInTags,
    /// `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>`
    /// (Qwen3.5 and later, Qwen3-Coder).
    XmlFunctionParams,
    /// `<function name="NAME"><param name="K">V</param></function>` (`MiniCPM5`).
    MiniCpmXml,
    /// `<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>` (Gemma 4).
    GemmaCall,
    /// No calls are parsed; the whole answer is text.
    PlainText,
}

/// How a template marks the model's reasoning.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningDialect {
    /// `<think>...</think>` (Qwen3, `MiniCPM5`).
    ThinkTags,
    /// `<|channel>thought...<channel|>` (Gemma 4).
    GemmaChannel,
    /// No reasoning is split from the answer.
    None,
}

/// The tool and reasoning dialects of one checkpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct TurnFormat {
    /// How the template marks tool calls.
    pub tools: ToolDialect,
    /// How the template marks reasoning.
    pub reasoning: ReasoningDialect,
}

impl TurnFormat {
    /// No tool calls and no reasoning: what an unrecognized template gets.
    pub const PLAIN: Self = Self {
        tools: ToolDialect::PlainText,
        reasoning: ReasoningDialect::None,
    };

    /// Selects each dialect whose markers the template source spells. A
    /// template matching several tool dialects selects none, except that
    /// [`ToolDialect::XmlFunctionParams`] wraps its calls in `<tool_call>`
    /// and so also spells [`ToolDialect::JsonInTags`]' marker.
    #[must_use]
    pub fn from_template(source: &str) -> Self {
        let spells = |markers: &[&str]| markers.iter().all(|marker| source.contains(marker));
        let mut tools: Vec<ToolDialect> = [
            (ToolDialect::JsonInTags, &["<tool_call>"][..]),
            (
                ToolDialect::XmlFunctionParams,
                &["<tool_call>", "<function=", "<parameter="],
            ),
            (
                ToolDialect::MiniCpmXml,
                &["<function name=", "<param name="],
            ),
            (ToolDialect::GemmaCall, &["<|tool_call>", "<tool_call|>"]),
        ]
        .into_iter()
        .filter(|(_, markers)| spells(markers))
        .map(|(dialect, _)| dialect)
        .collect();
        if tools.contains(&ToolDialect::XmlFunctionParams) {
            tools.retain(|&dialect| dialect != ToolDialect::JsonInTags);
        }
        let reasoning = match (
            spells(&["<think>", "</think>"]),
            spells(&["<|channel>", "<channel|>"]),
        ) {
            (true, false) => ReasoningDialect::ThinkTags,
            (false, true) => ReasoningDialect::GemmaChannel,
            _ => ReasoningDialect::None,
        };
        Self {
            tools: match tools.as_slice() {
                [dialect] => *dialect,
                _ => ToolDialect::PlainText,
            },
            reasoning,
        }
    }
}

impl ToolDialect {
    /// The token that ends a turn which issued calls and now awaits their
    /// results, when the dialect has one; it is a stop token, not text.
    #[must_use]
    pub const fn tool_end_marker(self) -> Option<&'static str> {
        match self {
            Self::GemmaCall => Some("<|tool_response>"),
            Self::JsonInTags | Self::XmlFunctionParams | Self::MiniCpmXml | Self::PlainText => None,
        }
    }

    /// `tools` give parameter types to dialects that write every value as
    /// text.
    fn parse(self, answer: &str, tools: &[Value]) -> Result<ParsedTurn, String> {
        match self {
            Self::JsonInTags => json_in_tags::parse(answer),
            Self::XmlFunctionParams => xml_function_params::parse(answer, tools),
            Self::MiniCpmXml => minicpm_xml::parse(answer, tools),
            Self::GemmaCall => gemma_call::parse(answer),
            Self::PlainText => Ok(ParsedTurn {
                text: answer.to_owned(),
                calls: Vec::new(),
            }),
        }
    }
}

impl ReasoningDialect {
    pub(crate) const fn markers(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::ThinkTags => Some(("<think>", "</think>")),
            Self::GemmaChannel => Some(("<|channel>thought", "<channel|>")),
            Self::None => None,
        }
    }

    /// Splits a thinking turn into its reasoning and its answer. The model
    /// may open the block itself, or the template may pre-fill the opening
    /// marker and leave only the close.
    fn split(self, text: &str) -> (&str, &str) {
        let Some((open, close)) = self.markers() else {
            return ("", text);
        };
        let (opened, body) = match text.trim_start().strip_prefix(open) {
            Some(body) => (true, body),
            None => (false, text),
        };
        match body.split_once(close) {
            Some((reasoning, answer)) => (reasoning.trim(), answer.trim_start()),
            None if opened => (body.trim(), ""),
            None => ("", text),
        }
    }

    /// Drops an empty block the model opens and closes on its own while
    /// thinking is off: Gemma 4 writes `<|channel>thought\n<channel|>` after
    /// each tool result. A block with reasoning in it stays text.
    fn strip_empty(self, text: &str) -> &str {
        self.markers()
            .and_then(|(open, close)| text.trim_start().strip_prefix(open)?.split_once(close))
            .filter(|(reasoning, _)| reasoning.trim().is_empty())
            .map_or(text, |(_, answer)| answer.trim_start())
    }
}

/// A finished assistant turn split into reasoning, visible text and tool calls
/// checked against their declared schemas. Each protocol only reshapes it.
#[derive(Clone, Debug, PartialEq)]
pub struct AssistantTurn {
    /// The reasoning, empty when thinking was off or the dialect has none.
    pub reasoning: String,
    /// The visible answer, without reasoning or tool-call markup.
    pub text: String,
    /// The tool calls, in the order the model wrote them.
    pub calls: Vec<ChatToolCall>,
    /// The model ended its turn; otherwise it hit the output limit.
    pub complete: bool,
}

/// Parses one generation. `tools` are template-shaped definitions
/// (`{"type": "function", "function": {...}}`). A truncated tool turn is an
/// error, never a partial call.
///
/// # Errors
///
/// Returns a message for any failure of [`parse_turn_unchecked`], or when a
/// call fails [`check_call`].
pub fn parse_turn(
    format: TurnFormat,
    text: &str,
    tools: &[Value],
    enable_thinking: bool,
    complete: bool,
) -> Result<AssistantTurn, String> {
    let turn = parse_turn_unchecked(format, text, tools, enable_thinking, complete)?;
    for call in &turn.calls {
        check_call(tools, call)?;
    }
    Ok(turn)
}

/// [`parse_turn`] without checking calls against their declarations, for
/// a caller that reports an invalid call back to the model (the agent
/// loop) through [`check_call`] instead of failing the turn.
///
/// # Errors
///
/// Returns a message when the dialect cannot parse the tool-call markup,
/// or when a turn with tool calls is not `complete`.
pub fn parse_turn_unchecked(
    format: TurnFormat,
    text: &str,
    tools: &[Value],
    enable_thinking: bool,
    complete: bool,
) -> Result<AssistantTurn, String> {
    let (reasoning, answer) = if enable_thinking {
        format.reasoning.split(text)
    } else {
        ("", format.reasoning.strip_empty(text))
    };
    let turn = format.tools.parse(answer, tools)?;
    if !turn.calls.is_empty() && !complete {
        return Err("truncated tool turn; no function calls returned".into());
    }
    Ok(AssistantTurn {
        reasoning: reasoning.to_owned(),
        text: turn.text,
        calls: turn
            .calls
            .into_iter()
            .map(|call| ChatToolCall {
                name: call.name,
                arguments: call.arguments,
            })
            .collect(),
        complete,
    })
}

/// Checks one call against the declared `tools`: the tool must exist and
/// its arguments must satisfy the parameter schema.
///
/// # Errors
///
/// Returns a message when the tool is undeclared, its parameter schema
/// does not compile under [`crate::validator`], or the arguments do not
/// satisfy it.
pub fn check_call(tools: &[Value], call: &ChatToolCall) -> Result<(), String> {
    let definition = tools
        .iter()
        .find(|tool| tool["function"]["name"] == call.name)
        .ok_or("model requested an undeclared tool")?;
    tools::validator(&definition["function"]["parameters"])?
        .validate(&call.arguments)
        .map_err(|error| format!("model tool arguments do not match the declared schema: {error}"))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::{
        ReasoningDialect, ToolDialect, TurnFormat, check_call, parse_turn, parse_turn_unchecked,
    };
    use crate::ChatToolCall;

    const QWEN3_TEMPLATE: &str = include_str!("../../../fixtures/qwen3-0.6b/chat-template.jinja");
    const MINICPM5_TEMPLATE: &str =
        include_str!("../../../fixtures/minicpm5-2b/chat-template.jinja");

    const QWEN3: TurnFormat = TurnFormat {
        tools: ToolDialect::JsonInTags,
        reasoning: ReasoningDialect::ThinkTags,
    };

    fn read_file() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"read_file","parameters":{
            "type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}),
        ]
    }

    #[test]
    fn dialects_follow_the_template_and_fail_closed() {
        assert_eq!(TurnFormat::from_template(QWEN3_TEMPLATE), QWEN3);
        assert_eq!(
            TurnFormat::from_template("{{ messages }}"),
            TurnFormat::PLAIN
        );
        assert_eq!(
            TurnFormat::from_template(MINICPM5_TEMPLATE),
            TurnFormat {
                tools: ToolDialect::MiniCpmXml,
                reasoning: ReasoningDialect::ThinkTags,
            }
        );
        // Qwen3.5-0.8B's chat_template.jinja (sha256 273d8e0e...): the
        // lines that spell its call format and its generation prompt.
        let qwen35 = concat!(
            r"{{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}",
            r"{{- '<parameter=' + args_name + '>\n' }}",
            r"{{- '</function>\n</tool_call>' }}",
            r"{{- '<think>\n\n</think>\n\n' }}",
        );
        assert_eq!(
            TurnFormat::from_template(qwen35),
            TurnFormat {
                tools: ToolDialect::XmlFunctionParams,
                reasoning: ReasoningDialect::ThinkTags,
            }
        );
        // gemma-4-12B-it's chat_template.jinja (sha256 ae53464b...).
        let gemma4 = concat!(
            r"{{- '<|channel>thought\n' + thinking_text + '\n<channel|>' -}}",
            r"{{- '<|tool_call>call:' + function['name'] + '{' -}}",
            r"{{- '}<tool_call|>' -}}",
        );
        assert_eq!(
            TurnFormat::from_template(gemma4),
            TurnFormat {
                tools: ToolDialect::GemmaCall,
                reasoning: ReasoningDialect::GemmaChannel,
            }
        );
        // A template spelling two dialects' markers selects neither.
        assert_eq!(
            TurnFormat::from_template(&format!("{QWEN3_TEMPLATE}{MINICPM5_TEMPLATE}")).tools,
            ToolDialect::PlainText
        );
    }

    /// `MiniCPM5-2B`'s greedy output for the `chat_tool_call` reference
    /// case, up to its `<|im_end|>` stop token.
    #[test]
    fn minicpm5_reference_call_parses_and_qwen3_leaves_it_as_text() {
        let text = r#"<function name="read_file"><param name="path">README.md</param></function>"#;
        let minicpm5 = TurnFormat::from_template(MINICPM5_TEMPLATE);
        let turn = parse_turn(minicpm5, text, &read_file(), false, true).unwrap();
        assert_eq!(turn.calls[0].arguments, json!({"path":"README.md"}));
        let qwen3 = parse_turn(QWEN3, text, &read_file(), false, true).unwrap();
        assert!(qwen3.calls.is_empty());
    }

    /// The turns that failed on the Qwen-only parser: each family's call
    /// syntax under its own dialects.
    #[test]
    fn every_family_call_parses_in_its_own_dialect() {
        let format = |tools, reasoning| TurnFormat { tools, reasoning };
        let cases = [
            (
                format(ToolDialect::MiniCpmXml, ReasoningDialect::ThinkTags),
                r#"<function name="read_file"><param name="path">README.md</param></function>"#,
            ),
            (
                format(ToolDialect::XmlFunctionParams, ReasoningDialect::ThinkTags),
                "<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                format(ToolDialect::GemmaCall, ReasoningDialect::GemmaChannel),
                r#"<|tool_call>call:read_file{path:<|"|>README.md<|"|>}<tool_call|>"#,
            ),
        ];
        for (format, text) in cases {
            let turn = parse_turn(format, text, &read_file(), false, true).unwrap();
            assert_eq!(turn.calls.len(), 1, "{text}");
            assert_eq!(turn.calls[0].arguments, json!({"path":"README.md"}));
        }
        let gemma = format(ToolDialect::GemmaCall, ReasoningDialect::GemmaChannel);
        let turn = parse_turn(
            gemma,
            "<|channel>thought\nadd them<channel|>4",
            &[],
            true,
            true,
        )
        .unwrap();
        assert_eq!(
            (turn.reasoning.as_str(), turn.text.as_str()),
            ("add them", "4")
        );
    }

    /// Greedy outputs of gemma-4-12B-it@707f0a3 under transformers, each
    /// without the stop token that ended it.
    #[test]
    fn gemma4_measured_turns_parse() {
        let gemma = TurnFormat {
            tools: ToolDialect::GemmaCall,
            reasoning: ReasoningDialect::GemmaChannel,
        };
        let weather = [
            json!({"type":"function","function":{"name":"get_weather","parameters":{
            "type":"object","required":["city"],"properties":{"city":{"type":"string"},
            "unit":{"type":"string","enum":["celsius","fahrenheit"]}}}}}),
        ];
        // "Weather in Paris?", stopped on <|tool_response> (50).
        let call = r#"<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>"#;
        let turn = parse_turn(gemma, call, &weather, false, true).unwrap();
        assert_eq!(turn.text, "");
        assert_eq!(
            turn.calls,
            [ChatToolCall {
                name: "get_weather".into(),
                arguments: json!({"city":"Paris"}),
            }]
        );
        // After the tool result with thinking off, stopped on <turn|> (106):
        // the model opens and closes an empty channel before answering.
        let answer =
            "<|channel>thought\n<channel|>The current weather in Paris is 18°C with clear skies.";
        let turn = parse_turn(gemma, answer, &weather, false, true).unwrap();
        assert_eq!(
            (turn.reasoning.as_str(), turn.text.as_str()),
            ("", "The current weather in Paris is 18°C with clear skies.")
        );
        // Thinking on ("Is 91 prime?", trace shortened).
        let thought = "<|channel>thought\n91 = 7 x 13.<channel|>No, **91 is not a prime number.**";
        let turn = parse_turn(gemma, thought, &[], true, true).unwrap();
        assert_eq!(
            (turn.reasoning.as_str(), turn.text.as_str()),
            ("91 = 7 x 13.", "No, **91 is not a prime number.**")
        );
        // With thinking off, a channel that holds reasoning stays text.
        let turn = parse_turn(gemma, thought, &[], false, true).unwrap();
        assert_eq!(turn.text, thought);
    }

    #[test]
    fn think_tags_split_reasoning_from_the_answer() {
        let split = |text| ReasoningDialect::ThinkTags.split(text);
        assert_eq!(split("<think>\nadd them\n</think>\n\n4"), ("add them", "4"));
        assert_eq!(split("pre-filled\n</think>\n\n4"), ("pre-filled", "4"));
        assert_eq!(split("<think>\nunfinished"), ("unfinished", ""));
        assert_eq!(split("plain answer"), ("", "plain answer"));
        let channel = |text| ReasoningDialect::GemmaChannel.split(text);
        assert_eq!(
            channel("<|channel>thought\nadd them<channel|>4"),
            ("add them", "4")
        );
        // The Gemma 4 template pre-fills the opening marker when thinking.
        assert_eq!(channel("add them\n<channel|>4"), ("add them", "4"));
        assert_eq!(channel("4"), ("", "4"));
        assert_eq!(
            ReasoningDialect::None.split("<think>x</think>y"),
            ("", "<think>x</think>y")
        );
    }

    #[test]
    fn turns_validate_calls_against_declared_tools() {
        let call = "<think>\nread it\n</think>\n\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.md\"}}\n</tool_call>";
        let turn = parse_turn(QWEN3, call, &read_file(), true, true).unwrap();
        assert_eq!(turn.reasoning, "read it");
        assert_eq!(turn.text, "");
        assert_eq!(
            turn.calls,
            [ChatToolCall {
                name: "read_file".into(),
                arguments: json!({"path":"a.md"}),
            }]
        );
        assert!(parse_turn(QWEN3, call, &read_file(), true, false).is_err());
        assert!(parse_turn(QWEN3, call, &[], true, true).is_err());
        let wrong =
            "<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": 5}}</tool_call>";
        assert!(parse_turn(QWEN3, wrong, &read_file(), false, true).is_err());
        // Unchecked parsing keeps the call for a caller that reports the
        // error to the model; check_call gives that error.
        let kept = parse_turn_unchecked(QWEN3, wrong, &read_file(), false, true).unwrap();
        assert!(
            check_call(&read_file(), &kept.calls[0])
                .is_err_and(|error| error.contains("declared schema"))
        );
        assert!(parse_turn_unchecked(QWEN3, call, &read_file(), true, false).is_err());
        // Plain text keeps any markup the model writes as text.
        let plain = parse_turn(TurnFormat::PLAIN, call, &read_file(), true, true).unwrap();
        assert_eq!((plain.reasoning.as_str(), plain.text.as_str()), ("", call));
        assert!(plain.calls.is_empty());
    }
}
