//! An assistant turn streamed as typed deltas while it is generated.
//!
//! [`TurnStream`] reads decoded text as it arrives and releases reasoning
//! and visible text only once no later text can change them: a suffix that
//! could still become a reasoning or tool-call marker is held back, and a
//! tool call is never streamed; it arrives whole in the final
//! [`AssistantTurn`]. Every released delta is a prefix of what
//! [`parse_turn_unchecked`] returns for the whole text, however the text
//! was chunked.

use serde_json::Value;

use crate::turn::{AssistantTurn, ToolDialect, TurnFormat, parse_turn_unchecked};

/// A piece of a turn a protocol may show as it arrives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TurnDelta {
    Reasoning(String),
    Text(String),
    /// A token arrived whose text is still held back (markup, or text that
    /// may become markup). It carries nothing to show; a protocol can use
    /// it to notice a disconnected client.
    Held,
}

/// Splits one turn's decoded text into [`TurnDelta`]s.
pub struct TurnStream {
    format: TurnFormat,
    enable_thinking: bool,
    raw: String,
    reasoning_sent: String,
    text_sent: String,
}

impl TurnStream {
    #[must_use]
    pub fn new(format: TurnFormat, enable_thinking: bool) -> Self {
        Self {
            format,
            enable_thinking,
            raw: String::new(),
            reasoning_sent: String::new(),
            text_sent: String::new(),
        }
    }

    /// Adds decoded text and returns what it settled.
    ///
    /// # Panics
    ///
    /// If settled text stops extending what was released, which the
    /// property tests rule out.
    pub fn push(&mut self, decoded: &str) -> Vec<TurnDelta> {
        self.raw.push_str(decoded);
        // shortcut: re-reads the whole turn on every push, O(n^2) in its
        // length; upgrade to incremental state when long reasoning turns
        // show up in profiles.
        let (reasoning, text) = settled(self.format, self.enable_thinking, &self.raw);
        let (reasoning, text) = (reasoning.to_owned(), text.to_owned());
        self.release(&reasoning, &text)
            .expect("settled text extends what was released")
    }

    /// Parses the whole turn and returns it with the deltas not yet
    /// released. Calls are not checked against their declarations; see
    /// [`crate::check_call`].
    pub fn finish(
        mut self,
        tools: &[Value],
        complete: bool,
    ) -> Result<(AssistantTurn, Vec<TurnDelta>), String> {
        let turn = parse_turn_unchecked(
            self.format,
            &self.raw,
            tools,
            self.enable_thinking,
            complete,
        )?;
        let deltas = self.release(&turn.reasoning, &turn.text)?;
        Ok((turn, deltas))
    }

    /// Releases the part of `reasoning` and `text` not yet sent.
    fn release(&mut self, reasoning: &str, text: &str) -> Result<Vec<TurnDelta>, String> {
        let mut deltas = Vec::new();
        for (settled, sent, delta) in [
            (
                reasoning,
                &mut self.reasoning_sent,
                TurnDelta::Reasoning as fn(String) -> TurnDelta,
            ),
            (text, &mut self.text_sent, TurnDelta::Text),
        ] {
            let fresh = settled
                .strip_prefix(sent.as_str())
                .ok_or("streamed turn diverged from its parse")?;
            if !fresh.is_empty() {
                sent.push_str(fresh);
                deltas.push(delta(fresh.to_owned()));
            }
        }
        Ok(deltas)
    }
}

/// The reasoning and visible text of `raw` that no continuation can change.
fn settled(format: TurnFormat, enable_thinking: bool, raw: &str) -> (&str, &str) {
    let markers = format.reasoning.markers();
    let (reasoning, answer) = match markers.filter(|_| enable_thinking) {
        None => ("", raw),
        Some((open, close)) => {
            let trimmed = raw.trim_start();
            if let Some(body) = trimmed.strip_prefix(open) {
                if let Some(end) = body.find(close) {
                    (body[..end].trim(), body[end + close.len()..].trim_start())
                } else {
                    // Still thinking; trailing whitespace may yet be trimmed.
                    let safe = &body[..body.len() - held_suffix(body, &[close])];
                    return (safe.trim(), "");
                }
            } else if open.starts_with(trimmed) {
                return ("", "");
            } else if let Some(end) = raw.find(close) {
                // The template opened the block itself.
                (raw[..end].trim(), raw[end + close.len()..].trim_start())
            } else {
                // Reasoning if a close follows, otherwise the answer.
                return ("", "");
            }
        }
    };
    let answer = if enable_thinking {
        answer
    } else {
        match markers {
            Some((open, close)) => match without_empty_block(answer, open, close) {
                Some(answer) => answer,
                None => return (reasoning, ""),
            },
            None => answer,
        }
    };
    let text = match call_open(format.tools) {
        Some(open) => match answer.find(open) {
            Some(start) => &answer[..start],
            None => &answer[..answer.len() - held_suffix(answer, &[open])],
        },
        None => answer,
    };
    (reasoning, text)
}

/// `answer` without an empty reasoning block it opens and closes, as
/// [`crate::parse_turn`] drops one when thinking is off; `None` while the
/// text could still turn out to be such a block.
fn without_empty_block<'a>(answer: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let trimmed = answer.trim_start();
    if open.starts_with(trimmed) {
        return None;
    }
    let Some(body) = trimmed.strip_prefix(open) else {
        return Some(answer);
    };
    match body.find(close) {
        Some(end) if body[..end].trim().is_empty() => Some(body[end + close.len()..].trim_start()),
        None if body[..body.len() - held_suffix(body, &[close])]
            .trim()
            .is_empty() =>
        {
            None
        }
        _ => Some(answer),
    }
}

/// The marker that opens a call in `dialect`.
const fn call_open(dialect: ToolDialect) -> Option<&'static str> {
    match dialect {
        ToolDialect::JsonInTags => Some("<tool_call>"),
        ToolDialect::XmlFunctionParams => Some(crate::tools::xml_function_params::OPEN),
        ToolDialect::MiniCpmXml => Some(crate::tools::minicpm_xml::OPEN),
        ToolDialect::GemmaCall => Some(crate::tools::gemma_call::OPEN),
        ToolDialect::PlainText => None,
    }
}

/// The length of the longest suffix of `text` that is a proper prefix of
/// one of `markers`, which more text could complete.
fn held_suffix(text: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .flat_map(|marker| (1..marker.len()).map(move |length| &marker[..length]))
        .filter(|prefix| text.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use serde_json::json;

    use super::{TurnDelta, TurnStream};
    use crate::turn::{ReasoningDialect, ToolDialect, TurnFormat, parse_turn_unchecked};

    const FORMATS: [TurnFormat; 5] = [
        TurnFormat {
            tools: ToolDialect::JsonInTags,
            reasoning: ReasoningDialect::ThinkTags,
        },
        TurnFormat {
            tools: ToolDialect::XmlFunctionParams,
            reasoning: ReasoningDialect::ThinkTags,
        },
        TurnFormat {
            tools: ToolDialect::MiniCpmXml,
            reasoning: ReasoningDialect::ThinkTags,
        },
        TurnFormat {
            tools: ToolDialect::GemmaCall,
            reasoning: ReasoningDialect::GemmaChannel,
        },
        TurnFormat::PLAIN,
    ];

    /// Pieces of every dialect's markup, partial markers and plain text.
    const PIECES: [&str; 24] = [
        "<think>",
        "</think>",
        "<|channel>thought",
        "<channel|>",
        "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>",
        "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>",
        "<function name=\"f\"><param name=\"a\">1</param></function>",
        "<|tool_call>call:f{a:1}<tool_call|>",
        "<tool_",
        "<|chan",
        "</thi",
        "<function",
        "<",
        " ",
        "\n",
        "  \n",
        "Paris",
        "is 4.",
        "a",
        "é",
        "think",
        "|>",
        ">",
        "x y",
    ];

    fn stream(
        format: TurnFormat,
        thinking: bool,
        chunks: &[&str],
    ) -> Result<(String, String, (String, String)), String> {
        let mut stream = TurnStream::new(format, thinking);
        let mut deltas = Vec::new();
        for chunk in chunks {
            deltas.extend(stream.push(chunk));
        }
        let (turn, last) = stream.finish(&[], true)?;
        deltas.extend(last);
        let mut reasoning = String::new();
        let mut text = String::new();
        for delta in deltas {
            match delta {
                TurnDelta::Reasoning(piece) => reasoning.push_str(&piece),
                TurnDelta::Text(piece) => text.push_str(&piece),
                TurnDelta::Held => unreachable!("the stream itself never holds a token"),
            }
        }
        Ok((reasoning, text, (turn.reasoning, turn.text)))
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(512))]

        /// However a turn is chunked, the released deltas add up to the
        /// parse of the whole text, in every dialect, thinking on or off.
        #[test]
        fn deltas_add_up_to_the_parse_for_any_chunking(
            pieces in prop::collection::vec(0..PIECES.len(), 0..12),
            cuts in prop::collection::vec(any::<prop::sample::Index>(), 0..8),
            format in 0..FORMATS.len(),
            thinking in any::<bool>(),
        ) {
            let format = FORMATS[format];
            let text: String = pieces.iter().map(|&piece| PIECES[piece]).collect();
            let whole = parse_turn_unchecked(format, &text, &[], thinking, true);
            let mut boundaries: Vec<usize> = cuts
                .iter()
                .map(|cut| cut.index(text.len() + 1))
                .filter(|&cut| text.is_char_boundary(cut))
                .collect();
            boundaries.sort_unstable();
            let mut chunks = Vec::new();
            let mut start = 0;
            for end in boundaries {
                chunks.push(&text[start..end]);
                start = end;
            }
            chunks.push(&text[start..]);
            match (whole, stream(format, thinking, &chunks)) {
                (Ok(turn), Ok((reasoning, text, parsed))) => {
                    prop_assert_eq!(&parsed, &(turn.reasoning.clone(), turn.text.clone()));
                    prop_assert_eq!(reasoning, turn.reasoning);
                    prop_assert_eq!(text, turn.text);
                }
                (Err(whole), Err(streamed)) => prop_assert_eq!(whole, streamed),
                (whole, streamed) => prop_assert!(false, "{whole:?} vs {streamed:?}"),
            }
        }
    }

    /// Reasoning and text arrive before the turn ends; a tool call does not.
    #[test]
    fn reasoning_and_text_stream_while_markers_are_held() {
        let qwen = FORMATS[0];
        let mut stream = TurnStream::new(qwen, true);
        assert_eq!(
            stream.push("<think>\nadd"),
            [TurnDelta::Reasoning("add".into())]
        );
        assert_eq!(
            stream.push(" them\n</th"),
            [TurnDelta::Reasoning(" them".into())]
        );
        assert_eq!(
            stream.push("ink>\n\nIt is 4. <tool_"),
            [TurnDelta::Text("It is 4. ".into())]
        );
        assert_eq!(stream.push("call>\n{\"name\": \"f\""), []);
        assert_eq!(stream.push(", \"arguments\": {}}\n</tool_call>"), []);
        let (turn, last) = stream.finish(&[], true).unwrap();
        assert_eq!(last, []);
        assert_eq!(turn.calls[0].arguments, json!({}));
        // Gemma, thinking off: the empty channel after a tool result is held,
        // then dropped.
        let mut gemma = TurnStream::new(FORMATS[3], false);
        assert_eq!(gemma.push("<|channel>thought\n<chan"), []);
        assert_eq!(
            gemma.push("nel|>It is 18°C."),
            [TurnDelta::Text("It is 18°C.".into())]
        );
    }
}
