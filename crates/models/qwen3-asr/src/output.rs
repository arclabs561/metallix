//! Turning Qwen3-ASR's decoded text into a language and a transcript.
//!
//! This follows `parse_asr_output` and `detect_and_fix_repetitions` in
//! `qwen_asr/inference/utils.py` at
//! <https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/utils.py>,
//! so transcripts compare equal to the reference package's. Python indexes
//! strings by code point, so this works on `char`s.

const ASR_TEXT_TAG: &str = "<asr_text>";
const LANGUAGE_PREFIX: &str = "language ";
/// Runs longer than this many repeats are collapsed.
const REPEAT_THRESHOLD: usize = 20;
/// Longest repeated pattern, in characters, that is collapsed.
const MAX_PATTERN: usize = 20;

/// A parsed transcription.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transcript {
    /// Detected or forced language name, empty when the model reported none.
    pub language: String,
    /// The transcript text.
    pub text: String,
}

/// Parses the decoded assistant output.
///
/// With `forced_language`, the prompt already ended in
/// `language X<asr_text>`, so the whole output is text. Otherwise the model
/// writes `language X<asr_text>text`, and `language None` means it heard no
/// speech.
#[must_use]
pub fn parse_output(raw: &str, forced_language: Option<&str>) -> Transcript {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Transcript::empty();
    }
    let fixed = fix_repetitions(trimmed);
    if let Some(language) = forced_language.filter(|language| !language.is_empty()) {
        return Transcript {
            language: language.to_owned(),
            text: fixed,
        };
    }
    let Some((meta, text)) = fixed.split_once(ASR_TEXT_TAG) else {
        return Transcript {
            language: String::new(),
            text: fixed.trim().to_owned(),
        };
    };
    if meta.to_lowercase().contains("language none") {
        return Transcript {
            language: String::new(),
            text: text.trim().to_owned(),
        };
    }
    let language = meta
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .filter(|line| line.to_lowercase().starts_with(LANGUAGE_PREFIX))
        .map(|line| normalize_language_name(line[LANGUAGE_PREFIX.len()..].trim()))
        .unwrap_or_default();
    Transcript {
        language,
        text: text.trim().to_owned(),
    }
}

impl Transcript {
    fn empty() -> Self {
        Self {
            language: String::new(),
            text: String::new(),
        }
    }
}

/// `normalize_language_name`: first character upper case, the rest lower.
#[must_use]
pub fn normalize_language_name(language: &str) -> String {
    let mut characters = language.trim().chars();
    characters.next().map_or_else(String::new, |first| {
        first
            .to_uppercase()
            .chain(characters.flat_map(char::to_lowercase))
            .collect()
    })
}

fn fix_repetitions(text: &str) -> String {
    let characters: Vec<char> = text.chars().collect();
    let collapsed = fix_char_repeats(&characters);
    fix_pattern_repeats(&collapsed).into_iter().collect()
}

/// Replaces a run of more than [`REPEAT_THRESHOLD`] equal characters with one.
fn fix_char_repeats(text: &[char]) -> Vec<char> {
    let mut out = Vec::with_capacity(text.len());
    let mut index = 0;
    while index < text.len() {
        let mut count = 1;
        while index + count < text.len() && text[index + count] == text[index] {
            count += 1;
        }
        if count > REPEAT_THRESHOLD {
            out.push(text[index]);
        } else {
            out.extend_from_slice(&text[index..index + count]);
        }
        index += count;
    }
    out
}

/// Replaces the first pattern of 1 to [`MAX_PATTERN`] characters repeated at
/// least [`REPEAT_THRESHOLD`] times with one copy, then repeats on the rest.
fn fix_pattern_repeats(text: &[char]) -> Vec<char> {
    let length = text.len();
    let minimum = REPEAT_THRESHOLD * 2;
    if length < minimum {
        return text.to_vec();
    }
    let mut out = Vec::with_capacity(length);
    let mut index = 0;
    while index <= length - minimum {
        for width in 1..=MAX_PATTERN {
            if index + width * REPEAT_THRESHOLD > length {
                break;
            }
            let pattern = &text[index..index + width];
            let repeated = (1..REPEAT_THRESHOLD).all(|repeat| {
                let start = index + repeat * width;
                &text[start..start + width] == pattern
            });
            if repeated {
                let mut end = index + REPEAT_THRESHOLD * width;
                while end + width <= length && &text[end..end + width] == pattern {
                    end += width;
                }
                out.extend_from_slice(pattern);
                out.extend(fix_pattern_repeats(&text[end..]));
                return out;
            }
        }
        out.push(text[index]);
        index += 1;
    }
    out.extend_from_slice(&text[index..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(language: &str, text: &str) -> Transcript {
        Transcript {
            language: language.to_owned(),
            text: text.to_owned(),
        }
    }

    #[test]
    fn splits_the_language_tag_from_the_text() {
        assert_eq!(
            parse_output("language English<asr_text> Hello there. ", None),
            transcript("English", "Hello there.")
        );
        assert_eq!(
            parse_output("language cHINese\n\n<asr_text>你好", None),
            transcript("Chinese", "你好")
        );
        assert_eq!(
            parse_output("plain words", None),
            transcript("", "plain words")
        );
        assert_eq!(
            parse_output("language None<asr_text>", None),
            transcript("", "")
        );
        assert_eq!(parse_output("   ", None), transcript("", ""));
    }

    #[test]
    fn a_forced_language_makes_the_whole_output_text() {
        assert_eq!(
            parse_output(" Hello <asr_text>there", Some("English")),
            transcript("English", "Hello <asr_text>there")
        );
    }

    #[test]
    fn collapses_runaway_repetition_like_the_reference() {
        // 25 of one character becomes one; 20 or fewer are kept.
        let long = format!("a{}b", "x".repeat(25));
        assert_eq!(parse_output(&long, Some("English")).text, "axb");
        let short = format!("a{}b", "x".repeat(20));
        assert_eq!(parse_output(&short, Some("English")).text, short);
        // A two-character pattern repeated 22 times, then a tail.
        let pattern = format!("ok{}end", "ha".repeat(22));
        assert_eq!(parse_output(&pattern, Some("English")).text, "okhaend");
        // Below the threshold nothing changes.
        let few = format!("ok{}end", "ha".repeat(19));
        assert_eq!(parse_output(&few, Some("English")).text, few);
    }
}
