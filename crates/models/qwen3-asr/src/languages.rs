//! OpenAI-style language codes to the names Qwen3-ASR prompts with.
//!
//! The transcription API takes ISO-639-1 codes (`en`); Qwen3-ASR writes and
//! reads English names (`language English<asr_text>`). Names come from the
//! released checkpoints' `support_languages`.

/// ISO 639 code and the checkpoint's name for each listed language.
const CODES: [(&str, &str); 30] = [
    ("zh", "Chinese"),
    ("en", "English"),
    ("yue", "Cantonese"),
    ("ar", "Arabic"),
    ("de", "German"),
    ("fr", "French"),
    ("es", "Spanish"),
    ("pt", "Portuguese"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ko", "Korean"),
    ("ru", "Russian"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("ja", "Japanese"),
    ("tr", "Turkish"),
    ("hi", "Hindi"),
    ("ms", "Malay"),
    ("nl", "Dutch"),
    ("sv", "Swedish"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("fil", "Filipino"),
    ("fa", "Persian"),
    ("el", "Greek"),
    ("ro", "Romanian"),
    ("hu", "Hungarian"),
    ("mk", "Macedonian"),
];

/// Resolves a request's language, an ISO 639 code or a name in any case, to
/// one of `supported` (the checkpoint's list), or `None` if it is not there.
#[must_use]
pub fn resolve<'a>(requested: &str, supported: &'a [String]) -> Option<&'a str> {
    let requested = requested.trim();
    let name = CODES
        .iter()
        .find(|(code, _)| code.eq_ignore_ascii_case(requested))
        .map_or(requested, |(_, name)| name);
    // `tl` is Tagalog's ISO 639-1 code; Filipino is its standardized form.
    let name = if requested.eq_ignore_ascii_case("tl") {
        "Filipino"
    } else {
        name
    };
    supported
        .iter()
        .find(|known| known.eq_ignore_ascii_case(name))
        .map(String::as_str)
}

/// The ISO 639 code for a checkpoint language name, for API responses.
#[must_use]
pub fn code(name: &str) -> Option<&'static str> {
    CODES
        .iter()
        .find(|(_, known)| known.eq_ignore_ascii_case(name))
        .map(|(code, _)| *code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_codes_and_names_to_the_checkpoint_list() {
        let supported: Vec<String> = ["English", "Chinese", "Filipino"]
            .map(String::from)
            .to_vec();
        assert_eq!(resolve("en", &supported), Some("English"));
        assert_eq!(resolve("EN", &supported), Some("English"));
        assert_eq!(resolve("english", &supported), Some("English"));
        assert_eq!(resolve("tl", &supported), Some("Filipino"));
        // A code the model knows but this checkpoint does not list.
        assert_eq!(resolve("de", &supported), None);
        assert_eq!(resolve("xx", &supported), None);
        assert_eq!(code("Chinese"), Some("zh"));
    }
}
