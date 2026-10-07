//! Typed Julia requests, mirroring the pinned source's `typed.predict_typed`
//! rows, `data.sequence` serialization under the published strict inference
//! policy, and its option-softmax readout.

use std::{collections::HashMap, fmt, fmt::Write as _};

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Value, json};
use thiserror::Error;

/// `max_length` from the pinned `inference-policy.json`.
pub const MAX_LENGTH: usize = 8192;
/// `head_length` from the pinned `inference-policy.json`.
pub const HEAD_LENGTH: usize = 512;
const OPTION_TOKENS: usize = 48;
const MIN_OPTIONS: usize = 2;
const MAX_OPTIONS: usize = 20;
const MAX_QUESTIONS: usize = 16;

/// A typed request the strict inference policy rejects, or a tokenizer
/// failure, with a human-readable reason.
#[derive(Debug, Error, PartialEq, Eq)]
#[error("{0}")]
pub struct JuliaRequestError(String);

fn fail<T>(message: impl Into<String>) -> Result<T, JuliaRequestError> {
    Err(JuliaRequestError(message.into()))
}

/// A JSON value that keeps object member order, as Python's `json.loads` does.
///
/// Duplicate keys keep their first position and last value, like a Python
/// dict. Integers outside the 64-bit range are parsed as floats by
/// `serde_json`, unlike Python; such states serialize differently.
#[derive(Clone, Debug, PartialEq)]
pub enum OrderedJson {
    /// JSON `null`.
    Null,
    /// JSON `true` or `false`.
    Bool(bool),
    /// An integer that fits in a signed or unsigned 64-bit integer.
    Int(i128),
    /// Any other number.
    Float(f64),
    /// A string.
    String(String),
    /// An array, in order.
    Array(Vec<OrderedJson>),
    /// An object's members in first-seen key order.
    Object(Vec<(String, OrderedJson)>),
}

impl<'de> Deserialize<'de> for OrderedJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct OrderedVisitor;
        impl<'de> Visitor<'de> for OrderedVisitor {
            type Value = OrderedJson;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Int(v.into()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Int(v.into()))
            }
            fn visit_f64<E>(self, v: f64) -> Result<OrderedJson, E> {
                Ok(OrderedJson::Float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<OrderedJson, E> {
                Ok(OrderedJson::String(v.to_owned()))
            }
            fn visit_string<E>(self, v: String) -> Result<OrderedJson, E> {
                Ok(OrderedJson::String(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<OrderedJson, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element::<OrderedJson>()? {
                    items.push(item);
                }
                Ok(OrderedJson::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<OrderedJson, A::Error> {
                let mut members: Vec<(String, OrderedJson)> = Vec::new();
                let mut index: HashMap<String, usize> = HashMap::new();
                while let Some((key, value)) = map.next_entry::<String, OrderedJson>()? {
                    if let Some(&at) = index.get(&key) {
                        members[at].1 = value;
                    } else {
                        index.insert(key.clone(), members.len());
                        members.push((key, value));
                    }
                }
                Ok(OrderedJson::Object(members))
            }
        }
        deserializer.deserialize_any(OrderedVisitor)
    }
}

impl OrderedJson {
    fn get(&self, key: &str) -> Option<&OrderedJson> {
        match self {
            Self::Object(members) => members.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Python `json.dumps(value, ensure_ascii=False)` with default separators.
    #[must_use]
    pub fn to_python_json(&self) -> String {
        let mut out = String::new();
        self.write_python(&mut out);
        out
    }

    fn write_python(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(true) => out.push_str("true"),
            Self::Bool(false) => out.push_str("false"),
            Self::Int(v) => {
                let _ = write!(out, "{v}");
            }
            Self::Float(v) => out.push_str(&python_float_repr(*v)),
            Self::String(s) => write_python_string(s, out),
            Self::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    item.write_python(out);
                }
                out.push(']');
            }
            Self::Object(members) => {
                out.push('{');
                for (i, (key, value)) in members.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    write_python_string(key, out);
                    out.push_str(": ");
                    value.write_python(out);
                }
                out.push('}');
            }
        }
    }
}

fn write_python_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if u32::from(c) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `repr(float)`: shortest round-trip digits, fixed notation for
/// decimal exponents in `-4..16`, otherwise `d.ddde+XX`.
fn python_float_repr(x: f64) -> String {
    if x == 0.0 {
        return String::from(if x.is_sign_negative() { "-0.0" } else { "0.0" });
    }
    let sign = if x < 0.0 { "-" } else { "" };
    // Rust's `{:e}` also prints the shortest round-trip digits.
    let scientific = format!("{:e}", x.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("LowerExp always has an exponent");
    let exponent: i32 = exponent.parse().expect("LowerExp exponent is an integer");
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    if (-4..16).contains(&exponent) {
        if exponent < 0 {
            let zeros = "0".repeat(exponent.unsigned_abs() as usize - 1);
            return format!("{sign}0.{zeros}{digits}");
        }
        let point = exponent.unsigned_abs() as usize + 1;
        if digits.len() <= point {
            let zeros = "0".repeat(point - digits.len());
            format!("{sign}{digits}{zeros}.0")
        } else {
            format!("{sign}{}.{}", &digits[..point], &digits[point..])
        }
    } else {
        let mantissa = if digits.len() == 1 {
            digits
        } else {
            format!("{}.{}", &digits[..1], &digits[1..])
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{sign}{mantissa}e{exponent_sign}{:02}",
            exponent.unsigned_abs()
        )
    }
}

/// The kind of answer a question asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuestionType {
    /// One of the caller's named options.
    Choice,
    /// The probability-weighted index over an ordered rubric.
    Score,
    /// The probability of the `true` option.
    Noul,
}

impl QuestionType {
    /// The request's `type` string for this kind.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    /// Source `QTYPES` row of the type embedding.
    #[must_use]
    pub fn qtype(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }
}

/// One named question as the source turns it into a `sequence` row.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedRow {
    /// The question's ID in the request.
    pub name: String,
    /// The kind of answer it asks for.
    pub kind: QuestionType,
    /// Answer IDs in option order.
    pub keys: Vec<String>,
    /// The question's `instructions` text.
    pub question: String,
    /// Option descriptions scored by the model, aligned with `keys`.
    pub options: Vec<String>,
}

/// A parsed typed request: its state and one row per question.
#[derive(Clone, Debug, PartialEq)]
pub struct TypedRequest {
    /// `state` as `sequence` renders it: strings verbatim, JSON via Python `json.dumps`.
    pub state_text: String,
    /// The questions, in request order.
    pub rows: Vec<TypedRow>,
}

/// Parses `{"state": ..., "questions": {name: {type, instructions, criteria}}}`,
/// keeping caller order for questions and choice criteria.
///
/// # Errors
///
/// Returns [`JuliaRequestError`] when the bytes are not a JSON object, the
/// state is not a string, array or object, `questions` is not an object of
/// 1 through 16 questions, or a question has an unknown type, non-text
/// instructions, or other than 2 through 20 nonempty option descriptions.
pub fn parse_typed_request(bytes: &[u8]) -> Result<TypedRequest, JuliaRequestError> {
    let request: OrderedJson = serde_json::from_slice(bytes)
        .map_err(|error| JuliaRequestError(format!("request JSON could not be parsed: {error}")))?;
    if !matches!(request, OrderedJson::Object(_)) {
        return fail("request must be a JSON object");
    }
    let state_text = match request.get("state") {
        Some(OrderedJson::String(text)) => text.clone(),
        Some(state @ (OrderedJson::Array(_) | OrderedJson::Object(_))) => state.to_python_json(),
        _ => return fail("state must be a string, JSON array, or JSON object"),
    };
    let Some(OrderedJson::Object(questions)) = request.get("questions") else {
        return fail("questions must be a JSON object");
    };
    if questions.is_empty() || questions.len() > MAX_QUESTIONS {
        return fail(format!(
            "request requires 1 through {MAX_QUESTIONS} questions; received {}",
            questions.len()
        ));
    }
    let rows = questions
        .iter()
        .map(|(name, question)| typed_row(name, question))
        .collect::<Result<_, _>>()?;
    Ok(TypedRequest { state_text, rows })
}

fn typed_row(name: &str, question: &OrderedJson) -> Result<TypedRow, JuliaRequestError> {
    if name.is_empty() || !matches!(question, OrderedJson::Object(_)) {
        return fail("questions require nonempty IDs and question objects");
    }
    let criteria = question.get("criteria");
    let (kind, pairs): (_, Vec<(String, &OrderedJson)>) = match question.get("type") {
        Some(OrderedJson::String(kind)) if kind == "choice" => {
            let Some(OrderedJson::Object(members)) = criteria else {
                return fail(format!(
                    "{name:?}: choice criteria must map IDs to descriptions"
                ));
            };
            if members.iter().any(|(key, _)| key.is_empty()) {
                return fail(format!("{name:?}: choice criteria IDs must be nonempty"));
            }
            let pairs = members.iter().map(|(k, v)| (k.clone(), v)).collect();
            (QuestionType::Choice, pairs)
        }
        Some(OrderedJson::String(kind)) if kind == "score" => {
            let Some(OrderedJson::Array(items)) = criteria else {
                return fail(format!("{name:?}: score requires an ordered rubric"));
            };
            let pairs = items
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v))
                .collect();
            (QuestionType::Score, pairs)
        }
        Some(OrderedJson::String(kind)) if kind == "noul" => {
            // Without criteria the source scores the literal labels "false" and "true".
            let literal = [
                OrderedJson::String("false".into()),
                OrderedJson::String("true".into()),
            ];
            let labels = match criteria {
                None | Some(OrderedJson::Null) => [&literal[0], &literal[1]],
                Some(c @ OrderedJson::Object(members)) if members.len() == 2 => {
                    match (c.get("false"), c.get("true")) {
                        (Some(f), Some(t)) => [f, t],
                        _ => {
                            return fail(format!(
                                "{name:?}: noul criteria must map false and true"
                            ));
                        }
                    }
                }
                _ => return fail(format!("{name:?}: noul criteria must map false and true")),
            };
            return labelled_row(
                name,
                QuestionType::Noul,
                question,
                vec![("false".into(), labels[0]), ("true".into(), labels[1])],
            );
        }
        _ => return fail(format!("{name:?}: type must be choice, score, or noul")),
    };
    labelled_row(name, kind, question, pairs)
}

fn labelled_row(
    name: &str,
    kind: QuestionType,
    question: &OrderedJson,
    pairs: Vec<(String, &OrderedJson)>,
) -> Result<TypedRow, JuliaRequestError> {
    let question_text = match question.get("instructions") {
        Some(OrderedJson::String(text)) => text.clone(),
        _ => return fail(format!("{name:?}: instructions must be text")),
    };
    if !(MIN_OPTIONS..=MAX_OPTIONS).contains(&pairs.len()) {
        return fail(format!(
            "{name:?}: options must contain {MIN_OPTIONS} through {MAX_OPTIONS} descriptions"
        ));
    }
    let mut keys = Vec::with_capacity(pairs.len());
    let mut options = Vec::with_capacity(pairs.len());
    for (key, value) in pairs {
        match value {
            OrderedJson::String(text) if !text.is_empty() => {
                keys.push(key);
                options.push(text.clone());
            }
            _ => {
                return fail(format!(
                    "{name:?}: option descriptions must be nonempty strings"
                ));
            }
        }
    }
    Ok(TypedRow {
        name: name.to_owned(),
        kind,
        keys,
        question: question_text,
        options,
    })
}

/// Tokenizer IDs and the literal mask string that `sequence` relies on.
#[derive(Clone, Debug)]
pub struct SpecialTokens {
    /// The mask token's text. Request text that contains it is rejected,
    /// because the model reads the mask token as an option marker.
    pub mask_text: String,
    /// ID of the mask token that starts each option.
    pub mask: u32,
    /// ID of the token that starts the sequence.
    pub cls: u32,
    /// ID of the separator token.
    pub sep: u32,
}

/// Serialized model input for one row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedRow {
    /// Token IDs of the whole sequence, at most [`MAX_LENGTH`].
    pub ids: Vec<u32>,
    /// Position of each option's mask token in `ids`, in option order.
    pub markers: Vec<usize>,
    /// The type-embedding row, from [`QuestionType::qtype`].
    pub qtype: usize,
}

/// Source `sequence(tokenizer, row, MAX_LENGTH, HEAD_LENGTH, strict=True)`.
///
/// `encode` must tokenize without special tokens. Strict mode rejects every
/// input the source would sanitize or truncate, so no clean/truncate branch
/// can change the result.
///
/// # Errors
///
/// Returns [`JuliaRequestError`] when any request text contains the mask
/// token's text, `encode` fails, an option encodes to more than 48 tokens,
/// the question and options pass the [`HEAD_LENGTH`] budget, or the state
/// does not fit in [`MAX_LENGTH`].
pub fn sequence(
    mut encode: impl FnMut(&str) -> Result<Vec<u32>, String>,
    special: &SpecialTokens,
    row: &TypedRow,
    state_text: &str,
) -> Result<EncodedRow, JuliaRequestError> {
    let mut encode = |text: &str| encode(text).map_err(JuliaRequestError);
    let name = &row.name;
    if std::iter::once(state_text)
        .chain(std::iter::once(row.question.as_str()))
        .chain(row.options.iter().map(String::as_str))
        .any(|text| text.contains(&special.mask_text))
    {
        return fail(format!("{name:?}: reserved model marker in request"));
    }
    let head = encode(&format!("{} question: {}", row.kind.name(), row.question))?;
    let mut options = Vec::with_capacity(row.options.len());
    for option in &row.options {
        let ids = encode(&format!(" {option}"))?;
        if ids.len() > OPTION_TOKENS {
            return fail(format!(
                "{name:?}: option exceeds {OPTION_TOKENS}-token model contract"
            ));
        }
        options.push(std::iter::once(special.mask).chain(ids).collect::<Vec<_>>());
    }
    // Strict mode: any option truncation (budget below 16) or head overflow is an error.
    let marked: usize = options.iter().map(Vec::len).sum();
    if marked + 16 > HEAD_LENGTH || head.len() > HEAD_LENGTH - marked {
        return fail(format!(
            "{name:?}: question/options exceed lossless head budget"
        ));
    }
    let mut ids = Vec::with_capacity(HEAD_LENGTH);
    ids.push(special.cls);
    ids.extend(head);
    ids.push(special.sep);
    let mut markers = Vec::with_capacity(options.len());
    for option in options {
        markers.push(ids.len());
        ids.extend(option);
    }
    ids.push(special.sep);
    let state = encode(state_text)?;
    // Source: room = max_length - len(ids) - 1 must be at least 1 and hold the whole state.
    if ids.len() + 2 > MAX_LENGTH || ids.len() + 1 + state.len() > MAX_LENGTH {
        return fail(format!(
            "{name:?}: game state exceeds lossless context budget"
        ));
    }
    ids.extend(state);
    ids.push(special.sep);
    Ok(EncodedRow {
        ids,
        markers,
        qtype: row.kind.qtype(),
    })
}

/// Source `predict_typed` readout: option softmax, then `choice`, expected
/// `score`, or `noul` (probability of `true`), plus raw marker scores.
///
/// # Errors
///
/// Returns [`JuliaRequestError`] unless there is one finite score per option.
pub fn typed_answer(row: &TypedRow, scores: &[f32]) -> Result<Value, JuliaRequestError> {
    if scores.len() != row.keys.len() || scores.iter().any(|s| !s.is_finite()) {
        return fail(format!("{:?}: invalid model scores", row.name));
    }
    let maximum = scores
        .iter()
        .map(|&s| f64::from(s))
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = scores
        .iter()
        .map(|&s| (f64::from(s) - maximum).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
    let probabilities: serde_json::Map<_, _> = row
        .keys
        .iter()
        .cloned()
        .zip(p.iter().map(|&v| json!(v)))
        .collect();
    let mut answer = json!({
        "type": row.kind.name(),
        "option_order": row.keys,
        "scores": scores,
        "probabilities": probabilities,
    });
    let max_probability = p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    match row.kind {
        QuestionType::Choice => {
            // Python's max() keeps the first maximal index.
            let best = (0..p.len()).fold(0, |best, i| if p[i] > p[best] { i } else { best });
            answer["choice"] = json!(row.keys[best]);
        }
        QuestionType::Score => {
            answer["score"] = json!(
                (0_u32..)
                    .zip(&p)
                    .map(|(i, v)| f64::from(i) * v)
                    .sum::<f64>()
            );
        }
        QuestionType::Noul => answer["noul"] = json!(p[1]),
    }
    if row.kind != QuestionType::Noul {
        answer["max_probability"] = json!(max_probability);
    }
    Ok(answer)
}

#[cfg(test)]
#[allow(clippy::float_cmp, reason = "readouts are exact for these inputs")]
mod tests {
    use super::*;

    const SPECIAL: SpecialTokens = SpecialTokens {
        mask_text: String::new(),
        mask: 4,
        cls: 2,
        sep: 1,
    };

    fn special() -> SpecialTokens {
        SpecialTokens {
            mask_text: "<mask>".into(),
            ..SPECIAL
        }
    }

    #[allow(clippy::unnecessary_wraps, reason = "matches the tokenizer callback")]
    /// Stub tokenizer: one ID per whitespace-separated word, `100 + byte length`.
    fn words(text: &str) -> Result<Vec<u32>, String> {
        Ok(text
            .split_whitespace()
            .map(|word| 100 + u32::try_from(word.len()).unwrap())
            .collect())
    }

    fn request(json: &str) -> Result<TypedRequest, JuliaRequestError> {
        parse_typed_request(json.as_bytes())
    }

    #[test]
    fn state_serializes_like_python_json_dumps() {
        // Expected strings are Python 3 `json.dumps(json.loads(s), ensure_ascii=False)`.
        let parsed = request(
            r#"{"state": {"b": [1, 2.0, {"z": null, "a": true}], "a": "x\"\\\n\t\u0001é\u007f",
                "e": [], "o": {}, "b": [1, 2.0, {"z": null, "a": true}]},
                "questions": {"q": {"type": "noul", "instructions": ""}}}"#,
        )
        .unwrap();
        assert_eq!(
            parsed.state_text,
            "{\"b\": [1, 2.0, {\"z\": null, \"a\": true}], \"a\": \"x\\\"\\\\\\n\\t\\u0001é\u{7f}\", \"e\": [], \"o\": {}}"
        );
        for (input, python) in [
            ("1e-05", "1e-05"),
            ("1e16", "1e+16"),
            ("0.0001", "0.0001"),
            ("123456789012345.6", "123456789012345.6"),
            ("1.5e16", "1.5e+16"),
            ("-2.5", "-2.5"),
            ("100.0", "100.0"),
            ("1E2", "100.0"),
            ("1e100", "1e+100"),
            ("5e-324", "5e-324"),
            ("0.1", "0.1"),
            ("1.7976931348623157e308", "1.7976931348623157e+308"),
            ("12345678901234567890.0", "1.2345678901234567e+19"),
            ("-0.0", "-0.0"),
            ("-7", "-7"),
        ] {
            let value: OrderedJson = serde_json::from_str(&format!("[{input}]")).unwrap();
            assert_eq!(value.to_python_json(), format!("[{python}]"), "{input}");
        }
        let text = request(
            r#"{"state": "raw {text}", "questions": {"q": {"type": "noul", "instructions": ""}}}"#,
        );
        assert_eq!(text.unwrap().state_text, "raw {text}");
    }

    #[test]
    fn typed_rows_keep_caller_order_and_source_labels() {
        let parsed = request(
            r#"{"state": [], "questions": {
                "z": {"type": "choice", "instructions": "pick", "criteria": {"b": "bee", "a": "ay"}},
                "y": {"type": "score", "instructions": "rate", "criteria": ["low", "high", "top"]},
                "x": {"type": "noul", "instructions": "ok?"},
                "w": {"type": "noul", "instructions": "ok?", "criteria": {"true": "yes", "false": "no"}}}}"#,
        )
        .unwrap();
        let summary: Vec<_> = parsed
            .rows
            .iter()
            .map(|r| {
                (
                    r.name.as_str(),
                    r.kind,
                    r.keys.join(","),
                    r.options.join(","),
                )
            })
            .collect();
        assert_eq!(
            summary,
            [
                ("z", QuestionType::Choice, "b,a".into(), "bee,ay".into()),
                (
                    "y",
                    QuestionType::Score,
                    "0,1,2".into(),
                    "low,high,top".into()
                ),
                (
                    "x",
                    QuestionType::Noul,
                    "false,true".into(),
                    "false,true".into()
                ),
                (
                    "w",
                    QuestionType::Noul,
                    "false,true".into(),
                    "no,yes".into()
                ),
            ]
        );
        assert_eq!(parsed.state_text, "[]");
    }

    #[test]
    fn typed_rows_reject_what_the_source_rejects() {
        let options = |n: usize| {
            let criteria: Vec<_> = (0..n).map(|i| format!("\"o{i}\"")).collect();
            format!(
                r#"{{"state": "", "questions": {{"q": {{"type": "score", "instructions": "", "criteria": [{}]}}}}}}"#,
                criteria.join(",")
            )
        };
        assert!(request(&options(2)).is_ok());
        assert!(request(&options(20)).is_ok());
        for bad in [
            options(1),
            options(21),
            r#"{"state": 3, "questions": {"q": {"type": "noul", "instructions": ""}}}"#.into(),
            r#"{"state": "", "questions": {}}"#.into(),
            r#"{"state": "", "questions": {"": {"type": "noul", "instructions": ""}}}"#.into(),
            r#"{"state": "", "questions": {"q": {"type": "noul"}}}"#.into(),
            r#"{"state": "", "questions": {"q": {"type": "rank", "instructions": ""}}}"#.into(),
            r#"{"state": "", "questions": {"q": {"type": "noul", "instructions": "", "criteria": {"true": "y"}}}}"#.into(),
            r#"{"state": "", "questions": {"q": {"type": "noul", "instructions": "", "criteria": {"true": "y", "maybe": "n"}}}}"#.into(),
            r#"{"state": "", "questions": {"q": {"type": "choice", "instructions": "", "criteria": {"a": "x", "b": ""}}}}"#.into(),
            r#"{"state": "", "questions": {"q": {"type": "choice", "instructions": "", "criteria": {"a": "x", "": "y"}}}}"#.into(),
            r#"{"state": "", "questions": {"q": {"type": "choice", "instructions": "", "criteria": ["x", "y"]}}}"#.into(),
        ] {
            assert!(request(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn sequence_matches_hand_derived_source_layout() {
        let parsed = request(
            r#"{"state": "hp 3", "questions": {"q": {"type": "choice", "instructions": "go?", "criteria": {"y": "yes", "n": "no way"}}}}"#,
        )
        .unwrap();
        let encoded = sequence(words, &special(), &parsed.rows[0], &parsed.state_text).unwrap();
        // [CLS] "choice" "question:" "go?" [SEP] [MASK] "yes" [MASK] "no" "way" [SEP] "hp" "3" [SEP]
        assert_eq!(
            encoded,
            EncodedRow {
                ids: vec![2, 106, 109, 103, 1, 4, 103, 4, 102, 103, 1, 102, 101, 1],
                markers: vec![5, 7],
                qtype: 0,
            }
        );
    }

    #[test]
    fn strict_sequence_rejects_markers_long_options_and_head_overflow() {
        let row = |options: &[String], question: &str| TypedRow {
            name: "q".into(),
            kind: QuestionType::Score,
            keys: (0..options.len()).map(|i| i.to_string()).collect(),
            question: question.into(),
            options: options.to_vec(),
        };
        let short = ["a".to_owned(), "b".to_owned()];
        assert!(sequence(words, &special(), &row(&short, "q"), "").is_ok());
        assert!(sequence(words, &special(), &row(&short, "q"), "x<mask>").is_err());
        assert!(sequence(words, &special(), &row(&short, "<mask>"), "").is_err());
        let long = ["w ".repeat(49), "b".to_owned()];
        assert!(sequence(words, &special(), &row(&long, "q"), "").is_err());
        let edge = ["w ".repeat(48), "b".to_owned()];
        assert!(sequence(words, &special(), &row(&edge, "q"), "").is_ok());
        // 49 + 2 marked tokens leave a 461-token budget; the head has 2 + n words.
        let fits = "w ".repeat(459);
        assert!(sequence(words, &special(), &row(&edge, &fits), "").is_ok());
        let overflows = "w ".repeat(460);
        assert!(sequence(words, &special(), &row(&edge, &overflows), "").is_err());
        assert!(
            sequence(
                words,
                &special(),
                &row(&short, "q"),
                &"s ".repeat(MAX_LENGTH)
            )
            .is_err()
        );
    }

    #[test]
    fn readout_follows_predict_typed() {
        let row = |kind, keys: &[&str]| TypedRow {
            name: "q".into(),
            kind,
            keys: keys.iter().map(|&k| k.to_owned()).collect(),
            question: String::new(),
            options: keys.iter().map(|&k| k.to_owned()).collect(),
        };
        let ln3 = 3.0_f32.ln();
        let choice = typed_answer(
            &row(QuestionType::Choice, &["a", "b", "c"]),
            &[0.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(choice["choice"], "b", "first maximum wins ties");
        assert_eq!(choice["max_probability"], choice["probabilities"]["c"]);
        let score = typed_answer(&row(QuestionType::Score, &["0", "1"]), &[0.0, ln3]).unwrap();
        assert!((score["score"].as_f64().unwrap() - 0.75).abs() < 1e-7);
        let noul = typed_answer(&row(QuestionType::Noul, &["false", "true"]), &[ln3, 0.0]).unwrap();
        assert!((noul["noul"].as_f64().unwrap() - 0.25).abs() < 1e-7);
        assert!(noul.get("max_probability").is_none());
        assert!(
            typed_answer(
                &row(QuestionType::Noul, &["false", "true"]),
                &[f32::NAN, 0.0]
            )
            .is_err()
        );
        assert!(typed_answer(&row(QuestionType::Noul, &["false", "true"]), &[0.0]).is_err());
    }
}
