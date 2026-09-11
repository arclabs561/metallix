//! Bounded, terminal-safe human preview for a generation JSON receipt.

use std::{fmt::Write, io::IsTerminal};

use serde_json::Value;

const MAX_PREVIEW_TOKENS: usize = 256;
const MAX_PREVIEW_TEXT_CHARS: usize = 4_096;

/// Emits an optional human preview to stderr without changing the JSON receipt on stdout.
pub(crate) fn emit(report: &Value) {
    let color = std::io::stderr().is_terminal() && terminal_allows_color();
    let preview = render(report, color);
    eprint!("{preview}");
}

/// Renders a bounded preview. `color` must only be enabled for a terminal.
#[must_use]
pub(crate) fn render(report: &Value, color: bool) -> String {
    let mut output = String::from("generation preview\n");
    line(
        &mut output,
        "operation",
        report_string(report, "operation")
            .as_deref()
            .unwrap_or("unavailable"),
    );
    line(
        &mut output,
        "finish_reason",
        report_string(report, "finish_reason")
            .as_deref()
            .unwrap_or("unavailable"),
    );

    render_timing_summary(&mut output, report);
    render_constraint_preview(&mut output, report);
    render_logprob_preview(&mut output, report, color);
    output
}

fn render_timing_summary(output: &mut String, report: &Value) {
    let load_ms = report_number(report, "load_ms");
    let prefill_ms = report_number(report, "prefill_ms");
    let (decode_total_ms, decode_steps) =
        report
            .get("decode_ms")
            .and_then(Value::as_array)
            .map_or((None, 0), |values| {
                (
                    Some(values.iter().filter_map(Value::as_f64).sum::<f64>()),
                    values.iter().filter(|value| value.is_number()).count(),
                )
            });
    output.push_str("timings_ms: load=");
    output.push_str(&format_number(load_ms));
    output.push_str(" prefill=");
    output.push_str(&format_number(prefill_ms));
    output.push_str(" decode_total=");
    output.push_str(&format_number(decode_total_ms));
    output.push_str(" decode_steps=");
    output.push_str(&decode_steps.to_string());
    output.push('\n');
}

fn render_constraint_preview(output: &mut String, report: &Value) {
    let constraint = report.get("constraint").and_then(Value::as_object);
    if let Some(constraint) = constraint {
        let status = constraint
            .get("status")
            .and_then(Value::as_str)
            .map_or_else(|| String::from("unavailable"), sanitize_terminal_text);
        line(output, "constraint_status", &status);
        if let Some(text) = constraint.get("generated_text").and_then(Value::as_str) {
            line(output, "output_text", &sanitize_terminal_text(text));
            return;
        }
    } else {
        line(output, "constraint_status", "not_requested");
    }

    let ids = generated_ids(report);
    output.push_str("raw_token_ids: ");
    if ids.is_empty() {
        output.push_str("unavailable\n");
    } else {
        output.push_str(&ids.join(","));
        output.push('\n');
    }
}

fn render_logprob_preview(output: &mut String, report: &Value, color: bool) {
    let Some(tokens) = report
        .get("logprobs")
        .and_then(Value::as_object)
        .and_then(|logprobs| logprobs.get("tokens"))
        .and_then(Value::as_array)
    else {
        if report.get("logprobs").is_none() {
            output.push_str("selected_token_logprobs: not requested (add --logprobs)\n");
        } else {
            output.push_str("selected_token_logprobs: unavailable\n");
        }
        return;
    };
    if tokens.is_empty() {
        output.push_str("selected_token_logprobs: unavailable\n");
        return;
    }

    output.push_str(
        "selected_token_logprobs: first 256; green >= -1, yellow >= -3, red < -3 nats; likelihood/surprisal only, not confidence or correctness\n",
    );
    for token in tokens.iter().take(MAX_PREVIEW_TOKENS) {
        let Some(token) = token.as_object() else {
            continue;
        };
        let token_id = token
            .get("token_id")
            .and_then(Value::as_i64)
            .map_or_else(|| String::from("?"), |id| id.to_string());
        let model = token.get("model_logprob").and_then(Value::as_f64);
        let constrained = token.get("constrained_logprob").and_then(Value::as_f64);
        let allowed_mass = token.get("allowed_log_mass").and_then(Value::as_f64);
        let model = color_logprob(&format_number(model), model, color);
        output.push_str("  id=");
        output.push_str(&token_id);
        output.push_str(" model_logprob=");
        output.push_str(&model);
        output.push_str(" constrained_logprob=");
        output.push_str(&format_number(constrained));
        output.push_str(" allowed_log_mass=");
        output.push_str(&format_number(allowed_mass));
        output.push('\n');
    }
    if tokens.len() > MAX_PREVIEW_TOKENS {
        output.push_str("  … ");
        output.push_str(&(tokens.len() - MAX_PREVIEW_TOKENS).to_string());
        output.push_str(" additional selected-token rows omitted\n");
    }
}

fn generated_ids(report: &Value) -> Vec<String> {
    report
        .get("generated_ids")
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |ids| {
            ids.iter()
                .filter_map(Value::as_i64)
                .take(MAX_PREVIEW_TOKENS)
                .map(|id| id.to_string())
                .collect()
        })
}

fn report_number(report: &Value, field: &str) -> Option<f64> {
    report.get(field).and_then(Value::as_f64)
}

fn format_number(value: Option<f64>) -> String {
    match value.filter(|value| value.is_finite()) {
        Some(value) => format!("{value:.3}"),
        None => String::from("unavailable"),
    }
}

fn color_logprob(value: &str, logprob: Option<f64>, color: bool) -> String {
    if !color {
        return String::from(value);
    }
    let code = match logprob.filter(|value| value.is_finite()) {
        Some(value) if value >= -1.0 => "32",
        Some(value) if value >= -3.0 => "33",
        Some(_) => "31",
        None => return String::from(value),
    };
    format!("\x1b[{code}m{value}\x1b[0m")
}

fn report_string(report: &Value, field: &str) -> Option<String> {
    report
        .get(field)
        .and_then(Value::as_str)
        .map(sanitize_terminal_text)
}

fn line(output: &mut String, label: &str, value: &str) {
    output.push_str(label);
    output.push_str(": ");
    output.push_str(value);
    output.push('\n');
}

fn sanitize_terminal_text(value: &str) -> String {
    let mut result = String::new();
    for (index, character) in value.chars().enumerate() {
        if index == MAX_PREVIEW_TEXT_CHARS {
            result.push_str("… <truncated>");
            break;
        }
        if character.is_control() || is_bidirectional_control(character) {
            write!(&mut result, "\\u{{{:04X}}}", u32::from(character))
                .expect("writing to a String cannot fail");
        } else {
            result.push(character);
        }
    }
    result
}

fn is_bidirectional_control(character: char) -> bool {
    matches!(
        character,
        '\u{061C}' | '\u{200E}' | '\u{200F}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}'
    )
}

fn terminal_allows_color() -> bool {
    let no_color = std::env::var_os("NO_COLOR").is_some_and(|value| !value.is_empty());
    let dumb_terminal = std::env::var_os("TERM").is_some_and(|value| value == "dumb");
    !no_color && !dumb_terminal
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::render;

    #[test]
    fn ansi_is_disabled_when_color_is_off() {
        let preview = render(
            &json!({
                "operation": "qwen3_generation",
                "finish_reason": "length",
                "generated_ids": [3],
                "load_ms": 1.0,
                "prefill_ms": 2.0,
                "decode_ms": [3.0],
                "logprobs": {"tokens": [{"token_id": 3, "model_logprob": -0.5}]},
            }),
            false,
        );

        assert!(!preview.contains('\x1b'));
        assert!(preview.contains("model_logprob=-0.500"));
    }

    #[test]
    fn ansi_is_owned_and_enabled_only_for_likelihood_values() {
        let preview = render(
            &json!({"logprobs": {"tokens": [{"token_id": 3, "model_logprob": -0.5}]}}),
            true,
        );

        assert!(preview.contains("\x1b[32m-0.500\x1b[0m"));
    }

    #[test]
    fn terminal_controls_and_bidi_text_are_escaped() {
        let preview = render(
            &json!({
                "operation": "safe\u{1b}[31m",
                "constraint": {"status": "ok\r", "generated_text": "x\u{202E}y"},
            }),
            false,
        );

        assert!(preview.contains("safe\\u{001B}[31m"));
        assert!(preview.contains("ok\\u{000D}"));
        assert!(preview.contains("x\\u{202E}y"));
        assert!(!preview.contains('\x1b'));
    }

    #[test]
    fn absent_constrained_text_is_explicitly_raw_ids() {
        let preview = render(
            &json!({"constraint": {"status": "incomplete"}, "generated_ids": [7, 8]}),
            false,
        );

        assert!(preview.contains("raw_token_ids: 7,8"));
        assert!(!preview.contains("output_text:"));
    }

    #[test]
    fn sparse_reports_and_probability_rows_are_safe() {
        let preview = render(&json!({"logprobs": {"tokens": [null, {}]}}), false);

        assert!(preview.contains("timings_ms: load=unavailable"));
        assert!(preview.contains("raw_token_ids: unavailable"));
        assert!(preview.contains("selected_token_logprobs:"));
    }
}
