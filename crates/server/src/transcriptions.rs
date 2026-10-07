//! Bounded upload and response contract for experimental WAV transcription.

use std::{io::Write, ops::Range};

use serde_json::{Value, json};

use crate::{
    generation_routes::error_response,
    http_transport::{BodyLimit, Connection, TransportLimits},
    multipart::{self, MultipartLimits},
    responses::json_response,
};

pub(crate) const PATH: &str = "/v1/audio/transcriptions";
pub(crate) const BODY_BYTES: usize = 25_000_000;
const FIELD_BYTES: usize = 16 * 1024;

/// Cooperative stop reasons, independent of a model implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stop {
    Cancelled,
    DeadlineExceeded,
}

pub(crate) struct Control<'a> {
    deadline: crate::chat_generation::GenerationDeadline,
    cancelled: &'a mut dyn FnMut() -> bool,
}

impl<'a> Control<'a> {
    pub(crate) fn new(
        timeout: std::time::Duration,
        cancelled: &'a mut dyn FnMut() -> bool,
    ) -> Self {
        Self {
            deadline: crate::chat_generation::GenerationDeadline::after(timeout),
            cancelled,
        }
    }

    pub(crate) fn check(&mut self) -> Result<(), Stop> {
        self.deadline.check().map_err(|_| Stop::DeadlineExceeded)?;
        if (self.cancelled)() {
            Err(Stop::Cancelled)
        } else {
            Ok(())
        }
    }
}

impl From<Stop> for Error {
    fn from(stop: Stop) -> Self {
        match stop {
            Stop::Cancelled => Self {
                status: 499,
                message: "transcription cancelled".into(),
            },
            Stop::DeadlineExceeded => Self {
                status: 408,
                message: "generation time budget exceeded".into(),
            },
        }
    }
}

/// Same route policy at the front proxy and model child; other limits survive.
pub(crate) struct UploadLimits(pub(crate) TransportLimits);

impl BodyLimit for UploadLimits {
    fn body_limit(&self, method: &str, target: &str) -> usize {
        let path = target.split_once('?').map_or(target, |(path, _)| path);
        if method == "POST" && path == PATH {
            BODY_BYTES
        } else {
            self.0.body_bytes
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Format {
    Json,
    Text,
    VerboseJson,
}

/// Validated form metadata and a borrowed file slice, before copying a job.
#[derive(Debug)]
pub(crate) struct Form<'a> {
    pub(crate) model: &'a str,
    pub(crate) prompt: &'a str,
    pub(crate) language: Option<&'a str>,
    pub(crate) format: Format,
    file: Range<usize>,
}

/// The model worker owns the original body, with no second copy of its file.
#[derive(Debug)]
pub(crate) struct Request {
    pub(crate) model: String,
    pub(crate) prompt: String,
    pub(crate) language: Option<String>,
    pub(crate) format: Format,
    body: Vec<u8>,
    file: Range<usize>,
}

impl Request {
    pub(crate) fn from_body(body: Vec<u8>, content_type: Option<&str>) -> Result<Self, Error> {
        let form = parse(&body, content_type)?;
        Ok(Self {
            model: form.model.to_owned(),
            prompt: form.prompt.to_owned(),
            language: form.language.map(str::to_owned),
            format: form.format,
            file: form.file,
            body,
        })
    }

    pub(crate) fn file(&self) -> &[u8] {
        &self.body[self.file.clone()]
    }
}

#[derive(Debug)]
pub(crate) struct Error {
    pub(crate) status: u16,
    pub(crate) message: String,
}

impl Error {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: message.into(),
        }
    }

    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self {
            status: 415,
            message: message.into(),
        }
    }
}

pub(crate) enum Response {
    Json(Value),
    Text(String),
}

pub(crate) fn parse<'a>(body: &'a [u8], content_type: Option<&str>) -> Result<Form<'a>, Error> {
    if body.len() > BODY_BYTES {
        return Err(Error {
            status: 413,
            message: "transcription body exceeds 25000000 bytes".into(),
        });
    }
    let boundary = multipart::boundary(content_type.unwrap_or_default())
        .map_err(|error| Error::unsupported(error.to_string()))?;
    let parts = multipart::parse(
        body,
        boundary,
        &["file"],
        MultipartLimits {
            parts: 8,
            headers: 4,
            header_bytes: 2048,
            file_bytes: BODY_BYTES,
            field_bytes: FIELD_BYTES,
        },
    )
    .map_err(|error| Error::invalid(error.to_string()))?;
    let mut form = Form {
        model: "",
        prompt: "",
        language: None,
        format: Format::Json,
        file: 0..0,
    };
    let mut seen = Vec::with_capacity(parts.len());
    for part in parts {
        if seen.contains(&part.name) {
            return Err(Error::invalid(format!("duplicate field {:?}", part.name)));
        }
        seen.push(part.name);
        if part.name == "file" {
            if part.content_type.is_some_and(|kind| {
                ![
                    "audio/wav",
                    "audio/x-wav",
                    "audio/wave",
                    "application/octet-stream",
                ]
                .iter()
                .any(|allowed| kind.eq_ignore_ascii_case(allowed))
            }) {
                return Err(Error::unsupported("only WAV uploads are implemented"));
            }
            if part.data.is_empty() {
                return Err(Error::invalid("file is empty"));
            }
            // Both slices come from this same retained body.
            let start = part.data.as_ptr().addr() - body.as_ptr().addr();
            form.file = start..start + part.data.len();
            continue;
        }
        if part.filename.is_some() {
            return Err(Error::invalid("only file may have a filename"));
        }
        let value = std::str::from_utf8(part.data)
            .map_err(|_| Error::invalid("form fields must be UTF-8"))?;
        match part.name {
            "model" if !value.trim().is_empty() && value.len() <= 256 => form.model = value,
            "prompt" => form.prompt = value,
            "language" if !value.trim().is_empty() && value.len() <= 64 => {
                form.language = Some(value);
            }
            "response_format" => {
                form.format = match value {
                    "json" => Format::Json,
                    "text" => Format::Text,
                    "verbose_json" => Format::VerboseJson,
                    _ => {
                        return Err(Error::invalid(
                            "response_format must be json, text or verbose_json",
                        ));
                    }
                }
            }
            "stream" if value == "false" => {}
            "temperature" if value.parse::<f64>().is_ok_and(|v| v == 0.0) => {}
            "stream" => {
                return Err(Error::invalid(
                    "streaming transcription is not implemented; use stream=false",
                ));
            }
            "temperature" => {
                return Err(Error::invalid("only greedy temperature=0 is implemented"));
            }
            _ => {
                return Err(Error::invalid(format!(
                    "unsupported or invalid transcription field {:?}",
                    part.name
                )));
            }
        }
    }
    if form.model.is_empty() || form.file.is_empty() {
        return Err(Error::invalid("file and model are required"));
    }
    Ok(form)
}

pub(crate) fn result(format: Format, text: String, language: &str, duration: f64) -> Response {
    match format {
        Format::Text => Response::Text(text),
        Format::Json => Response::Json(json!({"text":text})),
        Format::VerboseJson => Response::Json(
            json!({"task":"transcribe", "text":text, "language":language, "duration":duration}),
        ),
    }
}

pub(crate) fn respond(mut connection: Connection, outcome: Option<Result<Response, Error>>) {
    match outcome {
        Some(Ok(Response::Json(value))) => json_response(connection, 200, &value),
        Some(Ok(Response::Text(text))) => {
            connection.begin_response();
            let request_id = connection.request_id_header();
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\n{request_id}Connection: close\r\n\r\n",
                text.len()
            );
            if connection
                .write_all(head.as_bytes())
                .and_then(|()| connection.write_all(text.as_bytes()))
                .is_err()
            {
                tracing::warn!("transcription response write failed");
            }
        }
        Some(Err(error)) if error.status == 499 => drop(connection),
        Some(Err(error)) => {
            let code = (error.status == 408).then_some("generation_timeout");
            error_response(connection, error.status, code, &error.message);
        }
        None => error_response(
            connection,
            400,
            Some("unsupported_capability"),
            "model does not support transcribe",
        ),
    }
}

#[cfg(test)]
pub(crate) mod tests;
