//! A server-sent-event response whose HTTP head is written with its first
//! event, shared by the Chat Completions and Messages adapters.

use std::io::{BufWriter, Write};

use serde_json::Value;

use crate::http_transport::Connection;

/// The head waits for the first event so a failure before any output
/// (context overflow, schema compilation, an expired budget) still answers
/// with an ordinary HTTP error status instead of a `200` stream.
pub(crate) struct LazySse {
    pending: Option<Connection>,
    writer: Option<BufWriter<Connection>>,
}

impl LazySse {
    pub(crate) const fn new(connection: Connection) -> Self {
        Self {
            pending: Some(connection),
            writer: None,
        }
    }

    /// The connection, while nothing has been written to it.
    pub(crate) fn take_unopened(&mut self) -> Option<Connection> {
        self.pending.take()
    }

    fn open(&mut self) -> Result<&mut BufWriter<Connection>, String> {
        if self.writer.is_none() {
            let mut connection = self
                .pending
                .take()
                .ok_or("response connection is unavailable")?;
            connection.begin_response();
            let request_id = connection.request_id_header();
            let mut writer = BufWriter::new(connection);
            write!(writer, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{request_id}Connection: close\r\n\r\n").map_err(|e| e.to_string())?;
            self.writer = Some(writer);
        }
        Ok(self.writer.as_mut().expect("opened stream writer"))
    }

    /// Writes one event. `name` becomes an `event:` line when present;
    /// serialized JSON never contains a raw newline, so one `data:` line
    /// carries the whole payload.
    pub(crate) fn event(&mut self, name: Option<&str>, data: &Value) -> Result<(), String> {
        self.raw(name, &data.to_string())
    }

    /// Writes one event with a literal payload, such as `[DONE]`.
    pub(crate) fn raw(&mut self, name: Option<&str>, data: &str) -> Result<(), String> {
        let writer = self.open()?;
        if let Some(name) = name {
            writeln!(writer, "event: {name}").map_err(|e| e.to_string())?;
        }
        write!(writer, "data: {data}\n\n").map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())
    }

    /// Writes an SSE comment, which clients ignore, so a disconnect is
    /// noticed while output is withheld.
    pub(crate) fn keepalive(&mut self) -> Result<(), String> {
        let writer = self.open()?;
        writer
            .write_all(b": generating\n\n")
            .map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())
    }
}

/// Socket and backend fixtures shared by the protocol adapters' tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::{
        io::{Read as _, Write as _},
        net::{Shutdown, TcpListener, TcpStream},
        thread,
        time::Duration,
    };

    use serde_json::Value;

    use crate::{
        chat_generation::{
            ChatBackend, ChatFinishReason, ChatGeneration, ChatGenerationError,
            ChatGenerationMetrics, ChatRequest,
        },
        http_transport::{Connection, TransportLimits},
    };

    /// Sends one body through the real transport to `handle`, returning the
    /// raw HTTP response.
    pub(crate) fn exchange(
        path: &str,
        body: &str,
        handle: impl FnOnce(Connection, &[u8]),
    ) -> String {
        exchange_with(path, "", body, handle)
    }

    /// [`exchange`] with extra header lines, each ending in CRLF.
    pub(crate) fn exchange_with(
        path: &str,
        headers: &str,
        body: &str,
        handle: impl FnOnce(Connection, &[u8]),
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\n{headers}Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream.write_all(request.as_bytes()).unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
            let mut response = Vec::new();
            stream.read_to_end(&mut response).unwrap();
            String::from_utf8(response).unwrap()
        });
        let (stream, _) = listener.accept().unwrap();
        let mut connection = Connection::accept(stream, TransportLimits::default());
        let wire = connection.read_request().unwrap();
        connection.set_cache_salt(wire.trace.cache_salt.clone());
        handle(connection, &wire.body);
        client.join().unwrap()
    }

    /// The status line and JSON body of a non-streamed response.
    pub(crate) fn json_body(wire: &str) -> (String, Value) {
        let (head, body) = wire.split_once("\r\n\r\n").unwrap();
        (
            head.lines().next().unwrap().to_owned(),
            serde_json::from_str(body).unwrap(),
        )
    }

    /// Each event's optional name and raw `data` of a `200` event stream;
    /// comments are skipped.
    pub(crate) fn events(wire: &str) -> Vec<(Option<String>, String)> {
        let (head, payload) = wire.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 200 OK\r\n"), "{head}");
        assert!(head.contains("Content-Type: text/event-stream"), "{head}");
        payload
            .split("\n\n")
            .filter_map(|frame| {
                let mut name = None;
                let mut data = None;
                for line in frame.lines() {
                    if let Some(value) = line.strip_prefix("event: ") {
                        name = Some(value.to_owned());
                    } else if let Some(value) = line.strip_prefix("data: ") {
                        data = Some(value.to_owned());
                    }
                }
                data.map(|data| (name, data))
            })
            .collect()
    }

    /// Streams fixed text in two-character deltas, or fails before output.
    pub(crate) struct Scripted {
        text: String,
        failure: Option<String>,
        pub(crate) deltas: usize,
        /// Whether the last request carried tools and enabled thinking.
        pub(crate) seen: Option<(bool, bool)>,
        /// The last request's tenant cache salt.
        pub(crate) salt: Option<String>,
        /// Prompt tokens to report, and how many of them the cache served.
        pub(crate) prompt_tokens: usize,
        pub(crate) cached_prompt_tokens: usize,
    }

    impl Scripted {
        pub(crate) fn new(text: &str) -> Self {
            Self {
                text: text.into(),
                failure: None,
                deltas: 0,
                seen: None,
                salt: None,
                prompt_tokens: 1,
                cached_prompt_tokens: 0,
            }
        }

        pub(crate) fn failing(message: &str) -> Self {
            Self {
                failure: Some(message.into()),
                ..Self::new("")
            }
        }
    }

    impl ChatBackend for Scripted {
        fn load_ms(&self) -> f64 {
            0.0
        }

        fn generate_with_timeout(
            &mut self,
            request: ChatRequest<'_>,
            _timeout: Duration,
            on_token: &mut dyn FnMut(&str) -> Result<(), String>,
        ) -> Result<ChatGeneration, ChatGenerationError> {
            self.seen = Some((!request.tools.is_empty(), request.enable_thinking));
            self.salt = request.cache_salt.map(str::to_owned);
            if let Some(failure) = &self.failure {
                return Err(ChatGenerationError::Message(failure.clone()));
            }
            let characters: Vec<char> = self.text.chars().collect();
            for piece in characters.chunks(2) {
                self.deltas += 1;
                on_token(&piece.iter().collect::<String>())
                    .map_err(ChatGenerationError::Message)?;
            }
            Ok(ChatGeneration {
                text: self.text.clone(),
                generated_token_ids: vec![1, 2],
                finish_reason: ChatFinishReason::Eos,
                metrics: ChatGenerationMetrics {
                    context_tokens: 2048,
                    planned_kv_bytes: 0,
                    session_load_ms: 0.0,
                    render_ms: 0.0,
                    prefill_ms: 0.0,
                    time_to_first_token_ms: None,
                    decode_ms: vec![],
                    decode_total_ms: 0.0,
                    prompt_tokens: self.prompt_tokens,
                    cached_prompt_tokens: self.cached_prompt_tokens,
                    generated_tokens: 2,
                },
                logprobs: Vec::new(),
                sampling: None,
            })
        }
    }
}
