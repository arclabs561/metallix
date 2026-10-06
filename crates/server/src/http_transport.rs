//! Bounded one-request HTTP intake for the loopback Responses control plane.
//!
//! This deliberately owns no worker threads. A connection is read, answered,
//! and dropped by the serving loop, so a timed-out peer cannot retain a handler.

use std::{
    io::{self, Read, Write},
    net::TcpStream,
    time::{Duration, Instant},
};

/// Limits enforced before a request reaches the Responses protocol layer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TransportLimits {
    /// Maximum request-header bytes, including the terminating empty line.
    pub(crate) header_bytes: usize,
    /// Maximum parsed HTTP headers.
    pub(crate) headers: usize,
    /// Maximum declared and retained request-body bytes.
    pub(crate) body_bytes: usize,
    /// Absolute deadline covering the whole header and body intake.
    pub(crate) read_deadline: Duration,
    /// Maximum time one socket write or flush may block.
    pub(crate) write_idle: Duration,
    /// Absolute deadline for one response, reset by [`Connection::begin_response`].
    pub(crate) response_deadline: Duration,
}

impl Default for TransportLimits {
    fn default() -> Self {
        Self {
            header_bytes: 16 * 1024,
            headers: 64,
            body_bytes: 1024 * 1024,
            read_deadline: Duration::from_secs(5),
            write_idle: Duration::from_secs(5),
            response_deadline: Duration::from_secs(120),
        }
    }
}

/// A complete, bounded HTTP request. Connections accept exactly one request.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct Request {
    pub(crate) method: String,
    pub(crate) path: String,
    pub(crate) body: Vec<u8>,
    pub(crate) trace: TraceHeaders,
}

/// Request-correlation and router headers; every other header is dropped
/// after validation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct TraceHeaders {
    pub(crate) traceparent: Option<String>,
    pub(crate) request_id: Option<String>,
    /// `x-metallix-cache-salt`, the prefix-cache namespace. The router in
    /// front of the server owns this header: it must set or strip it on every
    /// request, since a client that can choose a salt can join another
    /// tenant's namespace. Validated as 1 to 256 visible ASCII bytes.
    pub(crate) cache_salt: Option<String>,
}

/// Longest accepted `x-metallix-cache-salt` value.
const MAX_CACHE_SALT_BYTES: usize = 256;

/// A transport rejection suitable for a small JSON error response.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HttpError {
    pub(crate) status: u16,
    pub(crate) message: &'static str,
}

impl HttpError {
    const fn bad_request(message: &'static str) -> Self {
        Self {
            status: 400,
            message,
        }
    }

    const fn timeout() -> Self {
        Self {
            status: 408,
            message: "request read deadline exceeded",
        }
    }
}

/// Generated deltas between client checks on a reply written only at the
/// end. A check is a nonblocking peek, cheap next to a decode step.
pub(crate) const CLIENT_POLL_DELTAS: usize = 16;

/// One accepted loopback connection with bounded read and response-write time.
pub(crate) struct Connection {
    stream: TcpStream,
    limits: TransportLimits,
    read_deadline: Instant,
    response_deadline: Option<Instant>,
    request_id: Option<String>,
    cache_salt: Option<String>,
    /// The request path, once the head is read; it picks the error shape.
    path: Option<String>,
    /// HTTP minor version of the request read, once one is read.
    minor_version: Option<u8>,
    /// Whether [`Connection::client_gone`] has sent its one probe.
    probed: bool,
    /// Internal forwards reserve write EOF for cancellation, while retaining
    /// the read half to wait for the worker to release admission.
    cancel_on_eof: bool,
}

impl Connection {
    /// Wraps an accepted stream. Callers retain listener and loopback policy.
    #[must_use]
    pub(crate) fn accept(stream: TcpStream, limits: TransportLimits) -> Self {
        Self {
            stream,
            limits,
            read_deadline: deadline_after(limits.read_deadline),
            response_deadline: None,
            request_id: None,
            cache_salt: None,
            path: None,
            minor_version: None,
            probed: false,
            cancel_on_eof: false,
        }
    }

    /// Reads exactly one HTTP/1.0 or HTTP/1.1 request before the absolute deadline.
    pub(crate) fn read_request(&mut self) -> Result<Request, HttpError> {
        let deadline = self.read_deadline;
        let (mut received, header_end, method, path, body_length, trace) =
            self.read_head(deadline)?;
        self.path = Some(path.clone());
        let prefix = received.split_off(header_end);
        if prefix.len() > body_length {
            return Err(HttpError::bad_request("pipelined requests are unsupported"));
        }
        received = prefix;
        received.reserve(body_length.saturating_sub(received.len()));
        while received.len() < body_length {
            let remaining = body_length - received.len();
            let mut chunk = [0_u8; 8192];
            let count = remaining.min(chunk.len());
            let read = self.read_with_deadline(&mut chunk[..count], deadline)?;
            if read == 0 {
                return Err(HttpError::bad_request(
                    "request body ended before Content-Length",
                ));
            }
            received.extend_from_slice(&chunk[..read]);
        }
        Ok(Request {
            method,
            path,
            body: received,
            trace,
        })
    }

    /// Sets the id that responses on this connection report in `X-Request-Id`.
    pub(crate) fn set_request_id(&mut self, request_id: &str) {
        self.request_id = Some(request_id.to_owned());
    }

    pub(crate) fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Sets the router-supplied prefix-cache salt for this request.
    pub(crate) fn set_cache_salt(&mut self, cache_salt: Option<String>) {
        self.cache_salt = cache_salt;
    }

    pub(crate) fn cache_salt(&self) -> Option<&str> {
        self.cache_salt.as_deref()
    }

    /// The path of the request read, once its head is read.
    pub(crate) fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// The `X-Request-Id` response header line, or nothing before an id is set.
    pub(crate) fn request_id_header(&self) -> String {
        self.request_id
            .as_deref()
            .map_or_else(String::new, |id| format!("X-Request-Id: {id}\r\n"))
    }

    /// Whether the client has closed its connection while its request waits,
    /// without consuming input. Read EOF alone also means a client that
    /// half-closed and is still waiting, so on the first EOF an HTTP/1.1 client
    /// is sent one `100 Continue`, which clients must accept before the final
    /// response; a closed peer answers it with a reset that a later call sees.
    pub(crate) fn client_gone(&mut self) -> bool {
        if self.stream.set_nonblocking(true).is_err() {
            return true;
        }
        let peeked = self.stream.peek(&mut [0_u8; 1]);
        if self.stream.set_nonblocking(false).is_err() {
            return true;
        }
        match peeked {
            Ok(0) if self.cancel_on_eof => true,
            Ok(0) if !self.probed && self.minor_version == Some(1) => {
                self.probed = true;
                self.stream
                    .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                    .is_err()
            }
            Ok(_) => false,
            Err(error) => error.kind() != io::ErrorKind::WouldBlock,
        }
    }

    /// An `on_token` for a reply written only when generation ends: every
    /// [`CLIENT_POLL_DELTAS`] deltas it checks [`Self::client_gone`] and stops
    /// the generation once the client has left, so a dropped request stops
    /// costing model time. A streamed reply learns the same from its writes.
    pub(crate) fn stop_when_gone<T>(&mut self) -> impl FnMut(T) -> Result<(), String> + '_ {
        let mut deltas = 0_usize;
        move |_| {
            deltas += 1;
            if deltas.is_multiple_of(CLIENT_POLL_DELTAS) && self.client_gone() {
                return Err(String::from("the client disconnected"));
            }
            Ok(())
        }
    }

    /// A second handle on the socket, which keeps it open after this
    /// connection is dropped until the handle is dropped too.
    pub(crate) fn hold_open(&self) -> Option<TcpStream> {
        self.stream.try_clone().ok()
    }

    /// Starts a new bounded response-write interval.
    pub(crate) fn begin_response(&mut self) {
        self.response_deadline = Some(deadline_after(self.limits.response_deadline));
    }

    #[allow(
        clippy::type_complexity,
        reason = "private parse result consumed once by read_request"
    )]
    fn read_head(
        &mut self,
        deadline: Instant,
    ) -> Result<(Vec<u8>, usize, String, String, usize, TraceHeaders), HttpError> {
        let mut input = Vec::with_capacity(self.limits.header_bytes.min(1024));
        loop {
            validate_header_wire(&input)?;
            let mut headers = vec![httparse::EMPTY_HEADER; self.limits.headers];
            let mut request = httparse::Request::new(&mut headers);
            match request.parse(&input) {
                Ok(httparse::Status::Complete(header_end)) => {
                    let method = request
                        .method
                        .ok_or(HttpError::bad_request("request method is missing"))?
                        .to_owned();
                    let path = request
                        .path
                        .ok_or(HttpError::bad_request("request path is missing"))?
                        .to_owned();
                    let version = request
                        .version
                        .ok_or(HttpError::bad_request("HTTP version is missing"))?;
                    if version > 1 {
                        return Err(HttpError {
                            status: 505,
                            message: "only HTTP/1.0 and HTTP/1.1 are supported",
                        });
                    }
                    let body_length = validate_headers(&request, &method, version, self.limits)?;
                    self.minor_version = Some(version);
                    // This opt-in affects only this connection's cancellation
                    // semantics; ordinary clients retain half-close support.
                    self.cancel_on_eof = request.headers.iter().any(|header| {
                        header.name.eq_ignore_ascii_case("x-metallix-cancel-on-eof")
                            && header.value == b"1"
                    });
                    let trace = trace_headers(&request);
                    if trace.cache_salt.as_deref().is_some_and(|salt| {
                        salt.is_empty()
                            || salt.len() > MAX_CACHE_SALT_BYTES
                            || !salt.bytes().all(|byte| byte.is_ascii_graphic())
                    }) {
                        return Err(HttpError::bad_request(
                            "x-metallix-cache-salt must be 1 to 256 visible ASCII characters",
                        ));
                    }
                    return Ok((input, header_end, method, path, body_length, trace));
                }
                Ok(httparse::Status::Partial) => {
                    if input.len() == self.limits.header_bytes {
                        return Err(HttpError {
                            status: 431,
                            message: "request headers exceed the 16 KiB limit",
                        });
                    }
                }
                Err(httparse::Error::TooManyHeaders) => {
                    return Err(HttpError {
                        status: 431,
                        message: "request has too many headers",
                    });
                }
                Err(_) => return Err(HttpError::bad_request("malformed HTTP request")),
            }

            let available = self.limits.header_bytes - input.len();
            let mut chunk = [0_u8; 1024];
            let count = available.min(chunk.len());
            let read = self.read_with_deadline(&mut chunk[..count], deadline)?;
            if read == 0 {
                return Err(HttpError::bad_request("request ended before headers"));
            }
            input.extend_from_slice(&chunk[..read]);
        }
    }

    fn read_with_deadline(
        &mut self,
        buffer: &mut [u8],
        deadline: Instant,
    ) -> Result<usize, HttpError> {
        self.stream
            .set_read_timeout(Some(remaining(deadline)?))
            .map_err(|error| map_read_error(error.kind()))?;
        self.stream
            .read(buffer)
            .map_err(|error| map_read_error(error.kind()))
    }

    fn write_timeout(&mut self) -> io::Result<()> {
        let deadline = self
            .response_deadline
            .get_or_insert_with(|| deadline_after(self.limits.response_deadline));
        let timeout = remaining_io(*deadline)?.min(self.limits.write_idle);
        self.stream.set_write_timeout(Some(timeout))
    }
}

impl Write for Connection {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.write_timeout()?;
        self.stream.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.write_timeout()?;
        self.stream.flush()
    }
}

fn validate_headers(
    request: &httparse::Request<'_, '_>,
    method: &str,
    version: u8,
    limits: TransportLimits,
) -> Result<usize, HttpError> {
    let mut host = None;
    let mut content_length = None;
    for header in request.headers.iter() {
        if header.name.eq_ignore_ascii_case("host") {
            if host.replace(header.value).is_some() {
                return Err(HttpError::bad_request("duplicate Host header"));
            }
        } else if header.name.eq_ignore_ascii_case("content-length") {
            if content_length.is_some() {
                return Err(HttpError::bad_request("duplicate Content-Length header"));
            }
            content_length = Some(parse_content_length(header.value, limits.body_bytes)?);
        } else if header.name.eq_ignore_ascii_case("transfer-encoding") {
            return Err(HttpError::bad_request("Transfer-Encoding is unsupported"));
        } else if header.name.eq_ignore_ascii_case("expect") {
            return Err(HttpError {
                status: 417,
                message: "Expect is unsupported",
            });
        }
    }
    if version == 1 && !host.is_some_and(valid_host) {
        return Err(HttpError::bad_request(
            "HTTP/1.1 requires one valid Host header",
        ));
    }
    match method {
        "POST" => content_length.ok_or(HttpError::bad_request("POST requires Content-Length")),
        "GET" => match content_length.unwrap_or(0) {
            0 => Ok(0),
            _ => Err(HttpError::bad_request("GET requests cannot include a body")),
        },
        _ => match content_length.unwrap_or(0) {
            0 => Ok(0),
            _ => Err(HttpError::bad_request(
                "request method with a body is unsupported",
            )),
        },
    }
}

/// The first `traceparent`, `x-request-id` and `x-metallix-cache-salt` values
/// that are UTF-8; callers validate their content.
fn trace_headers(request: &httparse::Request<'_, '_>) -> TraceHeaders {
    let value = |name: &str| {
        request
            .headers
            .iter()
            .find(|header| header.name.eq_ignore_ascii_case(name))
            .and_then(|header| std::str::from_utf8(header.value).ok())
            .map(str::to_owned)
    };
    TraceHeaders {
        traceparent: value("traceparent"),
        request_id: value("x-request-id"),
        cache_salt: value("x-metallix-cache-salt"),
    }
}

fn parse_content_length(value: &[u8], maximum: usize) -> Result<usize, HttpError> {
    if value.is_empty() || !value.iter().all(u8::is_ascii_digit) {
        return Err(HttpError::bad_request("Content-Length must be decimal"));
    }
    let value = std::str::from_utf8(value)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or(HttpError::bad_request("Content-Length is invalid"))?;
    if value > maximum {
        return Err(HttpError {
            status: 413,
            message: "request body exceeds the 1 MiB limit",
        });
    }
    Ok(value)
}

fn valid_host(value: &[u8]) -> bool {
    !value.is_empty()
        && value
            .iter()
            .all(|byte| byte.is_ascii_graphic() && *byte != b',')
}

/// Keeps the framing contract stricter than permissive HTTP parsers: headers
/// use CRLF only and cannot contain obsolete continuation lines. We inspect
/// only the header prefix, leaving an arbitrary POST body untouched.
fn validate_header_wire(input: &[u8]) -> Result<(), HttpError> {
    let header_bytes = input
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map_or(input, |end| &input[..end + 4]);
    for (index, &byte) in header_bytes.iter().enumerate() {
        if byte == b'\n' && (index == 0 || header_bytes[index - 1] != b'\r') {
            return Err(HttpError::bad_request(
                "HTTP headers require CRLF line endings",
            ));
        }
        if byte == b'\r' && index + 1 < header_bytes.len() && header_bytes[index + 1] != b'\n' {
            return Err(HttpError::bad_request(
                "HTTP headers require CRLF line endings",
            ));
        }
        if index >= 2 && header_bytes[index - 2..index] == *b"\r\n" && matches!(byte, b' ' | b'\t')
        {
            return Err(HttpError::bad_request(
                "obsolete folded HTTP headers are unsupported",
            ));
        }
    }
    Ok(())
}

fn remaining(deadline: Instant) -> Result<Duration, HttpError> {
    time_left(deadline, Instant::now()).ok_or(HttpError::timeout())
}

/// Positive time before `deadline`. A deadline reached exactly counts as
/// expired: sockets reject a zero timeout as invalid input, not a timeout.
fn time_left(deadline: Instant, now: Instant) -> Option<Duration> {
    deadline
        .checked_duration_since(now)
        .filter(|left| !left.is_zero())
}

fn deadline_after(duration: Duration) -> Instant {
    Instant::now()
        .checked_add(duration)
        .unwrap_or_else(Instant::now)
}

fn remaining_io(deadline: Instant) -> io::Result<Duration> {
    time_left(deadline, Instant::now())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "response write deadline exceeded"))
}

fn map_read_error(kind: io::ErrorKind) -> HttpError {
    if matches!(kind, io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock) {
        HttpError::timeout()
    } else {
        HttpError::bad_request("request socket read failed")
    }
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Read as _, Write as _},
        net::{Shutdown, TcpListener, TcpStream, UdpSocket},
        thread,
        time::{Duration, Instant},
    };

    use proptest::prelude::*;

    use super::{Connection, HttpError, Request, TraceHeaders, TransportLimits, time_left};

    fn limits() -> TransportLimits {
        TransportLimits {
            read_deadline: Duration::from_millis(150),
            ..TransportLimits::default()
        }
    }

    fn listener() -> TcpListener {
        TcpListener::bind("127.0.0.1:0").expect("loopback listener")
    }

    fn request_at(listener: &TcpListener) -> Connection {
        let (stream, _) = listener.accept().expect("test client connects");
        Connection::accept(stream, limits())
    }

    fn write_fragments(mut stream: TcpStream, bytes: &[u8], fragments: &[usize]) {
        let mut offset = 0;
        for &size in fragments {
            if offset == bytes.len() {
                break;
            }
            let end = (offset + size).min(bytes.len());
            stream
                .write_all(&bytes[offset..end])
                .expect("test fragment writes");
            offset = end;
        }
        if offset < bytes.len() {
            stream
                .write_all(&bytes[offset..])
                .expect("test tail writes");
        }
    }

    #[test]
    fn slow_header_hits_one_absolute_deadline_and_next_connection_recovers() {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let slow = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(b"POST /v1/responses HTTP/1.1\r\n")
                .unwrap();
            thread::sleep(Duration::from_millis(100));
            stream.write_all(b"Host: localhost\r\n").unwrap();
            thread::sleep(Duration::from_millis(100));
        });
        let mut connection = request_at(&listener);
        assert_eq!(connection.read_request(), Err(HttpError::timeout()));
        slow.join().unwrap();

        let valid = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
                .unwrap();
        });
        let mut connection = request_at(&listener);
        assert_eq!(
            connection.read_request().unwrap(),
            Request {
                method: "GET".into(),
                path: "/healthz".into(),
                body: vec![],
                trace: TraceHeaders::default(),
            }
        );
        valid.join().unwrap();
    }

    #[test]
    fn slow_body_hits_one_absolute_deadline_and_next_connection_recovers() {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let slow = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(
                    b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\na",
                )
                .unwrap();
            thread::sleep(Duration::from_millis(200));
        });
        let mut connection = request_at(&listener);
        assert_eq!(connection.read_request(), Err(HttpError::timeout()));
        slow.join().unwrap();

        let valid = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
                .unwrap();
        });
        let mut connection = request_at(&listener);
        assert_eq!(connection.read_request().unwrap().body, b"{}".to_vec());
        valid.join().unwrap();
    }

    #[test]
    fn body_receives_only_the_header_deadlines_remaining_time() {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .write_all(b"POST /v1/responses HTTP/1.1\r\n")
                .unwrap();
            thread::sleep(Duration::from_millis(200));
            stream
                .write_all(b"Host: localhost\r\nContent-Length: 2\r\n\r\na")
                .unwrap();
            // A per-stage body deadline would survive until EOF and report a
            // malformed body. The one intake deadline must expire first.
            thread::sleep(Duration::from_millis(200));
        });
        let (stream, _) = listener.accept().unwrap();
        let mut connection = Connection::accept(
            stream,
            TransportLimits {
                read_deadline: Duration::from_millis(300),
                ..TransportLimits::default()
            },
        );
        assert_eq!(connection.read_request(), Err(HttpError::timeout()));
        client.join().unwrap();
    }

    #[test]
    fn parser_rejects_bare_lf_and_obsolete_header_folding() {
        for wire in [
            b"GET /healthz HTTP/1.1\nHost: localhost\n\n".as_slice(),
            b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n folded: no\r\n\r\n".as_slice(),
        ] {
            let listener = listener();
            let address = listener.local_addr().unwrap();
            let bytes = wire.to_vec();
            let client = thread::spawn(move || {
                let mut stream = TcpStream::connect(address).unwrap();
                stream.write_all(&bytes).unwrap();
            });
            let mut connection = request_at(&listener);
            assert_eq!(connection.read_request().unwrap_err().status, 400);
            client.join().unwrap();
        }
    }

    #[test]
    fn cache_salt_header_is_read_and_bounded() {
        let wire = |salt: &str| {
            format!(
                "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nX-Metallix-Cache-Salt: {salt}\r\nContent-Length: 2\r\n\r\n{{}}"
            )
            .into_bytes()
        };
        let request = parse_wire(wire("tenant-7"), limits()).unwrap();
        assert_eq!(request.trace.cache_salt.as_deref(), Some("tenant-7"));
        let longest = "s".repeat(256);
        let request = parse_wire(wire(&longest), limits()).unwrap();
        assert_eq!(request.trace.cache_salt, Some(longest));
        for salt in ["", "two words", &"s".repeat(257)] {
            assert_eq!(parse_wire(wire(salt), limits()).unwrap_err().status, 400);
        }
        let plain =
            b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}";
        assert_eq!(
            parse_wire(plain.to_vec(), limits())
                .unwrap()
                .trace
                .cache_salt,
            None
        );
    }

    fn parse_wire(wire: Vec<u8>, limits: TransportLimits) -> Result<Request, HttpError> {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream.write_all(&wire).unwrap();
        });
        let (stream, _) = listener.accept().unwrap();
        let result = Connection::accept(stream, limits).read_request();
        client.join().unwrap();
        result
    }

    #[test]
    fn parser_rejects_ambiguous_or_unsupported_body_framing_before_body_reads() {
        for (wire, expected) in [
            (
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n".to_vec(),
                HttpError::bad_request("duplicate Content-Length header"),
            ),
            (
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec(),
                HttpError::bad_request("Transfer-Encoding is unsupported"),
            ),
            (
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nExpect: 100-continue\r\nContent-Length: 0\r\n\r\n".to_vec(),
                HttpError { status: 417, message: "Expect is unsupported" },
            ),
            (
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1048577\r\n\r\n".to_vec(),
                HttpError { status: 413, message: "request body exceeds the 1 MiB limit" },
            ),
        ] {
            assert_eq!(parse_wire(wire, limits()), Err(expected));
        }
    }

    #[test]
    fn parser_enforces_header_count_and_byte_caps() {
        let count_limited = TransportLimits {
            headers: 1,
            ..limits()
        };
        assert_eq!(
            parse_wire(
                b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n"
                    .to_vec(),
                count_limited,
            ),
            Err(HttpError {
                status: 431,
                message: "request has too many headers"
            })
        );
        let byte_limited = TransportLimits {
            header_bytes: 32,
            ..limits()
        };
        assert_eq!(
            parse_wire(
                b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n".to_vec(),
                byte_limited,
            ),
            Err(HttpError {
                status: 431,
                message: "request headers exceed the 16 KiB limit"
            })
        );
    }

    #[test]
    fn request_half_close_still_receives_the_complete_json_response() {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .write_all(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
                .unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
            let mut received = Vec::new();
            stream.read_to_end(&mut received).unwrap();
            received
        });
        let (stream, _) = listener.accept().unwrap();
        let mut connection = Connection::accept(stream, TransportLimits::default());
        let request = connection.read_request().unwrap();
        assert_eq!(request.path, "/v1/responses");
        assert_eq!(request.body, b"{}");
        // Read EOF proves only that the peer has finished sending. It is not
        // evidence that the peer stopped waiting for the response.
        assert_eq!(connection.stream.peek(&mut [0]).unwrap(), 0);
        connection.begin_response();
        connection.write_all(response).unwrap();
        connection.flush().unwrap();
        drop(connection);
        assert_eq!(client.join().unwrap(), response);
    }

    #[test]
    fn a_deadline_reached_exactly_is_expired_not_a_zero_timeout() {
        let now = Instant::now();
        let later = now + Duration::from_millis(1);
        // The socket layer rejects a zero timeout as invalid, so a deadline
        // equal to now must not reach it as `Some(Duration::ZERO)`.
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        assert_eq!(
            socket
                .set_write_timeout(Some(Duration::ZERO))
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(time_left(now, now), None);
        assert_eq!(time_left(now, later), None);
        assert_eq!(time_left(later, now), Some(Duration::from_millis(1)));
    }

    #[test]
    fn response_deadline_failure_drops_connection_and_next_request_recovers() {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let failed_client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .write_all(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}")
                .unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
            let mut received = Vec::new();
            stream.read_to_end(&mut received).unwrap();
            received
        });
        let (stream, _) = listener.accept().unwrap();
        let mut connection = Connection::accept(
            stream,
            TransportLimits {
                response_deadline: Duration::ZERO,
                ..limits()
            },
        );
        assert_eq!(connection.read_request().unwrap().path, "/v1/responses");
        connection.begin_response();
        let error = connection.write_all(b"response").unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        drop(connection);
        assert!(failed_client.join().unwrap().is_empty());

        let healthy_client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            stream
                .write_all(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n")
                .unwrap();
            stream.shutdown(Shutdown::Write).unwrap();
            let mut received = Vec::new();
            stream.read_to_end(&mut received).unwrap();
            received
        });
        let (stream, _) = listener.accept().unwrap();
        let mut connection = Connection::accept(stream, limits());
        assert_eq!(connection.read_request().unwrap().path, "/healthz");
        let response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
        connection.begin_response();
        connection.write_all(response).unwrap();
        connection.flush().unwrap();
        drop(connection);
        assert_eq!(healthy_client.join().unwrap(), response);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        #[test]
        fn fragmented_post_round_trips_without_changing_body(
            path in "/[a-z]{0,24}",
            body in prop::collection::vec(any::<u8>(), 0..256),
            fragments in prop::collection::vec(1_usize..64, 1..32),
        ) {
            let listener = listener();
            let address = listener.local_addr().unwrap();
            let expected_path = path.clone();
            let expected_body = body.clone();
            let client = thread::spawn(move || {
                let stream = TcpStream::connect(address).unwrap();
                let mut bytes = format!(
                    "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
                    body.len(),
                )
                .into_bytes();
                bytes.extend_from_slice(&body);
                write_fragments(stream, &bytes, &fragments);
            });
            let mut connection = request_at(&listener);
            let actual = connection
                .read_request()
                .map_err(|error| TestCaseError::fail(format!("{error:?}")))?;
            prop_assert_eq!(actual.method, "POST");
            prop_assert_eq!(actual.path, expected_path);
            prop_assert_eq!(actual.body, expected_body);
            client.join().expect("fragment client joins");
        }
    }
    #[test]
    fn client_gone_tells_a_closed_client_from_a_waiting_one() {
        let listener = listener();
        let address = listener.local_addr().expect("listener address");
        let request =
            b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}";
        let probe_until = |connection: &mut Connection, gone: bool| {
            let deadline = Instant::now() + Duration::from_secs(2);
            while connection.client_gone() != gone {
                assert!(Instant::now() < deadline, "client_gone never became {gone}");
                thread::sleep(Duration::from_millis(10));
            }
        };

        // Still sending nothing more, still connected: present.
        let mut open = TcpStream::connect(address).expect("connect");
        open.write_all(request).expect("request");
        let mut waiting = request_at(&listener);
        waiting.read_request().expect("request reads");
        for _ in 0..3 {
            assert!(!waiting.client_gone());
        }

        // Half-closed but reading: present, and sent exactly one 100 Continue.
        let mut half = TcpStream::connect(address).expect("connect");
        half.write_all(request).expect("request");
        half.shutdown(Shutdown::Write).expect("half-close");
        let mut half_closed = request_at(&listener);
        half_closed.read_request().expect("request reads");
        for _ in 0..5 {
            assert!(!half_closed.client_gone());
            thread::sleep(Duration::from_millis(10));
        }
        half_closed.begin_response();
        half_closed
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
            .unwrap();
        drop(half_closed);
        let mut received = Vec::new();
        half.read_to_end(&mut received).expect("response arrives");
        assert_eq!(
            received,
            b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"
        );

        // Internal forwards opt in: write EOF cancels while the read half
        // stays open to receive the response and final socket close.
        let mut forward = TcpStream::connect(address).expect("connect");
        forward
            .write_all(b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\nx-metallix-cancel-on-eof: 1\r\n\r\n{}")
            .expect("request");
        let mut forwarded = request_at(&listener);
        forwarded.read_request().expect("request reads");
        assert!(!forwarded.client_gone());
        forward.shutdown(Shutdown::Write).expect("cancel");
        probe_until(&mut forwarded, true);
        forwarded.begin_response();
        forwarded
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
            .unwrap();
        drop(forwarded);
        let mut response = Vec::new();
        forward
            .read_to_end(&mut response)
            .expect("response and close");
        assert_eq!(response, b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}");

        // Fully closed: gone once its reset to the probe arrives.
        let mut closed = TcpStream::connect(address).expect("connect");
        closed.write_all(request).expect("request");
        let mut departed = request_at(&listener);
        departed.read_request().expect("request reads");
        drop(closed);
        probe_until(&mut departed, true);

        // An HTTP/1.0 client is never sent a 1xx response.
        let mut old = TcpStream::connect(address).expect("connect");
        old.write_all(b"POST / HTTP/1.0\r\nContent-Length: 0\r\n\r\n")
            .expect("request");
        old.shutdown(Shutdown::Write).expect("half-close");
        let mut legacy = request_at(&listener);
        legacy.read_request().expect("request reads");
        for _ in 0..3 {
            assert!(!legacy.client_gone());
        }
        drop(legacy);
        let mut received = Vec::new();
        old.read_to_end(&mut received).expect("close arrives");
        assert!(received.is_empty());
        drop(open);
    }
}
