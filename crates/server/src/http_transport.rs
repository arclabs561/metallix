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
}

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

/// One accepted loopback connection with bounded read and response-write time.
pub(crate) struct Connection {
    stream: TcpStream,
    limits: TransportLimits,
    read_deadline: Instant,
    response_deadline: Option<Instant>,
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
        }
    }

    /// Reads exactly one HTTP/1.0 or HTTP/1.1 request before the absolute deadline.
    pub(crate) fn read_request(&mut self) -> Result<Request, HttpError> {
        let deadline = self.read_deadline;
        let (mut received, header_end, method, path, body_length) = self.read_head(deadline)?;
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
        })
    }

    /// Starts a new bounded response-write interval.
    pub(crate) fn begin_response(&mut self) {
        self.response_deadline = Some(deadline_after(self.limits.response_deadline));
    }

    fn read_head(
        &mut self,
        deadline: Instant,
    ) -> Result<(Vec<u8>, usize, String, String, usize), HttpError> {
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
                    return Ok((input, header_end, method, path, body_length));
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
    deadline
        .checked_duration_since(Instant::now())
        .ok_or(HttpError::timeout())
}

fn deadline_after(duration: Duration) -> Instant {
    Instant::now()
        .checked_add(duration)
        .unwrap_or_else(Instant::now)
}

fn remaining_io(deadline: Instant) -> io::Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
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
        io::Write as _,
        net::{TcpListener, TcpStream},
        thread,
        time::Duration,
    };

    use proptest::prelude::*;

    use super::{Connection, HttpError, Request, TransportLimits};

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
    fn response_deadline_fails_closed_before_a_socket_write() {
        let listener = listener();
        let address = listener.local_addr().unwrap();
        let client = thread::spawn(move || TcpStream::connect(address).unwrap());
        let (stream, _) = listener.accept().unwrap();
        let mut connection = Connection::accept(
            stream,
            TransportLimits {
                response_deadline: Duration::ZERO,
                ..limits()
            },
        );
        connection.begin_response();
        let error = connection.write_all(b"response").unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::TimedOut);
        drop(client.join().unwrap());
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
}
