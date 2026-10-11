//! A bounded `multipart/form-data` reader (the RFC 7578 subset HTTP clients
//! send for file uploads).
//!
//! The body is already in memory under the transport's size limit. Parts are
//! borrowed slices of it. Each part needs a `Content-Disposition: form-data`
//! header with a `name`; `filename` and `Content-Type` are optional. Any
//! other part header, a preamble before the first delimiter, or bytes after
//! the closing delimiter other than one trailing CRLF are refused, so a
//! malformed body never reaches a handler half-parsed.

/// Limits checked while reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MultipartLimits {
    /// Most parts in one body.
    pub(crate) parts: usize,
    /// Most header lines of one part.
    pub(crate) headers: usize,
    /// Largest header block of one part.
    pub(crate) header_bytes: usize,
    /// Largest part named in `file_fields`.
    pub(crate) file_bytes: usize,
    /// Largest part not named in `file_fields`.
    pub(crate) field_bytes: usize,
}

/// One form field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Part<'a> {
    /// The field name.
    pub(crate) name: &'a str,
    /// The uploaded file name, when the part is a file.
    pub(crate) filename: Option<&'a str>,
    /// The part's declared media type.
    pub(crate) content_type: Option<&'a str>,
    /// The part's bytes.
    pub(crate) data: &'a [u8],
}

/// Why a body was refused.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum MultipartError {
    /// The request is not `multipart/form-data` with a usable boundary.
    #[error("Content-Type must be multipart/form-data with a boundary")]
    ContentType,
    /// The body does not follow the delimiter structure.
    #[error("malformed multipart body: {0}")]
    Malformed(&'static str),
    /// A part's headers are missing, unsupported, or invalid.
    #[error("malformed multipart part headers: {0}")]
    PartHeaders(&'static str),
    /// More parts than the limit.
    #[error("multipart body has more than {0} parts")]
    TooManyParts(usize),
    /// A part exceeds its size limit.
    #[error("multipart field {name:?} exceeds {limit} bytes")]
    PartTooLarge {
        /// The field name.
        name: String,
        /// The limit it exceeds.
        limit: usize,
    },
}

/// Reads the boundary from a `Content-Type` header value.
pub(crate) fn boundary(content_type: &str) -> Result<&str, MultipartError> {
    let parameters = split_parameters(content_type).map_err(|_| MultipartError::ContentType)?;
    let mut parameters = parameters.into_iter();
    if !parameters.next().is_some_and(|(kind, value)| {
        value.is_none() && kind.eq_ignore_ascii_case("multipart/form-data")
    }) {
        return Err(MultipartError::ContentType);
    }
    let mut found = None;
    for (key, value) in parameters {
        if key.eq_ignore_ascii_case("boundary") {
            let value = value.ok_or(MultipartError::ContentType)?;
            if found.is_some()
                || value.is_empty()
                || value.len() > 70
                || value.ends_with(' ')
                || !value
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"'()+_,-./:=? ".contains(&b))
            {
                return Err(MultipartError::ContentType);
            }
            found = Some(value);
        }
    }
    found.ok_or(MultipartError::ContentType)
}

/// Splits `body` into parts.
///
/// `file_fields` names the fields held to `limits.file_bytes`; every other
/// field is held to `limits.field_bytes`.
pub(crate) fn parse<'a>(
    body: &'a [u8],
    boundary: &str,
    file_fields: &[&str],
    limits: MultipartLimits,
) -> Result<Vec<Part<'a>>, MultipartError> {
    let opening = format!("--{boundary}");
    let delimiter = format!("\r\n--{boundary}");
    let mut rest = body
        .strip_prefix(opening.as_bytes())
        .ok_or(MultipartError::Malformed(
            "body does not start with the boundary",
        ))?;
    let mut parts = Vec::new();
    loop {
        if let Some(after) = rest.strip_prefix(b"--") {
            return if after.is_empty() || after == b"\r\n" {
                Ok(parts)
            } else {
                Err(MultipartError::Malformed("data after the closing boundary"))
            };
        }
        rest = rest
            .strip_prefix(b"\r\n")
            .ok_or(MultipartError::Malformed("boundary not followed by CRLF"))?;
        if parts.len() == limits.parts {
            return Err(MultipartError::TooManyParts(limits.parts));
        }
        let header_end = find(
            &rest[..rest.len().min(limits.header_bytes.saturating_add(4))],
            b"\r\n\r\n",
        )
        .filter(|&end| end <= limits.header_bytes)
        .ok_or(MultipartError::PartHeaders(
            "headers unterminated or too long",
        ))?;
        let (name, filename, content_type) = part_headers(&rest[..header_end], limits.headers)?;
        let content = &rest[header_end + 4..];
        let limit = if file_fields.contains(&name) {
            limits.file_bytes
        } else {
            limits.field_bytes
        };
        let end = find_boundary(content, delimiter.as_bytes()).ok_or(MultipartError::Malformed(
            "part is not closed by a boundary",
        ))?;
        if end > limit {
            return Err(MultipartError::PartTooLarge {
                name: name.to_owned(),
                limit,
            });
        }
        parts.push(Part {
            name,
            filename,
            content_type,
            data: &content[..end],
        });
        rest = &content[end + delimiter.len()..];
    }
}

type PartHeaders<'a> = (&'a str, Option<&'a str>, Option<&'a str>);

fn part_headers(block: &[u8], max_headers: usize) -> Result<PartHeaders<'_>, MultipartError> {
    let block = std::str::from_utf8(block)
        .map_err(|_| MultipartError::PartHeaders("headers are not UTF-8"))?;
    let mut disposition = None;
    let mut content_type = None;
    for (index, line) in block.split("\r\n").enumerate() {
        if index >= max_headers {
            return Err(MultipartError::PartHeaders("too many headers"));
        }
        let (key, value) = line
            .split_once(':')
            .ok_or(MultipartError::PartHeaders("header line without a colon"))?;
        let value = value.trim();
        if key.eq_ignore_ascii_case("content-disposition") {
            if disposition.replace(value).is_some() {
                return Err(MultipartError::PartHeaders("repeated Content-Disposition"));
            }
        } else if key.eq_ignore_ascii_case("content-type") {
            if content_type.replace(value).is_some() {
                return Err(MultipartError::PartHeaders("repeated Content-Type"));
            }
        } else {
            return Err(MultipartError::PartHeaders("unsupported part header"));
        }
    }
    let disposition =
        disposition.ok_or(MultipartError::PartHeaders("missing Content-Disposition"))?;
    let mut parameters = split_parameters(disposition)?.into_iter();
    if !parameters
        .next()
        .is_some_and(|(kind, value)| value.is_none() && kind.eq_ignore_ascii_case("form-data"))
    {
        return Err(MultipartError::PartHeaders("disposition is not form-data"));
    }
    let (mut name, mut filename) = (None, None);
    for (key, value) in parameters {
        let value = value.ok_or(MultipartError::PartHeaders("parameter without a value"))?;
        let slot = if key.eq_ignore_ascii_case("name") {
            &mut name
        } else if key.eq_ignore_ascii_case("filename") {
            &mut filename
        } else {
            return Err(MultipartError::PartHeaders(
                "unsupported disposition parameter",
            ));
        };
        if slot.replace(value).is_some() {
            return Err(MultipartError::PartHeaders(
                "repeated disposition parameter",
            ));
        }
    }
    let name = name
        .filter(|name| !name.is_empty())
        .ok_or(MultipartError::PartHeaders("part has no name"))?;
    Ok((name, filename, content_type))
}

/// Splits `form-data; name="a"; filename="b"` into `(key, value)` pairs.
/// Quoted values may not contain quotes; HTTP clients percent-encode them.
fn split_parameters(value: &str) -> Result<Vec<(&str, Option<&str>)>, MultipartError> {
    let mut pairs = Vec::new();
    let mut rest = value;
    while !rest.trim().is_empty() {
        if pairs.len() >= 16 {
            return Err(MultipartError::PartHeaders("too many parameters"));
        }
        let item_end = if let Some(equals) = rest
            .find('=')
            .filter(|&equals| rest.find(';').is_none_or(|semicolon| equals < semicolon))
        {
            let key = rest[..equals].trim();
            let after = rest[equals + 1..].trim_start();
            let consumed = rest.len() - after.len();
            if let Some(quoted) = after.strip_prefix('"') {
                let close = quoted
                    .find('"')
                    .ok_or(MultipartError::PartHeaders("unterminated quoted parameter"))?;
                pairs.push((key, Some(&quoted[..close])));
                consumed + close + 2
            } else {
                let end = after.find(';').unwrap_or(after.len());
                pairs.push((key, Some(after[..end].trim())));
                consumed + end
            }
        } else {
            let end = rest.find(';').unwrap_or(rest.len());
            pairs.push((rest[..end].trim(), None));
            end
        };
        rest = &rest[item_end..];
        let trimmed = rest.trim_start();
        if trimmed.is_empty() {
            break;
        }
        rest = trimmed
            .strip_prefix(';')
            .ok_or(MultipartError::PartHeaders(
                "parameters must be separated by ';'",
            ))?;
    }
    Ok(pairs)
}

fn find_boundary(content: &[u8], delimiter: &[u8]) -> Option<usize> {
    content
        .windows(delimiter.len())
        .enumerate()
        .find_map(|(index, window)| {
            let suffix = &content[index + delimiter.len()..];
            (window == delimiter && (suffix.starts_with(b"\r\n") || suffix.starts_with(b"--")))
                .then_some(index)
        })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const LIMITS: MultipartLimits = MultipartLimits {
        parts: 8,
        headers: 4,
        header_bytes: 1024,
        file_bytes: 4096,
        field_bytes: 64,
    };

    fn body(boundary: &str, parts: &[(&str, Option<&str>, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, filename, data) in parts {
            out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
            let file = filename.map_or_else(String::new, |file| format!("; filename=\"{file}\""));
            out.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"{file}\r\n").as_bytes(),
            );
            if filename.is_some() {
                out.extend_from_slice(b"Content-Type: audio/wav\r\n");
            }
            out.extend_from_slice(b"\r\n");
            out.extend_from_slice(data);
            out.extend_from_slice(b"\r\n");
        }
        out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        out
    }

    #[test]
    fn rejects_duplicate_boundaries_and_preserves_boundary_prefix_in_file() {
        assert!(boundary("multipart/form-data; boundary=a; boundary=b").is_err());
        let file = b"binary\r\n--abcX\0tail";
        let bytes = body("abc", &[("file", None, file)]);
        assert_eq!(
            parse(&bytes, "abc", &["file"], LIMITS).unwrap()[0].data,
            file
        );
        let bytes = b"--b\r\nContent-Disposition: form-data; name=\"file\"\r\nContent-Type: audio/wav\r\n\r\nx\r\n--b--";
        assert!(
            parse(
                bytes,
                "b",
                &["file"],
                MultipartLimits {
                    headers: 1,
                    ..LIMITS
                }
            )
            .is_err()
        );
    }

    #[test]
    fn reads_fields_and_a_file() {
        let wav = b"RIFF\r\n--not-the-boundary\r\n\x00\xff";
        let bytes = body(
            "XyZ",
            &[
                ("model", None, b"qwen3-asr"),
                ("file", Some("a b.wav"), wav),
            ],
        );
        let parts = parse(&bytes, "XyZ", &["file"], LIMITS).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(
            (parts[0].name, parts[0].filename, parts[0].data),
            ("model", None, &b"qwen3-asr"[..])
        );
        assert_eq!(parts[1].filename, Some("a b.wav"));
        assert_eq!(parts[1].content_type, Some("audio/wav"));
        assert_eq!(parts[1].data, wav);
    }

    #[test]
    fn reads_the_boundary_from_the_content_type() {
        assert_eq!(
            boundary("multipart/form-data; boundary=abc123").unwrap(),
            "abc123"
        );
        assert_eq!(
            boundary("Multipart/Form-Data; charset=utf-8; boundary=\"a b:c\"").unwrap(),
            "a b:c"
        );
        for refused in [
            "application/json",
            "multipart/form-data",
            "multipart/form-data; boundary=",
            "multipart/form-data; boundary=\"trailing \"",
            "multipart/mixed; boundary=abc",
        ] {
            assert_eq!(
                boundary(refused),
                Err(MultipartError::ContentType),
                "{refused}"
            );
        }
    }

    #[test]
    fn refuses_oversized_parts_and_too_many_parts() {
        let big = vec![b'a'; 65];
        let bytes = body("b", &[("prompt", None, &big)]);
        assert_eq!(
            parse(&bytes, "b", &["file"], LIMITS),
            Err(MultipartError::PartTooLarge {
                name: "prompt".to_owned(),
                limit: 64
            })
        );
        let parts: Vec<(&str, Option<&str>, &[u8])> = vec![("x", None, b"1"); 9];
        assert_eq!(
            parse(&body("b", &parts), "b", &[], LIMITS),
            Err(MultipartError::TooManyParts(8))
        );
    }

    #[test]
    fn refuses_malformed_structure_and_headers() {
        let ok = body("b", &[("model", None, b"m")]);
        let cases: [(Vec<u8>, &str); 6] = [
            ([b"preamble\r\n", &ok[..]].concat(), "preamble"),
            ([&ok[..], b"junk"].concat(), "epilogue"),
            (ok[..ok.len() - 8].to_vec(), "unclosed"),
            (
                b"--b\r\nContent-Disposition: attachment; name=\"m\"\r\n\r\nx\r\n--b--".to_vec(),
                "not form-data",
            ),
            (
                b"--b\r\nContent-Disposition: form-data; name=\"m\"\r\nX-Other: 1\r\n\r\nx\r\n--b--"
                    .to_vec(),
                "extra header",
            ),
            (
                b"--b\r\nContent-Disposition: form-data; filename=\"f\"\r\n\r\nx\r\n--b--".to_vec(),
                "no name",
            ),
        ];
        for (bytes, label) in cases {
            assert!(parse(&bytes, "b", &[], LIMITS).is_err(), "{label}");
        }
    }

    fn arbitrary_parts() -> impl Strategy<Value = Vec<(String, Option<String>, Vec<u8>)>> {
        proptest::collection::vec(
            (
                "[a-z_\\[\\]]{1,12}",
                proptest::option::of("[a-zA-Z0-9 ._-]{1,16}"),
                proptest::collection::vec(any::<u8>(), 0..48),
            ),
            0..6,
        )
    }

    proptest! {
        // A body built from arbitrary fields round-trips exactly, as long as
        // no field happens to contain the delimiter.
        #[test]
        fn round_trips_arbitrary_fields(
            boundary_text in "[A-Za-z0-9'()+_,./:=?-]{1,70}",
            fields in arbitrary_parts(),
        ) {
            let delimiter = format!("\r\n--{boundary_text}");
            // The first delimiter after each field's data must be its own.
            prop_assume!(fields.iter().all(|(_, _, data)| {
                let framed = [data.as_slice(), delimiter.as_bytes()].concat();
                find(&framed, delimiter.as_bytes()) == Some(data.len())
            }));
            let borrowed: Vec<(&str, Option<&str>, &[u8])> = fields
                .iter()
                .map(|(name, file, data)| (name.as_str(), file.as_deref(), data.as_slice()))
                .collect();
            let bytes = body(&boundary_text, &borrowed);
            let parsed = parse(&bytes, &boundary_text, &[], MultipartLimits { field_bytes: 64, ..LIMITS });
            let parsed = parsed.unwrap();
            prop_assert_eq!(parsed.len(), fields.len());
            for (part, (name, file, data)) in parsed.iter().zip(&fields) {
                prop_assert_eq!(part.name, name.as_str());
                prop_assert_eq!(part.filename, file.as_deref());
                prop_assert_eq!(part.data, data.as_slice());
            }
        }

        // Every strict prefix of a valid body is refused, and arbitrary bytes
        // never panic.
        #[test]
        fn truncation_is_refused_and_noise_never_panics(
            fields in arbitrary_parts(),
            cut in 1_usize..200,
            noise in proptest::collection::vec(any::<u8>(), 0..300),
        ) {
            prop_assume!(fields.iter().all(|(_, _, data)| find(data, b"\r\n--Bnd").is_none()));
            let borrowed: Vec<(&str, Option<&str>, &[u8])> = fields
                .iter()
                .map(|(name, file, data)| (name.as_str(), file.as_deref(), data.as_slice()))
                .collect();
            let bytes = body("Bnd", &borrowed);
            // Dropping only the final CRLF still leaves a complete body.
            let keep = bytes.len().saturating_sub(cut);
            if keep < bytes.len() - 2 {
                prop_assert!(parse(&bytes[..keep], "Bnd", &[], LIMITS).is_err());
            }
            let _ = parse(&noise, "Bnd", &[], LIMITS);
            let mut prefixed = b"--Bnd\r\n".to_vec();
            prefixed.extend_from_slice(&noise);
            let _ = parse(&prefixed, "Bnd", &[], LIMITS);
        }
    }
}
