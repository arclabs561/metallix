use super::*;

pub(crate) fn form(fields: &[(&str, &[u8])]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--boundary\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(b"--boundary--\r\n");
    body
}

const CONTENT_TYPE: &str = "multipart/form-data; boundary=boundary";

#[test]
fn owns_the_exact_binary_file_and_refuses_unsupported_controls() {
    let wav = include_bytes!("../../../audio/tests/fixtures/pcm16_stereo.wav");
    let body = form(&[
        ("model", b"asr"),
        ("file", wav),
        ("response_format", b"text"),
        ("language", b"en"),
    ]);
    let request = Request::from_body(body, Some(CONTENT_TYPE)).unwrap();
    assert_eq!(request.file(), wav);
    assert_eq!(request.format, Format::Text);
    assert_eq!(request.language.as_deref(), Some("en"));
    assert_eq!(
        audio::decode_wav(request.file()).unwrap().samples().len(),
        6
    );
    for (key, value) in [
        ("stream", "true"),
        ("temperature", "0.1"),
        ("temperature", "NaN"),
        ("response_format", "srt"),
        ("timestamp_granularities[]", "word"),
        ("include[]", "logprobs"),
    ] {
        let body = form(&[("model", b"asr"), ("file", wav), (key, value.as_bytes())]);
        assert_eq!(
            parse(&body, Some(CONTENT_TYPE)).unwrap_err().status,
            400,
            "{key}"
        );
    }
}

#[test]
fn refuses_duplicates_missing_fields_and_oversized_text() {
    for fields in [
        vec![("model", b"a".as_slice())],
        vec![("file", b"x".as_slice())],
        vec![
            ("file", b"x".as_slice()),
            ("model", b"a".as_slice()),
            ("model", b"b".as_slice()),
        ],
    ] {
        assert!(parse(&form(&fields), Some(CONTENT_TYPE)).is_err());
    }
    let prompt = vec![b'x'; FIELD_BYTES + 1];
    assert!(
        parse(
            &form(&[("model", b"asr"), ("file", b"x"), ("prompt", &prompt)]),
            Some(CONTENT_TYPE)
        )
        .is_err()
    );
    assert_eq!(parse(b"x", Some("audio/wav")).unwrap_err().status, 415);
}

#[test]
fn body_policy_raises_only_transcription_post_including_query() {
    let limits = UploadLimits(TransportLimits {
        body_bytes: 123,
        ..Default::default()
    });
    assert_eq!(limits.body_limit("POST", PATH), BODY_BYTES);
    assert_eq!(
        limits.body_limit("POST", "/v1/audio/transcriptions?x=1"),
        BODY_BYTES
    );
    assert_eq!(limits.body_limit("GET", PATH), 123);
    assert_eq!(limits.body_limit("POST", "/v1/responses"), 123);
}
