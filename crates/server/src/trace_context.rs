//! W3C `traceparent` and `x-request-id` handling for `mx serve`.
//!
//! The front process accepts a caller's `traceparent` when it is well formed
//! and mints one otherwise, then hands each child a new parent id under the
//! same trace id. The request id is the caller's `x-request-id` when it is a
//! short token, else the trace id, so one id follows a request through the
//! front, the child, its logs and its response.

use std::{
    collections::hash_map::RandomState,
    hash::{BuildHasher as _, Hasher as _},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_REQUEST_ID: usize = 128;

/// A version-00 trace context: 16-byte trace id, 8-byte parent id, flags.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TraceParent {
    trace_id: u128,
    parent_id: u64,
    flags: u8,
}

impl TraceParent {
    /// Parses `00-<32 hex>-<16 hex>-<2 hex>`. Lowercase hex only, and neither
    /// id may be all zeros, per the W3C Trace Context recommendation.
    pub(crate) fn parse(value: &str) -> Option<Self> {
        let mut fields = value.split('-');
        let (version, trace, parent, flags) = (
            fields.next()?,
            fields.next()?,
            fields.next()?,
            fields.next()?,
        );
        if fields.next().is_some() || version != "00" {
            return None;
        }
        let trace_id = lower_hex(trace, 32).and_then(|v| u128::from_str_radix(v, 16).ok())?;
        let parent_id = lower_hex(parent, 16).and_then(|v| u64::from_str_radix(v, 16).ok())?;
        let flags = lower_hex(flags, 2).and_then(|v| u8::from_str_radix(v, 16).ok())?;
        (trace_id != 0 && parent_id != 0).then_some(Self {
            trace_id,
            parent_id,
            flags,
        })
    }

    /// A new sampled trace.
    pub(crate) fn mint() -> Self {
        Self {
            trace_id: (u128::from(random_u64()) << 64) | u128::from(random_u64()),
            parent_id: random_u64(),
            flags: 1,
        }
    }

    /// The same trace with a new parent id, for a downstream hop.
    pub(crate) fn child(self) -> Self {
        Self {
            parent_id: random_u64(),
            ..self
        }
    }

    pub(crate) fn trace_id(self) -> String {
        format!("{:032x}", self.trace_id)
    }
}

impl std::fmt::Display for TraceParent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "00-{:032x}-{:016x}-{:02x}",
            self.trace_id, self.parent_id, self.flags
        )
    }
}

/// The trace and request id for one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RequestContext {
    pub(crate) trace: TraceParent,
    pub(crate) request_id: String,
}

impl RequestContext {
    /// Keeps a valid incoming `traceparent` and `x-request-id`; replaces
    /// either when it is missing or malformed.
    pub(crate) fn from_headers(traceparent: Option<&str>, request_id: Option<&str>) -> Self {
        let trace = traceparent
            .and_then(TraceParent::parse)
            .unwrap_or_else(TraceParent::mint);
        let request_id = request_id
            .filter(|id| valid_request_id(id))
            .map_or_else(|| trace.trace_id(), str::to_owned);
        Self { trace, request_id }
    }
}

/// A request id is echoed into logs and headers, so it must be a short token.
fn valid_request_id(id: &str) -> bool {
    (1..=MAX_REQUEST_ID).contains(&id.len())
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:".contains(&byte))
}

fn lower_hex(value: &str, length: usize) -> Option<&str> {
    (value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    .then_some(value)
}

/// Non-cryptographic random ids: `RandomState` keys are seeded per process,
/// and the counter and clock separate ids within it.
fn random_u64() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    hasher.write_u128(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos()),
    );
    // Zero ids are invalid; the chance is negligible but the fix is free.
    hasher.finish().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    #[test]
    fn parses_the_recommendation_example_and_round_trips() {
        let parsed = TraceParent::parse(VALID).expect("valid traceparent");
        assert_eq!(parsed.to_string(), VALID);
        assert_eq!(parsed.trace_id(), "4bf92f3577b34da6a3ce929d0e0e4736");
    }

    #[test]
    fn rejects_malformed_or_zero_traceparents() {
        for invalid in [
            "",
            "garbage",
            "01-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
            "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-0g",
        ] {
            assert_eq!(TraceParent::parse(invalid), None, "{invalid:?}");
        }
    }

    #[test]
    fn minted_and_child_contexts_are_valid_and_distinct() {
        let minted = TraceParent::mint();
        let reparsed = TraceParent::parse(&minted.to_string()).expect("minted is valid");
        assert_eq!(reparsed, minted);
        let child = minted.child();
        assert_eq!(child.trace_id(), minted.trace_id());
        assert_ne!(child.to_string(), minted.to_string());
        assert_ne!(TraceParent::mint().trace_id(), minted.trace_id());
    }

    #[test]
    fn request_id_prefers_a_valid_header_then_the_trace_id() {
        let kept = RequestContext::from_headers(Some(VALID), Some("req-1.a:b_c"));
        assert_eq!(kept.request_id, "req-1.a:b_c");
        assert_eq!(kept.trace.to_string(), VALID);

        let derived = RequestContext::from_headers(Some(VALID), Some("has space"));
        assert_eq!(derived.request_id, "4bf92f3577b34da6a3ce929d0e0e4736");
        let long = "a".repeat(MAX_REQUEST_ID + 1);
        let derived = RequestContext::from_headers(Some("bogus"), Some(&long));
        assert_ne!(derived.trace.to_string(), "bogus");
        assert_eq!(derived.request_id, derived.trace.trace_id());
    }
}
