//! Typed, bounded schedule requirements for verifier-controlled generation.
//!
//! This module accepts only a local JSON descriptor. It deliberately does not
//! infer requirements from prompts, and it compares integer ticks exactly.

use std::{fs::File, io::Read, path::Path};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const MAX_DESCRIPTOR_BYTES: usize = 64 * 1024;
const MAX_DURATIONS: usize = 128;
const MIN_EXACT_TICK: i64 = -9_007_199_254_740_992;
const MAX_EXACT_TICK: i64 = 9_007_199_254_740_992;

/// A validated, exact-tick schedule target from one bounded descriptor file.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ScheduleRequirements {
    durations: Vec<i64>,
    window: TickWindow,
    source_sha256: String,
    source_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TickWindow {
    start: i64,
    end: i64,
}

/// Result of applying the typed task target after semantic non-overlap passes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RequirementsVerification {
    rejection: Option<&'static str>,
}

impl RequirementsVerification {
    #[cfg(test)]
    const fn accepted(self) -> bool {
        self.rejection.is_none()
    }
}

impl ScheduleRequirements {
    /// Loads and validates a descriptor without accepting unknown fields.
    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let bytes = read_descriptor(path)?;
        Self::from_bytes(&bytes)
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|_| String::from("schedule requirements descriptor is not valid JSON"))?;
        let (durations, window) = parse_descriptor(&value)?;
        let total_duration = durations
            .iter()
            .map(|duration| i128::from(*duration))
            .sum::<i128>();
        let window_length = i128::from(window.end) - i128::from(window.start);
        if total_duration > window_length {
            return Err(String::from(
                "schedule requirements total duration exceeds window length",
            ));
        }
        Ok(Self {
            durations,
            window,
            source_sha256: format!("{:x}", Sha256::digest(bytes)),
            source_bytes: bytes.len(),
        })
    }

    /// Receipt-safe source identity and the fully validated typed target.
    pub(crate) fn report(&self) -> Value {
        json!({
            "sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "durations": self.durations,
            "window": {"start": self.window.start, "end": self.window.end},
            "scope": "SHA-256 of exact descriptor bytes; integer ticks are compared exactly without epsilon or prompt extraction",
        })
    }

    /// Checks exact task adherence. The caller must run its independent
    /// non-overlap check first; this method checks endpoint representation,
    /// bounds, count, duration multiset, and window membership.
    fn verify_candidate(&self, value: &Value) -> RequirementsVerification {
        let Some(intervals) = value.get("intervals").and_then(Value::as_array) else {
            return rejected("missing_intervals");
        };
        if intervals.len() != self.durations.len() {
            return rejected("interval_count_mismatch");
        }
        let mut actual_durations = Vec::with_capacity(intervals.len());
        for interval in intervals {
            let Some(start) = integer_tick(interval.get("start")) else {
                return rejected("interval_start_not_integer_tick");
            };
            let Some(end) = integer_tick(interval.get("end")) else {
                return rejected("interval_end_not_integer_tick");
            };
            if start < self.window.start || end > self.window.end {
                return rejected("interval_outside_window");
            }
            // Existing semantic verification establishes end > start. Keeping
            // this defensive check makes the verifier safe if called directly.
            let Some(duration) = end.checked_sub(start) else {
                return rejected("interval_duration_not_positive");
            };
            if duration <= 0 {
                return rejected("interval_duration_not_positive");
            }
            actual_durations.push(duration);
        }
        actual_durations.sort_unstable();
        let mut expected_durations = self.durations.clone();
        expected_durations.sort_unstable();
        if actual_durations != expected_durations {
            return rejected("duration_multiset_mismatch");
        }
        RequirementsVerification { rejection: None }
    }

    /// Returns the stable task-adherence rejection, if any, after the caller's
    /// independent non-overlap verification has passed.
    pub(crate) fn rejection(&self, value: &Value) -> Option<&'static str> {
        self.verify_candidate(value).rejection
    }
}

fn rejected(rejection: &'static str) -> RequirementsVerification {
    RequirementsVerification {
        rejection: Some(rejection),
    }
}

fn read_descriptor(path: &Path) -> Result<Vec<u8>, String> {
    let metadata = path
        .metadata()
        .map_err(|_| String::from("schedule requirements descriptor must be a regular file"))?;
    if !metadata.is_file() {
        return Err(String::from(
            "schedule requirements descriptor must be a regular file",
        ));
    }
    let file = File::open(path)
        .map_err(|_| String::from("schedule requirements descriptor must be a regular file"))?;
    if !file
        .metadata()
        .map_err(|_| String::from("schedule requirements descriptor must be a regular file"))?
        .is_file()
    {
        return Err(String::from(
            "schedule requirements descriptor must be a regular file",
        ));
    }
    let mut bytes = Vec::new();
    file.take(u64::try_from(MAX_DESCRIPTOR_BYTES).expect("byte limit fits u64") + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| String::from("schedule requirements descriptor could not be read"))?;
    if bytes.len() > MAX_DESCRIPTOR_BYTES {
        return Err(String::from(
            "schedule requirements descriptor exceeds 64 KiB byte limit",
        ));
    }
    Ok(bytes)
}

fn parse_descriptor(value: &Value) -> Result<(Vec<i64>, TickWindow), String> {
    let object = value.as_object().ok_or_else(|| {
        String::from("schedule requirements descriptor must be an object with durations and window")
    })?;
    if object.len() != 2 || !object.contains_key("durations") || !object.contains_key("window") {
        return Err(String::from(
            "schedule requirements descriptor has unknown or missing fields",
        ));
    }
    let durations = object
        .get("durations")
        .and_then(Value::as_array)
        .ok_or_else(|| String::from("schedule requirements durations must be an array"))?;
    if durations.is_empty() || durations.len() > MAX_DURATIONS {
        return Err(String::from(
            "schedule requirements durations must contain 1 through 128 entries",
        ));
    }
    let mut parsed_durations = Vec::with_capacity(durations.len());
    for duration in durations {
        let Some(duration) = integer_tick(Some(duration)) else {
            return Err(String::from(
                "schedule requirements durations must contain integer ticks",
            ));
        };
        if duration <= 0 {
            return Err(String::from(
                "schedule requirements durations must contain positive ticks",
            ));
        }
        parsed_durations.push(duration);
    }
    let window = parse_window(object.get("window"))?;
    Ok((parsed_durations, window))
}

fn parse_window(value: Option<&Value>) -> Result<TickWindow, String> {
    let object = value.and_then(Value::as_object).ok_or_else(|| {
        String::from("schedule requirements window must be an object with start and end")
    })?;
    if object.len() != 2 || !object.contains_key("start") || !object.contains_key("end") {
        return Err(String::from(
            "schedule requirements window has unknown or missing fields",
        ));
    }
    let start = integer_tick(object.get("start")).ok_or_else(|| {
        String::from("schedule requirements window start must be an integer tick")
    })?;
    let end = integer_tick(object.get("end"))
        .ok_or_else(|| String::from("schedule requirements window end must be an integer tick"))?;
    if start >= end {
        return Err(String::from(
            "schedule requirements window must have start less than end",
        ));
    }
    Ok(TickWindow { start, end })
}

fn integer_tick(value: Option<&Value>) -> Option<i64> {
    let tick = value?.as_i64()?;
    (MIN_EXACT_TICK..=MAX_EXACT_TICK)
        .contains(&tick)
        .then_some(tick)
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::{MAX_EXACT_TICK, MIN_EXACT_TICK, ScheduleRequirements};

    fn requirements() -> ScheduleRequirements {
        let value = json!({"durations": [2, 2, 3], "window": {"start": -2, "end": 12}});
        let (durations, window) = super::parse_descriptor(&value).expect("valid descriptor");
        ScheduleRequirements {
            durations,
            window,
            source_sha256: String::from("test"),
            source_bytes: 0,
        }
    }

    #[test]
    fn exact_requirements_accept_duplicates_and_order_independent_durations() {
        let accepted = requirements().verify_candidate(&json!({
            "intervals": [
                {"start": 3, "end": 5}, {"start": 5, "end": 8}, {"start": -2, "end": 0}
            ]
        }));
        assert!(accepted.accepted());
    }

    #[test]
    fn exact_requirements_reject_noninteger_wrong_multiset_and_window_escape() {
        let requirements = requirements();
        for (candidate, rejection) in [
            (
                json!({"intervals": [{"start": 0.0, "end": 2}, {"start": 2, "end": 4}, {"start": 4, "end": 7}]}),
                "interval_start_not_integer_tick",
            ),
            (
                json!({"intervals": [{"start": 0, "end": 2}, {"start": 2, "end": 4}, {"start": 4, "end": 8}]}),
                "duration_multiset_mismatch",
            ),
            (
                json!({"intervals": [{"start": -3, "end": -1}, {"start": 0, "end": 2}, {"start": 2, "end": 5}]}),
                "interval_outside_window",
            ),
        ] {
            assert_eq!(
                requirements.verify_candidate(&candidate).rejection,
                Some(rejection)
            );
        }
    }

    #[test]
    fn descriptor_parsing_enforces_shape_exact_bounds_and_impossibility() {
        let valid = json!({
            "durations": [1, 1],
            "window": {"start": MIN_EXACT_TICK, "end": MIN_EXACT_TICK + 2}
        });
        assert!(super::parse_descriptor(&valid).is_ok());
        for (value, error) in [
            (
                json!({"durations": [], "window": {"start": 0, "end": 1}}),
                "schedule requirements durations must contain 1 through 128 entries",
            ),
            (
                json!({"durations": [1], "window": {"start": 0, "end": 1}, "extra": true}),
                "schedule requirements descriptor has unknown or missing fields",
            ),
            (
                json!({"durations": [1.0], "window": {"start": 0, "end": 1}}),
                "schedule requirements durations must contain integer ticks",
            ),
            (
                json!({"durations": [1], "window": {"start": 0, "end": MAX_EXACT_TICK + 1}}),
                "schedule requirements window end must be an integer tick",
            ),
        ] {
            assert_eq!(super::parse_descriptor(&value).unwrap_err(), error);
        }
        assert_eq!(
            ScheduleRequirements::from_bytes(
                br#"{"durations":[2,2],"window":{"start":0,"end":3}}"#
            )
            .unwrap_err(),
            "schedule requirements total duration exceeds window length"
        );
    }

    #[test]
    fn descriptor_hashes_exact_bytes_and_bounds_reads() {
        let directory =
            std::env::temp_dir().join(format!("metallix-requirements-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("temporary descriptor directory");
        let path = directory.join("requirements.json");
        let bytes = br#"{ "durations": [1], "window": {"start": 0, "end": 1} }"#;
        fs::write(&path, bytes).expect("descriptor write");
        let requirements = ScheduleRequirements::load(&path).expect("valid descriptor file");
        assert_eq!(
            requirements.report()["sha256"],
            format!("{:x}", Sha256::digest(bytes))
        );
        assert_eq!(requirements.report()["source_bytes"], bytes.len());
        assert!(ScheduleRequirements::load(Path::new("does-not-exist")).is_err());
        let oversized = directory.join("oversized.json");
        fs::write(&oversized, vec![b' '; super::MAX_DESCRIPTOR_BYTES + 1])
            .expect("oversized write");
        assert_eq!(
            ScheduleRequirements::load(&oversized).unwrap_err(),
            "schedule requirements descriptor exceeds 64 KiB byte limit"
        );
        fs::remove_file(path).expect("temporary descriptor cleanup");
        fs::remove_file(oversized).expect("temporary descriptor cleanup");
        fs::remove_dir(directory).expect("temporary descriptor cleanup");
    }
}
