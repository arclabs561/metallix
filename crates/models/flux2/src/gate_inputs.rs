//! Inputs for the checkpoint-backed parity gates.
//!
//! A gate that cannot find its checkpoint or reference fails; it never
//! prints a notice and passes. Resolution takes a variable lookup instead of
//! reading the process environment directly, so the failure path itself is
//! unit-tested without a checkpoint and without mutating the environment.

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use crate::transformer::Flux2Precision;

/// Reads one variable; the gates pass [`process_env`].
pub(crate) type Lookup<'a> = &'a dyn Fn(&str) -> Option<OsString>;

pub(crate) fn process_env(name: &str) -> Option<OsString> {
    std::env::var_os(name)
}

#[derive(Debug)]
pub(crate) enum InputError {
    Unset(String),
    Missing { var: String, path: PathBuf },
    Empty { var: String, path: PathBuf },
    Sidecar { path: PathBuf, message: String },
    Precision(String),
    Field(String),
    Device { path: PathBuf, device: String },
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unset(var) => write!(f, "{var} is not set; the gate cannot run"),
            Self::Missing { var, path } => write!(f, "{var}={} does not exist", path.display()),
            Self::Empty { var, path } => {
                write!(
                    f,
                    "{var}={} is empty (a placeholder, not a reference)",
                    path.display()
                )
            }
            Self::Sidecar { path, message } => write!(f, "sidecar {}: {message}", path.display()),
            Self::Device { path, device } => write!(
                f,
                "{} was produced on {device:?}; the bf16 bar needs a CPU source like the f32 one",
                path.display()
            ),
            Self::Field(key) => write!(f, "reference sidecar lacks numeric field {key:?}"),
            Self::Precision(value) => {
                write!(
                    f,
                    "METALLIX_FLUX2_PRECISION={value:?} is neither f32 nor bf16"
                )
            }
        }
    }
}

/// A variable naming an existing, non-empty file.
pub(crate) fn file(lookup: Lookup<'_>, var: &str) -> Result<PathBuf, InputError> {
    let path = PathBuf::from(lookup(var).ok_or_else(|| InputError::Unset(var.to_owned()))?);
    match fs::metadata(&path) {
        Ok(meta) if meta.is_file() && meta.len() > 0 => Ok(path),
        Ok(meta) if meta.is_file() => Err(InputError::Empty {
            var: var.to_owned(),
            path,
        }),
        _ => Err(InputError::Missing {
            var: var.to_owned(),
            path,
        }),
    }
}

/// A variable naming an existing directory.
pub(crate) fn dir(lookup: Lookup<'_>, var: &str) -> Result<PathBuf, InputError> {
    let path = PathBuf::from(lookup(var).ok_or_else(|| InputError::Unset(var.to_owned()))?);
    if path.is_dir() {
        Ok(path)
    } else {
        Err(InputError::Missing {
            var: var.to_owned(),
            path,
        })
    }
}

/// The JSON sidecar `scripts/flux2-reference.py` writes next to a reference.
pub(crate) fn sidecar(reference: &Path) -> Result<serde_json::Value, InputError> {
    let path = reference.with_extension("json");
    let text = fs::read_to_string(&path).map_err(|error| InputError::Sidecar {
        path: path.clone(),
        message: error.to_string(),
    })?;
    serde_json::from_str(&text).map_err(|error| InputError::Sidecar {
        path,
        message: error.to_string(),
    })
}

/// A source bf16 reference for a relative bar: an existing file whose
/// sidecar declares `"device": "cpu"`, the device of the f32 references.
/// A bar measured against an MPS or undeclared source would compare two
/// backends instead of two precisions.
pub(crate) fn cpu_reference(lookup: Lookup<'_>, var: &str) -> Result<PathBuf, InputError> {
    let path = file(lookup, var)?;
    let meta = sidecar(&path)?;
    match meta["device"].as_str() {
        Some("cpu") => Ok(path),
        other => Err(InputError::Device {
            device: other.unwrap_or("undeclared").to_owned(),
            path,
        }),
    }
}

/// Reject unrelated images as the source noise floor for the bf16 gate.
pub(crate) fn matching_pipeline(reference: &Path, bf16: &Path) -> Result<(), InputError> {
    let baseline = sidecar(reference)?;
    let source = sidecar(bf16)?;
    validate_pipeline_pair(&baseline, &source).map_err(|message| InputError::Sidecar {
        path: bf16.with_extension("json"),
        message,
    })
}

fn validate_pipeline_pair(
    baseline: &serde_json::Value,
    source: &serde_json::Value,
) -> Result<(), String> {
    for (meta, dtype) in [(baseline, "f32"), (source, "bf16")] {
        if meta["kind"] != "pipeline"
            || meta["dtype"] != dtype
            || meta["device"] != "cpu"
            || meta["decoded"] != true
        {
            return Err(format!("requires a decoded CPU {dtype} pipeline reference"));
        }
    }
    for key in ["prompt", "seed", "width", "height", "steps"] {
        if baseline.get(key).is_none_or(serde_json::Value::is_null) || baseline[key] != source[key]
        {
            return Err(format!("pipeline reference mismatch or missing {key}"));
        }
    }
    Ok(())
}

/// A non-negative integer field of a sidecar.
pub(crate) fn field(meta: &serde_json::Value, key: &str) -> Result<u64, InputError> {
    meta[key]
        .as_u64()
        .ok_or_else(|| InputError::Field(key.to_owned()))
}

/// `METALLIX_FLUX2_PRECISION`: unset or `f32` is f32, `bf16` is bf16, and
/// anything else is an error rather than a silent f32 run.
pub(crate) fn precision(lookup: Lookup<'_>) -> Result<Flux2Precision, InputError> {
    match lookup("METALLIX_FLUX2_PRECISION") {
        None => Ok(Flux2Precision::F32),
        Some(value) => match value.to_str() {
            Some("f32") => Ok(Flux2Precision::F32),
            Some("bf16") => Ok(Flux2Precision::Bf16),
            _ => Err(InputError::Precision(value.to_string_lossy().into_owned())),
        },
    }
}

/// Unwraps resolved gate inputs, failing the test with the reason.
pub(crate) fn require<T>(resolved: Result<T, InputError>) -> T {
    resolved.unwrap_or_else(|error| panic!("{error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_noise_floor_requires_the_same_completed_workload() {
        let baseline = serde_json::json!({
            "kind": "pipeline", "dtype": "f32", "device": "cpu", "decoded": true,
            "prompt": "a lake", "seed": 3, "width": 1024, "height": 1024, "steps": 4
        });
        let mut source = baseline.clone();
        source["dtype"] = "bf16".into();
        assert!(validate_pipeline_pair(&baseline, &source).is_ok());
        for (key, value) in [
            ("prompt", serde_json::json!("another prompt")),
            ("seed", serde_json::json!(4)),
            ("decoded", serde_json::json!(false)),
            ("device", serde_json::json!("mps")),
            ("kind", serde_json::json!("vae")),
        ] {
            let mut wrong = source.clone();
            wrong[key] = value;
            assert!(validate_pipeline_pair(&baseline, &wrong).is_err(), "{key}");
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("flux2-gate-inputs-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    #[test]
    fn unset_missing_and_empty_references_are_errors() {
        let unset: Lookup<'_> = &|_| None;
        assert!(matches!(file(unset, "X"), Err(InputError::Unset(_))));
        assert!(matches!(dir(unset, "X"), Err(InputError::Unset(_))));

        let absent = scratch("absent.safetensors");
        let lookup_absent = |_: &str| Some(absent.clone().into_os_string());
        assert!(matches!(
            file(&lookup_absent, "X"),
            Err(InputError::Missing { .. })
        ));

        // The zero-byte placeholders that once stood in for bf16 references.
        let empty = scratch("empty.safetensors");
        fs::write(&empty, b"").unwrap();
        let lookup_empty = |_: &str| Some(empty.clone().into_os_string());
        assert!(matches!(
            file(&lookup_empty, "X"),
            Err(InputError::Empty { .. })
        ));
        assert!(matches!(sidecar(&empty), Err(InputError::Sidecar { .. })));

        let real = scratch("real.safetensors");
        fs::write(&real, b"x").unwrap();
        let lookup_real = |_: &str| Some(real.clone().into_os_string());
        assert_eq!(file(&lookup_real, "X").unwrap(), real);
    }

    #[test]
    fn a_bf16_source_must_declare_the_cpu() {
        let reference = scratch("bf16.safetensors");
        fs::write(&reference, b"x").unwrap();
        let lookup = |_: &str| Some(reference.clone().into_os_string());
        for (sidecar, ok) in [
            (r#"{"device": "cpu"}"#, true),
            (r#"{"device": "mps"}"#, false),
            ("{}", false),
        ] {
            fs::write(reference.with_extension("json"), sidecar).unwrap();
            assert_eq!(cpu_reference(&lookup, "X").is_ok(), ok, "{sidecar}");
        }
    }

    #[test]
    fn an_unknown_precision_is_an_error_not_f32() {
        assert_eq!(precision(&|_| None).unwrap(), Flux2Precision::F32);
        assert_eq!(
            precision(&|_| Some("bf16".into())).unwrap(),
            Flux2Precision::Bf16
        );
        assert!(matches!(
            precision(&|_| Some("bfloat16".into())),
            Err(InputError::Precision(_))
        ));
    }

    #[test]
    #[should_panic(expected = "is not set")]
    fn require_fails_the_test() {
        let unset: Lookup<'_> = &|_| None;
        require(file(unset, "METALLIX_FLUX2_MODEL"));
    }
}
