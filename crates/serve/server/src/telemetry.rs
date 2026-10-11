//! Process-wide tracing subscriber for the `mx` and `metallix` binaries.
//!
//! Library crates only emit spans and events; this module is the one place a
//! subscriber is installed. Logs go synchronously to stderr so the last lines
//! before an abort (an MLX `SIGABRT`, say) are not lost in a buffer.

#[cfg(feature = "timeline")]
use std::sync::Mutex;
use std::{io::IsTerminal as _, path::Path};

use tracing_subscriber::{
    EnvFilter, Layer as _, Registry,
    layer::{Layered, SubscriberExt as _},
};

/// The filtered registry every output layer sits on.
type Base = Layered<EnvFilter, Registry>;
type BoxedLayer = Box<dyn tracing_subscriber::Layer<Base> + Send + Sync>;

/// Filter used when neither `METALLIX_LOG` nor `RUST_LOG` is set.
const DEFAULT_FILTER: &str = "warn";

#[cfg(feature = "timeline")]
static TIMELINE: Mutex<Option<tracing_chrome::FlushGuard>> = Mutex::new(None);

/// Installs the subscriber. `trace_out` adds a Chrome/Perfetto JSON timeline,
/// which needs the `timeline` feature.
pub(crate) fn init(trace_out: Option<&Path>) -> Result<(), String> {
    let (filter, invalid) = filter();
    let fmt = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_thread_ids(true)
        .with_thread_names(true)
        .boxed();
    let mut layers: Vec<BoxedLayer> = vec![fmt];
    if let Some(path) = trace_out {
        layers.push(timeline_layer(path)?);
    }
    let subscriber = Registry::default().with(filter).with(layers);
    // `try_init` also routes `log` records from dependencies into tracing.
    tracing_subscriber::util::SubscriberInitExt::try_init(subscriber)
        .map_err(|error| format!("could not install the log subscriber: {error}"))?;
    if let Some((name, error)) = invalid {
        tracing::warn!("ignoring invalid {name}: {error}; using {DEFAULT_FILTER}");
    }
    Ok(())
}

/// `METALLIX_LOG`, else `RUST_LOG`, else [`DEFAULT_FILTER`]; an invalid
/// directive falls back to the default and is reported once logging is up.
fn filter() -> (EnvFilter, Option<(&'static str, String)>) {
    for name in ["METALLIX_LOG", "RUST_LOG"] {
        if let Ok(directives) = std::env::var(name) {
            return match EnvFilter::try_new(&directives) {
                Ok(filter) => (filter, None),
                Err(error) => (
                    EnvFilter::new(DEFAULT_FILTER),
                    Some((name, error.to_string())),
                ),
            };
        }
    }
    (EnvFilter::new(DEFAULT_FILTER), None)
}

#[cfg(feature = "timeline")]
fn timeline_layer(path: &Path) -> Result<BoxedLayer, String> {
    let file = std::fs::File::create(path)
        .map_err(|error| format!("could not create {}: {error}", path.display()))?;
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .writer(file)
        .include_args(true)
        .build();
    *TIMELINE.lock().expect("timeline lock") = Some(guard);
    Ok(layer.boxed())
}

#[cfg(not(feature = "timeline"))]
fn timeline_layer(path: &Path) -> Result<BoxedLayer, String> {
    Err(format!(
        "--trace-out {} requires a build with --features timeline",
        path.display()
    ))
}

/// Requests that the writer flush buffered timeline entries to disk.
/// This does not wait for the writer or complete the JSON array; [`finish`]
/// waits for the writer and closes the array for strict JSON parsing.
#[cfg(feature = "metal")]
pub(crate) fn flush() {
    #[cfg(feature = "timeline")]
    if let Some(guard) = TIMELINE.lock().expect("timeline lock").as_ref() {
        guard.flush();
    }
}

/// On SIGTERM or SIGINT, completes the timeline and exits 0. `mx serve` ends
/// by signal, and the default action would drop entries still buffered on
/// the timeline's writer thread: [`flush`] only asks that thread to write,
/// while [`finish`] waits for it.
#[cfg(feature = "metal")]
pub(crate) fn finish_on_signal() -> Result<(), String> {
    use signal_hook::consts::{SIGINT, SIGTERM};
    let mut signals = signal_hook::iterator::Signals::new([SIGTERM, SIGINT])
        .map_err(|error| format!("could not watch for SIGTERM: {error}"))?;
    std::thread::Builder::new()
        .name(String::from("signals"))
        .spawn(move || {
            if signals.forever().next().is_some() {
                finish();
                std::process::exit(0);
            }
        })
        .map_err(|error| format!("could not start the signal thread: {error}"))?;
    Ok(())
}

/// Completes the timeline file. Call before `std::process::exit`, which skips
/// destructors.
pub(crate) fn finish() {
    #[cfg(feature = "timeline")]
    drop(TIMELINE.lock().expect("timeline lock").take());
}

/// The per-model file a `mx serve` child writes for a front-process path:
/// `trace.json` becomes `trace.<model>.json`.
#[cfg(feature = "metal")]
pub(crate) fn child_path(path: &Path, model: &str) -> std::path::PathBuf {
    let stem = path.file_stem().map_or_else(
        || std::ffi::OsString::from("trace"),
        std::ffi::OsStr::to_os_string,
    );
    let mut name = stem;
    name.push(".");
    name.push(model);
    if let Some(extension) = path.extension() {
        name.push(".");
        name.push(extension);
    }
    path.with_file_name(name)
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[cfg(feature = "timeline")]
    #[test]
    fn initialized_trace_finishes_as_strict_json_with_completed_span() {
        const CHILD: &str = "METALLIX_TRACE_LIFECYCLE_CHILD";
        const TRACE: &str = "METALLIX_TRACE_LIFECYCLE_PATH";
        if let Ok(mode) = std::env::var(CHILD) {
            let path = PathBuf::from(std::env::var_os(TRACE).expect("child trace path"));
            init(Some(&path)).expect("initialize actual subscriber");
            if mode != "normal" {
                finish_on_signal().expect("install actual signal finalizer");
            }
            {
                let span = tracing::info_span!("trace.lifecycle.completed");
                let _entered = span.enter();
                tracing::info!("completed lifecycle fixture");
            }
            match mode.as_str() {
                "normal" => {
                    finish();
                    finish(); // Repeated finalization must remain harmless.
                    std::process::exit(0);
                }
                "term" | "int" => {
                    let signal = if mode == "term" {
                        signal_hook::consts::SIGTERM
                    } else {
                        signal_hook::consts::SIGINT
                    };
                    signal_hook::low_level::raise(signal).expect("raise termination signal");
                    std::thread::sleep(std::time::Duration::from_secs(5));
                    panic!("signal finalizer did not exit");
                }
                _ => panic!("unknown lifecycle child mode"),
            }
        }

        // A subprocess isolates the global subscriber and termination signals
        // from the test harness. Exiting skips destructors in all three modes.
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("metallix-trace-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&directory).expect("fixture directory");
        for mode in ["normal", "term", "int"] {
            let path = directory.join(format!("{mode}.json"));
            let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args([
                    "--exact",
                    "telemetry::tests::initialized_trace_finishes_as_strict_json_with_completed_span",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, mode)
                .env(TRACE, &path)
                .env("METALLIX_LOG", "info")
                .output()
                .expect("run lifecycle child");
            assert!(output.status.success(), "{mode}: {output:?}");
            let trace: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path).expect("trace file"))
                    .expect("finalized trace must be strict JSON");
            let spans: Vec<_> = trace
                .as_array()
                .expect("Chrome trace array")
                .iter()
                .filter(|event| event["name"] == "trace.lifecycle.completed")
                .collect();
            assert_eq!(spans.len(), 2, "{mode}: completed span begin and end");
            assert_eq!(spans[0]["ph"], "B");
            assert_eq!(spans[1]["ph"], "E");
            assert_eq!(spans[0]["pid"], spans[1]["pid"]);
            assert_eq!(spans[0]["tid"], spans[1]["tid"]);
            assert!(
                spans[0]["ts"].as_f64().expect("begin timestamp")
                    <= spans[1]["ts"].as_f64().expect("end timestamp")
            );
        }
        std::fs::remove_dir_all(directory).expect("remove owned trace fixture");
    }

    #[test]
    fn child_paths_keep_the_directory_and_extension() {
        assert_eq!(
            child_path(Path::new("/tmp/run/trace.json"), "qwen"),
            PathBuf::from("/tmp/run/trace.qwen.json")
        );
        assert_eq!(
            child_path(Path::new("capture.gputrace"), "embed"),
            PathBuf::from("capture.embed.gputrace")
        );
        assert_eq!(
            child_path(Path::new("trace"), "julia"),
            PathBuf::from("trace.julia")
        );
    }
}
