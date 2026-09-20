//! Complete-call parsing and bounded, read-only workspace tools.

use std::{
    fs,
    io::Read,
    os::fd::OwnedFd,
    path::{Component, Path, PathBuf},
};

use rustix::fs::{self as rustix_fs, Dir, FileType, Mode, OFlags};
use serde::Deserialize;
use serde_json::{Value, json};

const MAX_FILE_BYTES: u64 = 32 * 1024;
const MAX_ENTRIES: usize = 128;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolCall {
    pub(crate) name: String,
    pub(crate) arguments: Value,
}

/// One model turn with complete tool calls removed from its visible text.
#[derive(Debug, PartialEq)]
pub(crate) struct ParsedTurn {
    pub(crate) text: String,
    pub(crate) calls: Vec<ToolCall>,
}

/// Parses complete Qwen tool-call envelopes while preserving surrounding text.
pub(crate) fn parse_turn(input: &str) -> Result<ParsedTurn, String> {
    let mut remaining = input;
    let mut text = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find("<tool_call>") {
        let before = &remaining[..start];
        if before.contains("</tool_call>") {
            return Err("unmatched tool-call closing marker".into());
        }
        text.push_str(before);
        remaining = &remaining[start + "<tool_call>".len()..];
        let end = remaining
            .find("</tool_call>")
            .ok_or("incomplete tool call")?;
        let call: ToolCall = serde_json::from_str(remaining[..end].trim())
            .map_err(|error| format!("invalid tool call: {error}"))?;
        if call.name.is_empty() || !call.arguments.is_object() {
            return Err("tool call requires a name and an arguments object".into());
        }
        calls.push(call);
        if calls.len() > 8 {
            return Err("at most eight calls per turn".into());
        }
        remaining = &remaining[end + "</tool_call>".len()..];
    }
    if remaining.contains("</tool_call>") {
        return Err("unmatched tool-call closing marker".into());
    }
    text.push_str(remaining);
    Ok(ParsedTurn { text, calls })
}

pub(crate) fn definitions() -> Vec<Value> {
    vec![
        json!({"type":"function","function":{"name":"read_file","description":"Read a UTF-8 file inside the workspace (maximum 32 KiB).","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}),
        json!({"type":"function","function":{"name":"list_files","description":"List up to 128 entries in one workspace directory.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}),
        json!({"type":"function","function":{"name":"search_file","description":"Find literal text in one UTF-8 workspace file (maximum 32 KiB).","parameters":{"type":"object","properties":{"path":{"type":"string"},"query":{"type":"string"}},"required":["path","query"],"additionalProperties":false}}}),
    ]
}

pub(crate) struct WorkspaceTools {
    root: OwnedFd,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PathArgs {
    path: PathBuf,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchArgs {
    path: PathBuf,
    query: String,
}

impl WorkspaceTools {
    pub(crate) fn new(root: &Path) -> Result<Self, String> {
        let root = root.canonicalize().map_err(|e| e.to_string())?;
        let root =
            rustix_fs::open(&root, directory_flags(), Mode::empty()).map_err(|e| e.to_string())?;
        Ok(Self { root })
    }

    fn open_relative(&self, relative: &Path, final_flags: OFlags) -> Result<OwnedFd, String> {
        let mut components = Vec::new();
        for component in relative.components() {
            match component {
                Component::Normal(component) => components.push(component),
                Component::CurDir => {}
                Component::RootDir | Component::ParentDir | Component::Prefix(_) => {
                    return Err("tool paths must stay below the workspace root".into());
                }
            }
        }
        if components.is_empty() {
            return rustix_fs::openat(&self.root, ".", final_flags, Mode::empty())
                .map_err(|e| e.to_string());
        }
        let mut directory = rustix_fs::openat(&self.root, ".", directory_flags(), Mode::empty())
            .map_err(|e| e.to_string())?;
        let last = components.len() - 1;
        for (index, component) in components.into_iter().enumerate() {
            let flags = if index == last {
                final_flags
            } else {
                directory_flags()
            };
            directory = rustix_fs::openat(&directory, component, flags, Mode::empty())
                .map_err(|e| e.to_string())?;
        }
        Ok(directory)
    }

    fn read(&self, relative: &Path) -> Result<String, String> {
        let fd = self.open_relative(relative, file_flags())?;
        if !FileType::from_raw_mode(rustix_fs::fstat(&fd).map_err(|e| e.to_string())?.st_mode)
            .is_file()
        {
            return Err("read requires a regular file".into());
        }
        let file = fs::File::from(fd);
        let metadata = file.metadata().map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES {
            return Err("read requires a regular file of at most 32 KiB".into());
        }
        let mut text = String::new();
        file.take(MAX_FILE_BYTES + 1)
            .read_to_string(&mut text)
            .map_err(|e| e.to_string())?;
        if text.len() as u64 > MAX_FILE_BYTES {
            return Err("file grew beyond the read limit".into());
        }
        Ok(text)
    }

    fn list(&self, relative: &Path) -> Result<Vec<String>, String> {
        let fd = self.open_relative(relative, directory_flags())?;
        let mut entries = Vec::new();
        for entry in Dir::read_from(&fd).map_err(|e| e.to_string())? {
            if entries.len() == MAX_ENTRIES {
                return Err("directory exceeds 128 entries; choose a narrower path".into());
            }
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name();
            if matches!(name.to_bytes(), b"." | b"..") {
                continue;
            }
            entries.push(name.to_string_lossy().into_owned());
        }
        entries.sort();
        Ok(entries)
    }

    pub(crate) fn execute(&self, call: &ToolCall) -> Result<Value, String> {
        match call.name.as_str() {
            "read_file" => {
                let args: PathArgs =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                self.read(&args.path).map(|text| json!({"text":text}))
            }
            "list_files" => {
                let args: PathArgs =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                self.list(&args.path)
                    .map(|entries| json!({"entries":entries}))
            }
            "search_file" => {
                let args: SearchArgs =
                    serde_json::from_value(call.arguments.clone()).map_err(|e| e.to_string())?;
                if args.query.is_empty() || args.query.len() > 1024 {
                    return Err("query must contain 1..=1024 bytes".into());
                }
                let text = self.read(&args.path)?;
                let matches: Vec<_> = text
                    .lines()
                    .enumerate()
                    .filter(|(_, line)| line.contains(&args.query))
                    .take(MAX_ENTRIES + 1)
                    .map(|(line, text)| json!({"line":line+1,"text":text}))
                    .collect();
                if matches.len() > MAX_ENTRIES {
                    return Err("more than 128 matches; narrow the query".into());
                }
                Ok(json!({"matches":matches}))
            }
            _ => Err(format!("unknown tool {}", call.name)),
        }
    }
}

fn directory_flags() -> OFlags {
    OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK | OFlags::DIRECTORY
}

fn file_flags() -> OFlags {
    OFlags::RDONLY | OFlags::CLOEXEC | OFlags::NOFOLLOW | OFlags::NONBLOCK
}

/// Compile only bounded local schemas; tool schemas must not cause retrieval.
pub(crate) fn validator(schema: &Value) -> Result<jsonschema::Validator, String> {
    fn check_refs(value: &Value) -> Result<(), String> {
        match value {
            Value::Object(map) => {
                for (key, value) in map {
                    if matches!(key.as_str(), "$ref" | "$dynamicRef")
                        && value.as_str().is_none_or(|s| !s.starts_with('#'))
                    {
                        return Err("only document-local schema references are supported".into());
                    }
                    check_refs(value)?;
                }
            }
            Value::Array(values) => {
                for value in values {
                    check_refs(value)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
    if schema.to_string().len() > 32 * 1024 {
        return Err("tool schema exceeds 32 KiB".into());
    }
    check_refs(schema)?;
    jsonschema::validator_for(schema).map_err(|e| format!("invalid tool schema: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        os::unix::fs::symlink,
        path::PathBuf,
        process::Command,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static NEXT_WORKSPACE: AtomicU64 = AtomicU64::new(0);

    struct TestRoot(PathBuf);

    impl TestRoot {
        fn new() -> Self {
            let sequence = NEXT_WORKSPACE.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "metallix-chat-tools-{}-{nanos}-{sequence}",
                std::process::id(),
            ));
            std::fs::create_dir(&path).expect("unique test root");
            Self(path)
        }

        fn workspace(&self) -> PathBuf {
            let workspace = self.0.join("workspace");
            std::fs::create_dir(&workspace).expect("workspace");
            workspace
        }
    }

    impl Drop for TestRoot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn complete_calls_only_and_typed_arguments() {
        let turn = parse_turn("before <tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"README.md\"}}\n</tool_call> after").unwrap();
        assert_eq!(turn.text, "before  after");
        assert_eq!(turn.calls.len(), 1);
        assert_eq!(turn.calls[0].arguments["path"], "README.md");
        assert!(parse_turn("<tool_call>{\"name\":\"x\"").is_err());
        assert!(
            parse_turn("<tool_call>{\"name\":\"x\",\"arguments\":\"{}\"}</tool_call>").is_err()
        );
        assert!(parse_turn("</tool_call>").is_err());
    }

    #[test]
    fn workspace_tools_keep_regular_reads_below_pinned_root() {
        let root = TestRoot::new();
        let workspace = root.workspace();
        std::fs::write(workspace.join("inside.txt"), "inside").expect("inside file");
        let outside = root.0.join("outside");
        std::fs::create_dir(&outside).expect("outside directory");
        std::fs::write(outside.join("secret.txt"), "outside").expect("outside file");
        let nested = outside.join("nested");
        std::fs::create_dir(&nested).expect("outside nested directory");
        std::fs::write(nested.join("secret.txt"), "outside nested file")
            .expect("outside nested file");
        symlink(outside.join("secret.txt"), workspace.join("leaf-link")).expect("leaf symlink");
        symlink(&nested, workspace.join("directory-link")).expect("directory symlink");
        let tools = WorkspaceTools::new(&workspace).expect("pinned workspace");

        assert_eq!(
            tools.read(Path::new("inside.txt")).expect("regular read"),
            "inside"
        );
        assert!(tools.read(Path::new("../outside/secret.txt")).is_err());
        assert!(tools.read(Path::new("leaf-link")).is_err());
        assert!(tools.read(Path::new("directory-link/secret.txt")).is_err());
        assert!(tools.list(Path::new("directory-link")).is_err());
    }

    #[test]
    fn workspace_fifo_is_rejected_without_a_reader() {
        let root = TestRoot::new();
        let workspace = root.workspace();
        let status = Command::new("mkfifo")
            .arg(workspace.join("blocked"))
            .status()
            .expect("mkfifo available on supported Unix targets");
        assert!(status.success(), "mkfifo test fixture");
        let tools = WorkspaceTools::new(&workspace).expect("pinned workspace");

        assert!(tools.read(Path::new("blocked")).is_err());
    }

    #[test]
    fn schema_validation_stays_local_and_checks_required_arguments() {
        assert!(validator(&json!({"$ref":"https://example.invalid/schema"})).is_err());
        assert!(validator(&json!({"$dynamicRef":"file:///tmp/schema"})).is_err());
        assert!(validator(&json!({"type":"not-a-json-schema-type"})).is_err());
        let local = validator(&json!({
            "$defs":{"path":{"type":"string"}},
            "type":"object", "properties":{"path":{"$ref":"#/$defs/path"}},
            "required":["path"], "additionalProperties":false
        }))
        .unwrap();
        assert!(local.is_valid(&json!({"path":"README.md"})));
        assert!(!local.is_valid(&json!({"path":5})));
        assert!(!local.is_valid(&json!({})));
    }

    #[test]
    fn workspace_tools_reject_unknown_arguments() {
        let tools = WorkspaceTools::new(Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
        assert!(
            tools
                .execute(&ToolCall {
                    name: "read_file".into(),
                    arguments: json!({"path":"Cargo.toml","extra":true})
                })
                .is_err()
        );
        let result = tools
            .execute(&ToolCall {
                name: "search_file".into(),
                arguments: json!({"path":"Cargo.toml","query":"name = \"server\""}),
            })
            .unwrap();
        assert_eq!(result["matches"][0]["text"], "name = \"server\"");
    }
}
