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
        // The marker may occur inside a JSON string argument. Let JSON consume
        // its complete value before checking the outer envelope delimiter.
        let mut values = serde_json::Deserializer::from_str(remaining).into_iter::<ToolCall>();
        let call = values
            .next()
            .ok_or("incomplete tool call")?
            .map_err(|error| format!("invalid tool call: {error}"))?;
        remaining = remaining[values.byte_offset()..]
            .trim_start()
            .strip_prefix("</tool_call>")
            .ok_or("incomplete tool call or trailing JSON data")?;
        if call.name.is_empty() || !call.arguments.is_object() {
            return Err("tool call requires a name and an arguments object".into());
        }
        calls.push(call);
        if calls.len() > 8 {
            return Err("at most eight calls per turn".into());
        }
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
    use proptest::prelude::*;
    use std::{
        fmt::Write as _,
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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn json_arguments_and_unicode_text_survive_envelopes(
            prefix in "[a-zA-Z0-9 \n🦀]{0,48}",
            suffix in "[a-zA-Z0-9 \n🦀]{0,48}",
            payloads in prop::collection::vec(prop_oneof![
                any::<String>(),
                Just("</tool_call>".to_owned()),
                Just("<tool_call>".to_owned()),
                Just("quotes \\\" and \\n and 🦀".to_owned()),
            ], 1..=8),
        ) {
            let mut input = prefix.clone();
            let expected: Vec<ToolCall> = payloads.into_iter().map(|payload| ToolCall {
                name: "search_file".into(),
                arguments: json!({"path":"README.md", "query":payload}),
            }).collect();
            for call in &expected {
                let body = json!({"name":call.name,"arguments":call.arguments});
                write!(input, "<tool_call>\n{body}\n</tool_call>").unwrap();
            }
            input.push_str(&suffix);
            let parsed = parse_turn(&input).map_err(TestCaseError::fail)?;
            prop_assert_eq!(parsed.calls, expected);
            prop_assert_eq!(parsed.text, prefix + &suffix);
        }

        #[test]
        fn incomplete_batch_never_exposes_completed_prefix_calls(payload in any::<String>()) {
            let body = json!({"name":"search_file","arguments":{"query":payload}});
            let input = format!("<tool_call>{body}</tool_call><tool_call>{{");
            prop_assert!(parse_turn(&input).is_err());
        }

        #[test]
        fn call_limit_is_enforced_for_every_oversized_batch(count in 9_usize..32) {
            let input = r#"<tool_call>{"name":"list_files","arguments":{"path":"."}}</tool_call>"#.repeat(count);
            prop_assert!(parse_turn(&input).is_err());
        }

        #[test]
        fn every_truncated_envelope_is_rejected(
            payload in any::<String>(),
            cut_seed in any::<usize>(),
        ) {
            let body = json!({"name":"read_file","arguments":{"path":payload}});
            let input = format!("<tool_call>{body}</tool_call>");
            let boundaries: Vec<_> = input.char_indices()
                .map(|(offset, _)| offset)
                .filter(|&offset| offset >= "<tool_call>".len())
                .collect();
            let cut = boundaries[cut_seed % boundaries.len()];
            prop_assert!(parse_turn(&input[..cut]).is_err());
        }

        #[test]
        fn extra_json_value_cannot_be_smuggled_inside_one_envelope(
            payload in any::<String>(),
            extra in prop_oneof![Just(json!(null)), Just(json!(false)), any::<i64>().prop_map(|x| json!(x)), any::<String>().prop_map(|x| json!(x))],
        ) {
            let body = json!({"name":"read_file","arguments":{"path":payload}});
            let input = format!("<tool_call>{body} {extra}</tool_call>");
            prop_assert!(parse_turn(&input).is_err());
        }

        #[test]
        fn nested_argument_values_survive_json_and_envelope_roundtrip(
            strings in prop::collection::vec(any::<String>(), 0..12),
            number in any::<i64>(),
            flag in any::<bool>(),
        ) {
            let arguments = json!({"nested":[{"strings":strings,"number":number,"flag":flag,"nothing":null}],"marker":"</tool_call>"});
            let body = json!({"name":"custom","arguments":arguments});
            let parsed = parse_turn(&format!("<tool_call>{body}</tool_call>")).map_err(TestCaseError::fail)?;
            prop_assert_eq!(parsed.calls.len(), 1);
            prop_assert_eq!(&parsed.calls[0].arguments, &arguments);
            prop_assert!(parsed.text.is_empty());
        }

        #[test]
        fn literal_search_preserves_source_line_numbers_and_unicode(
            rows in prop::collection::vec((any::<bool>(), "[a-z🦀]{0,24}"), 0..=140),
        ) {
            let root = TestRoot::new();
            let workspace = root.workspace();
            let lines: Vec<_> = rows.iter().map(|(hit, text)| {
                if *hit { format!("{text}NEEDLE🦀") } else { text.clone() }
            }).collect();
            std::fs::write(workspace.join("rows.txt"), lines.join("\n") + "\n").unwrap();
            let tools = WorkspaceTools::new(&workspace).unwrap();
            let result = tools.execute(&ToolCall {
                name: "search_file".into(),
                arguments: json!({"path":"rows.txt","query":"NEEDLE🦀"}),
            });
            let expected: Vec<_> = rows.iter().enumerate().filter(|(_, (hit, _))| *hit)
                .map(|(index, _)| json!({"line":index + 1,"text":lines[index]})).collect();
            if expected.len() > 128 {
                prop_assert!(result.is_err());
            } else {
                prop_assert_eq!(result.map_err(TestCaseError::fail)?, json!({"matches":expected}));
            }
        }

        #[test]
        fn pinned_workspace_survives_pathname_replacement(
            contents in "[a-zA-Z0-9 🦀]{0,96}",
            replacement_is_symlink in any::<bool>(),
        ) {
            let root = TestRoot::new();
            let workspace = root.workspace();
            let original = format!("original:{contents}");
            std::fs::write(workspace.join("note.txt"), &original).unwrap();
            let tools = WorkspaceTools::new(&workspace).unwrap();
            std::fs::rename(&workspace, root.0.join("retained")).unwrap();
            let replacement = root.0.join("replacement");
            std::fs::create_dir(&replacement).unwrap();
            std::fs::write(replacement.join("note.txt"), "replacement").unwrap();
            std::fs::write(replacement.join("extra.txt"), "replacement only").unwrap();
            if replacement_is_symlink {
                symlink(&replacement, &workspace).unwrap();
            } else {
                std::fs::rename(&replacement, &workspace).unwrap();
            }
            let read = tools.execute(&ToolCall {
                name: "read_file".into(),
                arguments: json!({"path":"note.txt"}),
            }).map_err(TestCaseError::fail)?;
            prop_assert_eq!(read, json!({"text":original}));
            let listed = tools.execute(&ToolCall {
                name: "list_files".into(),
                arguments: json!({"path":"."}),
            }).map_err(TestCaseError::fail)?;
            prop_assert_eq!(listed, json!({"entries":["note.txt"]}));
        }

        #[test]
        fn listing_limit_counts_real_entries_and_returns_sorted_names(count in 124_usize..=132) {
            let root = TestRoot::new();
            let workspace = root.workspace();
            let mut expected = Vec::new();
            for index in (0..count).rev() {
                let name = format!("entry-{index:03}");
                std::fs::write(workspace.join(&name), "").unwrap();
                expected.push(name);
            }
            expected.sort();
            let tools = WorkspaceTools::new(&workspace).unwrap();
            let result = tools.execute(&ToolCall {
                name: "list_files".into(),
                arguments: json!({"path":"."}),
            });
            if count <= 128 {
                prop_assert_eq!(result.map_err(TestCaseError::fail)?, json!({"entries":expected}));
            } else {
                prop_assert!(result.is_err());
            }
        }

        #[test]
        fn utf8_file_limit_counts_bytes_not_characters(characters in 8188_usize..=8196) {
            let root = TestRoot::new();
            let workspace = root.workspace();
            let text = "🦀".repeat(characters);
            std::fs::write(workspace.join("unicode.txt"), &text).unwrap();
            let tools = WorkspaceTools::new(&workspace).unwrap();
            let result = tools.read(Path::new("unicode.txt"));
            if characters <= 8192 {
                prop_assert_eq!(result.map_err(TestCaseError::fail)?, text);
            } else {
                prop_assert!(result.is_err());
            }
        }
    }
}
