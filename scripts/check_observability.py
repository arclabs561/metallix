# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Keep Rust telemetry passive, using a bounded lexical check (not type analysis).

Recognizes tracing-qualified macros/attributes, direct/grouped imports and
aliases, and tracing crate aliases. Imports are file-wide; reexports, macro
expansion, cfg_attr and shadowing need review. All .record(...) arguments are
checked conservatively because receiver types cannot be resolved here. Only
named readback/RNG calls are forbidden; helper bodies are not followed.
Comments, strings (including raw/byte strings) and character literals are
masked before matching. No tensor-name blacklist or timing heuristics.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MACROS = frozenset(
    [
        "event",
        "span",
        "trace",
        "debug",
        "info",
        "warn",
        "error",
        "trace_span",
        "debug_span",
        "info_span",
        "warn_span",
        "error_span",
        "enabled",
        "event_enabled",
        "span_enabled",
    ]
)
FORBIDDEN = frozenset(
    [
        "eval",
        "eval_all",
        "async_eval",
        "as_slice",
        "to_vec",
        "item",
        "item_cast",
        "item_exact",
        "next_u32",
        "next_u64",
        "next_f32",
        "next_f64",
        "fill_bytes",
        "try_fill_bytes",
        "random",
        "random_range",
        "gen",
        "gen_range",
        "sample",
        "sample_iter",
        "sample_categorical",
        "sample_masked",
        "sample_nucleus",
        "sample_top_k_candidates",
        "sample_constrained",
    ]
)
TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|::|[^\s]")


def masked(source: str) -> str:
    """Blank comments and literals while retaining offsets and line breaks."""
    result = list(source)
    i = 0
    while i < len(source):
        start = i
        if source.startswith("//", i):
            end = source.find("\n", i)
            i = len(source) if end < 0 else end
        elif source.startswith("/*", i):
            depth = 1
            i += 2
            while i < len(source) and depth:
                if source.startswith("/*", i):
                    depth += 1
                    i += 2
                elif source.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
        elif match := re.match(r'(?:br|cr|r)(#*)"', source[i:]):
            stop = '"' + match[1]
            end = source.find(stop, i + len(match[0]))
            i = len(source) if end < 0 else end + len(stop)
        elif source[i] == '"':
            i += 1
            while i < len(source):
                if source[i] == "\\":
                    i += 2
                elif source[i] == '"':
                    i += 1
                    break
                else:
                    i += 1
        elif match := re.match(
            r"'(?:\\(?:u\{[0-9a-fA-F_]+\}|x[0-9a-fA-F]{2}|.)|[^'\\\n])'", source[i:]
        ):
            i += len(match[0])
        else:
            i += 1
            continue
        for pos in range(start, min(i, len(source))):
            if result[pos] != "\n":
                result[pos] = " "
    return "".join(result)


def closing(tokens: list[str], start: int) -> int:
    """Find the end of a balanced Rust token group."""
    stack = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token in pairs:
            stack.append(pairs[token])
        elif token in pairs.values():
            if not stack or stack.pop() != token:
                return len(tokens)
            if not stack:
                return index
    return len(tokens)


def imports(code: str) -> tuple[set[str], dict[str, str]]:
    """Resolve direct tracing imports and one-level grouped aliases."""
    namespaces = {"tracing"}
    names = {}
    for alias in re.findall(r"\buse\s+(?:::)?\s*tracing\s+as\s+(\w+)\s*;", code):
        namespaces.add(alias)
    for group in re.findall(r"\buse\s+(?:::)?\s*tracing\s*::\s*\{([^{}]+)\}\s*;", code):
        namespaces.update(re.findall(r"\bself\s+as\s+(\w+)", group))
    for namespace in sorted(namespaces):
        for match in re.finditer(
            rf"\buse\s+(?:::)?\s*{namespace}\s*::\s*([^;]+);", code
        ):
            body = match[1].strip().strip("{}").strip()
            for item in body.split(","):
                parts = item.strip().split()
                if parts == ["*"]:
                    names.update({name: name for name in MACROS | {"instrument"}})
                elif parts and parts[0] in MACROS | {"instrument"}:
                    if len(parts) == 1:
                        names[parts[0]] = parts[0]
                    elif len(parts) == 3 and parts[1] == "as":
                        names[parts[2]] = parts[0]
    return namespaces, names


def after_type_arguments(tokens: list[str], start: int) -> int:
    """Skip a call's optional turbofish, including nested generic arguments."""
    if tokens[start : start + 2] != ["::", "<"]:
        return start
    start += 2
    depth = 1
    while start < len(tokens) and depth:
        depth += (tokens[start] == "<") - (tokens[start] == ">")
        start += 1
    return start


def violations(source: str) -> list[tuple[int, str]]:
    """Return line-numbered policy violations in one Rust source file."""
    code = masked(source)
    matches = list(TOKEN.finditer(code))
    tokens = [match[0] for match in matches]
    namespaces, names = imports(code)
    found = []
    for i, token in enumerate(tokens):
        name = names.get(token)
        qualified = i >= 2 and tokens[i - 1] == "::" and tokens[i - 2] in namespaces
        if qualified:
            name = token
        path_start = i - 2 if qualified else i
        if path_start > 0 and tokens[path_start - 1] == "::":
            path_start -= 1
        attribute = name == "instrument" and tokens[
            max(0, path_start - 2) : path_start
        ] == ["#", "["]
        macro = name in MACROS and tokens[i + 1 : i + 2] == ["!"]
        record = token == "record" and i > 0 and tokens[i - 1] == "."
        start = i + (2 if macro else 1)
        if record:
            start = after_type_arguments(tokens, start)
        if not (attribute or macro or record):
            continue
        line = source.count("\n", 0, matches[i].start()) + 1
        if start >= len(tokens) or tokens[start] not in "([{":
            if attribute:
                found.append((line, "instrument requires top-level skip_all"))
            continue
        end = closing(tokens, start)
        if attribute:
            depth = 0
            skipped = False
            for j in range(start + 1, end):
                t = tokens[j]
                if (
                    t == "skip_all"
                    and depth == 0
                    and tokens[j - 1] in {"(", ","}
                    and tokens[j + 1] in {",", ")"}
                ):
                    skipped = True
                if t in "([{":
                    depth += 1
                elif t in ")]}":
                    depth -= 1
            if not skipped:
                found.append((line, "instrument requires top-level skip_all"))
        for j in range(start + 1, end):
            if tokens[j] not in FORBIDDEN:
                continue
            after = after_type_arguments(tokens, j + 1)
            if tokens[after : after + 1] == ["("]:
                call_line = source.count("\n", 0, matches[j].start()) + 1
                found.append((call_line, f"telemetry must not call {tokens[j]}"))
    return sorted(set(found))


def main() -> int:
    """Check Rust files under declared workspace members, including tests."""
    import tomllib

    members = tomllib.loads((ROOT / "Cargo.toml").read_text())["workspace"]["members"]
    files = sorted(
        {
            path
            for member in members
            for path in (ROOT / member).rglob("*.rs")
            if "target" not in path.parts
        }
    )
    total = 0
    for path in files:
        for line, message in violations(path.read_text()):
            print(f"{path.relative_to(ROOT)}:{line}: {message}", file=sys.stderr)
            total += 1
    print(f"observability: {len(files)} Rust files checked, {total} violations")
    return int(total != 0)


if __name__ == "__main__":
    raise SystemExit(main())
