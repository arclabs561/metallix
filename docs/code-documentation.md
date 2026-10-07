# Code documentation

This is the standard for Rust documentation in this workspace: crate and
module docs, item docs, error types, examples, inline comments and `unsafe`
justifications. It is modeled on Andrew Gallant's crates, studied at these
commits:

- [regex-automata](https://github.com/rust-lang/regex/tree/72d650cb0a880a01ab6dc2137c0888e8f89740f7/regex-automata/src)
  (`lib.rs`, `meta/regex.rs`, `util/search.rs`)
- [memchr](https://github.com/BurntSushi/memchr/tree/bd6068c30e9074a90c285e47912fa0b047d07597/src)
  (`lib.rs`, `arch/x86_64/avx2/memchr.rs`, `arch/generic/memchr.rs`)
- [jiff](https://github.com/BurntSushi/jiff/tree/4100a7c71125b9523029566d1d18f8b227ecd18c/crates/jiff/src)
  (`lib.rs`, `timestamp.rs`, `error/mod.rs`)
- [aho-corasick](https://github.com/BurntSushi/aho-corasick/tree/6c0abf5681bfc30bb9d8f7f52b68a350b436fffa/src)
  (`lib.rs`)

The common thread: docs state the contract a caller can rely on, examples
prove it, and comments explain what the code cannot say for itself.

## Crate docs

`lib.rs` opens with a crate doc that a new reader can act on without opening
another file. In order:

1. One paragraph saying what the crate does and, as plainly, what it does not.
   memchr's first line is "This library provides heavily optimized routines
   for string search primitives."
2. A runnable example of the most common call, near the top.
3. A map of the entry points: which function or type to use for which job.
   regex-automata has "Should I be using this crate?" and "Available regex
   engines"; aho-corasick has "Overview" and "Lower level APIs".
4. Cross-cutting themes that every item shares, documented once here instead
   of on each item. regex-automata's "Error reporting" section explains the
   `try_` naming and when search can fail.
5. Cargo features, one bullet each, with what enabling one changes.

For metallix, the scope sentence matters most: most crates are references or
contracts with deliberate limits. Say which limits, for example "a scalar CPU
reference, not a Metal kernel or a hardware rounding oracle", in the first
paragraph rather than scattering it across items.

## Item docs

The first paragraph is one sentence that makes sense alone in rustdoc's
module index. Use a verb for functions ("Expands", "Returns"), a noun phrase
for types ("An invalid packed runtime block").

The body states the contract, not the implementation:

- Shapes, units and layouts of every buffer argument: `[rows, reduction]`,
  bytes or elements, low nibble first.
- What happens to caller output on failure. Prefer atomic output and say so.
- Allocation and cost when a caller budgets for them.
- Guarantees that callers may rely on across versions. jiff writes "It is a
  semver guarantee that the only way for this to return an error is if the
  given value is out of range."
- Links to the item a reader needs next, as intra-doc links.

Refer to other code by item, not by path. In Rust docs, write an intra-doc
link such as [`ChatTemplate::render`] or [`blockfloat::decode_e4m3fn`]:
rustdoc resolves it on every gate run, so a rename or a crate move fails the
build instead of leaving a stale `crates/...` path. Public docs cannot link
private items, and the CPU gate documents crates without `metal`, so a link
from always-built docs to a `metal`-only item breaks there; name both kinds
in backticks. Markdown under `docs/` is not
checked, so it names the crate and item ("the server crate's `gpu` module")
and links a source file only when a line number matters.

Then these sections, in this order, each only when it applies:

`# Errors` names the conditions, grouped by variant where useful, and what
the caller's buffers hold afterwards:

```rust
/// # Errors
///
/// Returns [`BlockDecodeError::ScaleLength`] unless there is one scale per
/// 16 packed bytes, and [`BlockDecodeError::NonFiniteScale`] for a NaN scale
/// code. `output` is unchanged on every error.
```

`# Panics` lists reachable panics as concrete conditions, as
regex-automata's `hybrid::regex::Regex::find` does. Prefer code that cannot
panic. When Clippy requires the section only because the body contains an
`expect` on a proven invariant, say in one line why it cannot fire.

`# Safety` lists every obligation on the caller of an `unsafe fn` as a
bullet, as memchr's `One::find_raw` does.

`# Example` (or `# Example: <what it shows>` when there are several) comes
last. See [Examples are tests](#examples-are-tests).

Document a type's cross-cutting behavior on the type, in its own named
section, not on each method. regex-automata documents cache synchronization
once under `meta::Regex`'s "Synchronization and cloning".

## Error types

An error enum's doc says when it is returned. Each variant's doc states the
condition in the caller's terms, and each field states its index space or
unit:

```rust
/// An E4M3FN activation code denotes NaN.
#[error("nonfinite E4M3FN activation at element {element}")]
NonFiniteActivation {
    /// Flat index into `activation_codes`.
    element: usize,
},
```

Display messages are lowercase with no trailing period, and carry the values
that let someone find the bad input. If an error type deliberately offers
little introspection, say so and why, as jiff's `Error` does under
"Introspection is limited" and "Design".

## Examples are tests

Every example runs under `cargo test --doc`, so it must assert, not print.

- Propagate errors with `?` and end with a hidden
  `# Ok::<(), CrateError>(())` line, so the example reads like caller code.
- Assert exact values. For floats, compare `to_bits()` when the contract is
  bit exactness.
- Hide setup that distracts with `#`, never the behavior being shown.
- An example that needs Metal or a checkpoint is `no_run` and says why in its
  first comment line. Prefer a CPU example of the same contract.
- When an item has several examples, name each in its heading:
  `# Example: an error leaves the output alone`.

## Module docs

A module's `//!` doc says what the module owns and the design choice a reader
would otherwise rediscover. memchr's AVX2 module explains why it stops at
three needle bytes and why a dedicated count routine exists. Keep long
rationale in `docs/` and link it.

## Inline comments

Comments explain why: the invariant the next line depends on, the source the
behavior is pinned to, or the performance reason for a less obvious shape.
memchr's generic module first gives the simple algorithm in pseudocode, then
lists each optimization and why it pays.

```rust
// Compute into scratch and copy on success, so an overflow partway through
// still leaves `output` unchanged.
```

Cite pinned public sources with a commit-pinned URL. Private helpers get a
doc comment when they carry an invariant that is not visible at the call
site, as memchr does for its struct fields ("Used for haystacks less than 32
bytes").

## `unsafe`

The workspace denies `unsafe_code`; a module that needs it allows it at the
module level and says in its `//!` doc why, as the server crate's `gpu`
module does. Every `unsafe` block gets a `// SAFETY:` comment naming the
obligation it meets and how, every `unsafe fn` a `# Safety` section.

```rust
// SAFETY: both functions only write one `size_t` through the pointer.
```

## Lints

The gate already builds rustdoc with `-D warnings`, so broken intra-doc links
fail it. Each crate adds `#![deny(missing_docs)]` once its public items are
documented. Clippy's `missing_errors_doc` and `missing_panics_doc` keep the
sections honest: `missing_panics_doc` is on through `clippy::pedantic`, and
`missing_errors_doc` is enabled per crate as each is retrofitted.

`scripts/check_doc_ratchet.py` keeps that progress. A crate not in its
`PENDING` list must carry both attributes in its `lib.rs`, and a listed crate
that already carries them fails until it is removed from the list. Retrofit a
crate by documenting it, adding the attributes and removing it from `PENDING`
in one change. A new crate starts documented.

## Anti-patterns

- Narrating the code: `// loop over rows` above `for row in 0..rows`.
- History in comments: "previously X", "changed after the review". History
  belongs in commit messages.
- Restating the signature: "`rows`: the rows". Give units, shape or range.
- Claims the code does not support, such as "crate-private" on a `pub`
  item, or "exact" where the tests check a tolerance.
- Marketing words ("powerful", "blazing", "seamless") and undefined acronyms.
- Session context: internal tickets, agent notes or local paths.
- An `# Errors` section that says only "returns an error if the input is
  invalid".
