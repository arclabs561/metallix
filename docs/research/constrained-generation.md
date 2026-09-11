# Constrained generation: a portable boundary for Rust serving

## Decision

Use **LLGuidance 1.8.0** as the first experimental constrained-decoding
backend by adding a real Qwen tokenizer bridge to the existing cached greedy
generation path.
Keep it behind a small engine-owned *per-request constraint session*, not in a
model adapter and not in a universal transformer trait.  It is a native Rust
crate with an explicit `compute_mask` / `commit_token` loop, clone support and
typed parser limits.  Its current release requires Rust 1.87, which the
workspace now declares. Local smoke results are recorded below; model-loop
integration and performance measurement remain separate gates.

Do not write a JSON-schema compiler or grammar parser.  The first public
contract should accept a JSON Schema document as an opaque `serde_json::Value`
or a named grammar source, report the selected backend/version and supported
mode, and validate the emitted bytes independently before returning success.
Grammar acceptance establishes **syntax only**.  It does not establish
application-specific semantics, authorization, a tool's argument validity, or
successful tool execution.

Evidence below was checked on 2026-09-11 against pinned upstream source.
The native Rust dependency and both smoke tests have been compiled and run;
no constrained model-generation benchmark has been run.

## Why this is an engine concern

A grammar consumes bytes and returns allowed token IDs.  It neither knows nor
should know whether the model is dense, MoE, multimodal, recurrent, MLA, has
cross-attention, or stores K/V in pages.  The engine owns the loop; each model
adapter later supplies logits and transactional sequence-state operations.

The byte mapping is a correctness boundary.  Store the tokenizer's token-ID to
**exact byte sequence** table and special/EOS IDs with the tokenizer revision;
do not derive it by repeatedly decoding a token to display text.  Tokenizer
normalization, byte fallback, byte-level encodings and special tokens make that
shortcut unsafe.  XGrammar's `TokenizerInfo` makes vocabulary type,
post-processed vocabulary, stop IDs and prefix-space policy explicit; its
matcher is constructed from that tokenizer-specific compiled grammar
([`tokenizer_info.h`](https://github.com/mlc-ai/xgrammar/blob/d02ad2b155a5f0c3eaa711950d154218d982ae7e/include/xgrammar/tokenizer_info.h),
[`matcher.h`](https://github.com/mlc-ai/xgrammar/blob/d02ad2b155a5f0c3eaa711950d154218d982ae7e/include/xgrammar/matcher.h)).
LLGuidance likewise builds a `ParserFactory` around a `TokEnv` and reuses that
factory for per-request parsers
([`factory.rs`](https://github.com/guidance-ai/llguidance/blob/v1.8.0/parser/src/factory.rs)).

EOS is permitted only when the grammar is in an accepting state.  A user stop
string is a separate policy: either model it in the grammar or return an
explicit `StoppedBeforeComplete` result, never label a truncated byte stream
as schema-valid.  Respect the model's EOS ID set rather than assuming the
single `eos_token_id` currently read by
[`qwen_forward.rs`](../../crates/server/src/qwen_forward.rs); different
tokenizers and chat modes can have multiple terminal IDs.

## Two credible backends

| Option | Evidence and fit | Cost / reversal criterion |
|---|---|---|
| **LLGuidance 1.8.0 (recommended first)** | MIT, Rust `rlib`/`staticlib`/`cdylib`; its `Constraint` exposes `compute_mask`, `commit_token`, `deep_clone`, forced-token splices and typed stop results ([crate manifest](https://github.com/guidance-ai/llguidance/blob/v1.8.0/parser/Cargo.toml), [`constraint.rs`](https://github.com/guidance-ai/llguidance/blob/v1.8.0/parser/src/constraint.rs)).  It supports a documented large subset of JSON Schema, regex and Lark-like CFG ([README](https://github.com/guidance-ai/llguidance/tree/v1.8.0)). | Dependency-lock and schema-coverage validation remain; exercise the public subset.  Reverse this choice if its token-table construction cannot faithfully represent Qwen's tokenizer or its measured CPU mask + Metal application cost breaches the decode latency budget. |
| **XGrammar v0.2.6** | Apache-2.0 C++ library with a portable static library/header surface.  Its compiler is tokenizer-scoped and caches compiled grammars; matcher fills preallocated packed masks, supports rollback, fork and speculative-tree traversal ([`compiler.h`](https://github.com/mlc-ai/xgrammar/blob/d02ad2b155a5f0c3eaa711950d154218d982ae7e/include/xgrammar/compiler.h), [`matcher.h`](https://github.com/mlc-ai/xgrammar/blob/d02ad2b155a5f0c3eaa711950d154218d982ae7e/include/xgrammar/matcher.h)).  The upstream README lists macOS/Apple Silicon and Qwen/DeepSeek among its targets ([README](https://github.com/mlc-ai/xgrammar/tree/d02ad2b155a5f0c3eaa711950d154218d982ae7e)). | First-party APIs are C++/Python/JS/Swift, not Rust; a new `unsafe` FFI boundary conflicts with the workspace's `unsafe_code = "deny"` unless isolated and audited.  Prefer it later only if its batch/speculative APIs materially beat the Rust option on representative schemas and the binding has a maintained, pinned safety story. |

The quoted upstream timing claims are not Mac results: they omit or vary Metal
mask application, sampling, scheduling and model decode.  Keep LLGuidance's
on-the-fly trie and parser/lexer limits request-configurable and return limit
errors, not panics ([technical details](https://github.com/guidance-ai/llguidance/tree/v1.8.0)).

## Identity rules

A numeric `TokenId` newtype alone does not prevent vocabulary mixing.  When
the serving seam exists, bind its tokens, masks and sampler input to one opaque
tokenizer fingerprint and reject a mismatch at construction/commit.  Keep that
small vocabulary boundary local to serving; it must not become a transformer
trait carrying cache shape, attention architecture or quantization.

For **future speculative decoding**, checkpoint both parser and sequence/KV
state, commit the same accepted prefix on both, and roll back both rejected
suffixes.  XGrammar has `Fork`/`Rollback` and tree masks; LLGuidance exposes
`deep_clone` and backtracking.  This is not needed for the first greedy path.

## Mask and measurement contract

Use a packed, preallocated token bitset of
`ceil(vocab_size / 32)` `u32` words per active sequence.  Define its bit order
once in tests (`word = token / 32`, `bit = token % 32`) and include the tail
word's out-of-vocabulary bits in rejection tests.  This matches XGrammar's
documented preallocated `int32` mask shape.  The grammar backend runs on CPU;
for the first one-sequence Mac path, mask the final logits on the same side
where logits are actually sampled.  Do not add a host round trip merely for
constraints.  When logits remain in Metal, upload/reuse the packed mask and
apply `-infinity` (or the sampler's defined rejected value) in the final
logits/sampling kernel.  This is an optimization hypothesis, not an assumed
win: tiny vocabularies or forced-token paths may be faster on CPU.

After warmup, split schema compile/cache, CPU `compute_mask`, mask
transfer/application, sampler and model decode.  Record TTFT, ITL, completion
time, tokens/s, grammar completion, independently validated-schema rate,
forced tokens and parser-limit failures.  A syntax-valid object still needs
schema semantics and separate tool argument/execution policy.

## Supported surface and gates

Start with objects/properties/required, arrays/items, scalar types,
enums/const and bounded strings/numbers.  Add unions only after fixtures;
reject unsupported keywords and limit depth/size.  This profile is Metallix
policy, not a claim that an upstream "large subset" equals all JSON Schema.

The existing Qwen cached greedy diagnostic is sufficient as a model-loop
starting point; it does **not** need HTTP serving or speculative rollback for a
first constrained generation. The new optional session now owns the token-byte
trie, grammar matcher, mask application and independent validator. Remaining
integration prerequisites beyond that diagnostic are:

- no canonical text encoder or immutable checkpoint/tokenizer revision binding;
- only a single parsed `eos_token_id`, no multi-EOS/stop policy;
- no stochastic, speculative or batched constrained decoding; and
- no device-resident masked sampling path.

The first implementation is feature-gated in
[`llguidance_qwen_tokenizer.rs`](../../crates/engine/tests/llguidance_qwen_tokenizer.rs):
a deterministic one-byte test covers valid/invalid JSON and EOS; an ignored
local-Qwen test reads `METALLIX_QWEN_TOKENIZER`, builds LLGuidance's exact
token-byte table, reconstructs consumed bytes for `serde_json` parsing, and
checks its trie masks.  The latter uses
`ApproximateTokEnv` greedy trie tokenization: it qualifies vocabulary/mask
compatibility only, **not** Hugging Face canonical BPE encoding, forced-token
correctness, model logits, or local performance.  Do not enable fast-forward
tokens until canonical encoding is separately qualified.

Executed locally:

```sh
METALLIX_QWEN_TOKENIZER=/path/to/Qwen3-0.6B/tokenizer.json \
  cargo test -p engine --features structured-output \
  --test llguidance_qwen_tokenizer -- --include-ignored
```

Both tests passed, with zero ignored tests and no compiler warnings in this
run (`artifacts/llguidance-qwen-smoke.log`). This initial qualification predates
the engine session and CLI integration below.
That workspace also passed `rustup run 1.87.0 cargo check --workspace
--all-features --all-targets --locked` on Apple Silicon, including LLGuidance
and Metal (`artifacts/msrv-187-final.log`).
See [AICI and newer research](structured-generation-frontiers.md) for the
separate fast-forward, precompiled-mask and reasoning-quality experiments.

## Cached generation integration

`engine::constraint::JsonConstraintSession` binds a grammar to one exact token
byte table and model logit width. Model padding beyond the tokenizer vocabulary
cannot be selected. EOS is consumed but not appended as JSON bytes. Only an
accepting non-resource stop can complete; truncation never validates as success.
The CLI reads bounded local files, rejects grammar compilation warnings and
external schema references, and independently validates completed output with
[`jsonschema` 0.56.0](https://docs.rs/jsonschema/0.56.0/jsonschema/).
Its default features are disabled and the validator uses `.offline()`; no
schema retrieval is allowed. The dialect is draft 2020-12. Acceptance by these
two implementations is not a claim of support for every JSON Schema keyword.
Format annotations are not a universal semantic-validation guarantee.

```sh
cargo build -p server --release --all-features
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 64 --verify-cache \
  --json-schema fixtures/constraints/record.json
```

The local release run produced `{"status":"ready","count":1}` in 12 tokens.
Boolean and Unicode/escaped-string fixtures also completed and passed independent
validation. Every step passed cached/full-prefix logit parity. A one-token
record budget emitted `constraint.status: incomplete`, `output: null`, and
exit code 1. Receipts use the `artifacts/constrained-serial-` prefix, the
`boolean`, `record` or `unicode` fixture name, and `.json`/`.stderr` suffixes;
the negative case is `artifacts/constrained-truncated.{json,stderr}`.

Serial, quiet, parity-enabled baseline measurements on the M3 Max control host:

| Fixture | Output tokens | Constraint setup ms | Mask/select/consume total ms | Model decode total ms |
|---|---:|---:|---:|---:|
| Boolean | 3 | 388.611 | 0.524 | 20.315 |
| Record | 12 | 389.624 | 2.012 | 103.539 |
| Unicode constant | 14 | 393.021 | 2.198 | 127.566 |

Setup includes local reads, JSON hashing, tokenizer/trie construction, grammar
compilation and independent-validator construction; it is not grammar compile
time alone. Decode excludes prefill and final sampled token (no next-step forward
is needed). Verification is outside timings but can warm execution. These are
single runs, not throughput, quality or comparative speedup evidence. Earlier
`constrained-{boolean,record,unicode}` smoke launches are not used for timings
because their process completion was not serialized explicitly.

This baseline makes reusable tokenizer/compiled-grammar setup the next measured
startup experiment; it does not yet justify speculative parser acceleration.
Keep per-request matcher state separate from any shared immutable artifact.
The CLI intentionally retains raw prompt IDs and has no forced-token skipping,
canonical token healing, HTTP serving or V4.1 generation yet.

### Selected-token log probabilities

`--logprobs` records natural-log probabilities under the original full model
vocabulary, the grammar-renormalized distribution, and the model mass remaining
in the allowed set. Padded model rows participate in the original partition
but never become legal output tokens. Max-shifted FP64 reductions avoid both
exponential overflow and loss of the normalization term at large equal offsets.
Analytic equal-logit, padded-row and `1e30` translation tests cover this boundary.

On the record fixture, the first selected token had raw logprob `-11.908898`,
conditional logprob `-0.737955`, and allowed log-mass `-11.170943`. The difference
illustrates grammar conditioning, not increased answer correctness. All 12
generated IDs matched the no-logprob control, and cached/full-prefix parity
passed at every step. The four-token unconstrained control likewise preserved
its IDs; both quiet stderr captures were empty. Receipts:
`artifacts/logprobs-{plain,constrained}.{json,stderr}`.

The score-enabled record run spent `6.486 ms` in mask/select/consume plus score
collection. This is one instrumented run, not an estimated regression or a
throughput comparison. Scores are opt-in; default selection does not compute
these exponential reductions. See [uncertainty signals](uncertainty.md) for
what additional data is needed before interpreting any score as confidence.

## Provenance

| Source | Pinned identity | Coverage |
|---|---|---|
| [LLGuidance](https://github.com/guidance-ai/llguidance) | tag `v1.8.0`, commit `dbaf504d498b6aeede06ae57adc6f7c2c4848c59` | README, crate manifest, `lib.rs`, `factory.rs`, `constraint.rs`, `api.rs` selected interfaces |
| [XGrammar](https://github.com/mlc-ai/xgrammar) | release tag `v0.2.6` resolves to commit `bc09a30ec10ba30a6c1ab0c79eaeba3ca518d11f`; selected headers inspected at later `main` commit `d02ad2b155a5f0c3eaa711950d154218d982ae7e` | README, CMake build surface, `compiler.h`, `matcher.h`, `tokenizer_info.h` selected interfaces |

The selected XGrammar `main` headers are not an ABI claim about `v0.2.6`.
These pins identify inspected upstream source, not ABI stability, a security
review, compatibility with this workspace's MSRV, or a local performance result.
