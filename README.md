<p align="center">
  <img src="docs/assets/metallix.png" alt="" width="160" />
</p>
<h1 align="center">metallix</h1>

A local inference engine for Apple Silicon, built around Rust and Metal.

Generate JSON that follows your schema. Replay a sampled sequence. Inspect
token probabilities and check the KV cache—all from `mx`.

Qwen3-0.6B runs today. DeepSeek-V4.1-Flash is the main target; its
configuration and isolated Metal operators work today, but it cannot generate
yet.

## Setup

Requires Rust 1.87+, Apple Silicon, CMake and the Xcode Metal toolchain. From
the repository root, build once and make the release binary available in this
shell:

```sh
cargo build -p server --release --all-features
export PATH="$PWD/target/release:$PATH"
MODEL=/path/to/Qwen3-0.6B
```

`MODEL` is an already-downloaded Qwen3-0.6B safetensors directory containing
`config.json` and `tokenizer.json`; its checkpoint shards must be locally
accessible. Shards may be regular files or symlinks to regular files. `mx` and
`metallix` are native executables with the same CLI; no shell alias is needed.

## Complete a prompt

`mx gen --prompt` encodes plain text with the local `tokenizer.json` and
returns a JSON receipt on stdout. Its `generated_text` field is decoded model
output; `--preview` writes a bounded rendering of that receipt to stderr. This
is text completion, not a chat interface: the CLI adds no chat template or
special tokens.

```sh
mx gen --model "$MODEL" --prompt "The capital of France is" --max-tokens 8 \
  --verify-cache --preview
```

Qwen runs through `mlx-rs` and MLX on Metal in this control path. The command
checks each cached decode step against a complete uncached forward outside the
reported generation timings.

Excerpt from a local run (`--preview`):

```text
finish_reason: length
output_text:  Paris. The capital of Italy is Rome
cache_checks: 8/8 passed
```

It is one deterministic completion and cache-consistency observation, not a
quality or performance benchmark.

## Generate a JSON record

The schema constrains token selection after the plain prompt is encoded. The
same JSON receipt carries the decoded constrained output in `generated_text`.

```sh
mx gen --model "$MODEL" --prompt "Return only the status." --max-tokens 8 \
  --json-schema-inline '{"type":"string","enum":["ready","waiting"]}' --preview
```

Successful schema output is independently validated. A token budget that ends
before the grammar completes reports `incomplete` and exits nonzero.

Excerpt from a local run:

```json
{"finish_reason":"grammar_complete","generated_text":"\"waiting\"","constraint":{"status":"validated","output":"waiting"}}
```

This demonstrates grammar completion and validation for the shown input, not
general structured-output quality.

## Experimental chat and local control plane

`chat`, `agent`, and `serve` are experimental Metal-only controls for one
locally available Qwen3 checkpoint. They use that checkpoint's
`tokenizer_config.json` chat template. A resident session loads model weights
once, then starts with fresh KV state for every chat turn or HTTP request.
The `--context-tokens` control applies only to these three commands: it accepts 1
through 2048 and defaults to 2048 total prompt-plus-generated tokens.
`--max-tokens` reserves part of that total and accepts 1 through 256. Before
checkpoint payloads load, the control path admits the selected context against
a fixed 512 MiB *logical* f32 KV budget. At 2048 tokens, Qwen3-0.6B's plan is
448 MiB. That estimate excludes weights, activations, scratch, and allocator
headroom; it is not a process-RSS guarantee or an MLX allocation reservation.
Use `--context-tokens 512` to retain the earlier compatibility bound.

Start a one-turn chat that streams text, or omit `--prompt` for the interactive
loop (`/reset` clears history and `/quit` exits):

```sh
mx chat --model "$MODEL" --prompt "Give a two sentence summary of Rust." \
  --max-tokens 96
```

Add `--json` for a complete structured generation receipt instead of streamed
text. Its metrics include `context_tokens` and `planned_kv_bytes`, alongside
load, render, prefill, time-to-first-token, and decode timings. Chat is a
Qwen3 control path, not a general model API or a qualified serving claim.

`agent` may ask the model to read files below one supplied workspace root. Its
only tools are `read_file` (a UTF-8 regular file of at most 32 KiB),
`list_files` (at most 128 entries), and `search_file` (literal text in the same
file bound). Tool paths cannot escape the root, and the command has no write,
shell, network, or process tool. It executes a completed, schema-valid tool
call only; truncated model output runs no tools.

```sh
mx agent --model "$MODEL" --workspace . \
  --prompt 'Use list_files with path "." and summarize the result.' \
  --max-tokens 128 --max-turns 4
```

Add `--json` for an execution receipt with final text, per-turn generation
timings, and ordered executed tool calls. Calls record valid relative paths,
argument hashes, and success or error outcomes; raw tool results are omitted.
A `completed` receipt means the model ended its tool loop, not that it solved
the task. The [qualification runner](DEVELOPMENT.md) checks task evidence
separately.

`serve` keeps one Qwen session resident and exposes a loopback-only,
single-request-at-a-time `/v1/responses` control endpoint, plus `/healthz` and
`/v1/models`:

```sh
mx serve --model "$MODEL" --model-id metallix-qwen3
```

With the server running, send a request from another terminal:

```sh
curl http://127.0.0.1:8321/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"metallix-qwen3","input":"Say hello","max_output_tokens":32}'
```

The Responses surface is deliberately a subset: text input and text parts,
complete input history, function-call round trips, greedy sampling, JSON or
SSE responses, and automatic tool choice are supported. Response storage,
`previous_response_id`, images, nonzero temperature, seeds, top-p changes,
and non-automatic tool choice are rejected. It is not ready to serve as a full
Codex backend. A 2048-token control context is still insufficient evidence for
that role. Accepted connections have a five-second total header/body read
deadline and bounded writes; each connection handles one request. Transfer
encoding, `Expect`, and duplicate body lengths are rejected. Generation has a
separate cooperative 60-second budget (`--generation-timeout-ms`, 1–120000).
Checks surround prefill and each decode, including tokens with no visible text.
An in-flight Metal operation must return before the budget can stop further
work; client disconnects are still detected through failed output writes.

A future user-level Codex profile could target this endpoint only after an
end-to-end compatibility and safety qualification. This is an example of that
future target, **not** a file to add today:

```toml
# ~/.codex/metallix.config.toml — qualification target only
model_provider = "metallix"
model = "metallix-qwen3"

[model_providers.metallix]
name = "Local Metallix"
base_url = "http://127.0.0.1:8321/v1"
wire_api = "responses"
```

Codex custom providers and profiles are user-level configuration, and a
profile is selected with `--profile`; repository-local provider settings are
ignored. See the
[official Codex configuration reference](https://developers.openai.com/docs/config-file/config-reference).

## Sampling and diagnostics

`--input-ids` remains available for deterministic forward, cache, sampling,
and probability diagnostics. It conflicts with `--prompt`.

**Try a different sequence—and replay it.** Turn on sampling, set the
temperature, and keep a seed:

```sh
mx gen --model "$MODEL" --prompt "The capital of France is" --max-tokens 4 \
  --sample --temperature 0.7 --seed 42 --logprobs
```

Replay requires the same model execution and sampling policy, not only the
same seed.

**Inspect probabilities and a terminal preview.** `--logprobs` adds
selected-token natural-log probabilities to stdout; `--preview` writes a
bounded summary to stderr. The probabilities are not answer-confidence scores.

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 4 --logprobs
```

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 4 --preview
```

**Check cached decoding.** Compare each cached step with a full forward. This
adds reference work outside reported generation timings:

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 4 \
  --verify-cache --verbose
```

See [the generation guide](DEVELOPMENT.md) for streamed weights, memory
budgets, profiles, benchmarks, and the full JSON report fields.

## Build features

| Feature | Enables |
|---|---|
| Default | Configuration and checkpoint inspection; no Metal execution |
| `metal` | Qwen execution and V4.1 operator diagnostics on Apple Silicon |
| `structured-output` with `metal` | Qwen JSON Schema constrained generation |

## What is qualified

[V4.1 operator checks](docs/experiments/v41-candidates.md) cover candidate
masks, final selection, index scores, and rotary tails against pinned official
expressions on synthetic inputs. They do not establish full-model or BF16/FP4
execution parity.

The [reduced V4.1 forward checks](docs/research/v41-forward-reference.md)
derive the layer-three attention input through native HC pre-mix and RMSNorm,
then connect native owner KV and producer-selected indices to layer-three
attention, HC, and FFN through the layer-four entry, then continue through
final-layer attention, HC, FFN, final normalization and logits. Fixed
source-derived numerical bounds and wrong-index/omitted-norm controls guard
this suffix.
The focused Engram continuation decodes selected FP8 embedding rows with
row-local E8M0 scales into BF16, projects WKV, and gates the residual before
the same native layer-three-to-logits suffix. Its pre-Engram residual and
incoming HC pre-mix state remain captured from earlier blocks;
full-model generation is still pending.

[Resident chat measurements](docs/experiments/chat-performance.md) cover
repeated CLI/HTTP output agreement at 1983 prompt tokens plus 64 generated
tokens, short requests after long ones, and process-scoped CPU profiling. The
maintained `scripts/qualify-chat.py` runner repeats these checks against an
already-running local server.

[Qwen experiments](docs/experiments/qwen-metal.md) record independent CPU
logit comparisons and measured decode changes.
[Streamed loading checks](docs/experiments/loader-qualification.md) include
teacher-forced cached prefill and appends compared with resident controls.
Streamed generation reuses the same constraints and sampling path. Its logical
budgets are not process-memory ceilings or proof of larger-than-RAM serving.

## Development

The [developer guide](DEVELOPMENT.md) covers profiling, benchmarks and
diagnostic commands. The [research reference](docs/research/README.md) tracks
source versions, reading coverage, implementation status and next tests.
The [architecture](docs/architecture.md) records the serving contract.

Checks additionally require uv, Node.js, and Ruff:

```sh
uv run scripts/check.py
uv run scripts/check.py --metal
```

These run formatting, tests, strict Clippy, rustdoc, and Python/Node harness
checks without downloading model weights. Run them sequentially.

## Limitations

Generation is a single FP32 Qwen sequence. Selection defaults to greedy;
seeded temperature sampling is opt-in, with or without a JSON schema. The
plain `gen` resident diagnostic allows at most `min(model context, 512)` total
prompt-plus-generated tokens; streamed mode allows at most 32 total and
separately checks weight/staging and retained-KV budgets. The experimental
`chat`, `agent`, and `serve` controls have their separate 1–2048 context flag
and fixed logical-KV admission described above.
Experimental Qwen3 chat, a bounded read-only workspace agent, and a loopback
Responses subset exist with the limits above. There is no V4.1 decoder,
continuous batching, execution-backed paged KV, quantization conversion, or
tuning workflow yet. Beyond-RAM execution remains a goal, not a demonstrated
capability. V4.1 weight download is gated on its own small text-forward
numerical fixture.

## License

MIT.
