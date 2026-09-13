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
seeded temperature sampling is opt-in, with or without a JSON schema.
Resident mode allows at most `min(model context, 512)` total prompt-plus-generated tokens;
streamed mode allows at most 32 total and separately checks weight/staging
and retained-KV budgets.
There is no checkpoint chat-template pipeline, messages API, V4.1 decoder,
HTTP serving, continuous batching, execution-backed paged KV, quantization
conversion, or tuning workflow yet. Beyond-RAM execution remains a goal,
not a demonstrated capability. V4.1 weight download is gated on its own small
text-forward numerical fixture.

## License

MIT.
