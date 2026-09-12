<p align="center">
  <img src="docs/assets/metallix.png" alt="" width="160" />
</p>
<h1 align="center">metallix</h1>

A local inference engine for Apple Silicon, built around Rust and Metal.

Run Qwen3-0.6B with KV caching, layer-streamed weights, and JSON Schema
constraints. The `mx` CLI exposes token probabilities, timing, and correctness
checks; GPU execution uses MLX.

DeepSeek-V4.1-Flash is the main target. Checkpoint inspection and isolated
operator tests work today; full-model generation is still in development.
Larger-than-memory inference and an OpenAI-compatible server are goals,
not supported features yet.

## Build and try it

From the repository root, with Rust 1.87+, Apple Silicon, CMake, and a working
Xcode Metal toolchain:

```sh
cargo build -p server --release --all-features
target/release/mx --help
```

Both `mx` and `metallix` are native executables with the same CLI; no shell
alias is required. Nothing is installed on your PATH by these commands.

With an already-downloaded Qwen3-0.6B safetensors checkpoint, including its
`config.json` and `tokenizer.json`, replace the model path below. Input is raw
token IDs, not a text prompt or chat template:

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 4
```

Stdout is a JSON diagnostic report with generated token IDs and timings.
For example, the local control run returned these fields (excerpt):

```json
{"generated_ids":[13,358,2776,264],"finish_reason":"length"}
```

To constrain generation to the included JSON Schema:

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 64 \
  --json-schema fixtures/constraints/record.json
```

The local control run produced this value in `constraint.output`:

```json
{"status":"ready","count":1}
```

The object above is not the entire stdout report. Successful constrained runs
report `constraint.status: "validated"`; a token limit reached before grammar
completion reports `incomplete` and exits nonzero.

Both examples use resident weights and greedy selection. See
[the generation guide](DEVELOPMENT.md) for readable previews, token scores,
cache verification, and layer-streamed generation with explicit memory budgets.

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
seeded temperature sampling is opt-in and currently excludes schema constraints.
Resident mode allows at most `min(model context, 512)` total prompt-plus-generated tokens;
streamed mode allows at most 32 total and separately checks weight/staging
and retained-KV budgets.
There is no canonical text-prompt/chat-template pipeline, V4.1 decoder,
HTTP serving, continuous batching, execution-backed paged KV, quantization
conversion, or tuning workflow yet. Beyond-RAM execution remains a goal,
not a demonstrated capability. V4.1 weight download is gated on its own small
text-forward numerical fixture.

## License

MIT.
