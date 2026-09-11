# Metallix

Metallix qualifies model layouts and builds Metal execution paths for one
Apple-Silicon Mac. Its first target is DeepSeek-V4.1-Flash.

The intended service is OpenAI-compatible local text generation with bounded
concurrency, reusable context, and measured memory behavior.

## Status

The current milestone runs a complete dense Qwen3 decoder through MLX on
Metal, with a reproducible CPU comparison and single-sequence KV reuse. DeepSeek-V4.1
configuration and checkpoint-index inspection are available; V4.1 execution
and HTTP serving remain unfinished.
[V4.1 operator diagnostics](docs/experiments/v41-candidates.md) compare CPU
candidate masks and Metal FP32 index scores with pinned official expressions
on synthetic inputs. They do not qualify the BF16/FP4 execution path.

```sh
cargo run -p server -- inspect-v41 --config /path/to/deepseek-v41-config.json
```

Expected output includes the sparse routing, Engram, and dynamic-FP8/FP4
checkpoint layout required by the V4.1 adapter.

Qwen3 is the small comparison model used to qualify shared server behavior:

```sh
cargo run -p server -- inspect-qwen --config /path/to/qwen3-config.json
```

For Qwen3-0.6B this reports a head dimension of 128 and 114,688 BF16 KV bytes
per token before allocator overhead. It is a sizing preflight, not inference.

Validate a local Qwen3 checkpoint without reading its tensor payloads:

```sh
cargo run -p server -- inspect-qwen-checkpoint --model /path/to/Qwen3-0.6B
```

On Apple Silicon, this optional command proves the pinned Rust MLX binding can
evaluate a GPU graph. It does not load model weights:

```sh
cargo run -p server --features metal -- smoke-qwen-metal
```

Metallix does not currently convert quantizations. V4.1 support begins by
validating the expected FP8/FP4 layout of supplied artifacts; conversion or lower-bit
formats need separate parity and quality gates.

On Apple Silicon, load an inspected Qwen3 safetensors checkpoint as MLX Metal
arrays and force evaluation of its token embedding:

```sh
cargo run -p server --features metal -- load-qwen-metal --model /path/to/Qwen3-0.6B
```

This checks checkpoint-payload compatibility without running the decoder.
The loaded tensors live only for the process lifetime.

The embedding command runs the same fixed raw IDs `[1, 2, 3]` recorded in the
reference fixture and reports the output shape:

```sh
cargo run -p server --features metal -- embed-qwen-metal --model /path/to/Qwen3-0.6B
```

Run one excluded warmup and three measured uncached forwards:

```sh
cargo run -p server --features metal -- forward-qwen-metal \
  --model /path/to/Qwen3-0.6B --input-ids 1,2,3 --repeats 3
```

The command emits JSON with timings and top logits. `--reference PATH` plus
`--reference-manifest JSON` compare every vocabulary logit against a local
float32 reference, checking input IDs and checkpoint/reference hashes first.
Disagreement exits nonzero. Float32 weights are prepared once before
warmup. This diagnostic path accepts at most 512 raw tokens per sequence.
An absent reference is reported as `"parity": null`.

Generate greedy token IDs with contiguous KV reuse:

```sh
cargo run -p server --release --features metal -- generate-qwen-metal \
  --model /path/to/Qwen3-0.6B --input-ids 785,6722,315,9625,374 \
  --max-tokens 12 --verify-cache
```

Those input IDs encode “The capital of France is”. The output begins
`12095,13,576` (“ Paris. The” with the Qwen tokenizer). `--verify-cache`
compares each cached result with a full Metal recomputation outside the timed
regions. This diagnostic uses greedy sampling, one sequence, and at most 512
input plus generated tokens. KV byte counts describe live arrays, excluding
allocator overhead and temporary copies.

For independent CPU parity, install [uv](https://docs.astral.sh/uv/) and run
the four-case suite. The scripts declare pinned Torch/Transformers dependencies
and use an already-downloaded local Qwen3-0.6B checkpoint:

```sh
cargo build -p server --release --features metal
uv run scripts/qwen-parity-suite.py --binary target/release/metallix \
  --model /path/to/Qwen3-0.6B
```

This compares all logits for prompts of 1, 3, 17, and 64 tokens. Temporary
reference files are cleaned up by the harness. See
[the local experiment](docs/experiments/qwen-metal.md) for measured results.
Removing layer-by-layer GPU waits reduced warm cached decode from 20.3 to
8.74 ms per step on an M3 Max (three 32-token runs, 90 measured decode steps).
This is a single-sequence diagnostic, not a serving-throughput comparison.
The experiment page also documents a repeatable local benchmark harness with
checkpoint hashes and per-run statistics.

See [the architecture](docs/architecture.md) for the serving contract and
delivery gates, and [the efficiency requirements](docs/research/efficiency-methods.md)
for the paper-derived implementation checks.

## Build

Build from source with Rust. The default build provides checkpoint inspection;
`--features metal` enables Qwen execution and V4.1 operator checks on Apple
Silicon. It builds native MLX, requiring CMake and a working Xcode Metal toolchain.

```sh
cargo build --workspace
```

## Benchmark a compatible server

The checked-in benchmark client measures TTFT to the first non-empty content
delta, time to the first SSE event, per-request completion time, and aggregate
reported completion throughput from an OpenAI-compatible endpoint. It also
reports two clearly labeled estimates: mean time between content-delta events,
and mean inter-token time inferred from server-reported token usage. It can
compare a reference server with Metallix once Metallix has an endpoint.

```sh
node scripts/benchmark-openai.mjs \
  --url http://127.0.0.1:8010 \
  --model /path/to/Qwen3-0.6B \
  --cache-condition cold-start \
  --warmup 0 \
  --requests 4 \
  --concurrency 4 \
  --max-tokens 64
```

The command prints versioned JSON with a prompt digest (not prompt text),
sampling parameters, warmup count, server-reported usage coverage, and an
explicit cache-condition declaration. `cold-start` requires `--warmup 0`; it
describes the state at the start of the run, not a cache reset between requests.
`warm` and `mixed` are operator declarations because the client cannot inspect
or clear a server cache. Keep the same model revision, cache condition, request
shape, and prompt digest when comparing runs. It does not claim a model
benchmark for the metadata-inspection commands.

Requests have a 120-second deadline, configurable with `--timeout-ms`.
Truncated streams, SSE errors, and invalid token usage fail the run; `[DONE]`
ends measurement even if the server keeps the connection open.

## Development

See the [developer guide](DEVELOPMENT.md) for measurement workflows and the
[research index](docs/research/README.md) for source-to-code provenance.

Checks require Node.js and Ruff on `PATH`, in addition to Rust and uv.

```sh
uv run scripts/check.py
```

On Apple Silicon, include the optional Metal execution path:

```sh
uv run scripts/check.py --metal
```

The runner checks formatting, tests, strict Clippy, rustdoc, and the Python/Node
harness tests. It stops at the first failure and does not download model weights.

## Limitations

There is no HTTP server or V4.1 decoder yet. Qwen's diagnostic forward works
on raw IDs; tokenizer/chat templates, paged KV, continuous batching, and
quantization conversion are unfinished. The HTTP benchmark client measures
an independently running compatible server. V4.1 weight download remains
gated on its own small text-forward parity fixture.

## License

MIT.
