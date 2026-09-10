# Metallix

Metallix qualifies model layouts and builds Metal execution paths for one
Apple-Silicon Mac. Its first target is DeepSeek-V4.1-Flash.

The intended service is OpenAI-compatible local text generation with bounded
concurrency, reusable context, and measured memory behavior.

## Status

The current milestone validates model configuration, checkpoint layout, Qwen
grouped-query KV sizing, and an optional Rust-to-MLX Metal checkpoint load. It
does not run a Qwen decoder or serve requests yet.

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
validating the official FP8/FP4 artifact layout; conversion or lower-bit
formats need separate parity and quality gates.

On Apple Silicon, load an inspected Qwen3 safetensors checkpoint as MLX Metal
arrays and force evaluation of its token embedding:

```sh
cargo run -p server --features metal -- load-qwen-metal --model /path/to/Qwen3-0.6B
```

This is a checkpoint-payload compatibility gate, not text generation. It holds
the loaded tensors only for the process lifetime and the next gate is
fixed-token decoder-logit parity.

See [the architecture](docs/architecture.md) for the serving contract and
delivery gates, and [the efficiency requirements](docs/research/efficiency-methods.md)
for the paper-derived implementation checks.

## Build

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

## Development

```sh
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Limitations

There is no Qwen decoder forward path or HTTP server yet. The optional Metal
commands qualify a 1×1 graph and the Qwen checkpoint payload; neither is model
inference. The repository includes an HTTP benchmark client and a
checkpoint-index parsing microbenchmark, neither of which is an inference
result. V4.1 weight download remains blocked on a small text-forward parity
fixture.

## License

MIT.
