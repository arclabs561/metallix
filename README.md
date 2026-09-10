# Metallix

Metallix qualifies model layouts and builds Metal execution paths for one
Apple-Silicon Mac. Its first target is DeepSeek-V4.1-Flash.

The intended service is OpenAI-compatible local text generation with bounded
concurrency, reusable context, and measured memory behavior.

## Status

The current milestone validates model configuration, checkpoint layout, and
Qwen grouped-query KV sizing. It does not load weights or serve requests yet.

```sh
cargo run -p server -- inspect-v41 --config /path/to/deepseek-v41-config.json
```

Expected output includes the sparse routing, Engram, and dynamic-FP8/FP4
checkpoint layout required by the V4.1 adapter.

Qwen3 is the small comparison model used to qualify shared server behavior:

```sh
cargo run -p server -- inspect-qwen --config /path/to/qwen3-config.json
```

For Qwen3-0.6B this reports a head dimension of 64 and 57,344 BF16 KV bytes
per token before allocator overhead. It is a sizing preflight, not inference.

Metallix does not currently convert quantizations. V4.1 support begins by
validating the official FP8/FP4 artifact layout; conversion or lower-bit
formats need separate parity and quality gates.

See [the architecture](docs/architecture.md) for the serving contract and
delivery gates, and [the efficiency requirements](docs/research/efficiency-methods.md)
for the paper-derived implementation checks.

## Build

```sh
cargo build --workspace
```

## Benchmark a compatible server

The checked-in benchmark client measures streamed TTFT, inter-token latency,
aggregate completion throughput, and per-request completion time from an
OpenAI-compatible endpoint. It can compare a reference server with Metallix
once Metallix has an endpoint.

```sh
node scripts/benchmark-openai.mjs \
  --url http://127.0.0.1:8010 \
  --model /path/to/Qwen3-0.6B \
  --requests 4 \
  --concurrency 4 \
  --max-tokens 64
```

The command prints JSON so results can be kept with the exact workload. It
does not claim a model benchmark for the metadata-inspection commands.

## Development

```sh
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Limitations

There is no model loader, Metal execution path, or HTTP server yet. The
repository includes an HTTP benchmark client and a checkpoint-index parsing
microbenchmark, neither of which is an inference result. V4.1 weight download
remains blocked on a small text-forward parity fixture.

## License

MIT.
