# Metallix

Metallix validates and will serve model architectures that need Metal-specific
execution and memory management on one Apple-Silicon Mac. Its first target is
DeepSeek-V4.1-Flash.

The product target is OpenAI-compatible local text serving that stays
responsive under concurrent agent workloads, safely reuses shared context, and
makes performance and memory behavior observable.

## Status

The current milestone validates model configuration and checkpoint-quantization
contracts. It does not load weights or serve requests yet.

```sh
cargo run -p server -- inspect-v41 --config /path/to/deepseek-v41-config.json
```

Expected output includes the sparse routing, Engram, and dynamic-FP8/FP4
checkpoint layout required by the V4.1 adapter.

Qwen3 is the small comparison model used to qualify shared server behavior:

```sh
cargo run -p server -- inspect-qwen --config /path/to/qwen3-config.json
```

Metallix does not currently convert quantizations. V4.1 support begins by
validating the official FP8/FP4 artifact layout; conversion or lower-bit
formats need separate parity and quality gates.

See [the architecture](docs/architecture.md) for the serving contract and
delivery gates.

## Build

```sh
cargo build --workspace
```

## Development

```sh
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## Limitations

There is no model loader, Metal execution path, HTTP server, or benchmark
result yet. The checked-in contracts and tests exist to make those later stages
measurable and model-specific.

## License

MIT.
