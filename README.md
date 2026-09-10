# Metallix

Metallix is a single-Mac, Apple-Silicon model-serving runtime for models whose
sparse architectures and memory layouts need more than a generic local
inference server. It begins with DeepSeek-V4.1-Flash.

The product target is OpenAI-compatible local text serving that stays
responsive under concurrent agent workloads, safely reuses shared context, and
makes performance and memory behavior observable.

## Status

The initial milestone is a model-contract probe: parse and validate a V4.1
configuration, then derive the execution requirements a Metal backend must
satisfy. It does not load model weights or serve requests yet.

```sh
cargo run -p metallix-server -- inspect-v41 --config /path/to/config.json
```

See [the architecture](docs/architecture.md) for the serving contract and
delivery gates.

## Development

```sh
cargo fmt --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```
