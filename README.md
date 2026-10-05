<p align="center">
  <img src="docs/assets/metallix.png" alt="" width="160" />
</p>
<h1 align="center">metallix</h1>

A programmable local inference engine for Apple Silicon, built around Rust,
Metal, and explicit model-state contracts.

## What works today

- `mx serve`: a loopback HTTP server for the models in a registry file. Each
  model runs in its own child process, so stopping it returns its memory.
  Models start with the server or on first request, and idle on-demand models
  are stopped, least recently used first, to stay within `--memory-budget-mib`.
- Generation with Qwen3-0.6B and Qwen3-4B-Instruct-2507 through a subset of the
  Responses API (`POST /v1/responses`), and from the CLI with `mx gen`,
  including JSON Schema constrained output.
- [Typed decisions](docs/typed-decisions.md) (`POST /v1/decisions`, `mx decide`):
  choice, score and Boolean questions scored from option logits with Qwen3, or on
  the CPU with Julia-1. Probabilities are uncalibrated.
- [Embeddings](docs/embeddings.md) (`POST /v1/embeddings`) in three scopes:
  single vectors with Qwen3-Embedding-0.6B, contextual chunk vectors with
  pplx-embed-context-v1-0.6b, and per-token vectors with pplx-embed-v1-late-0.6b.
  Each matches a pinned source reference within a declared tolerance.
- Reranking (`POST /v1/rerank`) with MaxSim over the per-token vectors.

DeepSeek-V4.1-Flash is the main target and is not usable for generation yet. A
scalar CPU reference path runs all 40 layers from the real checkpoint, reading
experts and embedding rows on demand, and predicts the source's next token on
two real prompts (3 and 17 tokens). It is slow: the 17-token prompt takes
about 32 minutes. Metal versions of individual stages exist but are not yet
wired into requests. [Progress](docs/progress.md) records the evidence.

## Requirements

- Apple Silicon Mac with the Xcode Metal toolchain
- Rust 1.87 or newer, and CMake (MLX is built from source)
- Locally downloaded checkpoints (safetensors plus `tokenizer.json`)
- Optional: the `hf` CLI for `mx fetch`; uv, Node.js and Ruff for the checks

## Build

```sh
cargo build -p server --release --all-features
export PATH="$PWD/target/release:$PATH"
```

This builds `mx` (and the identical `metallix`). Without the `metal` feature
only `inspect`, `fetch` and the reduced DeepSeek runner are available.

## Quick start

Write a registry, `models.json`:

```json
{"models": [
  {"id": "qwen", "kind": "qwen", "path": "/path/to/Qwen3-0.6B"},
  {"id": "embed", "kind": "qwen_embedding", "path": "/path/to/Qwen3-Embedding-0.6B"},
  {"id": "late", "kind": "pplx_late", "path": "/path/to/pplx-embed-v1-late-0.6b",
   "residency": "on_demand", "memory_mib": 3000}
]}
```

`kind` is one of `qwen`, `julia`, `qwen_embedding`, `pplx_context` or
`pplx_late`. `residency` defaults to `resident`. Under `--memory-budget-mib`
every entry must declare `memory_mib`. `mx serve --model PATH` is shorthand for
a one-entry Qwen registry.

```sh
mx serve --registry models.json    # listens on 127.0.0.1:8321
```

```sh
curl -s localhost:8321/v1/models

curl -s localhost:8321/v1/responses -H 'Content-Type: application/json' \
  -d '{"model": "qwen", "input": "Say hello", "max_output_tokens": 32}'

curl -s localhost:8321/v1/decisions -d '{"model": "qwen", "state": {"color": "blue"},
  "questions": {"color": {"type": "choice", "instructions": "Select the color stated in the input.",
  "criteria": {"blue": "Blue", "red": "Red"}}}}'

curl -s localhost:8321/v1/embeddings -d '{"model": "embed",
  "input": "Explain gravity", "input_type": "query", "dimensions": 256}'

curl -s localhost:8321/v1/rerank -d '{"model": "late",
  "query": "What motivates scientific discovery?",
  "documents": ["Scientists explore the universe driven by curiosity.",
                "Children learn through curious exploration."]}'
```

`GET /healthz` reports readiness. A route sent to a model without that
capability returns 400 `unsupported_capability`; a busy model returns 503
`server_busy`.

Without the server (`decision.json` holds the `state` and `questions` from the
decisions request above):

```sh
mx gen --model /path/to/Qwen3-0.6B --prompt "The capital of France is" \
  --max-tokens 8 --preview
mx decide --model /path/to/Qwen3-0.6B --request decision.json
```

## Documentation

- [Command-line examples](docs/cli.md): generation, schemas, sampling,
  DeepSeek artifacts and the reduced runner
- [Embeddings and reranking](docs/embeddings.md) and
  [typed decisions](docs/typed-decisions.md): request and response formats,
  qualification
- [Developer guide](DEVELOPMENT.md): checks, profiling, chat, the read-only
  agent and the Responses endpoint in detail
- [Progress](docs/progress.md), [delivery roadmap](docs/delivery-roadmap.md),
  [model adapters and serving design](docs/model-adapters.md) and
  [architecture](docs/architecture.md)
- [Research notes](docs/research/README.md)

Run the checks with `uv run scripts/check.py` (add `--metal` for the Metal
tests). They do not download model weights.

## Limitations

- The server binds only to loopback. Each model handles one request at a time;
  there is no batching across requests and no request queue.
- The Responses endpoint is a subset: text input, function calls and SSE are
  supported; stored responses, `previous_response_id`, images, nonzero
  temperature, `top_p` other than 1 and seeds are rejected. Output is greedy.
- Qwen context is at most 16,384 tokens (default 2048) with at most 256 output
  tokens per request.
- Served weights are float32. There is no quantization conversion, LoRA or
  training workflow.
- Rerank encodes documents one at a time.
- DeepSeek-V4.1-Flash cannot generate text yet (see above). Running models
  larger than memory is a goal, not a demonstrated capability.

## License

MIT. See [LICENSE](LICENSE).
