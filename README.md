<p align="center">
  <img src="docs/assets/metallix.png" alt="" width="160" />
</p>
<h1 align="center">metallix</h1>

An LLM serving engine for Apple Silicon, built around Rust and Metal.

`mx serve` exposes models over an HTTP API. Apple Silicon machines in the cloud
are a target deployment, as well as local Macs. Authentication and TLS are out
of scope by design: put a router or reverse proxy in front of the server.

## What works today

- `mx serve`: an HTTP server for the models in a registry file. Each
  model runs in its own child process, so stopping it returns its memory.
  Models start with the server or on first request, and idle on-demand models
  are stopped, least recently used first, to stay within `--memory-budget-mib`.
- Generation through subsets of the Responses (`POST /v1/responses`), Chat
  Completions (`POST /v1/chat/completions`) and Anthropic Messages
  (`POST /v1/messages`) APIs, and from the CLI with `mx gen`. The Responses
  API and `mx gen` support JSON Schema constrained output. Qwen3-0.6B and Qwen3-4B-Instruct-2507 are the
  measured checkpoints; the registry also accepts Gemma 4 dense text
  checkpoints and Qwen3.5-family checkpoints for generation.
- [Typed decisions](docs/typed-decisions.md) (`POST /v1/decisions`, `mx decide`):
  choice, score and Boolean questions scored from option logits with Qwen3, or on
  the CPU with Julia-1. Probabilities are uncalibrated.
- [Embeddings](docs/embeddings.md) (`POST /v1/embeddings`) in three scopes:
  single vectors with Qwen3-Embedding-0.6B, contextual chunk vectors with
  pplx-embed-context-v1-0.6b, and per-token vectors with pplx-embed-v1-late-0.6b.
  Each matches a pinned source reference within a declared tolerance.
- Reranking (`POST /v1/rerank`) with MaxSim over the per-token vectors.
- Scoring with Qwen3 (`POST /v1/score`, and `POST /v1/completions` with
  `echo`): teacher-forced log probabilities of a continuation or a prompt. A
  text continuation is encoded alone and appended to the prompt's tokens
  (`metallix.continuation_tokenization: "separate"`); callers that need
  lm-eval's joint encoding pass token IDs.
- Experimental WAV transcription with Qwen3-ASR
  (`POST /v1/audio/transcriptions`).

DeepSeek-V4.1-Flash is the main target and is not usable for generation yet. The
primary target is its [1017 update](docs/research/README.md#v41-source-identity);
the measurements below are from the earlier revision. A
scalar CPU reference path runs all 40 layers from the real checkpoint, reading
experts and embedding rows on demand, and predicts the source's next token on
two real prompts (3 and 17 tokens). On that CPU path the 17-token prompt takes
about 32 minutes. A Metal request backend (b7d8c91) runs the FP8 attention
linears, the shared and routed experts, Engram's FP8 WKV projection and the
output head inside a request step; on the 17-token prompt it falls back to the
CPU for none of them and predicts the same source token; the rest of each step
(attention core, router, Hyper-Connections, norms) still runs on the CPU. No
whole-request Metal timing is published yet. [Progress](docs/progress.md)
records the evidence.

## Requirements

- Apple Silicon Mac with the Xcode Metal toolchain
- Rust 1.87 or newer, and CMake (MLX is built from source)
- Locally downloaded checkpoints (safetensors plus `tokenizer.json`)
- Optional: the `hf` CLI for `mx fetch`; uv, Node.js and Ruff for the checks

## Build

```sh
cargo build -p metallix-cli --release --all-features
export PATH="$PWD/target/release:$PATH"
```

The root manifest is a virtual workspace. The `metallix-cli` package in
`crates/app/cli` declares both `mx` and the identical `metallix` binary. It is
the default workspace member and defaults to `mx`, so
`cargo run --all-features -- --help` runs `mx`.
`serve` is one CLI subcommand. Without the `metal` feature
only the inspection commands, `fetch`, `generate` (raw token IDs) and the
DeepSeek CPU runners are available.

A `metal` build also writes MLX's compiled kernels, `mlx.metallib`, to
`~/.mlx/lib/<key>/`, or to `$MLX_RS_METAL_PATH` if that is set at build time.
`mx` loads `mlx.metallib` from its own directory if one is there, and
otherwise from that build-time path. To run `mx` on another machine or
account, copy `mlx.metallib` next to the binary.

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

`kind` is one of `qwen`, `gemma4`, `qwen35`, `qwen3_asr`, `julia`,
`qwen_embedding`, `pplx_context` or `pplx_late`. `residency` defaults to `resident`. Under `--memory-budget-mib`
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
capability returns 400 `unsupported_capability`. While a model is busy,
requests wait in its queue (`--queue-depth`, default 8; `--queue-wait-ms`,
default 60000); a full queue returns 503 `server_busy` and an expired wait 503
`queue_timeout`.

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

- The server binds only to loopback. A generating model decodes one request
  at a time unless `--max-num-seqs` is above 1, which enables batched decoding.
- The Responses endpoint is a subset: text input, function calls, SSE and
  sampling (`temperature`, `top_p`, `seed`) are supported; stored responses,
  `previous_response_id` and images are rejected.
- Qwen3 serves in the checkpoint's float dtype (BF16 by default) or from
  affine-quantized checkpoints. There is no quantization conversion, LoRA or
  training workflow.
- Rerank encodes documents one at a time.
- DeepSeek-V4.1-Flash cannot generate text yet (see above). Running models
  larger than memory is a goal, not a demonstrated capability.

## License

MIT. See [LICENSE](LICENSE).
