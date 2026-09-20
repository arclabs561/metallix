# Local DeepSeek-V4.1-Flash profile

A local OpenAI-compatible DeepSeek route is validated outside Metallix's native
Rust adapter using oMLX 0.7.0.dev4 and the cached
`mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed` snapshot. The snapshot is
MIT-licensed, 92.8 GB on disk, and requires roughly 80–90 GB of unified memory.

The machine-local Codex profile is `/Users/arc/.codex/deepseek-v41-local.config.toml`:

```toml
model = "deepseek-v41-flash"
model_provider = "deepseek_v41_local"
model_reasoning_effort = "low"

[model_providers.deepseek_v41_local]
name = "DeepSeek V4.1 Flash local oMLX"
base_url = "http://127.0.0.1:18000/v1"
wire_api = "responses"
```

Start the local server with the cached snapshot exposed as one model directory:

```sh
mkdir -p /tmp/metallix-omlx-models
ln -sfn "$DEEPSEEK_SNAPSHOT" /tmp/metallix-omlx-models/deepseek-v41-flash
uv run --no-project --with 'omlx @ git+https://github.com/jundot/omlx.git' \
  omlx serve --model-dir /tmp/metallix-omlx-models \
  --host 127.0.0.1 --port 18000 --memory-guard-gb 110
```

The direct local smoke gate passed: the model loaded in 22.5 seconds and
returned `Ready.` from a 16-token greedy generation. The OpenAI-compatible
`/v1/models`, `/v1/chat/completions`, and `/v1/responses` routes respond for
`deepseek-v41-flash`.

Metallix now accepts this snapshot's standard MLX weight-map index in its
native `inspect-v41-index` command. The real artifact reports 2,757 tensors
across 18 shards. This is the first native artifact gate; it deliberately does
not claim tensor execution yet.

The current server logs report 1.5 generated tokens/second for the short 16-token
request after load. The native DSA indexer extension is not built, so oMLX falls
back to MLX for index scoring and warns that long-context prefill is several
times slower. A Codex smoke with the full local system prompt reached the server
but timed out during a roughly 50k-token prefill. This profile is therefore a
working local model route and short-prompt smoke path, not yet a successful
full-session Codex backend or a super-fast profile.

Metallix's native DeepSeek adapter remains separately gated on checkpoint-backed
Rust loading, full stateful execution, Metal parity, and end-to-end Codex
qualification. The oMLX profile gives us a usable external baseline while those
native gates continue.

The native header gate also accepts an individual shard. On the first local
shard it validated a 3,956,262,156-byte file and 1,625 tensor headers without
reading payload bytes:

```sh
mx inspect-v41-shard /path/to/model-00001-of-00018.safetensors
```

The DeepSeek crate now contains a bounded CPU decoder for the local embedding
layout: packed 8-bit codes with BF16 scale/bias groups of 64. Its tests cover
packing, shape rejection, and finite decoded values. It is a tensor primitive,
not yet a full checkpoint loader or model executor.
