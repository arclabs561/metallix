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

The native row gate now decodes the real token-0 embedding from shard two:
width `4096`, checksum `9aa497cb4f7c9e00`. This is a bounded file-range read
through Metallix's own header and affine decoder, with no full checkpoint load.

With `--all-features`, the same row gate now evaluates the decoded 4096-wide
embedding on Metal and reports `metal_eval: passed`. This is the first real
DeepSeek tensor to cross Metallix's native device boundary; full block execution
and logits remain the next gate.

The row gate also decodes layer-zero attention `wq_a` row 0 from shard one:
width `3072`, checksum `e965ea9adea61000`, with `metal_eval: passed`. This is a
real attention projection tensor crossing the native device boundary; the full
projection and block graph remain next.

The bounded tensor mode now decodes all 1,024 rows of layer-zero `wq_a`:
`1024 × 3072` logical FP32 values, checksum `f21d39db440e1f00`, with
`metal_eval: passed`. This validates the first complete native attention
projection tensor; the next step is applying it to the decoded hidden state.

The DeepSeek crate now exposes a native Metal matrix projection primitive for
applying decoded row-major affine weights to a hidden-state vector. Its device
test validates a small independent projection; the next integration step is
using it with the real embedding row and full layer-zero `wq_a` matrix.

Attempting to apply the raw 4,096-wide embedding directly to `wq_a` correctly
fails closed: layer-zero `wq_a` consumes a 3,072-wide latent. The missing native
boundary is the preceding latent projection/normalization stage, now identified
by a real artifact shape check rather than a guessed matrix multiply.

The next real layer-zero parameter gate is now covered: `attn_hc.fn` decodes as
an F32 `24 × 16384` matrix and evaluates on Metal, checksum
`6150937c7aee9697`. This anchors the hyper-connection parameter path before its
latent mixing is implemented.

The DeepSeek crate now validates the `hc_mult=4` hidden-state expansion layout,
repeating a 4,096-wide hidden state into the 16,384-wide `attn_hc.fn` input
shape. The next gate is the native HC coefficient mix itself.

The native crate now bridges the HC coefficient formula: RMS normalization of
the four-way expanded hidden state, `attn_hc.fn` products, and the existing
Sinkhorn coefficient splitter. The bridge has independent finite-coefficient
coverage; real parameter/input integration remains the next step.

The real parameter/input HC mix now runs through Metallix: token-0 embedding
from shard two plus layer-zero `attn_hc.fn`, base, and scale from shard one
produce four-copy coefficients with checksum `5fc4d375e7620dac`. Full block
execution remains separate, but the real latent-preparation coefficient stage
is now bound to the checkpoint.

The real HC mix now also performs the pre-collapse step: four-copy coefficients
collapse back to a 4,096-wide hidden stream with checksum
`b5e5dc3e838c352b`. The remaining gap is the model-specific latent conversion
from that stream into the 3,072-wide `wq_a` input.

The apparent 4,096→3,072 blocker was a decoder bug: attention tensors use
6-bit/group-128 packing, while embeddings use 8-bit/group-64. With the corrected
packing, the complete real layer-zero `wq_a` tensor applies to the 4,096-wide
token embedding and evaluates on Metal, producing width `1024` with projection
checksum `a1d46c0641124b5d`. The next native boundary is Q normalization and
`wq_b` projection.

The next attention tensor gate decodes the first `wq_b` head block from the
real shard: `512 × 1024`, checksum `60c20b2f8354c400`, with `metal_eval:
passed`. This matches the 1,024-wide Q-A output contract and bounds the next
Q-normalization/Q-B integration step.

The first native query chain now runs end-to-end for token 0: real embedding →
full `wq_a` → BF16 `q_norm` → first `wq_b` head block. Metal evaluation passes
with Q-B checksum `9caf5173ef2cfb1c`. Remaining attention work is the other Q
heads, rotary/norm details, KV path, and attention output projection.

The native Q chain now covers all 64 Q-B heads: token-0 embedding → full
`wq_a` → BF16 `q_norm` → 32,768 Q-B outputs, with Metal evaluation passing and
checksum `43a0197452b33b03`. The next attention boundary is per-head RMS
normalization, rotary query preparation, and KV construction.
