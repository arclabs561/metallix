<p align="center">
  <img src="docs/assets/metallix.png" alt="" width="160" />
</p>
<h1 align="center">metallix</h1>

A programmable local inference engine for Apple Silicon, built around Rust,
Metal, and explicit model-state contracts.

Generate JSON that follows your schema. Replay a sampled sequence. Inspect
token probabilities and check the KV cache—all from `mx`.

Qwen3-0.6B and Qwen3-4B-Instruct-2507 run today. DeepSeek-V4.1-Flash is the main target; its
metadata and a reduced native forward graph are qualified, but it cannot
generate yet through Metallix's native adapter. A separate machine-local oMLX smoke
route can load a cached quantized DeepSeek V4 snapshot with different model
geometry; it does not qualify the V4.1 target. See the [local
profile ledger](docs/experiments/deepseek-local-profile.md).

Metallix is intended to grow into a programmable inference substrate for
stateful, sparse, hybrid, speculative, and memory-constrained models. That
direction includes symbolic controllers, grammar and schema control,
mechanistic observation and activation steering, verifier and tool feedback,
hierarchical/recursive inference, episodic and semantic memory, continual
adaptation, probabilistic programs, sequential Monte Carlo (SMC), broader MLX
computation, and selected multimodal capabilities. These are research
directions, not claims that the current binaries implement AICI, mechanistic
steering, recursive agents, continual learning, or model-backed SMC. The
[programmatic inference design](docs/design/programmatic-inference.md) records
the boundary and gates.

Training/LoRA remains a separate, later capability. See the
[adapter/config proposal](docs/model-adapters.md) and [model landscape
survey](docs/research/hf-trending-landscape.md) for the broader planning
context.
The [delivery roadmap](docs/delivery-roadmap.md) orders that work and
defines the correctness, performance and resource gates.

## Current capabilities

| Surface | Working scope | Limit |
| --- | --- | --- |
| Qwen text and tools | Schema-constrained generation, chat, local agent, experimental Responses API | Bounded local Qwen3 controls |
| [Typed decisions](docs/typed-decisions.md) | `mx decide`: choice, score, Boolean probabilities; flattened leaf-path labels | Up to 16 options; probabilities are uncalibrated |
| [Verified candidates](docs/candidate-control.md) | Isolated retries with schema, non-overlap, and optional exact task requirements | Requirements must be supplied explicitly |
| [DeepSeek](docs/progress.md) | Bounded artifact-driven token-to-logit requests through all five reduced blocks, persistent shared state, invalidation/restart, and opt-in mixed CPU/Metal execution | Synthetic parameters and fixed qualified schedules; checkpoint loading, tokenizer/template integration, text decoding and native checkpoint generation remain open |
| [SMC](docs/research/sampling-next-gates.md) | Finite accounting, checkpoint-backed proposal correction, resampling and cache tests | Test-only composition, no particle-serving API |
| [Julia-1](docs/research/julia-decision-contract.md) | Tokenizer/header checks, native CPU head and ModernBERT block parity, two-block-to-head composition | Full 22-layer numerical qualification is open; no checkpoint or serving integration |

On the same 72 public decision tasks, local Qwen3-4B-Instruct-2507 scored
65/72 (90.3%), versus 35/72 for Qwen3-0.6B; all tasks produced valid receipts.
This is [local adapter qualification](docs/typed-decisions.md#local-qualification-evidence),
not an official JevBench score. Optional verifier-guided retries have a small
[synthetic interval-scheduling regression](docs/candidate-control.md#fixed-synthetic-task-quality-check);
it tests acceptance and exhaustion, not model quality or inference-request scheduling.
Retries are not required for ordinary model execution.

## Setup

Requires Rust 1.87+, Apple Silicon, CMake and the Xcode Metal toolchain. From
the repository root, build once and make the release binary available in this
shell:

```sh
cargo build -p server --release --all-features
export PATH="$PWD/target/release:$PATH"
MODEL=/path/to/Qwen3-0.6B
```

`MODEL` is an already-downloaded Qwen3-0.6B safetensors directory containing
`config.json` and `tokenizer.json`; its checkpoint shards must be locally
accessible. Shards may be regular files or symlinks to regular files. `mx` and
`metallix` are native executables with the same CLI; no shell alias is needed.

DeepSeek-V4.1-Flash artifact work is currently native metadata and row
qualification. The friendlier grouped commands are:

```sh
mx inspect deepseek artifact /path/to/deepseek
mx inspect deepseek index /path/to/model.safetensors.index.json
mx inspect deepseek shard /path/to/model-00001-of-00018.safetensors
```

They validate the local artifact, index, or shard without loading tensor
payloads. The legacy `inspect-v41-*` spellings remain available. The bounded
`mx inspect-v41-embedding-row <shard> --row 0` command decodes one real
embedding row.
These commands do not claim full DeepSeek generation yet.

`mx` can acquire the ordinary Hugging Face artifact layout through the `hf`
CLI; the per-artifact registry and pinned revisions live in
[`config/artifacts/`](config/artifacts/):

```sh
mx fetch deepseek /path/to/deepseek
mx fetch deepseek /path/to/deepseek --yes
mx fetch deepseek /path/to/deepseek --metadata-only
mx fetch deepseek /path/to/deepseek --dry-run
```

Weights are included by default; `--metadata-only` is the explicit opt-out.
Because the full snapshot is large, noninteractive weight downloads require
`--yes`. `--dry-run` shows the Hugging Face plan without downloading files.
Metallix stores no credentials and introduces no custom container format.

## Complete a prompt

`mx gen --prompt` encodes plain text with the local `tokenizer.json` and
returns a JSON receipt on stdout. Its `generated_text` field is decoded model
output; `--preview` writes a bounded rendering of that receipt to stderr. This
is plain text completion by default. Add `--chat-template` to render the prompt
as one user message using the checkpoint's template, with thinking disabled
and an assistant-generation prefix. The receipt records `input_format` and the
exact selected template's SHA-256. This option requires `--prompt`.

```sh
mx gen --model "$MODEL" --prompt "The capital of France is" --max-tokens 8 \
  --verify-cache --preview
```

Qwen runs through `mlx-rs` and MLX on Metal in this control path. The command
checks each cached decode step against a complete uncached forward outside the
reported generation timings.

Excerpt from a local run (`--preview`):

```text
finish_reason: length
output_text:  Paris. The capital of Italy is Rome
cache_checks: 8/8 passed
```

It is one deterministic completion and cache-consistency observation, not a
quality or performance benchmark.

## Generate a JSON record

The schema constrains token selection after the plain prompt is encoded. The
same JSON receipt carries the decoded constrained output in `generated_text`.

```sh
mx gen --model "$MODEL" --prompt "Return only the status." --max-tokens 8 \
  --json-schema-inline '{"type":"string","enum":["ready","waiting"]}' --preview
```

Successful schema output is independently validated. A token budget that ends
before the grammar completes reports `incomplete` and exits nonzero.
The JSON receipt records this separately as `constraint.verification`, with
`passed` only after independent JSON Schema validation; grammar masking alone
is not reported as semantic verification.

Excerpt from a local run:

```json
{"finish_reason":"grammar_complete","generated_text":"\"waiting\"","constraint":{"status":"validated","output":"waiting"}}
```

This demonstrates grammar completion and validation for the shown input, not
general structured-output quality.

## Verify a schedule against explicit requirements

For exact task acceptance, put integer-tick durations and the allowed time
window in `requirements.json`:

```json
{"durations":[2,2],"window":{"start":0,"end":8}}
```

Save this as `schedule-schema.json`:

```json
{
  "type": "object",
  "properties": {
    "intervals": {
      "type": "array",
      "minItems": 1,
      "maxItems": 128,
      "items": {
        "type": "object",
        "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
        "required": ["start", "end"],
        "additionalProperties": false
      }
    }
  },
  "required": ["intervals"],
  "additionalProperties": false
}
```

```sh
mx gen --model "$MODEL" --prompt 'Return two non-overlapping intervals, each two ticks long, between 0 and 8.' \
  --chat-template \
  --json-schema schedule-schema.json --sample --temperature 1 --seed 41 \
  --verify-schedule --schedule-requirements requirements.json \
  --max-attempts 4 --max-tokens 128
```

The local 0.6B run accepted this schedule on its first attempt:

```json
{"intervals":[{"start":0,"end":2},{"start":2,"end":4}]}
```

The schema describes `{"intervals":[{"start":integer,"end":integer},...]}`.
Acceptance requires non-overlap, the exact duration multiset (and therefore
interval count), and containment in the window. The receipt includes the
validated requirements and their file hash. Requirements are not extracted
from prompt text; omitted requirements leave the non-overlap-only behavior.
Exhaustion returns nonzero and exposes no accepted candidate. See the
[candidate control guide](docs/candidate-control.md) for bounds and evaluation.

## Experimental chat and local control plane

`mx decide --model "$MODEL" --request decision.json` scores typed `choice`,
`score`, and `noul` questions directly from Qwen3 option logits, with zero
generated answer tokens. It uses the same context and KV admission bounds
below, with fresh KV state per question and no generated-token reservation.
See [typed decisions](docs/typed-decisions.md) for the request format,
uncalibrated-probability contract, and public JevBench adapter qualification.

`chat`, `agent`, and `serve` are experimental Metal-only controls for one
locally available Qwen3 checkpoint. They use that checkpoint's
`tokenizer_config.json` chat template. A resident session loads model weights
once, then starts with fresh KV state for every chat turn or HTTP request.
For these three commands, `--context-tokens` accepts 1 through 16,384 and
defaults to 2048 total prompt-plus-generated tokens.
`--kv-budget-mib` sets the corresponding logical f32 K/V admission budget from
1 through 8192 MiB and defaults to 512 MiB. `--max-tokens` reserves part of the
context and accepts 1 through 256. Before checkpoint payloads load, the control
path checks both limits. At 2048 tokens, Qwen3-0.6B's plan is 448 MiB. These
are retained-K/V admission estimates, excluding weights, activations, scratch,
and allocator headroom; they are neither process-RSS guarantees nor MLX
allocation reservations. The expanded ceilings are experimental. A bounded 4B
agent qualification passed 12/12 trials at 2048 tokens and 1024 MiB (checkpoint
revision `cdbee75f17c01a7cc42f958dc650907174af0554`); it does not qualify the
16,384-token or 8192-MiB ceilings, general coding work, or a Codex backend. Use
`--context-tokens 512` to retain the earlier compatibility bound.

Start a one-turn chat that streams text, or omit `--prompt` for the interactive
loop (`/reset` clears history and `/quit` exits):

```sh
mx chat --model "$MODEL" --prompt "Give a two sentence summary of Rust." \
  --max-tokens 96
```

Add `--json` for a complete structured generation receipt instead of streamed
text. Its metrics include `context_tokens` and `planned_kv_bytes`, alongside
load, render, prefill, time-to-first-token, and decode timings. Chat is a
Qwen3 control path, not a general model API or a qualified serving claim.

`agent` may ask the model to read files below one supplied workspace root. Its
only tools are `read_file` (a UTF-8 regular file of at most 32 KiB),
`list_files` (at most 128 entries), and `search_file` (literal text in the same
file bound). Tool paths cannot escape the root, and the command has no write,
shell, network, or process tool. It executes a completed, schema-valid tool
call only; truncated model output runs no tools.

```sh
mx agent --model "$MODEL" --workspace . \
  --prompt 'Use list_files with path "." and summarize the result.' \
  --max-tokens 128 --max-turns 4
```

Add `--json` for an execution receipt with final text, per-turn generation
timings, and ordered executed tool calls. Calls record valid relative paths,
argument hashes, and success or error outcomes; raw tool results are omitted.
A `completed` receipt means the model ended its tool loop, not that it solved
the task. The [qualification runner](DEVELOPMENT.md) checks task evidence
separately.

`serve` keeps one Qwen session resident and exposes a loopback-only,
single-request-at-a-time `/v1/responses` control endpoint, plus `/healthz` and
`/v1/models`:

```sh
mx serve --model "$MODEL" --model-id metallix-qwen3
```

With the server running, send a request from another terminal:

```sh
curl http://127.0.0.1:8321/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"metallix-qwen3","input":"Say hello","max_output_tokens":32}'
```

The Responses surface is deliberately a subset: text input and text parts,
complete input history, function-call round trips, greedy sampling, JSON or
SSE responses, and automatic tool choice are supported. Response storage,
`previous_response_id`, images, nonzero temperature, seeds, top-p changes,
and non-automatic tool choice are rejected. It is not ready to serve as a full
Codex backend. Native 4B passed six ordered two-call replay trials across JSON
and SSE with fresh synthetic tool results; that and the bounded Codex
command-tool check remain insufficient evidence for general coding. Accepted
connections have a five-second total header/body read
deadline and bounded writes; each connection handles one request. Transfer
encoding, `Expect`, and duplicate body lengths are rejected. Generation has a
separate cooperative 60-second budget (`--generation-timeout-ms`, 1–120000).
Checks surround prefill and each decode, including tokens with no visible text.
An in-flight Metal operation must return before the budget can stop further
work. Streaming responses detect client disconnects when an output write fails.
Non-streaming JSON responses do not write during generation, so a disconnected
client can occupy the single-request server until generation completes or the
budget expires. Use streaming when early disconnect detection matters.

The server admits one request at a time with no application queue. While it is
occupied, new connections receive HTTP 503 with `error.code: "server_busy"`,
including health and model-list connections. A failed model worker returns
`model_worker_unavailable`. These are explicit admission failures; the server
does not automatically retry requests. A slow client can still delay the
bounded serial acceptor.

A bounded transport regression fills a non-reading client's socket until the
streaming callback fails, then verifies admission release and a successful
request on the same worker. Its synthetic backend qualifies transport recovery;
physical model-memory release and held-out task quality remain separate gates.

A bounded native Codex command-tool run passed six fresh trials against the
cached 4B checkpoint at `--context-tokens 16384 --kv-budget-mib 8192`:
three single-file reads and three two-file pointer chains. Each chain required
reading a filename, then that file, before returning its exact fresh marker.
All workspaces remained unchanged and every turn completed. The controlled
CLI used fallback model metadata and isolated instructions/features; no user
profile was installed. These read-only tasks do not establish general coding,
complete tool grammar, steady-state performance, or a production-ready Codex
backend.

The corresponding server and optional user-level profile shape are:

```sh
mx serve --model /path/to/Qwen3-4B --model-id metallix-qwen3-4b \
  --listen 127.0.0.1:18321 --context-tokens 16384 --kv-budget-mib 8192
```

```toml
# ~/.codex/config.toml — optional experimental profile

[model_providers.metallix]
name = "Local Metallix"
base_url = "http://127.0.0.1:18321/v1"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false

[profiles.metallix]
model_provider = "metallix"
model = "metallix-qwen3-4b"
model_context_window = 16384
```

This remains a qualification target, not a configuration change. Codex custom
providers and profiles are user-level configuration, and a profile is selected
with `--profile metallix`; repository-local provider settings are ignored. The
controlled test additionally used short model instructions and disabled hooks,
apps, multi-agent features, remote plugins, shell snapshots, and web search.
An optional profile is therefore not equivalent to that test. See the
[official Codex configuration reference](https://developers.openai.com/docs/config-file/config-reference).

## Sampling and diagnostics

`--input-ids` remains available for deterministic forward, cache, sampling,
and probability diagnostics. It conflicts with `--prompt`.

**Try a different sequence—and replay it.** Turn on sampling, set the
temperature, and keep a seed:

```sh
mx gen --model "$MODEL" --prompt "The capital of France is" --max-tokens 4 \
  --sample --temperature 0.7 --seed 42 --logprobs
```

Replay requires the same model execution and sampling policy, not only the
same seed.

**Inspect probabilities and a terminal preview.** `--logprobs` adds
selected-token natural-log probabilities to stdout; `--preview` writes a
bounded summary to stderr. The probabilities are not answer-confidence scores.

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 4 --logprobs
```

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 4 --preview
```

**Check cached decoding.** Compare each cached step with a full forward. This
adds reference work outside reported generation timings:

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 4 \
  --verify-cache --verbose
```

With resident execution, `--verify-cache` also forks the materialized KV
state for each candidate decode and compares child logits with the parent;
the receipt reports this as `branch_verification`. Streamed execution reports
that branch verification is unavailable while retaining its full-forward
cache check. See [the generation guide](DEVELOPMENT.md) for streamed weights,
memory budgets, profiles, benchmarks, and the full JSON report fields.

## Build features

| Feature | Enables |
|---|---|
| Default | Configuration and checkpoint inspection; no Metal execution |
| `metal` | Qwen execution and V4.1 operator diagnostics on Apple Silicon |
| `structured-output` with `metal` | Qwen JSON Schema constrained generation |

## Research direction: programmable inference

The engine is being shaped around a separation between a model-owned executor
and programmable control/orchestration. A controller may eventually observe
qualified internal state, apply a calibrated activation intervention, express
a grammar or tool protocol, call a verifier, recurse over external context,
maintain memory, or run a proposal/particle policy; the adapter still owns KV,
recurrent, routing, expert, and SSD state. This is the intended path for
combining research-grade control with Metal execution without turning every
model into a generic tensor interface.

The current boundary is deliberately conservative:

- JSON Schema masking and seeded categorical sampling are qualified only on
  the bounded Qwen diagnostic.
- The SMC primitive and particle accounting are test-only; they do not provide
  particle-aware Qwen or DeepSeek serving.
- AICI, llguidance, probabilistic-programming, and SMC integrations remain
  research/design work until logits, tokenizer, snapshot/restore, and ancestry
  gates pass on a real adapter.

See [Programmatic and probabilistic inference](docs/design/programmatic-inference.md),
[programmatic inference frontiers](docs/research/programmatic-inference-frontiers.md),
[structured-generation research](docs/research/structured-generation-frontiers.md),
and [sampling gates](docs/research/sampling-next-gates.md).

## What is qualified

[V4.1 operator checks](docs/experiments/v41-candidates.md) cover candidate
masks, final selection, index scores, and rotary tails against pinned official
expressions on synthetic inputs. They do not establish full-model or BF16/FP4
execution parity.

The same report records owner-transaction scaling, CPU profiles, and measured
scalar projection improvements. These are operator measurements, not
full-model throughput results.

The [reduced V4.1 forward checks](docs/research/v41-forward-reference.md)
derive the layer-three attention input through native HC pre-mix and RMSNorm,
then connect native owner KV and producer-selected indices to layer-three
attention, HC, and FFN through the layer-four entry, then continue through
final-layer attention, HC, FFN, final normalization and logits. Fixed
source-derived numerical bounds and wrong-index/omitted-norm controls guard
this suffix.
The focused Engram continuation decodes selected FP8 embedding rows with
row-local E8M0 scales into BF16, projects WKV, and gates the residual before
the same native layer-three-to-logits suffix. Its isolated diagnostic retains
captured pre-Engram residual and incoming HC pre-mix state; the joined layer-two
test below supplies both natively. Full-model generation is still pending.

Earlier, test-only layer-one owner replay executes the ratio-two compressor's
FP32 WKV/wgate projections within a fixed IEEE dot-product envelope, then
checks the BF16 latent and its owned compressed-KV and index-key publications,
native index query and score stages, and causal selected IDs at captured starts
0, 5, and 6. The pinned source's partial decode still scores against a captured
layer-three shared score-key prefix, distinct from that unchanged layer-one
owner prefix. A six-check source fixture pins layer-one attention
operands and outputs at SHA-256
`a13bb6cd53406f04e436f119aa8dec2184ca43a4d4f4969205bff8bdf26ac31b`.
Its three focused Rust checks run `LayerAttentionState` from the native ratio-two
owner KV/IDs at all three calls, compare diagnostics and final output, reject a
non-owner publication, and show that a legal wrong index changes output. The
partial score operand is derived from a strict prior layer-three candidate
capture, rather than relabelled as layer-one score state.

A source-only layer-two attention fixture proves that layer two borrows the
current layer-one KV/index publication and retains its exact historical FFN
handoff. A standalone native layer-two attention check consumes the native
layer-one KV/IDs, rejects a non-owner publication, and shows that a legal but
wrong layer-one ID changes the result.

A source-only layer-one tail fixture, SHA-256
`5f31036c71b797e7195a6b93cf2d656b8744b1e6a80a04dc68007d26e326b89b`,
pins the attention-to-FFN tail and exact layer-two residual/pre-mix handoff.
Its six source checks include layer identity and restoration after an injected
failure. A focused `forward_moe` test now joins native layer-one attention, HC,
and FFN into native layer two, then continues through Engram3 and the native
layer-three/four suffix to final logits. BF16 boundaries remain exact and HC
coefficients use fixed analytic bounds; discarded attention fails before FFN.
The captured layer-one initial residual/pre-mix and the partial-call layer-three
shared score keys remain boundaries. This is not a production API, whole graph,
Metal path, or checkpoint execution. The layer-zero bridge now joins native
window-only attention, HC mixing and RMSNorm/MoE FFN, passing the exact BF16
residual to native Engram1 at starts 0, 5 and 6. Native HC pre-mix and RMSNorm
now also reconstruct the attention input exactly. Token embedding, HC-copy
expansion and identity pre-mix reconstruct the incoming block state from the
same trace's token IDs and weights. HC coefficients now come from native
projection: F32 values satisfy the existing analytic bounds, and their native
layer-one consumer reproduces the exact BF16 attention input. The reduced graph
now retains layer-one, layer-two, Engram3 and layer-three state through starts
0, 5 and 6. Layer two consumes the live layer-one KV/index publication;
bootstrap and final continuation reuse prior outputs without replaying them.
The request rejects continuation after failure and requires reconstruction.
Layer four directly consumes committed layer-three key/KV prefixes while computing
its own query and selection. The unified reduced-runner fixture supplies all
L0–L4 and head numerical operands in this composition. The partial L1 call
requires the preceding live L3 publication. Checkpoint loading and production
recovery remain gates before a production decoder. A separate [source partition probe](docs/research/v41-forward-reference.md#partition-experiment)
matched common-endpoint logits and final cache state for `4 + 1 + 1 + 1`, while
retaining intermediate scratch differences. An observer-controlled capture pins
that alternate schedule's operands independently of the canonical capture.

The alternate native composition now runs token startup, L0, persistent L1
Engram/owner/attention, L2, and persistent L3 Engram/owner/attention. Partial L1
calls at starts four and six consume the preceding computed L3 publication;
completed L1 groups publish their own keys. Native L1 publications drive L2,
and computed L2 residuals and coefficients drive L3. The resulting L3 residuals,
attention outputs and key/KV/candidate publications feed the L3/L4 tails and
head, matching final logits under the existing source-derived bounds. BF16
boundaries and discrete routing remain exact. Invalid Engram streams and L3
owner inputs are rejected before their respective state advances.

This qualifies the fixed reduced `4 + 1 + 1 + 1` composition alongside the
canonical `5 + 1 + 1` path. `RequestSession` now executes all five blocks and
the final head within each call, using ordinary typed numerical operands.
Both schedules match the existing source-derived final-logit bounds and replay
identically after whole-request restart. The bounded artifact CLI now accepts
supplied numerical artifacts and token IDs. The Rust request model can explicitly
select bounded BF16 Metal scoring across L1, L3 and L4 and Metal key rotation
inside the L1/L3 owners. Each option and their combination match
scalar scores, selections, publications and final outputs, pass the source-derived
bounds, and replay after whole-request restart. This is mixed execution: other
arithmetic remains on CPU. The artifact CLI exposes this choice explicitly and
keeps scalar execution as its default.
Broader schedules, complete Metal execution, checkpoint loading and serving remain open.
Runtime components under `deepseek::reduced` now own
final-head arithmetic, attention HC/FFN block tails, and persistent Engram hashing,
embedding, projection and gating. They accept supplied operands without fixture
readers or expected outputs. Engram publishes history only after a successful
step. `StartupSession` now executes caller token IDs through embedding, HC/norm,
window attention and the first block tail, retaining its window across calls.
Its late failures require reset. L1's ratio-two compressed owner and L3's
candidate projection also run from typed operands; partial groups retain their
state, and candidate masks derive from live scores. `LayerThreeSession` joins
ratio-one compression, candidate scoring/selection and attention under one
failure/reset lifecycle. It derives publication identity and window offsets
from live state and returns owned publications for downstream layers.
`LayerOneSession` owns ratio-two compression, direct score selection and attention;
partial calls require the preceding L3 publication while retaining L1 KV.
`LayerFourSession` uses L3's actual committed candidate mask, keys and KV;
it validates candidate geometry before computing its own query and attention.
`RequestSession` owns the complete ordering and invalidates any admitted failed
call. Restart reconstructs every owner from immutable operands. L0 retains its
separate rotary table; L1–L4 use their qualified shared table. Fixture decoding
and source comparisons remain test-only.

Run the reduced scalar model without Metal or checkpoint downloads:

```sh
python3 scripts/export_v41_reduced_artifact.py \
  --source fixtures/deepseek-v41/reduced-runner-reference.json \
  --output /tmp/metallix-reduced-artifact.json
cargo run -p server --bin mx -- run-deepseek-reduced \
  --artifact /tmp/metallix-reduced-artifact.json \
  --input-ids 0,1,2,3,4,5,6 --prefill-tokens 5
```

On Apple Silicon, select Metal index scoring explicitly:

```sh
cargo run -p server --features metal --bin mx -- run-deepseek-reduced \
  --artifact /tmp/metallix-reduced-artifact.json \
  --input-ids 0,1,2,3,4,5,6 --prefill-tokens 5 \
  --score-execution metal-bf16
```

This mode reports `backend: "mixed-cpu-metal"` and
`score_execution: "metal-bf16"`; other arithmetic remains on CPU. The default
scalar receipt keeps its existing format. Builds without the `metal` feature
reject the Metal options. Metal scoring evaluates its four BF16 stage views
together, then reads and validates each intermediate in source order.
Add `--key-rotary-execution metal-fp32` to run index-key
rotation on Metal as well, or use it independently with scalar scoring. The
receipt adds `key_rotary_execution: "metal-fp32"`. Key projection, normalization,
FP4 staging and other request arithmetic remain scalar. Both schedules retain
exact key stages, publications and final outputs; this is not a throughput claim.

`--head-execution metal-fp32` independently moves the final vocabulary projection
to Metal and adds `head_execution: "metal-fp32"` to the mixed-backend receipt.
HC collapse and normalization keep their BF16 staging. Head logits satisfy the
existing source-derived error bounds; scalar bit identity is not required.
This path returns logits to the host and does not establish a resident GPU graph
or a speed improvement.

The exporter creates a new file and refuses to overwrite an existing one. Its
artifact contains configuration and checksummed numerical tensors; captured
outputs and source acceptance bounds stay in tests. The CLI prints a JSON
receipt with the artifact SHA-256 and final-token logit bits for each call.
Use `--prefill-tokens 4` for the alternate qualified schedule. The expanded
[first-chunk sweep](docs/research/v41-forward-reference.md#expanded-first-chunk-sweep)
checks terminal stability for sizes 2, 3 and 6; size 7 rejects an ambiguous L4
selection tie. Those additional sizes do not yet have full per-call qualification. This is bounded
synthetic execution, with no sampling or text decoding. Admission rejects
unknown fields/tensors, invalid encodings and oversized artifacts; the encoded
file limit is 64 MiB and the decoded tensor limit is 32 MiB.

[Resident chat measurements](docs/experiments/chat-performance.md) cover
repeated CLI/HTTP output agreement at 1983 prompt tokens plus 64 generated
tokens, short requests after long ones, and process-scoped CPU profiling. The
maintained `scripts/qualify-chat.py` runner repeats these checks against an
already-running local server. A stepped-capacity resident cache now backs the
production chat executor. It matched
whole-logit traces in 50 paired rows, reduced 1983-token decode time by 18.16%,
and regressed short prompts by about 1%; 128-token prompts plus 64 decode steps retained 112 MiB of
logical KV rather than fixed capacity's 448 MiB. Matched real 2048-token
requests preserved output hashes and 64 generated token IDs for both Qwen3-0.6B
and Qwen3-4B; the 4B decode median fell from 3796.7 to 3600.3 ms across three
fresh CLI processes. These are local M3 Max measurements. Separate 4B
Responses tool-result replay passed three JSON and three SSE trials; its stricter
qualifier validates model, output-item, and content identities, and three live
tool-stream disconnect recoveries also passed. See the measurement ledger for
memory results and qualification limits.

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
seeded temperature sampling is opt-in, with or without a JSON schema. The
plain `gen` resident diagnostic allows at most `min(model context, 512)` total
prompt-plus-generated tokens; streamed mode allows at most 32 total and
separately checks weight/staging and retained-KV budgets. The experimental
`chat`, `agent`, and `serve` controls accept 1–16,384 total context tokens
(default 2048), subject to the configurable logical-KV admission described above.
Experimental Qwen3 chat, a bounded read-only workspace agent, and a loopback
Responses subset exist with the limits above. There is no V4.1 decoder,
continuous batching, execution-backed paged KV, quantization conversion, or
tuning workflow yet. Beyond-RAM execution remains a goal, not a demonstrated
capability. Full V4.1 checkpoint acquisition requires both native reference
parity and an explicit storage, memory, context and latency budget.

## License

MIT.
