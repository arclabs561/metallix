# MLX capabilities and model adapters

Status: proposed interfaces; accepted priorities. The existing Qwen and
DeepSeek implementations remain the immediate delivery work. Broader compute,
multimodal execution, SMC/sampling, training, LoRA, and fine-tuning are explicit
directions. Existing inference completion and profiling lead implementation;
training requirements inform the interfaces now. This document extends the earlier serving-focused product boundary in
[architecture](architecture.md), without claiming these interfaces exist today.

## Problem and current state

Adding a model should not require threading its tensor layout through the CLI,
server, admission policy, and sampler. At the same time, a common decoder
interface cannot describe a vision encoder, audio codec, forecast model, or
iterative denoiser accurately. Fast execution depends on preserving those
model-specific shapes, state lifetimes, precision choices, and device graphs.

Today the workspace pins `mlx-rs 0.32.0`. MLX arrays and graph execution live in
the Qwen and DeepSeek crates. Qwen has the working decoder; DeepSeek has
source-grounded scalar and bounded Metal operators plus a connected reduced
five-block request and head. The artifact CLI exposes mixed scoring/key rotation;
complete device and real-checkpoint execution remain separate gates.
`engine::sampling::sample_categorical` already supplies a stateless
categorical policy. The shared engine now also exposes a bounded SMC primitive
for log-weight normalization, ESS, deterministic systematic resampling, and
absorbing particle state. Model/cache transitions remain adapter-owned; this
does not claim particle-aware Qwen or DeepSeek serving. See [sampling gates](research/sampling-next-gates.md).

Upstream MLX covers arrays, streams, transforms, compilation, serialization,
and distributed operations; availability in upstream Python does not imply
availability or qualification in the pinned Rust binding. Its
[function transforms](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html)
and [compilation](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
are relevant to future compute APIs as well as model inference.

## Options and proposed direction

1. A universal tensor/backend framework would centralize every operation but
   put a new abstraction between MLX and performance-critical graphs. It also
   recreates functionality already provided by MLX and its Rust binding.
2. Independent end-to-end commands per model preserve control but duplicate
   loading, admission, provenance, sampling, and measurement plumbing.
3. **Proposed:** a thin MLX-specific compute surface plus typed, coarse model
   adapters. Shared services depend on declared task capabilities. Graphs,
   kernels, checkpoint codecs, preprocessing, and state stay adapter-owned.

Option 3 best fits the accepted directions. Exact Rust trait signatures and
crate placement remain open until Qwen, DeepSeek, and a concrete non-text
consumer exercise them. The compute API may expose MLX concepts directly;
service and task APIs should not need raw arrays. Generic accelerator
portability is not a prerequisite for using MLX well.

The Rust operation API is the single compute authority: it exposes a closed,
typed set of qualified operations and bounded evaluated outputs. The CLI invokes
those operations by name for diagnostics and task commands; it does not accept
arbitrary graphs, shader names, or executable model code. This leaves graph and
kernel choice with the adapter while giving compute work one testable Rust and
CLI boundary.

## Configuration contract

A versioned model manifest should separate these independently validated parts:

| Part | Contents and owner |
| --- | --- |
| Identity | Publisher/repository, immutable revision, license/notice references, and the identities of the base, adaptation, tokenizer/template, and preprocessing revisions. |
| Artifact layout and codec | Closed container codec and version, index format, exact member-name grammar, and hash map for every required artifact. The loader validates those local bytes before allocation; unknown codecs or versions fail explicitly. |
| Architecture | Discriminated family/version with adapter-owned dimensions and invariants. Unknown variants fail explicitly. |
| Weight encoding | Per-tensor dtype, quantization scheme, block/group shape, scales and sharding. The adapter validates kernel compatibility. |
| Input/output | Named task, declared tokenizer/template revision or media/numeric preprocessing, output shape and units. The control plane applies declared text tokenization/templates; adapters own architecture-specific and media preprocessing. |
| Execution | Device, storage dtype, compute/accumulator/reduction precision, resident/streamed policy, context or shape bounds, logical budgets, and optional validated kernel variant. A qualified execution profile identifies the permitted combination. |
| Capabilities | Task-specific supported/experimental/unavailable states, each tied to a qualification receipt rather than inferred from a model name. |
| Sampling | Explicit policy, RNG seed/stream, proposal and target where applicable, stopping rule and constraints. Deterministic tasks need no sampler. |
| Adaptation/training | Base checkpoint identity, named adapter composition, trainable parameter selection, gradient/optimizer precision, optimizer-state format and checkpoint/export provenance. Inference-only adapters may declare this unavailable. |

Configuration describes supported graphs; it is not an arbitrary graph
interpreter or permission to execute repository-supplied code. Model metadata
alone cannot select an unknown architecture or silently substitute a codec.
Defaults belong to a validated architecture/execution profile. Overrides must
pass the same shape, precision, and memory checks as defaults. Credentials and
machine paths do not belong in published manifests.

## Adapter responsibilities

Discovery and inspection should be cheap and work before weight loading. A
loaded adapter should expose only the task operations it actually implements:
text continuation, embeddings, image/audio encoding, iterative generation, or
numeric prediction. Load, warmup, health, cancellation boundaries, measurements,
and unload can share lifecycle conventions without requiring identical tensor
state. The task dispatcher translates external inputs into typed requests;
it does not dispatch shader names or manipulate model caches. For text tasks,
it applies the adapter-declared tokenizer/template revision under the control
plane's existing ownership; task-specific media preprocessing stays adapter-local.

State is opaque and bound to checkpoint identity, preprocessing, execution
precision, and a request. A decoder cache is not a denoiser trajectory or a
forecast window. Branch/reindex support is an explicit optional capability:
all relevant state must move together or the operation must fail atomically.
The first SMC gate is a real next-logit replay differential after duplicate,
discarded, and EOS ancestry. Only afterward should a driver compose existing
sampling with log weights, ESS, resampling, and a declared terminal rule.

The proposed first non-text consumer is Chronos-2 numeric inference from the
[trending survey](research/hf-trending-landscape.md). This is a bounded consumer
choice, not an implemented or approved runtime architecture: it exercises typed
numeric shapes, covariates/grouping, units, and quantile outputs without
presuming a decoder cache or sampler. A vision encoder, bounded
image-conditioned decoder, or speech model remains a later, separately
qualified multimodal consumer. None needs to wait for a universal adapter SDK.

Training shares parameter naming, storage, graph construction, RNG identity,
and serialization with inference. Keep immutable base parameters distinct from
trainable adapters; record adapter ordering, rank/scaling, merge precision,
and base revision. Optimizer and gradient state have separate residency and
checkpoint lifecycles. A training-capable implementation must demonstrate a
small reference-matched gradient/update, deterministic resume, and inference
agreement before/after adapter export. Inference-only paths must not carry
optimizer allocations. Full training follows stable compute/parameter ownership
and existing-model completion; it is not a reason to freeze the architecture
around immutable inference weights.

## Performance contract

Correctness and performance advance together. Every optimization has a
specific workload, independent numerical/task oracle, baseline binary and
checkpoint identity, warmup policy, repeated measurements, and rollback
criterion. Track load, prefill/encoding, decode/iteration, first visible output,
end-to-end latency, throughput, logical bytes, and observed process/GPU memory
separately. An allocation estimate is not observed residency.

Candidate techniques include compilation and fusion, preserving asynchronous
graph execution, avoiding unnecessary readback, state/prefix sharing, shape
bucketing, quantized kernels, expert residency/prefetch, and speculative or
particle execution. These are hypotheses, not promised gains. Novel methods
must beat a matched baseline at the same correctness/quality target; compare
against current specialized MLX/Metal runtimes before claiming state of the art.
Do not expose experimental tuning flags as stable configuration until their
semantics and measurement benefit are reproducible.

## Implementation sequence and gates

1. Finish existing adapters, keeping the reduced DeepSeek source oracle and
   Qwen client/numerical checks intact. An operator pass is not full-model support.
2. Inventory the pinned Rust binding and add narrowly useful, closed Rust
   operations with CLI commands that invoke the same operations. Record
   shape/dtype/device/evaluation and bounded output/memory receipts; do not
   accept raw graph, shader, or code input. Exercise one operation outside a
   decoder before extracting shared binding mechanics.
3. Introduce a small typed manifest and capability dispatcher using existing
   Qwen/DeepSeek configuration as consumers. Preserve existing command behavior;
   prove invalid architecture/codec/budget combinations fail before allocation.
4. Add the proposed Chronos-2 numeric adapter with a source reference and
   bounded input fixture. Demonstrate that task dispatch does not require
   decoder assumptions. Reconfirm terms, artifacts, and measured resource cost
   before implementation.
5. Qualify model-state branching against independent replay, then measure a
   two/four-particle experiment with explicit target, proposal, EOS semantics,
   ESS, quality, latency, and memory. Keep inference estimates distinct from
   task-quality confidence.
6. Extract only interfaces demonstrated by those consumers. If an abstraction
   requires copying state, forces synchronization, or obscures kernel selection,
   retain the adapter-specific path and revise the interface.

Steps are reversible within this unpublished workspace. Public API and manifest
stability should be declared only after the consumer checks above. Distributed
deployment and automatic execution of arbitrary Hub code are separate future
decisions. Training and adaptation stay in scope, with explicit correctness
and memory gates rather than support implied by an inference adapter.

## Serving

Proposal, extending step 3. Metallix should serve generation, decision and
embedding models from one `mx serve` process, so a client sees one model list
and one set of routes. The principles above still hold: adapters own graphs
and state, the server only routes typed requests, and every advertised
capability names a qualification receipt.

Status: steps 1 through 3 below are delivered. `mx serve --registry
models.json` loads `{"models": [{"id", "kind", "path"}]}` entries of kind
`qwen` (generate and decide), `julia` (decide) or `qwen_embedding` (embed);
`--model PATH` remains a one-entry Qwen shorthand. `/v1/models` lists each
entry with its capabilities and whether it is loaded. `/v1/decisions` returns
the same receipt as `mx decide` or `mx decide-julia` (see
[typed decisions](typed-decisions.md#serving-decisions-over-http)), and
`/v1/embeddings` returns single pooled vectors (see [embeddings](embeddings.md)).

Each model runs in its own child `mx serve` process on a loopback port, and the
front process forwards requests to it unchanged. A model unloaded inside one
process kept its memory (about 1.2 GB for Julia, 3.4 GB for Qwen3-0.6B), so
stopping the child is how memory is returned. Entries are `resident` (started
with the server) or `on_demand` (started on first request) and declare a
measured `memory_mib`; under `--memory-budget-mib`, idle on-demand children are
stopped least recently used first to make room. A child that fails to start or
dies makes only its model unavailable until a later request restarts it.
Capabilities follow the model kind; per-capability receipts in the manifest
remain open.

### Registry

- Declaration: a local serving manifest lists entries, each pointing at a
  model manifest from the configuration contract plus serving fields:
  `id`, `path`, `capabilities`, `residency` (`resident` or `on_demand`) and an
  optional memory estimate. `mx serve --model PATH` stays as the one-entry
  shorthand for today's behavior. Machine paths stay in the local manifest,
  never in published model manifests.
- Identity: `id` is the client-facing name; the receipt behind each
  capability binds it to the immutable revision and artifact hashes.
- Validation: at startup every entry is inspected (manifest, header, tokenizer)
  without loading weights. An unknown architecture, unqualified capability or a
  resident set over budget fails startup, not the first request.
- Residency on one 128 GB Mac: resident entries load at startup; on-demand
  entries load on first use and are evicted least recently used when the
  declared resident budget would be exceeded. Eviction only unloads an idle
  worker. Larger-than-RAM models such as DeepSeek keep their own streamed
  residency inside the adapter; the registry budgets only their resident part.
- Selection: the request's `model` field picks the entry, as `/v1/responses`
  already checks `model_id`. Unknown models return 404 `model_not_found`.

### Capability-typed routes

| Route | Capability | Body and response |
| --- | --- | --- |
| `POST /v1/responses` | `generate` | Unchanged Responses shape. |
| `POST /v1/decisions` | `decide` | The existing `mx decide` request plus `model`; responds with the same receipt the CLI prints. |
| `POST /v1/embeddings` | `embed` | OpenAI-compatible `{model, input}` with `data[].embedding`, plus the adapter's pooling, normalization and dimension in metadata. |
| `GET /v1/models` | none | Every entry with `capabilities` and `loaded` state. |

A route sent to a model without that capability returns 400
`unsupported_capability` naming the model's capabilities. The decision body is
already the de facto typed-decision shape: Julia's source `predict_typed` and
the third-party Decision 2.0 models use the same `state` plus typed `questions`.
Each decision adapter keeps its own option order and normalization (Qwen sorts
choice IDs and applies a temperature; Julia keeps caller order and a plain
softmax) and reports both in its receipt. Ollama's `/v1/systemone` stays a
separate compatibility decision after its semantics are compared.

### Workers and admission

- One worker per loaded model, each owning its adapter state on its own thread,
  fed by today's zero-capacity channel. The `Admission` flag becomes one flag
  per model, so a busy Qwen request does not block a Julia decision. Admission
  must move after the request is read, because the model is named in the body;
  transport limits already bound that read.
- Device sharing: CPU models (the native Julia encoder) run concurrently with
  GPU models. GPU workers share one Metal lock, as the tests already do, until
  measured overlap shows concurrent GPU work is safe and faster.
- Workers expose capabilities, not a common trait over all of them:
  `ChatBackend` stays generation-only, and decision and embedding workers get
  their own narrow traits. A worker panic marks only that model unavailable.

### Embedding candidates

From live Hub API metadata, 2026-10-04 (pinned revisions in sources):

| Model | Arch / dims / pooling | License, access | MLX and reuse |
| --- | --- | --- | --- |
| Qwen3-Embedding-0.6B | Qwen3, 28 layers, 1024 dims, last-token pooling | Apache-2.0, open; 9.4M downloads | Reuses the Qwen3 decoder adapter; `mlx-community` 8-bit and 4-bit ports exist |
| gte-modernbert-base | ModernBERT, 22 layers, 768 dims, CLS pooling | Apache-2.0, open | Same encoder family as the native Julia encoder (width differs) |
| EmbeddingGemma-300m | Gemma 3 text, 24 layers, 768 dims | Gemma terms, gated (manual approval) | `mlx-community` ports exist; gated, so not first |

Proposed first: Qwen3-Embedding-0.6B, because it reuses the qualified Qwen3
path and is open. gte-modernbert-base is the second, encoder-side check.
Neither is qualified; each needs a source-oracle receipt before `embed` is
advertised.

### Decision candidates

| Model | Shape | License | Fit |
| --- | --- | --- | --- |
| Decision-2.0-Kai-0.6B (vLLM Semantic Router) | Qwen3-0.6B-Base backbone plus `decision_head.safetensors`; choice, yes/no and score in one pass | Apache-2.0, open; custom code | Same request shape; reuses the Qwen3 adapter plus a new head. Sizes up to 27B exist. |
| Qwen3-Reranker-0.6B | Qwen3 causal model scored as a yes/no reranker | Apache-2.0, open | A `rerank` capability or a two-option decision; needs its own semantics |
| gte-reranker-modernbert-base | ModernBERT sequence classifier | Apache-2.0, open | Encoder-side reranker close to the Julia encoder |

The 26B MATILDA-jev decision model is also Apache-2.0 but needs about 49 GiB
and custom multimodal code; it is not a first candidate. Custom Hub code is a
reference to port, never code the server executes.

### Sequence and gates

1. Delivered: registry plus `/v1/models` plus `/v1/decisions` for Qwen decide
   and Julia. Gate: decision receipts over HTTP equal the CLI receipts for the same
   request (Julia: the six reference requests); existing `/v1/responses` tests
   pass unchanged; a Julia request completes while a Qwen request is busy.
2. Delivered: on-demand loading and eviction. Gate: measured resident memory stays under
   the declared budget across a load/evict cycle; an evicted model reloads
   with an identical receipt.
3. Delivered: `/v1/embeddings` with Qwen3-Embedding-0.6B. Gate: embeddings match the
   source within a recorded tolerance on fixed inputs, with pooling and
   normalization recorded.
4. Decision-2.0-Kai as the second Qwen-backbone decision model, then further
   generation adapters from the October priorities as they qualify.

Non-goals for this proposal: a universal model trait, request batching across
models, multi-host serving, binding beyond loopback, authentication, running
Hub-supplied code, and matching Ollama's `/v1/systemone` before the comparison
above.

Sources: Hub model API (`/api/models/{id}`, pipeline-tag listings for
`sentence-similarity`, `feature-extraction` and `text-ranking`) and each
repository's `config.json` and `1_Pooling/config.json`:
[Qwen3-Embedding-0.6B@97b0c61](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3),
[gte-modernbert-base@e7f32e3](https://huggingface.co/Alibaba-NLP/gte-modernbert-base/tree/e7f32e3c00f91d699e8c43b53106206bcc72bb22),
[embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m) (gated;
dims from [the MLX port@2f420ef](https://huggingface.co/mlx-community/embeddinggemma-300m-bf16/tree/2f420efb007317a1ef2aa1608f87ef24226e6807)),
[Decision-2.0-Kai-0.6B@cd49ea3](https://huggingface.co/vllm-sr/Decision-2.0-Kai-0.6B/tree/cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764),
[Qwen3-Reranker-0.6B@e61197e](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/tree/e61197ed45024b0ed8a2d74b80b4d909f1255473),
[gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base),
[MATILDA-jev](https://huggingface.co/Maincode/matilda-jev-v1).
