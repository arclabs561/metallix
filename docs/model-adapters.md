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

Today the workspace pins `mlx-rs 0.25.3`. MLX arrays and graph execution live in
the Qwen and DeepSeek crates. Qwen has the working decoder; DeepSeek has
source-grounded scalar and bounded Metal operators plus a connected reduced
suffix. `engine::sampling::sample_categorical` already supplies a stateless
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
