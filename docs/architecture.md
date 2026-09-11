# Metallix architecture

## Product boundary

Metallix v1 targets a single-Mac, macOS / Apple-Silicon / Metal serving runtime.
It is designed around portable service contracts and model-specific Metal
execution plug-ins, without a generic tensor or compute-backend abstraction.
V4.1 Flash is the primary target because its CED, sparse-MoE, Engram, and cache layout are not
a drop-in fit for existing local runtimes. Qwen is the first running
qualification adapter; V4.1 remains the primary target.

The target is not a generic HTTP wrapper and not a claim of immediate feature
parity with vLLM or SGLang. The target is their useful serving semantics for
the models Metallix supports: bounded concurrent requests, safe KV reuse,
streaming, cancellation, measurable scheduling, and explicit capability
coverage per model.

## Planned control plane

The control plane owns request parsing, tokenization and chat templates,
admission, continuous batching, sampling, streaming, cancellation, metrics,
and model lifecycle. It owns logical KV allocation, reference counts,
prefix-hash identity, eviction policy, and queue fairness. It talks to each
execution backend through coarse load, generate, embed, health, capability,
metric, and unload operations. It must not contain architecture-specific tensor
code or prescribe tensor, allocator, graph, or kernel traits.

Cache identity includes the model revision, tokenizer and template revisions,
adapter identity, media hashes when applicable, and a trust-domain salt.
Prompt content is not logged by default.

Model lifecycle is explicit: `unloaded`, `loading`, `warming`, `ready`, or
`failed`. Only `ready` admits inference. Readiness is therefore not inferred
from process startup, a listening port, or model discovery; it includes the
adapter's required plan compilation and warmup.

## Planned execution plug-ins

Each adapter under `crates/models/` owns validated configuration, checkpoint
loading, quantization-manifest validation, prefill and decode graphs, the
physical KV page representation, and kernel capabilities. The control plane
asks the adapter for a capacity estimate and page handle; it never assumes a
uniform KV tensor. An adapter reports exactly which server capabilities it
supports: `supported`, `experimental`, or `unavailable`.

The V4.1 plug-in will own CED execution, CSA2 sparse attention, sparse-MoE
routing, Engram lookup, DSpark-compatible lookahead state, and tiered expert /
Engram weights. It must use an SSD tier on this machine; active parameter
counts do not make the full checkpoint resident.

## Delivery order

Qwen3-0.6B is the resident control model used to qualify MLX loading, numerical
forward agreement, and single-sequence cached-decode agreement before attempting
V4.1's larger sparse execution path. Passing Qwen tests does not qualify CED,
CSA2, Engram, or V4.1 quantization. The current Qwen diagnostic uses float32 weights and raw
token IDs; its prompt limit is deliberately smaller than the model context.

1. Parse and validate V4.1 configuration; establish reference-parity tests
   for a small text-only forward pass.
2. Build the V4.1 loader and a correct single-request Metal decode path with
   explicit memory accounting.
3. Add paged KV, chunked prefill, continuous batching, fair admission, and
   automatic prefix caching.
4. Expose health, model discovery, and OpenAI chat/completions with SSE,
   cancellation, limits, and Prometheus metrics.
5. Add constrained output and tool-call response shaping; calls are returned,
   never executed.
6. Measure and then add MTP / speculative decoding, kernel fusion, and
   tiered expert-pager policy.
7. Prove a second text architecture before promising a general model runtime.

Vision, LoRA, embeddings, Anthropic/Responses compatibility, and distributed
serving are feature-gated follow-ons. Metallix v1 is intentionally one Mac:
no remote KV transport, replicas, pipeline parallelism, tensor parallelism, or
expert parallelism. Cluster cache tiers and prefill/decode disaggregation are
not single-Mac v1 work.

## Pivot conditions

Metallix is not committed to rebuilding commodity serving work. vLLM-Metal,
SGLang's MLX backend, llama.cpp, and oMLX are behavioral and performance
oracles throughout development.

| Evidence observed | Pivot | Work retained |
| --- | --- | --- |
| V4.1 executes correctly and meets the single-Mac memory/latency target in an existing runtime | Stop building a standalone serving plane; contribute or maintain a narrow V4.1 backend and qualification suite there. | V4.1 config parser, parity fixtures, quant manifest, benchmark corpus. |
| V4.1 parity passes but SSD-tiered expert/Engram traffic makes interactive decode unacceptable | Make expert residency, prefetch, and byte-per-token admission the project focus; do not add API compatibility features. | Engine contracts, Metal graph, trace and benchmark harness. |
| Metallix serves V4.1 but a second architecture requires unrelated scheduler or cache semantics | Keep it model-specific. Do not generalize the executor from one adapter. | Stable HTTP/request contracts and the first adapter. |
| Three independently-used adapters need identical service behavior | Extract only the proven service seam into a backend-neutral interface. | Existing adapter APIs become conformance tests. |
| A concrete second accelerator and model have a measured acceptance case | Add a separate backend crate with its own executor and page representation; keep Metal internals private. | Control-plane protocol, capability matrix, benchmark schema. |

The immediate competing hypothesis is that upstream Apple support catches up
before a native runtime is useful. The decision gate is practical, not
ideological: compare V4.1 parity, 512/8k/32k TTFT and inter-token latency,
shared-prefix behavior, peak resident memory, and SSD bytes per generated
token against the best available upstream path before expanding Metallix.

## Gates

- No full V4.1 checkpoint download before a V4.1 text-only reference-parity
  test passes. The small resident Qwen comparison checkpoint is used to
  establish and exercise the numerical test harness.
- No serving claim before bounded request, cancellation, and memory-admission
  tests pass.
- No default-model recommendation before 512, 8k, and 32k cold/shared-prefix
  measurements record TTFT, inter-token latency, throughput, cache behavior,
  and memory.
- No generic capability claim: every feature is reported per model plug-in.
