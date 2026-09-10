# Metallix architecture

## Product boundary

Metallix v1 is a single-Mac, macOS / Apple-Silicon / Metal serving runtime. It
has portable service contracts and model-specific Metal execution plug-ins; it
does not have a generic tensor or compute-backend abstraction. V4.1 Flash is
the first plug-in because its CED, sparse-MoE, Engram, and cache layout are not
a drop-in fit for existing local runtimes.

The target is not a generic HTTP wrapper and not a claim of immediate feature
parity with vLLM or SGLang. The target is their useful serving semantics for
the models Metallix supports: bounded concurrent requests, safe KV reuse,
streaming, cancellation, measurable scheduling, and explicit capability
coverage per model.

## Control plane

The control plane owns request parsing, tokenization and chat templates,
admission, continuous batching, sampling, streaming, cancellation, metrics,
and model lifecycle. It talks to each execution backend through coarse load,
generate, embed, health, capability, metric, and unload operations. It must
not contain architecture-specific tensor code or prescribe tensor, allocator,
KV-page, graph, or kernel traits.

Cache identity includes the model revision, tokenizer and template revisions,
adapter identity, media hashes when applicable, and a trust-domain salt.
Prompt content is not logged by default.

## Execution plug-ins

Each adapter under `crates/models/` owns validated configuration, checkpoint
loading, quantization-manifest validation, prefill and decode graphs, KV and
prefix-cache layout, batch formation, and kernel capabilities. A plug-in
reports exactly which server capabilities it supports: `supported`,
`experimental`, or `unavailable`.

The V4.1 plug-in will own CED execution, CSA2 sparse attention, sparse-MoE
routing, Engram lookup, DSpark-compatible lookahead state, and tiered expert /
Engram weights. It must use an SSD tier on this machine; active parameter
counts do not make the full checkpoint resident.

## Delivery order

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

## Gates

- No full checkpoint download before a text-only reference-parity test passes.
- No serving claim before bounded request, cancellation, and memory-admission
  tests pass.
- No default-model recommendation before 512, 8k, and 32k cold/shared-prefix
  measurements record TTFT, inter-token latency, throughput, cache behavior,
  and memory.
- No generic capability claim: every feature is reported per model plug-in.
