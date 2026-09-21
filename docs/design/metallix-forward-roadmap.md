---
status: proposal
scope: Metallix native MLX backends, DeepSeek Codex readiness, extensibility, and performance
review-trigger: after the first resident DeepSeek token loop or any change to model acquisition/licensing assumptions
---

# Metallix forward roadmap

This roadmap turns the current qualified-operator work into a usable native
backend without claiming that the current diagnostics are already a serving
runtime. It is grounded in the existing execution, attention, sublayer, parity,
and training design documents plus the pushed implementation through `8d770f4`.

## Current position

Done and reusable:

- Qwen resident K/V generation, chat/tool boundaries, responses serving,
  profiling receipts, and property coverage.
- DeepSeek source-shaped CPU attention, indexer, HC, Engram, MoE, cache, and
  suffix fixtures.
- Real MLX header/index validation and 6-bit/group-128 attention decoding.
- Real layer-zero HC collapse, full Q preparation, per-head Q normalization,
  full KV projection, KV normalization, and a 64-value KV rotary-tail gate.
- Bulk contiguous affine reads and focused rotary/KV property tests.
- SMC primitive and test-only LoRA micrograph remain bounded, non-serving gates.

Still unwired:

- Real token hidden-state input into the KV/Q path as one resident executor.
- Attention score/cache execution joining the real Q and KV tensors.
- `wo_a`/`wo_b`, HC post, FFN/MoE, logits, tokenizer, stateful decode, and
  OpenAI-compatible serving for DeepSeek.
- A Codex profile backed by Metallix itself. The oMLX profile is only an
  external baseline.
- A resident token benchmark with load/prefill/decode phase separation.

## Dependency order

### Phase 0: freeze contracts and evidence

Inventory the exact local artifact, index hash, source revision, quantization
geometry, and license provenance. Add one machine-readable receipt describing
which gates are operator-only, which use real payloads, and which use synthetic
inputs.

Gate: a fresh clone can reproduce headers, checksums, and focused tests without
network access. Reversible.

### Phase 1: resident tensor loader

Build an adapter-private DeepSeek loader that maps only the tensors needed for
layer zero and one, validates every shape/dtype/packing rule before allocation,
and keeps decoded weights resident. Reuse the bulk affine reader. Do not add a
generic model loader yet.

Parallel lanes: loader memory accounting; malformed-range/property tests;
artifact/license receipt review.

Gate: one process loads the selected tensors once, reports resident bytes and
allocation phases, and passes repeated-load checksum equality. Partially
reversible.

### Phase 2: real hidden-state and KV/Q preparation

Replace the bounded self-row input with the actual layer-zero HC-prepared hidden
state from a token embedding. Apply attention normalization, Q preparation, KV
projection, KV normalization, and trailing rotary tails in one resident call.
Keep the 448-value compressed prefix and 64-value rotary tail explicit.

Gate: source-shaped token-0 and one decode-position intermediates match within
existing BF16 tolerances; no tensor is decoded inside the token loop. Reversible
within the adapter.

### Phase 3: attention and cache join

Wire the real Q/KV outputs into the existing sparse/window attention primitives.
Implement single shared KV-head broadcasting, sink handling, causal/window masks,
compressed/index cache ownership, and request-local state. Preserve the existing
cache identities and failure-atomic behavior.

Gate: prefill plus one decode step matches source attention outputs and selected
indices for a captured token sequence; reset and retry produce identical state.
Partially reversible.

### Phase 4: output projection and block execution

Decode and execute `wo_a`/`wo_b`, inverse rotary ordering, HC post, FFN/MoE, and
final residual publication. Use the existing sublayer composition contracts
rather than creating a second MoE implementation.

Gate: one complete layer and then the reduced multi-layer suffix match exact
source intermediates and logits for the pinned fixture. Partially reversible.

### Phase 5: logits, tokenizer, and resident decode loop

Add the DeepSeek tokenizer adapter, logits projection, greedy/sampling controls,
EOS handling, and a resident stateful decode API. Keep the API private to the
DeepSeek adapter until parity is established.

Gate: fixed prompt produces identical token IDs across repeated runs, reset
requests, and bounded prefill/decode chunkings. Reversible at the API boundary.

### Phase 6: Metallix Codex profile and tool calling

Expose the resident DeepSeek executor through the existing Responses/chat
server contract. Add tool-call framing, cancellation, request limits, and a
local profile that points Codex at Metallix. Validate with the existing Codex
qualifier and a real local profile smoke test.

Gate: Codex can complete a short prompt and one tool call through Metallix with
stable terminal markers and no external oMLX process. Partially reversible.

### Phase 7: performance and optimization

Add resident benchmarks with one excluded warmup and separate load, prefill,
TTFT, decode, and teardown phases. Measure tokens/sec, peak resident memory,
Metal command latency, and cache growth. Optimize only after receipts exist:
weight residency, fused rotary/norm, projection staging, command scheduling,
quantized activation reuse, and cache layout.

Gate: every optimization preserves token IDs/checksums and improves a named
receipt metric on the same artifact and prompt. Reversible per optimization.

### Phase 8: extensibility and non-LLM MLX capabilities

After DeepSeek and Qwen share a proven resident/runtime boundary, define a
small model adapter manifest and capability registry for text, vision, audio,
embedding, diffusion, and training/LoRA tasks. Keep each capability explicit;
do not make a universal tensor graph contract prematurely.

Gate: adding a new model requires only a manifest, adapter, tokenizer/input
contract, and parity fixture, with no changes to the server core. This phase is
partially reversible and requires an architecture decision first.

## Parallel work lanes

- **Correctness:** source captures, independent numerical oracles, property and
  mutation tests, malformed checkpoint tests.
- **Performance:** resident memory accounting, bulk I/O, Metal command timing,
  token receipts, and regression thresholds.
- **Serving:** Responses/tool protocol, cancellation, profile configuration,
  Codex qualification.
- **Model coverage:** Qwen maintenance, DeepSeek suffixes, then one carefully
  chosen non-LLM MLX adapter.
- **Training:** keep LoRA test-only until a real adapter consumer and checkpoint
  format are selected; then add resume/export parity before production APIs.

Each lane owns separate paths and must merge through the parent’s canonical
checks. No lane may claim end-to-end readiness from an operator-only receipt.

## Decision-required forks

1. **Resident executor shape:** private DeepSeek executor first, or immediate
   shared runtime trait. Recommended: private first, because cache semantics and
   quantization are still model-specific.
2. **Input artifact policy:** local-only supplied checkpoints, a managed download
   cache, or a user-provided path. Recommended: user-provided/local cache until
   license and distribution metadata are explicit.
3. **Codex serving contract:** extend the current Responses route or introduce a
   backend-neutral generation service. Recommended: extend the existing route
   with a backend adapter, preserving the public protocol.
4. **Model capability registry:** one unified manifest versus capability-specific
   manifests. Recommended: capability-specific manifests composed by a small
   registry; this keeps non-LLM MLX work from inheriting text-only assumptions.
5. **Training scope:** test-only LoRA micrographs versus production adapter
   loading/training. Recommended: keep the micrograph test-only until a real
   model checkpoint and export/import contract are selected.

Do not start Phase 6 until Phases 2–5 pass their parity gates. Do not start
Phase 8 until the resident executor and capability-registry fork are recorded in
an ADR. Do not report DeepSeek Codex readiness until Metallix itself owns the
resident load, decode loop, tokenizer, logits, and tool protocol.
