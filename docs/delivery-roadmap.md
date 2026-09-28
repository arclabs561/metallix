# Delivery roadmap

Status: active sequence; API choices remain proposals. Scope: existing-model completion, measured performance, useful
local agents, programmable inference, then broader MLX capabilities. Grounded in
[architecture](architecture.md), [current progress](progress.md),
[adapter/config direction](model-adapters.md),
[performance evidence](experiments/chat-performance.md), and
[sampling gates](research/sampling-next-gates.md).

Baseline: `d92c70d`. Review this sequence after each milestone, a failed
feasibility gate, a materially better upstream runtime, or a change to the
pinned MLX binding. This proposal records sequencing; it does not declare new
interfaces stable or turn synthetic parity into model support.

## Current checkpoint

Qwen is the usable adapter: text/chat/tools and typed decisions execute on
local 0.6B and 4B checkpoints. The public-task decision qualification reached
35/72 and 65/72 respectively; this is not an official benchmark score.
Exact schedule verification now accepts an explicit count/duration/window
contract. Plain-prompt 4B retries exhausted all eight cases; using the checkpoint
chat format reached 7/8 on the same observed tasks (schema-only baseline 6/8).
A separately frozen confirmation set reached 3/8 versus 2/8, at 1,606 versus
478 generated tokens. Strict acceptance holds; broad quality remains unqualified.
SMC proposal
correction and steering pass checkpoint mechanics tests; application quality
and serving integration remain separate gates. Julia has source-pinned encoding
and complete published-header checks, plus six real-tokenizer source-parity
vectors. A synthetic CPU head reference now matches the pinned source; native
encoder/head execution remains open.

The stepped-capacity feasibility experiment passed 50 paired whole-logit trace
rows and isolated memory probes, with 18.16% lower 1983-token decode time and
about 1% short-prompt regression. Matched real 2048-token requests for Qwen3-
0.6B and 4B preserved output/token parity, so bounded stepped storage is now
the resident production path. DeepSeek's source gates now cover Engram1,
layer-one, and exact layer-zero token embeddings; native layer-one attention/HC/FFN now feeds
the layer-two-to-logits reduced suffix; Engram1 now feeds the native layer-one
path through the reduced suffix with corruption rejection. Native layer-zero
window-only attention, HC mixing and RMSNorm/MoE FFN now reproduce its residual
into Engram1. Incoming block residual, prior HC pre-mix and HC-coefficient
production remain captured;
remove those boundaries before claiming a complete reduced oracle. Metal
execution and real previous-call shared state follow those gates. Native
Responses tool-result replay passed three JSON and three SSE trials, and three
tool-stream disconnect recoveries passed. See the
[measurement ledger](experiments/chat-performance.md) and
[current progress](progress.md) for the evidence and limits. The metadata-only
[DeepSeek traffic sensitivity](research/host-memory.md) does not yet establish a
feasible checkpoint-serving envelope.

The bounded `mx inspect deepseek artifact <directory>` gate now validates the
DeepSeek config, tokenizer/template, index, referenced shard containment, and
shard headers without loading tensor payloads. It is an artifact-admission
check, not generation or Codex compatibility evidence.

## Position and constraints

Qwen is the usable vertical: resident chat, bounded read tools, experimental
Responses, and qualified 0.6B/4B controls. DeepSeek's native reduced suffix now
connects layer-one attention/HC/FFN through layer two and final logits. Layer-one
initial residual/pre-mix and partial-call layer-three shared score keys still
cross captured boundaries. The new layer-zero bridge removes the captured FFN
output; native window-only attention is also joined, with its input reconstructed
by native HC pre-mix and RMSNorm. Incoming block residual, prior pre-mix and
coefficient synthesis remain open.
Its scalar numerical oracle and bounded
Metal operators are not a complete GPU decoder.

Performance work has located a useful next experiment: cache concatenation
scales with prefix length in standalone graphs. Transpose construction costs
under 0.3% of measured decode, and the earlier argmax replacement failed its
serving comparison. Neither should be reopened without new evidence.

Keep three completion milestones separate:

1. **Complete reduced oracle:** tokens and synthetic encoded parameters reach
   final logits through native state, without captured intermediate operands.
2. **Complete reduced Metal executor:** the same graph executes on device with
   independently checked numerical, routing, cache and lifecycle behavior.
3. **Usable checkpoint support:** actual checkpoint loading, tokenizer/template,
   prefill/decode, memory admission and task/latency gates pass together.

The single-Mac boundary, architecture-specific graphs, and no-full-DeepSeek-
checkpoint-download-before-reference-parity gate remain in force. Training,
multimodal work and model discovery are accepted directions, not reasons to
start a universal backend framework or a cluster scheduler. Symbolic
controllers, mechanistic observation, recursive orchestration, memory and SMC
are broader research lanes over the same model-owned state contract, not
permission to skip the real logits/state gates.

## Programmable inference milestone

Metallix's broader product thesis is a measured runtime for interventions over
model execution: symbolic constraints, activation/probe controls, verifier and
tool feedback, recursive subqueries, hierarchical memory, and probabilistic
search. These are distinct mechanisms with distinct evidence classes. The
[programmatic inference design](design/programmatic-inference.md) and
[frontiers ledger](research/programmatic-inference-frontiers.md) define their
shared boundary and limits.

The first useful vertical should be one end-to-end controller over the working
Qwen path: a declared specification, a transactional intervention or
constraint, an independent verifier, accept/rollback behavior, and a receipt.
Only after that should the same contract be exercised by a real DeepSeek
adapter, recursive context handles, activation steering, or particles.

**Open before expensive checkpoint work:** define the target context, acceptable
first-token latency, minimum useful decode rate and SSD/RAM budget for the
intended workflow. The current measurements do not settle those requirements.
Correctness work can continue while the feasibility lane makes the tradeoffs
concrete; an unspecified performance target cannot establish usable serving.

## Immediate parallel work

Limit active implementation to two lanes. A third agent can review an
independent boundary or update docs. One owner controls builds and device
measurements; competing GPU workloads invalidate performance comparisons.

| Lane | Next deliverable and consumer | Exit gate | Reversibility |
| --- | --- | --- | --- |
| Qwen performance | Profile the adopted stepped resident path across model sizes and repeated request shapes. | Preserve full-logit/token and branch/EOS replay; retain the concatenation receipt as a regression control. | Reversible implementation; no public tuning flag. |
| DeepSeek completion | Remove remaining layer-zero input/coefficient and previous-call state boundaries. Consumer: the complete reduced text oracle. | Preserve exact same-trace FFN/Engram1 handoff, source-derived numerical bounds, routes and retry behavior before Metal composition. | Reversible implementation; source semantics must not be silently changed. |
| Bounded feasibility/review | Estimate checkpoint storage, expert/Engram residency and bytes transferred per token from inspected metadata; compare to measured local I/O and an explicit latency target. | Record assumptions and a feasible envelope, or trigger the architecture's pager/runtime pivot. Do not download the full checkpoint to discover an obvious capacity failure. | Reversible analysis. |

For each boundary, use existing operators and fixtures first. Add a capture only
when a specific missing operand blocks the next join. Once the complete reduced
oracle passes, stop extending fixture infrastructure for its own sake and move
the execution graph onto Metal.

DeepSeek now joins window-only layer-zero attention to its native FFN/Engram
consumer. After removing the upstream input and coefficient boundaries,
execute the reduced graph in token order with a request-local previous-call
shared index publication so layer
three's real previous publication supplies the next partial layer-one call. That
last gate removes the captured shared-key shortcut. Exercise more than the fixed
5/1/1 trace, including multiple compression boundaries and reset/retry sequences.

### Performance adoption contract

The capacity candidate must use the pinned binding's actual semantics. A
functional slice update is not evidence of buffer donation or reduced copying.
Preserve independent parent/child state; do not trade away future SMC correctness
for single-stream speed. Keep inference KV and training activation ownership
distinct.

For the pinned 0.6B FP32 geometry, reserving 2048-token K/V uses 448 MiB versus
28 MiB at 128 tokens in exact-length storage. That 420 MiB short-context cost
is why fixed capacity is a feasibility probe, not an assumed default. If its
update primitive wins, test stepped allocation and every growth boundary
before selecting a shipping representation. Logical allocation is not RSS.

Start with the existing 128/512/1983-token, 64-step workload on the 0.6B control,
then validate the 4B consumer. Freeze precision, tokens, checkpoint, binary,
warmup and memory limits. Use interleaved baseline/candidate runs rather than
two long sequential blocks. Suggested initial decision budget: at least ten
paired runs; seek a repeatable 10% long-context decode improvement, reject a
short-context regression above 5%, and require request-time benefit as well as
phase benefit. These are proposed adoption thresholds, not measured outcomes.
An ambiguous result earns another measurement, not a speedup claim.

Measure observed peak memory and allocation behavior alongside logical KV
bytes. A candidate that exceeds admission, mutates an ancestor, changes the
declared numerical result, or merely moves work outside the timer fails. Keep
the baseline until these gates pass. Novel fusion or cache methods follow the
same rule; compare useful latency/quality against a qualified current runtime
before using a state-of-the-art claim.

## Subsequent milestones

| Milestone | Consumer and dependency | Required result |
| --- | --- | --- |
| Metal DeepSeek execution | Depends on the complete reduced oracle and explicit shared-state semantics. | Device graph matches the independent reference across prefill, decode, compression boundaries, reset and rejected-call retry. Profile before custom kernel work. |
| Checkpoint-backed DeepSeek | Depends on Metal correctness and a viable storage/residency plan. | Validate codecs, scales, sharding and notices before payload allocation; qualify a bounded real single-request path with explicit RAM/SSD traffic and latency receipts. Full-checkpoint acquisition remains gated. |
| Agent/Codex qualification | Can advance on the working Qwen adapter without waiting for DeepSeek. Preserve the existing bounded read-tool baseline. | Multi-step held-out tasks, tool/result replay, cancellation recovery and declared context limits pass. Test real Codex tool requirements in disposable workspaces before advertising a backend profile. Separate model task failure from protocol failure. |
| Typed compute and config | Extract common mechanics after real Qwen/DeepSeek consumers demonstrate them. | One Rust operation API with CLI parity; manifests reject unknown architecture, codec, precision and budget combinations before allocation. Keep graph/state implementation adapter-local. |
| Model-backed sampling | Depends on stable decoder state and measured fork/release costs; numerical fork replay already has a bounded passing gate. | A private two/four-particle experiment declares target, proposal, EOS, log weights, ESS and terminal selection; compare quality/distribution error, latency and physical memory. No particle-serving API without a useful result. |
| Non-text capability | Depends on the small typed task boundary, not a universal adapter SDK. Chronos-2 remains a proposed first consumer, subject to artifact/license/resource checks. | Source-matched numeric inputs/outputs and preprocessing; then select one vision/audio consumer to exercise a different state lifecycle. |
| LoRA and fine-tuning | Prioritized after current inference milestones and stable parameter/serialization ownership. | Tiny reference-matched gradient/update, immutable base plus trainable adapter ownership, deterministic checkpoint resume, and exported-adapter inference agreement. Then measure memory and useful training throughput. |

The long-term API must leave room for trainable parameters now, but inference
must not allocate optimizers or carry speculative training machinery. Sampling
experiments do not require a non-text adapter first; likewise useful read-only
Codex qualification does not depend on completing DeepSeek.
LoRA should not wait for every future model, continuous batching or distributed
serving: revisit its implementation gate when the current inference milestones
pass or the documented DeepSeek feasibility pivot changes that scope.

## Decisions to resolve at their gate

These are design-record stubs, not accepted ADRs or permission to cross an
existing product boundary.

| Decision / governing surface | Options and tradeoff | Recommended next move |
| --- | --- | --- |
| KV storage, Qwen decoder/cache | Exact-length concat is simple; fixed capacity stabilizes shapes but may copy unused storage; stepped capacity limits waste but adds growth transitions. | Stepped growth, failure-parity, and matched 0.6B/4B resident requests passed with exact output/token parity. Keep the bounded stepped path and retain the concatenation measurements as a regression control. |
| DeepSeek partial shared index state, model execution | Reproduce pinned source publication order; or intentionally correct it and qualify against a separately identified reference. | Reproduce the pinned source for the parity baseline. Never silently substitute owner keys. Decide before claiming complete model parity. |
| Native runtime versus upstream integration | Continue the specialized executor; or retain qualification/adapter work and integrate a runtime that demonstrably meets the same target. | Apply the existing architecture pivot using matched evidence, not popularity. |
| MLX binding upgrade | Keep the qualified pinned stack; or upgrade to unlock a demonstrated blocking operation/performance gain. | Avoid combining an upgrade with a cache-layout change. Qualify an upgrade as its own change before depending on new semantics. |
| Adapter/config shape, model and engine boundary | Closed typed task adapters; versus a universal tensor/graph interface. | Follow the existing thin-MLX proposal. Decide exact signatures only with concrete consumers; keep manifest versions explicit. |
| Workspace mutation and broader Codex tools | Retain read-only tools; or introduce scoped execution/write authority with cancellation and isolation. | Keep protocol qualification separate. Decide the execution policy before adding shell/write tools to `mx agent`. |

Do not start production cache replacement until its storage/aliasing contract
is decided. Do not start full-checkpoint DeepSeek serving until reference
parity and storage/latency feasibility pass. Do not publish a generic adapter
API until actual consumers establish the shared interface.

## Research, maintenance and stop rules

- Refresh model discovery when a chosen implementation needs it or a verified
  architecture release changes the plan. The existing 100-page survey is a
  discovery snapshot, not an obligation to implement every trending model.
  Uncensored variants use the same architecture/codec/resource/notice gates;
  their label alone neither establishes compatibility nor creates a new adapter.
- Fold new numerical properties into invariant tests: chunking/reset/retry,
  ancestry isolation, malformed metadata, exact routing and unused-result
  detection. Prefer independent oracles and counterexamples over test counts.
- Consolidate shared test readers only when three active consumers demonstrate
  the same contract. Preserve historical fixture bytes and provenance. Retire
  duplicate runbooks and superseded experiments after their conclusions are
  captured; retain enough raw evidence to reproduce an adoption decision.
- Keep [progress](progress.md) factual and this roadmap prospective. Update
  both when a gate changes; do not append conflicting status paragraphs.
- Each work batch ends with a user-visible capability or an experiment that
  accepts/rejects a named hypothesis, the relevant canonical/source/checkpoint
  checks, docs/notices, and a verified push. A new diagnostic alone must name
  the implementation decision it unblocks.
- If reduced correctness passes but memory/I/O cannot meet the agreed use case,
  revisit the executor/pager focus before building more API compatibility.
  Distributed serving remains outside the single-Mac plan.
