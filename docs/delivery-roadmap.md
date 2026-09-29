# Delivery roadmap

Status: active sequence; API choices remain proposals. Scope: existing-model completion, measured performance, useful
local agents, programmable inference, then broader MLX capabilities. Grounded in
[architecture](architecture.md), [current progress](progress.md),
[adapter/config direction](model-adapters.md),
[performance evidence](experiments/chat-performance.md), and
[sampling gates](research/sampling-next-gates.md).

Review baseline: `ebf5ba3`. Review this sequence after each milestone, a failed
feasibility gate, a materially better upstream runtime, or a change to the
pinned MLX binding. This proposal records sequencing; it does not declare new
interfaces stable or turn synthetic parity into model support.

## Current checkpoint

Qwen is the usable adapter: text/chat/tools and typed decisions execute on
local 0.6B and 4B checkpoints. The public-task decision qualification reached
35/72 and 65/72 respectively; this is not an official benchmark score.
The interval-scheduling example has qualified optional verification/retry
mechanics. Keep it as a regression; its small synthetic results do not establish
model quality and do not justify more task-specific features.
SMC proposal correction and steering pass checkpoint mechanics tests;
application quality and serving integration remain separate gates. Julia has
source-pinned encoding, complete published-header checks, six real-tokenizer
source-parity vectors, and a bounded native 22-layer path. Its numerical
qualification currently fails, so encoder prefill, checkpoint execution, typed
decisions, and serving integration remain open.

The stepped-capacity feasibility experiment passed 50 paired whole-logit trace
rows and isolated memory probes, with 18.16% lower 1983-token decode time and
about 1% short-prompt regression. Matched real 2048-token requests for Qwen3-
0.6B and 4B preserved output/token parity, so bounded stepped storage is now
the resident production path. DeepSeek has fixed, test-private native startup
through final-head composition for both the canonical 0/5/6 and alternate
0/4/5/6 schedules. It carries token startup, persistent Engram and L1–L3
state, the prior L3 publication for L1 partial calls, live L1 owner
publications into L2, and native L3/L4 tails to the source-qualified final
head. Failure invalidation and fresh reconstruction remain bounded fixture
controls. The alternate L3/L4 tails and head still run after the upstream loop;
interleaved request execution and alternate full-request recovery remain open.
The final head, HC/FFN block tails and persistent Engram arithmetic/state now
have fixture-independent runtime components. A stateful first-block session
now accepts caller token IDs and weights through its complete window-attention
and HC/FFN path. L1 ratio-two owner publication and L3 candidate projection
are also runtime components. Complete L1–L4 request assembly remains the next
extraction gate.
Downstream weights and source checks remain fixture-backed; broader partitions,
checkpoint loading and Metal execution remain open. Native
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
Responses, and qualified 0.6B/4B controls. DeepSeek's fixed test-private
canonical and alternate startup-to-head compositions are numerical
qualification, not a decoder: they retain fixture-backed weights and source
oracles, while the alternate tail/head runs after the upstream loop. Checkpoint
loading, broader partitions, interleaved request execution, alternate recovery,
and Metal execution remain open. Julia's bounded 22-layer path exists but has
not passed numerical qualification. Its scalar numerical oracle and bounded
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

The Qwen controller mechanism now has accept/reject, rollback and receipt
evidence. Its interval example is not the product milestone. The next user-visible
milestones are complete model execution: native DeepSeek generation and a native
Julia typed-decision path. Preserve Qwen's working text, tools and decision
commands while pursuing those consumers. SMC and steering remain research and
regression surfaces until a concrete application needs them.

**Open before expensive checkpoint work:** define the target context, acceptable
first-token latency, minimum useful decode rate and SSD/RAM budget for the
intended workflow. The current measurements do not settle those requirements.
Correctness work can continue while the feasibility lane makes the tradeoffs
concrete; an unspecified performance target cannot establish usable serving.

## Immediate parallel work

Use two bounded implementation lanes plus one independent reviewer when useful.
One owner integrates and runs Cargo/device checks. Research must resolve a named
implementation uncertainty and produce a pin, contract correction, or executable
gate; another broad model survey is not on the critical path.

| Lane | Next deliverable and consumer | Exit gate | Reversibility |
| --- | --- | --- | --- |
| DeepSeek primary | Promote the reduced composition into an executable runner, carrying request invalidation/reconstruction across alternate partitions before checkpoint loading. The test-private graph now uses unified-bundle numerical operands throughout L0–L4/head, retains live L1–L3 state, and invalidates failed requests before reconstruction. L2 requires live L1 publications; L1 partial decode requires the prior live L3 prefix; L4 reuses committed L3 key/KV while computing its own query and selection. The source-only `4+1+1+1` probe matches logits at common endpoints and final state, while intermediate scratch differs. Native L3 WKV/compressor/key/KV, query/score stages, candidate masks, selected IDs, attention outputs and both partial L3→L1 prefixes now match the alternate source exactly; computed attention also reaches the L4 entry through native L3 HC/FFN under the existing numerical bounds. Retained L3 candidate sets/key/KV and native residual/pre now drive L4 and final logits under the same numerical bounds. The alternate composition now derives L3 input from native L0–L2 and persistent Engram, with preceding computed L3 publications feeding both L1 partial calls. L3/L4 tails and head run after the upstream loop in this test; interleaved request execution and alternate full-request recovery remain open. Owner cancellation/retry/reset controls pass. Consumer: the existing reduced text graph. | Existing numerical policy, exact discrete routing/BF16 boundaries, multiple prefill/decode partitions; preserve publication, reset and retry controls. | Reversible scalar implementation; fixed synthetic fixture shapes, no public decoder API yet. |
| Julia secondary | Resolve the bounded 22-layer prototype's numerical qualification before expanding prefill to ordinary typed requests. Consumer: typed decisions. | Preserve the enabled `1e-5` hidden-state gate and qualified two-block/head controls; then qualify longer padded and window-crossing inputs. Keep the preregistered [independent F64 limits](experiments/julia-accuracy-contract.md) fixed. Same-input replay separates QKV/RoPE/score error. Both F64-dot and balanced-F32 trials failed calibration and were reverted; balanced F32 passing the legacy source test was insufficient. Held-out evaluation and promotion remain closed until calibration and independent-reference controls pass. | Reversible CPU reference before Metal or checkpoint loading. |
| Feasibility and review | Refine DeepSeek residency/selected-byte traffic from existing metadata and source routes in parallel. Consumer: the checkpoint acquisition decision. | Explicit RAM/SSD/context/latency budget; distinguish metadata sensitivity from measured hit rates. | Read-only estimates and bounded local probes. |
| Qwen maintenance | Keep text/tools/typed decisions and optional controller behavior working. | Existing regression checks; new work needs a concrete bug or representative application requirement. | Reversible fixes; no further scheduling benchmark variants. |

### Dependency order and stopping gates

1. **Preserve the qualified numerical boundaries.** DeepSeek coefficient comparisons
   use the existing stage-specific policy, with exact downstream BF16 and routing
   checks. Julia's complete block follows the pinned Transformers RoPE, masking
   and GEGLU behavior. These gates pass; retain their controls during composition.
   Stop and diagnose a new mismatch rather than widening tolerances.
2. **Compose execution.** DeepSeek gets one bounded stateful runner from tokens
   to logits, with native previous-call layer-three publication. Julia gets
   encoder-to-head prefill from its qualified block. A new fixture is
   justified only by a missing operand or independent oracle for these consumers.
3. **Move the qualified graph onto Metal.** Preserve full-output, state, routing
   and failure-atomicity comparisons. This is an execution milestone, not evidence
   that the published checkpoint fits the machine.
4. **Admit real checkpoint work.** Before acquisition or loader expansion, settle
   payload, resident memory, context and latency budgets. DeepSeek's existing
   [traffic sensitivity](research/host-memory.md#deepseek-routed-expert-traffic-sensitivity)
   makes locality a feasibility question now, not after Metal completion. Julia
   needs its own bounded checkpoint plan. Then qualify loading, prefill/decode or
   typed scores, and an ordinary CLI request together.

Do not start full-checkpoint acquisition until its resource plan is decided.
If DeepSeek's feasibility gate rules out useful single-Mac execution under the
chosen limits, explicitly revisit residency/quantization scope; do not silently
start a pager framework. Julia can advance independently within a separately established
resource envelope. Performance optimization follows a measured bottleneck
on the relevant execution path, not another microbenchmark by default.

### Work held at its current boundary

Keep the schedule verifier, SMC accounting, and steering mechanics covered by
existing tests. Resume application-quality work only for a named consumer and
an independent outcome metric. Retain broad architecture research as background;
no generic controller, particle-serving API, hierarchical router, training path,
or universal tensor framework is needed for the next model-execution milestone.
These remain accepted future directions rather than implicit parallel work.

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
| DeepSeek request failure lifecycle | Roll back every owner to the last committed call; or invalidate the entire request after any late failure and reconstruct all state before a fresh request. | Prefer invalidation first. A corrupted previous-L3 prefix can fail after L1 Engram/compressor state has advanced; component-local transactions cannot make that call retryable. The reduced test-private wrapper now rejects continuation after poisoning and rebuilds its L1–L3 state for a fresh request, checked through final-logit source bounds. Carry this contract into the real runner once fixture operands are replaced; production recovery remains unimplemented. |
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
