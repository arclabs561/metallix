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
JevBench-shaped requests already have a qualified Qwen direct-option-scoring
bridge. That is benchmark-interface support through a decoder adapter, not
direct Julia checkpoint support. A Julia result must preserve its source
encoding, type embedding, marker-position gather, and decision head.

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
controls. The library `RequestSession` now interleaves all five blocks and the
head per call; both schedules pass final-logit source bounds and complete replay
after request restart.
The final head, HC/FFN block tails and persistent Engram arithmetic/state now
have fixture-independent runtime components. A stateful first-block session
now accepts caller token IDs and weights through its complete window-attention
and HC/FFN path. L1 ratio-two owner publication and L3 candidate projection
are also runtime components. `LayerThreeSession` joins its owner, selection
and attention with coordinated invalidation/reset. `LayerOneSession` now does
the same for ratio-two ownership and direct selection, including the preceding
L3 score-prefix rule. `LayerFourSession` consumes the actual L3 candidate mask;
`RequestSession` composes all layers with whole-request invalidation and restart.
The bounded synthetic-weight/token-input CLI now exports and reads one numerical
artifact, passes existing final-logit source bounds for both schedules, and has
focused server integration coverage for byte-hash/output agreement and
fail-closed inputs. The canonical Metal check passes the DeepSeek tests and stops at the unchanged
Julia numerical mismatch; focused CLI, strict lint and documentation gates pass. Downstream weights and
source checks remain fixture-backed; broader partitions, checkpoint loading and
Metal execution remain open. Native
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
Responses, and qualified 0.6B/4B controls. DeepSeek's library `RequestSession`
now interleaves all five blocks and the final head per call. Both established
synthetic schedules pass source bounds and replay after restart. The bounded
artifact loader and scalar CLI have focused qualification. The workspace check
remains red at the unchanged Julia gate; broader partitions and DeepSeek Metal
execution remain open. Synthetic numerical
qualification does not establish checkpoint support. Julia's bounded 22-layer
path exists but has not passed numerical qualification. Its scalar numerical
oracle and bounded Metal operators are not a complete GPU encoder/decision
runtime.

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
milestones are complete model execution: native DeepSeek generation and, only
after its numerical gate, a direct Julia typed-decision path. Preserve Qwen's
working text, tools and decision commands while pursuing those consumers. The
existing Qwen bridge can qualify JevBench-shaped request and receipt behavior;
it must identify its decoder route and cannot stand in for Julia checkpoint
evidence. SMC and steering remain research and regression surfaces until a
concrete application needs them.

**Open before expensive checkpoint work:** define the target context, acceptable
first-token latency, minimum useful decode rate and SSD/RAM budget for the
intended workflow. The current measurements do not settle those requirements.
Correctness work can continue while the feasibility lane makes the tradeoffs
concrete; an unspecified performance target cannot establish usable serving.

## Immediate parallel work

The next device slice now has a passing live-L3 index-score qualification:
computed queries and committed owner keys reach the existing Metal kernel,
with CPU score/selection controls and a measured cutoff-margin gate. The bounded
library BF16 scorer also matches every dot, ReLU, weighted and final score
boundary exactly on the alternate schedule. Shared shape/work/finite-input
checks precede GPU construction; each stage rejects nonfinite results, and
finite calls recover after overflow. An explicit candidate-projector choice now
runs scoring on Metal inside L3, with alternate-session publication/output parity
and pre/post-commit failure/reset controls. Canonical L4 score operands also pass
exact stages. `RequestModel::with_score_execution` now propagates the explicit
choice through L1, L3 and L4. Both complete reduced schedules match scalar scores,
selections, publications and final outputs, preserve independent source bounds,
and replay after restart. A malformed late L4 attention operand still poisons the
whole request. The artifact API and CLI now expose the same explicit choice;
Metal receipts identify mixed CPU/Metal execution. A score-device-error injection
remains open. Scalar remains default.
Promote a device execution component only after
its source staging, downstream state and failure behavior remain qualified;
operator agreement alone is not a complete Metal request.

The expanded first-chunk sweep establishes terminal stability for 2, 3 and 6;
per-call qualification is still separate. First-chunk 7 reaches a native L4
ambiguous cutoff tie while the source completes with changed output. Preserve
the explicit rejection: a full source capture confirms equal zero scores at
reachable L4 positions 4 and 6, and the source API offers no portable tie order.
Admission needs an explicit stable policy with quality qualification or a pinned
backend oracle; it is not a prerequisite for Metal work on established schedules.
See the [source tie audit](research/v41-forward-reference.md#expanded-first-chunk-sweep).

Use two bounded implementation lanes plus one independent reviewer when useful.
One owner integrates and runs Cargo/device checks. Research must resolve a named
implementation uncertainty and produce a pin, contract correction, or executable
gate; another broad model survey is not on the critical path.

| Lane | Next deliverable and consumer | Exit gate | Reversibility |
| --- | --- | --- | --- |
| DeepSeek primary | Keep the bounded artifact CLI as the single executable consumer of `RequestModel` and `RequestSession`. It now reads an exported numerical artifact and has focused byte-hash/output and fail-closed integration coverage. It now offers explicit mixed CPU/Metal scoring. Next: remaining reduced arithmetic on Metal and broader schedule qualification. | Existing final-logit source bounds pass for both schedules; focused CLI integration and strict DeepSeek/server Clippy pass. The canonical check has no new DeepSeek failure and stops at the known Julia mismatch; a green workspace claim remains closed. Preserve fail-closed malformed input/weight geometry and keep fixture readers, captured intermediates, and expected outputs out of the executable. | Fixed synthetic shapes and the numerical artifact remain reversible; full GPU arithmetic, checkpoint loading and serving remain separate gates. |
| Julia secondary | Resolve the numerical fidelity target for the bounded 22-layer prototype before expanding prefill to ordinary typed requests. Consumer: direct Julia decisions. | The fixed calibration source control now matches actual eager attention exactly; this does not clear the full native gate. **Decision-required before further native arithmetic:** retain pinned-source-backend compatibility or record a separately declared portable scalar contract. Preserve the enabled `1e-5` hidden-state gate and qualified two-block/head controls; do not widen limits. Under the selected target, use a calibration-only layer-zero source-oracle control to separate QKV/RoPE/score reconstruction from backend behavior, then qualify longer padded and window-crossing inputs. The head contract is type embedding, two bidirectional encoder layers, marker-position gather, and a scorer, not ordinary pooled classification. Held-out evaluation and promotion remain closed until the chosen calibration and independent-reference controls pass. | Reversible CPU control and contract evidence before Metal or checkpoint loading; changing the fidelity target requires its own recorded decision. |
| Feasibility and review | Establish DeepSeek route locality before checkpoint acquisition. Consumer: the checkpoint acquisition decision. | Declare RAM/SSD/context/latency envelope; capture an actual source-compatible route trace without expert payloads; replay explicit cache capacities before reading selected real ranges. Keep metadata sensitivity distinct from measured hit rates, bytes, and latency. | Read-only metadata/trace work and bounded local probes; residency, prefetch, and pager policy remain decision-required. |
| Qwen maintenance | Keep text/tools/typed decisions working; qualify concurrent admission and cancellation under load before a Codex-ready claim. The retained six read-only trials include three pointer chains, now checked against ordered direct file-read events. | Preserve focused protocol/assessor regressions and exact tool-result replay. Separate task quality from protocol failure; no general coding claim from the fixed read trace. | Reversible serving checks; no further scheduling benchmark variants. |

### Dependency order and stopping gates

Next delivery sequence, grounded in `7894d99`, the
[reduced-executor decision](design/deepseek-reduced-executor.md), and the
[architecture gates and pivot conditions](architecture.md#pivot-conditions).
Review after the first request using Metal scoring, any state/parity failure,
or evidence that the intended checkpoint cannot meet the resource envelope.
The Rust API and reduced artifact CLI now execute the mixed CPU/Metal request.
The next implementation gate is the remaining reduced arithmetic on Metal. Scoring qualification alone does not establish full GPU execution.

| Order | Deliverable and consumer | Gate before proceeding |
| --- | --- | --- |
| 0 | Reconcile the completed isolated lane with the intended integration branch. Consumer: a reproducible project baseline. | Establish checkout ownership, classify peer edits and local commits, integrate in an owned checkout, and rerun the relevant gates. Preserve the known Julia failure explicitly. Do not mutate the peer-owned dirty main checkout or discard its work. |
| 1 | Qualify the library BF16 scorer on canonical as well as alternate operands. Consumer: the scored-query path. | Exact existing BF16/routing controls pass on both schedules; malformed input, overflow and recovery remain covered. Retain the prefill-7 tie rejection. Stop adding score microbenchmarks unless they resolve a named integration uncertainty. |
| 2 | Add an explicit DeepSeek-local device-scoring route: scored-query preparation → L3 projector/session → L1/L4 → request. Consumer: the reduced artifact CLI. | First compare an L3 session using device scoring against its scalar counterpart, including committed state and failures. Then the actual request must invoke Metal scoring and preserve final-logit source bounds, publication identity and candidate/selection results on both schedules. A late device error invalidates the whole request; restart reconstructs state and reproduces the clean result. Make backend choice observable; report this as mixed execution. |
| 3 | Move the remaining reduced request operations onto Metal in dependency order. Consumer: the same token/artifact CLI. | Identify residual CPU computation/transfers on that executable path, then move query/key preparation, attention, block tails/Engram and the head with their existing independent oracles. Complete GPU arithmetic plus routing/cache/reset checks is the exit gate; per-stage readbacks are diagnostic instrumentation, not a throughput implementation. |
| 4 | Establish useful performance and the checkpoint resource envelope. Consumer: the checkpoint acquisition/residency decision. | Record matched artifact/input/schedule timing and peak memory for scalar and device paths. In parallel, obtain source-compatible route locality evidence and declare hardware, RAM/SSD, context, TTFT and decode-rate limits. A small synthetic graph cannot establish full-checkpoint fit or speed. |
| 5 | Admit one real checkpoint-to-request vertical, then serving. Consumer: ordinary local generation. | After reference parity and a viable resource plan, qualify actual loading, tokenizer/template, prefill/decode and output behavior together. Serving additionally needs cancellation, admission and bounded-resource recovery. Apply the existing upstream-runtime pivot if a competing implementation meets the same declared target. |

The next bounded candidate within step 3 is index-key RoPE in the existing
L1/L3 owner path, reusing `rotate_tail_metal`. Compare the actual BF16 `post_rope`
and FP4 reconstruction against the current source/key-prefix controls on both
schedules; a small FP32 rotation error alone is insufficient. Qualify owner
publication and reset behavior, then thread it through the same request/artifact
consumer. Keep it scalar if exact downstream staging fails. This is a proposed
integration step, with no timing or GPU-residency claim. A final-head alternative
needs a new validated FP32 projection; full query preparation additionally needs
Metal FP8 projection, normalization and FP4 staging.

Steps 1–3 are reversible model-local changes. Integration in step 0 must preserve
recoverable peer state. Checkpoint acquisition and residency choices in steps 4–5
are consequential resource commitments. Do not start full-checkpoint acquisition
until the resource plan is decided and the architecture's reference-parity gate
passes. Do not make the default path Metal until the request-level gate passes.

Use a small execution budget: one DeepSeek implementation owner, one independent
Qwen/feasibility owner when there is concrete work, and an independent reviewer.
The parent owns integration and serial Cargo/device execution. Research supplies
answers to a named gate; broad model surveys, new scheduling examples and generic
framework work do not occupy the critical path.

Parallel work has separate exit gates:

- **Qwen:** real-model callback cancellation and reuse of the loaded session now
  pass, with a fresh executor reproducing clean-baseline token IDs. Next qualify
  socket-driven cancellation under load and measure resource release, then choose
  bounded admission versus queuing under concurrent arrivals. The callback test
  does not establish recovery of cancelled KV state or physical memory release.
  Preserve the existing typed-decision and tool replay controls; useful Qwen
  delivery need not wait for DeepSeek.
- **Julia:** retain pinned-source-backend compatibility as the current contract
  while the fidelity decision is pending. A portable scalar contract needs its
  own recorded decision and independent oracle. Do not change native arithmetic,
  widen the hidden-state gate or use held-out data to tune the failing calibration.
  Under the selected target, resolve the calibration mismatch, then qualify
  encoder-to-head decisions and broader inputs.
- **Feasibility:** source-compatible route traces and explicit cache-capacity
  replay can precede completion of the Metal executor. The hardware/latency
  envelope is a user decision; generic I/O measurements cannot substitute for it.

Retire duplicate test-only arithmetic when a library component replaces it, as
already done for the BF16 Metal graph. Keep one scalar oracle, one runtime
consumer and independent source assertions. Each implementation batch should
close a request-execution gate, not merely add another standalone diagnostic.

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
| DeepSeek device-path selection, `src/indexer/query.rs` and `src/reduced/**` | Explicit per-operation Metal siblings keep each surface obvious but multiply call paths; a DeepSeek-local immutable execution choice centralizes dispatch but touches request construction. | Decide at step 2 after the call-site survey. Prefer one model-local choice carried through the request, preserving the scalar default and existing validation/state logic. No engine-wide backend trait; keep the diagnostic scorer's readbacks explicit until a separately qualified resident form replaces them. |
| Qwen concurrent admission, Responses/chat serving | Reject while busy; or queue a bounded number of requests with deadlines and cancellation. | Decide after real model cancellation/recovery is measured. Start from bounded single-model admission; no unbounded queue or concurrent mutation of one session. Record the accepted policy before changing the serial serving loop. |
| DeepSeek request failure lifecycle | Roll back every owner; or invalidate a failed request and reconstruct all state. | `RequestSession` now implements invalidation and full reconstruction from immutable operands. Both schedules replay through source-qualified logits; malformed L4 execution blocks retries and restart reconstructs the earlier owners. This does not promise component rollback or checkpoint-serving recovery. |
| DeepSeek checkpoint resource envelope | Acquire/load the checkpoint before locality evidence; or establish the intended RAM/SSD/context/latency envelope, trace source-compatible routes, then measure selected ranges at explicit cache capacities. | Take the latter path. Metadata sensitivity and generic Qwen I/O probes are not a V4.1 serving envelope. Do not start acquisition, a pager, or prefetch policy until this gate has a measured result. |
| Julia fidelity and benchmark route | Run JevBench-shaped requests through the qualified Qwen decoder bridge; or claim a direct Julia result after its source-compatible encoder/head path qualifies. Separately, choose pinned-source-backend fidelity or a portable scalar contract for the failing Julia numerical gate. | Keep Qwen as the current benchmark-interface route. Record the Julia fidelity target before another arithmetic change; direct Julia scoring remains closed until its marker-gather/head path passes the resulting independent gate. |
| Native runtime versus upstream integration | Continue the specialized executor; or retain qualification/adapter work and integrate a runtime that demonstrably meets the same target. | Apply the existing architecture pivot using matched evidence, not popularity. |
| MLX binding upgrade | Keep the qualified pinned stack; or upgrade to unlock a demonstrated blocking operation/performance gain. | Avoid combining an upgrade with a cache-layout change. Qualify an upgrade as its own change before depending on new semantics. |
| Adapter/config shape, model and engine boundary | Closed typed task adapters; versus a universal tensor/graph interface. | Follow the existing thin-MLX proposal. Decide exact signatures only with concrete consumers; keep manifest versions explicit. |
| Workspace mutation and broader Codex tools | Retain read-only tools; or introduce scoped execution/write authority with cancellation and isolation. | Keep protocol qualification separate. Decide the execution policy before adding shell/write tools to `mx agent`. |

Do not start production cache replacement until its storage/aliasing contract
is decided. Do not start full-checkpoint DeepSeek serving until reference
parity and storage/latency feasibility pass. Do not promote Julia as a direct
checkpoint adapter until its fidelity target and numerical gate are decided and
passed. Do not publish a generic adapter API until actual consumers establish
the shared interface.

## Research, maintenance and stop rules

The [September 29 evidence update](research/architecture-landscape.md#september-29-update-evidence-that-changes-the-next-gates)
adds Jeff, Ollama's decision endpoint, Nimble/Tev1, and inference-time visual
control to the comparison corpus. Immediate order remains DeepSeek device
qualification, Julia oracle/fidelity resolution, Qwen protocol reliability, and
route-locality feasibility in parallel. Small trained decoder decision models
are the next adapter comparison, not automatically a replacement for Julia.
World/video models and non-autoregressive decoding remain conditional research.

- Keep one executable path for the reduced DeepSeek request: the synthetic
  artifact binds the CLI to `RequestModel`/`RequestSession`, while source
  fixtures remain test-owned oracles. Consolidate any duplicate executable
  runbook around that path; do not add a second fixture-backed runner or general
  backend seam.
- Keep the newly inspected Jeff decision checkpoints at watch status. The pinned
  source at `f06788292874c21a5b5c41549ac220dd9e15da7f` describes a causal decoder
  with chat-template serialization, tokenizer-validated one-token option codes,
  a custom trained 255-way final-hidden-state readout, option masking, and fitted-
  temperature softmax. It is neither raw Qwen option logits nor Julia's
  bidirectional marker-head path. Admit it only through a separately pinned
  backend/artifact contract with source-qualified state, layout, precision, and
  variable-shape controls; its reported hard-task result is a local diagnostic,
  not an official JevBench result.
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
