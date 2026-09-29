# Metallix progress and next gates

The [delivery roadmap](delivery-roadmap.md) proposes sequencing,
adoption gates and decision points; this page records delivered evidence.

The broader direction is recorded in [model adapters and MLX capabilities](model-adapters.md):
compute API/CLI, multimodal execution, SMC/sampling, and training/LoRA support.
Existing-model completion and profiling lead delivery. The
[100-page Hub survey](research/hf-trending-landscape.md) informs future adapter
choices without expanding current support claims.

## Delivered in this lane

- A frozen four-task/two-seed schedule comparison now separates non-overlap
  validity from requested count/duration/window adherence. Verified retries
  did not improve task adherence on either local checkpoint: 0/8 for both,
  versus 0/8 (0.6B) and 3/8 (4B) baselines. All 32 runs completed without
  infrastructure errors. The [candidate control note](candidate-control.md)
  records compute budgets, exact task hash and the task-aware-verifier gap.

- Optional `--schedule-requirements` now enforces exact interval count,
  duration multiset and window from a bounded local descriptor. Four local
  smoke cases accepted a compliant schedule and rejected each mismatch.
  The unchanged frozen 4B follow-up exhausted all eight retry runs without
  publishing an invalid candidate; baseline task adherence remained 3/8.
  Stronger verification is established, improved task completion is not.
  A subsequent serialization-only diagnostic using the checkpoint chat template
  reached 7/8 verified successes versus 6/8 schema-only successes on those same
  tasks. A fresh, separately frozen same-family set reached 3/8 verified
  successes versus 2/8 baseline, using 1,606 versus 478 generated tokens.
  Five verified runs exhausted their budgets; no invalid schedule was accepted.
  This small confirmation does not establish general task quality.

- The ignored local 0.6B steering checkpoint test passes exact zero-coefficient
  identity, observable finite intervention, fork inheritance and parent-cache
  immutability. Full/chunk logit differences meet the existing `5e-5` bound
  for both unsteered and steered controls. Behavioral benefit remains unmeasured.

- Layer-zero native window-only attention, BF16 HC post/pre/post mixing and RMSNorm/MoE FFN
  reconstruct the exact residual consumed by layer-one Engram at starts 0, 5
  and 6. Native HC pre-mix and RMSNorm also reproduce the attention input
  exactly, with an incoming-residual corruption control. Corrupted HC
  coefficients fail the source oracle; rejected out-of-order
  attention calls leave the state retryable, including reset and replay.
  Token embedding, HC-copy expansion and identity pre-mix now reconstruct the
  incoming block state from the same trace's IDs and weights. Both attention
  and FFN HC coefficients now come from native projection; next-pre values use
  the existing analytic F32 envelopes, while the native layer-one pre-mix and
  RMSNorm consumer matches the exact BF16 source attention input. These joined
  cases now compose one bounded prefill/decode partition through native Engram1,
  layer one and the existing final reduced suffix. The checked-in unified source
  bundle feeds layer zero and the final L3 HC/FFN/MoE operands. The request
  retains a typed L3 projection with matching bundle capture/revision/model
  identity and unchanged numerical limits. Mixed-capture metadata is rejected;
  zeroing a supplied L3 normalization weight fails the final numerical oracle.
  L4 uses the same bundle for attention, candidate selection, MoE operands and
  the final head. It directly consumes the persistent L3 session's committed
  key/KV prefixes and publication identities, then computes its own query and
  selection. It no longer reruns the L3 owner at finalization. Snapshots retain
  successfully consumed L3 inputs, and reset clears publication history.
  The persistent L3 Engram, owner, compressor, candidate selection and attention now
  retain caller-supplied unified-bundle operands. The bootstrap uses unified L3
  HC parameters too. Standalone legacy wrappers retain their identity gates.
  L2 now retains unified HC, attention and FFN operands and requires the live
  L1 publication. L1 now retains unified Engram, owner, attention and HC/FFN
  tail operands too. The fixed reduced trace uses the unified bundle throughout;
  it remains a test-private, synthetic-shape composition, not checkpoint-backed
  generation. The previous-call layer-three
  key prefix now comes from native execution of starts zero and five and feeds
  the actual layer-one start-six selection used by the final-logit path.
  Rejected publication preserves nonempty owner state for a same-ID retry;
  reset clears it for a new request. One live layer-three owner now advances
  through starts zero, five and six, and its attention outputs feed the final
  logits without replaying that owner. Owner and attention epochs advance
  together on reset. Layer one now also retains Engram, compressed-owner/shared-score
  and attention state across the three calls. Its starts-zero/five results feed
  the layer-three bootstrap directly; the same layer-one session then consumes
  the published prefix at start six. Native attention inputs feed its owner
  projections, with captured values used as exact checks. Layer-two attention
  and Engram3 hashing now also retain state across those calls. Layer two
  consumes the exact publication produced by live layer one; its independent
  fixture-fed owner remains only for standalone controls. Bootstrap consumes
  the already-computed layer-two/Engram3 prefix, and final continuation advances
  the same sessions once at start six. This removes that prefix replay from
  the data-producing path. Fixture-backed operands and alternate call partitions still separate this
  test-private composition from a production token-to-logits runner.
  A completion audit found that both live Engram helpers checked supplied
  residuals after committing hash history. They now reject bad inputs first.
  Regressions at starts zero and five prove rejected calls cannot make future
  token history available and that correct retries reproduce the control
  continuation. This covers input preflight only: a late layer-one publication
  mismatch can still advance earlier owners before rejection. A test-private
  request wrapper now marks the composition poisoned before mutable work,
  rejects continuation after a late failure, and reconstructs all its L1–L3
  owners on restart. Regressions compare restarted and fresh final-block
  outputs and check both against the final-logit source oracle. A finalization
  corruption control also rejects retry after the operand is repaired; removing
  finalization poisoning makes the regression fail. This establishes
  invalidation/rebuild for the reduced fixture path, not production rollback.
- Julia-1's source-pinned encoding contract now has nine pure-stdlib fixtures
  for marker placement, option ordering, mask sanitation, strict truncation,
  and padding. Its bounded artifact inspector validates the complete pinned
  published safetensors header without reading tensor payloads.
  Six published-tokenizer sequence cases now match the audited pinned source,
  including option permutation and strict rejection. A synthetic F32 head
  reference independently spells out attention, residuals, feed-forward, gather
  and scoring against the pinned source. The CPU-only Julia crate executes
  that head from supplied weights and hidden states, with all five frozen cases
  and bounded-input failures covered. It also executes one bounded F32
  ModernBERT encoder block from caller-supplied weights: layer-zero identity
  normalization, later affine normalization, Q/K RoPE, global and +/-64 local
  attention, and GEGLU. Eight source-backed synthetic output cases cover
  padding, local/global windows, distant-token isolation, and an all-masked
  local query. Five additional source-backed cases join two encoder blocks to
  the head, including padding and marker controls.
  A bounded eight-token, 22-layer prototype adds selected token lookup,
  embedding normalization and final normalization, but its full hidden-state
  comparison against the pinned SDPA source fails the fixed `1e-5` bound on the
  unmasked control. Three isolated reduction-precision experiments did not
  close that gap and were reverted. Full-stack qualification remains open;
  neither the tolerance nor the fixture weights were changed. A preregistered
  independent F64 experiment now localizes a separate calibration failure to
  layer zero on `cal_len7`: native error exceeds the source-derived envelope
  by approximately 2.00335 times. Traces locate the excess inside attention,
  before the output projection. Accumulating only attention QK dots in F64
  reduced that error but introduced a second calibration failure, so the
  experiment was reverted and its baseline/trial evidence retained. Same-input
  replay then separated QKV propagation, rotary transforms and score reduction.
  A balanced-F32 score reduction passed the legacy source test but worsened
  frozen calibration from seven passing cases to six, introducing a new
  `cal_len5` failure; it too was reverted. Held-out
  inputs remain unopened; see the
  [numerical experiment](experiments/julia-accuracy-contract.md). This is not a
  full-checkpoint encoder, Metal implementation, or serving integration; see
  the [Julia contract](research/julia-decision-contract.md).

- `mx decide` scores Qwen3 answer-letter logits directly for bounded `choice`,
  `score`, and `noul` questions, returning normalized option probabilities and
  zero generated tokens. Each question gets fresh KV state. The local 0.6B
  checkpoint passed three typed receipt/replay checks and completed all 72
  pinned public JevBench tasks without execution or receipt failures, with
  35 correct (48.6%). The same prompt/tasks on Qwen3-4B-Instruct-2507
  produced 65/72 correct (90.3%), again with complete valid receipts. This is
  local adapter qualification, not an official benchmark result or calibrated confidence. See [typed decisions](typed-decisions.md).
- SMC importance updates now reject proposal zero support and invalid log
  probabilities atomically; target zero support rejects the particle. The
  finite independent oracle checks this boundary and composed stage means
  across two resampling rounds. An ignored checkpoint test now composes real
  Qwen proposals, cache forks and particle weights under explicit synthetic
  potentials; the local 0.6B run passed, including fresh-prefix and parent-cache
  replay. A second checkpoint test checks raw temperature-one target versus
  temperature-0.7 proposal correction, normalized weights, ESS, ancestry and
  fresh-prefix replay, rejecting lost proposal support. This is a test-only
  driver, not a particle-serving API.

- Native Qwen3 chat uses the checkpoint template and keeps model weights loaded
  across turns. KV state is rebuilt per turn. `--context-tokens` defaults to
  2048 and `--kv-budget-mib` to 512 MiB, with experimental ceilings of 16,384
  and 8192 MiB; diagnostic commands retain their 512-token limit. The pinned
  4B checkpoint revision `cdbee75f17c01a7cc42f958dc650907174af0554` passed the
  bounded four-case agent workload in all 12 trials at 2048 tokens and 1024 MiB.
  Its 151,936 Metal logits matched the CPU reference, with maximum absolute
  difference 0.0000581741 and RMSE 0.00001409596. Eight cached-versus-uncached
  positions also matched across all 151,936 logits, with maximum absolute
  difference 0.000020980835. This does not qualify the expanded ceilings,
  general coding, or general Codex coding.
- A fresh native Codex run passed all six read-only tasks against the same 4B
  checkpoint at 16,384 tokens and 8192 MiB: three single-file reads and three
  two-file pointer chains. The assessor verified ordered command evidence,
  exact final answers, terminal completion and unchanged synthetic workspaces.
  The temporary server stopped after the run. Receipts are retained locally at
  `.agents/receipts/qwen-codex-chain-20260928/`; `run.json` pins the release
  binary SHA and checkpoint revision. This remains bounded tool qualification,
  with controlled CLI instructions/features and fallback model metadata.
- The reusable Responses qualifier passed three JSON and three SSE native 4B
  tool-result replay trials at 2048 tokens and 1024 MiB. It now validates model,
  output-item, and content identities; three live tool-stream disconnect
  recoveries also passed. A new workload requires two ordered calls and both
  fresh result values in the final answer. All six new JSON/SSE trials passed
  with the same checkpoint and admission limits. Results are synthetic client-provided values, not
  filesystem tool execution. The private stepped-capacity Qwen experiment passed
  all 50 whole-logit trace pairs, reduced the long-prompt median by 18.16%, and
  regressed short prompts by about 1%; 128-plus-64-token logical KV is 112 MiB rather
  than fixed capacity's 448 MiB. Production adoption still requires matched real
  4B requests. The resident executor now uses the stepped storage path; matched
  2048-token real-server requests preserved output hashes and 64 generated IDs
  for Qwen3-0.6B and Qwen3-4B, with the 4B decode median moving from 3796.7 ms
  to 3600.3 ms. See the [measurement ledger](experiments/chat-performance.md).
- `mx agent` runs a bounded read-only workspace tool loop. Complete tool calls
  and surrounding assistant text survive history replay. Paths are opened
  relative to a pinned workspace descriptor without following symlinks.
  `--json` records per-turn generation costs and executed tool names, valid
  relative paths, argument hashes, and outcomes. Execution completion is
  distinct from task success.
- `mx serve` offers an experimental loopback text/function Responses subset.
  A live independent HTTP client checked JSON, SSE ordering, length limits,
  post-header failure, invalid requests, a function result round trip, and a
  disconnected client followed by another request. Socket intake now has one
  absolute read deadline, and live stalled-header/body cases recover to healthy
  subsequent requests. Writes are bounded; a separate cooperative generation
  budget checks every decode boundary. In-flight Metal work cannot be interrupted.
- DeepSeek compressed-owner transactions allow scoring/selection against staged
  key/KV prefixes before commit. A source fixture forces downstream scorer
  rejection, discards the transaction, checks unchanged state, and retries.
- Property tests exercise context sizing/overflow, budget monotonicity, tool
  envelope truncation and nested JSON, Unicode byte limits, and literal search
  line numbers. A generated delimiter-in-string case found and fixed an actual
  tool parser bug. Further properties cover HTTP fragmentation, SSE framing,
  and function-result ordering; reused completed call IDs are now rejected.
- Native layer-two FFN output now feeds Engram3 and the connected layer-three/
  four suffix through final logits. Both native residual and HC pre-mix are
  retained as operands. Fixed HC arithmetic bounds handle FP32 rounding, and
  exact BF16 attention-input checks reject corrupted handoffs. Its isolated FFN
  control retains captured post-attention input; the joined attention test below
  supplies that boundary natively. This is still a reduced graph.
- Six test-only native layer-one owner checks execute the ratio-two compressor's
  FP32 WKV/wgate projections, then check the BF16 latent plus owned compressed
  KV/index-key publications, native index query and score stages, and causal
  selected IDs at starts 0, 5, and 6. A same-trace bridge now reconstructs the
  preceding layer-three ratio-one compressor/key publication natively from its
  captured attention input, then feeds the consumed leading three keys through
  request-local index state into the partial layer-one score/selection gate.
  It rejects a substituted prefix and preserves distinct owner key/KV state.
  Layer-three block input and attention remain captured boundaries; this is not
  a production API or complete native layer-one execution.
- The six-check source fixture
  `layer1-attention-reference.json` pins layer-one attention at SHA-256
  `a13bb6cd53406f04e436f119aa8dec2184ca43a4d4f4969205bff8bdf26ac31b`.
  Three focused Rust checks run `LayerAttentionState` from the native ratio-two
  owner KV/IDs, compare exposed adapter diagnostics and final output, reject a non-owner
  publication, and show that a legal wrong index changes output. The partial
  score operand derives from a strict prior layer-three candidate capture.
- The six-check source fixture `layer1-tail-reference.json` pins layer-one's
  attention-to-FFN tail and exact layer-two residual/pre-mix handoff at SHA-256
  `5f31036c71b797e7195a6b93cf2d656b8744b1e6a80a04dc68007d26e326b89b`.
  Its checks include source layer identity and HC binding restoration after an
  injected failure. A focused `forward_moe` join now carries native layer-one
  attention, HC, and FFN into native layer two and through the existing suffix
  to final logits. Layer-one initial residual/pre-mix and the partial shared
  layer-three score keys remain captured boundaries, so this is not whole-graph,
  Metal, or checkpoint execution.
- A source-only layer-two attention exporter/fixture captures exact source
  storage for the borrowed layer-one publication, local window state, attention
  boundaries, and the historical layer-two FFN handoff. Its nine checks pin
  source and helper provenance, storage hashes and raw finite values, reject a
  missing, altered, or paired-forged owner publication at the sparse boundary,
  and retain the FFN seam. This exporter remains source-only; native layer-two
  attention and full-model parity are separate claims.
- Standalone native layer-two attention consumes the native layer-one KV/IDs at
  all captured calls. It rejects a non-owner publication and shows that a legal
  but wrong layer-one index changes output. A focused joined suffix derives the
  normalized layer-two attention input from captured layer-one residual/pre-mix,
  then carries native attention through HC post-mixing, native FFN, Engram3,
  native layer-three/four, and final logits. BF16 boundaries are exact and HC
  coefficients satisfy fixed analytic bounds; discarded attention fails before
  FFN. Layer-one terminal residual/pre-mix, owner inputs, and the partial shared
  layer-three score keys remain captured. This is not full DeepSeek execution.
- A seven-check source-only layer-two HC fixture pins the layer-one terminal
  residual/pre into layer-two block input, HC mixes/coefficients, attention
  boundaries, and exact attention/FFN handoffs. It validates pinned and live
  source hashes, strict storage, HC configuration, and captured parameter
  identity. It is not native HC, native attention, or full-forward parity.
- A source capture supplies layer-three block-entry residual/pre-mix;
  native HC and RMSNorm now derive the owner/candidate attention input and
  cross-check it against preserved historical captures.
- The layer-three continuation now consumes native staged owner keys/KV and
  native producer-selected indices through attention output. Its separate
  source fixture captures layer three's actual rotary schedule. Properties
  check wrong-owner rejection with a valid retry and output sensitivity to
  legal-but-wrong selected indices; production candidate storage is unchanged.
- Native layer-three attention now continues through HC and FFN to the
  layer-four entry. Residuals match source storage; FP32 pre-mix coefficients
  satisfy fixed source-derived bounds. Zeroed attention propagated through FFN
  fails the same entry gate. That native state now feeds the layer-four suffix
  through final logits: an exact BF16 attention-input check closes the handoff,
  and corrupted incoming coefficients fail at that boundary.
- A reduced DeepSeek suffix connects owner-produced KV/indices through native
  final-layer attention, HC, FFN, final norm and logits. Fixed source-derived
  envelopes reject omitted normalization; earlier layer inputs remain captured.
  This is not full-model generation.
- Performance receipts separate startup, prefill, decode, visible first-token
  latency, request time, and process RSS. See the
  [measurement ledger](experiments/chat-performance.md).
- A private Qwen cache-fork gate passed full-logit replay on the pinned 0.6B and
  4B checkpoints in FP32, plus 16 generated tiny-model ancestry cases. Duplicate
  branches leave parent and EOS cache snapshots unchanged. Particle scheduling
  and physical cache-sharing costs remain separate [sampling gates](research/sampling-next-gates.md).
- The programmable-control vertical now has same-session grammar checkpoints,
  explicit independent JSON Schema verification receipts, and transactional
  rollback of the constrained controller plus seeded sampling RNG on rejected
  draws. The Qwen executor exposes its validated KV fork operation. Resident
  `--verify-cache` now exercises that fork before each candidate decode and
  compares branch/parent next logits; streamed mode reports the branch
  limitation explicitly.
- The [model/runtime refresh](research/model-runtime-refresh.md) records primary
  sources for Qwen3.8, vLLM-Metal, and Whallm, with explicit qualification gates.
- The engine now has a bounded SMC state primitive with deterministic
  systematic resampling and absorbing particles. A Qwen test-only LoRA
  micrograph checks gradient/update ownership and adapter export parity; neither
  is a production training or particle-decoding API yet.

See [verified schedule candidates](candidate-control.md) for the current resident
Qwen consumer, receipt semantics, qualification command, and remaining evidence
gates for candidate retries, steering, SMC, and the DeepSeek producer join.

## Next, in dependency order

1. **Performance owner: native inference lane.** Profile the measured decode
   workload before changing another kernel or precision. CPU sampling points
   to Metal completion waits; host argmax replacement did not improve repeated
   request time and was rejected. The maintained `scripts/qualify-chat.py` runner
   checks CLI/HTTP parity, long/short recovery and optional process CPU samples.
   Keep output-token parity, cached/full-forward checks, and repeated
   wall-time measurements as gates. The source-checked owner benchmark separates prepare/drop/commit
   costs; captured prefixes five and six do not establish a scaling bottleneck.
   Current profiling places context growth inside MLX evaluation while transpose
   construction is below 0.3% of decode time. Standalone component graphs show
   material cache-concatenation growth with prefix length. Test a bounded
   The adopted stepped capacity-buffer path has exact logit/fork parity and
   matched real 0.6B/4B request receipts. Continue profiling it across request
   shapes; the pinned MLX binding exposes functional slice updates, not
   guaranteed in-place buffer reuse.
   Prefix reuse, quantization, and batching require separate evidence.
2. **Model owner: DeepSeek forward lane.** Extend the source-grounded reduced
   forward through the earlier text blocks. Layers three and four now connect
   natively through final logits, preceded by native Engram3 and layer-two FFN.
   The native layer-two attention/HC/FFN suffix now consumes layer-one's native
   KV/IDs, and native layer-one attention/HC/FFN now feeds it. Layer-one initial
   residual/pre-mix remains captured. Engram1 now feeds the native layer-one
   path through the reduced suffix with exact fixture and corruption gates. The
   layer-zero token embedding now has an exact source fixture and OOV/mutation
   controls. Native layer-zero window-only attention, HC mixing, RMSNorm and
   MoE FFN now feed Engram1. Native embedding and identity HC initialization
   supply the same-trace incoming block state. Native HC projection now supplies
   coefficients under the existing F32 envelope policy and exact BF16 consumer
   checks. Compose these joins into one stateful runner while preserving the
   source-grounded partial-call layer-three shared score keys and discrete
   routing gates. This pair does not use the layer-three/four candidate-mask
   path. Operator and joined-suffix parity do not establish full model generation.
   The partial-group source trace also distinguishes the unchanged owned index
   cache from the shared keys actually scored; the latter retain the preceding
   layer-three publication. Resolve that [source behavior](research/v41-forward-reference.md)
   explicitly before assuming every layer-one score reads its own keys.
   Operator and transaction parity do not establish full-model generation. Keep full checkpoint
   acquisition behind the existing reduced-forward numerical gate.
3. **Agent owner: CLI lane.** Improve multi-step task completion before claiming
   coding-agent readiness. The maintained `scripts/qualify-agent.py` gate runs
   three trials each of literal search, a two-file chain, long-file reading,
   and a factual join across two files, checking exact execution evidence.
   The 0.6B control passed search, long reads, and the factual join in all trials
   but stopped after the first file in all pointer-chain trials, despite
   returning exit code zero. The join used two extra successful searches,
   recorded as an efficiency cost. The qualification runner rejects unfinished
   tasks. Shell and write tools
   need a separate execution policy; this delivery does not execute them.
4. **Serving owner: Responses lane.** Socket read/write deadlines are bounded;
   qualify concurrent admission, client-driven cancellation, the experimental
   16,384-token/8192-MiB resident limits, custom-tool requirements, and broader
   Codex tasks. A bounded native Codex command-tool run passed three single-file and three
   two-file-chain trials against the 4B server at those limits, but used fallback model metadata and
   does not establish general coding or complete tool grammar support.
   No live profile was changed.
5. **Runtime owner: comparison lane.** Run matched quality/tool/performance
   workloads on an independently owned current local runtime before deciding
   whether a new native model adapter is worth its implementation cost.

## Validation

A separate [source partition probe](research/v41-forward-reference.md#partition-experiment)
compares `5 + 1 + 1` against `4 + 1 + 1 + 1` using fresh models and identical
synthetic parameters. Baseline head and L3 KV match their frozen oracles.
Logits match exactly at common endpoints 5, 6 and 7; all 18 recorded final
cache/state fields match. Intermediate compressor scratch and call-shaped
candidate/selection state differ and remain recorded. This is source evidence;
An independently controlled capture now verifies all four L1→L2 and L3→L4
bridges plus the preceding L3→L1 partial-prefix handoff. The unobserved control
disables both intermediate hooks and kernel tracing; all four calls preserve
logits and recorded cache identities. A compact source fixture retains these
operands with provenance. Rust alternate-partition qualification is still open.
Nine probe tests and six bridge/extraction tests pass. The canonical Metal gate still reaches the
unchanged Julia failure below.


The composed L1 path now retains unified Engram, owner, attention and tail
operands. Its start-six owner call requires the preceding live L3 prefix before
mutating compressor state. Changed source metadata and weights are rejected;
the complete reduced path retains its request invalidation/reconstruction tests.
All 73 DeepSeek `forward_moe` tests and strict all-target/all-feature Clippy pass.
The canonical Metal check reaches the unchanged Julia encoder mismatch below.
Receipts: `.agents/receipts/candidate-control/unified-l1-*`.


The persistent L2 session retains unified HC, attention and FFN operands, and
requires the live L1 publication instead of replaying a legacy owner. All 70
DeepSeek `forward_moe` tests pass, including changed-weight/source rejection
and the absent-publication control. Strict all-target/all-feature Clippy passes;
the canonical Metal check still stops at the unchanged Julia encoder mismatch.
Receipts: `.agents/receipts/candidate-control/unified-l2-*`.


The persistent L3 path now consumes unified-bundle Engram, owner/compressor,
candidate and attention operands, with bootstrap HC from the same bundle.
Changed weights fail the existing numerical oracles before a successful owner
publication is retained; reset preserves operands and advances the epoch while
reproducing the original outputs and key/KV prefixes. Exact source metadata
checks cover these projections and both bundled MoE tails, including loader
identity. Legacy standalone gates retain their original fixtures.
All 67 DeepSeek `forward_moe` tests and strict all-target/all-feature Clippy
pass. The canonical Metal check, run before the final MoE metadata tightening,
passed its 66 DeepSeek tests and stopped at the unchanged Julia mismatch below.
Validation receipts: `.agents/receipts/candidate-control/unified-persistent-l3-*`.


The committed L3 publication handoff passes all 62 DeepSeek `forward_moe`
tests and strict all-target/all-feature Clippy against the existing final-logit
oracle. The canonical Metal check still stops at the unchanged Julia encoder
mismatch below. Receipts: `.agents/receipts/candidate-control/committed-publication-*`. Its negative controls reject malformed call histories, altered owner
inputs, wrong layer/epoch/call identities, and changed or truncated key/KV
prefixes without changing the producer. A valid finalization passes afterward.
L4 recomputes its own query/selection but never stages another L3 owner.
The obsolete post-publication compressor-weight mutation control was removed:
L4 no longer consumes those weights. Candidate, L4 normalization and head weight
controls remain, as do exact bundle provenance checks.

The unified L4 attention/owner/MoE/head migration passes all 61 DeepSeek
`forward_moe` tests and strict all-target/all-feature DeepSeek Clippy. Its controls
reject mixed owner capture, changed observer metadata, and changed compressor,
candidate, L4 normalization and head weights. The new path checks complete source
metadata against the committed bundle while consuming caller-supplied numerical
operands. The canonical Metal gate still stops at the unchanged Julia full-encoder
parity failure below. Receipts: `.agents/receipts/candidate-control/bundle-l4-*`.

The unified L3 operand migration passes all 60 DeepSeek `forward_moe` tests,
including mixed-capture and changed-weight rejection. The expanded exporter
integrity suite passes all nine checks, including full pinned source regeneration.
All 15 existing projections remain unchanged; three owner projections were added
under the same complete-capture identity. The calibration-only
Julia normalization replay reproduced the saved native and source embedding
outputs exactly and found no improvement from scalar Welford. No Julia runtime
arithmetic or acceptance limits changed.

The recovery batch passed all 58 DeepSeek `forward_moe` tests and strict
all-target/all-feature DeepSeek Clippy. Seven Julia diagnostic tests and all 22
Codex assessor tests passed. The canonical `RUST_TEST_THREADS=1 just check-metal`
run stops at the unchanged Julia full-encoder source-parity failure:
`unmasked_control hidden[0]: -0.2049238 != -0.20494038`
(12 passed, one failed, four ignored). Earlier green batches below are historical;
the current full gate is red. New receipts are retained locally under
`.agents/receipts/candidate-control/request-recovery-*`.

The cooperative budget's live probes covered JSON and SSE at 1 ms and 100 ms,
followed by healthy requests. The warm 100 ms stream emitted ten text deltas
before exactly one timeout failure. With the default budget, both JSON and SSE
clients that half-closed their sending side received the expected 32-token
response and matching text. A 1 ms budget still took 22–40 ms to return:
the currently running synchronous phase must finish before expiry is observed.

The canonical checks are `uv run scripts/check.py` and
`uv run scripts/check.py --metal`; preserve the configured `RUSTC_WRAPPER`.
The historical cooperative-budget batch passed both checks, followed by a release build
and the live HTTP checks above. The owned test server was stopped after
validation; use the README command to start a new one.

The earlier structured-agent and connected layer-three/four suffix batch passed both
canonical checks, the 15 qualifier tests, the source exporter rejection tests,
and observer restoration checks. Live JSON receipts also retain failure status
when workspace initialization fails. The earlier 0.6B four-task qualification remains
partially failing as described above; no Codex-readiness claim follows from
passing protocol and numerical checks.

The subsequent 4B/layer-two batch passed both canonical checks, including
the resident-budget properties and 17 Codex qualifier tests. Both source-backed
layer-two FFN and historical Engram suites passed seven tests. All 12 native
4B tool trials and three actual Codex command-tool trials passed; the preserved
Codex logs also passed the stricter ordered-event reassessment. The owned
qualification server was stopped. These results extend the bounded controls;
earlier DeepSeek blocks and general coding qualification remain open.

The layer-one owner/layer-two capture batch passed both canonical checks,
including all six new native owner/query/score/selection tests. Separate
source-backed suites passed 12 layer-one owner, nine layer-two attention,
seven historical layer-two FFN and seven Engram checks. The new source fixture
preserves the historical numerical handoffs. An optimized Qwen3-0.6B phase
probe passed five repeated rows at each of three prompt lengths with matching
final logit bits and whole-trace fingerprints; its measured host intervals are
recorded in the performance ledger. No production speedup follows from that probe.

The subsequent native layer-two attention join passed both canonical checks,
including three standalone attention tests and the 46-test FFN/suffix target.
The join includes 16 generated nonempty subsets of trace positions whose
attention outputs are discarded; each must fail at the HC post-mix boundary.
The new HC source exporter passed seven checks. The standalone Qwen cache
component probe passed a warmup and three measured rows at each prompt length,
with unchanged ordinary-decoder logits before and after the component graphs.
