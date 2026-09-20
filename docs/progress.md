# Metallix progress and next gates

The [delivery roadmap](delivery-roadmap.md) proposes sequencing,
adoption gates and decision points; this page records delivered evidence.

The broader direction is recorded in [model adapters and MLX capabilities](model-adapters.md):
compute API/CLI, multimodal execution, SMC/sampling, and training/LoRA support.
Existing-model completion and profiling lead delivery. The
[100-page Hub survey](research/hf-trending-landscape.md) informs future adapter
choices without expanding current support claims.

## Delivered in this lane

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
- A separate native Codex command-tool check passed 3/3 fresh synthetic-fact
  trials against the 4B server at 16,384 tokens and 8192 MiB. Strict
  reassessment verified command execution before each exact final answer and
  terminal completion; the controlled CLI was 0.153.4. It used fallback model
  metadata and controlled instructions/features, so no user profile was added.
- The reusable Responses qualifier passed three JSON and three SSE native 4B
  tool-result replay trials at 2048 tokens and 1024 MiB. It now validates model,
  output-item, and content identities; three live tool-stream disconnect
  recoveries also passed. Results are synthetic client-provided values, not
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
  selected IDs at starts 0, 5, and 6. The twelve-check source suite preserves
  the partial-start distinction: its score-key operand is a captured layer-three
  shared prefix, not layer one's retained owner key/KV state. This is not a
  production API or complete native layer-one execution.
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
- The [model/runtime refresh](research/model-runtime-refresh.md) records primary
  sources for Qwen3.8, vLLM-Metal, and Whallm, with explicit qualification gates.
- The engine now has a bounded SMC state primitive with deterministic
  systematic resampling and absorbing particles. A Qwen test-only LoRA
  micrograph checks gradient/update ownership and adapter export parity; neither
  is a production training or particle-decoding API yet.

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
   controls. The next boundary is a full stateful runner, while preserving the
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
   Codex tasks. A bounded native Codex command-tool run passed 3/3 fresh trials
   against the 4B server at those limits, but used fallback model metadata and
   does not establish general coding or complete tool grammar support.
   No live profile was changed.
5. **Runtime owner: comparison lane.** Run matched quality/tool/performance
   workloads on an independently owned current local runtime before deciding
   whether a new native model adapter is worth its implementation cost.

## Validation

The cooperative budget's live probes covered JSON and SSE at 1 ms and 100 ms,
followed by healthy requests. The warm 100 ms stream emitted ten text deltas
before exactly one timeout failure. With the default budget, both JSON and SSE
clients that half-closed their sending side received the expected 32-token
response and matching text. A 1 ms budget still took 22–40 ms to return:
the currently running synchronous phase must finish before expiry is observed.

The canonical checks are `RUSTC_WRAPPER= uv run scripts/check.py` and
`RUSTC_WRAPPER= uv run scripts/check.py --metal`. The wrapper override avoids a
local sccache startup failure and does not modify global compiler settings.
Both checks passed on the final implementation, followed by a release build
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
