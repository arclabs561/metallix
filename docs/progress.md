# Metallix progress and next gates

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
  exact BF16 attention-input checks reject corrupted handoffs. The layer-two
  post-attention input remains captured; this is still a reduced graph.
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

## Next, in dependency order

1. **Performance owner: native inference lane.** Profile the measured decode
   workload before changing another kernel or precision. CPU sampling points
   to Metal completion waits; host argmax replacement did not improve repeated
   request time and was rejected. The maintained `scripts/qualify-chat.py` runner
   checks CLI/HTTP parity, long/short recovery and optional process CPU samples.
   Keep output-token parity, cached/full-forward checks, and repeated
   wall-time measurements as gates. The source-checked owner benchmark separates prepare/drop/commit
   costs; captured prefixes five and six do not establish a scaling bottleneck.
   Measure longer prefixes before changing staged allocations. Prefix reuse,
   quantization, and batching require separate evidence.
2. **Model owner: DeepSeek forward lane.** Extend the source-grounded reduced
   forward through the earlier text blocks. Layers three and four now connect
   natively through final logits, preceded by native Engram3 and layer-two FFN.
   Layer-two post-attention state still comes from capture. The next boundary
   is layer one's ratio-two compressed KV/index publication into layer two;
   this pair does not use the layer-three/four candidate-mask path. Replace the
   captured state with native earlier-block output while
   preserving the source-grounded arithmetic and discrete routing gates.
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
