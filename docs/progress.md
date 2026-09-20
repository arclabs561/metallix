# Metallix progress and next gates

## Delivered in this lane

- Native Qwen3 chat uses the checkpoint template and keeps model weights loaded
  across turns. KV state is rebuilt per turn. Resident chat admits up to 2048
  tokens after checking model limits and a 512 MiB logical KV budget; diagnostic
  commands retain their 512-token limit.
- `mx agent` runs a bounded read-only workspace tool loop. Complete tool calls
  and surrounding assistant text survive history replay. Paths are opened
  relative to a pinned workspace descriptor without following symlinks.
- `mx serve` offers an experimental loopback text/function Responses subset.
  A live independent HTTP client checked JSON, SSE ordering, length limits,
  post-header failure, invalid requests, a function result round trip, and a
  disconnected client followed by another request.
- DeepSeek compressed-owner transactions allow scoring/selection against staged
  key/KV prefixes before commit. A source fixture forces downstream scorer
  rejection, discards the transaction, checks unchanged state, and retries.
- Property tests exercise context sizing/overflow, budget monotonicity, tool
  envelope truncation and nested JSON, Unicode byte limits, and literal search
  line numbers. A generated delimiter-in-string case found and fixed an actual
  tool parser bug.
- A reduced DeepSeek suffix connects owner-produced KV/indices through native
  final-layer attention, HC, FFN, final norm and logits. Fixed source-derived
  envelopes reject omitted normalization; earlier layer inputs remain captured.
  This is not full-model generation.
- Performance receipts separate startup, prefill, decode, visible first-token
  latency, request time, and process RSS. See the
  [measurement ledger](experiments/chat-performance.md).
- The [model/runtime refresh](research/model-runtime-refresh.md) records primary
  sources for Qwen3.8, vLLM-Metal, and Whallm, with explicit qualification gates.

## Next, in dependency order

1. **Performance owner: native inference lane.** Profile the measured decode
   workload before changing another kernel or precision. CPU sampling points
   to Metal completion waits; host argmax replacement did not improve repeated
   request time and was rejected. The maintained `scripts/qualify-chat.py` runner
   checks CLI/HTTP parity, long/short recovery and optional process CPU samples.
   Keep output-token parity, cached/full-forward checks, and repeated wall-time measurements as
   gates. Measure staged DeepSeek prefix allocation before making KV snapshots
   lazy. Prefix reuse, quantization, and batching require separate evidence.
2. **Model owner: DeepSeek forward lane.** Extend the source-grounded reduced
   forward through remaining text blocks to logits. Operator and transaction
   parity do not establish full-model generation. Keep full checkpoint
   acquisition behind the existing reduced-forward numerical gate.
3. **Agent owner: CLI lane.** Evaluate more than one read task, including longer
   tool results and history growth. The 2048-token path passed repeated long
   requests and a read-tool task with an 800-word result. Shell and write tools
   need a separate execution policy; this delivery does not execute them.
4. **Serving owner: Responses lane.** Add request deadlines and qualify longer
   context beyond the qualified 2048-token control, custom-tool requirements,
   cancellation, and real Codex task completion. Only then activate a personal Codex profile. The documentation's
   profile example is a qualification target; no live profile was changed.
5. **Runtime owner: comparison lane.** Run matched quality/tool/performance
   workloads on an independently owned current local runtime before deciding
   whether a new native model adapter is worth its implementation cost.

## Validation

The canonical checks are `RUSTC_WRAPPER= uv run scripts/check.py` and
`RUSTC_WRAPPER= uv run scripts/check.py --metal`. The wrapper override avoids a
local sccache startup failure and does not modify global compiler settings.
Both checks passed on the final implementation, followed by a release build
and the live HTTP checks above. The owned test server was stopped after
validation; use the README command to start a new one.
