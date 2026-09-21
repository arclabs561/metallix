# Changelog

User-visible changes and implementation milestones. Unreleased work is not a
published release; reference operators do not imply full-model support.

## Unreleased

### Added

- Native `mx inspect-v41-index` now recognizes the real MLX/Hugging Face
  weight-map index format when `metadata.total_size` is absent, reporting its
  2,757 tensors and 18 shards while keeping the scope explicitly metadata-only.
- Added `mx inspect-v41-shard` to validate a real DeepSeek/MLX safetensors
  header with bounded I/O before any tensor payload is read.
- Added the first native MLX affine-row decoder for packed 8-bit groups with
  BF16 scales and biases, including malformed-layout and non-finite guards.
- Added `mx inspect-v41-embedding-row`, which decodes a real 4096-wide
  DeepSeek MLX embedding row from a shard and emits a deterministic checksum.
- The all-features embedding-row command now evaluates that decoded row on
  Metal, proving the first real MLX tensor crosses the native device boundary.
- The same native gate now decodes layer-zero attention `wq_a` rows (3,072
  logical values) and evaluates them on Metal.
- The row gate can now decode all 1,024 layer-zero `wq_a` rows as a bounded
  3,072-wide affine tensor and evaluate the assembled matrix on Metal.
- The attempted embedding-to-`wq_a` application now reports the real latent
  width mismatch explicitly: embeddings are 4,096-wide while this projection
  consumes a 3,072-wide pre-attention latent.
- Added native decoding and Metal evaluation for the real layer-zero
  hyper-connection matrix (`24 × 16384`, checksum `6150937c7aee9697`).
- Added the bounded HC input expansion primitive for repeating a hidden state
  across the model's four-way hyper-connection layout.
- Added the checked HC coefficient mix bridge: RMS-normalized expanded hidden
  state, `attn_hc.fn` row products, and existing Sinkhorn coefficient splitting.
- The native CLI now binds that bridge to the real token-0 embedding and
  layer-zero HC parameters, producing four-copy coefficients from the local
  checkpoint.
- The real HC mix gate now collapses those coefficients back to a 4,096-wide
  hidden stream, checksum `b5e5dc3e838c352b`, before the remaining latent path.
- Corrected MLX attention quantization to contiguous 6-bit/group-128 packing;
  the full real `wq_a` tensor now applies to the 4,096-wide embedding and
  evaluates on Metal, producing a 1,024-wide output.
- Added a bounded first-head `wq_b` gate: 512 output rows × 1,024 Q-A inputs,
  decoded from the real 6-bit shard and evaluated on Metal.
- The native Q chain now runs token-0 embedding → full `wq_a` → real BF16
  `q_norm` → first 512-row `wq_b` head on Metal, checksum `9caf5173ef2cfb1c`.
- The same native chain now decodes and evaluates all 64 Q-B heads (`32768`
  outputs), checksum `43a0197452b33b03`.
- Added the source-accurate unweighted per-head Q-B RMS normalization stage;
  the normalized 32,768-output checksum is `dfa3530172fc18ba`.
- Q-B head loading now reuses one shard handle per contiguous row range,
  avoiding one file open/close cycle per head during the native gate.
- Added the native Metal matrix-projection primitive that applies decoded
  affine weights to a hidden-state vector, with shape checks and a device test.
- Documented and locally validated a machine-local oMLX DeepSeek-V4.1-Flash
  Codex profile path, including the direct API smoke result and the measured
  full-Codex prefill limitation.
- Added a typed engine SMC primitive with finite log weights, ESS, deterministic
  systematic resampling, absorbing particles, and fail-closed input checks.
- Added a test-only MLX LoRA micrograph for Qwen: frozen base ownership,
  `value_and_grad`, independent gradient checks, one SGD update, adapter-only
  safetensors export, and exact reload output parity. No production training API
  is implied yet.
- Qwen benchmark receipts now report decode tokens per second per run and in
  aggregate, alongside the existing millisecond measurements.
- A source-grounded DeepSeek layer-zero token-embedding fixture now rebuilds
  exact BF16 rows for the captured prefill/decode calls and rejects changed or
  out-of-vocabulary token IDs.
- DeepSeek's test-only reduced forward now qualifies Engram1 output into the
  native layer-one attention/HC/FFN path and onward to the existing layer-two
  suffix, with exact output hashes and corrupted-entry rejection. The Codex
  qualifier also covers an ordered two-command pointer chain while retaining
  its single-file control.
- Resident Qwen chat now uses bounded stepped K/V storage with valid-prefix
  attention and whole-request reset on append failure. The 2048-token real
  server gate preserved exact output/token parity for Qwen3-0.6B and 4B; local
  three-process measurements reduced 4B decode median from 3796.7 to 3600.3 ms.
- A source-only DeepSeek V4.1 layer-one Engram fixture pins stream/hash shape,
  layer binding, encoded parameter finiteness, and observer restoration with
  negative controls for layer substitution and tampering.
- The reduced DeepSeek continuation now starts at the native layer-two FFN,
  carrying its residual and HC pre-mix through Engram3 and layers three/four
  to final logits. Source-derived arithmetic bounds and mutated handoff tests
  preserve the distinction from complete model execution.
- Resident chat, agent, and Responses commands accept an explicit logical K/V
  budget and an experimental context ceiling of 16,384 tokens. Defaults remain
  2048 tokens and 512 MiB. Loading and request admission share the same budget.
- Sharded Qwen reference receipts bind the checkpoint index and exact shard set.
  Qwen3-4B-Instruct-2507 passed full-logit and cached-forward comparisons and
  all 12 bounded native read-tool trials.
- A reproducible Codex qualification runner checks native command execution,
  hidden fixture values, event ordering, terminal completion, and workspace
  preservation. Three real client trials passed against the 4B Responses
  endpoint; general coding and custom grammar tools remain unqualified.
- Third-party notices explicitly inventory Qwen numerical extracts and newer
  synthetic DeepSeek captures alongside the retained Apache-2.0 and MIT terms.
- Profile-guided BF16 projection row iteration preserves scalar arithmetic and
  error semantics while improving the measured captured owner workload.
  Property tests cover row splitting, output permutation, and late-overflow
  atomicity.
- Agent qualification rejects internally inconsistent completed receipts,
  malformed generation metrics, and tool executions attributed to truncated turns.
- Native DeepSeek layer-three terminal state now feeds layer four through final
  logits. Exact BF16 handoff checks preserve the existing arithmetic bounds,
  and corrupted incoming coefficients fail before attention execution.
- A bounded scalar DeepSeek Engram lookup decodes selected supplied FP8 rows
  with row-local E8M0 scales into BF16 and maps masked or out-of-table IDs to
  zero rows. A source fixture records the layer-three hash, embedding, WKV,
  residual-gate, and block-entry boundary. The focused native continuation
  carries that entry through layer three, layer four, and the final logits.
- The source-grounded DeepSeek layer-three continuation reaches layer-four
  entry through native HC and FFN. Fixed propagated bounds check coefficient
  rounding and reject an omitted-attention control through the same boundary.
- `mx agent --json` reports execution status, final text, per-turn generation
  metrics, and executed tool evidence without raw tool-result payloads. The
  qualifier checks exact tool paths and argument hashes separately from answers.
- Workspace tool properties cover pathname replacement after opening the root
  and listing boundaries around 128 entries.

- A source-grounded DeepSeek layer-three continuation joins native HC/RMSNorm,
  compressed owner KV, producer selection, and attention output. It retains
  layer-specific rotary provenance and property checks for publication identity
  and selection sensitivity.
- Responses generation has a configurable cooperative wall-clock budget checked
  around prefill and each decode, including tokens that emit no visible text.
  Timeout failures retain distinct JSON/SSE error semantics.
- A read-tool qualification runner checks task answers and observed tool calls
  independently of process exit status. Its synthetic multi-file task exposes
  unfinished work from the current small control model.

- Native layer-three HC pre-mix and RMSNorm now feed DeepSeek owner and
  candidate inputs in the connected suffix tests, using a new pinned source
  capture and exact cross-checks against the preserved historical fixtures.

- Loopback Responses intake now owns its sockets, bounds header/body reads by
  one absolute deadline, and limits response writes. Malformed framing is
  rejected before generation; stalled clients no longer leave handler threads.
- Function-call history rejects reused IDs even after prior results complete.
  Properties cover reordered results, SSE payload framing, and fragmented HTTP.
- A source-checked DeepSeek owner-transaction benchmark separates prepare,
  discard, commit, and direct-forward costs at captured prefix lengths.

- Resident context and tool-call property tests cover sizing overflow, budget
  boundaries, malformed batches, nested JSON, and Unicode. JSON delimiters
  inside tool arguments no longer terminate an envelope early.
- A repeatable chat qualification runner checks CLI/HTTP output agreement,
  context accounting, short-request recovery, timings, and optional process CPU
  sampling.
- A reduced DeepSeek owner-to-logits test connects native compressed KV and
  selection through the final block and head, with fixed source-derived
  HC/RMSNorm error bounds. Upstream activations remain captured inputs.
- Experimental native Qwen chat, a bounded workspace-read agent, and a loopback
  text/function Responses endpoint. Sessions retain weights, use checkpoint
  chat templates, and report load/prefill/decode timings. Resident chat supports
  up to 2048 tokens with a checked logical KV budget; diagnostics retain their
  512-token limit.
- DeepSeek compressed-owner prepare/commit transactions expose staged key/KV
  prefixes to downstream consumers without publishing them on rejection.
- Responses benchmark receipts distinguish text bytes, model tokens, first-token
  latency, completion status, and sample variability.
- A single-batch DeepSeek scored-query adapter composes query preparation and
  BF16 score reduction over a distinct borrowed index-key view. It checks the
  aggregate scoring workload before preparing queries, rather than relying on
  per-row limits. Source tests now use this production orchestration for both
  producer and consumer scores.
- A stateless DeepSeek selection adapter now applies causal masks, produces
  opaque candidate results and selects indices from BF16 scores. Validated
  geometry and publication/batch metadata prevent accidental cross-call reuse.
  The source-oracle owner-to-attention test now calls this adapter instead of
  test-local masking and selection loops. Query scoring and cache ownership
  remain separate.
- The DeepSeek owner-to-attention source test now derives both producer and
  consumer QR natively. Captured QR is an exact expected result rather than a
  scoring input; layer activations and full-model orchestration remain outside
  this test's scope.
- Stateless DeepSeek candidate-query preparation now derives QR from input
  activations through the same private FP8 projection/RMSNorm helper as attention.
  Validated candidate layouts keep query dimensions consistent without requiring
  a KV publication or advancing attention state. Source tests compare both QR
  stages before continuing through candidates, selection and attention.
- Native DeepSeek candidate queries, BF16 scores and block masks now compose
  with owner-produced keys, layer-four selection and native-KV attention in
  source-oracle tests. Prefill and two decode calls match exact captured stages;
  removing a selected candidate changes selection and attention output.
  Layer-three QR and input activations remain captured boundaries, not full-model
  execution.
- Source capture now distinguishes DeepSeek candidate producers and consumers
  with explicit roles and quantization phases. The new candidate fixture keeps
  query/key stages and prefill/decode masking distinct. Real-source tests verify
  observation leaves outputs unchanged and restores hooks after failure.
  Observation integrity is checked separately from native numerical parity.
- A ratio-one DeepSeek owner now publishes separate index-key and compressed-KV
  caches atomically with compressor progress. Prepared appends validate both
  caches before either commits, without cloning capacity-sized buffers.
  Native KV prefixes drive the captured attention sequence and match the source
  byte-for-byte; wrong-format and wrong-position controls guard the comparison.
  The existing key-only API remains available. Full-model execution is still open.
- Atomic owner output now drives the V4.1 selection-to-attention capture test:
  raw owner input → compressor → prepared key cache → scores → selected indices
  → attention. Exact stage checks connect the supplementary compressor capture
  to the older attention oracle. Candidate masks remain supplied; compressed KV
  now comes from the coupled native owner.
- Atomic ratio-one index-key owner calls compose BF16 compressor projection,
  normalization, key preparation and cache append. Failed key preparation does
  not advance compressor or cache state; explicit reset starts a new epoch.
  Captured-stage comparisons and randomized retry tests cover the adapter.
- Native ratio-one owner compressor projection and RMSNorm match exact
  source-captured BF16 stages and now feed the owner-key/cache test. A
  supplementary capture preserves the earlier attention oracle's identity;
  full-model transactions and generation remain open.
- Bounded, per-request V4.1 index-key cache ownership with atomic prepared-key
  append/reset, source/epoch/call checks and borrowed per-batch prefixes.
  Source-capture tests join native owner-key preparation and cache retention
  to index selection and attention. Owner-layer inputs, candidates and
  compressed KV remain source-supplied; full-model generation is still open.
- Native V4.1 owner-key preparation matches the reduced source-forward
  key-cache append regions exactly at starts 0, 5 and 6. An offline extractor
  retains the actual indexer weights and pre-mutation compressor latents;
  corruption, wrong-layer weight and wrong-rotary-position controls guard the
  comparison. Released-checkpoint execution remains open.
- Stateless V4.1 index-key preparation from the original compressor latent:
  bounded BF16 projection, RMSNorm, rotary and G32/E8M0 FP4 reconstruction.
  Hand-staged numerical checks and batch/position properties cover the API;
  the existing premature-latent-rotation control now calls it. Owner-layer
  checkpoint parity remains open.
- Native V4.1 index selection now feeds the layer-attention integration test
  across captured prefill and decode calls, with exact attention-boundary
  comparisons. QR, owner-layer inputs, candidates and compressed KV remain
  source-supplied; the test does not execute the full model.
- DeepSeek-local BF16 index-score reference API with explicit dot, rectified,
  weighted and final-score boundaries. It preserves negative zero through
  ReLU, checks shape/work limits before allocation, and rejects FP32 and BF16
  narrowing overflow. Exact source-fixture selection checks now call this
  implementation; key-position permutation properties exercise its layouts.
  Existing FP32 APIs are unchanged. This is scalar staging, not GPU execution.
- Source-fixture composition of native V4.1 queries, BF16 score stages,
  causal/candidate masks and final index selection across prefill and decode.
  Every captured score stage and selected index matches exactly. Shared keys
  and candidates remain source-supplied; this is not a complete indexer.
  Masked-score previews are checked against their exact storage, including
  invalid-JSON and wrong-sign negative controls.
- Native V4.1 source-fixture index-query preparation through FP4-reconstructed
  rotary queries and projected, scaled BF16 head weights at prefill/decode
  starts 0, 5 and 6. The source observer captures the projection boundary and
  serializes nonfinite preview values as strict-JSON strings while retaining
  exact storage bytes. Shared index cache publication and complete reindexing
  remain open; this is not full-model execution.
- Joined native V4.1 attention, HC residual mixing and FFN comparison. Native
  HC/normalization input feeds one cache-continuous attention sequence, whose
  output now feeds the existing numerical-envelope checks. Capture-identity,
  call-order and discarded-output controls guard the join. Upstream block
  inputs, incoming coefficients, compressed KV and selected indices are still
  source-supplied; this is not full-model execution.
- Native V4.1 layer-four attention composition from encoded projections through
  query normalization, rotary tails, the window ring, sparse attention and
  output projection. Captured prefill and decode boundaries match BF16 storage
  and window indices exactly. Compressed KV and selected compressed indices
  remain source-supplied; native reindexing and full-model execution are open.
  Publication checks reject stale identity, wrong prefix lengths, phantom
  empty-prefix slots and duplicate selected positions without committing state.
- Native V4.1 FFN composition joining HC coefficient projection, incoming
  residual mixing, RMSNorm, routed/shared experts and HC post-mixing.
  Synthetic source comparisons use input-derived arithmetic envelopes and
  explicit BF16 rounding cells, with exact observed MoE checkpoints and
  wrong-handoff/omitted-attention controls. Attention output remains
  source-provided; this is not complete native block or model execution.
- Bounded native V4.1 MoE sublayer combining BF16 gate routing, packed FP4
  routed experts, the FP8 shared expert and source-order FP32 accumulation.
  A source-forward fixture checks exact BF16 outputs at seven positions;
  inputs are still source-provided, so this is not full-model execution.
- Shrinking property tests for MoE expert renaming and SwiGLU up-sign
  symmetry, plus BF16 finite-limit rounding regressions.
- CPU V4.1 index-score reference with bounded scalar work, explicit FP32
  reduction order and overflow rejection. Tests join scoring to causal
  masking, CSA2 candidates and final selection without requiring Metal.
  Projection, quantization and cache ownership remain caller responsibilities.
- Reduced V4.1 CPU source-forward harness with an explicit shape/schedule
  manifest, hash-checked source loading and independent quantized-kernel
  replacements. Synthetic prefill/decode captures include final logits and
  candidate-filter checks; this is not native Rust full-model execution or
  upstream GPU parity.
- Native FP32 output-head comparison using hidden states and encoded weights
  from the complete synthetic V4.1 source capture, with a fixed dot-product
  error bound and wrong-position/vocabulary-order negative controls.
- Native final HC collapse and RMSNorm joined to that output head. All seven
  captured prefill/decode positions match the source's BF16 intermediate bits;
  final-block state and coefficients remain source-provided.
- Bounded FP32 linear reference with atomic output writes and no BF16
  narrowing. Compressor fixture tests now execute BF16/FP32 projection
  matrices before native pooling and normalization.
- Model-local scalar compressor pooling and partial-group state with separate
  BF16 ratio-one and FP32 gated inputs. Learned projection results are supplied
  by the caller; cache publication and scheduling remain separate.
- Native compressor prefill and singleton completion joined with index-key
  preparation, Metal selection and mathematical attention in a batch-one
  composition test, retaining the premature-latent-rotation counterexample.
- Text prompts for `mx gen --prompt`, with decoded generated text in its JSON
  report. Raw `--input-ids` remains available for numerical diagnostics.
  Prompts are encoded as plain text, without a chat template.
- Inline constraints through `--json-schema-inline`, alongside schema files.
- Bounded V4.1 Engram hash-state and preprojected BF16 residual-gate references,
  checked against captures from pinned upstream code. Gate tests cover
  per-copy normalization, shared values, signed-zero masking, multiplication
  order, and failure without partial output writes.
- `just check`, `just check-metal`, and `just check-fixtures` development
  commands. The Engram fixture integrity checker also runs in the canonical
  quality gate.
- Source-captured V4.1 compressor and two-block sequencing tests, using
  explicit projection/sublayer stubs. Compressor checks cover partial groups,
  sequential decode, reset and exact BF16 output; block checks cover the
  attention-to-FFN and inter-block pre-mix handoff.
- Offline particle ancestry tests for absorbing EOS, resampling, effective
  sample size and weighted final selection through the categorical sampler.
  These do not provide model-backed SMC or physical cache branching.
- Pinned V4.1 compressed-cache publication traces covering indexer visibility
  before mutation, post-write cache reads, partial groups and shared consumers.
  These check source control flow with explicit numerical stubs, not a native
  Rust attention implementation.
- Library FP4 activation reconstruction through
  `deepseek::precision::requantize_bf16_activations_e2m1`, with typed
  compressed-KV E4M3 and indexer E8M0 modes. The former test-only implementation
  now serves the oracle and composition tests through the public API, using
  one fallible staging allocation and no diagnostic trace arrays.
  E2M1 rounding remains a software assumption, not qualified GPU behavior.
- Compressed-only attention composition tests joining rotary, FP4
  reconstruction and mathematical sparse attention with supplied latents
  and indices. Intermediate BF16 values and final attention are checked
  against an independent CPU oracle.
- Metal-feature composition of FP4-prepared index operands, GPU index scores,
  final selection and compressed-vector attention, checked against closed-form
  scores and output. That test supplies post-projection index operands.
- Synthetic-weight index-key projection and RMSNorm joined with rotary,
  FP4 preparation and Metal selection. A numerical ordering check detects
  reading compressed latents after attention has rotated them.
- Synthetic BF16/FP8 index-query projection and signed BF16 head-weight projection
  joined with rotary, FP4 reconstruction and Metal scoring. Closed-form checks
  distinguish omitted rotation, early quantization and omitted head scaling;
  FP8 intermediate checks expose skipped activation quantization and shared
  output-block scales. Checkpoint loading and GPU GEMM parity remain open.

### Fixed

- The Engram source exporter checks numeric storage for NaN and infinity,
  including receipts with a consistent checksum and an incorrect finite flag.
- The owner transaction benchmark accepts Cargo's injected `--bench` flag,
  so its documented default, scaling, and profiling commands run through
  `cargo bench` as well as directly.
- Index-key rotary narrowing errors report the full key-buffer element, not
  the offset within one rotary tail. A late-row overflow regression covers it.
- New Metal composition tests serialize process-global MLX device use,
  matching the library tests' guard convention.
- Qwen checkpoint discovery accepts Hugging Face cache shard symlinks that
  resolve to regular files, while rejecting broken and non-file targets.
- Generation tokenization disables saved tokenizer padding and truncation,
  and checks tokenizer IDs against the model's vocabulary width.

### Current limits

- Full DeepSeek-V4.1-Flash generation is not implemented. The Engram scalar
  reference decodes only selected rows of a supplied bounded, unsharded table.
  Checkpoint table loading and sharding, full-model qualification, and Metal
  execution remain separate work.
- The new Engram references are scalar correctness tools, not performance
  improvements or evidence of larger-than-memory serving.
