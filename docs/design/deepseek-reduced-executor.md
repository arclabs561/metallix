# DeepSeek reduced executor extraction

Status: implementing in bounded slices. Governing sequence:
[delivery roadmap](../delivery-roadmap.md). Consumer: a runnable scalar reduced
DeepSeek request accepting token IDs and supplied synthetic weights.

The canonical and alternate startup-to-head tests establish numerical evidence.
Their fixture readers, captured intermediate assertions and numerical envelopes
remain test infrastructure. They cannot be imported into the executable path.

Options considered:

- Wrap the test harness: quick to invoke, but still reads expected intermediate
  records and does not expose ordinary execution. Rejected as the runtime.
- Extract DeepSeek-owned arithmetic and state composition into `src/reduced.rs`:
  retain the independent fixture checks while giving the runner typed weights,
  input values and fallible results. Selected.
- Introduce a shared multi-model executor interface: defer until Qwen and
  DeepSeek demonstrate a shared execution contract.

The first slice is `FinalHead`: supplied BF16 final residual, incoming HC
coefficients, norm weights and F32 output weights produce final logits. It has
no fixture parser, source identity, expected output, or numerical acceptance
bounds. Both existing compositions must consume this implementation and retain
their unchanged independent source comparisons. Malformed inputs and excessive
shapes must fail before unbounded allocation.

The next extracted components are `BlockTailReference` (attention HC post-mix
and FFN) and `EngramSession` (persistent hashing, embedding, projection and
residual gating). Engram stages hash history and commits it only after every
numerical stage succeeds; reset reconstructs pristine history. Fixture decoding
and every source-stage comparison remain test-owned.

`AttentionInput` now owns incoming HC collapse and RMSNorm. `StartupSession`
composes embedding, attention preparation, window-only attention and the first
block tail over caller token IDs. It retains the window between calls and
invalidates itself after any admitted-call failure; reset clears the window
and cursor. Both established schedules retain their independent source checks.

The remaining assembly owns L1/L2/L3/L4 request state with the preceding-call
L3 publication. `RatioTwoCompressedOwner` now stages the L1 compressor and
paired key/KV prefixes over typed per-call weights; its partial calls retain
incomplete groups. `CandidateProjector` now derives L3 masks from supplied
input and key scores, checking matrix geometry before scoring. These components
remove the test-owned owner/projection arithmetic. `LayerThreeSession` now
joins ratio-one owner preparation, candidate scoring/selection, owner commit
and attention. It derives publication identity and selection offsets from live
state. Any admitted failure poisons the session; reset clears both owners and
advances their epoch together. This does not promise rollback after the owner
commits and attention fails. `LayerOneSession` similarly joins ratio-two
ownership, query scoring, direct causal selection and attention. Partial groups
require the preceding L3 publication with matching source, epoch and call ordinal,
and consume its leading completed-group rows while keeping L1 KV. The session
derives token/group frequency positions and publication identity from live state.
`LayerFourSession` now consumes the actual committed L3 candidate mask, keys
and KV, checking batch, ratio, key count, window offset and publication identity.
`RequestModel` and `RequestSession` compose the fixed five-block path over ordinary
operands. Each call reaches the final head before the next call starts. A failed
admitted call poisons the whole request; restart reconstructs every owner and
Engram history together, dropping prior publications. Startup retains its own
rotary table: source qualification exposed that L0 and L1–L4 tables differ after
position zero. Both established schedules pass their final-logit source bounds
and replay identically after restart. A malformed L4 weight exercises late
failure after the earlier owners have advanced.
`ReducedArtifact` now admits a self-contained JSON numerical artifact, with
explicit tensor dtype, shape, little-endian storage and SHA-256. It checks the
exact operand inventory, finite floating-point values, and bounded dimensions
before assembling the model. Encoded input is limited to 64 MiB and decoded
tensors to 32 MiB. Runtime code never reads captured cases or expected outputs.
The exporter alone extracts immutable numerical operands from the existing
source bundle; it rejects conflicting projections and keeps L0 rotary separate.

The server-owned `mx run-deepseek-reduced` command accepts this artifact and
supplied token IDs, executes the scalar request, and returns per-call final-token
logit bits plus an artifact hash. It is available without Metal. Both established
schedules pass the unchanged source bounds through exported artifacts, while
CLI integration checks compare actual JSON output and fail-closed admission.

Options for artifact transport were self-contained JSON or metadata plus a
separate tensor file. Bounded JSON is selected for this small scalar milestone:
it gives one hashable input without introducing a checkpoint payload loader.
This is not the eventual checkpoint format. Broader schedules, Metal,
checkpoint loading and serving retain the roadmap's separate gates. No generic
backend abstraction or token-generation claim follows from this executable.
