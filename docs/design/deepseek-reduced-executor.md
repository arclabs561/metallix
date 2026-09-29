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
Weights must be separated from the source cases; synthetic layouts remain explicitly
bounded. Both established schedules must work. A late failure invalidates the
request, and restart reconstructs all mutable state from immutable weights.
No component-local rollback claim is sufficient for the entire request.

The library path now returns independently checked logits. The next slice is a
CLI accepting supplied synthetic weights and token IDs. A head-only call is a component
diagnostic, not token generation. Metal, checkpoint loading and serving follow
the roadmap's separate gates. Review this extraction after the first executable
stateful request; do not grow this module into a generic backend abstraction.
