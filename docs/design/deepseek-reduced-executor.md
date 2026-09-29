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

Subsequent slices extract HC/block and Engram orchestration, then assemble the
request-owned L1/L2/L3/L4 state with the preceding-call L3 publication. Weights
must be separated from the source cases; synthetic layouts remain explicitly
bounded. Both established schedules must work. A late failure invalidates the
request, and restart reconstructs all mutable state from immutable weights.
No component-local rollback claim is sufficient for the entire request.

Only after that library path returns independently checked logits should a CLI
accept supplied synthetic weights and token IDs. A head-only call is a component
diagnostic, not token generation. Metal, checkpoint loading and serving follow
the roadmap's separate gates. Review this extraction after the first executable
stateful request; do not grow this module into a generic backend abstraction.
