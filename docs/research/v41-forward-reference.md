# Reduced V4.1 CPU reference

This development harness connects the pinned source text graph to independent
CPU numerical kernels. Synthetic weights exercise implementation mechanics;
they do not produce a useful pretrained model. A successful capture is not
Rust agreement, upstream GPU parity, or checkpoint support.

## Run it

With the pinned source files already retained under `artifacts/`:

```sh
uv run scripts/v41_forward_manifest.py --require-execution-ready
uv run scripts/v41-forward-reference.py --output artifacts/v41-forward-reference.json
uv run --with torch==2.13.0 --with numpy==2.5.3 scripts/test_v41_cpu_kernels.py
```

The manifest command validates shapes and the call schedule without allocating
tensors. The capture command runs the source graph with synthetic weights and
writes encoded parameters, intermediate outputs, caches and final logits.
The runner pins its Python dependencies and rejects source-hash mismatches.
Neither command downloads model weights.

The default `just check` runs the dependency-free manifest, source-loader and
attention-fixture integrity tests. Numerical kernel tests require Torch and
are run explicitly above.
The loader's source-body comparison is skipped if retained source artifacts
are absent; that skip is not source verification.

## Boundaries and provenance

The source loader verifies SHA-256 before executing retained `model.py` and
`engram.py`. It replaces the six kernel imports and supplies the real Engram
classes; unused vision imports are excluded. Model class and function bodies
remain unchanged, including the final head and token-hash history.

- [Pinned model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
- [Pinned numerical kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py)
- [Pinned Engram source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/engram.py)

The CPU backend retains quantization encodings, scale placement and explicit
BF16 boundaries. Torch CPU dot products are not the original GPU reduction
sequence. Captures must identify the numerical backend, source revision,
synthetic manifest and initialized tensor encodings.

## Why this fixture is not just smaller defaults

The source defaults disable Engram and candidate filtering. The manifest
instead enables both, with window-only attention, two compression ratios,
source/consumer cache sharing and nontrivial HC and expert routing.

Candidate block selection always includes the newest block. Keeping just one
block would remove score-driven choice. Asking for more final indices than
retained candidates can also admit a masked position: the pinned indexer's
final check tests causal position, not the candidate mask. The reduced
schedule therefore retains two single-position blocks and selects one final
index, with calls exposing more than two positions.

FP4 has distinct roles: compressed KV uses groups of 16 with E4M3 scales;
index query/key reconstruction uses groups of 32 with E8M0 scales; packed
expert weights use their separate group-32 representation. One generic
“4-bit” setting would erase these distinctions.

## Acceptance remains separate

The capture must expose actual mechanism coverage and complete final logits
before it can become an oracle fixture. Integer decisions and encoded storage
need exact checks; floating comparisons require a policy fixed before running
the Rust candidate. Retained source files and generated captures are local
artifacts, not fetched automatically by the ordinary quality gate.

HC kernel calls are observed without changing the source graph: ten calls per
forward record their inputs and pre/post/combination coefficients. Finite
logits, candidate-mask checks and these recorded boundaries are a
source-execution milestone, not acceptance of the full Rust comparison gate.

## First Rust consumer: the output head

```sh
uv run scripts/v41-forward-reference.py --output artifacts/v41-forward-reference.json \
  --head-fixture-output fixtures/deepseek-v41/forward-head-reference.json
cargo test -p deepseek --test forward_head
```

The small checked-in subset contains exact BF16 final-normalization outputs,
FP32 head weights and FP32 logits from all three source calls. Rust executes
`fp32_linear_reference` using the last prefill position, then both decode
positions. Negative controls select the wrong prefill position or reverse
vocabulary rows; neither may satisfy the comparison.

The tolerance is fixed before candidate execution: the sum of two FP32
dot-product roundoff bounds, each `gamma(2K) * sum(abs(x*w))`, with
`gamma(n) = n*u/(1-n*u)` and `u = 2^-24`. The bound is evaluated in FP64 with
a guard for that evaluation's own roundoff. This is input-dependent absolute
error, not a blanket relative tolerance near cancellation. Finite normal
arithmetic without underflow/overflow is required. The test qualifies this
head boundary only.

The tail-composition test now starts earlier, from the captured final-block
residual copies and returned pre-mix coefficients. Native HC collapse and
RMSNorm must match the source BF16 storage exactly at all seven positions;
the native normalized last row then feeds the output head under the same
unchanged dot-product bound. The collapsed reference is observed by a pre-hook
on the source's final norm, not recomputed by the fixture exporter. Controls
reject passing through one residual copy and omitting normalization.

Final-block execution and derivation of its pre-mix coefficients still come
from the source graph; this qualifies the native tail, not the full Rust model.

## Native MoE sublayer

```sh
uv run scripts/v41-forward-reference.py --output artifacts/v41-forward-reference.json \
  --head-fixture-output fixtures/deepseek-v41/forward-head-reference.json \
  --moe-fixture-output fixtures/deepseek-v41/forward-moe-reference.json
cargo test -p deepseek --test forward_moe
```

The MoE subset retains layer 4's source input, actual gate decisions and output,
plus its encoded gate, four FP4 routed experts and one FP8 shared expert.
`deepseek::moe::MoEReference` computes routing and executes the selected experts
from those weights. All seven captured positions must match the final BF16
output exactly. Selected expert IDs must match exactly; route weights are
compared by ID with a fixed absolute tolerance of `2^-20`. That tolerance is
a diagnostic policy, not a proved bound on transcendental implementations.

Execution preserves the source's BF16 projection boundaries and applies route
weights after SwiGLU but before the hidden BF16 cast and W2. Routed expert
outputs accumulate in ascending expert-ID order in FP32; the unweighted shared
expert contributes once before the final BF16 cast. A source-fixture negative
control zeros the shared output projection and must fail output agreement.
Separate analytic tests cover nonzero clamps and route-weight placement; this
source manifest has its SwiGLU clamp disabled.

Two bounded property tests add coverage beyond the source trace: flipping the
up branch's sign must flip SwiGLU's sign, and renaming two routed experts
together with their gate rows and biases must preserve their combined output.
Each runs 64 generated cases with shrinking. The permutation property is
deliberately limited to two selected experts; it does not assume arbitrary
FP32 summation order is invariant. Run the properties and analytic tests with
`cargo test -p deepseek moe:: --lib`.

The API is a bounded scalar diagnostic over runtime-encoded weight views, not
a checkpoint loader, scheduler, or optimized serving path. MoE input still
comes from the source block. Attention, cache ownership and complete block
composition remain necessary before native full-graph parity can be claimed.

## Native FFN composition and numerical contract

`deepseek::ffn::FfnSublayerReference` joins HC projection, incoming pre-mixing,
RMSNorm, the native MoE and HC post-mixing. Fresh FFN coefficients come from
the residual; the attention sublayer's incoming coefficients select the FFN
input. A 64-case shrinking property test varies positive incoming weights
and checks that this handoff affects execution without changing the freshly
derived coefficients.

The extended source fixture includes layer-four block inputs, attention
outputs, raw HC projection mixes, split coefficients and intermediate BF16
residuals. These are observations from the pinned source graph, not values
recomputed by the exporter. Repeated captures reproduce both fixture files
byte for byte.

The original block-tail exact-storage comparison found a rounding crossing.
With captured source
coefficients, native HC pre/post mixing reproduces all seven captured
positions exactly. Native coefficient derivation instead changes element 117
of the attention residual at decode start 6 from BF16 bits `16600` to `16599`.
Using the captured pre-mix afterward retains this discrepancy, so the first
BF16 divergence precedes the FFN. Source raw mixes also expose FP32-bit
differences in scalar coefficient splitting before projection is involved.
The coefficient diagnostics do not claim exact transcendental parity. The
source-fed exact checks remain; native coefficient propagation now uses the
explicit arithmetic-envelope contract below. This is neither native attention
execution nor full-block qualification.

A follow-up isolation separates benign coefficient differences from the
observed rounding crossing. At decode start 6, splitting the captured raw
mixes produces one differing combination coefficient but still reproduces
the BF16 residual with captured post coefficients. The combination matrix
derived through native projection differs in three entries and reproduces
the element-117 failure even with captured post coefficients. Conversely,
scalar-split post coefficients with the captured combination matrix produce
the exact residual. Thus the extra projection-derived combination drift is
necessary for this observed crossing; sigmoid drift alone is not its cause.

The numerical contract distinguishes source CPU bits from portable arithmetic.
The scalar projection declares ascending FP32 reductions, whereas the source
graph uses Torch `F.linear` and `rsqrt`. Even the CPU replacement's
`torch.softmax` uses reciprocal multiplication in
[PyTorch 2.13.0](https://github.com/pytorch/pytorch/blob/v2.13.0/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L62),
while the retained DeepSeek HC kernel expresses division by the row sum.
Matching a particular Torch CPU execution bitwise and validating independent
source-mathematical implementations are different acceptance targets. The
oracle arithmetic remains unchanged. Native composition is now checked using
input-derived arithmetic envelopes rather than requiring Torch CPU bit identity.
The fixture retains its original exact-storage policy fields as capture
provenance; those fields are not the acceptance thresholds for native-derived
HC propagation. The native policy is implemented in the test's `support/`
interval helpers, while source-fed HC replay continues to require exact bits.

### What the new comparison establishes

`tests/forward_moe.rs` derives independent intervals from the encoded weights,
input magnitudes and operation counts. It propagates HC projection and
coefficient uncertainty through attention post-mixing, FFN collapse and
RMSNorm, checking whether each source and native BF16 value's round-to-nearest,
ties-to-even cell intersects the permitted interval. It propagates the full
BF16 rounded enclosure, not just the two observed values. Signed zero,
subnormals and midpoint parity have dedicated tests.

The target model uses FP32 unit roundoff `2^-24`, reduction bounds based on
`gamma(2n)`, and absolute gradual-underflow terms. It rejects overflow-capable
reductions, including a signed dot whose absolute total could overflow under
regrouping even when ascending partial sums remain finite. Normalization uses
a correctly rounded square root followed by a rounded reciprocal. Coefficient
splitting declares an exponential relative-error assumption of at most `2u`
over its bounded domain; outward interval evaluation includes a Taylor
remainder and the softmax/Sinkhorn operation sequence.

These are explicit target-arithmetic assumptions, **not proofs of Torch's
`rsqrt`, platform exponentials, fast-math or GPU accuracy**. Both observed
executions must fit the same independently derived envelopes. No measured
error is used to set their widths.

At the MoE boundary, native/source selected expert IDs and observed BF16
outputs must still agree exactly. Only then does that observed output enter
the final HC post-mix comparison. This qualifies the paired captured
executions; it is not a theorem that every input inside the propagated
interval produces the same routes or MoE output. Wrong-HC-handoff and
omitted-attention controls must fail the contract. Full-model composition and
native attention remain separate gates.

Run the boundary probes with:

```sh
cargo test -p deepseek --test forward_moe -- --nocapture
cargo test -p deepseek ffn:: --lib
```

## Layer-four attention capture and native composition

```sh
uv run scripts/v41-forward-reference.py --output artifacts/v41-forward-reference.json \
  --head-fixture-output fixtures/deepseek-v41/forward-head-reference.json \
  --moe-fixture-output fixtures/deepseek-v41/forward-moe-reference.json \
  --attention-fixture-output fixtures/deepseek-v41/forward-attention-reference.json
cargo test -p deepseek --test forward_attention
```

The attention subset records the actual layer-four input, FP8 projection
weights and scales, BF16 normalization weights, rotary frequencies, query
stages, window-ring state, compressed numerical KV, selected indices, sparse
attention output and output projections. Hooks copy observed tensors before
subsequent in-place operations; the exporter does not recompute intermediate
answers. A repeated capture reproduced the full capture and attention subset
byte for byte. Regenerating the head and MoE subsets changed their provenance
only, not their tensor payloads.

Two window views must remain distinct: prefill returns the newly prepared
chunk, while decode returns the complete physical ring. The fixture records
both the newly prepared rows and the returned read, including the wrap at
decode position six. Its dependency-free integrity gate checks encoded tensor
lengths and SHA-256 digests, with a deliberately corrupted-byte control.

`deepseek::attention::layer::LayerAttentionState` joins query projection,
row-wise normalization, rotary tails, quantized window preparation, sparse
attention and output projection. It owns a bounded numerical BF16 ring, not
a packed cache or serving scheduler. A call stages state until execution
succeeds; the publication identifies its source layer, request epoch and
successful-call ordinal.

The source comparison requires exact BF16 query projection, normalization,
rotary, prepared-window, returned-window, ring, sparse-output and final-output
bits, plus exact native window indices, across the prefill and both decodes.
All these checks pass without widening the comparison policy. The captured
grouped `wo_b` input remains an available diagnostic, not a separately checked
native intermediate in this test.

Publication validation rejects an incorrect compressed-prefix length, stale
identity, noncausal or out-of-range IDs, duplicate nonnegative IDs within a
query and phantom slots when the prefix is empty. Valid empty prefixes execute
window-only attention. Negative controls retry successfully against the
unchanged state. This validates structural ownership and index invariants;
it does not authenticate supplied values, reproduce indexer scoring or prove
that the supplied slot count is the source indexer's selected count.

The compressed values still come from layer three in the source capture.
The supplied selected indices are **layer four's own source-computed
reindexing results**, not indices reused from layer three. Native reindexing,
the native compressed-cache producer and full-block composition remain
separate acceptance gates. This diagnostic is not pretrained-model execution
or a performance benchmark.

## Joined native attention, HC and FFN

The `native_attention_hc_ffn_chain_matches_source_numerical_contract` test in
`forward_moe.rs` removes the source-provided attention-output boundary from
the block-tail comparison. Native HC collapse and normalization derive the
attention input from the captured block residual and incoming pre-mix. That
actual derived buffer feeds one native attention state through all three
calls. Native attention output then feeds HC post-mixing and the native FFN.
The outgoing pre-mix is the freshly derived FFN pre-mix, not the attention
pre-mix; the existing contract checks it against the captured outgoing
coefficients.

The shared attention harness requires equal complete-capture identities,
matching call counts and positions, and exact derived attention-input bits.
Native attention output must also equal the MoE fixture's attention tensor
before that tensor can serve as an exact point in the HC error envelope.
The numerical policy is unchanged. A control executes native attention but
discards its output; the downstream exact MoE checkpoint must reject it.
Separate controls reject a mismatched capture identity and reordered calls.

Upstream block residuals and incoming coefficients, including upstream Engram
effects, still come from the source graph. Compressed KV and the layer-four
indexer results are also supplied. This establishes the joined arithmetic
path for the captured executions, not end-to-end native model generation or
block-level transactional state rollback if a later sublayer fails.

## Owner-layer index keys

```sh
uv run --python 3.13 python scripts/v41_index_key_capture.py \
  --input artifacts/v41-indexer-source.json \
  --output fixtures/deepseek-v41/forward-index-key-reference.json
cargo test -p deepseek --test forward_index_key
```

This offline extractor selects layer three's `indexer.wk.weight` and
`indexer.k_norm.weight`, not the compressor's projection and normalization
weights. It retains the compressor output and the completed index-cache
prefix after each source call. It copies or slices captured storage bytes;
it does not calculate expected keys. Repeated extraction is byte-identical.
Source revision, model hash, complete-capture hash, tensor hashes, storage
types, geometry and the three-call schedule identify the oracle.

The timing of the capture matters: the compressor forward hook serializes
its output before attention rotates and quantizes that same latent storage.
The source indexer must read that original latent. The extractor reuses the
captured layer-four frequencies only for the pinned layer-three/four
ratio-one layout: both layers take the same nonzero-compression branch in
the [pinned attention constructor](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L678).
This is not a general rule for sharing rotary buffers between models.

`prepare_index_keys` matches the captured final BF16 key bytes at starts
zero, five and six. `IndexKeyState` retains these keys and exposes the complete
per-batch prefix, compared byte-for-byte with the source cache after each call.
Replaying position-zero frequencies during decode changes the result.
`forward_index_attention` feeds this native prefix into the BF16 scorer and
selector, then passes their selected indices to native layer attention.
Owner-layer input and candidate masks still come from the capture. Compressed
KV comes from the coupled native owner described below. The ratio-one compressor
executes natively through the atomic owner;
these checks do not establish arbitrary Torch/GPU reduction parity. The captured
weights are from the synthetic reduced model, not the released checkpoint.

The cache is DeepSeek-local and request-local: it does not reproduce the
source's process-global shared slots. Its offsets count completed compressed
positions, not tokens. The caller remains responsible for compression-group
completion and token continuity. An empty append advances the call ordinal
without adding a key. Source-layer, epoch and call checks are local safeguards,
not features of the source implementation. Candidates and selected indices
have separate ownership. Identity checks guard appends; the current scorer
accepts a numeric slice and does not authenticate its owner. A future integrated
indexer must carry publication identity through that consumer boundary. Borrowing
prevents mutation while a prefix is live, but copied values carry no identity.

Backing storage is `[batch, capacity, key_dimension]`; a valid prefix is a
separate borrowed slice for each batch. Truncating the entire backing buffer
would mix capacity padding into multi-batch scoring. Chunking properties cover
this layout, empty appends, late invalid input, retry and reset. Cache updates
are atomic individually, not across a later attention or FFN failure. The
current one-million-element cap bounds a CPU reference cache, not a general
GPU or larger-than-memory cache manager.

### Native ratio-one compressor input

```sh
uv run scripts/v41-forward-reference.py --output artifacts/v41-owner-compressor-source.json
uv run --python 3.13 python scripts/v41_index_key_capture.py \
  --input artifacts/v41-owner-compressor-source.json \
  --output artifacts/v41-owner-compressor-index-key.json \
  --compressor-output fixtures/deepseek-v41/forward-compressor-reference.json
cargo test -p deepseek --test forward_index_key
```

The supplementary capture retains layer three's attention input, BF16 `wkv`
output and normalized compressor latent. This reduced model uses compression
ratio one, with no grouped pooling. Native BF16 linear projection and
`CompressorState` match both captured stages exactly at starts zero, five and
six. Their returned latent now feeds the owner-key/cache test; a zero-weight
control changes that latent. The selection-to-attention test now calls the
atomic owner from the supplementary raw input, checks its projection and latent,
and passes its committed key prefix into the scorer. It compares that latent
with the older key fixture before continuing across the capture boundary.

The new complete capture has SHA-256
`2f2ff3f1734f959b33a673773cf6fe9c056fabb06a82531e5465562af5480c39`;
a repeated execution was byte-identical. Older fixtures retain their historical
capture identities. The native compressor result is checked against both the
new latent and the old key fixture's latent before continuing, so these captures
are not assumed numerically interchangeable just because the model is unchanged.
The compressor and attention fixtures pin their historical observer hashes.
The newer candidate fixture below gates the live observer hash.

`RatioOneIndexKeyOwner` composes these production primitives in one atomic
owner call. Projection and compressor progress are staged first; key preparation
then operates on the staged latent. Only after the cache accepts the complete
append does the owner replace its compressor state. This avoids cloning the
capacity-sized key cache on each call. A malformed rotary input after successful
compression leaves both states unchanged; retrying the same call reproduces
the captured projection, latent and key prefix. Randomized batch/input tests
also compare failed-then-retried calls with clean runs.

Reset is explicit and advances the epoch. Unlike the source compressor's
implicit restart at token zero, the owner rejects a second prefill until reset;
an accidental replay cannot replace a live prefix. This API deliberately covers
ratio one, not grouped compression or candidate selection. Its transaction ends
at key publication: downstream attention or FFN failure does not roll back the
owner, and a complete model runner still needs a broader transaction boundary.

### Coupled index-key and compressed-KV publication

The second owner product is compressed attention KV, distinct from the
index keys used to select positions. The [source compressed-KV path](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L739)
rotates the original compressor latent and uses G16/E4M3 reconstruction; index
keys instead use `wk`, key normalization and G32/E8M0 reconstruction.
`prepare_compressed_kv` rotates the original native owner latent and reconstructs
G16/E4M3 values. `RatioOneCompressedOwner` retains these values in a distinct
`CompressedKvState`; the attention integration consumes its complete native
prefix, matching the captured consumer prefix exactly at starts zero, five and
six. G32/E8M0 substitution and
position-zero frequencies during decode produce detectable differences. The
captured consumer prefix suffices as the final-byte oracle; no new intermediate
capture was needed for this check.

The combined owner derives and validates its KV layout before allocation.
It stages the small compressor state and both numerical products, then prepares
both cache appends before committing either. Each pending append holds an
exclusive cache borrow and an immutable values borrow; commit performs only
validated per-batch copies and metadata assignments. A rejected second append
cannot publish the first. Reset similarly validates both epoch transitions
before clearing either cache. No capacity-sized cache clone is needed per call.

`IndexKeyState` and `CompressedKvState` remain separate public types over a
private bounded prefix store. Sharing storage mechanics does not make their
precision formats or consumer roles interchangeable. The existing key-only
owner retains its narrower contract and valid latent geometries; only the
combined owner requires the G16-compatible latent width and rotary tail.

The transaction ends at the coupled owner publication. A subsequent attention
or FFN failure still requires a broader model-runner transaction. Full candidate
production remains a separate execution boundary; this reduced scalar
reference is not released-checkpoint generation or a GPU performance result.

### Candidate composition and remaining query prefix

Layer three produces the candidate mask consumed by layer four. The pinned
[indexer path](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L550)
prepares its query, reduces weighted scores, masks future compressed positions,
then selects candidate blocks. The existing `csa2::candidate_mask` implements
the block-selection rule. The native query, score and mask chain is now joined
in source-oracle tests, including native producer QR formation. A complete
candidate-selection runtime adapter remains open.

The candidate capture observes layer-three `wq_a`, `q_norm`, and indexer query,
weight and score stages, alongside the final candidate mask. It is separate
from historical fixtures, which retain their capture identities.
Candidate composition reuses index-query preparation, BF16 scoring and
the coupled key prefix. `CandidateQueryLayout` combines an existing validated
index-query layout with the matching private QR layout. The stateless
`prepare_candidate_query` derives QR through the same private `wq_a`/RMSNorm
helper used by attention. It takes no KV publication, window, or mutable state.
The helper retains G32 FP8 activation quantization and BF16 narrowing at the
projection and normalization boundaries; it is model-local, not a universal
query interface.

`forward_candidate.rs` and `forward_index_attention.rs` now check exact
source-stage and candidate-mask parity at starts zero, five and six, followed
by unchanged layer-four selected IDs and attention results. The integrated
path passes the native owner's complete index-key prefix into candidate
scoring, then feeds the generated mask into consumer selection. Attention
consumes the native compressed-KV prefix. The source key append is checked
as a suffix, not mistaken for the full prefix used by decode scoring.

Unmasked future scores are rejected. Perturbing query weights changes the
projection; removing a selected candidate changes both selection and attention.
The latter control holds captured consumer scores and KV fixed to isolate the
mask effect; it does not replace the positive native-owner integration.

Layer-three input activations remain captured operands; its `wq_a` and QR are
now computed and checked against captured expectations before the native
index-query path consumes them. The layer-four consumer now uses the same
stateless adapter with its own captured weights and input, checking its native
projection and QR against the attention oracle before scoring. Neither producer
nor consumer scoring consumes captured QR.

`indexer::selection` now owns causal masking, candidate-block selection and
final selection over supplied head-summed BF16 scores. Its `SelectionGeometry`
requires the exact completed-group key count, bounded score storage, an
in-range index offset, and prefill or single-position decode geometry.
Zero-key calls bypass this initial nonempty-prefix adapter. Each call processes
one batch and carries its explicit batch index and key-publication identity.
An opaque `CandidateSelection` can be consumed only with matching call metadata.
This is a consistency check on caller assertions, not proof of key provenance.
Producer and consumer scores are intentionally different: the shared result is
the producer's candidate mask, not its score matrix. Requiring score equality
would reject the source's cross-layer CSA2 path.

All raw scores must be finite, including future entries that will be masked.
Candidate bits may include future positions in a selected partial block;
final selection independently reapplies causality before candidate filtering.
The joined source test uses these production stages. Its isolated mask-bit
perturbation control keeps test-local masking to change a single bit without
providing a public constructor for arbitrary candidate results.

Scoring orchestration remains in the test harness. The remaining runtime work
joins prepared queries, owner key prefixes and these selection stages into
the model runner, followed by whole-block transaction handling.

The earlier consumer-only observer could not simply be pointed at layer three:

- It required an already-published key prefix and candidate mask at entry.
  The owner publishes its updated keys during the call and produces the new
  candidate mask afterward. Read the scoring operand at the actual einsum
  boundary, not from a possibly stale entry snapshot.
- Its first active FP4 call was labeled `q_after_rope_fp4`. In the owner, key
  quantization precedes query quantization. Identify the query operand through
  its source operation, rather than assuming the first quantized tensor is Q.
- The compressor fixture checked the live observer hash. It now pins its
  historical hash, not silently relabeled old bytes. The new fixture has its
  own capture identity and observer hash.

`IndexerRole` now distinguishes the two fixed producer/consumer roles, with
separate per-call state. A quantization-phase enum is set by the actual
`k_norm` and `wq_b` module hooks, not tensor shape or call ordinal. The key
prefix is copied from the actual score-einsum operand. Mask observations track
the summed-score tensor identity: prefill has a causal-mask operation, while
single-token decode does not. The final source index offset follows the window
width (five, six, six for this trace), not the final token position.

```sh
uv run scripts/test_v41_observer_runtime.py
uv run scripts/v41-forward-reference.py --output artifacts/v41-candidate-formatted-source.json
uv run --python 3.13 python scripts/v41_candidate_capture.py \
  --input artifacts/v41-candidate-formatted-source.json \
  --output fixtures/deepseek-v41/forward-candidate-reference.json
```

The complete candidate capture has SHA-256
`7c5cc8541da338fa3426d63e32b9a66e9132e07ab68ee26d86fbf9e29f62f48d`;
a repeated capture is byte-identical. Its observer hash is
`0235926c7fbd884433021d5ddcef6731884123e0df867466845fb2b39bf33c16`.
The opt-in runtime test executes the pinned graph with and without observers,
compares outputs and cache bytes exactly, checks historical attention values,
and injects a producer-query exception to verify hook and binding restoration.
It also checks that the produced candidate mask equals the consumer's input.
These runtime checks establish observation integrity. The Rust integration
tests separately qualify the native candidate chain from captured input onward;
neither establishes full-model generation.
