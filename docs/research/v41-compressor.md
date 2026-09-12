# V4.1 compressed-KV state and block ordering

## Source and reading coverage

The complete `Compressor` implementation and its `Attention`, `Indexer`,
`SharedAttentionRuntime`, `Block`, and `Transformer.forward` call sites were
inspected in the pinned
[inference/model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py).
Revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
Retained source SHA-256:
`4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
This is code reading and bounded CPU capture, not a paper reproduction.

Terminology correction: `Compressor` is learned KV pooling. It does not
compress tokenizer IDs or implement Engram, and the source does not name a
separate `CED` class. Architectural CED terminology must not be treated as an
extra operator to invent independently of the actual layer schedule.

## Pooling and state contract

- Ratio one uses BF16 projection followed by RMSNorm. There is no score gate
  or partial-group state.
- Larger ratios promote input to FP32 and compute separate KV and score
  projections. Softmax is over the group's **token axis, independently for
  each feature**. Weighted pooling remains FP32.
- The pooled result narrows to the input dtype **before** RMSNorm. For BF16
  input, this creates two distinct BF16 boundaries: pre-norm and norm output.
- Prefill (`start_pos == 0`) emits complete groups and saves the trailing
  partial group. It returns no latent when no group is complete.
- Subsequent source calls expect one token. They fill `start_pos % ratio`
  and emit only at a group boundary. Arbitrary multi-token continuation is
  not a qualified source path; chunked prefill needs explicit orchestration.
- Reset does not clear every source buffer slot. Under sequential writes,
  all slots of a new complete group are replaced before pooling. A runtime
  must separately enforce position continuity, batch identity and capacity.

## Native pooling boundary

The model-local `deepseek::compressor` boundary owns preprojected pooling,
normalization and partial-group state. Learned projection execution remains
upstream; rotary, index-key publication, attention cache ownership and request
scheduling remain downstream or engine concerns. This keeps the model-specific
token-axis gate out of generic cache/scheduler APIs.

`CompressorInput::ProjectedBf16` supplies the ratio-one projection result.
For larger ratios, `CompressorInput::Gated` supplies separate FP32 KV and score
projections. `CompressorState` fixes batch count, width, ratio and normalization
parameters at construction. Inputs and outputs use batch/position/feature
order. A successful start-zero call resets the sequence; larger-ratio
continuations must be sequential singleton calls. This is a bounded scalar
reference, not a learned checkpoint compressor or a GPU implementation.
All ratios enforce stream continuity as an API safety policy; the upstream
ratio-one branch itself is stateless and does not impose that restriction.
Failed validation, allocation, pooling or normalization leaves the prior
partial group and next position unchanged. State is staged before commit;
`MAX_COMPRESSOR_ELEMENTS` bounds each buffer, not aggregate process memory.
Nonfinite scalar intermediates are rejected rather than treated as a claim
about upstream overflow behavior.

The public API is exercised against the pinned source capture by
`crates/models/deepseek/tests/compressor_api.rs`. The fixture adapter only
constructs its declared identity KV and reversed/scaled gate projections;
pooling and normalization execute in the library.
Additional tests cover ratio four, independent feature-wise token weights,
finite softmax underflow and failed completion/reset retries. Restoring an
early zero-denominator check makes the underflow regression fail: the
denominator is only required to be positive after all token terms are summed.

```sh
cargo test -p deepseek --test compressor_api
```

## The attention join that must remain ordered

1. A KV-source layer produces the **unrotated** pooled latent.
2. Its indexer derives index keys from that latent, normalizes, rotates and
   quantizes them, then writes/publishes its index cache.
3. Attention rotates the latent at the group's starting position, performs
   compressed-KV quantization, and writes its own cache.
4. Consumer layers reuse the published cache and, where configured, sparse
   indices. Source layers must run before their consumers.

The representations differ: compressed KV uses a 16-element FP4 activation
group with E4M3 scales; index keys use a 32-element FP4 activation group; raw
window KV uses the separate FP8/E8M0 path. A shared representation chosen only
because the tensors have similar dimensions would erase source semantics.
The current pooling capture does **not** qualify these downstream joins.

These are **numerical quantization boundaries, not packed-cache memory
claims**. `inference/kernel.py:fp4_act_quant` with `inplace=True` writes
dequantized values back into `x` in its original dtype. The model allocates
cache tensors with the current Torch default dtype, set to BF16 by its local
example entry point. Copying this path does not produce four-bit physical
cache storage. A packed cache would require a separate layout, scale ownership,
kernel consumer and measured allocation contract.

This distinction was checked against the complete `fp4_quant_kernel` and
`fp4_act_quant` bodies in the same revision's
[kernel.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py),
retained SHA-256
`1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.
Neither TileLang kernel was executed locally. FP4 cast rounding and E4M3 versus
E8M0 scale behavior still need a qualified numerical implementation before
removing the compressed-publication harness's quantization stub.

For that implementation, the kernel's scale paths are not interchangeable:

| Consumer | Group | Scale computation before E2M1 encoding |
|---|---|---|
| Compressed KV | 16 | Floor `amax` at `6 * 2^-9`, divide by 6, cast to E4M3, then widen the stored scale for division/reconstruction |
| Index keys/queries | 32 | Floor `amax` at `6 * 2^-126`, then use the power-of-two `fast_round_scale` path |

Both clamp normalized values to `[-6, 6]`. In-place reconstruction casts to
E2M1, widens to FP32, multiplies by the selected scale, and narrows to the
input dtype. Tests must cover all-zero groups, scale-bin boundaries, E2M1 ties,
signed zero and reconstruction overflow before this replaces a stub.

### Software FP4 arithmetic qualification

`deepseek::precision::requantize_bf16_activations_e2m1` is the bounded scalar
BF16 → E2M1 → BF16 library reference, with `Fp4ActivationMode` distinguishing
compressed-KV and indexer modes. The implementation lives in
`crates/models/deepseek/src/precision/fp4_activation.rs`; the composition tests
in `crates/models/deepseek/tests/fp4_activation.rs` call that API directly.
`scripts/v41-fp4-activation-reference.py` provides an independent CPU
arithmetic oracle: Torch performs E4M3 scale casts and BF16 narrowing; explicit
midpoint intervals provide assumed E2M1 round-to-nearest, ties-to-even.
Its fixture is `fixtures/deepseek-v41/fp4-activation-reference.json`.
This is **not** an upstream kernel capture or a hardware-parity result.

The API writes reconstructed BF16 values to caller-provided storage, not
packed four-bit codes. It validates shape and finite inputs before reserving
one temporary output buffer, reports allocation failure, and copies results
only after all groups succeed. Diagnostic scale/code assertions remain in
unit tests; callers do not allocate or receive trace arrays. The request
limit is `MAX_FP4_ACTIVATION_ELEMENTS`.

For example, a group with `amax=9` uses E4M3 scale `1.5` for compressed KV,
but E8M0 scale `2` for the indexer. The indexer scale expression multiplies by
the FP32 reciprocal of six; do not silently substitute division when porting
the expression. The all-zero index group uses scale `2^-126`; the compressed
KV group uses the minimum E4M3 subnormal scale `2^-9`.

The bounded software reference rejects raw E4M3 scales above 448 and nonfinite
reconstructions without changing caller output. That is a deliberate safety
restriction, not a claim about upstream overflow/saturation behavior. A reduced
scalar composition can use this reference under its explicit rounding
assumption. Upstream cast qualification and Metal execution remain separate
gates; passing this oracle does not establish either.

Scale-encoder tests cover every nonnegative finite E4M3 code and every adjacent
midpoint, including immediate FP32 neighbors. A targeted mutation reversing
the even-tie preference was rejected by the midpoint test. Shape overflow,
work limits and short buffers are tested without allocating the requested
oversized shape, with caller output required to remain unchanged.

### Compressed-only attention composition

The FP4 test harness also joins the existing `rotate_tail` and
`sparse_attention_reference` implementations. Supplied BF16 compressed latents
are widened for tail rotation, narrowed to BF16, reconstructed with the
group-16 E4M3-scale FP4 reference, then consumed as shared keys/values.
The independent CPU oracle is
`scripts/v41-compressed-attention-reference.py`; its capture is
`fixtures/deepseek-v41/compressed-attention-reference.json`.

The test compares intermediate BF16 words exactly and mathematical attention
outputs within a predeclared `2e-6` absolute tolerance. It includes duplicate
sparse slots, a denominator-only sink and an all-masked query. Quantizing
before rotation, or omitting quantization, must change the final result by
more than `1e-3` in at least one component.

This is supplied-latent, supplied-index composition: compressor projections,
index scoring, window/cache ownership, BF16 attention-kernel rounding, output
projections and full-model generation are not exercised. Frequencies and
already-prepared query vectors are supplied rather than generated here.

With the `metal` feature, a second composition test replaces the supplied
final indices with the existing GPU score core and CPU final selection.
Its supplied index operands pass through group-32 E8M0-scale FP4 reconstruction.
Two signed query heads and two keys yield independently derived scores
`[-1024, 96]`, selecting position 1. The calculated index then drives attention
over rotary/FP4-prepared compressed KV; a zero query and zero sink give the
closed-form result of half the selected vector. This checks real Metal score
execution and the handoff, not projection/norm/RoPE preparation of index
operands, candidate filtering or the upstream BF16 score-kernel rounding.
The same handoff checks a shorter causal prefix selecting position 0 and an
empty prefix producing a `-1` sentinel and zero attention output.

A further Metal-feature test executes index-key `wk` and `k_norm` using the
existing BF16 linear and RMSNorm references with synthetic weights, followed
by key rotation, group-32 FP4 reconstruction, scoring, selection and attention.
The projection reads a latent tail component whose sign changes under the
attention rotation: correct pre-rotation inputs yield scores `[-992, 248]`,
whereas prematurely rotated inputs yield an ambiguous `[248, 248]` cutoff.
This turns the shared-latent ordering requirement into a numerical check.
Compressor latents, query heads and projection/norm weights remain synthetic;
checkpoint loading and the full indexer are not qualified by this test.
The integration binary serializes its Metal operations using the same
test-only device guard convention as the library tests. An unguarded parallel
run crashed, while an explicitly serial run passed. These tests do not qualify
concurrent MLX requests or solve runtime device ownership for serving.

The query-side composition test
`projected_index_queries_rotate_before_fp4_and_scale_signed_head_weights`
executes synthetic BF16 and FP8 `wq_b` branches and the explicitly BF16
`weights_proj` branch of pinned `Indexer.forward` (lines 550–555). `wq_b`
inherits the configured default weight dtype; this test does not load
checkpoint tensors or cover the preceding `wq_a`/`q_norm` producing `qr`.
With synthetic weights, two projected heads are constant +9 and -9. Tail
rotation by `0.6 + 0.8i`, BF16 narrowing and group-32 FP4 reconstruction give
head sums +244 and -244. A separate projection gives weights `[-32, 8]`;
the source's `index_head_dim^-0.5 * n_heads^-0.5` factor is 1/8 for this
32-wide, two-head case, giving BF16 weights `[-4, 1]`.
The weight projection consumes `x`, not `qr`; `n_heads` is the global head
count, not the local shard count. This single-rank test does not exercise
distributed head reduction or distinguish those counts.
Against supplied keys +1 and -1, the real Metal core produces exact scores
`[-976, 244]` and selects position 1. Omitting rotation gives `[-1024, 256]`,
omitting scaling gives `[-7808, 1952]`, and quantizing before rotation also
changes the scores. These analytical checks qualify the composition under the
scalar rounding assumptions, not upstream BF16 GEMM/reduction parity or
full indexer cache/candidate orchestration.

The FP8 query branch uses the existing activation quantizer and
`fp8_linear_runtime_f32`, then explicitly narrows the result to BF16.
Its supplied `qr` has one value 9.25 in a 32-element group. Scale `2^-5`
(E8M0 code 122) normalizes it to 296; software E4M3 RNE yields 288
(code `0x79`), reconstructing 9. The two 32-row weight blocks select that
element with codes +1 and -2 and separate scales 1 and 1/2. The resulting
heads are exactly +9 and -9, equal to the BF16 branch's outputs before the
shared rotary/FP4/scoring path. Exact intermediate assertions detect skipped
activation quantization (9.25 instead of 9) and shared output-block scales
(-18 instead of -9). This follows the G32 FP8 branch of pinned `linear`
(lines 196–204), using synthetic runtime buffers rather than checkpoint
decoding or upstream GPU GEMM execution.

```sh
uv run scripts/v41-compressed-attention-reference.py > artifacts/compressed-attention-reference.json
cargo test -p deepseek --test fp4_activation rotated_fp4_compressed_keys_feed_sparse_attention_in_source_order
cargo test -p deepseek --features metal --test fp4_activation fp4_index_scores_select_the_compressed_vector_consumed_by_attention
cargo test -p deepseek --features metal --test fp4_activation projected_index_queries_rotate_before_fp4_and_scale_signed_head_weights
```

```sh
uv run scripts/v41-fp4-activation-reference.py > artifacts/fp4-activation-reference.json
cargo test -p deepseek --test fp4_activation
```

## Qualification artifacts

`scripts/v41-compressor-reference.py` executes only SHA-checked
`Compressor.forward` and `RMSNorm.forward`. Identity KV projection and a
reversed/scaled score projection are explicit stubs. Two batches, width four,
ratios 1/2/3, and prefix lengths 1/4/7 exercise complete groups, partial groups,
singleton continuation and dirty-state reset. The checked-in capture is
`fixtures/deepseek-v41/compressor-reference.json`.

```sh
uv run scripts/v41-compressor-reference.py > artifacts/v41-compressor-reference.json
cargo test -p deepseek --test compressor_composition
cargo test -p deepseek --test block_composition
```

`scripts/v41-block-reference.py` independently captures the pinned
`Block.forward`, its HC helpers and `RMSNorm.forward`, emitting
`fixtures/deepseek-v41/block-reference.json`. Regenerate it with stdout
redirected to an artifact path, as with the compressor capture.

The block harness composes real scalar HC pre/post and RMSNorm with explicit
attention/FFN and coefficient stubs. It tests the handoff: incoming pre-mix
feeds attention; current attention pre-mix feeds the current FFN; current FFN
pre-mix feeds the next block. Normalized sublayer inputs, block outputs and
returned pre-mix tensors are checked against the independent source capture.
It is not a complete learned block.

`scripts/v41-compressed-publication-reference.py` executes only the pinned
`Attention._compress_kv` and `_compress_topk_idxs` methods, using visibly
mutating rotary/quantization stubs and explicit compressor/indexer stubs.
`fixtures/deepseek-v41/compressed-publication-reference.json` captures prefill,
singleton completion, nonboundary decode, zero compressed length and consumer
reuse. `cargo test -p deepseek --test compressed_publication` checks that source
trace; it does not execute a Rust publication implementation. In particular,
`latent=None` still invokes the indexer when an existing compressed prefix is
available; only zero compressed length skips it on an index-source layer.
Consumer reuse checks the current source's returned indices as well as cache
contents. Regenerate with:

```sh
uv run scripts/v41-compressed-publication-reference.py > artifacts/compressed-publication-reference.json
```

Next consumer: replace the projection stubs with qualified weight execution,
then join pre-RoPE index-key production, compressed-cache publication, sparse
attention, and the block harness. Full reduced logits remain the acceptance
gate; none of these captures alone establishes full-model parity or speed.
