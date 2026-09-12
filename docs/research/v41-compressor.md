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

`crates/models/deepseek/tests/fp4_activation.rs` is a bounded test-only
BF16 → E2M1 → BF16 implementation with separate compressed-KV and indexer
modes. `scripts/v41-fp4-activation-reference.py` provides an independent CPU
arithmetic oracle: Torch performs E4M3 scale casts and BF16 narrowing; explicit
midpoint intervals provide assumed E2M1 round-to-nearest, ties-to-even.
Its fixture is `fixtures/deepseek-v41/fp4-activation-reference.json`.
This is **not** an upstream kernel capture or a hardware-parity result.

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

```sh
uv run scripts/v41-compressed-attention-reference.py > artifacts/compressed-attention-reference.json
cargo test -p deepseek --test fp4_activation rotated_fp4_compressed_keys_feed_sparse_attention_in_source_order
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
