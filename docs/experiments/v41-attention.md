# V4.1 sparse-attention semantic reference

`sparse_attention_reference` is a bounded CPU mathematical reference for the
official V4.1 [`sparse_attn_kernel`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py#L311).
It is deliberately an operator gate, not a V4.1 text-forward implementation or
a performance result.

The explicit input layouts are query `[batch, query, head, dimension]`, shared
KV `[batch, key, dimension]`, sparse indices `[batch, query, slot]`, and one
sink logit per head. An index is either `-1` (empty) or a shared-KV position.
The caller must apply causality before this operation: a nonnegative index has
no implicit temporal restriction here.

For each query/head, the reference gathers every valid slot, scores query dot
KV times a positive finite scale, then takes a stable FP64 softmax. Repeated
slots intentionally count repeatedly. The sink is a denominator-only logit, so
it lowers output magnitude but supplies no value vector. An all-`-1` row
returns zeros, matching the upstream kernel's stated convention.

This differs deliberately from native-kernel parity. The pinned kernel gathers
BF16 Q/KV, uses block-64 online softmax, and casts unnormalized probabilities
to BF16 before its numerator GEMM, then stores the output as BF16. Metallix's
CPU reference takes validated finite FP32 inputs, performs stable FP64
arithmetic, and narrows its final result to
FP32. It neither executes CUDA/TileLang nor claims BF16, kernel, or Metal
parity.

Regenerate the pinned synthetic fixture after obtaining the already
hash-qualified source:

```sh
uv run scripts/v41-attention-reference.py \
  --source artifacts/v41-kernel-pinned.py \
  > artifacts/v41-attention-capture.json
# Only after capture succeeds:
cp artifacts/v41-attention-capture.json fixtures/deepseek-v41/sparse-attention-reference.json
cargo test -p deepseek attention
```

The capture script verifies the complete source SHA-256
`1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`, then
checks that it contains exactly one `sparse_attn_kernel` definition. It does
not execute the source or import TileLang. Its independent dense-gather oracle
records multi-batch/multi-head, duplicate-slot, masked/all-masked, and
sink-dominance cases. The generated fixture is the external numerical oracle
for the Rust test; hand-written unit tests additionally cover invalid
indices, non-finite values, invalid scales, and numerically large finite
scores.

Fixture parity passes for every captured scalar. Neither this reference nor
the fixture establishes sparse-attention throughput, device residency, cache
ownership, attention projections, RoPE integration, or complete V4.1 decoder
parity.

## Bounded FP32 Metal diagnostic

The `metal` feature adds
`deepseek::attention::sparse_attention_metal_f32`. This is a host-gather
diagnostic: it gathers validated sparse slots on the CPU, preserving duplicate
indices, and sends each nonempty query/head through FP32 MLX score, stable
softmax and value-reduction operations. The sink is appended as a softmax logit
but excluded from the value reduction. All-masked rows return zero.

Two possible next steps were native BF16 numerical emulation or a bounded
FP32 Metal semantic check. The latter gives an executable device-side operator
gate without combining block-64 online rounding, cache ownership and decoder
composition in one change. It does not replace the later BF16 precision gate.

The admission contract is intentionally narrower than the CPU reference:

- The existing length, finite-input, index and positive-scale checks still apply.
- An explicit `max_work` limits `batch * query * head * slot * dimension`.
  This is a query-key scalar-product estimate, not the complete operation count.
- Output and per-query gathered FP32 vectors are individually limited to
  1,048,576 elements. These are not process-memory limits; index staging,
  MLX intermediates and allocator retention remain outside the limits.
- Sink and scaled query-key logits must have absolute value at most 80.
  Host FP64 preflight also checks unscaled product accumulation with FP32
  headroom, so a tiny scale cannot hide an overflowing device dot product.
  These are diagnostic limits, not restrictions inferred for the model.
- Nonfinite device output is an error, not a successful comparison result.

Run the bounded application tests on Apple Silicon:

```sh
cargo test -p deepseek --features metal attention
```

The small Metal cases are checked against the CPU FP64 mathematical reference
at `1e-4 + 1e-5 * abs(reference)`. This tolerance qualifies the tested inputs,
not every admitted FP32 input or the native BF16 kernel. Metal tests share the
crate's GPU lock. No checkpoint download or full V4.1 decoder is involved.

The focused command passed 13 tests, including the preexisting attention and
configuration checks. New device cases cover multiple batches/heads, masks,
duplicates and hand-derived sink results (one slot gives 1; duplicating it
gives 4/3). Rejection cases cover invalid inputs, work/gather limits and huge
finite products hidden by a tiny scale or cancellation. Receipt:
`artifacts/v41-attention-metal-final-tests.log`. Both canonical default and
Metal checks passed; receipts are `artifacts/check-v41-attention-default.log`
and `artifacts/check-v41-attention-metal-clean.log`.

Next, independently compose forward RoPE on the query tail, sparse attention,
then inverse RoPE on the output tail, as in the pinned
[`Attention.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L763).
Projections, compressed-cache/index ownership, grouped output projection,
BF16 probability rounding and production kernel scheduling remain separate
gates. No sparse-attention performance claim follows from this diagnostic.
