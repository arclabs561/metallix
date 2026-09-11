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
to BF16 before its numerator GEMM. The Metallix helper takes validated finite
FP32 inputs, performs stable FP64 arithmetic, and narrows its final result to
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

Fixture parity now passes for every captured scalar. The next gate is a
separately designed Metal path with its own precision reference.
Neither this reference nor the fixture establishes sparse-attention throughput,
device residency, cache ownership, attention projections, RoPE integration,
or complete V4.1 decoder parity.
