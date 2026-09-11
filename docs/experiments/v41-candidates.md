# V4.1 sparse-indexer qualification

The first executable V4.1-specific diagnostic is a CPU implementation of
`select_candidate_blocks` from the [official inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L583).
It checks the first-stage candidate mask, not index-score generation, the
second-stage Top-K, sparse attention, or a V4.1 decoder.

For each query, the reference takes each block's maximum score, pins the block
containing the newest reachable compressed position, selects the highest
scoring blocks, and drops blocks whose score remains negative infinity. Its
output covers entire blocks: future positions inside a selected partial block
can have `true` bits. The caller's causal score mask remains necessary.

`fixtures/deepseek-v41/candidate-block-reference.json` contains 11 synthetic
cases captured by executing the exact hash-pinned upstream helper with CPU
float32 Torch. No model weights are downloaded or used. The Rust fixture test
requires exact boolean masks, including newest-block pinning, partial blocks,
zero K, no reachable positions, and K exceeding the available blocks.

The diagnostic deliberately rejects NaN, positive infinity, unmasked future
scores, and finite score ties across the Top-K cutoff. The upstream helper
delegates tie selection to `torch.topk`; matching one CPU choice would not
establish a portable GPU tie policy. Ties entirely within the selected or
unselected set do not make the mask ambiguous.

```sh
mkdir -p artifacts
curl --fail --location --max-time 30 \
  --output artifacts/v41-reference-model.py \
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py
uv run scripts/v41-candidate-reference.py \
  --source artifacts/v41-reference-model.py > artifacts/v41-candidate-reference.json
cargo test -p deepseek csa2
```

The capture script checks the complete source SHA-256, extracts only the named
function, and does not import the CUDA/Triton model module. The fixture records
the source revision, hash, symbol, dtype, and Torch version. See
[third-party notices](../../THIRD_PARTY_NOTICES.md) for the upstream MIT license.

## FP32 score reduction on Metal

The optional Metal diagnostic also computes the indexer's query/key dot products,
rectifies each head, multiplies by signed head weights, then sums across heads.
It compares every result with CPU execution of the two corresponding expressions
from the same pinned `Indexer.forward` implementation.

```sh
cargo run -p server --release --features metal -- check-v41-indexer-metal
```

Five synthetic cases include negative head weights, zero weights, negative dot
products, and the configured 32-head, 128-dimension shape. Input vectors stand in
for already-projected, post-RoPE vectors and already-scaled head weights. This
qualifies float32 arithmetic only, not the official BF16/FP4 rounding path.

The command compares an excluded warmup and three timed evaluations per case.
Every score must agree within `0.0001 + 0.00001 * abs(reference)`. Timings include
CPU validation, array creation, GPU graph construction, execution, and readback;
they are not model throughput or resident-kernel-only measurements. The core
head-by-position matrix is capped at 16,777,216 elements before GPU allocation;
that is not a peak-memory guarantee.

On the M3 Max qualification run (20 measured repeats per case), the tiny cases
matched exactly and the 32×128 case's largest score error was
`1.9073486328125e-06`. Its median end-to-end operator time was 0.2503 ms, with
sample standard deviation 0.1924 ms. That variance is too large for a small
speedup claim; this run establishes correctness and a diagnostic baseline.
Increasing one expected score by 1 caused both warmup and measured comparisons
to fail and the command to exit nonzero.

Regenerate the score fixture with:

```sh
uv run scripts/v41-candidate-reference.py --kind index-scores \
  --source artifacts/v41-reference-model.py > artifacts/v41-index-score-reference.json
```

These are operator-level parity gates. They do not satisfy the text-forward
gate for downloading the full V4.1 checkpoint, and they are not a performance or
model-quality result. Next are BF16/FP4 score qualification, second-stage
selection, and sparse attention with the same numerical and masking checks.
