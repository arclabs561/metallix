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

## CPU final selection

The capture script also records ten synthetic cases for the final three
statements in the pinned `Indexer.forward`: clamp Top-K to row width, select by
score and re-sort by position, then apply the offset or causal `-1` sentinel.

```sh
uv run scripts/v41-candidate-reference.py --kind selection \
  --source artifacts/v41-reference-model.py > artifacts/v41-selection-reference.json
cmp fixtures/deepseek-v41/selection-reference.json artifacts/v41-selection-reference.json
```

The CPU `deepseek::select_indices` helper checks these ten reference cases.
It is not a Metal selection kernel or a complete indexer.
Inputs are already masked. A selected negative-infinity score does **not**
automatically produce `-1`: that conversion checks only causal reachability.
For example, `[8, -inf, -inf]` with three reachable positions, Top-K 3, and
offset 10 produces `[10, 11, 12]`. Fewer finite scores than requested positions
can therefore select candidate-masked but reachable entries. The fixture avoids
cutoff ties; sorting selected positions does not settle ambiguous Top-K membership.

The helper rejects ambiguous cutoff ties, including signed zero and negative
infinity. A cutoff tie entirely among future positions is permitted because
every possible choice maps to the same `-1` output. Ties wholly selected or
wholly excluded are also unambiguous. Inputs are bounded to 1,048,576 positions;
future scores must already be negative infinity, and reachable offsets must
fit signed 32-bit indices. The offset addresses compressed KV after the
sliding-window KV segment; it is not a token-position offset.

```sh
cargo test -p deepseek selection
cargo bench -p deepseek --bench selection
```

The CPU benchmark selects 512 positions from distinct synthetic score rows of
width 512, 4,096, and 16,384. Input construction is excluded; validation,
selection, allocations, and output disposal are included. These are diagnostic
operator costs, not model throughput.

At `ed732b1`, three release runs on M3 Max each measured 100 samples per shape
(default features, resident synthetic inputs, no concurrent benchmark). Medians
in microseconds were:

| Score positions | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| 512 | 7.582 | 7.832 | 7.833 |
| 4,096 | 51.74 | 52.58 | 51.24 |
| 16,384 | 215.7 | 213.5 | 213.4 |

The implementation fully sorts scores before sorting selected positions. These
measurements establish a baseline, not a speedup or a representative workload
distribution; real score ties and masking patterns need separate measurements.

These are operator-level parity gates and reference captures. They do not satisfy the text-forward
gate for downloading the full V4.1 checkpoint, and they are not a performance or
model-quality result. Next are BF16/FP4 score qualification, Metal selection,
and sparse attention with the same numerical and masking checks.
