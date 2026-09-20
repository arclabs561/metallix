# V4.1 sparse-indexer qualification

The first executable V4.1-specific diagnostic is a CPU implementation of
`select_candidate_blocks` from the [official inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L583).
That diagnostic checks the first-stage candidate mask, not index-score
generation, the second-stage Top-K, sparse attention, or a V4.1 decoder.
Separate [sparse-attention checks](v41-attention.md) now cover the mathematical
reference and bounded FP32 Metal path; they are not part of this mask check.

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

The separate [rotary-tail qualification](#fp32-rotary-tail-layout) below checks
the layout feeding this operator; neither gate implements full attention.

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

## FP32 rotary-tail layout

[`rotate_tail`](../../crates/models/deepseek/src/rotary.rs) checks the pinned
`apply_rotary_emb` operation with explicit batch, position, head and adjacent
complex-pair dimensions. Rank-three input maps to one head; frequencies
broadcast over batches and heads. The inverse flag conjugates frequencies.

```sh
uv run scripts/v41-rotary-reference.py \
  --source artifacts/v41-reference-model.py > artifacts/v41-rotary-regenerated.json
cmp fixtures/deepseek-v41/rotary-reference.json artifacts/v41-rotary-regenerated.json
cargo test -p deepseek rotary
```

The hash-checked Torch capture regenerated byte-identically in the local run.
Three cases cover rank-three, multi-batch/multi-head rank-four, and inverse
rotation. Rust compares each scalar within absolute tolerance `1e-6` and tests
length, shape-overflow and non-finite-input rejection without buffer mutation.
Finite inputs retain upstream IEEE FP32 overflow behavior; finite output is not
promised. This supplied-frequency operation does not itself generate frequencies or rotate BF16/FP4,
or implement an attention block. CPU receipts:
`artifacts/v41-rotary-regenerated.{json,stderr}`.

The optional `rotate_tail_metal` path now performs the same adjacent-pair
arithmetic with MLX indexing, broadcast arithmetic, stacking and logical
flattening before readback. It is a host-to-GPU diagnostic, not a fused kernel
or an in-place cache update. Reproduce its qualification and timing with:

```sh
cargo test -p deepseek --features metal rotary
cargo build -p server --release --features metal
target/release/mx check-v41-rotary-metal --repeats 5
```

On the M3 Max control host, all three pinned cases passed every warmup and
measured comparison. Maximum absolute error was `2.3841858e-7` against the
`1e-6` gate. Five measured round trips per case ranged from `0.204` to `1.042`
ms; startup warmups are excluded and reported separately. These tiny synthetic
cases measure graph/dispatch/readback costs, not useful model throughput or a
speedup over CPU. Receipts: `artifacts/rotary-metal-test.log` and
`artifacts/v41-rotary-metal.{json,stderr}`. The next integration gate remains
device-resident attention/cache use with a complete upstream numerical oracle.

### Generated RoPE and YaRN frequencies

`RotaryFrequencyParameters` now validates explicit FP32 parameters and generates
only a requested contiguous position range. It mirrors the pinned
`precompute_freqs_cis` expression: double-precision correction boundaries,
FP32 inverse frequencies and YaRN interpolation, then FP32 position/angle
arithmetic. It does not allocate the entire preceding context for a late range.
Configuration-to-parameter wiring remains separate; arbitrary FP64 parameters
must not be silently narrowed near a YaRN boundary.

[The capture script](../../scripts/v41-rope-reference.py) checks the source hash
before extracting `precompute_freqs_cis` and `apply_rotary_emb`. It removes only
the former's cache decorator and executes the expressions with pinned Torch
2.13.0 and NumPy 2.4.3. The
[fixture](../../fixtures/deepseek-v41/rope-frequency-reference.json) contains
three synthetic cases: local no-YaRN, compressed mixed-ramp YaRN, and inverse
rotation at absolute positions 65,533 through 65,536. The compressed parameters
are base 160,000, original length 65,536, factor 16, and beta values 32/1.

Both frequency components and the composed CPU/Metal rotations pass absolute
tolerance `1e-6`, without relaxing the gate for high positions. The Rust test
also checks invalid parameters and position overflow. Reproduce with
`cargo test -p deepseek --features metal rotary`.
This closes a small composed operator gate, not attention, frequency-cache
ownership, full-model inference or BF16/FP4 parity. There is no speed claim.
Receipts: `artifacts/v41-rope-captured.json`,
`artifacts/check-v41-frequencies-fixed.log`, and
`artifacts/check-cached-rope-metal-final.log`.
Fixture SHA-256: `9b4b06c131fbde15faba57a44d245e21a029313d4d6682797ed5bd0c8efa456c`.

The completion audit also tests interval composition: generating a requested
position range exactly equals slicing a generated prefix across small widths,
offsets and both scaling branches. The pinned Torch fixture remains the
independent numerical oracle. A separate rejection matrix covers non-finite
and nonpositive required parameters, ignored parameters in the no-YaRN path,
position/output overflow, and capacity overflow without a large allocation.
Canonical default and Metal checks passed afterward; receipts:
`artifacts/check-boundary-{default,metal}.log`.

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

That baseline fully sorted scores before sorting selected positions. A sampled
profile of the 16,384-position workload placed 2,497 of 3,001 leaf samples in
four score-sort routines. This motivated partitioning at the first excluded
score, checking the selected partition's minimum for cutoff ambiguity, and
sorting only selected positions. Selecting the entire row skips score ordering.

At `8b88312`, the same three-run benchmark produced these median microseconds:

| Score positions | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| 512 | 1.249 | 1.229 | 1.229 |
| 4,096 | 10.24 | 10.66 | 10.20 |
| 16,384 | 33.12 | 32.83 | 32.83 |

The median of run medians improved about 6.4×, 5.1×, and 6.5× respectively.
Divan retained 100 samples per shape, automatically batching two iterations per
sample for the faster 512-position case. These are short synthetic measurements,
not model speedups or a representative score distribution. Real score ties and
masking patterns need separate measurements.

An independent exhaustive oracle enumerates every Top-K subset for rows up to
five positions with scores drawn from `[-inf, -1, +0, -0, 1]`, across reachable
prefixes and K values. It compares all score-optimal subsets' observable outputs:
one distinct output is accepted, multiple outputs require tie rejection. Both
the full-sort baseline and partitioned implementation pass this oracle, as well
as the ten pinned upstream reference cases.

These are operator-level parity gates and reference captures. They do not satisfy the text-forward
gate for downloading the full V4.1 checkpoint, and they are not a performance or
model-quality result. Next are BF16/FP4 score qualification, Metal selection,
and sparse attention with the same numerical and masking checks.

## Ratio-one owner transaction microbenchmark

At `bf4ebb1` plus the benchmark working diff, three serial release executions
of `owner_transaction` measured the captured ratio-one owner fixture with
default CPU features and no concurrent benchmark. The fixture asserts its
pinned source revision, model hash, little-endian BF16 storage, and exact
prefill latent/key/KV outputs before it times anything. Each execution gathered
200 internal samples per phase; raw receipts are ignored under
`artifacts/owner-transaction/`.

The source capture contains only a five-token prefill and one-token decode
continuations at valid prefixes five and six. Setup, reset, and pre-publication
work are excluded where noted below. This is a bounded operator measurement,
not a model-throughput or prefix-scaling benchmark.

| Phase median (microseconds) | Run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Prefill-5 `prepare` | 109.333 | 109.250 | 109.708 |
| Decode `prepare` at prefix 5 | 29.584 | 27.833 | 29.542 |
| Decode `prepare` at prefix 6 | 29.666 | 27.896 | 29.562 |
| Decode transaction `drop` at prefix 5 | 0.125 | 0.125 | 0.125 |
| Decode transaction `commit` at prefix 5 | 0.125 | 0.125 | 0.125 |
| Decode `forward` at prefix 5 | 27.708 | 27.708 | 27.791 |
| Whole benchmark process wall time (milliseconds) | 91.710 | 89.244 | 90.226 |

`prepare` includes the production projection/compressor/key/KV work and the
transaction's complete staged key/KV prefix allocations. `drop` is timed after
preparation; `commit` is timed with an already prepared transaction; `forward`
is the production control that omits staged complete-prefix views. The
prefix-five to prefix-six prepare difference is much smaller than cross-run
variation, so this capture does not isolate prefix-copy cost as an optimization
target. The difference between decode `prepare` and `forward` varies across
runs and does not isolate allocation from other transaction work. Changing
allocation or copy semantics needs a source-grounded longer-prefix workload
and a new matched measurement.

### Synthetic prefix scaling

The opt-in `owner_transaction --synthetic-scale` mode exercises prefixes 8,
64, 512, and 2048 with the same batch/input/latent/key dimensions (1/128/64/64)
and deterministic nonzero BF16 inputs. Identity-style weights and rotary
frequencies make this a synthetic operator workload. Before timing each shape,
it compares complete staged and direct prefixes and checks drop/retry behavior.
Priming and owner cloning are outside the timed intervals. Preparation/drop
use 200 samples; commit/direct publication use 40.

Three serial processes on macOS 26.6.2, Mac15,9 (16 CPUs, 128 GiB), with no
concurrent project builds or captures, produced the following phase medians:

| Prefix | Complete staged key + KV bytes | Prepare μs, runs 1 / 2 / 3 | Direct publication μs, runs 1 / 2 / 3 |
| --- | ---: | --- | --- |
| 8 | 2,304 | 28.959 / 33.083 / 34.791 | 30.437 / 31.209 / 34.542 |
| 64 | 16,640 | 30.584 / 32.750 / 34.146 | 30.375 / 32.938 / 32.792 |
| 512 | 131,328 | 32.333 / 32.958 / 37.667 | 29.125 / 29.501 / 34.063 |
| 2048 | 524,544 | 39.125 / 44.334 / 50.625 | 31.500 / 33.480 / 41.480 |

The release executable SHA-256 was
`b89d3bccb625f2c29798e5970827b9ffc4559d73ead55b55b5968ead3ad67fa1`,
built from `10d95e5` plus the Engram lookup and benchmark working changes.
Raw timings, including within-process sample standard deviations, remain in
ignored `artifacts/owner-scaling/trial-{1,2,3}.txt`.

Preparation grows with prefix length in this workload. At prefix 2048 its
median exceeds direct publication by 7.625–10.854 μs within each process.
The phases run in a fixed order and differ in cache/allocation conditions;
this comparison does not isolate copy cost or establish a model speedup.
Matched timing order and a representative staged-view consumer remain gates
before changing the contiguous-prefix transaction interface.

```sh
RUSTC_WRAPPER= cargo bench -p deepseek --bench owner_transaction -- --synthetic-scale
```

The `--profile-synthetic` mode repeats prepare/drop 200,000 times at prefix
2048. A headless Samply run of an optimized build with debug information
recorded 7,444 unit-weight samples at nominal 1 ms intervals. The setup oracle
appeared in 45 samples; 7,354 samples contained `RatioOneCompressedOwner::prepare`
without the oracle. The remaining 45 samples are outside that selected stack
population. Within those 7,354 preparation samples:

| Leaf | Samples | Share |
| --- | ---: | ---: |
| BF16 linear projection | 3,107 | 42.25% |
| BF16-to-FP4 requantization | 1,377 | 18.72% |
| `_platform_memmove` | 1,336 | 18.17% |
| FP4 scale calculation | 1,077 | 14.65% |
| Index-key preparation | 273 | 3.71% |

Copying is a secondary hotspot in this synthetic workload; projection and
requantization account for more samples. Of the 1,336 copy samples, 1,335
appear directly under `prepare`, consistent with its staged-prefix copies.
`staged_prefixes` was inlined, so these stacks cannot uniquely identify that
copy site. Allocator growth appeared in only 65 inclusive samples (0.88%);
the profile does not establish allocation as the dominant cost. No runtime
optimization or interface change is claimed from this diagnostic.

The profiled binary SHA-256 was
`2e930c14623c94e4e2996a7f68d861b331b110a3226adfcb083357f06b1abdc7`.
The ignored `artifacts/owner-scaling/profile-2048.json.gz` and `.json.syms.json`
retain the samples and resolved symbol ranges. The loop reported 7,399.315 ms
under profiling; use the unprofiled measurements above for latency.

```sh
RUSTC_WRAPPER= CARGO_PROFILE_BENCH_DEBUG=1 cargo bench -p deepseek --bench owner_transaction --no-run
samply record --save-only --unstable-presymbolicate \
  -o artifacts/owner-scaling/profile-2048.json.gz -- \
  target/release/deps/owner_transaction-<build-hash> --profile-synthetic
```

## Scalar BF16 projection iteration

The owner profile above led to a small change in `bf16_linear_reference`:
iterate over validated activation/weight row slices instead of repeatedly
indexing the full matrices. Products, FP32 accumulation order, overflow checks,
BF16 narrowing, and transactional output publication remain the same.

At `309bd87` plus that working change, six alternating control/candidate pairs
ran the synthetic scaling workload. At prefixes 8 and 512, preparation medians
across processes changed from 28.917 to 26.813 μs and 31.031 to 27.615 μs.
All six paired preparation ratios improved at these shapes; the 2048-prefix
ratios ranged from 0.837 to 1.128, so the larger-prefix result is inconclusive.
The earlier sequential before/after trial showed stronger changes but also
large baseline drift; it is not used as the optimization result.

Three additional alternating pairs ran the captured source fixture as a
control. Values below are medians of process medians, with sample standard
deviation across those three process medians:

| Captured phase | Original μs | Row-slice iteration μs |
| --- | ---: | ---: |
| Prefill-5 preparation | 110.729 ± 0.626 | 100.500 ± 3.051 |
| Decode preparation at prefix 5 | 29.625 ± 0.673 | 27.833 ± 1.517 |
| Decode preparation at prefix 6 | 28.958 ± 0.804 | 25.730 ± 1.659 |
| Direct decode publication | 27.667 ± 0.244 | 25.605 ± 0.280 |

The control binary SHA-256 was
`d18b223a9c3ec9c2508dcb482a67f831d053882064973234e47c55581684ff4b`;
the candidate was
`a20248d97da24865eda40a05a77ee7a213208e289a1b22883ade90f7eea18682`.
Both used default release features on the same host as the profile, with no
concurrent project builds or captures. Retained raw receipts and both binaries
are under ignored `artifacts/owner-projection/`. This is an operator-level
improvement, not a full-model throughput claim. Property tests additionally
check row splitting, output-column permutation, and exact late-overflow
coordinates with unchanged output.

## Packed FP4 runtime expansion

`deepseek::precision` expands E2M1x2 bytes in low-nibble-first order and
contiguous 32-element blocks with one E8M0 scale. The representation follows
[PyTorch's pinned byte definition](https://github.com/pytorch/pytorch/blob/84e524623ea4754a748936bf1ba6ecaaa92c3ae6/torch/headeronly/util/Float4_e2m1fn_x2.h)
and the V4.1 reference's
[linear allocation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
and [FP4 kernel input layout](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py).
Read coverage is those representation boundaries, not a full checkpoint loader.

`cargo test -p deepseek precision::` covers every packed byte, signed zeros,
block-scale boundaries, subnormals, and rejection without partial output.
Expansion allocates nothing; the caller supplies the output buffer. All scales
and scaled values are checked before writes begin.

This qualifies runtime representation and scalar FP32 multiplication only.
Safetensors byte layout, loader transforms, row padding, and fused GEMM rounding
are not established by these tests. It is not a quantizer or a file decoder,
and there is no performance claim.
