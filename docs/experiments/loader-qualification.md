# Bounded Qwen loading and V4.1 shape qualification

This pass qualifies selected Qwen tensor reads and initial V4.1 configuration
relationships. The later one-block experiment below executes selected weights.
Neither experiment implements a full streamed decoder, a weight pager, V4.1
text inference, or a process-wide memory ceiling.

## Qwen selected-tensor reads

At `7399db3`, [`Qwen3CheckpointInspection`](../../crates/models/qwen/src/checkpoint.rs)
retains validated tensor locations for an adapter-private synchronous reader.
It checks the selected payload budget before allocating or opening the shard,
then reads the exact range and checks file length and modification time around
the read. Inputs must remain immutable. These metadata checks are not protection
against adversarial writes preserving length and timestamp.

[`qualify_tensor_range`](../../crates/models/qwen/src/metal.rs) widens little-endian
BF16 bytes to FP32 and compares every finite value, bit-for-bit, with the
existing MLX loader. The CLI landed at `e0c25f4`:

```sh
cargo build -p server --release --features metal
target/release/metallix check-qwen-tensor-metal --model /path/to/Qwen3-0.6B
target/release/metallix check-qwen-tensor-metal --model /path/to/Qwen3-0.6B \
  --tensor model.layers.0.self_attn.q_proj.weight --max-bytes 4194304
```

On M3 Max, 2026-09-11, both commands exited 0:

| Tensor | Shape | Raw bytes read | Values compared | Result |
|---|---|---:|---:|---|
| `model.layers.0.input_layernorm.weight` | `[1024]` | 2,048 | 1,024 | Bit-exact |
| `model.layers.0.self_attn.q_proj.weight` | `[2048,1024]` | 4,194,304 | 2,097,152 | Bit-exact |

Setting the second command's budget to `4194303` exited 1 with the expected
budget error. Raw-file unit tests additionally cover unknown names, exact
offsets, timestamp drift, truncation and budget rejection before file I/O.

The JSON reports payload, decoded host and candidate-array bytes separately.
For the projection, these were 4 MiB, 8 MiB and 8 MiB respectively. They are
logical counts, not peak resident memory. Headers, allocator retention and the
whole-checkpoint reference load are outside the selected-payload budget.
Single read timings are retained as diagnostics, not SSD or speedup evidence.

Receipts: `artifacts/qwen-bounded-{norm,qproj}.json`. Executable SHA-256:
`26849b73689d16b59fc48851d9e232de323c8b226dc760d7d02b5dc6dd0fca3d`.
Config SHA-256: `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.
Checkpoint SHA-256: `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.

The one-block experiment below advances the execution gate; full streamed
forward and cached-forward parity with a measured total working set remain.
Passing selected reads alone does not satisfy those gates. The subsequent
[nested-header fix](../research/README.md#nested-header-uniqueness) closes the
duplicate-key parser follow-up without changing the streamed-execution gate.

## Contiguous BF16 row reads

The adapter-private [`read_bf16_rows`](../../crates/models/qwen/src/checkpoint/read.rs)
reads a nonempty half-open range from a validated rank-two BF16 tensor. Checked
arithmetic establishes containment and the selected raw-byte budget before
payload allocation or opening the shard. Inspection still reads headers.
[`qualify_tensor_rows`](../../crates/models/qwen/src/metal/row_check.rs)
compares the selected values with a slice from the independent resident MLX
loader. It slices BF16 reference rows before widening them to FP32.

```sh
target/release/metallix check-qwen-rows-metal --model /path/to/Qwen3-0.6B \
  --start-row 65535 --rows 3 --max-bytes 6144
```

On M3 Max, the same Qwen checkpoint identified above produced:

| Embedding rows (half-open) | Raw bytes | Values compared | Result |
|---|---:|---:|---|
| `65535..65538` | 6,144 | 3,072 | Bit-exact |
| `151935..151936` | 2,048 | 1,024 | Bit-exact |
| `0..4096` | 8,388,608 | 4,194,304 | Bit-exact |

All three commands exited 0. The middle slice with `--max-bytes 6143` and
the out-of-bounds `--start-row 151936 --rows 1` each exited 1. Unit tests
also cover invalid dtype/rank, empty/reversed ranges, exact budgets and
rejection before payload I/O. Default and Metal canonical checks passed.

Receipts: `artifacts/qwen-rows-{middle,last,tile}.json`, corresponding stderr
files, `artifacts/qwen-rows-{budget,oob}.stderr`, and
`artifacts/check-rows-{default,metal}.log`. Executable SHA-256:
`ecdde624d791f885dcef99f2a8c3f018df1325371c9ffcfa9e914821e19fa5ab`.

This prepares embedding lookup and tiled output-projection experiments; it
does not implement either in the decoder. The raw budget excludes decoded
FP32 buffers, headers, allocator overhead and the full resident reference.
Single read timings do not establish SSD throughput or a performance gain.

## Token-ordered selected embedding

[`qualify_embedding`](../../crates/models/qwen/src/metal/embedding_check.rs)
uses the row reader for actual raw token IDs. It reads each distinct row once,
then assembles an FP32 array in input order, preserving repeated IDs. The
aggregate raw budget is checked before payload reads. A separate resident MLX
`take_axis` lookup supplies the oracle; this is not the decoder's embedding
path yet.

```sh
target/release/mx check-qwen-embedding-metal --model /path/to/Qwen3-0.6B \
  --input-ids 151935,0,151935,3 --max-bytes 6144
```

With the Qwen checkpoint identified above, this exited 0 and compared all
4,096 values bit-for-bit. Output shape was `[4,1024]`, with three distinct
rows and 6,144 selected raw bytes, versus the full embedding tensor's
311,164,928 bytes. The candidate output was 16,384 FP32 bytes. These are
logical payload counts, not measured physical I/O or a peak-memory claim;
the resident oracle still loads the checkpoint. A budget of 6,143 exited 1.

The API caps inputs at 512 tokens and 1,048,576 output elements. Unit tests
cover duplicate-aware budgeting, invalid IDs and oversized output. Both
canonical checks passed. Receipts: `artifacts/qwen-selected-embedding.json`,
its `.stderr`, `artifacts/qwen-selected-embedding-budget.stderr`, and
`artifacts/check-embedding-{default,metal}.log`. Executable SHA-256:
`d7586bfc773184f4bfcf4337f11f0172f41457167d594d1f30004f0ba99da385`.

Next: qualify tiled tied-output projection before composing a complete
streamed forward. No speedup follows from this single lookup measurement.

## One-block selected-weight execution

At `51f61d0`, [`qualify_layer`](../../crates/models/qwen/src/metal/layer_check.rs) reads the
11 BF16 tensors required by one Qwen transformer block, widens them to FP32,
and executes the same private block kernel used by uncached resident forward.
It evaluates and reads back the output before releasing candidate weights.
Inputs are deterministic nonzero hidden states, not outputs from preceding
layers. Keep checkpoint files immutable throughout inspection and execution.
The CLI is available at `1da7a18`.

```sh
target/release/metallix check-qwen-layer-metal --model /path/to/Qwen3-0.6B \
  --layer 0 --tokens 3
target/release/metallix check-qwen-layer-metal --model /path/to/Qwen3-0.6B \
  --layer 27 --tokens 17
/usr/bin/time -l target/release/metallix check-qwen-layer-metal \
  --model /path/to/Qwen3-0.6B --layer 0 --tokens 3 \
  --max-weight-bytes 81798144 --candidate-only
```

On the same M3 Max and checkpoint, 2026-09-11, both comparisons exited 0:
layer 0 matched all 3,072 hidden-state values bit-for-bit; layer 27 matched
all 17,408. The resident comparison qualifies the selected loader, not an
independent block implementation. Separately, the existing CPU-reference
suite passed full-vocabulary logits for prompt lengths 1, 3, 17 and 64 after
the block extraction: zero mismatches under its tolerances, maximum absolute
error `4.208087921142578e-5`. A small nonzero two-layer unit test also compares
the extracted blocks with the independent cached-prefill implementation.

Each layer reads 31,461,888 raw bytes and retains 62,923,776 logical FP32 bytes.
The peak loading plan is 81,798,144 bytes: previously retained arrays plus the
current raw tensor, host FP32 conversion and copied FP32 array. A budget of
81,798,143 exited 1 with the planned-versus-allowed error before payload loading.
The byte plan excludes hidden states, execution scratch, headers, allocator
retention and the resident comparison. It is not a process-memory limit.

Three serial, fresh-process candidate-only runs used the exact budget above:

| Run | Load ms | Block execution ms | Maximum RSS bytes | Peak footprint bytes |
|---|---:|---:|---:|---:|
| 1 | 17.963 | 9.853 | 111,017,984 | 288,424,584 |
| 2 | 18.516 | 9.662 | 111,001,600 | 288,408,200 |
| 3 | 18.138 | 9.430 | 111,001,600 | 288,391,792 |

Memory fields are the macOS `/usr/bin/time -l` process observations, including
startup and inspection, not isolated GPU allocation counters. RSS and footprint
are different OS accounting measures; neither is the logical weight plan.
Each process performs one block execution with no warmup; timings are diagnostic,
not steady-state decode throughput or a speedup. OS file-cache state was
uncontrolled. No global cache flush or host memory-limit change was used.
`--candidate-only` deliberately reports verification `not_run`; parity comes
from the separate comparison commands, not the memory runs.

Receipts: `artifacts/qwen-layer-{0,27}-compare.{json,stderr}`,
`artifacts/qwen-layer-candidate-{1,2,3}.{json,time}`,
`artifacts/qwen-layer-under-budget.{json,stderr}` and
`artifacts/qwen-layer-parity.log`. Executable SHA-256:
`3042adc141aba7b07c2ee4efd3f66a6e37eb49ad92e7236a79371446b1e99e6f`.
Input hashes were rechecked and are unchanged from the selected-tensor section;
the local hash receipt is `artifacts/qwen-layer-identities.txt`.
Both canonical checks passed; receipts are
`artifacts/check-layer-default.log` and `artifacts/check-layer-metal-r2.log`.

Next: carry real hidden states through all layers, bound embeddings/output
weights and KV separately, and measure repeated load/evaluate/release cycles.
This result does not prove allocator reuse over a long generation or inference
for a model exceeding RAM. V4.1 still needs its own layout and numerical gates.

## Repeated lifetimes and BF16 conversion

At `aa99484`, `check-qwen-layer-metal --repeats N` runs 1–64 complete diagnostic
cycles in one process. Each cycle reinspects the checkpoint, creates fresh
synthetic input, loads and executes the same layer, and releases its arrays.
Only small reports survive between cycles. One cycle preserves the original
JSON shape; multiple cycles emit a `qwen3_selected_layer_cycles` envelope with
one report per completed cycle. A failure exits before emitting success JSON.
This is not a sequence of distinct model layers or an inner compute-only loop.

```sh
/usr/bin/time -l target/release/metallix check-qwen-layer-metal \
  --model /path/to/Qwen3-0.6B --layer 0 --tokens 3 \
  --max-weight-bytes 81798144 --candidate-only --repeats 64
```

Before optimizing, three fresh processes per cycle count gave these process
peak ranges on the same checkpoint and M3 Max:

| Cycles per process | Maximum RSS bytes, min–max | Peak footprint bytes, min–max |
|---|---:|---:|
| 1 | 111,017,984–111,214,592 | 288,391,792–288,588,400 |
| 8 | 113,836,032–114,606,080 | 291,160,736–291,914,376 |
| 64 | 114,835,456–115,064,832 | 292,143,752–292,389,536 |

The observed peak did not grow by one retained layer per cycle. This supports
bounded reuse for this fixed-shape diagnostic, not a universal allocator bound.
Different layers/shapes, growing KV and long generations remain unmeasured.
Each 64-cycle process requested 2,013,560,832 payload bytes in total; this is
not evidence of physical SSD traffic. Cache state was uncontrolled. The
candidate-only reports all say `not_run`; a separate two-cycle resident
comparison matched all 3,072 output values on each cycle. A 64-cycle command
with a budget one byte below the plan exited 1 and produced empty stdout.

### Profile and measured change

A headless `samply record --save-only --unstable-presymbolicate --duration 10`
capture around that 64-cycle command identified `decode_bf16` as 56.8% of
main-thread weighted self samples, `read` as 16.8%, and `_platform_memmove` as
8.2%. Symbols were resolved using the captured sidecar's exact address map.
Some system frames remain unresolved. These are sampled main-thread stacks,
not GPU timings or exact CPU-time accounting.

Two safe-Rust hypotheses were measured with three processes each. Discarding
cycle 1 leaves 63 warm observations per process; the table reports per-process
load medians, then their mean and sample standard deviation. Load timing covers
selected reads, conversion and array creation/evaluation, not header inspection
or block execution. No profiled run contributes to these timing results.

| Implementation | Process load medians, ms | Mean ± sample SD, ms | Decision |
|---|---|---|---|
| Early finite rejection + capacity/`push` | 12.689, 12.725, 12.608 | 12.674 ± 0.060 | Baseline |
| Accumulated finite check + capacity/`push` | 13.593, 13.519, 13.628 | 13.580 ± 0.056 | Rejected: slower |
| Accumulated finite check + fixed-size destination | 6.588, 6.412, 6.356 | 6.452 ± 0.121 | Kept at `49575cd` |

The kept change fills a pre-sized FP32 slice and accumulates the finite check
without early exit. It preserves little-endian bits, signed zero and rejection
of every NaN/infinity. It retains one FP32 destination, so the logical staging
plan is unchanged. This reduced measured warm layer-load time by about 49%;
it is not a 49% model-throughput result. Adjacent block-execution medians were
1.250–1.297 ms before and 1.266–1.297 ms afterward. Optimized process peak
footprints were 292,258,488–292,340,408 bytes, within the earlier range.

A second profile lowered conversion's weighted self-sample share to 26.4%;
reads were then 26.6% and memory copying 14.9%. Sampling proportions do not
prove vectorization or predict a further end-to-end speedup. The next loading
optimization needs its own profile and parity gate, not a speculative rewrite.

Validation: all 65,536 BF16 bit patterns checked individually, all finite
patterns also checked in a batch, and nonfinite values rejected at the start,
middle and end of batches. Real norm/Q-projection tensor comparisons and
layer 0 (twice)/layer 27 comparisons remained bit-exact. Both canonical
default and Metal checks passed after the change.

Receipts: `artifacts/qwen-cycles-{1,8,64}-{1,2,3}.{json,time}`,
`artifacts/qwen-cycles-{compare,budget}.{json,stderr}`,
`artifacts/qwen-bf16-{branchless,fixed}-{1,2,3}.{json,time}`,
`artifacts/qwen-bf16-timing-summary.json`,
`artifacts/qwen-bf16-fixed-{compare,layer27,norm,qproj}.{json,stderr}`,
`artifacts/check-bf16-fixed-{default,metal}.log`.
Profiles and symbol sidecars are under `artifacts/qwen-cycles-profile*` and
`artifacts/qwen-bf16-fixed-profile*`; their weighted sample summary is
`artifacts/qwen-bf16-profile-comparison.jsonl`.
Executable SHA-256 identities (same input hashes as above):

- Baseline: `3470c648a66c6bd6b018f1fcc040462edff5061683ebcd3ce60552ae4bccb881`.
- Rejected: `8c4191b9d69676aa69c7ec70acd21f878c24f0302a271b8649271524a394b078`.
- Kept: `213d1f4a6fbe50b892f1de90efa994e8eba22d5e9fe80f2e02f3205b5717436a`.

## V4.1 initial dimensions and live cache sources

At `e7fbd92`, [`V41ExecutionShape`](../../crates/models/deepseek/src/lib.rs)
adds an opt-in gate separate from metadata inspection:

```sh
target/release/metallix inspect-v41 --config /path/to/v41-config.json \
  --execution-shape
```

The pinned full configuration passed locally; its output is retained in
`artifacts/v41-initial-shape-check.txt`. This validates initial attention
and MoE dimensions, rotary-width constraints, compression-schedule length,
source IDs, latest-publisher ratio compatibility and candidate-source linkage.
It does not validate all Engram/CED/DSpark fields, tensor names, packed weight
ranges or numerical execution. It is not an executable load plan.

Source identity is the [pinned V4.1 record](../research/README.md#v41-source-identity).
The config projection in the test names its source revision and full config hash.
The locally inspected `inference/model.py` capture matched SHA-256
`4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
This pass read the relevant source functions, not the whole report again.

Two review findings became red/green regressions:

- `apply_rotary_emb` pairs adjacent values, and `Indexer.forward` rotates the
  index-head tail. A rotary width must be even and fit both attention and
  index heads; fitting only the attention head is insufficient.
- `Attention._compress_kv` and `_compress_topk_idxs` overwrite single shared
  slots. An older matching-ratio publisher cannot rescue a newer incompatible
  publisher. Changing layer 21 to ratio 2 after layer 20 publishes ratio 1
  exposed the original existential-owner check.

`cargo test -p deepseek rejects_unexecutable -- --nocapture` first exited 101
with both tests failing, then exited 0 after the checks were corrected.
The idea of rejecting a candidate source merely because it has no downstream
consumer was rejected: unused candidate work alone is not an execution error.

Next gate: complete representation/layout contracts and independent operator
references, then the reduced composed text-forward fixture. These shape checks
do not unlock the full V4.1 checkpoint download.

## Resident decode control

The unchanged resident decode path was measured with the
[existing benchmark methodology](qwen-metal.md#decode-measurements): three
32-token runs, 90 retained warm-decode observations, same model and raw inputs.
At checkout `e0c25f4`, median was 8.14494 ms, mean 8.28021 ms, sample standard
deviation 0.66523 ms; per-run medians were 8.14533, 8.12023 and 8.18590 ms.

The comparator accepted the workload/hardware/checkpoint match and identical
generated IDs against `artifacts/qwen-tooling-validation.json` (8.74083 ms
median). This is an observed control measurement, not evidence that the bounded
reader accelerates decode: that reader is not wired into generation.
Receipts: `artifacts/qwen-bounded-loader-control.json` and the comparator output
`artifacts/qwen-bounded-loader-comparison.json`.

The independent CPU/Metal parity suite was rerun for 1, 3, 17 and 64 tokens.
All four cases passed with zero mismatches across all 151,936 logits per case;
the maximum absolute error was `4.208087921142578e-05`. Receipt:
`artifacts/qwen-bounded-loader-parity.log`. This protects the resident control;
it does not qualify a streamed decoder that has not been implemented.

Default and Metal canonical checks passed, including strict Clippy and rustdoc.
Logs: `artifacts/check-bounded-loader-default.log` and
`artifacts/check-bounded-loader-metal-r3.log`. Earlier failed check logs remain
retained alongside them rather than being overwritten.
