# Bounded Qwen loading and V4.1 shape qualification

This ledger follows selected Qwen tensor reads through component checks to a
complete synchronous uncached streamed forward. It also records initial V4.1
configuration relationships. A weight pager, cached streamed generation,
V4.1 text inference and a process-wide memory ceiling remain unfinished.

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

The tiled projection experiment below advances the next qualification gate.
No speedup follows from this single lookup measurement.

## Tiled tied-output projection

[`qualify_projection`](../../crates/models/qwen/src/metal/projection_check.rs)
parses the forward configuration before accepting tied output weights. For
one deterministic synthetic hidden row, it reads each BF16 embedding tile,
widens to FP32, computes its logits, evaluates and copies those logits to the
CPU before releasing the tile and advancing. A separately loaded resident
MLX embedding supplies a complete-vocabulary reference.

```sh
target/release/mx check-qwen-projection-metal --model /path/to/Qwen3-0.6B \
  --tile-rows 1024 --max-bytes 8388608
```

On M3 Max, 128 GiB, macOS 26.6.2, three fresh processes per tile size were
run in repeating order 256, 1024, 4096, following one excluded 1024-row run.
All nine compared 151,936 logits with zero observed absolute error. The
acceptance criterion remains `5e-5 + 1e-4 * abs(reference)`, not a general
promise of bit-exact matrix arithmetic. Inputs are synthetic, not final
decoder hidden states. Every size exercises a partial final tile.

| Tile rows | Maximum raw tile | Candidate load + execute median | Observed range |
|---|---:|---:|---:|
| 256 | 512 KiB | 247.06 ms | 231.90–510.43 ms |
| 1,024 | 2 MiB | 125.18 ms | 119.06–171.47 ms |
| 4,096 | 8 MiB | 111.83 ms | 109.37–176.71 ms |

These are exploratory fresh-process timings, including per-tile evaluation
and readback. Only one shape received an excluded warmup; compilation,
allocator and OS cache state are not controlled. Variance is substantial.
Keep the default at 1,024 rows pending a controlled sweep; this is neither
an optimal-size claim nor a comparison against resident decode throughput.
`reference_ms` includes resident loading and casting, so it is not a
steady-state kernel baseline.

Every candidate still reads 311,164,928 raw bytes cumulatively. Tiling bounds
the selected raw slab, not total I/O or process memory. FP32 staging, array
copies, operator scratch, allocator retention, accumulated logits, headers
and the resident oracle are outside the raw budget. With 1,024 rows, the
exact 2,097,152-byte budget passed; 2,097,151 exited 1 before payload reading.

Both canonical checks passed. Receipts:
`artifacts/projection-{256,1024,4096}-{1,2,3}.json`, corresponding stderr,
`artifacts/projection-warmup.json`, `artifacts/projection-budget.stderr`,
`artifacts/projection-host.txt`, and `artifacts/check-projection-{default,metal}.log`.
Executable SHA-256:
`d0b297637a806394ab4824938b3fdaaa0e931f274e7fe76e76ac6555a63acae2`.
The checkpoint/config identities are the Qwen identities recorded above.

The complete-forward experiment below composes these pieces. KV reuse remains
a subsequent qualification gate.

## Complete synchronous streamed forward

[`qualify_streamed_forward`](../../crates/models/qwen/src/metal/stream_check.rs)
composes selected embedding rows, all decoder layers, final RMS normalization
and tiled output projection. Each layer is evaluated and its hidden state
copied to the host and reconstructed as a fresh MLX array before its weights
are released. This explicitly severs lazy graph dependencies. Shared loader
and operator helpers also remain used by the component diagnostics.

```sh
target/release/mx check-qwen-stream-metal --model /path/to/Qwen3-0.6B \
  --input-ids 1,2,3 --tile-rows 1024 --max-weight-bytes 81798144
```

On the same M3 Max, four initial fresh-process runs used raw IDs `1..=N` for
N = 1, 3, 17 and 32. Each executed all 28 layers and compared every one of
151,936 final logits with an independently loaded resident FP32 Qwen decoder.
All observed absolute errors were zero; the acceptance criterion is still
`5e-4 + 1e-4 * abs(reference)`, not guaranteed bitwise arithmetic.

The sequential logical weight/staging peak was 81,798,144 bytes. That exact
budget passed; one byte less failed during planning before payload reads.
The plan takes the largest embedding/layer/final-norm/projection stage, not
their sum. It excludes hidden activations, operator scratch, headers, allocator
retention and the resident reference. No process-peak measurement or
larger-than-host-memory claim follows from these results.

Candidate wall times in that initial order were 1,216.77, 861.48, 478.89 and
413.36 ms. Cache/compilation state was not controlled, so the decreasing times
are not a sequence-length scaling result. The 32-token run attributed 214.34 ms
to layer loading, 52.31 ms to layer execution/readback, 71.32 ms to projection
loading and 51.21 ms to projection execution/readback. This is one diagnostic
trace, not an optimized serving benchmark. It read 1,192,165,376 raw bytes
cumulatively. Resident-reference time includes loading and FP32 preparation
and is not a steady-state decoder baseline.

The independent CPU regression suite also passed its 1/3/17/64-token cases
after the shared-helper refactor. That suite runs the resident decoder; the
stream diagnostic separately checks the resident implementation. Inputs over
32 tokens and invalid vocabulary IDs are rejected by the stream diagnostic.

Initial receipts: `artifacts/stream-{1,3,17,32}.json`, their stderr files,
`artifacts/stream-exact-budget.json`, `artifacts/stream-budget.stderr`, and
`artifacts/stream-cpu-regression.log`. Initial executable SHA-256:
`9337298d0a6e0caa90d739671855d78f54ae1f8c995f33a5d9149839cea10e89`.
The checkpoint/config identities are recorded above. Both canonical checks
pass; logs are `artifacts/check-stream-{default,metal}.log`.

After improving the input-length error message, streamed runs using the first
three CPU-suite inputs (last vocabulary row, ordinary three-token text and
17 repeated tokens) also matched resident logits exactly. Three further fresh
processes on the same three-token input took 375.72, 384.61 and 375.00 ms in
the candidate stage (median 375.72 ms). These followed the earlier reads, with
no enforced cache state; they establish a repeatable diagnostic workload, not
generation throughput. Receipts: `artifacts/stream-oracle-{0,1,2}.json` and
`artifacts/stream-repeat-{1,2,3}.json`, with corresponding stderr. Executable:
`3507f68f2a1647a94fa81ed44bac13809e74f4dbfa385cd836f2b57a29f74205`.

The candidate-only measurement below advances the memory gate. Sampled cached
streaming and DeepSeek-V4.1 text-forward parity remain unfinished; this
Qwen result does not establish V4.1 support.

### Candidate-only process footprint

At `2d387fa` (Qwen) and `44d14bf` (CLI), `--candidate-only` uses the same
planning and numerical path without constructing the resident oracle.
Its distinct Rust report contains complete candidate logits
and `verification: "candidate_only"`, never comparison/error-count fields.
Non-finite logits fail with an indexed typed error before JSON serialization.

```sh
/usr/bin/time -l target/release/mx check-qwen-stream-metal \
  --model /path/to/Qwen3-0.6B --input-ids 9707,11,1879 \
  --tile-rows 1024 --max-weight-bytes 81798144 --candidate-only
```

Three serial fresh processes on the same M3 Max/checkpoint, 2026-09-11:

| Trial | Maximum RSS, bytes | Peak footprint, bytes | Candidate time, ms |
|---|---:|---:|---:|
| 1 | 117,948,416 | 295,060,152 | 586.590 |
| 2 | 118,816,768 | 295,912,096 | 371.754 |
| 3 | 118,931,456 | 296,043,192 | 367.804 |

Footprint median: 295,912,096 bytes; min–max span: 983,040 bytes.
Candidate-time median: 371.754 ms. OS cache and compilation state were not
controlled; the first run is not evidence of steady-state performance.
Process peaks include inspection, runtime/allocator overhead and output
serialization. Candidate time excludes planning and serialization. Neither
memory counter is the 81,798,144-byte logical weight/staging budget.

The earlier oracle-inclusive command measured 3,606,627,960–3,934,291,792 bytes
of peak footprint. That comparison demonstrates oracle contamination, not an
inference optimization: the commands perform different work. The candidate
still requests 1,192,105,984 payload bytes cumulatively for this input. This is
not measured physical SSD traffic or evidence of a checkpoint exceeding RAM.

A separate CPU float32 eager capture (Transformers 5.12.1, Torch 2.13.0,
one thread) checked all 151,936 emitted logits from each candidate process.
Maximum absolute error was approximately 4.10e-5, with zero mismatches under
`5e-4 + 1e-4 * abs(reference)`. Unlike the earlier transitive check, this
compares candidate output directly with CPU output. CPU loading ran outside
the measured processes. Input/config/checkpoint identities matched those above.
The capture command was `uv run scripts/qwen-reference.py --model
/path/to/Qwen3-0.6B --input-ids 9707,11,1879 --logits-output
artifacts/stream-memory-cpu.f32`; stdout was retained as the JSON manifest.
The comparison decoded this sidecar as little-endian FP32, checked equal vector
lengths and finite values, then applied the stated tolerance elementwise to
each JSON `candidate_logits` vector.
The default resident comparison still passed with zero observed error;
candidate-only budget 81,798,143 failed with empty stdout.

Receipts: `artifacts/stream-memory-before-{1,2,3}.{json,time}`,
`artifacts/stream-memory-candidate-{1,2,3}.{json,time}`,
`artifacts/stream-memory-cpu.{json,f32,stderr}`,
`artifacts/stream-memory-direct-parity.json`, and
`artifacts/stream-memory-{qualified-after,under-budget}.{json,stderr}`.
Candidate executable SHA-256:
`79b1a25df32c6fa7472bb027d166ccc2c854cf0258493ee8043bdc4b6a5083a5`.
Baseline executable is the preceding `3507f68...` identity.
Canonical Metal/default checks and release build passed; logs are
`artifacts/check-stream-candidate-metal-final.log`,
`artifacts/check-stream-candidate-default.log` and
`artifacts/build-stream-candidate-release.log`.

Next: measure repeated complete forwards and varying shapes before making an
allocator-stability claim, then qualify cached streamed generation and growing
state. Keep Qwen's sequential weight schedule adapter-local.

## Cached streamed prefill and appends

`check-qwen-stream-cache-metal` now checks a supplied prompt followed by known
append IDs, keeping detached FP32 K/V between layer-streamed steps. It compares
all 151,936 final-position logits at each step against both resident cached
and resident full-prefix execution. This is teacher-forced qualification, not
text generation or an independent CPU oracle by itself.

```sh
target/release/mx check-qwen-stream-cache-metal --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --decode-ids 4,5,6 \
  --max-weight-bytes 81798144 --max-kv-bytes 1376256
```

On the same M3 Max and Qwen checkpoint identified above, three sequential
release processes passed. Every cached-reference error was zero; the largest
full-prefix error was `0.000018119812`, within `5e-4 + 1e-4 * abs(reference)`.
Final retained KV was 1,376,256 bytes. Reducing that budget by one byte failed
before candidate execution with empty stdout. A second case using repeated
vocabulary-edge IDs (`151935,1,151935`, then `0,151935`) and 127-row projection
tiles also passed both controls. ID 151936 was rejected.
A 31-token prefill plus one append passed at the exact 32-token qualification
limit and 7,340,032-byte KV budget (largest full-prefix error `0.00005555153`);
a total of 33 tokens was rejected. This 32-token limit belongs to streamed
qualification; resident Qwen execution has a separate 512-token diagnostic cap.

Candidate timings in milliseconds (resident controls excluded):

| Run | Prefill 3 tokens | Append 4 | Append 5 | Append 6 |
|---|---:|---:|---:|---:|
| 1 | 572.28 | 352.60 | 341.59 | 340.71 |
| 2 | 376.65 | 342.74 | 340.42 | 353.85 |
| 3 | 547.31 | 346.04 | 434.97 | 377.80 |

These include loading, execution, host detachment and finite-logit validation.
File/compilation caches were not controlled. This establishes a baseline, not
a speedup: each step still reloads layer weights and the output projection.
The weight/staging budget and retained-KV budget exclude scratch, transient
copies, allocator retention, output vectors and resident controls. No total
process-memory or beyond-RAM conclusion follows from this command.

The first multi-token run failed despite a one-token prompt passing. Root cause:
MLX's Rust `as_slice` reads underlying storage without reordering a transposed
view. Rebuilding `[batch, heads, sequence, dim]` K/V from those bytes scrambled
sequence/head order. The fix materializes logical flattened order through MLX
before copying and rebuilding. The focused regression compares a nontrivial
transpose against an explicit scalar-order oracle, not another strided read.
All stream-detachment helpers now use that same copy path.

Receipts: `artifacts/cached-stream-1.{json,stderr}` (failed baseline),
`artifacts/cached-stream-short.{json,stderr}` (one-token control),
`artifacts/cached-stream-fixed-{1,2,3}.{json,stderr}`,
`artifacts/cached-stream-{under-kv,varied,invalid-id}.{json,stderr}`.
Context-boundary receipts are `artifacts/cached-stream-{context-limit,over-context}.{json,stderr}`.
Passing release executable SHA-256:
`2c26b556fe9e585b5d6dbaca4808cd4198fbcc0a4d7035a06f397933dadacfef`.
The independent Torch resident-forward regression also passed all vocabulary
logits for lengths 1, 3, 17 and 64, with zero mismatches
(`artifacts/cached-stream-cpu-regression.log`). This is a separate reference
check, not a direct Torch cached-stream measurement.

Next: join this path to sampling and measure repeated variable-length
lifetimes. Candidate-only measurement is recorded below. Keep the
DeepSeek compressed/sparse state layout separate from Qwen GQA K/V.

### Candidate-only cached process footprint

At `6036110` (Qwen) and `de7be36` (CLI), the cached command accepts
`--candidate-only`. It shares the exact preflight and candidate executor with
the qualification path but never loads resident controls. Its distinct report
retains per-step logits, timings and K/V sizes, with `verification: "candidate_only"`
and no comparison/tolerance fields. Finite-logit validation remains mandatory.

```sh
/usr/bin/time -l target/release/mx check-qwen-stream-cache-metal \
  --model /path/to/Qwen3-0.6B --input-ids 9707,11,1879 --decode-ids 4,5,6 \
  --max-weight-bytes 81798144 --max-kv-bytes 1376256 --candidate-only
```

Three serialized fresh release processes on the same M3 Max/checkpoint:

| Trial | Maximum RSS, bytes | Peak footprint, bytes | Sum of four candidate steps, ms |
|---|---:|---:|---:|
| 1 | 132,431,872 | 308,806,304 | 1,500.292 |
| 2 | 130,220,032 | 306,578,056 | 1,311.260 |
| 3 | 134,496,256 | 310,887,096 | 1,358.580 |

Median footprint was 308,806,304 bytes, with a 4,309,040-byte min–max span.
Process peaks include inspection, allocator retention, all four host output
vectors and JSON serialization. Candidate timings exclude planning and report
serialization. File/compilation caches and unrelated host activity were not
controlled; the timing spread is not a latency improvement claim.
Final retained K/V was 1,376,256 bytes, separate from the 81,798,144-byte
logical weight/staging plan. Neither budget bounds process memory.

Three prior oracle-inclusive runs peaked at 3,789,735,544–4,166,207,288 bytes.
Omitting resident controls isolates the measurement; it is not a model-memory
optimization or proof that a checkpoint larger than host RAM can run.

Separately captured Torch CPU FP32 full-prefix references checked every
151,936-element emitted vector for prefixes of 3, 4, 5 and 6 tokens in all
three runs. All were finite and had zero mismatches under
`5e-4 + 1e-4 * abs(reference)`; maximum absolute error was about `5.857e-5`.
Reference sidecar SHA-256 and input IDs were checked before comparison.
Fresh checkpoint/config hashes matched the identities recorded above.
The default resident-qualified command also passed after rebuilding.
Reducing the candidate-only K/V budget by one byte failed with empty stdout.

Receipts: `artifacts/cached-memory-{oracle,candidate}-{1,2,3}.{json,time}`,
`artifacts/cached-memory-cpu-<comma-separated-prefix>.{json,f32,stderr}`,
`artifacts/cached-memory-direct-parity.json`, and
`artifacts/cached-memory-{qualified-after,under-kv}.{json,stderr}`.
The retained comparison recipe is `artifacts/compare-cached-memory.mjs`;
CPU capture uses the existing `scripts/qwen-reference.py --input-ids ...`
with `--logits-output` outside the measured process.
Candidate executable SHA-256:
`e7e8b7f6ec74923763652af32191d0253ab68a85e48fff078bda5ebe948a0bb6`.
Prior oracle-inclusive executable SHA-256:
`106b8497cb020c04dc1d96737bb42275c8473b10bad2aba9bd65cebf6ecd312c`.
Default/Metal canonical checks, Rust 1.87 all-feature locked checking, and
release build passed (`artifacts/check-cached-rope-{default,metal-final}.log`,
`artifacts/msrv-cached-rope.log`, `artifacts/build-cached-rope.log`).

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
