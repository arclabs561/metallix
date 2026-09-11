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
