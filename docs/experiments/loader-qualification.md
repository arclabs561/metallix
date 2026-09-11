# Bounded tensor and V4.1 shape qualification

This pass qualifies selected Qwen tensor reads and initial V4.1 configuration
relationships. It does not implement streamed layers, a weight pager, V4.1
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

Next gate: one-layer streamed execution, then full-logit and cached-forward
parity with a measured total working set. Passing selected reads does not
satisfy that gate. Nested JSON duplicate-key handling remains a parser follow-up.

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
