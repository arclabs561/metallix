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

Next consumer: replace the projection stubs with qualified weight execution,
then join pre-RoPE index-key production, compressed-cache publication, sparse
attention, and the block harness. Full reduced logits remain the acceptance
gate; none of these captures alone establishes full-model parity or speed.
