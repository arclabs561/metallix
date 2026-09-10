# Efficiency requirements

This is the implementation reading list for Metallix, not a claim that these
methods exist in the runtime today. The papers were read through their complete
primary PDFs before the requirements below were recorded.

## V4.1 execution

The [DeepSeek-V4.1-Flash technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
defines the first adapter's non-negotiable layout:

- CED divides the 40 layers into a 20-layer encoder and 20-layer decoder. The
  decoder global KV is projected from final encoder states, so it is not a
  conventional decoder-only KV cache.
- CSA2 uses statically assigned Full, Reindex, and Reuse modes. The adapter
  must represent shared global KV, indexer K, and Top-K index ownership
  explicitly rather than allocating one independent cache tensor per layer.
- Global KV is the durable prefix-cache candidate. Sliding-window KV is
  short-lived; a future persistent cache must test bounded replay after a
  global-cache hit instead of silently treating that reconstruction as exact.
- The report's reported 890 global-KV bytes/token is a model-specific target,
  not a generic engine constant. Metallix must measure resident bytes and SSD
  bytes per generated token against it on the actual adapter.
- DSpark comes after a correct greedy and sampled decode path. Its acceptance
  rate, extra draft work, and output-distribution parity decide whether it is
  retained.

The first V4.1 loader gate therefore expands from configuration validation to
a small text-only forward parity fixture. It must verify CED state flow, the
CSA2 mode schedule, sparse Top-K selection, and the FP4/FP8 tensor decoding
before any serving or full-checkpoint claim.

## Shared serving mechanics

[PagedAttention](https://arxiv.org/pdf/2309.06180) separates logical token
blocks from their physical KV allocation. Its transferable requirement is a
block table with reference counts and copy-on-write for shared prefixes. It is
not permission to assume V4.1 has Qwen-shaped KV pages: each adapter supplies
its physical page bytes and valid sharing operations.

[Speculative decoding](https://arxiv.org/pdf/2211.17192) makes the important
trade-off explicit: a draft improves latency only when its acceptance rate and
cost outweigh verification. For every future speculative path, record draft
length, accepted tokens, target verification time, and total bytes moved. Do
not report a speedup from a greedy-only comparison when sampled output parity
is the intended contract.

## Near-term experiments

1. Implement dense Qwen3 BF16 loading and single-request prefill/decode via
   the selected native Metal substrate; compare logits on the checked-in fixed
   raw-token reference at `fixtures/qwen3-0.6b/forward-reference.json`.
2. Run `scripts/benchmark-openai.mjs` against both the reference server and
   Metallix using 512, 8k, and 32k cold/shared-prefix profiles once Metallix
   streams tokens.
3. Add a V4.1 tensor-slice reader and parity fixture. Only a passing fixture
   permits a bounded artifact acquisition experiment.

MTP/DSpark, SSD expert/Engram prefetch, fusion, and KV quantization are
measured follow-ons. They are not substitutes for the first parity gate.
