# Efficiency requirements

This is the implementation reading list for Metallix, not a claim that these
methods exist in the runtime today. The papers were read through their complete
primary PDFs before the requirements below were recorded.

## V4.1 execution

The [DeepSeek-V4.1-Flash technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf)
defines the first adapter's non-negotiable layout:

- CED divides the 40 layers into a 20-layer encoder and 20-layer decoder. The
  decoder global KV is projected from final encoder states, so it is not a
  conventional decoder-only KV cache.
- CSA2 uses statically assigned Full, Reindex, and Reuse modes. The adapter
  must represent shared global KV, indexer K, and Top-K index ownership
  explicitly rather than allocating one independent cache tensor per layer.
- Global KV is the durable prefix-cache candidate. Bounded sliding-window
  replay is intentionally approximate, not exact cache reconstruction. Qualify
  the report's replay behavior separately from full-forward numerical parity.
- The reported 890 bytes/token covers accelerator-resident global KV after
  CSA2 and FP4 compression. It excludes other allocations and is not an SSD
  traffic metric or generic engine constant.
- Global-KV FP4 uses quantization-aware training and a specific E2M1 encoding
  with per-16-channel E4M3 scales (§2.4.4). A generic 4-bit conversion is not
  qualified by that result; test the actual representation and scale decoding.
- DSpark requires trained draft blocks and Markov/confidence heads, not just
  an engine switch. After a correct greedy and sampled path, measure acceptance,
  draft and verification cost, and output-distribution parity.

Reading anchors: CED §2.2, CSA2 §2.3, Engram §2.4.2 and §3.1.3,
DSpark §2.4.3, persistent cache and bounded replay §§3.2.1–3.2.2.
The pinned report's SHA-256 is
`ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.

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

1. Use the implemented Qwen3 BF16-to-FP32 Metal path as the resident control.
   Compare bounded tensor loading against the checked-in raw-token reference
   at `fixtures/qwen3-0.6b/forward-reference.json` before adding overlap.
2. Run `scripts/benchmark-openai.mjs` against both the reference server and
   Metallix using 512, 8k, and 32k cold/shared-prefix profiles once Metallix
   streams tokens.
3. Add a V4.1 tensor-slice reader and parity fixture. Only a passing fixture
   permits a bounded artifact acquisition experiment.

MTP/DSpark, SSD expert/Engram prefetch, fusion, and KV quantization are
measured follow-ons. They are not substitutes for the first parity gate.

The [host-memory investigation](host-memory.md) records the full Flash/Fiddler
readings, current implementation reports, and the initial local I/O curve.
Use [DEVELOPMENT.md](../../DEVELOPMENT.md) for repeatable checks and comparisons.
