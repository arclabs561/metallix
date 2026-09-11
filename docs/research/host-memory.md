# Models larger than host memory

This is an implementation direction, not a supported inference mode yet.
On Apple Silicon, CPU and GPU share host memory: moving weights from GPU to
CPU does not create another large memory tier. The proposed next tier is SSD.

## Evidence and limits

The full [LLM in a Flash paper, revision 3](https://arxiv.org/html/2312.11514v3),
including appendices A–G, was reviewed for this direction. Its transferable
lessons are to measure I/O separately from computation, combine small reads
into useful contiguous units, bound resident buffers, and exploit measured
reuse. It also reports costs from dynamic execution shapes and an unsuccessful
coactivation-bundling experiment. Its activation predictors depend on sparse
ReLU behavior and model-specific training; they are not a drop-in V4.1 method.
Its reported speedups do not predict performance here.

Two details matter when transferring that work: Appendix C hides dynamic
resident shapes inside shared-buffer kernels, and Appendix B reports quality
trade-offs from activation prediction. Appendix F discusses 4-bit loading but
does not implement the required kernels. Packed storage, exact execution, and
approximate prediction are separate qualification tasks.

[Fiddler, revision 3](https://arxiv.org/html/2402.07033v3) was also read through
Appendix F. Its exact router assignments feed a measured CPU-versus-GPU
execution choice. The decision depends on how many tokens reach an expert,
not just whether the expert is resident. However, its nonresident weights
still fit in CPU RAM; its PCIe cost model does not solve a unified-memory Mac's
SSD problem. Here, measure per-expert routed-token counts and fetch costs
separately for decode and prefill. Defer CPU execution until it earns a benefit
on the target Mac.

## Engineering evidence

The concrete warning from [tinygrad PR 17610](https://github.com/tinygrad/tinygrad/pull/17610)
is dequantization order: its contributor found that dequantizing all expert
weights before selection defeated sparse decode. The proposed patch selects
packed expert slices first, but retains the full-tensor path for symbolic
multi-token prefill. At review, the PR was closed without merging; its H100
timings and tests are contributor reports, not independently reproduced
Metallix results. The associated [issue](https://github.com/tinygrad/tinygrad/issues/17316)
remained open.

Our check should therefore count packed bytes fetched and decoded as expert
count grows with fixed top-K. Correct logits alone cannot establish sparse
execution. For prefill, count the union of selected experts across tokens;
that union can erase a decode-only residency advantage.

MLX's [allocator-cache limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_cache_limit.html)
governs free cached allocations, not eviction of live model arrays. Its
[wired-memory limit](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.set_wired_limit.html)
is another resource control, not a weight pager. Verify these APIs against
the pinned native substrate before adding a Rust wrapper. Do not substitute
allocator tuning for application-owned residency accounting.

For V4.1, qualify exact router-selected expert loading first. Any Engram fetch
unit must follow verified lookup semantics. Do not approximate routing, discard
weights, or change activations merely to reduce traffic. Keep each packed unit's
scales and other quantization metadata with the unit needed to execute it.

### Prefix reuse is a different cache problem

The complete [SGLang HiCache engineering post](https://www.lmsys.org/blog/2025-09-10-sglang-hicache/)
describes storage-friendly page-first layouts separate from compute-oriented
layer-first layouts, layer-wise load overlap, and configurable prefetch waits.
Selective write-through avoids backing up every KV page. These are useful
policy alternatives for a future prefix cache, not evidence that weight
offload is solved. Its benchmarks use distributed NVIDIA systems.

The complete [HiSparse post](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)
describes a hot KV buffer with top-K miss detection, LRU eviction, and page-table
updates. Importantly, it reports overhead at low concurrency and larger gains
when memory capacity limits batching. Its DSA examples do not qualify V4.1 CSA2
or Mac SSD-backed KV. Here, evaluate latency at concurrency one before adopting
a throughput-oriented offload policy.

The resulting experimental boundaries are:

| Resource | Fetch decision | Required correctness check |
|---|---|---|
| Packed expert weights | Actual expert routes; union across a microbatch | Resident versus fetched expert output, then full logits |
| Engram rows | Verified deterministic lookup addresses | Row contents, scale decoding, and lookup output |
| Prefix/attention state | Cache identity and adapter-specific KV dependencies | Prefix divergence, replay semantics, sharing and cancellation |

Keep these policies separate even if they eventually share a bounded file-read
primitive. A correct weight eviction does not imply that mutable KV can be
evicted or restored the same way.

## Current obstacles

The Qwen control adapter loads all shard arrays and expands them to float32.
Its forward path borrows the full weight map, while per-layer KV arrays grow
through concatenation. These are useful parity diagnostics, not an offload
implementation. The DeepSeek index parser inventories shards but does not yet
expose validated per-expert byte ranges.

The [selected-tensor diagnostic](../experiments/loader-qualification.md) now
qualifies Qwen's adapter-private range reader against the resident loader.
It bounds one raw payload, not the entire process. A subsequent
[one-block experiment](../experiments/loader-qualification.md#one-block-selected-weight-execution)
executes selected weights with a logical weight/staging plan and separately
measures candidate-only process memory. It uses synthetic hidden states;
full streamed forward remains unqualified. A later
[repeated-lifetime probe](../experiments/loader-qualification.md#repeated-lifetimes-and-bf16-conversion)
measured 1, 8 and 64 same-shape cycles; peak memory stayed near the one-cycle
working set, without establishing a bound for changing shapes or growing KV.
An allocation
must remain leased until every dependent GPU operation completes; dropping a
Rust handle is not itself proof of GPU completion.

Budget weights, KV, staging buffers, decoder scratch, and allocator retention
separately. Leave headroom for the OS. Admission must reserve the largest
in-flight working set, not just the steady-state cache size. Never raise global
memory limits or induce system swap as a substitute for a controlled pager.

## Measurement order

1. Measure file-read latency across chunk sizes using the existing checkpoint.
   Record requested bytes and cache mode. Buffered read throughput may be page
   cache throughput; even a no-cache hint is not physical SSD telemetry.
2. Compare a one-layer resident Qwen experiment with the fully resident control:
   exact logits, bytes read, peak allocations, load time, and decode time. A
   smaller resident budget emulates a capacity constraint without a huge download.
3. Qualify V4.1 expert and Engram slices against their reference operators.
   Replay actual routing traces before choosing cache policy or speculative prefetch.
4. Measure cache misses, useful and wasted prefetch bytes, stalls, and admission
   behavior over long generations. Add overlap only after synchronous correctness.

An optimistic bandwidth-only time floor is `miss bytes per token / sustained
read bytes per second`. Compute, decoding, read latency and scheduling add cost;
overlap cannot remove dependencies. A checkpoint fitting on SSD establishes
capacity, not usable token latency.

## Initial local read-size probe

On an M3 Max with 128 GiB RAM, three sequential invocations of
`scripts/benchmark-checkpoint-io.py --uncached` read the existing Qwen3-0.6B
checkpoint. Each invocation used seed 0 and 64 aligned random reads per size.
All completed; no global cache purge or memory-limit change was performed.

| Read size | Range of run median latency | Range of requested-byte throughput |
|---|---:|---:|
| 4 KiB | 0.115–0.124 ms | 28.6–32.7 MB/s |
| 32 KiB | 0.129–0.135 ms | 246.8–254.7 MB/s |
| 256 KiB | 0.203–0.208 ms | 1.245–1.282 GB/s |
| 1 MiB | 0.338–0.355 ms | 2.909–3.042 GB/s |

Throughput uses decimal units and sums time spent inside `pread`, excluding
loop overhead. These short, serial, repeated-offset workloads do not establish
sustained physical SSD throughput, optimal queue depth, thermal behavior, or
V4.1 token latency. `F_NOCACHE` was requested; cache misses were not independently
measured. The next comparison must include useful versus extra bytes from
coalescing and bounded read concurrency, using real tensor ranges.

Raw receipts are retained locally as
`artifacts/checkpoint-io-uncached-{1,2,3}.json`. They bind script identity and
file metadata, not checkpoint contents. The same seed deliberately repeats
offsets; these are repeatability measurements, not independent file samples.

KV offload, predictor-driven loading, speculative prefetch, and adaptive cache
replacement remain separate experiments. Compare each against a simple exact
baseline and retain it only with measured benefit and unchanged output semantics.
