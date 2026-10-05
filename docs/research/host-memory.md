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

## What bounds DeepSeek decode speed on one Mac

A byte-level cost model from the pinned configuration decides what matters.
Per decoded token, V4.1-Flash reads about 12.5 GB of weights across 40 layers:
about 160 MB per layer of attention projections (with `wo_a` as BF16 after
conversion), the 35 MB shared FP8 expert, the 3.9 MB router, and 113 MB for six
routed FP4 experts. The 8.0 GB of non-routed weights fit in memory; the 289 GB of
routed experts do not.

| Where expert bytes come from | Expert traffic per token | Decode floor |
| --- | ---: | ---: |
| All weights in unified memory (~400 GB/s) | 4.5 GB | ~31 ms, ~32 tok/s |
| Experts from SSD, no cache (~3 GB/s measured) | 4.5 GB | ~1.5 s, ~0.7 tok/s |
| 90% expert cache hit | 0.45 GB from SSD | ~180 ms, ~5.6 tok/s |
| 95% expert cache hit | 0.23 GB from SSD | ~105 ms, ~9.7 tok/s |

These are bandwidth floors, not predictions; they ignore compute, overlap and
read amplification. Two conclusions follow. Kernel speed matters only after
residency: the fused FP8 Metal kernel already runs a layer's attention
projections in about the time their bytes take to read. Expert residency and
read volume decide whether DeepSeek is usable: every percentage point of expert
cache hit rate above 90% is worth more than any further kernel work.

### Current work that addresses this bottleneck

Read from abstracts on the Hugging Face papers index; results are the authors'
claims until reproduced here.

- [Edge0 (2609.18063)](https://huggingface.co/papers/2609.18063) serves a 35B
  MoE from SSD at 20 tok/s in 3 GiB by training a per-layer prerouter that
  predicts the next layer's routing and then uses that prediction as the
  routing. That removes the dependency that keeps reads from starting early,
  but it changes the model and needs a recovery adapter.
- [Training-free halving of activated experts (2609.04575)](https://huggingface.co/papers/2609.04575)
  activates the top `k1` experts while normalizing by the top `k2` mass, keeping
  the expert branch's gain calibrated. It reports small quality loss from
  halving experts on Qwen MoEs. For V4.1 this would cut expert traffic in half,
  but it is an approximation and needs a held-out quality gate here.
- [EcoSpec (2607.12696)](https://huggingface.co/papers/2607.12696) observes that
  speculative drafts can route to disjoint experts and inflate the verified
  expert union, and selects drafts by predicted expert cost as well as
  acceptance.
- [ExFold (2608.24938)](https://huggingface.co/papers/2608.24938) projects
  excluded experts' contributions onto retained ones under a budget.
- [Lossy verification analysis (2607.26627)](https://huggingface.co/papers/2607.26627)
  shows relaxed speculative verification silently changes the sampling
  distribution; exact verification should remain the default.

V4.1 ships its own speculation path: three next-token-prediction layers
(`num_nextn_predict_layers = 3`) and a "DSpark" block drafter over layers
37-39 with 128 experts. Those are the natural drafters; no separate draft
model is needed.

### Closest prior art, from a wider arXiv search

A semantic arXiv search through Firecrawl's research index found much closer
work than the Hugging Face index, which changes what is worth proposing:

- [Speculating Experts (2603.19289)](https://arxiv.org/abs/2603.19289) predicts
  future experts from current internal representations to overlap transfers
  with compute, and reports that executing the predicted experts usually
  preserves accuracy.
- [SpecMD (2602.03921)](https://arxiv.org/abs/2602.03921) benchmarks caching
  policies and finds MoE expert access does not follow the temporal locality
  that LRU and LFU assume.
- [Reproducible MoE caching evaluation (2608.07911)](https://arxiv.org/abs/2608.07911)
  shows replay semantics and templated prompts can inflate recency policies by
  27-29% and invert policy rankings.
- [AcceptMoE (2608.02989)](https://arxiv.org/abs/2608.02989) selects verifier
  experts using commitment probabilities and, under offloading, cache
  residency; [MoE-Spec (2602.16052)](https://arxiv.org/abs/2602.16052) and
  [the limits of speculation (2609.22156)](https://arxiv.org/abs/2609.22156)
  budget experts and bound the gain from speculation in MoE.

### Proposed methods (hypotheses, not yet tested)

Revised after that search. Each lists its falsifying test; none is implemented.

1. **Exact lookahead prefetch.** Apply the layer-`N+1` router to the layer-`N`
   residual only to choose SSD reads, then route exactly and read misses. The
   prediction mechanism is the one in 2603.19289; what differs is the contract:
   never execute a predicted expert, so outputs stay identical to the source.
   Falsifier: on recorded V4.1 routes, the read time it hides must exceed the
   extra bytes it wastes on mispredictions.
2. **Exact speculation ranked by non-resident bytes.** Rank V4.1's own MTP and
   DSpark drafts by the bytes their verification would read from SSD given the
   exact resident set, and keep exact verification with full routing.
   AcceptMoE instead shrinks the verifier's expert set, which alters outputs;
   this proposal changes only which drafts are verified. Falsifier: accepted
   tokens per SSD byte versus acceptance-only draft selection on the same prompts.
3. **Frequency-first residency.** Given SpecMD's evidence against temporal
   locality, start from a static per-layer allocation of the most frequently
   routed experts measured on held-out prompts, with recency only as a
   secondary tier. Falsifier: hit rate at a fixed budget against LRU, using an
   event-atomic replay and varied prompt templates (2608.07911) so the
   comparison is not an artifact.
4. **No route-aware changes to sampled probabilities.** Any policy that alters
   which token is sampled to save reads is lossy, and is out of scope by
   default; 2607.26627 documents how such relaxations distort generation.

### Batches and probabilistic decoding under the same cost model

Because decode is bound by weight bytes, any work that shares one expert read
across several tokens is nearly free compute. Each extra token in a batch or a
verified draft only costs the experts its routes add to the union. That makes
batch prediction and exact probabilistic methods unusually attractive here, and
it changes how to evaluate them: the cost of a batch is the bytes of its
expert union, not its token count.

Relevant work (Firecrawl arXiv index):

- [Cacheable by design? (2608.18261)](https://arxiv.org/abs/2608.18261)
  measures Qwen3-235B decoding from SSD at 0.44 tok/s, matching a bytes-per-token
  model, with adjacent-token expert reuse 2.0x chance, 95% of traffic on 52.5%
  of experts, and a 13.4% LRU cache serving 66% of requests. It also reports a
  batching scheme collapsing at batch 32 from paging thrash. This is the closest
  published measurement to the DeepSeek setting and supports the cost model.
- [XShare (2602.07265)](https://arxiv.org/abs/2602.07265),
  [Opportunistic expert activation (2511.02237)](https://arxiv.org/abs/2511.02237)
  and [BASE (2609.36222)](https://arxiv.org/abs/2609.36222) re-route tokens
  toward experts the batch already loaded. They cut expert traffic but change
  outputs.
- Exact multi-draft speculative sampling ([SpecTr (2310.15141)](https://arxiv.org/abs/2310.15141),
  [SpecHub (2411.05289)](https://arxiv.org/abs/2411.05289),
  [Global Resolution (2511.15898)](https://arxiv.org/abs/2511.15898)) verifies
  several drafts per step while preserving the target distribution.
- Prefix-sharing parallel decoding ([Hydragen (2402.05099)](https://arxiv.org/abs/2402.05099),
  [Bifurcated attention (2403.08845)](https://arxiv.org/abs/2403.08845)) and
  [distinct-leaf enumeration (2604.20500)](https://arxiv.org/abs/2604.20500)
  make many samples from one prompt cheap and non-redundant.
- Sequential Monte Carlo steering ([2306.03081](https://arxiv.org/abs/2306.03081),
  [twisted SMC (2404.17546)](https://arxiv.org/abs/2404.17546),
  [self-distilled twisted SMC (2507.02315)](https://arxiv.org/abs/2507.02315)) already informs
  Metallix's [sampling gates](sampling-next-gates.md).

Hypotheses specific to an expert-streaming Mac (untested):

5. **Particles and parallel samples are cheap on an MoE.** SMC particles,
   best-of-n samples and self-consistency votes from one prompt share early
   routes, so their expert union grows sublinearly with particle count.
   Falsifier: measured expert-union bytes per step versus particle count on
   recorded routes; if the union grows nearly linearly, the advantage is small.
6. **Route-aware particle scheduling with exact weights.** When resampling SMC
   particles, advance first the particles whose next step reuses the resident
   expert set, deferring the rest within the same step. Ordering does not change
   particle weights or the target distribution; it only changes which reads
   overlap. Falsifier: SSD bytes per completed step versus fixed order.
7. **Exact multi-draft verification counted in bytes.** Score multi-draft
   trees from V4.1's MTP/DSpark heads by expected accepted tokens per
   non-resident byte, combining exact multi-draft verification (SpecTr-style)
   with the residency ranking in hypothesis 2. Falsifier: tokens per SSD byte
   against single-draft and acceptance-only multi-draft baselines.
8. **Shared-prefix batching across requests.** Serve concurrent agent requests
   as one batch per step so they share both the prompt prefix and each step's
   expert reads, with an explicit memory budget to avoid the paging collapse
   reported in 2608.18261. Falsifier: aggregate tokens per second and peak
   resident memory against serial serving at batch sizes 1-16.

Approximate methods that change routing (XShare, opportunistic activation,
expert halving) are worth measuring as opt-in modes with a held-out quality gate,
never as defaults.

The first measurement for all of these is the same: record actual V4.1 expert
routes per layer over real prompts, including several samples per prompt, then replay cache, prefetch and speculation
policies offline against those traces before building any of them.

## DeepSeek routed-expert traffic sensitivity

The pinned V4.1 configuration and inspected shard metadata make one capacity
constraint concrete before acquiring the checkpoint. These are metadata-derived
estimates, not measured DeepSeek decode or a completed residency plan.

At revision `dba1be0a40aa45a94ad051997016db3960a90277`, the configuration has
40 backbone layers, 384 routed experts per layer, six selected per token,
hidden width 5120 and expert intermediate width 2304. The inspected layer-six,
expert-zero header contains three packed I8 matrices, each with 5,898,240
payload bytes, and three E8M0 scale arrays, each with 368,640 bytes. This agrees
with `3 × 5120 × 2304 × (1/2 + 1/32) = 18,800,640` bytes per routed expert.
The [selected expert descriptor](../../crates/models/deepseek/src/checkpoint/source_fp4.rs)
checks this packed-weight/scale relationship; it does not load or qualify those
payloads.

| Quantity | Derived bytes | Interpretation |
| --- | ---: | --- |
| One routed expert, weights plus scales | 18,800,640 | Three projections; excludes host/device expansion |
| Backbone routed experts at this geometry | 288,777,830,400 | 268.95 GiB; excludes shared experts, Engram, attention and draft layers |
| One token, all 240 selected backbone experts missing residency | 4,512,153,600 | Useful expert payload demand before read amplification |
| Index-declared complete checkpoint payload | 510,286,023,000 | Metadata declaration, not verified shard sizes or acquired storage |

Using the earlier 1-MiB probe's 2.909–3.042 GB/s as a **hypothetical sustained
rate**, the all-miss expert payload alone would take 1.48–1.55 seconds per
token. That probe does not establish sustained bandwidth for these ranges;
this calculation is a sensitivity scenario, not a latency prediction.
At the same assumed rate, even a compute-free budget of five tokens/s permits
only 12.9–13.5% of selected expert bytes to miss residency; ten tokens/s permits
6.4–6.7%. Actual compute, Engram lookup, other weights and read amplification
tighten those budgets. Five and ten tokens/s are illustrative targets, not
adopted product requirements.

This makes measured routing locality and useful-byte residency prerequisites
for an interactive-serving claim. It does not establish an attainable hit
rate: a cache's fraction of stored experts is not its workload hit rate.
The Engram tables and their lookup locality still require a separate budget.
Next, replay source-derived routes against explicit RAM limits, measure reads
of real selected ranges, and settle the intended latency/context target before
choosing a pager or downloading the full checkpoint.

Evidence: the [pinned config](README.md#v41-source-identity), existing local
`v41-shard09-header-pinned.json` and `v41-index-pinned.json`; the calculation
receipt is `artifacts/deepseek-feasibility-metadata.json`. Its input hashes bind
the inspected metadata. The one inspected expert validates the arithmetic at
that boundary; extrapolation does not validate every expert header.

### Engram capacity and lookup traffic

Header-only HTTP range reads of pinned shards
[47](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/model-00047-of-00048.safetensors)
and [48](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/model-00048-of-00048.safetensors)
separate the Engram table budget from routed experts. Each request required
HTTP 206 and an exact `Content-Range`; only the eight-byte length prefix and
the declared header were read. No tensor payload was acquired.

| Engram layer | FP8 embedding bytes | E8M0 scale bytes | Total table bytes |
| --- | ---: | ---: | ---: |
| 1 | 98,305,579,008 | 3,072,049,344 | 101,377,628,352 |
| 14 | 98,308,270,592 | 3,072,133,456 | 101,380,404,048 |
| Both | 196,613,849,600 | 6,144,182,800 | 202,758,032,400 |

The two embedding tables occupy 188.83 GiB in the published encoding. Combined
with the extrapolated 268.95 GiB backbone routed experts, this leaves
18,750,160,200 bytes (17.46 GiB) of the index total for everything else. That
remainder is a subtraction, not a validated resident-weight inventory: it also
contains draft and vision parameters, shared experts, attention, projections
and other tensors. The index lists all six weight/scale tensor names for every
one of the 384 routed experts in each of the 40 backbone layers; their shapes
have not all been header-checked.

Capacity does not imply per-token table traffic. The pinned
[hash source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/inference/engram.py)
selects three n-gram sizes times eight heads in each of two layers. Each row
contains 256 FP8 bytes and eight scale bytes, so 48 row lookups require 12,672
useful embedding bytes per token before reuse. This excludes Engram projections,
decoding and physical read amplification. Separate weight and scale ranges can
make small random reads expensive even though the useful byte count is small.
Measure their page/range coalescing independently of routed-expert caching.

For perspective, allocating 32, 64 or 96 GiB **solely to packed routed experts**
would retain 11.9%, 23.8% or 35.7% of their payload. Under an explicitly
hypothetical uniform-independent routing workload, those fractions would imply
about 1.33, 1.15 or 0.97 seconds of expert reads per token at a sustained
3 GB/s, before computation. These are sensitivity scenarios, not cache hit-rate
or latency predictions; real skew and reuse could change them substantially.
The 96 GiB allocation is not an admission recommendation on a 128 GiB machine.

The immediate decision remains to measure source-derived routing locality and
reserve OS, non-expert weights, mutable state and scratch separately. Do not
equate table size with lookup traffic or stored expert fraction with observed
hit rate. Full-checkpoint acquisition remains gated by a concrete resource plan.

Reproduction identities: the unmodified shard-47 header is 656 bytes with
SHA-256 `e5c5c7fc900caec400644bef07aa5d7e28fc6ca941c8c47592b393963b218242`;
shard 48 is 664 bytes with SHA-256
`5d8fa4697d7a071c31baeee6947d493c08ea3243261c5d0f79c355ad191b8222`.
Read ranges are `0–7`, then `8–663` and `8–671`, respectively. The index hash is
`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`;
the inspected expert-header hash is
`139eeea4664aba161a4b4cb82a60a86a2429467601ee6584a887825f8137adfd`.
The fresh config and hash source matched the identities in the research index.

### Route evidence and the external baseline

The cached `mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed` snapshot is a
separate architecture: width 4096, 43 layers, 256 routed experts and affine
quantization, versus this target's width 5120, 40 backbone layers, 384 experts
and packed expert/E8M0 encoding. Snapshot revision
`10001e0065f8394e03e968e652cbbe7cd2ca122c`, config SHA-256
`44735712733fcf8f299bdf1faa1d87fac88f1917efe1d3876d6d4c582f79a68f`,
binds that distinction. Its local service alias does not identify the published
V4.1 target. External smoke throughput or routing cannot qualify these traffic
estimates; see the [baseline ledger](../experiments/deepseek-local-profile.md).

An additive route projector now extracts exact source-selected expert IDs from
the existing reduced capture without evaluating a router:

```sh
uv run scripts/v41_reduced_runner_capture.py \
  --input artifacts/v41-reduced-runner-source.json \
  --output artifacts/reduced-runner-projection.json \
  --synthetic-route-trace-output artifacts/synthetic-route-trace.json
```

It preserves source hashes, layer/start/offset, prefill/decode phase and ordered
top-k IDs. The fourteen rows cover layers three/four at starts 0/5/6 in a
four-expert, top-two synthetic graph. They qualify collector inputs, not real
model cache hit rates. Real locality remains gated on a source-compatible V4.1
router trace. Use such a trace to measure route-weighted misses and useful versus
requested bytes under an explicit RAM budget before selecting a pager or
acquiring full weights. The illustrative five-token/s target above requires
at least about 86.5–87.1% of selected expert bytes to hit residency even before
compute and read amplification; stored capacity fraction cannot establish that.

### First real V4.1 route trace

A recorder now runs the unmodified pinned V4.1 source on CPU over real
checkpoint tensors, fetching only the routed experts and Engram rows each layer
selects (exact HTTP byte ranges, checked against the shard index) and recording
every router decision. Its layer-zero choices match the earlier source capture
exactly for all three positions of the parity prompt. The first trace covers a
single 23-token coding prompt and 7 decoded tokens (1,200 router rows, 2,585
distinct layer/experts; 54.8 GB of expert payload fetched, within a 64 GiB
disk envelope). It ran at about 40 minutes per prompt on CPU, so it is a
locality probe, not a speed measurement.

Replaying the decode tokens after an empty cache sees the prefill
(18.8 MB per expert):

| Expert cache | LRU decode hit rate | Belady (offline optimum) |
| ---: | ---: | ---: |
| 2 GiB | 0.0% | 34.3% |
| 8 GiB | 33.0% | 68.9% |
| 16 GiB | 48.6% | 73.3% |
| 32 GiB | 66.4% | 73.3% |

Three facts bound what any cache can do on this trace. 65.2% of decode
selections reuse an expert the prompt's prefill already used; only 32.4%
repeat an expert the previous token chose at the same layer; and 26.7% are
first-ever uses of that layer/expert, which no cache can hit. Belady saturates
at 73.3% for exactly that reason. LRU needs about four times Belady's memory
for the same hit rate, which agrees with SpecMD's finding that recency is a
poor fit for expert access. Exact prefetch (method 1 above) addresses the
compulsory misses that caching cannot: they must be read, so the gain comes
only from overlapping the read with the preceding layers' compute.

With one prompt and seven decode tokens these rates are not yet held-out
evidence and do not pick a policy. The next trace adds four varied prompts
with 32 decoded tokens each, two used only to fit the frequency policy and two
held out, as 2608.07911 recommends.

### Offline route replay contract

`scripts/replay_v41_route_trace.py` now supplies the accounting step once a real
route trace is available. Both cache capacity and bytes per expert are required
inputs; no measured hit rate or operator budget is inferred from this machine's
capacity. It starts with an empty LRU keyed by `(layer, expert)` and reports
hits, misses, evictions, conditional useful miss bytes and prefill/decode totals.
Equal expert sizes and sequential accesses in the recorded order are explicit
simulation assumptions, not a chosen serving policy.

```sh
uv run scripts/replay_v41_route_trace.py --trace TRACE.json \
  --expert-cache-bytes BYTES --expert-bytes BYTES --output NEW_REPORT.json
```

The bounded schema-1 input carries the pinned revision, config and model-source
hashes, full 5120/40/384/top-6 geometry, and rows containing `request_id`,
`phase`, `token_position`, `layer` and ordered `expert_ids`. Each token must
cover layers 0 through 39. Rows retain the supplied chronological router-event
order, including layer-major prefill; they are never regrouped into token-major
order. A regression demonstrates that regrouping changes cache hits.

`--policy` selects `lru` (default), `belady` or `frequency`. Belady is the
offline optimum: it evicts the resident expert reused furthest in the future,
so it bounds every online cache policy at the same capacity. `frequency` keeps
a static set of the most often routed layer/experts, counted only on requests
named with `--calibration-request`; those requests are then excluded from the
replayed rows, so the comparison stays held out.

The report hashes the raw trace and labels its source identity
declared/unverified. Matching declared hashes cannot authenticate the collector
or its output. Synthetic reduced traces are rejected; a separately qualified
collector and real forward execution or externally obtained capture are still
needed. This tool reads no model payloads and measures neither SSD traffic nor
latency. Engram, dense weights, state, scratch, OS headroom and I/O amplification
remain outside its cache budget.
