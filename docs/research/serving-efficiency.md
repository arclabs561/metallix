# Serving efficiency methods

This is a technical reference for *serving* methods. It deliberately does not
specify Metallix behavior or claim that a candidate technique is implemented.
It excludes model quantization detail and Metal API/kernel implementation
detail, which belong in their respective references.

## Measurement before mechanism

Classify every result by phase. **Prefill** evaluates an input sequence in
parallel and is normally compute- and attention-heavy. **Decode** emits one
token per live sequence and is normally limited by parameter/KV memory
bandwidth. A single-stream decode rate is not service throughput; an upstream
CUDA result is not an M3 result.

For any serving change, collect time to first token (TTFT), inter-token latency
(ITL), prefill and decode tokens/s, completed requests/s, p50/p95/p99 latency,
active sequences, KV occupancy/hit rate, admission rejects/preemptions, and
cold versus warmed process/compile/cache state. Report the request mix, prompt
and output distributions, concurrency, model revision, runtime version, and
hardware topology.

Use these evidence labels:

- **Exact** preserves the target model's distribution or semantics, subject to
  ordinary implementation floating-point differences.
- **Approximate** changes a representation, routing, cache, or sampling
  distribution and needs an accuracy/tool-use gate.
- **Operational** helps only for a measured workload; it is not a universal
  speedup.

The multipliers below are results reported by their authors on their selected
models, runtimes, and hardware. They are not forecasts for Metallix.

## IO-aware attention, fusion, and graphs

FlashAttention is an exact attention primitive that tiles Q/K/V, performs
online softmax, and avoids materializing the quadratic score matrix in HBM.
It spends additional arithmetic on recomputation to avoid more expensive
memory traffic. The paper reports up to 7.6x attention-kernel speedup on GPT-2
and 3x faster GPT-2 training at sequence length 1K; its implementation and
measurements are CUDA/GPU-specific. The transferable lesson is to make data
movement explicit: fuse operations only where doing so removes intermediate
allocations or synchronization without breaking numerics or cache layout.
FlashAttention can improve prefill where an equivalent MLX/Metal kernel exists;
it does not remove decode's parameter-streaming cost.

Graph capture and compilation instead reduce dispatcher, launch, and Python
overhead. Current vLLM V1 documentation describes optimization levels,
compile-cache reuse, and CUDA graph capture. Its cache is valid only for the
matching model/config/environment/framework/GPU tuple. `--enforce-eager` is a
useful experimental control: it removes capture and compilation but trades
away steady-state decode performance. These are CUDA runtime features, not
Apple-Silicon capability claims. Pin the installed vLLM release and benchmark
eager versus warmed captured runs before treating a compile cache as
production-compatible.

For V4.1, a correct CED/CSA2/Engram forward path is prerequisite work. Generic
batching, prefix reuse, and structured-output experiments should first use
Qwen as a control. Model-specific grouped-expert/Engram kernels, speculation,
or graph capture must preserve V4.1 cache indexing and state semantics.

## KV cache as a scheduled resource

PagedAttention stores a request's KV state in fixed-size blocks addressed by a
block table, rather than in one contiguous maximum-length allocation. This
limits internal waste to a final block, prevents external fragmentation, and
allows block-granular copy-on-write sharing. The PagedAttention paper reports
that predecessor systems used only 20.4--38.2% of KV allocation for actual
token state under its experiment and reports 2--4x throughput at comparable
latency against its selected baselines. That establishes an allocation design,
not a current per-model performance guarantee.

Admit work against a conservative reservation: resident weights, current and
future KV, temporary prefill workspace, plus graph/cache headroom. Make each
hold, rejection, and preemption observable. Copy-on-write is safe only for
immutable full prefix blocks; a shared tail must be copied before appending.
On an M3, unified-memory and SSD/cache pressure must participate in admission;
macOS swapping is not a usable KV tier. On multi-GPU CUDA, account for tensor
or expert-parallel communication and topology before assuming that more shards
improve a small batch.

KV identity must include token IDs and every input which can change K/V:
model revision, tokenizer and chat template, system prompt, adapter,
multimodal preprocessing, RoPE/context configuration, and model mode. A text
prefix is not sufficient. Exact prefix reuse skips only repeated **prefill**;
it cannot accelerate the target's newly generated decode tokens. vLLM's APC
documentation makes this limit explicit, so it fits repeated long documents
and multi-turn histories but is not a generic decode optimization.

Eviction needs a stated policy and metrics. LRU is a reasonable general policy;
session preference helps active conversations but can reduce global hit rate or
starve new work. For hybrid models, cache/reuse rules are layer-specific: full
attention requires all prefix state, sliding-window attention needs the
retained tail, and state-space layers can require checkpoints. vLLM labels its
hybrid KV manager early-stage and documents grouping/padding limitations; pin
the version before relying on it as a contract.

KV quantization expands context/concurrency but is approximate unless an
architecture and kernel define an exact compressed form. Current vLLM FP8 KV
docs distinguish default, uncalibrated scales from calibration and note that
per-head support depends on the FlashAttention path. Pair memory/throughput
results with long-context retrieval, answer quality, schema validity, and
tool-call success. Those docs are CUDA/ROCm-facing; they do not establish an
M3 implementation.

## Continuous batching and prefill fairness

Iteration-level batching adds and retires sequences on each decode step instead
of waiting for a static batch's longest request. Paired with block KV storage,
it can keep capacity productive as output lengths vary. Chunked prefill places
part of a long compute-heavy prompt beside memory-heavy decode work. Current
vLLM V1 documentation says chunked prefill is enabled when possible, prioritizes
decode, and uses `max_num_batched_tokens` as the TTFT/ITL/throughput tradeoff:
smaller budgets favor ITL; larger budgets favor TTFT and throughput. Its
documented `>8192` throughput suggestion is for smaller models on large GPUs,
not a value to copy to Apple hardware.

Frequent recompute preemption is evidence that capacity estimation or admission
is wrong for the workload: it converts apparent concurrency into repeated
prefill and tail-latency loss. Separate interactive, background, and
long-context classes; cap queued work; reserve future output; and throttle new
long prefills when ITL misses. Sweep token budgets under a fixed trace rather
than optimizing a one-request benchmark.

## Radix reuse and structured generation

SGLang's RadixAttention retains reusable KV paths in a radix tree, matching
arbitrary shared token prefixes and making cache state visible to scheduling and
eviction. Its compressed finite-state machine (FSM) for constrained output can
emit a forced multi-token grammar path at once rather than sampling one token
at a time. The paper reports up to 6.4x throughput across its tested A10G/A100
workloads, models, and baselines; that aggregates workload-level methods and
does not establish a single-kernel or M3 multiplier.

Structured decoding can reduce token steps and parser retries, but grammar
construction/masking has overhead and unusual schemas may yield few forced
paths. Measure schema validity, tool-call repair rate, and end-to-end task time
as well as tokens/s. Current SGLang session-radix docs add an operational
constraint: a session ID is a soft reference to reusable KV, not prompt
reconstruction or a hard pin. Requests still carry the intended prompt, and
error/cancellation paths should close session references. The cache evicts
unreferenced data first and can still reclaim referenced data. This is a
candidate integration behavior only until the actual SGLang version is pinned.

## Exact speculative decoding

Speculative sampling uses a cheap draft to propose K tokens, scores them in a
batch with the target, then accepts from left to right with modified rejection
sampling. When the target correction and resampling are complete, the result is
exact with respect to the target distribution within hardware numerics. Shortcut
acceptance, lossy logits, or a different verifier is a separate, possibly
approximate method and must be named as such.

The original paper reports 2--2.5x distributed Chinchilla-70B decode speedup.
Actual benefit is acceptance times target batch efficiency minus draft and
verification overhead. Weak drafts, low acceptance under structured constraints,
or serial verification can lose. The method primarily targets decode; it does
not replace KV reuse or paging. Native MTP/DSpark paths require their own
model-specific verifier and cache semantics, so V4.1 cannot be presumed
compatible with a Qwen or generic draft path.

## MoE and storage offload

MoE adds routing-dependent expert selection, irregular expert batch sizes, and
possibly cross-device all-to-all traffic. Grouped GEMM/batched expert kernels
can reduce launch overhead; expert parallelism can spread capacity but adds
communication and load imbalance. Keep the router, shared experts, dense
attention, and a measured hot expert working set resident. Before increasing a
storage cache, observe expert reuse, queue depth, cache hits, bytes read, and
tail stalls. Lowering top-k, pruning, low-bit experts, approximate routing, or
aggressive eviction changes the model and requires a quality/tool-use gate.

Apple's *LLM in a Flash* is high-fit prior art for a storage-constrained Mac,
but not a generic MoE server. It uses activation sparsity, windowing to reuse
recently active neurons, and row-column bundling to convert small random reads
into larger contiguous transfers. The paper reports models up to twice DRAM
size and up-to-4x CPU, 7x Metal, and 20x NVIDIA gains over a naive offload
baseline. These are relative to naive repeated loading, not resident-model
performance. Its M1 Max experiments show sequential reads can exceed 6 GiB/s
while small random reads do not approach that bandwidth. Thus an SSD
expert/Engram candidate needs coalesced asynchronous reads and a measured hot
set; sparse active parameters alone do not imply usable 128GB V4.1 latency.

## Prioritized experiments

1. **Qwen control serving trace.** Use fixed short and 8K prompts while
   sweeping concurrency and token budget. Capture TTFT, ITL, throughput, KV
   peak, and preemptions. Disconfirm the batching hypothesis if there is no
   useful knee before memory pressure or p95 worsens at every setting.
2. **Exact prefix identity.** Compare identical requests to one-token,
   adapter, template, and RoPE changes. Measure saved prefill and false hits.
   Disconfirm it if a hit does not reduce prefill or a nonidentical request
   reuses KV.
3. **Structured output.** Run representative tool JSON/grammar traces and
   measure parser retries and schema/tool validity. Disconfirm it if no
   end-to-end improvement remains or semantic tool success falls.
4. **V4.1 feasibility microtrace.** Only after a correct forward path exists,
   run fixed 512/8K text and collect resident/SSD bytes, expert/Engram hits,
   decode stalls, and reference agreement. Disconfirm 128GB viability if a
   conservative unified-memory headroom cannot be maintained or SSD stalls
   dominate.
5. **Speculation after feasibility.** Sweep K and verifier batching with exact
   target-sampling parity and an acceptance distribution. Disconfirm it if
   acceptance times batching cannot pay draft/verifier overhead, or cache and
   structured-output semantics diverge.

## Sources and read coverage

Primary sources were read selectively: abstract, introduction, relevant method
and reported evaluation material; this is not a claim of exhaustive appendix
review.

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/html/2205.14135), 2022: IO-aware exact attention and reported evaluation.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/html/2309.06180v1), SOSP 2023: allocation, sharing, scheduling, and evaluation claims.
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/html/2312.07104), 2023: RadixAttention, compressed FSM, and evaluation claims.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/html/2302.01318), 2023: correction method and reported speedup.
- [LLM in a Flash: Efficient Large Language Model Inference with Limited Memory](https://arxiv.org/html/2312.11514), ACL 2024 / arXiv v3: storage cost model, bundling/windowing, and evaluation claims.

Supplementary runtime snapshots fetched on 2026-09-11 must be version-pinned
before implementation: [vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html), [automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html), [quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html), [hybrid KV manager](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager.html), and [SGLang session-aware radix cache](https://docs.sglang.io/docs/advanced_features/session_radix_cache).
