# Training efficiency: measured memory economics

This is a training research and validation plan, not a claim that Metallix
trains models today. It separates mechanisms that can help a single Apple
Silicon Mac from distributed techniques that become relevant only after a
future multi-accelerator pivot. Serving cache optimization is a separate
problem: saved training activations are not an inference KV cache.

## Memory is an accounting problem before it is a kernel problem

Start with a measured peak, but use a budget to identify the likely term to
attack. Let `P` be base-model parameters, `Q` be trainable adapter parameters,
`b_w`, `b_m`, `b_g`, and `b_o` be bytes per stored weight, master weight,
gradient, and optimizer state, `A` saved activations, and `T` temporary kernel,
allocator, and framework memory. A rough full-fine-tune model is:

```text
M_full ~= P * (b_w + b_m + b_g + b_o) + A + T
```

For example, a conventional mixed-precision Adam setup might retain 2-byte
model weights, 4-byte master weights, 2- or 4-byte gradients, and two 4-byte
moments. That is an illustrative assumption, not a framework invariant;
measure the optimizer and dtype actually selected.

Activations scale chiefly with the work shape:

```text
A ~= c * layers * microbatch * sequence_length * hidden_size * activation_bytes
     + attention_workspace
```

`c` depends on what autograd saves. A naive attention implementation can also
materialize a score-like `O(sequence_length^2)` temporary; a memory-efficient
attention kernel may avoid that materialization. Allocator *reserved* memory,
*allocated* memory, and process peak are different facts and should never be
silently substituted for one another.

For adapter tuning, the corresponding model is:

```text
M_adapter ~= P * b_base
           + Q * (b_adapter + b_master + b_grad + b_opt) + A + T
```

This makes the first question concrete: is the failure from trainable state,
activations, attention workspace, or an unprofiled temporary? Reducing the
wrong term can make a configuration more complicated without making it fit.

## Rematerialization trades time for activation capacity

Checkpointing saves fewer forward intermediates and recomputes selected work
during backward. In its idealized linear-chain analysis, Chen et al. split a
network of `n` operations into segments of length `k` and obtain approximate
saved-feature memory `O(n/k + k)`; `k ~= sqrt(n)` gives sublinear memory with
extra forward work. The result is valuable directionally, but real transformer
graphs have unequal layers, fusion, attention workspaces, and allocator peaks.
The paper also distinguishes feature-map storage from parameter and temporary
memory, so its result is not a whole-process capacity formula.
[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/html/1604.06174)

Current PyTorch checkpointing reruns forward work in backward. Use an explicit
`use_reentrant=False`: the non-reentrant variant records an autograd graph and
can stop recomputation after regenerating needed intermediates. Preserving RNG
state is normally necessary for dropout-equivalent results, though it carries
overhead; disabling it changes the numerical contract and needs a deliberate
test. [PyTorch checkpoint documentation](https://docs.pytorch.org/docs/stable/checkpoint)

JAX exposes the same choice through `jax.checkpoint`. Its documentation notes
that XLA may already remove some saved values under `jit`, while staged control
flow such as `lax.scan` can constrain those optimizations. Wrapping a whole
outer function that saves no useful residuals does not create a benefit.
[JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html)

On one Mac, qualify whole-transformer-block policies first: none, every block,
and alternating blocks. For an identical seed and batch, require logits and
gradients to agree within a stated tolerance. Then compare peak memory,
steady-state supervised tokens/s, and p95 step time. A lower peak that halves
useful throughput is a capacity trade, not an unconditional optimization.

## Microbatching and precision control the local execution shape

With data-parallel world size `D`, nominal effective batch size is:

```text
B_effective = B_micro * accumulation_steps * D
```

Today `D = 1`. Gradient accumulation can lower activation memory by reducing
`B_micro`, at the cost of more forward/backward passes between optimizer
updates. Correctly normalized accumulation can approximate a larger batch, but
is not always identical: dropout/randomness, batch-dependent layers, input
ordering, clipping timing, and scheduler step frequency may differ. Record
microbatch, accumulation count, optimizer steps/s, supervised tokens/s, loss,
gradient norm, clipping, and skipped/non-finite updates.

Mixed precision performs eligible work in a lower precision while retaining
selected accumulations and optimizer state in a wider representation. FP16
loss scaling computes `L' = S * L`, backpropagates scaled gradients, unscales
before clipping or update, and lowers/skips on non-finite results. It protects
small gradients from underflow, but a too-large scale can overflow.
[NVIDIA Mixed Precision Training User Guide](https://docs.nvidia.com/deeplearning/performance/pdf/Training-Mixed-Precision-User-Guide.pdf)

BF16 often needs less loss-scaling intervention because of its exponent range,
but Metal dtype support, accumulation precision, and chosen kernels are
backend-specific. Treat every claimed Mac precision win as a qualification:
same data and seed, a short loss curve, non-finite counters, peak memory, and
held-out quality. A successful first step does not establish a stable regime.

## LoRA is the primary single-host adaptation lever

LoRA freezes `W_0` and trains a low-rank update:

```text
W = W_0 + BA
B in R^(d_out x r), A in R^(r x d_in), r << min(d_out, d_in)
```

For a dense matrix, trainable elements become `r(d_in + d_out)` rather than
`d_in * d_out`. This reduces adapter gradients and optimizer state, while the
frozen base still occupies memory and participates in forward compute and
activation production. The original study found small ranks effective on its
evaluated tasks, but explicitly does not claim that every task has a suitable
small-rank update. [LoRA: Low-Rank Adaptation of Large Language
Models](https://arxiv.org/html/2106.09685)

When the base model fits, LoRA is therefore the first adapter method to
qualify on a Mac. Sweep target modules and rank under one fixed data and
evaluation contract; compare validation quality, trainable count, peak memory,
tokens/s, and adapter merge/load behavior. A small rank grid can start at 8,
16, and 32 when architecture and task make those plausible, but no rank is a
universal default.

QLoRA holds a frozen quantized base and backpropagates through dequantized
compute into LoRA adapters. Its NF4/double-quantization storage design and
NVIDIA-unified-memory paged optimizer demonstrate a particular CUDA-oriented
system, not an equivalent Metal implementation. [QLoRA: Efficient Finetuning
of Quantized LLMs](https://arxiv.org/html/2305.14314) Keep its detailed
quantization and QAT issues in [quantization-precision.md](quantization-precision.md).
The training conclusion here is narrower: QLoRA changes base residency and
adapter state only when the format, dequantization kernel, and numerical path
are actually supported. Qualify LoRA first; do not directly port CUDA paging
assumptions to Apple unified memory.

## Distributed methods are a deliberate future pivot

ZeRO shards optimizer state, then gradients, then parameters across
data-parallel ranks. Its savings are coupled to collectives: its analysis
replaces ordinary data-parallel synchronization with reduce-scatter and
all-gather work. [ZeRO: Memory Optimizations Toward Training Trillion Parameter
Models](https://arxiv.org/html/1910.02054) PyTorch FSDP implements a
ZeRO-stage-3-inspired parameter-sharding approach and documents behavioral
limits around CPU-offloaded accumulation, mixed frozen parameters, double
backward, and forwarding modules outside its wrapper.
[PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html)

With a single rank, FSDP/ZeRO has no physical state to shard and introduces no
current capacity benefit. Preserve a clean model/config boundary so a later
multi-device experiment can target a demonstrated bottleneck.

| Method | Main relief | Principal new cost | Use only when |
|---|---|---|---|
| FSDP / ZeRO | replicated parameter, gradient, optimizer state | collective traffic and transient gathers | state exceeds a device |
| Tensor parallel | layer weight/compute per rank | intra-layer collectives | a layer is too wide |
| Pipeline parallel | model state per stage | bubbles and microbatch scheduling | layers partition cleanly |
| Context parallel | long-context activation/attention state | sequence collectives | context, not weights, is limiting |

PyTorch labels tensor-parallel APIs experimental and its examples are
CUDA-oriented. [Tensor parallel documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
Its pipeline framework requires staged processes and microbatches, with
experimental composability. [Distributed pipelining documentation](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
Context-parallel support is likewise experimental.
[Distributed tensor documentation](https://docs.pytorch.org/docs/stable/distributed.tensor.html)

## Attention, inputs, and compiler measurement

Attention kernels can change both temporary memory and time. Prefer a
framework-supported fused or memory-efficient path only after proving
logit/loss/gradient agreement for the real causal mask, position encoding, and
sequence distribution. A synthetic square-attention result is not a training
result.

Packing can improve useful-token density, but only if attention boundaries,
position IDs, labels, and loss masks prevent cross-example leakage. Input
workers can avoid blocking compute, but their count and prefetch depth should
be profiled rather than inherited from CUDA examples. [PyTorch data loading
documentation](https://docs.pytorch.org/docs/stable/data.html)

Compilation must report first compile, recompiles, and steady state separately.
PyTorch recommends profiling graph breaks; dynamic shapes can trigger
recompilation and erase an expected gain. [PyTorch compiler profiling
guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)
Keep an eager baseline and measure post-warmup median and p95 step time rather
than treating first-run latency as model throughput.

Use this primary throughput metric:

```text
useful_train_tokens_per_second =
  non-padding, loss-supervised tokens / steady-state wall time
```

Every receipt should include model revision, seed, data slice, sequence policy,
packing, microbatch, accumulation, dtype, optimizer, checkpoint policy,
loader configuration, compiler state, warmup versus measured windows, peak
memory, loss, gradient norm, non-finite events, and validation quality. This
makes a lower-level memory or kernel claim falsifiable rather than merely
plausible.

## Single-Mac qualification order

1. Establish a fixed eager baseline with a warmup and measured window.
2. Eliminate padding or loader starvation while checking loss-mask correctness.
3. Find the largest stable microbatch, then add accumulation to reach an
   intentional effective batch.
4. Sweep block-level rematerialization with output and gradient parity gates.
5. Compare supported attention and precision paths with stability checks.
6. Sweep LoRA target modules and ranks; assess QLoRA only after backend support
   and numerical parity exist.
7. Compile only after the workload shape is stable.
8. Revisit FSDP, TP, PP, or CP only for a measured multi-accelerator need.

## Source and reading coverage

The four original papers below were the bounded deep-reading set. No claim is
made that every paper in the wider literature was read.

| Source | Version and portions read | Coverage note |
|---|---|---|
| [Chen et al., *Training Deep Nets with Sublinear Memory Cost*](https://arxiv.org/html/1604.06174) | arXiv v2; abstract through §5 and Appendix A | Full HTML body and available appendix read |
| [Rajbhandari et al., *ZeRO*](https://arxiv.org/html/1910.02054) | arXiv v3; introduction, memory/communication analysis, stages, implementation/evaluation excerpts | Core analytical sections read; not every evaluation table |
| [Hu et al., *LoRA*](https://arxiv.org/html/2106.09685) | arXiv v2; method, experiments, rank/target analysis, Appendix G excerpts | Main method and relevant evaluations read; not every appendix subsection |
| [Dettmers et al., *QLoRA*](https://arxiv.org/html/2305.14314) | arXiv HTML; §§2–5 and relevant Appendix A/B/G excerpts | Storage/compute method read; not every evaluation appendix |

The PyTorch and JAX citations above are current versioned practice sources,
rather than claims about an unchanged API across releases. Re-check their
release-specific behavior before binding Metallix tooling to one framework.
