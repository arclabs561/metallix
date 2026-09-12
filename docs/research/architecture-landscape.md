# Architecture landscape and implementation position

Checked 2026-09-12. This is a bounded primary-source comparison, not a newest
model ranking or a performance evaluation of other runtimes. Model cards and
runtime support matrices are mutable; their claims below are upstream claims,
not independently reproduced Metallix results.

## Runtime position

Metallix currently executes Qwen3-0.6B with MLX through Rust, including cached
and layer-streamed generation, grammar-constrained sampling, and probability
diagnostics. The V4.1 adapter is still being assembled from qualified numerical
and representation references. It does not yet generate with V4.1, provide a
production concurrent server, or establish larger-than-RAM operation.
See the [current CLI](../../README.md) and [measured streamed path](../experiments/streamed-generation.md).

[MLX LM](https://github.com/ml-explore/mlx-lm) already provides Apple-Silicon
generation, quantization, and fine-tuning. [vLLM-Metal](https://github.com/vllm-project/vllm-metal)
uses upstream vLLM's API server, scheduler, and block manager, MLX model layers,
and a request-aware Metal attention path. Its current
[support matrix](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md)
includes Qwen3-Next and Qwen3.8 hybrids, with hybrid prefix-cache qualifications.
V4.1 is not listed in that matrix; that is not proof it cannot run through
another upstream path. This review did not execute those runtimes.

Our engineering inference: a Rust CLI or Mac support alone is not a sufficient
reason to rebuild serving. The useful experimental focus is exact V4.1
execution, model-specific memory residency, and observable generation control.
No comparative speed advantage has been demonstrated. Keep the existing
[pivot conditions](../architecture.md#pivot-conditions): upstream success can
turn this work into a backend/conformance contribution rather than a duplicate
standalone serving system.

## Model families require different state

| Family | State that persists between work units | Consequence for an executor |
|---|---|---|
| Dense causal GQA, our Qwen3 control | Committed token prefix and per-layer K/V | Conventional append-only cache is a useful baseline. |
| V4.1 CED / CSA2 / MoE | Encoder-derived shared global KV, sliding-window state, published sparse indices, residual mixing, and sparse weight/Engram accesses | Cache ownership and numerical dependencies cross layer boundaries. |
| Recurrent-attention hybrids | Attention KV plus ordered recurrent state in linear-attention layers | Prefix reuse, branching, and rollback must preserve both kinds of state. |
| Masked text diffusion | Revisable token canvas, mask/noise schedule, and model-private attention state | A work step can revise multiple positions; provisional output is not committed text. |
| Particle-based generation control | Multiple prefixes, model-state references, grammar/proposal state, log weights, RNG, and ancestry | Forking and resampling require explicit state ownership and probability accounting. |

The last row is an inference algorithm, not a competing neural architecture.

### V4.1

The [official card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
describes a 20-layer causal encoder followed by a 20-layer decoder. Decoder
global KV comes from final encoder states. CSA2 shares KV/index representations
and sparse selections across Full, Reindex, and Reuse layers. Engram performs
token-based conditional-memory lookup; mHC mixes residual streams. DSpark adds
trained draft machinery rather than a generic decoder switch.

Its reported 890 bytes/token is global-KV accounting, not total model memory
or SSD traffic. Sparse active-parameter counts likewise do not describe total
checkpoint capacity. On a Mac, bytes actually fetched, reuse, and transient
allocations must be measured independently. See the pinned source and
full-report reading anchors in [efficiency requirements](efficiency-methods.md).

### Hybrids and newer checkpoints

[Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) uses
Gated DeltaNet and gated attention in a 3:1 pattern, with sparse MoE.
The newer [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B) describes
the same 3:1 attention pattern across 64 layers, but dense feed-forward blocks,
a vision encoder, and trained multi-token prediction. Dense versus MoE and
attention versus recurrence are independent architecture choices. Neither
model is implemented or newly acquired by this research pass.

### Diffusion and probabilistic control

[LLaDA](https://github.com/ML-GSAI/LLaDA) and
[Dream](https://arxiv.org/abs/2508.15487v1) motivate an adapter-owned refinement
loop, not a forced `decode_one_token` interface. Blockwise sampling does not by
itself prove a trained block-causal mask or reusable causal cache. The
[diffusion note](diffusion-text.md) records source pins and reading limits.

[GenLM Control](https://github.com/genlm/genlm-control) instead operates over
probabilistic generation/control state. Legal-next-token masking is not the
same as sampling a desired whole-sequence distribution: proposal probabilities,
potential weights, resampling, and particle ancestry matter. See
[the control reference](genlm-control.md) for the mathematical and implementation
gates. These mechanisms remain future work, not features of `mx gen` today.

## Shared contracts without a universal tensor engine

Our inference from these cases is to share lifecycle, admission, cancellation,
capability reporting, and measurement conventions where implemented consumers
justify it. Keep physical state, cache validity, progress/commit semantics, and
probability meaning adapter-owned. A common Rust trait is useful only after
real implementations demonstrate that its operations have compatible meaning.

Immediate order remains quantized V4.1 graph composition, independent reduced
forward agreement, budgeted artifact acquisition, then real-request latency and
memory measurements. The architecture survey does not move those gates aside.
