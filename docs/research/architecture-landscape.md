# Architecture landscape and implementation position

Checked 2026-09-12. This is a bounded primary-source comparison, not a newest
model ranking or a performance evaluation of other runtimes. Model cards and
runtime support matrices are mutable; their claims below are upstream claims,
not independently reproduced Metallix results.

## September 29 update: evidence that changes the next gates

This update prioritizes September releases, with July/August context where it
changes a runtime contract. It is a bounded sample of ML directions relevant to
Metallix, not a comprehensive survey or a ranking. External measurements below
are upstream reports, not local qualification. Earlier sections retain their
original review scope.

| Direction and dated primary evidence | What is actually available | Metallix consequence |
| --- | --- | --- |
| Small specialized decision models: [Jeff v1.1, September 29](https://github.com/firelex/jeff/blob/f06788292874c21a5b5c41549ac220dd9e15da7f/README.md) | Released Qwen3.5/Gemma4 fine-tunes and serving/training code. The Qwen v1.1 release expands its trained option range; that does not establish calibration on our tasks. | Evaluate a trained decoder readout as a separate candidate after a pinned forward oracle. Existing Qwen3 direct option logits do not imply support. |
| Typed decision serving: [Ollama v0.35.0, September 28](https://github.com/ollama/ollama/releases/tag/v0.35.0) | A prerelease with `/v1/systemone` for choice, boolean-like `noul`, and ordered score questions; Nimble and Tev1 are named consumers. | A concrete external compatibility target now exists. Compare serialization, option ordering, probability normalization and score semantics before sharing a wire contract. No Ollama dependency or backend support claim follows. |
| Specialized objectives and data: [Nimble](https://ollama.com/library/nimble) and [Tev1](https://ollama.com/library/tev1) | Qwen3.5-based decision fine-tunes. Nimble documents contrastive examples differing in one deciding fact; Tev1 documents programmatic-policy and routing data. Their training/evaluation sets and option limits differ. | Add contrastive and option-order controls to any future artifact qualification. Keep calibration, task accuracy, numerical parity and API compatibility as separate receipts. Confidence concentration is not correctness probability. |
| Inference-time control beyond text: [World in World, September 10](https://arxiv.org/abs/2609.11548), [released implementation](https://github.com/Westlake-AGI-Lab/WorldinWorld) | A frozen video backbone receives visual evidence through attention K/V. The released subset supports camera rerendering, bullet time and editing. Its README reports Linux/A100-80GB, roughly 72 GB peak memory and 86 GB weights; streaming and long-video memory remain roadmap items. | Watch the explicit evidence/state interface as prior art for programmable inference. It is not a current Mac workload or a reason to begin a video backend. A smaller locally qualified artifact and named consumer would change that decision. |
| Backend-specific numerical/quantization contracts: [MLX v0.32.0, July 7](https://github.com/ml-explore/mlx/releases/tag/v0.32.0) | Released changes include an SDPA block-count override and explicit rejection of tensor-scale NVFP4 in Metal qqmm. | Preserve the pinned binding until a blocking operation warrants a separately qualified upgrade. A quantization family name alone does not establish kernel compatibility. |

Our inference from this sample is that useful capability is increasingly spread
across task-trained readouts, model-specific state, and controlled inference,
rather than a single text-generation interface. That supports typed task
boundaries and exact model-owned execution contracts. It does not justify a
universal tensor API, speculative scheduler, training framework or video stack
before the current model gates pass. Non-autoregressive/diffusion decoding
remains a watch item: this pass did not establish a fresh, locally reproducible
artifact that should displace the current priorities.

### Jeff is a distinct execution contract

Pinned implementation: `firelex/jeff` at
`f06788292874c21a5b5c41549ac220dd9e15da7f` (September 29).
Its [Qwen implementation](https://github.com/firelex/jeff/blob/f06788292874c21a5b5c41549ac220dd9e15da7f/src/jeff/model.py)
serializes each decision through the checkpoint's chat template and prompt
layout, validates single-token answer codes, left-pads, takes the last hidden
state, and applies a separately trained 255-way readout. It masks unavailable
slots, divides logits by a saved positive temperature and applies softmax.
The readout is initialized from selected LM-head rows but saved independently
in `readout.safetensors`; substituting the ordinary LM head changes the model.
The code capacity and the release's trained/served option limit are distinct.

This differs from both Qwen3's current selected-token scorer and Julia's
bidirectional marker-position decision head. Jeff also contains a separate
[ModernBERT pairwise cross-encoder](https://github.com/firelex/jeff/blob/f06788292874c21a5b5c41549ac220dd9e15da7f/src/jeff/encoder.py);
sharing request types does not make those graphs interchangeable. Its
[JevBench converter](https://github.com/firelex/jeff/blob/f06788292874c21a5b5c41549ac220dd9e15da7f/src/jeff/jevbench.py)
uses pinned older public-hard data and omits score questions. Treat its reported
hard result as that diagnostic, not a current official benchmark result.

Next admission test: pin the actual checkpoint, tokenizer/template, answer-code
mapping, trained readout, temperature and dtype; capture source tokenization,
masked logits and probabilities on a small fixed suite. Include option
permutation, option-count boundaries, padding, malformed artifacts and changed
evidence that flips a label. Benchmark through the chosen benchmark's actual
scorer with revision and dataset provenance. Do not add a backend enum or a
new dependency until this experiment has a real consumer.

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

## Verified recurrent mixers: state, order, and numerical boundary

The following notation fixes the state orientation used by the verified FLA
reference implementations: for each batch, recurrent/value head, and layer,
`S` has shape `[K, V]`; keys and queries have `K` components, values have `V`,
and the output is `o = Sᵀq`. This is a persistent, ordered state: a decode
step consumes and replaces it; it is not an append-only KV cache.

### Gated DeltaNet

The recurrent reference in FLA applies scalar decay before predicting the
current value:

$$
D_t = \alpha_t S_{t-1},\qquad
e_t=\beta_t(v_t-k_t^\top D_t),\qquad
S_t=D_t+k_t e_t^\top,\qquad o_t=S_t^\top q_t.
$$

Equivalently, $S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$.
The placement of $\alpha_t$ inside the prediction is load-bearing: using
`v - kᵀS_prev` without the preceding decay is a different recurrence. Here
$\alpha_t=\exp(\mathrm{logg}_t)$; the reference supplies $q$ already scaled
by $1/\sqrt K$. The recurrent path maintains FP32 state;
projection/output storage precision is a separate contract.

### Kimi Delta Attention

KDA retains the same `[K,V]` state but replaces GDN's scalar retention with a
per-key-coordinate gate. With `g_t` shaped `[K]`, let
$D_t=\operatorname{Diag}(\exp(g_t))S_{t-1}$, then apply the delta write
$S_t=D_t+k_t[\beta_t(v_t-k_t^\top D_t)]^\top$. Thus “left multiplication”
means row-wise key-coordinate decay in this orientation, not value-coordinate
decay. In grouped-value attention, the implementation distinguishes query
heads from value heads: the state is `[B, H_v, K, V]`, and the gate is
`[B,T,H_v,K]`; callers must not assume one state matrix per query head.

### Prefill, decode, and speculative state

Decode executes the recurrence in token order. Prefill may use the chunked
algorithms: their causal triangular/WY-like factors summarize within-chunk
rank-one updates, then propagate the terminal state between chunks. They are
not permission to reorder tokens. A speculative branch therefore needs an
independent state snapshot, replay, or a proven append-only update journal.
The last option is only safe if the stored gate domain and its noncommuting
row-wise/rank-one update order are retained; a generic “truncate updates” claim
is not established by these implementations.

The equations and shapes above were verified against the FLA naive recurrence
implementations, not a full-paper reread. FLA is pinned at
`516143e31fce09925e6c39ac37148444bad176c4`:
[GDN naive recurrence](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/gated_delta_rule/naive.py) and
[KDA naive recurrence](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/kda/naive.py).

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
