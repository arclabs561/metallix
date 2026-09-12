# Diffusion text models: a separate generation family

**Decision:** Metallix could support a text diffusion model through the proposed
adapter-owned denoising/refinement loop.  Do not assume that the current
token-step decoder, logical KV pages, next-token logprob, grammar-mask, or SMC
contracts apply.  DeepSeek-V4.1-Flash remains the implementation priority; this
is a qualification plan for a later, distinct adapter.

## What is available

This is a bounded source review checked 2026-09-11, not an exhaustive inventory
or a claim to identify the newest diffusion model.  Links below are primary
project, model, or paper sources.  Git repository pins are the observed `HEAD`
at review time, not release tags.  The LLaDA and DiffuLLaMA source licenses
were not established in this pass; public availability does not grant copying
rights.

| Family | Availability and pin | Generation shape | Local-backend consequence |
| --- | --- | --- | --- |
| [LLaDA / iLLaDA](https://github.com/ML-GSAI/LLaDA/tree/9182493720ed723ef8031210d85959364e51cbe0) | Official PyTorch repository, observed at `9182493`; its README says it released LLaDA 8B Base/Instruct weights, later iLLaDA 8B weights, and LLaDA-MoE-7B-A1B weights.  The same README requires Transformers custom code and BF16 for its example.  Exact model and code licenses remain a pre-acquisition check. | Full-length masked discrete diffusion.  The [paper](https://arxiv.org/abs/2502.09992) describes a random forward masking process and a reverse Transformer that predicts masked tokens; the repository says sampling uses a fixed context and does not yet use KV cache. | No safe inference that the Python/Transformers reference or its cache behavior maps to Metal.  iLLaDA is a useful same-family follow-up because its project says the inference code is reusable with `mask_id=5`; it is not a shortcut to accepting its checkpoint format. |
| [Dream 7B](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B/tree/05334cb) | The observed Hugging Face revision is `05334cb`; the model card labels it Apache-2.0, Safetensors, `custom_code`, and requires `trust_remote_code=True` in its Transformers example.  It provides four weight shards and custom model/tokenizer Python. | Parallel iterative discrete denoising, with AR initialization and context-adaptive token-level noise rescheduling in the [v1 paper](https://arxiv.org/abs/2508.15487v1).  The paper describes arbitrary-order generation and infilling, rather than a left-to-right decoder. | It is the first real-model candidate only after a source audit of the pinned custom code and a checkpoint/config manifest audit.  The supplied examples mention vLLM/SGLang and CUDA-oriented containers; they do **not** establish a Metal path. |
| [Mercury](https://www.inceptionlabs.ai/blog/introducing-mercury) | Commercial API/on-premise family.  No public weights were verified in this review.  Its [technical report](https://arxiv.org/abs/2506.17298) describes Mini and Small Coder models and a public API; the company announcement promises API/on-prem access. | A Transformer predicts multiple tokens in parallel over a coarse-to-fine denoising process. | Treat it as a systems and product comparison point only.  Its published H100 throughput is neither an Apple-Silicon result nor an implementation specification. |

[DiffuLLaMA](https://github.com/HKUNLP/DiffuLLaMA/tree/c17e897f6476c174b4623da594e4c65554f1613d)
is useful conversion prior art, not the first serving target: the observed
source publishes adaptation/training/evaluation code for converting GPT-2 and
LLaMA models, describes a released GSM8K LoRA adapter, and specifies
PyTorch+CUDA plus optional FlashAttention.  Its source license and a generally
usable inference checkpoint were not verified here.  It therefore cannot be
treated as an implementation dependency.

### Reading coverage

The LLaDA abstract and official README/inference/FAQ passages were read; the
paper's full methods and appendices were not.  The Dream v1 abstract,
introduction, approach excerpts, and repository model card were read; this was
not a full-paper or custom-code audit.  Mercury's announcement and abstract
were read; its implementation is unavailable.  The DiffuLLaMA README and
environment/inference sections were read, but the OpenReview paper was not
available through the source renderer.  These limits deliberately bound the
claims above.

## Why a decoder-shaped interface is wrong

An AR decoder advances a prefix with `p(token | prefix)`, normally stops at an
EOS decision, and may append one new K/V position per forward.  A diffusion
text model instead maintains a *whole candidate sequence* plus its corruption
or timestep state.  A model pass can update many positions, re-mask positions,
or make an arbitrary infill decision.

| Concern | Full-sequence masked diffusion | Blockwise semi-autoregressive inference over a masked model |
| --- | --- | --- |
| State | Prompt/fixed positions, all mutable positions, mask/noise state, schedule position, RNG, and any model-private attention state. | A committed prefix plus one mutable block and the block's schedule/noise state. |
| Termination | Fixed schedule, no mutable positions, or an adapter-defined convergence condition; EOS is data to validate, not automatically the loop terminator. | Commit a completed block, then choose/denoise the next block; model-specific final-block/EOS rules remain necessary. |
| Cache reuse | Do not assume an AR K/V cache: a bidirectional pass can change every mutable position and invalidate it.  Immutable prompt encoding or explicitly supported fixed-token state may be reusable only after parity proof. | A completed block may be fixed for later rounds, but this does not make it a reusable causal K/V prefix.  Each denoising pass needs model-specific cache-validity proof. |
| Work shape | Repeated full-sequence or active-position forward passes; parallel positions exchange fewer serial token steps for more passes and potentially large activation traffic. | Repeated block passes; can trade block size against serial rounds, memory, and prefix reuse. |

This second column is already a LLaDA inference mode, not a hypothetical model
family.  At pin `9182493`, [`generate.py`](https://github.com/ML-GSAI/LLaDA/blob/9182493720ed723ef8031210d85959364e51cbe0/generate.py)
initializes the requested suffix as masks, then repeatedly selects tokens only
from the current block while later blocks remain masked; its `var_generate()`
variant grows one block at a time.  That is a semi-autoregressive *sampling
policy over a masked model*, not evidence that LLaDA was trained as a
block-causal model or that its Transformer has a reusable causal-prefix cache.
The project itself says its sampling does not yet leverage KV cache.

A genuinely trained block-causal diffusion architecture would be a separate,
unverified future case: it may have a committed causal prefix representation,
but must demonstrate the model mask and cache validity before it shares an AR
prefix optimization.  These distinctions are why a future service contract
should ask an adapter for capacity and progress, not prescribe
`decode_one_token`.

## Probability, constraints, and control are different contracts

A denoiser's score or logits at a position and noise level are not, by
themselves, an AR next-token probability conditional on a committed prefix.
Consequently:

- A displayed token logprob needs an adapter-defined meaning: e.g. a sampled
  reverse-transition probability or a conditional likelihood estimator.  It
  must not be presented as ordinary decoder `log p(token | prefix)` without a
  derivation and validation.
- LLGuidance-style legal-next-token masking is not directly reusable.  A
  diffusion adapter may freeze known bytes/tokens, restrict a transition, or
  use guidance at mutable positions, but it must prove tokenizer-boundary,
  schedule, and final-language validity.  Final validation remains required.
- The Feynman--Kac note's discrete FKC and continuous FKC rows are separate
  score/CTMC research paths.  SMC-SD remains specifically AR target/draft
  work: its incremental ratios and prefix-trie K/V sharing cannot be assumed
  for a denoiser.  A future diffusion-control experiment would need its own
  transition-density and particle-state ledger.

Pure categorical operations such as temperature transforms, a random draw, or
top-k selection could still be shared *inside* an adapter when their input is a
well-defined reverse transition and corresponding tests establish the intended
law.  What cannot be shared by default is the AR interpretation of that draw,
its prefix-derived logprob, or its cache lifetime.

This preserves the useful division in the existing control research: grammar
state is not model state, and control must not manufacture a false likelihood
contract.

## Smallest adapter boundary

Keep this private to a future model crate until at least two real diffusion
families need the same service behavior.  The conceptual loop is:

1. Validate model-specific prompt/template, requested output extent, schedule
   and any immutable/infill spans; create opaque refinement state.
2. Run one adapter-owned denoising transition over the active sequence or
   block; the adapter owns attention mask, packed representation, RNG use,
   candidate selection/remasking and physical cache allocation.
3. Emit only adapter-defined progress (round, mutable-token count, work and
   memory receipts).  Do not stream provisional bytes as final text unless the
   adapter can state its revision semantics.
4. Finish through the adapter's schedule/convergence rule, decode once, then
   apply independent requested-output validation before returning a completed
   response.

The prospective control plane can share cancellation, request bounds, model
lifecycle, final text/usage receipts, capability reporting, opaque adapter
capacity estimates, and proven pure categorical helpers.  It must not presume
Qwen's physical K/V pages, per-token final streaming, AR logprob meaning, AR
EOS handling, or SMC cache forking.  This matches the architecture document's
existing rule that an adapter owns its physical cache representation and reports
capabilities.

## Gates and recommendation

**Do not start runtime work before the V4.1 text execution gate.**  After that,
prefer Dream 7B for the first real candidate because its observed model-card
license is Apache-2.0 and it exposes Safetensors; its `trust_remote_code`
requirement is a mandatory source/config audit, not a reason to execute code.
LLaDA is an equally valuable reference and an especially relevant future MoE
stress test, but licensing and exact artifact metadata must be recorded first.

1. **Offline gate — no checkpoint.** Implement no common trait.  In a tiny
   categorical reference, define a masking schedule and an explicit reverse
   transition; assert fixed-token preservation, schedule termination,
   reproducibility, decoded-text validity, and the stated transition
   likelihood.  Record active positions, rounds, allocated bytes and elapsed
   time.  This proves the accounting vocabulary, not model quality or Metal.
2. **Real-model gate — one bounded request.** After reviewing the pinned Dream
   custom code/config and tensor map, run one short fixed-prompt reference
   generation on the upstream implementation and capture only permitted
   output/config/round receipts.  Reproduce the same seed, schedule, token IDs
   and final IDs with an adapter-local CPU reference before considering Metal.
   The acceptance comparison is final IDs plus round-by-round state where the
   reference exposes it; prose similarity is insufficient.
3. **Apple gate — qualification, not assumption.** Port the smallest correct
   bidirectional/full-or-active-position forward path, then compare CPU
   reference parity, peak unified-memory allocation, rounds, time per round,
   total latency, and final-ID match.  Only then test immutable-prompt or
   block-prefix reuse.  CUDA/FlashAttention examples and H100 results do not
   substitute for this measurement.

Defer dynamic constraints, speculative/draft methods, continuous batching,
provisional streaming, generic denoiser traits, model paging, and any claim of
vLLM/SGLang parity.  Those are structural forks that need a second qualified
diffusion adapter or a demonstrated shared service need.
