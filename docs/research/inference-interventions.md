# Inference interventions

These methods change different parts of generation. They are experiments in
behavior and diversity, not demonstrated serving-speed optimizations. Keep the
baseline model path unchanged and retain task-quality, cache and numerical
controls before introducing an intervention.

## The three linked projects

| Project and inspected revision | Mechanism | Metallix implication |
|---|---|---|
| [llms-on-drugs](https://github.com/lexdoudkin/llms-on-drugs/tree/a2ad9898c32ee2a1df7b03b41ea48ca1f5ead9df), `a2ad989` | Prepends persona/context strings to benchmark prompts sent to a hosted model. | A paired prompt experiment, not a tensor hook or sampler. |
| [WEISS](https://github.com/r1cc4r2o/weiss/tree/97b797797a19b6b224ee56a7774895314d74da4f), `97b7977` | Samples a continuous latent and applies a learned projection/inverse-projection difference to encoder states before decoder generation. | A trained encoder–decoder/checkpoint change, not a drop-in V4.1 decoding flag. |
| [DRµGS](https://github.com/EGjoni/DRUGS/tree/3c053ead813b2d903df23a33f9b25b6ba3c25c1e), `3c053ea` | Perturbs hidden, query, key, value or attention-output tensors inside patched Llama/Mistral attention. | An adapter-specific intervention with cache and kernel consequences, not ordinary temperature sampling. |

Source coverage: all three READMEs and selected concrete implementation hooks,
not exhaustive repository audits or reproduced experiments:

- [Prompt benchmark](https://github.com/lexdoudkin/llms-on-drugs/blob/a2ad9898c32ee2a1df7b03b41ea48ca1f5ead9df/src/run_benchmark.py).
  The code's default temperature is 0.2 while its accompanying TeX describes
  temperature zero. Do not treat those reports as a reproducible local baseline.
  GitHub metadata did not identify a license; no code is copied here.
- [WEISS model](https://github.com/r1cc4r2o/weiss/blob/97b797797a19b6b224ee56a7774895314d74da4f/src/model/weiss.py)
  and [beam-search module](https://github.com/r1cc4r2o/weiss/blob/97b797797a19b6b224ee56a7774895314d74da4f/src/module/sampling.py).
  Its research target is molecular/sequence diversity; the repository has an
  MIT license. A new trained latent mechanism would need its own adapter and
  checkpoint qualification.
- [DRµGS attention](https://github.com/EGjoni/DRUGS/blob/3c053ead813b2d903df23a33f9b25b6ba3c25c1e/drugs/models/llama/drugged.py)
  and [injector](https://github.com/EGjoni/DRUGS/blob/3c053ead813b2d903df23a33f9b25b6ba3c25c1e/drugs/inject_mixin.py).
  H is perturbed before Q/K/V projections, Q before positional rotation,
  cloned K/V after cache update, and attention output afterward. The hook
  preserves initial attention-sink positions. Its README discusses persistent
  noise-conditioned cache state and optional cache regeneration. The repository
  is MIT-licensed; no implementation is imported here. Informal output samples
  do not establish improved task quality.

## Related foundations

- [Contrastive Decoding](https://arxiv.org/abs/2210.15097): contrasts expert and
  amateur log probabilities under a plausibility restriction. Closest to a
  logits-level intervention, but requires compatible token identities and
  additional model/cache execution. No free memory or latency benefit follows.
- [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681): derives
  residual-stream directions from paired positive/negative examples and adds
  them during inference. Vectors depend on model, layer and capture protocol.
- [Instruction-following activation steering](https://arxiv.org/abs/2410.12877):
  derives intervention vectors from instructed/uninstructed activations.
  Format-following behavior remains a quality hypothesis, unlike a parser's
  explicit token-validity guarantee.

These are historical anchors, not a claim to cover the current frontier.
Coverage in this pass is abstracts and method descriptions, not full papers.

## Newer research directions

- [SWAI, 2601.10960v2](https://arxiv.org/html/2601.10960v2), revised May 2026:
  builds tokenizer-specific normalized one-vs-rest log-odds scores from labeled
  corpora, then biases a target-scored subset of the model's top-K candidates.
  This touches logits rather than attention or cache tensors. Its Llama
  experiments include additional decoding controls and model judging; stronger
  bias can degrade fluency/coherence. A local experiment would need pinned
  corpus/tokenizer/score-table identity, exact top-K/bias ordering, fixed-logit
  tests and paired task evaluation. It is not a demonstrated speed technique.
- [Conditional activation steering, 2505.12189v3](https://arxiv.org/html/2505.12189v3),
  revised April 2026: CAST/K-CAST condition the steering sign/scale on internal
  activation geometry; K-CAST uses nearest labeled condition vectors. It studies
  controlled reasoning, including Qwen 2.5, rather than general serving
  reliability. This needs model-specific capture sites, per-input condition
  state and evaluation, not one universal perturbation-strength flag.

Coverage: selected methodology, evaluation and limitation sections, not full
papers or local reproductions. Full-paper and implementation review remain
gates before adopting either method. The nearer experiment is a reversible
token-score bias on the existing logit surface; conditional tensor interventions
remain deferred. If combined with grammar constraints, define the ordering and
prove that biasing cannot re-enable a forbidden token. Report baseline,
intervened and grammar-conditioned probabilities separately.

## An experiment that fits the current implementation

First compare paired prompt variants on the Qwen control, with immutable token
inputs and a task oracle. This needs an evaluation harness, not runtime hooks.
Record output validity, task score, generated IDs, reference-model log
probability, latency and resource use. More diverse text is not necessarily
better text. Preserve a zero-intervention baseline and negative results.

The new `mx gen --logprobs` instrumentation is useful but not sufficient for
all these comparisons. It reports selected-token probabilities under the
current forward pass; it does not automatically rescore a perturbed sequence
under an unmodified reference model. Under grammar constraints, raw model
probability and the renormalized allowed-token probability are distinct.
Neither is the probability of the entire sequence conditioned on eventual
grammar acceptance. KL/JS comparisons need the full distributions on identical
prefixes, not just one selected token's log probability.

For a later activation experiment, specify the exact layer/tensor, perturbation
rule, magnitude, seed and token-position policy. Any cache identity must include
the intervention and its stochastic history. Cached and full-prefix executions
need the same perturbations for a meaningful parity oracle; a shared seed alone
does not establish that when their execution schedules differ. Do not reuse
baseline prefix state across intervention settings or quietly invalidate fused
kernel assumptions. Zero perturbation must recover baseline behavior.

No intervention is implemented by this note. DeepSeek-V4.1 forward/cache
qualification and production serving of constrained generation remain ahead of runtime tensor
perturbations; there is no universal hook/plugin interface to commit to yet.
