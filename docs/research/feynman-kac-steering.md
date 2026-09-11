# Feynman–Kac steering and particle speculation

**Decision:** keep these as research inputs, not a Metallix runtime feature. The
near-term priority remains correct DeepSeek-V4 generation, constrained decoding,
and measured single-Mac serving. All three methods make a finite-population
approximation or target a model class we do not run.

## What the methods actually target

| Method | Target and proposal/state evolution | Incremental weight and resampling | Guarantee / fit |
| --- | --- | --- | --- |
| Discrete FKC | A *masked discrete diffusion* marginal transformed into an annealed `p_t^beta`, product/geometric average, or reward tilt `p_t exp(beta_t r)`. Its proposal is a newly derived reverse CTMC rate, not an autoregressive draft model. | The derived Feynman–Kac potential `g_t(x)` accumulates in log weights; normalized particles are resampled (the reference implementation supports systematic, per-step or ESS-triggered resampling). | The paper derives the CTMC/Feynman–Kac construction and time-discretization limit, but explicitly leaves produced-sample convergence bounds open. Finite particles and finite steps are approximate. It is not a decoder-only/KV-cache algorithm. |
| Diffusion FKC | A continuous score-diffusion path targeting annealed, geometric-average, product, classifier-guided, or reward-tilted densities. A weighted SDE supplies the proposal path and potential. | Weight increments correct the PDE-derived path; systematic resampling is used over an active time interval in the reported implementation. | Self-normalized importance estimates become exact only as particle count tends to infinity; finite particles are approximate. This requires score/SDE and often reward-gradient contracts, not token logits. |
| SMC-SD | The target is an autoregressive target model `p`; each particle extends a different prefix under cheap autoregressive draft `q`. | For a draft block, multiply `p(d_j | prefix,d_<j) / q(d_j | prefix,d_<j)`; resample particle ancestors when ESS crosses a threshold. The target also supplies a bonus token. | Unlike ordinary speculative decoding, it deliberately trades exactness for a tunable finite-`N` approximation. Its single-round bounds depend on target/draft chi-squared divergence; the authors leave multi-round resampling/path-degeneracy error bounds for future work. This is the only one structurally adjacent to Metallix. |

ESS, conventionally `1 / sum(normalized_weight^2)`, measures proposal/path
degeneracy. It is neither token entropy nor probability that an answer is
correct.

## Serving implications

SMC-SD’s appealing systems statement is conditional: when `B * N * (K + 1)`
is below an accelerator-specific roofline ridge, extra particles can consume
otherwise idle compute. Beyond it, its model predicts speed falls with effective
batch size. Its reported headline results are multi-GPU; they do not establish a
single unified-memory Mac result.

The SMC-SD reference engine is a patched SGLang implementation, and documents the real
state cost: `N` request histories, per-particle target/draft work, and
ref-counted KV pages. Prefix pages are shared at fan-out; resampling releases a
destination tail and aliases/increments the selected ancestor tail. That avoids
blind full-cache copying but does not make divergent tails free. Its current
request validator rejects grammars, returned logprobs, and several other
features, so it is evidence of non-composability rather than a drop-in design.

For a Mac, particle count competes directly with KV capacity and Metal batch
efficiency. Measure target and draft forward time, allocated/shared KV bytes,
resample bytes and time, ESS, and output-distribution/quality drift. Do not
infer a speedup from CUDA roofline language.

## Constraints, logprobs, and uncertainty

Raw token probabilities belong to the unmodified model policy. For a grammar
allowed set `A`, feasible mass is `alpha = sum_{v in A} p(v)` and constrained
probability is `p(v) / alpha` for allowed `v`. An importance ratio is valid only
when target and proposal name the same policy: masks, temperature, top-p, and
other interventions must be reflected in both sides. A hard terminal grammar
indicator can be a steering potential, but late discovery produces zero-weight
particles and severe ESS collapse; it is not a substitute for incremental
grammar masking. Particle weights/ESS quantify mismatch, not semantic
uncertainty or calibrated correctness.

## Bounded next gate

After baseline autoregressive generation, raw/constrained logprob provenance,
and KV allocation metrics are stable, build an offline tiny-categorical reference
test only: enumerate `p`, draft from `q`, verify log-ratio normalization,
ancestry, ESS, and convergence as `N` grows. Then compare `N = 1, 2, 4` on one
Mac with a paired draft model and fixed prompt/seed suite. Advance only if it
reports memory, latency, ESS, and distribution/quality error against exact
speculative decoding or target sampling. Diffusion FKC and discrete FKC require
separate model-family adapters and should not shape the core decoder API.

## Primary sources and reading coverage

- Hasan et al., [*Discrete Feynman–Kac Correctors*, arXiv:2601.10403v1](https://arxiv.org/html/2601.10403): read abstract, Sections 2–5, Algorithm 1, and cited implementation’s steering/resampling core; not the entire appendices. The paper’s Sections 3 and 5 support its derived CTMC targets and stated convergence gap.
- Skreta et al., [*Feynman–Kac Correctors in Diffusion*, arXiv:2503.02819v2](https://arxiv.org/html/2503.02819): read abstract, method/resampling passages, and selected propositions; not the full paper or appendices. Its Section 4 describes active-interval systematic resampling and its background labels finite particle samples approximate.
- Emara et al., [*Faster LLM Inference via Sequential Monte Carlo*, arXiv:2604.15672](https://arxiv.org/html/2604.15672): read Sections 2–3, engine design, approximation analysis, conclusion, and linked engine design docs; not a full-paper read. Sections 3.1–3.2 establish the ratio, roofline assumptions, finite-`N` status, and stated multi-round gap.
- Code pins checked 2026-09-11: [`discrete_fkc`](https://github.com/hasanmohsin/discrete_fkc/tree/f42c3c9b0823614de468f9bcf4a37dc07c2cab47) `f42c3c9` (GitHub license metadata: none); [`fkc-diffusion`](https://github.com/martaskrt/fkc-diffusion/tree/aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6) `aa6f5ed` (metadata: none, README states MIT); [`smcsd`](https://github.com/abdelfattah-lab/smcsd/tree/6096b8980495bb1b4e5ced58aab604e87f86d8ef) `6096b89` (metadata: none). Treat unresolved licensing and the README/metadata discrepancy as due-diligence items, not permission to copy code.
