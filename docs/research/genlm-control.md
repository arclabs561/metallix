# GenLM: weighted control is not a grammar backend

## Scope and provenance

This is a bounded primary-source review for Metallix, checked on 2026-09-11.
It covers the linked GenLM organisation and three repositories, not the whole
ecosystem or a performance comparison.  The repositories were read at these
main-branch pins, each reporting Apache-2.0:

| Source | Pin | Read coverage |
|---|---|---|
| [genlm-grammar](https://github.com/genlm/genlm-grammar/tree/455c3a494f58122aaf3ce5af184dfbe6248cd7e8) | `455c3a4` | README plus language-model and CFG adapters |
| [genlm-control](https://github.com/genlm/genlm-control/tree/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d) | `533a36e` | README, potential and sampler docs, token sampler API |
| [llamppl](https://github.com/genlm/llamppl/tree/7ad531a506ccac939651553fa595bb036db18f80) | `7ad531a` | README, SMC implementation and cache documentation |

The original LLaMPPL paper, *Sequential Monte Carlo Steering of Large Language
Models using Probabilistic Programs*, was read in full through the Hugging Face
Markdown rendering and cross-checked against all 12 pages of arXiv
`2306.03081v2`: abstract, sections 1--4, algorithm, figures, footnotes and
references.  This version was submitted on 2023-11-26 and contains no
appendix.  Full reading establishes what the paper claims, not correctness,
current implementation parity, or Apple-Silicon performance.  The newer AWRS
paper was read only through its abstract and selected sections 1--3; its claims
below are correspondingly narrower.

## Separate three probability contracts

**Hard grammar masking.** A Boolean `BoolCFGLM` gives all legal next tokens the
same grammar weight; used against an LLM it is the familiar local legal-token
mask ([adapter](https://github.com/genlm/genlm-grammar/blob/455c3a494f58122aaf3ce5af184dfbe6248cd7e8/genlm/grammar/cfglm.py)).  That is the same probability level as
LLGuidance: renormalize the model over tokens legal *now*.  Its Earley/CKY
chart-prefix cache is parser state, not transformer KV state.  A weighted
grammar instead supplies a grammar language-model score; it does **not** by
itself mean the base LLM conditional on the grammar language
([LM factorization](https://github.com/genlm/genlm-grammar/blob/455c3a494f58122aaf3ce5af184dfbe6248cd7e8/genlm/grammar/lm.py)).

**Weighted target and proposal.** `genlm-control` makes a target explicit as
the LLM multiplied by potentials.  Potential products add log-weights;
`complete` can be a terminal reward and `prefix` can supply earlier information
or support pruning ([potentials](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/docs/potentials.md)).  A proposal is a computational choice, not a target change: its token sampler applies
the supported `target/proposal` importance ratio and rejects a cross-tokenizer
proposal ([API](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/genlm/control/sampler/token.py)).

This distinction resolves a common error.  A masked proposal with unit weights
only samples the local conditional. To target the global conditional
over finished strings, retain the omitted legal mass as the incremental
potential.  The LLaMPPL paper derives precisely that correction for a hard
prefix constraint; it also shows an LLM product-of-experts example.  Its
program semantics are concrete: `sample(dist, proposal)` contributes
`dist/proposal`, `observe(dist, value)` contributes `dist(value)`, and
`condition` contributes zero or one ([paper v2](https://arxiv.org/html/2306.03081v2)).

**Particle approximation.** SMC expands proposals, multiplies incremental
weights, then resamples.  The LLaMPPL algorithm returns a weighted particle
approximation and an unbiased estimate of the normalizer; it states posterior
consistency as particle count grows.  Therefore a finite particle completion is
not an exact global-posterior sample.  AWRS is more limited and attractive for
Boolean local constraints: uncapped adaptive weighted rejection samples the
local constrained token law exactly and supplies normalizer estimates useful
to SMC.  Its runtime is stochastic, so a cap, fallback or proposal truncation
must be labelled approximate unless a valid correction remains
([AWRS, selected §§1--3](https://arxiv.org/html/2504.05410)).

## Cache and Mac implications

The paper's useful systems idea is a shared token trie: cache next-token logits
and layer K/V at common prefixes, then compute only divergent suffixes.  It
also expands active particles by factor `K` and uses without-replacement
downsampling to preserve diversity.  This does **not** establish a low-cost
Metal implementation.  The present LLaMPPL code deep-copies particles before
beam expansion ([implementation](https://github.com/genlm/llamppl/blob/7ad531a506ccac939651553fa595bb036db18f80/llamppl/inference/smc_steer.py)); its
own cache notes say fixed shared-prompt KV is supported while divergent
per-particle KV is poorly supported ([cache limits](https://github.com/genlm/llamppl/blob/7ad531a506ccac939651553fa595bb036db18f80/docs/caching.md)).

For Metallix, there is no reason to add a PPL or universal control trait.  A
future offline experiment needs only a sequence-fork facility whose operations
are snapshot, append, release and measured copy-on-write allocation.  It must
not duplicate full Metal tensors per particle.

## Bounded test gate after the generation baseline

1. Use a tiny enumerable vocabulary/predicate offline.  Compare ordinary
   local mask and rejection/AWRS-like draws; assert legality and record legal
   mass, attempts, latency and exact distribution error.
2. If decoded KV can fork cheaply, test two then four offline particles on the
   same toy target.  Record `log p_target`, `log q`, incremental log-weight,
   normalizer estimate, ESS, resampling ancestry and resident/allocated KV
   bytes.  Enumerate the target independently.
3. Keep the current generation diagnostic single-sequence and hard-mask-only until those results
   demonstrate a useful accuracy/cost trade.  Terminal critics, learned
   lookahead and a user-visible PPL are deliberately out of scope.

The reusable requirement is a per-experiment likelihood ledger and cache-fork
telemetry, not a new abstraction family. DeepSeek-V4.1-Flash forward/cache
qualification remains the primary implementation target. The existing
log-probability and constrained-generation baseline runs Qwen3, not V4.1.
