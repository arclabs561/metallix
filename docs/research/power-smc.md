# Power-SMC: a bounded serving research memo

## Decision

Power-SMC is a plausible *optional, adapter-local* inference-time sampler once
cached decoding and cache reindexing are independently qualified.  It is not a
replacement for ordinary sampling, a generic cache abstraction, or evidence of
better latency on an Apple Silicon Mac.  The next implementation decision is
gated on a tiny exact oracle and a cache-ancestry differential test; neither
the paper nor its CUDA reference implementation closes those gates for
Metallix.

The primary source is [Power-SMC: Low-Latency Sequence-Level Power Sampling
for Training-Free LLM Reasoning](https://arxiv.org/abs/2602.10273v2),
arXiv:2602.10273v2, 23 March 2026.  This memo covers the complete v2 HTML
article: sections 1--8 and appendices A--E, including equations, proof,
algorithm, cost model, experimental setup, and cache/EOS appendices.  The
rendered HTML was used because it exposes the same versioned mathematical
content; no claim here depends on uninspected PDF-only material.

## What distribution it targets

For an EOS-terminated completion $y$ and base autoregressive model
$p_{\theta}$, the target is the *sequence-level* power distribution

$$
\pi_{\alpha}(y\mid x) =
  \frac{p_{\theta}(y\mid x)^\alpha}{Z_{\alpha}(x)},\qquad
p_{\theta}(y\mid x)=\prod_{t=1}^{T(y)}p_{\theta}(y_{t}\mid x,y_{<t}),\quad
\alpha>1.
$$

This is not the distribution produced by merely lowering a per-token
temperature to $\tau=1/\alpha$.  Locally normalized temperature sampling
adds a different normalization for every prefix, so its sequence probability
is generally not proportional to $p_{\theta}(y\mid x)^\alpha$.  That
distinction is the reason particle weights exist at all.

At prefix length $t$, the paper's Feynman--Kac unnormalized target is
$\gamma_{t}(y_{1:t}\mid x)=p_{\theta}(y_{1:t}\mid x)^\alpha$.  For any
prefix-only proposal $q_{t}$ whose support contains the target's support,
the incremental importance factor is

$$
\omega_{t} =
\frac{p_{\theta}(y_{t}\mid x,y_{<t})^\alpha}
     {q_{t}(y_{t}\mid x,y_{<t})}.
$$

The conditional-second-moment-optimal proposal is
$q_{t}^*(v\mid h)\propto p_{\theta}(v\mid h)^\alpha$, equivalently
`softmax(alpha * logits)` / temperature $1/\alpha$.  Under that proposal,
the token-choice part of the incremental weight is

$$
\log \omega_{t} =
\log\sum_{v} \exp\bigl(\alpha\log p_{\theta}(v\mid h)\bigr).
$$

It is deterministic conditional on a prefix, but differs between prefixes.
Thus the optimal local proposal removes token-choice variance, not all
particle-weight variance.

## Correctness boundary

Use log weights, normalize with `logsumexp`, and compute normalized
$w_{i}$ before $\mathrm{ESS}=1/\sum_{i} w_{i}^2$.  Resampling is triggered
only at the chosen ESS rule (the paper uses a threshold $\kappa N$) and
uses an explicit ancestor vector whose entries may repeat.  Systematic
resampling is an unbiased resampling scheme and usually lower variance than
multinomial resampling; it does not make a finite particle population an exact
draw from $\pi_{\alpha}$.

The paper's alpha bridge uses a schedule
$1=\alpha^{(0)}<\dots<\alpha^{(L)}=\alpha$.  At a stage change, its
correction is

$$
\log W \mathrel{+}= (\alpha^{(\ell)}-\alpha^{(\ell-1)})
                  \log p_{\theta}(y_{1:t}\mid x).
$$

Together with the stage proposal factors, this is an algebraically exact
change of the unnormalized path target in exact arithmetic.  It is separate
from the finite-$N$ quality of the self-normalized/resampled particle
approximation.  A serving implementation must retain the per-particle
cumulative base log probability across bridge stages and must not silently
replace the final weighted categorical selection with MAP selection.

EOS is an ordinary sampled token.  Once a particle emits it, the paper treats
the state as absorbing: force EOS/no-op thereafter, use incremental weight
one, and do not advance that particle's model cache.  This matters for both
the mathematical target and cache lifecycle.

Proposal changes require an explicit semantic choice.  Hard top-k/top-p,
grammar masks, and an EOS ban can assign zero proposal probability where the
original power target is positive; that violates the importance-support
condition.  A finite repetition penalty normally preserves support, but its
altered proposal probability still has to be used in the incremental weight.
For hard constraints, valid targets include a globally restricted
$p_{\theta}(y\mid x)^\alpha$ target or a power of a locally constrained base
distribution.  Those are different distributions, so the chosen definition
must determine the normalizers and weights.  Applying a constraint after
computing an unconstrained target is not an exact Power-SMC variant.

| Statement | Status |
| --- | --- |
| Prefix-target, incremental factor, and bridge identity | Exact algebraic specification |
| Locally optimal temperature proposal | Exact conditional variance result in the paper |
| A finite particle population plus selected final particle | Approximation to the target, with Monte Carlo/resampling variance |
| Paper's reported speedups | H100/Hugging Face experiment result, not Mac evidence |
| CUDA cache optimizations in the authors' repository | Candidate implementation ideas, not an adapter-independent proof |

## Cache and particle cost on one Mac

With cached decoding the paper counts $N$ particle forward evaluations per
generated token: $C_{\mathrm{SMC}}=N T$.  Its wall-time discussion assumes
batch throughput $s(N)$, yielding approximately $TN/s(N)$, plus linear
weight/resampling work.  That model does not establish that a unified-memory
Mac has useful $s(N)$, enough memory, or cheap cache gathers.  Measure all
three rather than extrapolating GPU results.

Naively, active decode state scales as $O(N\,\text{KV bytes})$, while a
resample may gather/copy it and duplicate ancestors.  The paper's companion
implementation proposes two useful but conditional ideas:

1. Process an immutable prompt once, then share/replicate its cache for the
   initially identical particles.
2. After resampling, retain only the $U$ unique ancestor caches, map the
   $N$ logical particles to those physical caches, and fork/expand only
   before divergent next-token forwards.

The latter can lower *post-resample* residency but cannot remove the normal
peak of divergent particles.  It also requires genuine copy-on-write or
otherwise immutable shared pages; aliasing a mutable Metal buffer is a
correctness bug, not an optimization.  On a single Mac, prompt length,
generated length, particle count, cache representation, resample frequency,
and memory pressure all have to be observed together.

Do not introduce a universal cache trait for this experiment.  Each model
adapter should expose a narrow, fallible particle operation with a typed
ancestor map (length $N$, values in `0..N`, duplicates permitted) and must
reindex **all** next-step state atomically: token histories/positions, EOS
flags, cumulative log probabilities, cache tensors and any architecture
specific recurrent/compressed state, plus the particle RNG/ancestry accounting
needed for reproducibility.  DeepSeek-family cache variants make a generic
recursive "first batch dimension" traversal particularly unsafe.

## Bounded gates before an implementation choice

1. **Math oracle.**  Enumerate a tiny EOS-terminated vocabulary/tree.  Verify
   the incremental log weight and every bridge transition against direct
   $p(y)^\alpha$ enumeration.  Test log-domain normalization and ESS with
   fixed values.  This proves identities, not a false claim that finite $N$
   samples are exact.
2. **Cache ancestry differential.**  Use an ancestor vector with both repeated
   and discarded parents.  For every resulting particle, compare the next
   logits/state after reindexing against independently replaying that
   particle's prefix; include one absorbed-EOS particle.  An adapter failing
   this does not opt into Power-SMC.
3. **Constrained-target contract.**  Before structured output support, define
   whether a mask is part of the base target and test nonzero proposal support
   for every permitted target token.  Refuse unsupported hard filters instead
   of silently changing the distribution.
4. **Mac measurement, not a throughput claim.**  For a qualified small model,
   record ordinary cached decode and $N=1,2,4,8$: TTFT, inter-token latency,
   peak/resident memory, active cache bytes, bytes/time spent reindexing,
   ESS/resample count, unique-ancestor count, and output-length distribution.
   Stop increasing $N$ when memory pressure or batch scaling erases the
   intended benefit.

Only after these gates should copy-on-write cache compaction or multi-round
execution be considered.  Independent rounds bound peak memory but remove
cross-round resampling, so they are a quality/latency tradeoff rather than a
transparent way to claim one large SMC population.

## Related sources and provenance

The paper links its [official reference implementation](https://github.com/ArminAzizi98/Power-SMC), inspected at commit
[`0f3b20f`](https://github.com/ArminAzizi98/Power-SMC/tree/0f3b20f88e0f3add4712f4106d0a9abed44fc50e)
(MIT).  Its `smc_samp_utils.py` is useful evidence for shared-prompt and
unique-ancestor cache layouts.  It is not the correctness authority: its
optional top-k/top-p/repetition transforms must be reconciled with the target
support rule above before reuse, and its CUDA/Torch cache hooks do not define
a Metal adapter contract.

The supplied [Nous SMC server repository](https://github.com/NousResearch/smc-inference-server)
currently resolves and was inspected at main commit
[`4c5fade`](https://github.com/NousResearch/smc-inference-server/tree/4c5fade5ef89965079c5866ff302893227491e3d).
It identifies itself as a wrapper for *Sequential Monte Carlo Steering*
(arXiv:2306.03081), `llamppl`, vLLM, and CUDA Docker workers; it is not an
alias for Power-SMC and selects its maximum-weight particle rather than this
paper's final weighted categorical sample.  The pinned tree and GitHub
metadata contain no license declaration, so it is a behavioral contrast only,
not code to copy.

[Awesome Inference-Time Scaling](https://github.com/ThreeSR/Awesome-Inference-Time-Scaling)
was boundedly inspected at commit
[`2d03846`](https://github.com/ThreeSR/Awesome-Inference-Time-Scaling/tree/2d03846447cbec8ecfa24e8c3c8564f4082c602e)
as a discovery catalog, not as evidence.  Its relevant SMC entry is *On the
Power of (Approximate) Reward Models for Inference-Time Scaling*
(arXiv:2602.01381), which concerns reward-guided SMC and Bellman error, not
sequence-level power sampling.  No verified primary direct post-v2 Power-SMC
follow-up was located in this bounded pass; the paper itself cites earlier
related power-sampling work, which is intentionally outside this memo's
separate-paper reading scope.
