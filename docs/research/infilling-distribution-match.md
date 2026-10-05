# Infilling and distribution matching: a staged plan

## Decision

For a template such as `A [s1] B [s2] C`, the target is the model's own
conditional: `pi(s) ∝ p(A s1 B s2 C)`, restricted to strings the template
admits. Left-to-right masked sampling does not draw from `pi`; each slot
ignores the fixed text after it. Every sampler below is judged by how close it
gets to `pi` per forward token, measured against an exact enumerated target
on small templates. No particle-serving API, controller runtime, or diffusion
adapter ships before its gate passes.

The first build is an offline benchmark plus a teacher-forced scoring API, not
SMC. Resampling with KV forks is built only if it beats reweighted best-of-N
at equal forward tokens on a template where it should (see the stop rule
below).

## The target and its weight

Infilling is the grammar-conditioned target with a template grammar. Fixed
text after a slot is an observation: its token probabilities multiply the
weight, and its proposal probability is 1. LLaMPPL defines this posterior
and weight over token sequences, with each fixed fragment tokenized once
([2306.03081v2](https://arxiv.org/abs/2306.03081) §2.2 p.4, read in full: the
potential is the forced fragment's token likelihood times `2^{|h|+1}`, the
inverse of the geometric slot-length proposal). That is the
fixed-tokenization target, not the byte-string target below. Twisted SMC's
infilling experiment is the one-slot, fixed-length special case: 15 generated
tokens with the next 10 tokens of a base-model sample as the observed suffix,
whose likelihood is the terminal potential
([2404.17546](https://arxiv.org/abs/2404.17546) §7.2.3 p.10, App. G.4;
Table 5 p.15 names this target).

Metallix already computes a weight of this kind, with one qualification.
`allowed_log_mass` in [`constraint.rs`](../../crates/engine/src/constraint.rs)
is the model mass of every token the grammar allows next, and fixed JSON keys
are decoded one token at a time. The constraint uses llguidance's approximate
(non-canonical) tokenizer environment, so a fixed key admits every
tokenization of its bytes: on the Qwen3 vocabulary, after `{"` the mask allows
six tokens (`d`, `de`, `des`, `desc`, `descr`, `description`). The sum of
`allowed_log_mass` over a run is therefore the importance log weight toward
the global conditional over byte strings, marginalizing tokenizations, at
temperature 1. It equals the weight for canonical token sequences only with a
canonical tokenizer environment, where those masks become single tokens.
Each sampling mode should declare which of the two targets it uses.
At other temperatures or with truncation, add `model_logprob -
sampling_logprob` per step, as [Gate 1](sampling-next-gates.md) records.

Two consequences:

- Forced-token fast-forward must not give forced tokens unit weight, or the
  target silently becomes the left-to-right one. A fast-forward path needs
  per-position log probabilities of the forced span; `extend_last_logits`
  returns only the final position's logits today.
- Masked decoding is exact only when the product of allowed masses is constant
  over the target's support. That depends on where the model puts mass, not
  on the grammar: a grammar where every prefix has the same number of
  completions still biases a real model.

Slot termination is part of the proposal. A model never emits "end of slot",
so the rule that ends a slot (a grammar delimiter or a proposed length) must
appear in `q`. Scoring fixed text under one tokenization gives a lower bound
on its byte-level probability
([2412.03719](https://arxiv.org/abs/2412.03719) eq. 10, p.4); the exact value
sums over a covering of token strings, which beam pruning approximates at
`K >= 8` for the models they tested (§3.2, §4). Use grammar-forced spans or
declare an aligned boundary.

## Approximation ladder

`N` is the particle or sample count. Costs are accounting identities, not
measurements.

| Rung | Method | Extra cost | Target |
|---|---|---|---|
| 0 | Left-to-right masked, fixed text appended, weight ignored | none | Locally normalized; each slot ignores later text |
| 1 | Best-of-N reweighted by the forced-text weight, weighted categorical pick | `N` times rung 0 | Consistent as `N` grows |
| 1b | Independence Metropolis-Hastings over the same `N` draws: start at draw 1, move to draw `j` with probability `min(1, w_j / w_current)`, return the final state | selection rule only, `O(N)` scalar work | At least as close to `pi` as rung 1 in every f-divergence for every `N`, if `w` is exactly proportional to `pi/q`, `q` and `pi` have the same finite support (lengths capped, `w > 0` on every draw), and draw 1 initializes the chain ([2610.03480](https://arxiv.org/abs/2610.03480) Thm 1 and §3 p.4; the proof does not use autoregressive structure, p.5); strict for `N >= 2` when `q != pi` in TV, KL (both directions) and chi-square |
| 2 | Importance-sampling ensemble | as rung 1 | Weighted set plus a normalizer estimate |
| 3 | SMC, resampling at fragment boundaries | KV forks, `N`-way residency | Helps only with two or more slots; equals rung 1 with one |
| 4 | Lookahead: score the next fixed text on a discarded fork | extra prefill per check | Any positive twist stays consistent |
| 5 | MCMC over slots (resample one slot, accept on the rest's likelihood) | slot regeneration plus rescoring per move | The only rung where later slots revise earlier ones |
| 6 | Exact rejection | about `1/p(fixed text)` attempts | Exact; practical only for short, predictable spans |

Ordering from the literature, not from local measurement: better proposals
and lookahead beat more particles; past 5 to 10 particles, returns diminished
in three of Loula et al.'s four domains but not in goal inference (0.419 at
10, 0.611 at 50); resampling raised accuracy in three of four domains, not
text-to-SQL ([Loula et al. 2504.13139](https://arxiv.org/abs/2504.13139)
§3.2, Table 6 p.21, [AWRS 2504.05410](https://arxiv.org/abs/2504.05410)
Table 1). Importance sampling needs roughly `exp(KL(pi || q))` samples,
within a factor `exp(O(s))` where `s` is the spread of the log weight under
`pi` ([1511.01437](https://arxiv.org/abs/1511.01437) Thm 1.1-1.2, pp. 3-4),
which the enumerated benchmark predicts before anything runs.

The 1b guarantee is lost under top-k or top-p, a sampling temperature the
weight does not record, a slot-ending rule that gives zero probability to some
target length, uncapped lengths, a first state that is not a fresh draw (for
example greedy), a pool size chosen after looking at the draws
([2610.03480](https://arxiv.org/abs/2610.03480) App. C.2 p.41, Lemma 5 remark
p.22), or weights computed on a different numerical
path from the sampler. It covers one returned sample per pool; several outputs
need independent pools.

MCMC notes. Gonzalez et al. truncate at a sampled position, regenerate the
suffix with grammar-constrained decoding, and accept with
`min(1, P(w') q(w | w') / (P(w) q(w' | w)))`, where the grammar normalizer
cancels. The paper does not write `q` out; its released code sums over every
truncation point up to the longest common prefix,
`q(w'|w) = sum_{i <= lcp} p_pos^w(i) P_GCD(w'_{>i} | w'_{<=i})`, which is
what the convergence proof needs (App. E, p.19). Their stated decoding used
top-p 0.9 (App. B), which gives the proposal less than full support and is
outside the theorem. On one representative SyGuS benchmark their KL falls
over 1 to 10 steps, but it is the GAD proxy computed over observed samples
([2506.05754](https://arxiv.org/abs/2506.05754) §3, §4.1,
Llama-3.1-8B-Instruct only). Their convergence theorem (App. E, Thm 3) needs
truncation at position 0 to have positive probability.
[Large Language Gibbs](https://arxiv.org/abs/2606.19264) regenerates one
variable with the others serialized into the prompt in random order. Its
stationary law is defined only implicitly, as a compromise among the model's
conditionals; it equals the random-order autoregressive law only under an
order-invariance assumption that the authors say fails in general (§3, Prop.
3.2). Its experiments sweep variables in a fixed order (App. A), which the
paper notes may change the stationary law again (§3.3), and set burn-in and
thinning without convergence diagnostics (Limitations). Neither law is `pi`,
which fixes the template's serialization, so we would use it only as a proposal inside an MH step; that use is our inference,
not the paper's. A single MH chain needs one fork, not `N`-way KV,
so it is not blocked behind particle cache work. No paper measures mixing for
2 to 5 slots.

## How to tell whether a sampler matches

Published "KL to target" numbers in this area are proxies. GAD
([2405.21047](https://arxiv.org/abs/2405.21047) §4, p.8-9) computes KL of the
empirical law in a 500-sample sliding window against the model's unnormalized
probabilities, which equals the KL to the target plus `-log C`, and notes that
GCD scored better on one benchmark because it favored the mode. We found no
paper that enumerates a real model's full constrained distribution; the
closest is [2610.03480](https://arxiv.org/abs/2610.03480) App. F, which
enumerates a top-8, four-token continuation tree of a Gemma adapter pair in
FP32/FP64 and computes MH and SIR output laws exactly (pp. 51-53).

The benchmark therefore tabulates once and simulates offline:

1. Enumerate every filled template with a prefix-trie walk (`fork_prefilled`
   plus one-token decodes), in FP32, recording each token probability, each
   slot's allowed mass and each forced-token probability. Re-score a sample
   without forks to measure the numerical noise floor.
2. Rungs 1 and 1b have closed-form output laws given the weight distribution
   under `q` ([2610.03480](https://arxiv.org/abs/2610.03480) App. C.1, eq.
   62-63), so they are computed exactly; rungs that need sampling (SMC) are
   simulated 1e5 to 1e6 times from the tables at no model cost, with
   forward-token cost from the trie.
3. About 1,000 model-backed runs per configuration test that the code draws
   from the same distribution as its own table simulation.

This separates "is the algorithm good at this budget" (Gate 2) from "does our
code implement it" (Gate 3).

Templates, each designed so later text pulls on earlier slots:

- T0: null control with uninformative trailing text.
- T1: only the final sentence identifies the pair, as in "[city] is the
  capital of [country]. Its currency is the yen." Prediction: SMC gains
  nothing over reweighted best-of-N at equal tokens.
- T2: the middle text is informative, as in "[animal] barks loudly. It is a
  [size] dog." Prediction: SMC beats reweighted best-of-N.
- T3: digit slots with a joint constraint, "[d1] + [d2] = 9."
- T4: a three-slot chain.

If T1 and T2 do not separate as predicted, the benchmark or the code is
wrong. Synthetic-table versions run first, including a control where masked
decoding is exact. Every template must pass a canonical-tokenization check
(encoding the concatenation equals concatenating the segment encodings). On
the Qwen3-0.6B tokenizer, no base-vocabulary token contains an ASCII digit
next to any other character, so digit slots have one tokenization (checked
2026-10-05; added special tokens not scanned).

Metrics: exact TV, KL in both directions and Jeffreys divergence where `q` is
computable; for sampled rungs, joint and per-slot TV against a null band
drawn from `pi` itself, mutual information between slots, `log Z - E[log
Zhat]`, `Var(log Zhat)`, a check that `Zhat` is unbiased, ESS and unique
ancestors. With `w` the product of allowed masses, `KL(q || pi) = log Z -
E_q[log w]` and `KL(pi || q) = E_pi[log w] - log Z`
([2404.17546](https://arxiv.org/abs/2404.17546) §5.1). The mean of
`sum log(1/alpha)` and `Var(log w)` are free on every masked run and are the
candidate triage signal for whether particles are worth spending; calibrate
them against exact KL here first. Calibrate `max w / sum w` alongside them:
empirical weight variance can report convergence long before IS has converged
([1511.01437](https://arxiv.org/abs/1511.01437) §2, Thm 2.1), and the
max-to-sum ratio is the diagnostic that paper proposes.

Report error against forward tokens per returned sample, in two columns:
logical (no sharing) and physical (with prefix sharing). Wall time and memory
are recorded separately.

## Stop rule, made concrete

The [existing stop rule](sampling-next-gates.md#stop-rule) needs a
predeclared gain. Proposed: SMC counts as a gain only if, at matched forward
tokens, its joint TV beats independence MH over the same number of draws
(rung 1b) by more than the null band on T2 or T4, and the T1 no-gain
prediction holds. The baseline is rung 1b rather than rung 1 because 1b is
free given the same draws and never worse than 1; that choice is our inference
from [2610.03480](https://arxiv.org/abs/2610.03480) Thm 1, whose enumerable
runs (Table 2 p.44, p.43) never test multi-slot templates. Otherwise KV-fork
resampling buys nothing over independent draws on this Mac, and the memory
measurement is skipped.

The `N=2` latency check must also wait for batched decode. Today two
particles are two sequential single-row forwards, so the check would measure
missing batching, not particle cost.

## Engine requirements, in build order

1. Teacher-forced scoring: return `model_logprob`, `constrained_logprob` and
   `allowed_log_mass` for caller-supplied tokens, and per-position log
   probabilities for a forced span. Needed by fixed-text weights, MH reverse
   moves and slot scoring. Today a token can only be consumed through the
   sampling transaction.
2. Grammar fork: build N constraint sessions from one checkpoint, sharing the
   tokenizer environment. The llguidance matcher already supports
   `deep_clone`, `rollback`, fast-forward and speculative token checks;
   metallix uses only `deep_clone` and masks.
3. Atomic resample over the block manager: validate the whole ancestor map and
   reserve the next step's copy-on-write blocks before any change, then fork
   duplicates and free losers. `BlockManager::fork` never allocates, so
   capacity failure surfaces at the next allocation; a loop of forks and frees
   is still not atomic because a free cannot be undone.
4. Particle tails stay out of the prefix cache, and finished (absorbed-EOS)
   particles release their blocks, since their KV is never read again.
5. Group scheduling: a request's particles are admitted, charged and
   preempted together.
6. Prefix-aware attention: read the shared prefix once per step with a tree
   mask. Without it, block sharing saves capacity but not memory traffic;
   prefix-aware kernels are where Hydragen
   ([2402.05099](https://arxiv.org/abs/2402.05099)) and DeFT
   ([2404.00242](https://arxiv.org/abs/2404.00242)) get their gains.

Items 1 and 2 serve the first experiments. Items 3 to 6 wait for the stop
rule.

## Prompts as programs

[AICI](https://github.com/microsoft/aici) ran user controllers in a Wasm
sandbox with mask, fork, backtrack and fast-forward operations. Its vLLM
integration ([PR 2888](https://github.com/vllm-project/vllm/pull/2888)) closed
unmerged and listed fast-forward, backtracking and forking as unsupported;
the tracking issue ([3714](https://github.com/vllm-project/vllm/issues/3714))
closed as not planned. The part that needed no engine-state changes became
llguidance, and its matcher and `toktrie` still carry AICI's `Splice` and
`Branch` vocabulary. Pie ([2510.24051](https://arxiv.org/abs/2510.24051))
runs sandboxed programs that own KV pages, at 3 to 12% latency overhead (2.4%
on an 8B model, 11.4% on 1B; Table 4), returns next-token distributions
truncated to the top 256 tokens by default (§4.2), and leaves security out of
scope (§8). SGLang's `fork` and `select` run in a client
interpreter over prefix caching; `select` is a length-normalized heuristic,
not a conditional probability.

Recommendation: a closed set of server-owned, typed operations plus declared
inference modes, with no user-code runtime. User code cannot declare a target,
so it would lose the `p`/`q`/`pi` receipt contract. Proposed operations, in
order: constrain (grammar as data, exists), score (item 1 above), fork and
release, splice (backtrack plus forced tokens, after tokenizer-boundary
qualification), and a declared mode naming target, proposal, method, `N` and
terminal weighted draw, where the engine owns the per-token loop. Resampling
is never a client operation. This reverses only if a named consumer needs
per-token logic that cannot be expressed as grammar data, potentials or these
operations.

## Diffusion models

Revised from the earlier [diffusion note](diffusion-text.md): the first
candidate is SDAR ([2510.06303](https://arxiv.org/abs/2510.06303)), a
block-diffusion model whose 4B configuration matches Qwen3-4B. It prefills the
prompt, denoises fixed-size blocks against the cache without writing KV, then
runs one more forward per block to write that block's KV under the
block-causal mask (§3.1-3.2, eq. 11; App. A.2 says the cache layout is
identical to an AR cache). So it reuses the Qwen forward with a block mask and
a pass that does not commit KV. Dream stays second because
it does multi-span infilling natively but needs a bias path and full-sequence
forwards. I-DLM ([2604.11035](https://arxiv.org/abs/2604.11035)) is third.
Batch-1 diffusion has not beaten autoregressive decoding on a Mac in public
reports, so SDAR stays experimental unless it beats Qwen3-4B tokens per second
at matched accuracy. SDAR cannot see right context for earlier blocks, so
infilling support is advertised per model. Diffusion denoiser outputs are not
autoregressive probabilities; using one as an MH proposal needs a derived
proposal density, which no paper provides.

## Order

DeepSeek V4.1 qualification still precedes particle serving, as in
[sampling-next-gates.md](sampling-next-gates.md#stop-rule). The offline
benchmark and the scoring API do not touch serving and can run alongside it:

1. Synthetic-table benchmark and the Gate 2 particle-count sweep, as offline
   tests.
2. Teacher-forced scoring API and grammar fork.
3. Qwen3-0.6B tabulation of T0 to T4; rungs 0, 1, 1b and 2 compared exactly.
4. SMC and the serving-core items, only if the stop rule's gain is met after
   batched decode exists.
5. Single-chain MH with an informed proposal.
6. SDAR reference capture and parity gates, after V4.1.

## Source limits

Read from the PDFs (2026-10-05): LLaMPPL v2, twisted SMC, GAD v4, Pie, MCMC
(2506.05754) and 2610.03480 in full, with the 2610.03480 App. A.2 proof
re-derived; Loula pp. 1-14 and 17-34; AWRS pp. 1-14, 20-26 and 42-50; Large
Language Gibbs pp. 1-14; SDAR pp. 1-23 and 29; 1511.01437 pp. 1-18;
2412.03719 pp. 1-12; Power-SMC v3 in full. I-DLM was not read. AICI, vLLM,
llguidance and toktrie
facts come from pinned source or issue history. No benchmark in this note has
been run.
