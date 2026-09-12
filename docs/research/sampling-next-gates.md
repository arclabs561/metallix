# Sampling: the next falsifiable gates

## Decision

Do not add a generic sampler framework or a particle-serving API yet.  A
narrow seeded categorical policy is now wired to the Qwen diagnostic, including
`--json-schema`: `mx gen --sample --temperature T --seed S` has no top-k/top-p
truncation and records both raw model `p` and deployed-policy `q` when
`--logprobs` is requested. The grammar path is transactional, and same-checkpoint
sampled grammar replay passed on the local Qwen control. The next work should prove
the remaining three small boundaries in order. The first two need no checkpoint,
Metal execution, or cache fork. The third needs a qualified decode path;
confidence calibration also needs independently labelled data.

This note follows [GenLM/LLaMPPL](genlm-control.md),
[Feynman--Kac methods](feynman-kac-steering.md),
[Power-SMC](power-smc.md), and [uncertainty](uncertainty.md).  It is a
consolidation, not a new literature survey.  The LLaMPPL v2 paper was read in
full there; the FKC, SMC-SD and AWRS sources have the explicitly selective
coverage recorded in their respective notes.

## Offline EOS particle accounting

`crates/engine/tests/eos_particle_target.rs` extends the finite target oracle
to EOS-terminated paths. It checks incremental log weights, normalization,
effective sample size, duplicate/discarded ancestors, absorbing EOS and
population extinction. Multinomial ancestor draws and weighted terminal
selection exercise the engine's categorical sampler against an independent
CDF oracle at fixed entropy values.

The exact enumerated target has normalizer 0.36; the chosen finite particle
trace estimates 23/30 and loses a supported path. That deliberate disagreement
guards against presenting finite-particle inference as exact conditioning.
This is test-only accounting, not a particle runtime, model-backed cache
fork, convergence experiment or calibrated-confidence feature. Physical
cache-fork/reindex parity remains a prerequisite for model-backed particles.

## Gate 1: one explicit sampled-policy distribution

The implemented Qwen slice accepts FP32 logits, a legal mask, positive finite
temperature, and a deterministic `ChaCha8Rng` variate. It returns a token and
temperature-conditioned `sampling_logprob`, using FP64 accumulation.
`sampling_policy` records the RNG, seed, temperature, uniform conversion and
absence of truncation; `model_logprob` remains raw temperature-one `p`. The
path rejects nonfinite logits and temperature scaling that overflows. Floating-
point underflow can remove tiny probabilities, so it is not an exact-real
importance proposal with guaranteed full support.

This closes seed management, `p`/`q` receipts, and bounded same-policy replay for
the diagnostic. It does not close cross-device replay, top-k/top-p,
device-versus-reference distribution parity, or any whole-sequence target
claim.

The engine now additionally exposes
`JsonConstraintSession::select_categorical_with_logprobs`: caller-supplied
temperature and uniform entropy, with four separate receipts (`model_logprob`,
`constrained_logprob`, `allowed_log_mass`, `sampling_logprob`). It prepares the
mask and consumes a token on a deep-cloned matcher, committing grammar state
and decoded bytes only on success. Padded model rows contribute to raw `p`
but never grammar `q`. Greedy selection shares this transactional boundary.
This is not a fast-forward or cache-fork implementation. The engine owns no
RNG. The CLI supplies candidate entropy from a cloned seeded policy, checks
that the model vocabulary fits signed server token IDs before a grammar advance,
and commits the candidate RNG only after `ConstraintRun` succeeds. Thus a
failed grammar draw commits neither RNG nor grammar state. This transactional
boundary is wired to `mx gen --sample --json-schema`.

Local sampled-schema qualification used FP32 Qwen3-0.6B, input IDs
`9707,11,1879`, the record schema, temperature 0.7, seed 42 and token limit 29.
Resident, repeated resident, unscored resident and streamed runs independently
validated `{"status":"ready","count":1}` and selected identical 12-token
sequences. All four probability fields matched exactly across the three scored
runs. The greedy and unconstrained sampled controls retained their prior IDs.
Redirected preview preserved diagnostics without ANSI escapes and stdout stayed
JSON. Receipts: `artifacts/sampled-schema-{resident-a,resident-b,unscored,streamed}.json`,
`artifacts/sampled-schema-{greedy-control,unconstrained-control}.json` and
`artifacts/sampled-schema-preview.stderr`. See [the development guide](../../DEVELOPMENT.md)
for the command and streamed budgets. This is bounded replay evidence, not a
cross-device reproducibility or throughput claim.

The shared transactional boundary was exercised through the existing greedy
Qwen schema command (prompt `9707,11,1879`, limit 64, record schema, logprobs).
Three before/after fresh-process runs preserved every generated ID and
probability receipt and independently validated `{"status":"ready","count":1}`.
Total selection time across 12 tokens was 6.411/6.687/6.551 ms before, versus
11.282/11.461/11.107 ms after. The roughly 4.7 ms/request overhead is retained
for error atomicity, not reported as an optimization. This measures the
whole selection-boundary change, not matcher cloning in isolation; the two
three-run blocks were sequential, not an interleaved crossover experiment.
Receipts: `artifacts/grammar-transaction-{before,after}-{1,2,3}.json`.
Release/all-features binary SHA-256 before (`9259433`):
`514b1e4f588a3f5e3e49c96fa8596ecca70a85b1485504b15107c14ee9b0b0bc`;
after: `f3d03a73ebdfb14b343eeb3ccadc5e20fa5d627a09422f261cd57305c11bef92`.
This used the same local M3 Max and Qwen checkpoint described in
[the streamed-generation ledger](../experiments/streamed-generation.md),
without a file-cache flush.

**Question.** Given raw logits and a grammar allowed set,
what distribution actually selected the token?

The stateless primitive now has an independent finite-toy `f64` oracle in
`crates/engine/src/sampling.rs`: direct uncentered exponentials and normalized
probabilities are compared with the production centered implementation at
fixed entropy values. Cases cover ties, masked/padded rows, temperature, and
a representable tiny legal mass, checking selected IDs and policy logprob.
The tiny row stands in for EOS only numerically; the primitive does not know
grammar or EOS semantics. This is not a statistical distribution test or an
oracle for underflowed tails.

The CLI policy tests now join a separately seeded ChaCha8 stream to a direct
finite-distribution oracle, checking eight scored and unscored draws plus
rollback after a rejected draw. A separate bit-basis test checks all 64 source
bits in the 53-bit uniform conversion, including discarded low bits and both
extrema. These check policy consumption and conversion, not the ChaCha
algorithm itself, statistical convergence, or cross-device/model parity.
Run `cargo test -p server --all-features qwen_forward::tests` for this join.
Run `cargo test -p engine independent_f64_oracle` for the new toy checks.
The engine tests now cover grammar masks analytically, accepting EOS without
special-token bytes, and rejected draws/output limits without committed state
changes. Top-k/top-p and their support changes remain future inputs, not
current `mx gen` behavior.

The receipt must name three distinct things: raw model `p`, deployed sampling
policy `q`, and (only when an SMC/control experiment names one) target `π`.
For normal constrained sampling, `q` may be the model conditioned on the
allowed set.  For a global constraint or a power target it usually is not `π`;
the incremental ledger must retain `log π_increment - log q_increment`.
Hard top-k/top-p or an EOS ban can remove target support, so they cannot be
quietly used as an importance proposal.  This is the support condition in the
LLaMPPL Feynman--Kac formulation and the Power-SMC target/proposal analysis
([LLaMPPL v2, §§2--3](https://arxiv.org/html/2306.03081v2),
[Power-SMC](power-smc.md#correctness-boundary)).

**Advance criterion:** analytic oracle tests pass and unit traces can make a
replayable claim about `p`, `q`, policy parameters, seed and grammar-mask
identity. A real-model grammar replay must reproduce IDs and all probability
receipts under the same checkpoint and execution policy. This does not prove
Metal sampler parity; that later needs a device-versus-reference distribution
test.

## Gate 2: target/weight/ancestry oracle before cache work

**Question.** Does a finite particle run implement the requested target rather
than a plausible-looking heuristic?

Build a separate tiny EOS-terminated tree oracle.  Enumerate completed strings
to obtain the normalized target, then run fixed-seed particles with a named
proposal.  Assert each incremental log-weight, `logsumexp` normalization,
ESS, ancestor vector, terminal (absorbing-EOS) handling, and final **weighted
categorical** selection.  Include repeated and discarded ancestors.  Sweep
particle count and report distribution error; finite `N` must remain labelled
an approximation even when the normalizer estimator is unbiased.

This closes the otherwise easy errors identified by both papers: using a local
mask with unit weight targets the *locally normalized* conditional, not the
global completed-string conditional; selecting its maximum token is the
separate greedy choice.  Choosing maximum-weight particle output likewise
changes a sampling method into optimization.
LLaMPPL defines a target through Markov transitions plus potentials and
identifies consistency only as particle count grows; Power-SMC spells out the
sequence-level target and absorbing EOS state
([LLaMPPL v2](https://arxiv.org/html/2306.03081v2),
[Power-SMC](power-smc.md#what-distribution-it-targets)).

This is also offline now.  It requires no model logits and no cache.  It must
finish before `N=2` is attached to a real request, because a real run cannot
diagnose whether an apparent output change is target error, proposal mismatch,
resampling variance, or model quality.

## Gate 3: cache ownership, then uncertainty calibration

**Question.** Can real decode state be forked/reindexed correctly and does any
reported uncertainty mean more than a token statistic?

The cache subgate requires an actual qualified model decode and architecture-
specific sequence state.  Take an ancestor map containing duplicates,
discarded parents and one EOS particle; fork/reindex all cache and logical
state atomically.  Compare each resulting next-logit vector with independent
replay of its token prefix.  Measure allocated, shared and resident KV bytes,
unique ancestors, fork/reindex time, and release behavior.  A shared immutable
prompt prefix plus copy-on-write divergent tails is admissible; aliasing a
mutable Metal buffer is not.  Only after this differential passes may a
2/4-particle Mac measurement record latency, ESS, normalizer estimates and
memory together.  The cited SMC designs require prefix/KV sharing, but neither
establishes that it is cheap on a unified-memory Mac
([cache discussion](genlm-control.md#cache-and-mac-implications),
[Power-SMC cache gate](power-smc.md#bounded-gates-before-an-implementation-choice)).

Uncertainty has a parallel two-stage rule.  A full-vocabulary `f64` reference
can now validate raw entropy and varentropy reductions, with separate raw and
constrained namespaces.  Turning either into confidence needs model execution
over a frozen checkpoint/template/policy plus labelled held-out data, then
calibration and risk--coverage reporting.  ESS and particle weights measure
proposal mismatch; they are not calibrated answer correctness
([uncertainty limits](uncertainty.md#calibration-gate),
[FK limitation](feynman-kac-steering.md#constraints-logprobs-and-uncertainty)).

## Stop rule

Ordinary single-sequence categorical sampling, including the transactional
grammar-mask path, has integrated with an explicit seed, temperature and
`p`/`q` receipt. Local Qwen sampled-grammar replay passed as described above.
Particle or global-control serving remains behind Gate 2 and the cache-replay
portion of Gate 3. Confidence claims remain separately behind Gate 3's labelled
calibration work. A result that reverses particle work is simple: if cache
replay is correct but `N=2` increases per-token latency or memory without a
predeclared distribution/quality gain, do not pursue particle serving on this
Mac. DeepSeek V4.1 Flash forward and cache qualification still take precedence
over all three gates.

## Fresh GenLM Control check: integration traps

Fresh primary-source read on 2026-09-11: `genlm-control` main commit
[`533a36e`](https://github.com/genlm/genlm-control/tree/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d), Apache-2.0. This is separate from
the full LLaMPPL-paper coverage recorded in `genlm-control.md`. Read coverage
here: sampler documentation plus `token.py`/`sequence.py`, and their
token/AWRS/sequence tests and stateful-potential test; no code was run.

1. **Keep target weight and sampling logprob separate.** `DirectTokenSampler`
   samples a *normalized* proposal but returns an *unnormalized* target weight:
   without an alternate proposal that is `log Z_target`; with one it is
   `log target_weight(token) - log q(token)`. Its third return value is the
   proposal's log probability. Metallix's external-variate primitive should
   therefore continue to return the normalized deployed-policy logprob, not
   call it a control weight. Add the GenLM-style forced-token test: enumerate
   toy target/proposal vectors and assert both values independently. Require
   one tokenizer fingerprint and EOS policy, not merely matching integer
   widths. ([implementation](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/genlm/control/sampler/token.py),
   [tests](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/tests/sampler/test_token_sampler.py))

2. **A bounded rejection path is an SMC experiment, not ordinary sampling.**
   AWRS's `proper_weights=False` deliberately omits correction; finite
   rejection can instead return zero (`-inf`) weight. At the length boundary,
   GenLM forces EOS and adds the target's unnormalized EOS weight, which is
   different from sampling EOS under a normal categorical policy. Any future
   Metallix AWRS/control mode must expose acceptance/rejection, zero-weight
   particle death and the exact terminal rule; it must not substitute an
   arbitrary fallback token. Transfer the tests for forced EOS and
   proposal-weight identity before a model run. ([AWRS implementation](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/genlm/control/sampler/token.py),
   [sequence contract](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/genlm/control/sampler/sequence.py),
   [sequence tests](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/tests/sampler/test_seq_sampler.py))

3. **Do not infer KV fork semantics from a stateful control API.** GenLM's
   default `ParticleState.clone` builds a new state then replays its full
   context; its stateful cache is a bounded host-side pool. That is a useful
   correctness fallback but no evidence of copy-on-write device KV or cheap
   resampling. Our Gate 3 next-logit replay differential remains required,
   including duplicate/discarded ancestors and resource release. ([state
   implementation](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/genlm/control/potential/stateful.py),
   [state tests](https://github.com/genlm/genlm-control/blob/533a36ecaffb2a2c4ebf6b18dab68f7e462d099d/tests/potential/test_stateful.py))

## LLaMPPL motivation: programmable inference, bounded

Fresh repository check: [genlm/llamppl at `7ad531a`](https://github.com/genlm/llamppl/tree/7ad531a506ccac939651553fa595bb036db18f80) is a non-archived, non-fork
Apache-2.0 GitHub repository (main commit 2026-06-07); its [README](https://github.com/genlm/llamppl/blob/7ad531a506ccac939651553fa595bb036db18f80/README.md) calls it a
research prototype formerly named `hfppl` and documents optional MLX backend
support. [probcomp/LLaMPPL](https://github.com/probcomp/LLaMPPL) also resolves
as a separate, non-archived non-fork repository, so this check establishes no
redirect or ownership-transfer claim. `genlm-control` is separately newer
(created 2025-01-10). The `llamppl` project file says MIT while GitHub metadata
reports Apache-2.0; treat that discrepancy as a no-copy boundary.

The attractive idea is not more sampling knobs. A program's `step` describes
base draws, hard `condition`, soft `observe`, termination, and optionally a
proposal; inference then exposes the resulting target and weights. In the
paper's formalization, conditions contribute a zero/one potential, observations
contribute likelihood, and `sample(dist, proposal)` contributes the ratio. A
proposal changes computation, not the named target. Hard constraints therefore
remove support, whereas strictly positive soft weights retain it with unequal
mass (a zero likelihood also removes support).
This does **not** justify exposing arbitrary user programs or embedding a
Python-like DSL in Metallix's generation API. The full v2 paper reading and
its finite-particle caveat remain recorded in [GenLM control](genlm-control.md).

**Implemented offline accounting experiment:**
[`programmed_target.rs`](../../crates/engine/tests/programmed_target.rs)
enumerates a private two-step program, not an interpreter API. It draws
`x ∈ {A,B}` then `y ∈ {0,1}`
from enumerated base probabilities; applies hard
`condition(!(x == B && y == 0))`; then soft-observes `true` from a tabulated
Bernoulli likelihood `r(x,y)`. It uses an alternate full-support proposal `q`
and records, for every leaf, `log p`, hard indicator, `log r`, `log q`, and
`log weight`. Direct enumeration checks the target proportional to
`p(x,y) 1[condition] r(x,y)`; the proposal-weighted sum recovers its
unnormalized mass. This checks program-to-potential accounting and distinguishes
hard from soft control without a model, RNG-owned public API, grammar backend,
or cache fork. The leaf target masses are `[0.21, 0.045, 0, 0.288]`, with
normalizer `0.543`. The nonuniform proposal `[0.25, 0.25, 0.125, 0.375]`
recovers that mass with correction; omitting correction gives `0.17175`
and changes the normalized distribution. A separate test exercises the
production categorical primitive's staged proposal log probabilities.

Run `cargo test -p engine --test programmed_target`. This is a deterministic
finite-sum identity, not evidence of finite-particle accuracy. EOS trees,
resampling, ESS, ancestor accounting, and cache replay remain separate gates.
