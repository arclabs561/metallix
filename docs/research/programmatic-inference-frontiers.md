# Programmatic inference frontiers

Status: research direction; no feature in this note is implied to be
implemented. Initial review 2026-09-22; contrastive decision models checked
2026-09-30. Sources are primary papers, author research posts,
and public implementation documentation. This note is a source ledger and
design input, not a ranking of methods.

## The larger opportunity

Metallix can become a runtime for interventions over model execution rather
than only a runtime that maps prompt tokens to output tokens. A useful
intervention may observe or change:

- token policy: masks, grammars, proposals, sampling and speculative commits;
- hidden state: probes, activation vectors, caps, feature ablations and
  model-specific residual-stream edits;
- external state: retrieval, tools, files, environments and persistent memory;
- search state: branches, particles, verifiers, recursion and hierarchical
  plans;
- model state: adapters, test-time parameters and carefully governed continual
  updates.

These are not one mechanism. They have different evidence, safety, state,
and reversibility contracts. The common runtime problem is to observe,
intervene, branch, verify, commit, and record the result.

## Research map

| Area | What it contributes | What Metallix would need | Status |
|---|---|---|---|
| Mechanistic observation | Features, circuits, probes, attribution graphs, latent workspaces, natural-language activation explanations | Activation capture, layer/model coordinates, probe artifacts, causal ablations, provenance and visualization | Research tooling, not a serving feature |
| Activation steering | Causal hidden-state edits, caps, vector interventions, behavior stabilization | Hook points, dtype/device-safe edits, per-model calibration, behavioral regression suite, emergency disable/fallback | Candidate adapter-local control |
| Symbolic control | Grammars, schemas, tools, programs, typed outputs | Tokenizer-aware masks, transactional controller state, byte/grammar provenance, bounded controller execution | Bounded JSON Schema path exists |
| Probabilistic control | Proposals, potentials, posterior targets, importance weights and SMC | Exact target/proposal declarations, support checks, ancestry, ESS, resampling, cache fork/restore | Offline/test-only primitive exists |
| Recursive inference | External context, subqueries, code-mediated reasoning, hierarchy and decomposition | Child-run budgets, context handles, recursion/cycle policy, partial results, aggregation and cancellation | Future orchestrator layer |
| Continual adaptation | Episodic memory, strategy memory, test-time updates and consolidation | Versioned writable state, provenance, rollback, evaluation gates, privacy and forgetting policy | Research only |
| Verifier/search loops | Outcome/process verifiers, tool feedback, tree search and self-correction | Candidate state, environment snapshots, reward/progress signals, search budget and stopping rules | Future functional layer |

## Mechanistic interpretability is an operational input

Anthropic's public research is useful here because it separates three claims
that are often conflated:

1. A representation can be identified or approximated.
2. A representation can be causally intervened on.
3. An intervention is useful and safe in deployment.

The [feature-mapping work](https://www.anthropic.com/research/mapping-mind-language)
and [scaling monosemanticity paper](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
show how sparse features can expose interpretable concepts. The
[circuit-tracing work](https://www.anthropic.com/research/tracing-thoughts-language-model)
connects features into pathways and reports both useful findings and serious
coverage limitations: the captured graph is only a fraction of total
computation, and interpretation still requires substantial human effort.

The [global-workspace research](https://www.anthropic.com/research/global-workspace)
describes a J-space of internal patterns that can be observed, modulated, and
used for multi-step reasoning, while explicitly presenting the method as
partial and imperfect. The [natural-language autoencoder work](https://www.anthropic.com/research/natural-language-autoencoders)
adds a human-readable activation explanation loop, but also reports
hallucinated explanations and high inference cost. These are excellent
arguments for receipts and corroboration, not permission to treat an
activation explanation as ground truth.

The [Assistant Axis work](https://www.anthropic.com/research/assistant-axis)
is closer to a functional runtime feature: monitor a representation, cap drift,
and test whether behavior stabilizes. The important design lesson is that a
steering artifact must include its extraction prompts/data, layer and token
position, intervention schedule, coefficient range, target model revision,
and behavioral side-effect evaluation.

## What real steering requires

Embeddings alone are insufficient. They are useful for retrieval, example
selection, clustering, memory indexing, and sometimes as inputs to probes, but
they do not identify a causal intervention site or guarantee a behavior change.

A functional steering stack needs:

```text
behavior specification
  → probe / feature / vector construction
  → intervention point and schedule
  → target-model forward hook
  → candidate generation or action
  → independent verifier/evaluator
  → rollback, accept, or quarantine
  → receipt and regression corpus
```

Minimum assets for one useful steering feature:

- a frozen target model and exact revision;
- positive/negative or contrastive calibration examples;
- a defined layer, position, normalization and coefficient schedule;
- a hook that preserves dtype, device, batch and cache semantics;
- a task metric and a side-effect metric;
- an unsteered control and a disable/fallback path;
- held-out prompts that test transfer, interference and adversarial failure;
- a versioned artifact containing the probe/vector and its provenance.

The output should be a qualified intervention, not an unbounded “personality
knob.” A safety or behavior monitor should be treated as a sensor with false
positives and false negatives, not as a proof of intent.

## Hierarchical and recursive inference

The [Recursive Language Models](https://arxiv.org/abs/2512.24601) work moves
long context and intermediate reasoning into an external environment where an
LM can inspect data, execute code, and recursively call another LM. This is
not merely a deeper prompt loop. It requires:

- a context handle rather than repeated token copying;
- bounded recursive child calls;
- isolation and cancellation;
- a typed result channel;
- aggregation or reduction over child results;
- cycle, depth and cost limits;
- an auditable tree of calls and evidence;
- explicit separation between executable code and untrusted model output.

Metallix should model this as an outer orchestrator over model adapters. The
inner decoder should not know whether a token came from a top-level request or
a recursive child. The orchestrator owns budgets, context stores, child
lifetimes and reduction; the adapter owns model state.

Hierarchical memory should likewise distinguish:

```text
working state       current tokens, activations, KV/recurrent state
episodic state      prior trajectories, tool results, receipts
semantic state      indexed facts, embeddings, summaries, procedures
parameter state     adapters or test-time learned weights
```

Embeddings are mainly useful in the semantic layer. They are not a substitute
for working-state snapshots or a safe continual-learning mechanism.

## Continual learning and writable state

“Continual learning” names several very different operations:

1. keep more context in a session;
2. store and retrieve episodic memory;
3. consolidate summaries or procedures into semantic memory;
4. update an adapter or small parameter set;
5. update the base model in place.

Metallix should implement and measure them in that order. Every writable layer
needs ownership, provenance, rollback, privacy/retention, conflict resolution,
and a held-out regression gate. Parameter updates should not be introduced as
an invisible side effect of inference.

Recent continual-learning work explores persistent memory, test-time learning,
and self-modification, but those results do not establish that online weight
updates are safe or useful for a local serving runtime. A first useful feature
is more likely to be a versioned strategy/episodic memory with evaluator-gated
promotion than automatic base-model mutation.

## Verifier and search loops

Useful steering needs an external notion of progress. Candidate sources
include:

- deterministic program/test execution;
- schema and grammar acceptance;
- a domain calculator or simulator;
- an outcome verifier;
- a process verifier;
- tool/environment feedback;
- a learned reward or preference model;
- a mechanistic monitor.

These sources have different trust levels. Metallix should represent them as
typed evidence and keep the distinction between “candidate scored well” and
“candidate is correct.” A recursive controller or SMC policy should be able to
use the signal without pretending it is a probability of correctness.

## Contrastive decision models

[CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)
uses a frozen Qwen3-8B encoder with last-token pooling and separate trainable
state/action projection heads. Normalized projected embeddings are compared
with a learned logit scale. Its contrastive objective and cached action
representations make it relevant to candidate ranking and verifiers; it is
not a replacement for autoregressive generation or the Julia encoder contract.
The model revision reviewed is `e939398d4556fcd9400c76fa8c5a513202f42b0a`.

The upstream [head implementation](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/src/clm/heads.py)
and [fine-tuning entry point](https://github.com/Contrastive-LM/CLM/blob/bb42c6c5bf914fd449bed2f6ca65be80602cb1f7/train/finetune.py)
support a useful narrow training boundary: frozen encoder embeddings feeding
trainable heads. The published `.pt` checkpoint needs a trusted conversion
and an explicit tensor manifest before native import. A small head does not
remove the cost of encoding new states with the 8B backbone.

A first Metallix experiment should fix encoder revision, tokenizer, pooling,
head weights, normalization and candidate ordering; prove head parity on
frozen embeddings; then measure a held-out ranking task against the raw encoder
and existing decision route. A native encoder path and action-cache invalidation
need their own tests. Probabilities remain relative to the candidate set and
must not be presented as calibrated correctness. This is a proposed adapter
qualification, not implemented CLM support. Reading covered the model card,
repository overview, heads and training entry point, not independent benchmark
reproduction.

## Gates before making this functional

1. **Observation gate:** capture a small set of model activations with exact
   model/layer/position provenance and no output-behavior claims.
2. **Intervention gate:** apply one calibrated vector or cap to an open model,
   compare against an unsteered control, and test transfer/interference.
3. **Verifier gate:** connect one deterministic tool or test evaluator to a
   controller and record candidate, score, evidence, and rollback.
4. **Recursive gate:** run bounded child calls over an external context handle,
   with cancellation, depth, cost, and cycle tests.
5. **Memory gate:** persist episodic/semantic state with versioning, provenance,
   deletion, and held-out promotion checks.
6. **Particle gate:** only then connect SMC to a real adapter with physical
   cache ancestry checks and finite-population error measurements.

## Source limits

Anthropic's public posts and papers are compelling primary evidence for
interpretability mechanisms, but they are not a turnkey open-weight runtime
for Metallix. The RLM and continual-learning results are fast-moving research;
their reported quality and cost do not establish Apple-Silicon behavior. No
claim here establishes that activation steering is robust across model
families, that internal explanations are faithful, or that online parameter
updates are safe. Those are explicit open questions.
