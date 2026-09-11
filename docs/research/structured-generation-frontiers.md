# Structured generation after AICI

Checked 2026-09-11. These are research inputs, not implemented Metallix
features or measured Mac speedups. Start with the
[baseline constraint qualification](constrained-generation.md).

## Historical anchors

These are selected milestones, not claims of invention or an exhaustive history.
For these older papers, this pass checked metadata and abstracts only.

| Period | Contribution |
|---|---|
| 2021 | [PICARD](https://arxiv.org/abs/2109.05093) uses incremental parsing to reject inadmissible tokens during generation, notably for SQL. |
| 2022–2023 | [LMQL](https://arxiv.org/abs/2212.06094) combines prompting, scripting and output constraints in a programming language. |
| 2023 | [Outlines' guided-generation paper](https://arxiv.org/abs/2307.09702) indexes the model vocabulary against automaton states for efficient constrained decoding. |
| 2024 | AICI exposes programmable generation control; [XGrammar](https://arxiv.org/abs/2411.15100) develops an efficient grammar engine. These are different layers, not interchangeable projects. |
| 2025 | [JSONSchemaBench](https://arxiv.org/abs/2501.10868) measures efficiency, schema coverage and output quality rather than JSON parsing success alone. |

Structured output is the desired result; constrained decoding is a mechanism;
a grammar specifies accepted forms; an AICI-style controller orchestrates
generation. The distinction matters when comparing capabilities and costs.

## AICI and LLGuidance

[AICI's pinned README](https://github.com/microsoft/aici/blob/ecc50362fe2c620c3c8487740267f6fae49397d3/README.md)
explicitly recommends LLGuidance as its maintained evolution and specialization
for constrained decoding. AICI is broader: token masks, forced tokens,
backtracking, generation forks and controller communication. Its Wasm runtime
and process protocol are not prerequisites for native Rust grammar masking.
GitHub metadata reports `archived: false` and last push 2025-01-22; low recent
activity is not the same as repository archival.

The transferable idea is to keep generation control separate from model
execution, with tokenizer-specific state. Ordinary masked sampling comes first.
Forks or rollback must eventually update model state and grammar state together;
they do not justify introducing empty universal interfaces now. AICI's published
latency measurements used an EPYC/A100 host, not Apple Silicon.

## Token boundaries, forced tokens and output dialects

LLGuidance's [fast-forward discussion](https://github.com/guidance-ai/llguidance/blob/bae21e505859b9a338b7aeefaa8a64319ba41831/docs/fast_forward.md)
explains why known bytes cannot simply be encoded independently and appended:
alternate tokenization and boundary healing affect which tokens can be forced.
Qualify exact tokenizer bytes and ordinary masks before enabling fast-forward.

XGrammar's [matcher API](https://github.com/mlc-ai/xgrammar/blob/d02ad2b155a5f0c3eaa711950d154218d982ae7e/include/xgrammar/matcher.h)
separates compiled grammar from mutable matching and exposes fork, rollback and
draft-tree traversal. Its
[tool/reasoning format documentation](https://github.com/mlc-ai/xgrammar/blob/d02ad2b155a5f0c3eaa711950d154218d982ae7e/docs/structural_tag/tool_calling_and_reasoning.md)
also distinguishes model-specific output formats. This is relevant to Qwen and
DeepSeek reasoning/tool boundaries, but is not evidence of Metallix support.
The inspected SHA is a main-branch snapshot, not a claim about the exact API of
the v0.2.6 release.

## Dynamic schemas in XGrammar-2

[XGrammar-2 v2](https://arxiv.org/html/2601.04426v2) adds tag-triggered switching
between free text and subgrammars, substructure-level cross-grammar caching,
Earley-based adaptive masks, JIT compilation and repetition compression.
JIT shifts work from compilation to decoding; both times need measurement.
Cache equivalence must include tokenizer and lookahead context. The cycle-hash
algorithm covers simple reference cycles, not arbitrary graph equivalence.
Repetition compression requires a nonempty repeated rule. Methods and evaluation
were read; appendices were only skimmed. Reported hardware is EPYC/Xeon with
NVIDIA GPUs, not a Mac.

## Parser Stack Classification

[PSC, arXiv:2608.03065v1](https://arxiv.org/html/2608.03065v1), targets deterministic
grammar-constrained decoding. It precomputes token legality over parser-stack
and lexer states, merges/minimizes automata, then maps the current parser state
to a precomputed vocabulary mask. Stack processing remains proportional to
stack depth; selecting the resulting mask does not scan the vocabulary.

The attractive result is low runtime mask cost. The counterweight is compilation
and storage: reported Qwen JSON preprocessing takes about 28.6 seconds and
3.13 GiB, while programming-language grammars can require much more; Qwen SQL
has multi-GiB runtime tables. Mask-only speedups exclude GPU transfer/application,
and end-to-end measurements use vLLM/A100, not Metal. The inherited lexer also
has Unicode boundary limitations.

Candidate experiment: one fixed JSON schema and tokenizer, measure artifact
bytes and compilation cost, compare every mask against an independent baseline,
then measure end-to-end decode. Reject the approach if artifacts compete with
the model's memory budget. Public [PSC code](https://github.com/Gompyn/PSC) was
located; it was not executed or license-audited here.

Reading coverage: main sections 1–6, figures/tables and references; not a claim
of a line-by-line reading of every supplementary page.

## Draft-conditioned constrained decoding

There is a separate distribution question:
[Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047) distinguishes local
valid-token masking from sampling the original model conditioned on the whole
output satisfying a grammar. They are not generally equivalent. This pass
checked its abstract only; no implementation or convergence claim is adopted.

[DCCD, arXiv:2603.03305v2](https://arxiv.org/html/2603.03305v2), separates semantic
planning from structural realization: generate an unconstrained draft, then
condition constrained generation on that draft. Its multi-draft variant scores
candidates using feasible probability mass. This deliberately changes the
generation distribution; it is not an exact acceleration of ordinary sampling.

Reported quality improvements need two-pass cost accounting. Evaluation uses
Qwen/Llama 1–14B, XGrammar and reasoning tasks; supplemental latency uses H100
and batched vLLM. A wrong draft is rarely repaired by the second stage in its
reported analysis. Correct syntax does not prove a correct answer.

Candidate experiment, after baseline sampling works: compare unconstrained,
ordinary constrained, and one-draft constrained generation on identical tasks.
Record independent schema validation, task accuracy, total tokens, end-to-end
latency, peak memory and cancellation behavior. Do not report only mask speed.
Public [DCCD code](https://github.com/avinashreddydev/dccd) was located but not
executed or license-audited here.

Reading coverage: main method/experiments/conclusion and appendices A–E/L;
not every example in the 45-page supplement. Related
[CRANE](https://arxiv.org/abs/2502.09061) alternates unconstrained reasoning and
constrained output; its full paper was not read in this pass.

## Order of implementation

1. Exact vocabulary bytes, invalid-token rejection and incomplete/complete EOS.
2. Real logits and ordinary constrained sampling with independent validation.
3. Measure schema compilation, CPU masks, Metal mask application and decode
   separately; include schema coverage and task quality.
4. Only then compare tokenizer-aware fast-forward, reusable compiled masks,
   reasoning-aware output regions and draft-conditioned generation.

DeepSeek execution and bounded-memory loading continue independently. No grammar
optimization substitutes for a correct model forward or a measured memory bound.
