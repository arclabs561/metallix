# Serving adoption is not serving performance

Source: [LLM Serving in the Wild, v1](https://arxiv.org/html/2608.03036v1),
checked 2026-09-11. Coverage: abstract, sections 1–9, all tables and figures;
references not individually followed, replication package not reproduced.
Captured HTML: `artifacts/paper-2608.03036v1.html`, SHA-256
`9b048842ab09a6f20e7d3f81eb7d6d00af11d1f5e96b947ce09ed97d614e7881`.

## Evidence and cautions

This mines Python API occurrences and repository descriptions across five
selected frameworks; it does not benchmark inference. Selection filters and
static detection limit conclusions about Mac runtimes, defaults and actual
production usage (§3). Co-occurrence does not prove runtime integration.

Audit flags: Table 6 maps Multi-LoRA to network pruning. Table 13 lists 121
SGLang repositories in one topic despite Table 2's 54 filtered repositories;
the population relationship needs clarification. Figure 3c's 40% quantization
plus memory combination exceeds Table 3's 13.3% quantization frequency.
Do not use these counts as feature priorities without auditing the mapping
and denominators. Observed model usage is not a support matrix.

## Metallix interpretation

Keep kernel specialization separate from orchestration. Measure optimization
combinations, not just isolated wins. Our next cache/offload experiments should
retain explicit compatibility and ownership contracts, parity gates and mixed
workloads. The paper motivates these checks; it does not validate our design
or establish a speedup. See [development guardrails](../../DEVELOPMENT.md#keep-model-assumptions-local).
