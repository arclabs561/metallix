# LLMShare: hardware design-space reading

## Finding

[LLMShare](https://shipxu123.github.io/papers/C15-DAC2025-LLMShare.pdf) is a
hardware-architecture design-space exploration (DSE) paper, not a model
runtime, a weight-sharing scheme, or a way to run one model above host memory.
It searches separate *prefill* and *decode* server pools for a cost/serving-time
Pareto frontier. Its core conclusion is credible only within its simulated,
multi-accelerator setting: phase demands should be measured separately before
provisioning or scheduling them.

## Mechanism

Each pool varies server/device/link counts, main/global/local memory, cores,
lanes, and array/vector dimensions (Table I). The proposed serving simulator
combines a cost model, LLM attributes, and a request trace with a scheduler,
latency model, and a KV-transfer delay between pools (Fig. 4). It says it uses
LLMCompass for latency/cost and Splitwise's scheduler; this note did not
independently read or validate either dependency.

Because that space is about 9 × 10^14 configurations, LLMShare uses
two search choices. Memory-Centric Initialization stratifies samples by total
pool main-memory capacity, then samples within percentile bins (Algorithm 1).
A Gaussian-process surrogate uses a recursively constructed tree embedding of
the hardware hierarchy as its kernel input, then selects by expected
hypervolume improvement (Sections III.B--D). The optimization target is
simulated serving time and cost, not numerical model correctness or output
quality.

## What the evaluation establishes

The experiment is an offline dataset of 1,055 random designs for GPT-3 175B,
with a two-minute Azure-derived trace of 2,454 requests. It initializes 10
designs, runs 20 exploration steps, and reports ten-run averages against four
DSE baselines (Section IV.A). On that simulator dataset, Table II reports
LLMShare ADRS 0.1589 and hypervolume 5.0552 × 10^8; Table III compares
one Pareto H100-cluster configuration against one found configuration at
normalized cost 0.87 and normalized RPS 4.11. Thus the often-quoted “13% lower
cost, over 4x throughput” is a normalized, simulator-selected comparison, not
an end-to-end run of a released serving engine on hardware. It does not show
quality, numerical parity, tail latency, or a single-device result.

## Fit for Metallix

This is not directly applicable to the current single-Mac target: its template
assumes 1--10 servers per pool, 4--16 devices per server, HBM-like memory
choices, accelerator links, and a KV hand-off across pools. A fixed Apple
Silicon machine cannot independently choose those properties, and moving KV
between local processes would add copying without the heterogeneous-cluster
benefit. It also says nothing about demand paging weights from SSD, streamed
layers, or architecture-specific operators such as MoE, MLA, state-space, or
hybrid attention.

The transferable experiment is smaller: record prefill and decode separately,
run a representative local request trace, and choose among *software* policy
budgets (resident-weight limit, KV admission reservation, batch/token budget,
and concurrency) on a measured Pareto frontier. Include TTFT, ITL,
throughput, p95/p99, unified-memory pressure, SSD reads, cache hit rate, and
reference agreement. A result would reverse the phase-specific policy if one
shared local queue wins on those measurements without violating latency or
memory headroom.

Rust types should protect those model-independent boundaries, rather than
freeze the paper's hardware tree or Qwen/DeepSeek layouts into the engine:

- make prefill output and appendable decode/KV state distinct, so a decoder
  cannot consume a cache whose compatibility key differs in model revision,
  token IDs, positional configuration, adapter, or cache layout;
- make architecture capabilities explicit data checked at planning time
  (attention/cache kind, layer dependencies, weight representation, supported
  kernels). Use exhaustive enums only for a closed local choice; a new model
  extension should provide its own validated capabilities, not enlarge a
  giant engine-wide architecture enum;
- keep residency/admission reservations separate from actual allocator
  measurements. A `PlannedBytes`-like value must not masquerade as process
  peak memory.

Those contracts scale across architectures while leaving operator lowering and
kernel choice behind a per-model implementation boundary. They are candidates
for design after cached generation exists, not evidence to add cluster routing
now.

## Provenance and coverage

Source captured locally as `artifacts/LLMShare-DAC2025.pdf`, SHA-256
`7b42e7db7092a72869b07aa270723273eac87e46f8795540f1fe42ba54b7cf1c`.
Read in full: all seven PDF pages, abstract, Sections I--V, Figures 1--8,
Tables I--III, Algorithm 1, equations, conclusion, acknowledgements, and the
reference list. The paper has no appendix. Its cited LLMCompass, Splitwise,
and Azure trace documentation were not read here; their claims remain
second-hand in this note.
