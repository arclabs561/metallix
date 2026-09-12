# Research and implementation provenance

Follow a finding from its source to the code, test, or measurement that uses it.
Research is not implementation, and an upstream benchmark is not a Metallix result.
This index covers the recorded work below; older unversioned references remain
explicit gaps, not retrospectively pinned evidence.

## Reading map

| Question | Notes |
|---|---|
| What does V4.1 require, and what do its optimizations actually mean? | [Efficiency requirements](efficiency-methods.md) |
| What might let a model exceed a Mac's RAM? | [Host-memory investigation](host-memory.md) |
| Which precision and quantization ideas are faithful formats versus new quality experiments? | [Quantization and precision](quantization-precision.md) |
| What gates separate categorical sampling, particle control, and calibrated uncertainty? | [Sampling implementation gates](sampling-next-gates.md) |
| How would diffusion text generation differ from the token-step decoder? | [Diffusion text models](diffusion-text.md) |
| Which other runtimes offer ideas worth testing? | [Runtime patterns](reusable-runtime-patterns.md) |
| What have we reproduced for V4.1? | [Sparse-indexer qualification](../experiments/v41-candidates.md) |
| What runs on Metal, and what improved? | [Qwen qualification ledger](../experiments/qwen-metal.md) |
| Can selected weights execute a block, and what checks precede V4.1 loading? | [Loader qualification](../experiments/loader-qualification.md) |
| How are experiments run? | [Developer guide](../../DEVELOPMENT.md) |
| What can we learn from serving adoption, hardware DSE and teaching implementations? | [Serving in the wild](serving-in-the-wild.md), [LLMShare](llmshare.md), [Inference book code](llm-model-inference.md) |

## Technical reference guide

Read these as engineering references, not a list of implemented features.
Each chapter records source coverage and applicability; full-paper reading
is still pending where the ledger says excerpts or selected sections.

| Topic | Reference | Primary decision |
|---|---|---|
| Memory and file I/O | [Metal memory](metal-memory.md) | Storage modes, allocation ownership, residency and measured working sets |
| GPU execution | [Metal execution](metal-execution.md) | Queue ordering, synchronization and resource lifetime |
| Kernels and profiling | [Metal kernels](metal-kernels.md) | Threadgroups, specialization, matrix APIs and counter-driven experiments |
| Training | [Training efficiency](training-efficiency.md) | Rematerialization, precision, microbatching and adapters |
| Serving | [Serving efficiency](serving-efficiency.md) | Attention, batching, KV reuse, speculation and offload |
| Structured generation | [Constraints and tokenizers](constrained-generation.md) | Grammar masks, exact token bytes, EOS and independently checked output |
| Inference interventions | [Prompt, logit and activation controls](inference-interventions.md) | Mechanism boundaries, reference probabilities and quality/cache controls |
| Uncertainty | [Probability, entropy and calibration](uncertainty.md) | Observable token statistics versus independently calibrated answer correctness |
| Probabilistic control | [GenLM and LLaMPPL](genlm-control.md) | Target versus proposal, weighted grammars and finite-particle guarantees |
| Feynman–Kac methods | [Correctors and particle speculation](feynman-kac-steering.md) | Diffusion versus autoregressive state and approximation costs |
| Power sampling | [Power-SMC](power-smc.md) | Sequence-level targets, importance weights, EOS and cache ancestry |
| Quantization | [Quantization and precision](quantization-precision.md) | Exact format decoding versus quality-changing conversion |

Version lookup: consult the Metal chapters' macOS and GPU-family gates before
using an API. Metal 4 availability does not establish MLX backend support;
legacy SIMD-group matrix operations must not be assigned the newer Metal 4
tensor feature's availability floor. CUDA paper results are hypotheses for a
Mac experiment, not portable performance guarantees.

The [complete uncached Qwen stream](../experiments/loader-qualification.md#complete-synchronous-streamed-forward)
now composes bounded embedding, layer and projection reads against the resident
oracle. [Candidate-only process measurements](../experiments/loader-qualification.md#candidate-only-process-footprint)
now isolate that execution and directly compare its emitted logits with CPU.
Teacher-forced cached streaming is also qualified against resident controls;
[cached candidate-only measurements](../experiments/loader-qualification.md#candidate-only-cached-process-footprint)
now isolate its process footprint and directly compare each step with CPU.
The [streamed generation path](../experiments/streamed-generation.md) now reuses
greedy sampling and grammar constraints with explicit weight/KV budgets.
Long-running allocator behavior and larger contexts remain separate gates.
The [V4.1 sparse-attention reference](../experiments/v41-attention.md) supplies
a small CPU semantic oracle, not native BF16 or Metal-kernel parity.
DeepSeek-V4.1 still needs its own text-forward numerical fixture;
Qwen correctness does not establish DeepSeek support. Training techniques
remain reference material, not an implemented training subsystem.

## What each new finding records

Keep the detail in the relevant topic/experiment note, and link it here when it
adds a new source or implementation relationship. Do not duplicate whole notes.

```text
Finding: one falsifiable statement
Source: primary URL, revision/version, section or symbol; hash when captured
Checked: date; full body/appendices, named sections, code-only, or abstract-only
Evidence: source claim / inspected implementation / local reproduction / inference
Application: adopted, experimental, deferred, or rejected; reason and limitations
Code: repository path and symbol; commit for a historical result
Validation: command, input identity, environment, result, raw artifact location
Next gate: what would establish or disprove the proposed benefit
```

Use commit permalinks for code, versioned paper links, and hashes for downloaded
inputs. Record mutable documentation as mutable, with the date checked; never
invent a historical revision. Separate a performance result's source commit
from its executable hash. A matching checkout does not authenticate a binary.

When a finding changes, explain the correction beside the original result or
link its replacement. Keep negative experiments and their revisit conditions.
Credit adapted code in [third-party notices](../../THIRD_PARTY_NOTICES.md);
a citation does not replace license obligations. Raw local receipts may remain
under ignored `artifacts/`, but public notes must include the method and limits.
Do not publish private paths, prompts, credentials, or checkpoint payloads.

## V4.1 source identity

The report, configuration, and executable reference use Hub revision
`dba1be0a40aa45a94ad051997016db3960a90277`:

| Artifact | SHA-256 |
|---|---|
| [Technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf) | `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d` |
| [Inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py) | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| [Configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/config.json) | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |

Report coverage: technical body §§1–6 and appendices A–C; referenced papers
are not included in that coverage. Source-code qualification executes only
selected hash-checked expressions, not the whole model. On 2026-09-11, the
older local config capture matched a fresh pinned download byte-for-byte.
This establishes its identity, not successful inference.

## From source to implementation

| Finding | Local chain | Evidence boundary |
|---|---|---|
| Candidate selection covers whole blocks; causal masking is still needed. | `select_candidate_blocks` → [capture script](../../scripts/v41-candidate-reference.py) → [fixture](../../fixtures/deepseek-v41/candidate-block-reference.json) → [`candidate_mask`](../../crates/models/deepseek/src/csa2.rs) | Exact synthetic CPU masks; ambiguous cutoff ties rejected. |
| Index scores rectify each head before signed weighting and reduction. | `Indexer.forward` → [score fixture](../../fixtures/deepseek-v41/index-score-reference.json) → [`index_scores_f32`](../../crates/models/deepseek/src/indexer.rs) | Metal FP32 diagnostic; not projections, FP4/BF16, or attention. |
| Final selection sorts selected positions and maps future indices to `-1`. | `Indexer.forward` → [selection fixture](../../fixtures/deepseek-v41/selection-reference.json) → [`select_indices`](../../crates/models/deepseek/src/selection.rs) | CPU fixture parity plus independent exhaustive small-row oracle. |
| Partitioning avoids a full score sort. | [Benchmark](../../crates/models/deepseek/benches/selection.rs) → [before/after ledger](../experiments/v41-candidates.md#cpu-final-selection) | Local optimization at `8b88312`; synthetic operator timings, not model throughput. |
| Removing per-layer waits improved the Qwen diagnostic. | [`forward.rs`](../../crates/models/qwen/src/forward.rs) → [measurement and parity ledger](../experiments/qwen-metal.md#wait-removal-result) | Local profile-derived change at `0abe5bf`, not a paper speedup. |
| Read size affects observed file-read cost. | [I/O probe](../../scripts/benchmark-checkpoint-io.py) → [initial curve](host-memory.md#initial-local-read-size-probe) | Actual reads; no proof of physical SSD traffic or model latency. |
| Selected BF16 reads can match the resident loader without reading every payload through the candidate path. | [`read_tensor`](../../crates/models/qwen/src/checkpoint/read.rs) → [`qualify_tensor_range`](../../crates/models/qwen/src/metal.rs) → [real-checkpoint results](../experiments/loader-qualification.md) | Selected raw-byte budget only; the independent reference still loads the whole checkpoint. |
| One Qwen block can execute using only its selected tensors. | Private `forward_layer` → [`qualify_layer`](../../crates/models/qwen/src/metal/layer_check.rs) → [comparison and process measurements](../experiments/loader-qualification.md#one-block-selected-weight-execution) | Bit-exact resident-loader comparison on synthetic hidden states; independent full-logit regression gate. Not a streamed decoder or process-memory ceiling. |
| Filling a fixed-size FP32 destination reduced selected-layer load time. | Main-thread profile → [`decode_bf16`](../../crates/models/qwen/src/metal.rs) → [kept/rejected measurements](../experiments/loader-qualification.md#profile-and-measured-change) | Exhaustive BF16 bit tests and real-checkpoint parity; about 49% lower warm layer-load time, not model throughput. |
| CSA2 consumers need the latest shared publisher, not any earlier matching publisher. | Pinned `Attention._compress_kv` / `_compress_topk_idxs` → [`V41Csa2Schedule`](../../crates/models/deepseek/src/lib.rs) → [red/green findings](../experiments/loader-qualification.md#v41-initial-dimensions-and-live-cache-sources) | Configuration relationship check, not V4.1 numerical parity. |

The Qwen oracle is generated by [qwen-reference.py](../../scripts/qwen-reference.py)
and checked by [qwen-parity-suite.py](../../scripts/qwen-parity-suite.py).
The [checked-in fixture](../../fixtures/qwen3-0.6b/forward-reference.json) records
library versions, checkpoint/config hashes and selected observations; it is
not the full-vocabulary oracle payload. The runtime suite compares all logits.

## Checkpoint header key uniqueness

Source: [safetensors format notes](https://github.com/safetensors/safetensors/blob/62e4d8b86063a6e5f8967547fa09a09f67420e2a/README.md#format),
format section reviewed 2026-09-11. Duplicate keys are disallowed.

Local reproduction: at `dd93bdc` with only the new raw-JSON test added,
`cargo test -p qwen rejects_duplicate_top_level_header_keys -- --nocapture`
exited 101: the reader returned one `weight` entry instead of rejecting the
duplicate. [`UniqueHeader`](../../crates/models/qwen/src/checkpoint.rs)
now rejects repeated tensor/metadata names, including differently escaped
spellings of the same name. `cargo test -p qwen` exited 0 afterward.
The fix is commit `68c1b85`; full default/Metal check logs are retained locally
as `artifacts/check-header-uniqueness-{default,metal}.log`.
This is a parser fix, not tensor-range loading; it does not establish uniqueness
inside nested JSON objects. The later completion pass below closes that gap.

### Nested header uniqueness

The same pinned safetensors format requirement also applies within tensor and
metadata objects. At `6a60644` with the new raw-JSON regression added,
`cargo test -p qwen rejects_duplicate_nested -- --nocapture` exited 101:
`dtype: F32` followed by `dtype: BF16` was accepted as a single BF16 declaration.
`UniqueJsonValue` now routes every object through the uniqueness check before
constructing a `serde_json::Value`. The same regression then passed, covering
dtype, shape, offsets, escaped metadata keys, and nested objects in arrays.

Positive tests compare valid JSON value types with the ordinary parser and
confirm the JSON recursion limit remains enforced. This closes header
ambiguity, not concurrent-file integrity or streamed execution. Validation
receipts are `artifacts/check-nested-header-{default,metal}.log`.
The real 4 MiB Qwen projection was checked again after rebuilding; all
2,097,152 values remained bit-exact against MLX. Receipt:
`artifacts/qwen-nested-header-qproj.json`.

## Research that has not become runtime support

[Host-memory notes](host-memory.md) distinguish full readings of LLM in a Flash
v3 and Fiddler v3 from engineering posts (HiCache and HiSparse), mutable MLX
API documentation, and the unmerged tinygrad expert-selection proposal.
Their methods are candidates; their benchmarks have not been reproduced here.

The older [runtime-pattern shortlist](reusable-runtime-patterns.md) links
`master` and `stable`, and the shared-serving paper links in
[efficiency requirements](efficiency-methods.md#shared-serving-mechanics) omit
paper revisions. Their historical source snapshots were not retained. Pin and
recheck the specific implementation before importing behavior or code; do not
treat these links as evidence of a current upstream capability inventory.
