# Verified schedule candidates

The resident Qwen diagnostic can generate bounded JSON candidates and accept a
schedule only after independent schema validation and a fixed semantic check.
The verifier requires a nonempty `intervals` array of at most 128 objects with
finite numeric `start` and `end`, `start < end`, and no overlaps. Intervals are
half-open: `[0, 1)` and `[1, 2)` may touch.

```sh
mx gen --model /path/to/Qwen3-checkpoint \
  --prompt 'Return a schedule as JSON.' \
  --json-schema /path/to/schedule-schema.json \
  --sample --temperature 1 --seed 41 \
  --verify-schedule --max-attempts 4 --max-tokens 128 \
  --max-candidate-ms 60000 --preview
```

Build with the `metal` and `structured-output` features. The command retains
prompt KV and forks a child for each attempt. It compiles the grammar once per
request, restoring its root checkpoint after each rejected or incomplete
attempt; output and sampling state do not carry across attempts. Each attempt
has its own entropy stream. Rejected children are dropped. After schema and
semantic acceptance, the terminal grammar token is materialized into the child
KV before that child becomes the request state. The command then ends; this is
not a persistent conversation API.

Attempt seeds are derived deterministically from the request seed and attempt
index. Repeating the same configuration is intended to reproduce the attempt
streams; cross-device numerical equivalence is not promised. Sampling retries
are not a confidence estimate, and finite attempt limits can exhaust.

`--max-tokens` bounds each attempt; `--max-attempts` defaults to four and is capped
at sixteen. Their product bounds total generated tokens. The cooperative time
limit covers model loading, prefill, grammar setup, attempts, and verification;
CLI parsing and prompt tokenization precede that timer. In-flight device work
cannot be interrupted, so wall time may exceed the limit. An expired request
cannot subsequently promote a candidate after observing expiration.

The JSON receipt records attempt identities/statuses, rejected-work costs,
configuration provenance, and accepted output. Exhaustion returns nonzero and
omits rejected candidate text, token IDs, and token probabilities from result
fields. Logical KV size is not process peak memory. The model directory and
configuration fingerprint do not authenticate checkpoint weight contents.

The workflow currently rejects streamed mode and combination with
`--verify-cache`. The latter is a separate parity diagnostic: it reports exact
comparison count and absolute input-token positions, and reports
`not_run_no_eligible_decode` when generation performs no candidate decode.
Forking a candidate alone is not evidence of parent/child numerical parity.

## Exact task requirements

`--schedule-requirements requirements.json` adds a typed task contract to
`--verify-schedule`. The file is at most 64 KiB, rejects unknown fields, and
contains only:

```json
{"durations":[2,2],"window":{"start":0,"end":8}}
```

Durations are positive integer ticks, one through 128 entries. Window bounds
and durations must stay within the exact integer range ±2^53; the window has
positive width, and total duration cannot exceed it. Candidate endpoints must
also be integer ticks in that range. The verifier requires the exact duration
multiset, hence interval count, and all intervals inside the window, as well
as the existing non-overlap check. There is no epsilon or free-text extraction.

The requirements file is read once before model loading. Receipts record its
exact-byte SHA-256 and parsed contract. Failed candidates expose stable reasons
such as `interval_count_mismatch`, `duration_multiset_mismatch`, and
`interval_outside_window`; they are discarded before another attempt. These
requirements do not constrain unlisted properties or guarantee generation will
find a feasible candidate within the budget.

The frozen evaluator accepts `--task-aware` to derive requirement files from
its existing task fields. This changes the verifier contract, not the prompts,
seeds, or schema, and is recorded separately from the frozen task hash.

## Qualification

`scripts/qualify-candidates.py` prints a dry-run plan by default. With
`--execute`, it invokes the actual CLI using an existing local checkpoint,
independently checks touching-interval acceptance and nested-overlap rejection,
and checks zero/one branch-comparison receipts. It downloads no model.

```sh
python3 scripts/qualify-candidates.py --binary target/debug/mx \
  --model /path/to/Qwen3-checkpoint --execute
```

These synthetic tests qualify mechanism wiring, not held-out schedule quality,
a latency improvement, or general semantic verification. A local Qwen3-0.6B
mixed-schema check rejected three candidates and reached the token limit on two
before accepting the sixth; replay preserved accepted IDs, attempt seeds,
statuses, and token counts. A one-millisecond cooperative budget exhausted
without publishing accepted output. Matched unconstrained/verified task
quality measurements remain a separate gate. A repeated local cost pilot and
its peak-memory limitations are recorded in the
[performance ledger](experiments/chat-performance.md#candidate-grammar-reuse-pilot).

## Fixed synthetic task-quality check

`scripts/evaluate-candidates.py` freezes four schedule requests and two seeds
before execution, then compares a schema-only baseline against the same
prompt/schema/token limit with at most four verified attempts. The independent
oracle checks both non-overlap and task adherence: requested interval count,
durations, and window. The CLI uses plain completion prompts, not chat
messages. Candidate attempt entropy is derived from each base seed; the two
arms do not consume an identical first sampled stream.

```sh
uv run scripts/evaluate-candidates.py --binary target/debug/mx --model /path/to/Qwen3 --execute --output /path/to/new-receipts
```

Local results at 128 tokens/attempt and temperature 1, with no task or prompt
changes between checkpoints:

| Checkpoint / arm | Valid non-overlap | Full task adherence | Generated tokens |
| --- | --- | --- | --- |
| Qwen3-0.6B baseline | 1/8 | 0/8 | 572 |
| Qwen3-0.6B verified | 3/8 | 0/8 | 2966 |
| Qwen3-4B-Instruct-2507 baseline | 6/8 | 3/8 | 445 |
| Qwen3-4B-Instruct-2507 verified | 7/8 | 0/8 | 1172 |

All 32 runs returned valid evaluation receipts without infrastructure errors.
Verified acceptance means only that the fixed non-overlap predicate passed;
it does not establish adherence to free-text counts, durations, or bounds.
This workload did not demonstrate a task-quality benefit. The retry arm has a
larger generation budget, and the small synthetic run does not establish a
speedup, general model quality, or a causal effect of retries alone.

The frozen task/schema/seed/budget hash is
`87f4e94e312389756108e582095caa1a64c87711989052a2df680c46d77745e8`.
Receipts preserve failure denominators, measured-cost counts, all commands,
and per-case output. Missing receipt costs are errors rather than zero work.

The explicit-requirements follow-up uses `--task-aware` with the same frozen
tasks, prompts, seeds and budgets. On Qwen3-4B-Instruct-2507, all eight candidate
runs exhausted: none published a schedule that violated the task contract,
but none completed the task. They consumed 1,687 generated tokens. The repeated
baseline remained 3/8 task-adherent with 445 tokens; all 16 runs had valid
evaluation receipts and no infrastructure errors. This establishes the stricter
acceptance boundary, not a task-completion improvement. Four separate forced
schema smoke cases accepted an exact valid schedule and rejected wrong count,
duration and window, confirming that acceptance is reachable.

All 32 strict attempts completed the grammar within their token budget;
rejections were overlap (17), nonpositive width (6), duration mismatch (6),
and count mismatch (3). No run exhausted its elapsed-time budget. A
schema-only replay using one candidate's derived seed reproduced three
unit-length intervals for a task requiring lengths 1, 2 and 3; the paired
one-attempt verifier rejected that same 41-token draw for duration mismatch.
This supports investigating proposal quality and prompt serialization before
increasing the token budget or changing retry lifecycle behavior.

A serialization-only follow-up rendered the same task text as one user message
with the pinned 4B checkpoint's chat template and assistant-generation prefix.
The schema, seeds, temperature, requirements and attempt budgets stayed fixed.
Schema-only generation reached 6/8 full-task successes (459 generated tokens);
strict verified retries reached 7/8 (757 tokens), with one exhausted run. All
16 runs produced valid evaluation receipts without infrastructure errors.
The renderer used Jinja2 3.1.6 and independently checked the inspected
single-user template branch. Template/config and binary hashes were retained.
This diagnoses a prompt-format sensitivity on already-observed synthetic
tasks; it does not establish held-out improvement, calibration, or a speedup.
The native `mx gen --chat-template` option now renders that single-user path
before weights load and records the selected template SHA in `input_format`.
Native baseline and candidate smoke runs preserved the experiment's input IDs,
generated IDs and parsed schedules. To reproduce the template-conditioned plan:

```sh
uv run scripts/evaluate-candidates.py --binary target/debug/mx --model /path/to/Qwen3 --chat-template --task-aware --execute --output /path/to/new-template-receipts
```

A fresh `confirmation-v1` suite freezes four different prompts and seeds 101
and 211 before execution. Task hash:
`513ecc4fcf05570ae5063e23042750e692790ad8cf9fcc9a580aea261ab27a0c`.
It retains the same schema, checkpoint, chat formatting, temperature and budgets.
On the pinned 4B checkpoint, schema-only generation met all task requirements
in 2/8 runs (478 generated tokens); verified retries accepted 3/8 (1,606 tokens)
and exhausted five. All 16 runs produced valid receipts without infrastructure
errors, and every accepted schedule met the task requirements. This smaller
result limits the earlier development-set observation: acceptance is reliable
on these cases, while completion remains weak and retries cost more work.
This is same-family synthetic confirmation, not an official benchmark or
an independently representative task sample. The suite must not be retuned
and then described as fresh confirmation.

```sh
uv run scripts/evaluate-candidates.py --binary target/debug/mx --model /path/to/Qwen3 --suite confirmation-v1 --chat-template --task-aware --execute --output /path/to/new-confirmation-receipts
```


```sh
uv run scripts/evaluate-candidates.py --binary target/debug/mx --model /path/to/Qwen3 --task-aware --execute --output /path/to/new-task-aware-receipts
```

## Adjacent research boundaries

Qwen's experimental residual-steering API validates a declared artifact against
its bound configuration and applies it at an absolute token-position range.
Changing it requires empty KV; forks retain the intervention. Caller-provided
model identity remains an assertion. No behavioral steering benefit is claimed
without a calibrated artifact and held-out regression measurements.
The ignored local 0.6B checkpoint test verifies exact zero-coefficient identity,
an observable finite nonzero effect, fork inheritance and parent-cache
immutability. Full/chunked prefill differences pass the existing `5e-5`
absolute logit bound: the unsteered control measured `1.38e-5`, and the
synthetically steered path `2.05e-5`. Same-shape fork/chunk replay is exact.

The SMC state primitive now distinguishes terminal and impossible particles,
rejects weight overflow atomically, and returns stage log-mean weight separately
from population log-sum weight. It is still not a model-backed particle runtime.
A same-capture DeepSeek test now publishes native layer-three index keys into
request-local storage and consumes them in partial layer-one scoring. Its
layer-three block input and attention boundaries remain captured.
