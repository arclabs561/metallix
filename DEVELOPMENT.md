# Developing Metallix

DeepSeek-V4.1-Flash is the primary target. Qwen3-0.6B is the running control for
loader, numerical, and single-sequence decode experiments. A new model earns
an adapter through a verified checkpoint contract and numerical reference, not
through a model-name entry alone.

The product priorities are larger-than-RAM model execution on one Mac, timely
support for new architectures, structured/constrained generation and fast
serving. Quantization and local adaptation are future workflows, not reasons
to turn the serving engine into a training framework now. Each diagnostic
should retire a gate toward those outcomes, not become the product itself.
Track fit and speed separately: SSD-backed execution can enable a model without
meeting a useful latency target. Constrained generation needs sampler-level
token validity and completion/error semantics, not post-hoc JSON repair.

Metallix should also make new inference methods easy to try as they appear.
An experimental path needs a runnable command, pinned source provenance, a
correctness or quality oracle, explicit resource bounds and comparable baseline
measurements. Experimental availability and reliable serving support are
different statuses. Keep negative results; promote a fast path only for the
shapes and conditions its evidence covers. Research may propose new experiments,
but it must not indefinitely displace executable DeepSeek-V4.1-Flash progress.

## Keep model assumptions local

The first two adapters are test cases, not a universal model architecture.
Keep layer ordering, attention/cache representation, expert routing, embedding
tying, positional encodings and packed weight layouts in their model adapters.
The synchronous Qwen stream is a diagnostic implementation, not a required
execution schedule for future models.

Before promoting an adapter helper into the engine, demonstrate the shared
contract with materially different execution/state needs. Stress-test proposed
interfaces against recurrent or hybrid state, sliding or compressed attention,
sparse expert loading, untied output heads and multimodal encoder state.
These are design probes, not claims of implemented model support. Do not add
empty traits or a capability flag for each hypothetical architecture.

The engine's logical KV-page pool is useful for paged KV reservations; it is
not a complete model-memory estimator or a mandatory state representation.
Future admission must account for the adapter's actual state, scratch and
weight-residency requirements. Prefix reuse and speculative rollback need
explicit state-compatibility rules, not an assumption that every state is a
token-indexed KV tensor.

Choose fast paths by validated layout, precision, workload shape and backend
capability, with a numerical fallback/reference and a measured crossover.
Do not select them solely by model name or force all models through the same
fusion, batching or offload strategy. Extract shared mechanisms when real
adapters demonstrate reuse; retain specialized kernels where they earn it.

Use Rust types to enforce these boundaries: validated layouts rather than raw
configuration fields, distinct units for bytes/tokens/pages, owned reservation
handles and explicit execution/verification outcomes. Construct validated values
through fallible APIs and keep their invariants private. A candidate run is not
a verified comparison; absent evidence must not become a zero-error result.
Use enums for genuinely exclusive states and newtypes where mixing values is
meaningfully wrong. Avoid boolean capability matrices, unchecked casts and
speculative trait hierarchies; add type-state transitions where they prevent
an actual invalid operation, not merely to encode the current implementation.

## Keep findings traceable

Use the [research index](docs/research/README.md) to connect sources, code,
fixtures, and measurements. Record a new optimization's source and limitations
in its topic note when implementing it, then link the result from the experiment
ledger. Distinguish copied/adapted code, conceptual inspiration, and changes
derived from our own profiles. Keep failed experiments and rejected approaches
with the evidence that would justify revisiting them.

Original methods are welcome, not just reproductions. For each proposed method,
record: the source or observation that inspired it; what is newly proposed;
the expected mechanism; baseline and falsifying test; exact reproduction command
and artifact identities; correctness/quality, memory and timing results; and
the decision to retain, revise or reject it. Label untested ideas as hypotheses.
Do not call an idea a breakthrough or claim novelty beyond the prior art that
was actually checked. Keep this record in the existing topic note and experiment
ledger rather than creating a second disconnected tracking system.

## Check first

Run from the repository root. Install Rust 1.87 or newer, uv, Node.js, and Ruff; Metal builds
also require Apple Silicon, CMake, and the Xcode Metal toolchain.

```sh
uv run scripts/check.py
uv run scripts/check.py --metal
```

The runner stops at the first failure. Each command has a timeout and owned
child processes are terminated if it expires. It runs no model downloads.
Do not run competing Cargo builds or benchmarks in this checkout concurrently.

`just check` and `just check-metal` call the same runner. For the focused
Engram integrity gate, use `just check-fixtures`; see the
[check recipes and custom-lint policy](docs/development.md).

## Measure one change

For a CPU selection baseline:

```sh
cargo bench -p deepseek --bench selection
```

For repeated Qwen decode, use a new output filename on every run:

```sh
cargo build -p server --release --features metal
uv run scripts/benchmark-qwen.py --model /path/to/Qwen3-0.6B \
  --output artifacts/qwen-before.json
```

After the change, repeat with `artifacts/qwen-after.json`, then compare:

```sh
uv run scripts/compare-qwen-benchmarks.py \
  --baseline artifacts/qwen-before.json --candidate artifacts/qwen-after.json
```

The comparator checks that the measurements describe the same model, hardware,
workload, backend, and generated IDs. A comparison is not a correctness oracle
or a statistical significance test. Keep the independent numerical parity gate.

For the larger-than-memory investigation:

```sh
uv run scripts/benchmark-checkpoint-io.py \
  --file /path/to/Qwen3-0.6B/model.safetensors \
  --output artifacts/checkpoint-io.json --uncached
```

This measures read sizes, not inference. `--uncached` requests a macOS cache
hint; it does not prove that every byte came from physical SSD. Omitting it
leaves OS cache state uncontrolled. No global cache flush or memory-limit change
is performed. See [host-memory constraints](docs/research/host-memory.md).

## Profile before optimizing

For the first selected-tensor loader gate and V4.1 shape checks, follow
[loader qualification](docs/experiments/loader-qualification.md). The tensor
diagnostic's byte limit applies to its selected raw payload, not the resident
reference loader or the process's total memory.

Both release binaries expose the same commands: use `target/release/mx` or
`target/release/metallix`. `check-qwen-rows-metal` checks contiguous BF16 rows;
`check-qwen-embedding-metal` gathers raw token IDs in input order and reads
repeated IDs once. Its byte budget is the aggregate selected raw payload,
not the output array or resident oracle. See the
[selected embedding result](docs/experiments/loader-qualification.md#token-ordered-selected-embedding)
for the exact command and checkpoint identity.

For resident Qwen generation, `gen` is a short alias for `generate-qwen-metal`:

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 4 --verify-cache --debug
```

Without a schema this generates raw token IDs, not decoded text; it does not run V4.1 yet.
`--debug`, `--verbose` and `-v` are equivalent opt-in generation diagnostics.
They emit phase timings, token counts and logical memory sizes to stderr,
without adding model paths, token values or logits. Stdout remains the existing
JSON diagnostic report, which intentionally includes input/generated IDs.
Timing regions exclude the stderr writes, but logging can perturb a benchmark;
leave verbosity off for comparisons. Resident generation has a 512-token
diagnostic context cap (or the model's smaller configured limit), distinct from
the streamed path's 32-token limit. Prompt and EOS IDs are checked before loading weights.
The local four-output-token smoke passed cache verification with and without
verbosity; generated IDs and JSON field sets matched, and quiet stderr was
empty. Receipts: `artifacts/gen-{debug,quiet}.{json,stderr}`. This verifies CLI
behavior, not a generation-quality or speed improvement.

For seeded categorical sampling instead of greedy selection:

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 4 \
  --sample --temperature 0.7 --seed 42 --logprobs
```

All three sampling flags are required together. Temperature must be finite and
positive; omitting them preserves greedy behavior. Sampling supports resident
and streamed execution, including `--json-schema`. It does not apply top-k or
top-p truncation.

For reproducible schema-constrained sampling:

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 29 \
  --json-schema fixtures/constraints/record.json \
  --sample --temperature 0.7 --seed 42 --logprobs --preview
```

The grammar mask is applied before sampling. A rejected draw commits neither
the random stream nor the grammar. `constrained_logprob` and `allowed_log_mass`
describe the temperature-one grammar restriction; `sampling_logprob` describes
the actual temperature-conditioned masked policy. This local conditioning is
not sampling from the model conditioned on eventual whole-sequence validity.

The report records `sampling_policy`, including the RNG, seed, temperature and
uniform conversion. With `--logprobs`, `model_logprob` remains the selected
token's natural-log probability under the raw temperature-one model;
`sampling_logprob` is its natural-log probability under the actual
temperature-conditioned sampling policy.
Replay requires the same policy and model execution, not just the same seed:
different logits or floating-point execution can select different tokens.

Local qualification on an M3 Max (FP32 Qwen3-0.6B): the unconstrained four-token
command produced IDs `13,21927,11,1879`. A repeated run reproduced IDs and both log-probability
fields exactly; disabling scores preserved IDs. The streamed path with the
budgets below matched IDs and scores exactly on this prompt. Greedy selection
still produced `13,358,2776,264`, and the schema example remained independently
validated. These are bounded replay checks, not cross-device reproducibility
or throughput guarantees. Local receipts are in
`artifacts/sampling-{resident-42-*,streamed-42,greedy-control,schema-control}.json`.

The sampled-schema command above also passed locally: all four runs (resident,
repeat, scores disabled, streamed) independently validated
`{"status":"ready","count":1}` and selected the same 12 token IDs. The three
scored runs matched all four probability fields exactly. Redirected preview
contained the probability diagnostics without ANSI escapes; stdout remained
JSON. These are same-checkpoint correctness checks, not speedup measurements.
Receipts: `artifacts/sampled-schema-{resident-a,resident-b,unscored,streamed}.json`
and `artifacts/sampled-schema-preview.stderr`.

For bounded layer-streamed generation, use:

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 16 --memory-mode streamed \
  --max-weight-bytes 81798144 --max-kv-bytes 7340032 \
  --json-schema fixtures/constraints/record.json --logprobs --preview
```

Resident mode remains the default. Streamed mode supports the same grammar,
logprob and preview options, but reserves the full prompt-plus-output budget
before reading candidate weights, even if generation later stops early.
Its default K/V budget is 7,340,032 bytes and default output budget is four
tokens (resident: 32). `--tile-rows` defaults to 1024. Stream-specific budget/tile
flags are rejected in resident mode. These are logical weight/staging and
retained-KV budgets, not process-memory caps: scratch, activations, headers,
projection output and allocator retention are excluded. `--verify-cache`
additionally loads a resident full-prefix oracle and excludes that verification
from candidate timings; omit it when measuring candidate memory or speed.
See [streamed generation measurements](docs/experiments/streamed-generation.md).
In streamed mode, `--debug` / `--verbose` also adds `streamed.phase_profiles`
to the JSON report and an aggregate stderr summary. These host wall-clock
intervals separate layer loading/conversion, execution/readback and tiled
projection; they are not GPU-only timings. Quiet generation omits them.

For constrained JSON generation, build with `--all-features`, then add
`--json-schema fixtures/constraints/record.json`. This optional path masks real
model logits and independently validates completed JSON. `--max-tokens` still
bounds work: an incomplete grammar reports failure rather than a successful
partial object. Schemas are local-only and limited to 32 KiB; tokenizer JSON is
limited to 64 MiB and decoded output to 1 MiB. See the
[integration and measurements](docs/research/constrained-generation.md#cached-generation-integration).

For greedy generation, add `--logprobs` to request selected-token natural-log
probabilities in the JSON report. `model_logprob` uses the original model distribution; constrained runs
also report `constrained_logprob` after masking and `allowed_log_mass` for the
grammar-allowed set. These are temperature-one categorical probabilities, not
the probability of the deterministic greedy policy or of an entire valid
sequence. Scores are opt-in and never appear in debug logs. Their computation
adds work, so compare runs with matching flags.

`--preview` adds a human-readable summary on stderr without changing stdout
JSON. It shows decoded constrained output (or explicitly raw IDs for the
unconstrained diagnostic), timings and any requested logprobs. Colors are
terminal-only and describe token likelihood, not answer correctness. Generated
terminal-control characters are escaped. Unlike metadata-only `--debug`,
`--preview` explicitly opts into displaying generated content.

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 64 \
  --json-schema fixtures/constraints/record.json --logprobs --preview
```

`check-qwen-stream-metal` composes a complete uncached forward for at most 32
raw tokens. `--max-weight-bytes` limits the planned maximum sequential
weight/loading stage, not process memory; the resident oracle runs afterward
and is outside that budget. Add `--candidate-only` to exclude the oracle from
process-memory measurement. That mode emits all final-token logits and a
distinct `candidate_only` verification status, with no comparison fields.
Treat those logits as explicit diagnostic output, not general telemetry.
Run the normal comparison separately; candidate timing excludes report
serialization, while an external process measurement includes it. See the
[complete stream ledger](docs/experiments/loader-qualification.md#complete-synchronous-streamed-forward).

`check-qwen-stream-cache-metal` adds teacher-forced cached prefill and appends:

```sh
target/release/mx check-qwen-stream-cache-metal --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --decode-ids 4,5,6 \
  --max-weight-bytes 81798144 --max-kv-bytes 1376256
```

It compares every vocabulary logit at each step with both resident cached and
full-prefix controls. Prompt plus appends must fit 32 tokens. The weight budget
covers logical weights/loading staging; the separate KV budget covers retained
FP32 K/V only, not transient copies, scratch, or process memory. This is a
cache-correctness diagnostic, not generated text or a beyond-RAM serving claim.

Add `--candidate-only` to this cached command to omit both resident controls.
It emits per-step logits and `verification: "candidate_only"`, not parity
fields. This permits isolated process-memory measurement; retained output
vectors and serialization still contribute to process peak. Compare the
emitted vectors with separately captured CPU references and keep the default
resident qualification as a separate run. See the
[cached process measurements](docs/experiments/loader-qualification.md#candidate-only-cached-process-footprint).

`check-qwen-layer-metal` extends this to one real block on synthetic hidden
states. Its `--max-weight-bytes` bounds logical weights and loading staging,
not scratch or process memory. Use `--candidate-only` when measuring process
memory; it skips the resident comparison and explicitly reports no verification.
Run the normal comparison separately. The qualification ledger records both.
Use `--repeats N` (1–64) to measure repeated diagnostic lifetimes in one process.
The default is one cycle. Multiple cycles return a typed envelope containing
each report; they repeat the same layer/input shape, not sequential model layers.

Use an optimized executable and a sustained workload. With Divan, directly
invoking a benchmark executable needs `--bench`; otherwise it only runs tests.
Confirm the output includes timed samples. Use headless `samply record
--save-only` and keep the profile under `artifacts/`. Profile results locate
work; unprofiled before/after measurements establish performance.

For each experiment record the commit and executable identity, command,
input shape, cache state, memory budget, correctness result, per-run timings,
and keep/reject decision. A checkout commit does not authenticate an arbitrary
binary's build provenance. Retain raw receipts locally; publish concise results
and caveats in `docs/experiments/`.

## Instrument what is real

Current tools emit local JSON receipts and explicit completion/failure status.
There is no remote telemetry exporter or HTTP metrics endpoint yet. Future
runtime metrics must separate load, prefill, decode, queue wait, GPU wait,
weights, KV, scratch, resident cache, and requested read bytes. Physical SSD
traffic needs an independent measurement source.

Do not log prompts, credentials, tensor contents, or unbounded labels. Model
identity and request IDs are useful; prompt strings and arbitrary file paths
are not metric labels. Add counters only where their ownership and units are
defined, and test both success and cancellation/error paths.
