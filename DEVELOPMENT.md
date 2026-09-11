# Developing Metallix

DeepSeek-V4.1-Flash is the primary target. Qwen3-0.6B is the running control for
loader, numerical, and single-sequence decode experiments. A new model earns
an adapter through a verified checkpoint contract and numerical reference, not
through a model-name entry alone.

## Keep findings traceable

Use the [research index](docs/research/README.md) to connect sources, code,
fixtures, and measurements. Record a new optimization's source and limitations
in its topic note when implementing it, then link the result from the experiment
ledger. Distinguish copied/adapted code, conceptual inspiration, and changes
derived from our own profiles. Keep failed experiments and rejected approaches
with the evidence that would justify revisiting them.

## Check first

Run from the repository root. Install Rust, uv, Node.js, and Ruff; Metal builds
also require Apple Silicon, CMake, and the Xcode Metal toolchain.

```sh
uv run scripts/check.py
uv run scripts/check.py --metal
```

The runner stops at the first failure. Each command has a timeout and owned
child processes are terminated if it expires. It runs no model downloads.
Do not run competing Cargo builds or benchmarks in this checkout concurrently.

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
