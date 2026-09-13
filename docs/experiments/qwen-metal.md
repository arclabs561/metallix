# Qwen3 Metal qualification ledger

This is a correctness and single-sequence decode experiment for
`Qwen/Qwen3-0.6B`, not an HTTP-serving, concurrency, or V4.1 result. The
baseline implementation is commit `66e3013`.

## Scope

- Host: one M3 Max with 128 GiB unified memory.
- Runtime: `mlx-rs 0.25.3` on Metal.
- Weights: safetensors checkpoint materialized as float32 for the diagnostic.
- Inputs: raw token IDs only for this numerical experiment. The CLI separately
  accepts a local-tokenizer plain-text prompt, but that user interface is not
  part of these parity observations.
- Cache: one sequence with contiguous K/V arrays. It is not paged KV, prefix
  caching, or continuous batching.

`logical_weight_bytes` and `logical_kv_bytes` count live logical arrays only.
They exclude allocator overhead, temporary concatenation buffers, peak memory,
and concurrent requests.

## Plain-text CLI boundary

`mx gen --prompt` encodes exactly the supplied text with the checkpoint-local
`tokenizer.json`, then includes decoded `generated_text` in its JSON receipt.
It adds neither a chat template nor special tokens. Therefore it is a
single-sequence text-completion interface, not a chat, HTTP-serving, or
multi-request result. Its cached-decode `--verify-cache` mode compares each
step with a full uncached Metal forward; that checks cache consistency and is
not a CPU-reference parity run.

For a local operator check, use a materialized Qwen directory:

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --prompt "The capital of France is" --max-tokens 8 --verify-cache --preview
```

The primary Qwen execution path uses `mlx-rs` and MLX on Metal. The JSON
receipt, rather than terminal preview text, is the machine-readable result.

## CLI smoke ledger

`artifacts/qwen-readme-fresh.json` records the prompt command above: greedy
completion ended by its length budget with decoded text ` Paris. The capital
of Italy is Rome`; all eight requested cache comparisons passed. Its companion
stderr preview reports `cache_checks: 8/8 passed`.

`artifacts/qwen-readme-schema-fresh.json` records the inline enum-schema
command: it ended `grammar_complete`, decoded `"waiting"`, and its constraint
status is `validated`. These are bounded operator checks for the shown
inputs—not quality, throughput, latency, or general structured-output claims.

## Numerical evidence

The checked-in reference fixtures record the Qwen3-0.6B config/checkpoint
hashes, float32 CPU eager reference provenance, and fixed raw-ID observations.
The full-parity harness uses four cases of 1, 3, 17, and 64 tokens, including
the final vocabulary row. For each case it compares all 151,936 final-token
logits against an independently executed CPU float32 eager reference.

The comparator requires finite values, exact vocabulary length, absolute
tolerance `0.0005`, and relative tolerance `0.0001`. The reference artifact is
also bound to the input IDs, model config, `model.safetensors`, and logit bytes
by SHA-256 before comparison.

After the wait-removal and last-row projection changes, all four cases passed
with zero mismatches. The largest observed absolute error was
`4.208087921142578e-05`; the largest observed RMSE was
`8.799367978007218e-06`.

Cached decoding has a separate invariant: with `--verify-cache`, every
cached step is recomputed through the complete uncached Metal forward outside
the timed region and must agree under the same full-logit comparator. This is a
cache-consistency check, not an independent CPU check at every generated step.

Reproduce the CPU/Metal full-logit suite after building the release binary:

```sh
cargo build -p server --release --features metal
uv run scripts/qwen-parity-suite.py \
  --binary target/release/metallix \
  --model /path/to/Qwen3-0.6B
```

## Decode measurements

For new measurements, build the release executable and use the local harness:

```sh
cargo build -p server --release --features metal
mkdir -p artifacts
uv run scripts/benchmark-qwen.py --model /path/to/Qwen3-0.6B \
  --output artifacts/qwen-benchmark.json
```

The output must be a new file: the harness refuses to overwrite prior evidence. Its defaults are
three runs, 32 generated tokens, and one discarded decode observation per run.
It records all timings, per-run medians, pooled statistics, hardware metadata,
and binary/config/weight SHA-256 hashes. The checkout revision is an observation,
not proof of which source built an arbitrary executable.

Hashing warms the filesystem cache. Load and first-prefill times are retained
separately, not folded into warm decode. A 300-second total deadline covers the
run; the receipt records running, completed, failed, or interrupted state.
Early EOS, differing generated IDs, enabled cache verification, invalid timings,
or changed checkpoint/executable bytes fail the measurement. Run the independent
parity suite separately before comparing performance.

## Release resident baseline

`artifacts/index-query-qwen-baseline.json` records a completed release-binary
resident run on an Apple M3 Max with 128 GiB unified memory. It used the MLX
float32 path, raw IDs `785,6722,315,9625,374`, 32 generated tokens, and three
runs. Each run started a fresh process and contiguous KV cache; the first
decode observation was discarded, leaving 90 retained warm-decode observations.
Load, prefill, and initial-decode time are excluded. The run did not request
`--verify-cache`, so it is neither a cache-consistency check nor an independent
correctness oracle.

| Metric | Recorded value |
| --- | ---: |
| Median warm decode | 9.169375 ms |
| Mean warm decode | 9.328383844 ms |
| Sample standard deviation | 0.47073026 ms |
| Per-run medians | 9.1741455 / 9.199625 / 9.1189375 ms |
| Retained observations | 90 |

The recorded binary, config, and weight SHA-256 values are respectively
`61e9da601f21c2ce6b22d8e7f2e87b789a30b500019c525f3f7201d12a296c4d`,
`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`, and
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
The receipt also records a dirty checkout revision, which is not binary-build
provenance.

The recorded workload can be reproduced with explicit flags:

```sh
uv run scripts/benchmark-qwen.py \
  --binary target/release/mx --model /path/to/Qwen3-0.6B \
  --output artifacts/index-query-qwen-baseline.json \
  --input-ids 785,6722,315,9625,374 --max-tokens 32 \
  --memory-mode resident --runs 3 --discard-decode 1
```

This is one single-sequence resident decode baseline, not an improvement claim,
throughput or serving result, V4.1 execution result, or DeepSeek measurement.

## Earlier decode baselines

The first harness run on the same M3 Max measured 90 retained observations:
median 8.812625 ms, mean 8.8707791 ms, sample standard deviation 0.2694133 ms.
This reproduces the optimized single-sequence range, not a new optimization.

The release baseline used greedy generation from raw IDs
`785,6722,315,9625,374`, generated 32 tokens, and ran three times. Each run
records 31 decode calls because the first generated token comes from prefill.
The first decode observation of each run was excluded, leaving 90 warm-decode
samples.

| Metric | `66e3013` baseline |
| --- | ---: |
| Median warm decode | 20.2876875 ms |
| Mean warm decode | 20.7975254 ms |
| Sample standard deviation | 1.5986179 ms |
| Samples | 90 |

The recorded command shape is:

```sh
cargo run -p server --release --features metal -- generate-qwen-metal \
  --model /path/to/Qwen3-0.6B \
  --input-ids 785,6722,315,9625,374 \
  --max-tokens 32
```

This is not a throughput, TTFT, long-context, or peak-memory benchmark. The
prefill is deliberately excluded from the warm-decode statistic; the first
prefill in a process is not warmed.

## Wait-removal result

Weighted samples in the baseline profile attributed 85.4% to GPU
`Event::wait`. Commit `0abe5bf` removed 56 layer-level waits. Rerunning the
same three-by-32-token protocol and exclusion rule produced 90 samples:

| Metric | Wait-removal result |
| --- | ---: |
| Median warm decode | 8.736521 ms |
| Mean warm decode | 8.8053898889 ms |
| Sample standard deviation | 0.2346757225 ms |
| Samples | 90 |

The three runs generated identical IDs, also identical to the baseline runs.
The associated 128 generated-token cache verification had 128 passing
full-vocabulary comparisons; when coalesced, its worst absolute error was
`9.822845458984375e-05`.

The final CPU parity suite above was run after both this change and the later
last-row projection change. A final 128-token cache verification after both
changes also passed all 128 comparisons, with worst absolute error
`9.632110595703125e-05`.

## Last-row projection experiment

Commit `b9ac0e5` projects only the final hidden row into
the vocabulary. It removes unnecessary projection rows for this last-token
diagnostic. It did not receive a robust speedup claim: each side contains only
20 uncached-forward samples per prompt length, and the observed median changes
are small.

| Input length | Before (ms) | After (ms) |
| --- | ---: | ---: |
| 1 | 8.5906245 | 8.474771 |
| 3 | 9.9868745 | 9.633437 |
| 64 | 23.2746875 | 22.770833 |

No result here supports HTTP serving, V4.1 execution, paged KV, prefix reuse,
continuous batching, quantization conversion, or a default-model decision.
