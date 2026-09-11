# Bounded Qwen streamed generation

The Qwen adapter can generate greedily from layer-at-a-time weight reads,
keeping detached live K/V between tokens. `mx gen --memory-mode streamed`
uses the same grammar, logprob and preview path as resident generation.
It does not load a resident oracle unless `--verify-cache` is requested.
This is a single-sequence, 32-total-token qualification path, not larger-than-RAM
serving or a full DeepSeek decoder.

## Contract

`Qwen3StreamExecutor` validates prompt IDs, configured context, the promised
total context, and separate weight/staging and final K/V budgets before
candidate payload reads. Every append verifies finite logits and exact logical
K/V size. Invalid token/context calls fail before mutation. An execution error
that might leave partial layer state clears and permanently poisons the cache.
The executor retains current logits, not a per-step logit history.

Budgets exclude activations, scratch, headers, projection outputs and allocator
retention. They are not process-memory limits. The CLI conservatively reserves
`prompt length + max_tokens`, even though the final sampled token may not need
another forward pass or the grammar may finish early. Defaults are four output
tokens, 81,798,144 weight/staging bytes, 7,340,032 K/V bytes and 1024 projection
rows per tile. Resident mode still defaults to 32 output tokens and has its
separate 512-token/configured context ceiling.

## Real-checkpoint qualification

The local Qwen3-0.6B safetensors SHA-256 was
`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`;
config SHA-256 was
`660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd`.

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 16 --memory-mode streamed \
  --max-weight-bytes 81798144 --max-kv-bytes 7340032 \
  --json-schema fixtures/constraints/record.json --logprobs --preview --verbose
```

The run completed the schema in 12 generated tokens with value
`{"status":"ready","count":1}`. It retained 14 cached tokens / 3,211,264
logical K/V bytes. `/usr/bin/time -l` observed 389,382,960 bytes peak footprint
and 266,977,280 bytes maximum RSS. This is one candidate-only constrained run,
not a repeated memory benchmark. Receipts: `artifacts/stream-gen-record.json`
and `artifacts/stream-gen-record.time`.

A separate run with `--verify-cache` compared all 151,936 vocabulary logits at
each of the 12 sampling steps to resident full-prefix Metal recomputation.
Every comparison passed `5e-4 + 1e-4 * abs(reference)`; maximum absolute error
was `2.09808349609375e-5`. That oracle shares the Metal implementation and is
not independent CPU evidence. Its memory is explicitly outside candidate
planning and its work outside candidate timing. Receipt:
`artifacts/stream-gen-record-parity.json`.

The ignored local-checkpoint test
`streamed_executor_repeated_lifetimes_keep_only_live_candidate_cache` also ran
two independent executor lifetimes in one process: 2→4 and 3→6 cached tokens,
ending at 917,504 and 1,376,256 logical K/V bytes. It checks repeat-prefill
rejection and invalid-token/context-limit non-mutation. The debug test took
85.16 seconds; this is test duration, not serving performance. Receipt:
`artifacts/stream-gen-lifetimes.log`. This test does not measure post-drop
allocator retention or establish stability under many requests.

## Initial speed baseline

```sh
uv run scripts/benchmark-qwen.py --model /path/to/Qwen3-0.6B \
  --binary target/release/mx --output artifacts/stream-gen-benchmark.json \
  --input-ids 9707,11,1879 --max-tokens 8 --memory-mode streamed \
  --max-weight-bytes 81798144 --max-kv-bytes 7340032 --tile-rows 1024 --runs 3
```

Three fresh processes generated identical IDs
`[13,358,2776,264,5458,315,279,3822]`. After discarding each run's first decode,
18 samples had median 363.652292 ms and sample standard deviation 8.576055 ms.
Per-run medians were 360.019792, 366.721063 and 361.381334 ms. That is roughly
2.75 tokens/s for this synchronous path; it is not a speedup claim. The receipt
binds measured binary SHA-256
`1e857c912bc4e7c83e2831dbe1d39611d8ff76be774f8cb5382a57c61293723b`;
later CLI-default/dispatch cleanup is not measured by this receipt.

A second three-process run on the final binary
`967ef964da45d60309cb0ea029ec40f25fdaaebc6314c4b0c470837445e31389`
produced the same IDs. Its 18 retained samples had median 347.839666 ms,
sample standard deviation 4.429296 ms and per-run medians 346.691687,
348.391271 and 348.421854 ms (about 2.87 tokens/s). Receipt:
`artifacts/stream-gen-benchmark-final.json`; validated comparison:
`artifacts/stream-gen-benchmark-comparison.json`. These sequential runs are not
an interleaved optimization experiment; the timing difference is not attributed
to the dispatch/default cleanup.

On that final binary, default streamed generation produced four expected IDs
with empty stderr. A K/V budget of 1,605,631 bytes rejected a reservation needing
1,605,632; 33 promised tokens and streamed flags in resident mode also failed
with empty stdout. Receipts: `artifacts/stream-gen-{default,under,over,wrong-mode}`
with `.json`/`.stderr` suffixes.

Model hashing warms the file cache. Load/prefill, verification, constraints and
preview are excluded from this decode baseline. Benchmark comparison tooling
rejects mixed memory modes or unequal streamed budgets; old mode-less receipts
are interpreted as resident. Planned stage bytes are never relabeled as total
resident weight bytes.

## Next gates

1. Profile weight reads, BF16→FP32 conversion, projection tiles and layer
   readbacks separately. Reduce the measured dominant cost while preserving
   all-logit parity and the explicit staging budget.
2. Measure repeated request lifetimes and allocator retention in one long-lived
   process, then qualify growing contexts before raising the 32-token ceiling.
3. Keep the V4.1 sparse-attention precision/composition gate independent:
   Qwen parity cannot establish V4.1 execution support.

HTTP scheduling, batching, quantized execution and beyond-RAM operation remain
separate implementation gates. These results do not qualify them.

## Sampled host profile

The shipped baseline binary was sampled headlessly during 16-token streamed
generation, without verification, constraints or preview:

```sh
samply record --save-only --unstable-presymbolicate --duration 20 \
  -o artifacts/stream-gen-samply.json.gz -- \
  target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --memory-mode streamed --max-tokens 16 \
  --max-weight-bytes 81798144 --max-kv-bytes 7340032
```

The capture completed normally. Its symbol sidecar resolved the Rust/MLX
frames used below. Of 5,465 nonempty main-thread sampled stacks, 3,011 included
`load_layer`, 1,734 included `decode_bf16`, 1,713 included `project_tiled`,
and 1,562 included `read_exact_range`. These are **inclusive, overlapping**
sample counts, not additive cost fractions. The profile also includes blocked
waits; it does not measure GPU kernel time. Summary receipt:
`artifacts/stream-gen-samply-summary.json`; symbol sidecar:
`artifacts/stream-gen-samply.json.syms.json`.

This supports investigating weight staging/conversion and projection before
changing sampling or attention algorithms. It does not distinguish physical
SSD traffic from warm filesystem-cache reads, nor prove that asynchronous I/O
or device-side conversion will be faster. Those need separate parity and
memory-budget-preserving experiments.

## Opt-in phase timing

`mx gen --memory-mode streamed --debug` now reports one consumed profile per
prefill/decode in `streamed.phase_profiles`, plus aggregate metadata on stderr.
Quiet streamed and resident reports have no profile field. The executor keeps
only its latest profile; failed operations cannot return a stale success
profile. Disabling profiling clears it. Profile data contains counts and times,
not model paths, token values or logits.
Implementation: Qwen executor/profile commit `80a598a`, CLI integration
`af95150`. Default/Metal quality gates and the Rust 1.87 all-feature check pass;
logs are `artifacts/check-stream-profile-{default,metal-pass}.log` and
`artifacts/msrv-stream-profile.log`.

The intervals are host wall-clock boundaries around reads/conversion,
explicit evaluation and readback. Layer execution includes dropping its
temporary arrays. Projection uses its existing sequential load and execution
timers. `total_ms` brackets the tracked append stages, not the sampler or
subsequent report serialization; the outer CLI timing remains authoritative
for the complete prefill/decode call. An explicit remainder accounts for work
outside the named phases. No isolated GPU timing or physical SSD claim follows.

Three fresh-process, eight-output-token runs used the same prompt and budgets
as the speed baseline, with `--verbose` enabled. The 21 decode profiles (seven
per run, no first-decode discard) produced identical token IDs and these means:

| Tracked phase | Mean ms/decode | Share of tracked total |
|---|---:|---:|
| Layer load/conversion | 188.85 | 54.1% |
| Projection load/conversion | 61.97 | 17.7% |
| Projection execution/readback | 51.89 | 14.9% |
| Layer execution/readback | 45.88 | 13.1% |

Mean tracked total was 349.36 ms. Embedding, final norm and remainder account
for the remaining time. These instrumented values diagnose cost, not speedup.
Every profile had finite nonnegative phases, exact token/K/V counts, and a
phase-plus-remainder sum equal to total within `1e-6` ms. Receipts:
`artifacts/stream-profile-{1,2,3}.{json,stderr}`, `stream-profile-decode-samples.json`
and `stream-profile-summary.json`. Measured binary SHA-256:
`9923d870082c5e9ba73e1fbf375759cb472fd61cd26f1bfec8b6b584c7e5d389`.

The next optimization hypothesis is narrower now: reducing repeated host-side
weight staging/conversion may save more than changing decoder math. A
device-side conversion candidate must still reject invalid payloads, reproduce
FP32 inputs bit-for-bit and satisfy the explicit live staging budget. It has
not been implemented or measured by this profiling pass.

The final real-checkpoint lifecycle test
`repeated_candidate_lifetimes_emit_one_profile_per_successful_append` passed
in release mode across both request sizes. It checks profile single-consumption,
clearing an unread profile after invalid input, exact K/V counts and cache
non-mutation on rejection. Receipt: `artifacts/stream-profile-lifetimes.log`.
The constrained record run with both profiling and `--verify-cache` also
validated its JSON and all 12 full-vocabulary comparisons:
`artifacts/stream-profile-parity.json`.

Quiet-path controls used three fresh processes and 18 post-discard decode
observations per capture. Streamed median was 347.84 ms before instrumentation,
342.68 ms in the first after-capture, and 349.03 ms in the repeat. The first
after-capture had two spikes (626.85 and 557.70 ms), increasing its sample
standard deviation to 80.87 ms; the repeat's standard deviation was 7.06 ms.
Resident control medians were 9.18 ms before, 9.87 ms after, then 9.17 ms on
repeat. Outputs matched throughout. This does not show a consistent regression,
but the sequential, non-interleaved measurements do not prove zero overhead
or attribute the noise. This is retained instrumentation, not an optimization.

Receipts: `artifacts/stream-profile-{streamed,resident}-{after,repeat}.json`,
`stream-profile-resident-before.json`, and their `*-comparison.json` sidecars.
The before captures used binary
`967ef964da45d60309cb0ea029ec40f25fdaaebc6314c4b0c470837445e31389`;
after/repeat captures used
`038bfe232e77fa7f2e8dffa7a750afdefe10bde26c7820d172bf89ef9c741214`.
This is a comparison across builds, not a same-binary instrumentation toggle.
