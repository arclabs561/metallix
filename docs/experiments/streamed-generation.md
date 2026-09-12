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

1. Re-profile after the retained CPU BF16 widening change below. Separate
   remaining weight-read/staging cost from conversion before trying device
   conversion or read-ahead; preserve all-logit parity and the staging budget.
2. Measure repeated request lifetimes and allocator retention in one long-lived
   process, then qualify growing contexts before raising the 32-token ceiling.
3. Keep the V4.1 sparse-attention precision/composition gate independent:
   Qwen parity cannot establish V4.1 execution support.

HTTP scheduling, batching, quantized execution and beyond-RAM operation remain
separate implementation gates. These results do not qualify them.

## Seeded replay profiling follow-up

At `9259433`, three fresh-process streamed runs were interleaved with three
resident controls. Each used prompt IDs `9707,11,1879`, four output tokens,
`--sample --temperature 0.7 --seed 42`. Streamed runs also used `--verbose`,
`--max-weight-bytes 81798144 --max-kv-bytes 7340032`; tile rows remained 1024.
No scores, grammar, preview or resident verification oracle were enabled.
All six runs produced `13,21927,11,1879`.

The existing local checkpoint and release/all-features build were used on the
M3 Max. File caches were not flushed; this is not a cold-storage measurement.
Binary SHA-256: `514b1e4f588a3f5e3e49c96fa8596ecca70a85b1485504b15107c14ee9b0b0bc`.
Across nine decode calls per mode (no first-call discard), streamed median was
295.805 ms, sample standard deviation 8.642 ms; resident median was 9.665 ms,
sample standard deviation 0.736 ms. Samples within a process are correlated;
three processes do not establish a stable performance distribution.

| Streamed host phase | Mean ms/decode | Sample standard deviation |
|---|---:|---:|
| Layer load/conversion | 155.053 | 2.587 |
| Layer execution/readback | 48.280 | 5.813 |
| Projection load/conversion | 50.289 | 0.783 |
| Projection execution/readback | 40.711 | 1.671 |

This confirms staging/conversion as the next profiling target; it does not
separate file-cache reads, copies and conversion within those intervals.
Nor is the difference from older runs an optimization result: policy, output
length and run conditions differ. No performance implementation changed in
this experiment. Raw receipts:
`artifacts/stream-profile-repeat-{1,2,3}.{json,stderr}` and
`artifacts/resident-profile-control-{1,2,3}.json`.

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

## CPU BF16 widening experiment

Retained change: `2628a71`. Instead of constructing a two-byte array from
separate indexed bytes, `decode_bf16` copies each exact two-byte chunk into
an array and calls `u16::from_le_bytes`. This stays safe Rust, accepts unaligned
slices, keeps the finite-value checks and allocates the same output vector.
Disassembly of the measured Apple-Silicon builds changed from interleaved byte
loads and table shuffles to paired vector loads and widening shifts, with
halfword loads in the scalar tail. This is compiler-specific evidence, not an
instruction-selection guarantee. Receipts: `artifacts/bf16-before.asm` and
`artifacts/bf16-candidate.asm`.

On an M3 Max with 128 GiB memory and Darwin 25.6.0, release binaries
built with Rust 1.98.0 / LLVM 22.1.8 ran serially in before-A, after-A, after-B,
before-B order. Each block used
three fresh processes, the same checkpoint hashes above, prompt
`9707,11,1879`, eight output tokens, tile rows 1024 and unchanged streamed
budgets. The command is the speed-baseline command above with the corresponding
`--binary` and `--output`; no verification, constraints or profiling was enabled.
Hashing warms the filesystem cache. Each block discards the first decode per
process and retains 18 observations; observations within a process are not
independent repetitions.

| Block | Before median ms | After median ms | Change | Before / after sample SD ms |
|---|---:|---:|---:|---:|
| A | 333.53 | 287.47 | −13.81% | 4.45 / 12.89 |
| B | 333.71 | 283.95 | −14.91% | 4.08 / 12.89 |

All generated IDs matched. Every run retained 10 cached tokens / 2,293,760
logical K/V bytes and planned 81,798,144 weight/staging bytes. This supports
lower warm-decode wall time for this workload, not end-to-end request latency,
HTTP throughput, cold-storage performance or larger-than-memory execution.

The resident negative control used the same prompt/output and ABBA order.
Before/after medians were 9.2865/9.4153 ms and 9.2721/9.4136 ms (1.39% and
1.53% slower); sample SDs were 0.2540/0.3868 and 0.4152/0.3394 ms. An additional
after-then-before block measured 9.0674/9.0783 ms before/after (0.12% slower),
with SDs 0.1298/0.4074 ms. The small differences are retained, not relabeled
as zero regression. The repeated streamed benefit justifies keeping the
bounded change; the controls do not establish statistical equivalence.

Timing receipts: `artifacts/bf16-streamed-{before,after}-{a,b}.json` and
`artifacts/bf16-resident-{before,after}-{a,b,c}.json`; comparator receipts:
`artifacts/bf16-{streamed,resident}-comparison-*.json`. The preserved before
binary is `artifacts/mx-bf16-before`, from `b10037e`, SHA-256
`038bfe232e77fa7f2e8dffa7a750afdefe10bde26c7820d172bf89ef9c741214`.
The measured candidate SHA-256 is
`1bd4b4d24bcd86c99711f8f50b7f4a3d77c5fa2b01429b0c8f53a90b1caa3c86`.
Both receipts record the checkout at measurement time (`2628a71`); the preserved
binary's provenance, not that checkout field, identifies the baseline source.

Correctness is separate: exhaustive BF16 bit-pattern tests, unaligned slices
at offsets 0–15, vector-tail lengths 0–65 and nonfinite rejection pass the
Metal quality gate. A real constrained record run with `--verify-cache`
validated the JSON and all 12 comparisons of 151,936 vocabulary logits, with
maximum absolute error `2.09808349609375e-5` against resident recomputation.
Receipt: `artifacts/bf16-parity.json`. This shared-Metal oracle is not an
independent CPU reference. The default and Metal quality-gate logs are
`artifacts/check-bf16-default.log` and `artifacts/check-bf16-metal-confirm.log`.
The Rust 1.87 all-feature/all-target check also passed (`artifacts/msrv-bf16.log`);
this checks compatibility, not that compiler's performance. The final release
rebuild reproduced the measured candidate binary hash.

Device-side BF16 conversion remains a separate experiment: a dtype view may
allocate rather than alias, so its live-memory budget and exactness must be
proved before adopting it. No new allocation, quantization or numerical
precision policy is introduced by the retained CPU change.

## Post-widening profile and projection tuning

Three eight-output-token runs with `--verbose`, using the same binary and
checkpoint identities as the BF16 candidate above, measured 21 decode profiles.
Mean tracked host-wall times were 280.31 ms total, 150.31 ms layer loading and
conversion, 47.69 ms projection loading, 39.97 ms projection execution/readback,
and 41.65 ms layer execution/readback. These diagnostic runs include the first
decode, unlike the quiet benchmark. Receipts: `artifacts/bf16-reprofile-{1,2,3}`
with `.json`/`.stderr` suffixes and `artifacts/bf16-reprofile-summary.json`.
The raw phase reports do not embed model/binary hashes; the surrounding sweep
receipts and post-run `artifacts/post-widening-identities.sha256` record those
identities separately.

This motivated revisiting the earlier synthetic projection-size experiment
with actual decoder hidden states. The existing `--tile-rows` option requires
no implementation change. At checkout `90229e1`, serial blocks used 1024-A,
4096-A, 4096-B, 1024-B order, three fresh processes per block, the same
eight-output-token prompt and budgets as above, and no diagnostics or oracle.

| Block | 1024-row median ms | 4096-row median ms | Change | 1024 / 4096 sample SD ms |
|---|---:|---:|---:|---:|
| A | 269.31 | 253.92 | −5.71% | 11.66 / 11.46 |
| B | 270.32 | 254.70 | −5.78% | 7.77 / 11.27 |

Each block has 18 post-discard observations, not 18 independent runs. All token
IDs, checkpoint/binary hashes, backend, host, remaining workload settings and
logical memory counts matched: 10 cached tokens, 2,293,760 K/V bytes and
81,798,144 planned weight/staging bytes. The raw projection tile grows from
2 MiB to 8 MiB, but the planned layer-loading peak still dominates this
checkpoint's staging budget. Receipts: `artifacts/tile-sweep-{1024,4096}-{a,b}.json`.
These intentionally differ in tile rows; the strict same-workload comparator
is not bypassed or weakened to compare them. The controlled difference was
checked explicitly against the receipts. Do not multiply this improvement by
the earlier BF16 percentage: those captures occurred at different times.

A candidate-only constrained record run per size observed maximum RSS of
264,339,456 / 282,034,176 bytes and peak footprint of 386,270,024 / 403,129,136
bytes for 1024 / 4096 rows. These are single-run observations, not repeated
memory benchmarks or enforced caps. The larger tile's measured process memory
increased even though its planned staging peak did not. Receipts:
`artifacts/tile-memory-{1024,4096}.{json,time}`.

A separate 4096-row `--verify-cache` constrained run passed all 12 comparisons
of 151,936 vocabulary logits and JSON validation, with maximum absolute error
`2.09808349609375e-5` (`artifacts/tile4096-parity.json`). The oracle is excluded
from timing and candidate memory observations.

Keep 1024 as the conservative default. For this qualified checkpoint, 4096 is
an opt-in latency/memory tradeoff:

```sh
mx gen --model /path/to/Qwen3-0.6B --input-ids 9707,11,1879 \
  --memory-mode streamed --max-tokens 8 --tile-rows 4096 \
  --max-weight-bytes 81798144 --max-kv-bytes 7340032
```

This is a warm-cache single-sequence result. Longer contexts, other model
shapes, sustained request lifetimes and tighter process-memory limits still
need qualification before automatic tile selection or a default change.
