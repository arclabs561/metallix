# MLX, Metal, and the Rust execution boundary

Metallix currently uses MLX through Rust bindings. Metal is the GPU execution
layer underneath MLX, not a competing choice already implemented alongside it.
The ordinary GPU path is:

```text
mx CLI and Rust model/state code
  → mlx-rs
    → MLX C bindings and C++ runtime
      → Metal backend
        → Apple GPU
```

MLX is an array-computation framework. `mlx-rs` exposes it to Rust. `mlx-lm`
is a separate model/generation library: Metallix consults its implementations
as references but does not call Python `mlx-lm` for its model execution.
Custom Metal kernels are a possible extension of the MLX path, not a reason
that all array operations must be rewritten immediately.

## What executes today

The [workspace manifest](../../Cargo.toml) pins `mlx-rs =0.25.3`, disables its
default features, and enables `metal` and `safetensors`.
The [lockfile](../../Cargo.lock) records the resolved dependency set.
The `safetensors` feature concerns data interoperation; it does not imply model
architecture support, quantization, scheduling or generation features.

The inspected release chain is `mlx-rs` revision
`93ed8db7255cdc0d3f9a890aa0b3ad40c50d1eaf` → MLX-C revision
`9ebe155864eab06d94ba18e01f9cb2666b2975a7` → MLX core `v0.25.1`.
See [the wrapper tree](https://github.com/oxiglade/mlx-rs/tree/93ed8db7255cdc0d3f9a890aa0b3ad40c50d1eaf)
and [MLX-C's build configuration](https://github.com/ml-explore/mlx-c/blob/9ebe155864eab06d94ba18e01f9cb2666b2975a7/CMakeLists.txt).
Do not infer the linked core version from a current Python installation.

| Local responsibility | Actual execution boundary |
|---|---|
| Qwen model operators | Rust constructs MLX array graphs; see [forward.rs](../../crates/models/qwen/src/forward.rs). |
| Qwen cached/streamed execution | Rust owns the model-specific executor and read/cache sequence; see [metal module](../../crates/models/qwen/src/metal.rs). |
| V4.1 index-score GPU check | MLX matmul, ReLU, weighting, head reduction, evaluation and readback; see [indexer.rs](../../crates/models/deepseek/src/indexer.rs). |
| V4.1 sparse-attention GPU check | Host-side sparse gather followed by small MLX graphs; see [attention/metal.rs](../../crates/models/deepseek/src/attention/metal.rs). |
| V4.1 numerical composition | Bounded CPU references establish model/precision contracts. They do not establish a complete GPU decoder. |

The module/feature name `metal` does not imply a direct Rust Metal API call.
Nor does a diagnostic's successful GPU result establish device-resident
selection, a fused graph, or serving throughput.

## Lazy evaluation and measurement

The [pinned binding documentation](https://github.com/oxiglade/mlx-rs/blob/93ed8db7255cdc0d3f9a890aa0b3ad40c50d1eaf/mlx-rs/src/lib.rs)
describes lazy array graphs. Evaluation, host inspection and scalar extraction
can force work to materialize. Treat diagnostic host reads as synchronization
boundaries, not harmless logging. This matters especially when measuring
launch overhead or trying to overlap reads with GPU computation.

For an optimization experiment:

1. Fix the inputs, model state and numerical reference before changing graph
   boundaries. Record dtype and rounding points explicitly.
2. Warm compilation and allocation separately from steady-state runs.
3. Time a completed computation, not just construction of lazy nodes. Keep
   synchronization placement identical in both variants.
4. Measure CPU preparation, GPU execution, host readback, active allocations,
   retained cache and process footprint separately where observable.
5. Keep debug introspection out of the timed path, or report its cost as part
   of the workload. A final-token read required by the sampler is different
   from reading every intermediate tensor.

Array ownership and in-flight work are related but distinct. An owning Rust
handle does not establish safe arbitrary cross-thread access to the runtime.
Streams, asynchronous work, cancellation, and foreign resource lifetime need
tests at the exact binding boundary before adding concurrency. Do not bypass
binding restrictions merely because unified memory makes a buffer addressable.

### Compilation and asynchronous evaluation in this release

The binding exposes `transforms::eval`, `transforms::async_eval`, `compile`
and `compile_with_state`. Asynchronous evaluation is not itself a Rust async
task or a cancellation protocol. Graph compilation can fuse operations, but
shape/dtype/input changes can trigger recompilation; recreating compiled
closures in a token loop defeats reuse.

The [release's compile module](https://github.com/oxiglade/mlx-rs/blob/93ed8db7255cdc0d3f9a890aa0b3ad40c50d1eaf/mlx-rs/src/transforms/compile/mod.rs)
explicitly warns against capturing `Array`s inside compiled closures. Pass
arrays as inputs and qualify explicit state handling through `compile_with_state`
where needed. This is not a guarantee that arbitrary side effects are safely
compiled: test changing weights, cache state, shape and repeated invocation.
Its `clear_cache` calls `mlx_detail_compile_clear_cache`; despite the generic
name/comment, it must not be presented as an allocator-cache budget control.

## Custom kernels: core support versus binding support

MLX documents [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
and [custom extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html).
Those core/Python/C++ facilities do not automatically exist in the pinned Rust
API. The locked MLX-C [fast header](https://github.com/ml-explore/mlx-c/blob/9ebe155864eab06d94ba18e01f9cb2666b2975a7/mlx/c/fast.h)
already declares custom-Metal-kernel functions; the inspected high-level Rust
`fast` module does not wrap them. A safe extension therefore need not start
by inventing a new Metal runtime, but must qualify that FFI path and its
ownership rather than assume it is a current high-level capability.
Do not write implementation plans that assume an immediately callable
`mlx_rs::metal_kernel`.

There are three possible paths, in increasing integration cost:

| Path | Benefit | Required gate |
|---|---|---|
| Compose existing MLX operations | Reuses graph evaluation, allocator and backend integration | Correct semantics and fewer actual transfers/launches in measurements |
| Extend the Rust wrapper to an MLX custom operation | Can preserve MLX graph ownership while specializing a hot operation | Safe FFI, input/output layouts, stream behavior, lifetimes and numerical tests |
| Add a direct-Metal operation boundary | Direct control of APIs not exposed by the current stack | Explicit resource sharing/copies, synchronization, errors, ownership and measurable benefit over MLX |

Recommendation: retain MLX composition as the baseline. Choose a custom path
only for a measured bottleneck with a stable numerical contract. A broad
backend rewrite is not justified by an API announcement or by another
runtime's benchmark. An upstream wrapper extension is preferable when it
solves the narrow problem without creating a second allocator/queue owner;
direct Metal becomes more attractive if the required operation cannot be
expressed or exposed that way.

## V4.1 optimization experiments

### Own semantics in the model, mechanics in the backend

| Concern | Appropriate owner | What must not leak across the boundary |
|---|---|---|
| Sparse positions, sink, causal rules, cache publications and model rounding | Model adapter and its conformance tests | Backend optimizations must not redefine these semantics. |
| Tiling, vector/matrix path, fusion, launch geometry and supported types | MLX/backend operation | CLI and scheduling policy should not select shader internals. |
| Safe foreign handles, stream ownership and capability exposure | Rust binding boundary | Raw FFI and resource lifetime should not be copied into model call sites. |
| Request admission, cancellation and work scheduling | Runtime owner as those facilities are implemented | Kernel dispatch is not request policy. |
| Arguments, formatting and diagnostics | CLI | Debug output must not secretly change the numerical operation. |

This is a responsibility map, not a proposal for a new generic tensor engine
or new crates. Keep model-specific implementations local until a second
consumer establishes a genuinely shared contract. A fused V4.1 kernel may
implement several model steps, but its caller must still supply an explicit,
testable operation contract.

### Is a one-query specialization worthwhile?

Yes: ordinary autoregressive decode issues one new query against many cached
keys. It can benefit from a vector-shaped, low-overhead reduction rather than
the tiling used for a large prefill. Query length one does not mean key length
one, one batch item, or a context-independent cost. Sparse decode replaces
the full key range with selected slots but still needs the model's mapping.

The linked core already specializes short-query attention: its vector path
admits query lengths up to eight under head-dimension/device checks, alongside
separate full-attention paths. Reuse a compatible existing operation first.
Only implement another specialization if profiling and conformance establish
an unmet need. Benchmark `Q=1`, short multi-query verification and prefill
separately, with fixed key lengths, dtypes and masks; include actual dispatch,
warm GPU time and end-to-end decode latency. Never force all attention through
a one-query kernel or infer a speedup merely from fewer operations in source.

First qualify the native model graph. Then replace host-staged diagnostics
with device-resident equivalents while retaining independent source fixtures.
For the indexer, candidate-only scoring must actually gather/score the selected
domain rather than compute the full context and mask it afterward. For sparse
attention, preserve duplicate slots, positional mapping and the denominator-only
sink. A dense attention helper is not automatically a compatible replacement.

Use fixed-capacity or blocked cache storage only after ownership and maximum
working-set requirements are explicit. Avoid repeatedly rebuilding a growing
cache if profiling establishes that movement as a bottleneck. Capacity bounds
on Rust vectors or checkpoint reads are not limits on MLX temporary allocation,
allocator caching, total process memory or operating-system paging.

Before promising a memory budget, prove which allocator controls the exact
Rust release exposes and what each control limits. Before promising asynchronous
overlap, prove which submissions can overlap and how completion is observed.
These are implementation gates, not capabilities inferred from Python examples.

The pinned [MLX memory header](https://github.com/ml-explore/mlx/blob/v0.25.1/mlx/memory.h)
and [MLX-C memory header](https://github.com/ml-explore/mlx-c/blob/9ebe155864eab06d94ba18e01f9cb2666b2975a7/mlx/c/memory.h)
provide memory metrics and controls. The inspected high-level Rust wrapper
does not expose a corresponding ergonomic memory module. Generated raw FFI
exposure is not a qualified application-budget API: a future wrapper needs
units, scope, thread safety and measured enforcement tests, including temporary
allocations and cache retention. Do not confuse memory, cache and wired-memory
limits with each other or with a process-wide hard limit.

## Source and coverage ledger

Checked 2026-09-13. No GPU benchmark, binding extension or backend switch was
performed by this research pass.

| Source | Coverage and limit |
|---|---|
| Local manifest, lockfile and linked call sites | Current Rust/MLX boundary; does not establish every transitive backend implementation detail |
| [mlx-rs revision 93ed8db](https://github.com/oxiglade/mlx-rs/tree/93ed8db7255cdc0d3f9a890aa0b3ad40c50d1eaf) | Release source, features, array/stream contracts and fast-op surface; core dispatch needs its own version check |
| MLX custom-kernel and extension documentation above | Official mutable documentation; capability of current MLX, not automatic exposure in this Rust release |
| [Metal kernel/version guide](metal-kernels.md) | Separates GPU family, SDK, runtime, compilation and numerical gates |
| [Math validation](math-validation.md) | Candidate-only work, quantized formats, source rounding and replay limits |

One concrete stale-comment example: the wrapper describes optimized scaled
dot-product attention as query-length-one only, but the linked
[MLX v0.25.1 dispatch](https://github.com/ml-explore/mlx/blob/v0.25.1/mlx/fast.cpp)
includes vector paths for more than one query and full-attention paths under
their own shape/type/mask checks. The comment is not a valid restriction.
Actual sparse/sink compatibility still needs its own test; broader dispatch
does not make a dense SDPA primitive a complete V4.1 attention implementation.
