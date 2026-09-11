# Metal kernel methods on M3 Max

This is a research guide for custom compute kernels behind Metallix's MLX
backend. It is deliberately about shader execution and measurement. It does
not choose model residency, SSD tiering, or command-stream synchronization.

## Capability boundary

M3 and A17 Pro are Apple GPU family 9. Treat that as a hardware-family fact,
not as shorthand for every current Metal API. A candidate kernel must pass all
three checks:

1. the device supports the required GPU family;
2. the selected macOS SDK/runtime exposes the required Metal language/API; and
3. the exact specialized pipeline compiles successfully.

The [Metal Feature Set Tables](https://developer.apple.com/metal/limits/) list
`MTLDataType.bfloat` scalar/vector cases from Apple family 6, so M3 clears that
hardware-family floor. This establishes a type-availability floor, not a
throughput claim or a V4.1 numerical-accuracy budget. FP4 is not established
by the reviewed Apple sources as a native arithmetic type: begin with explicit
packed-load, group-scale, conversion, and accumulation semantics.

### Important matrix-operation correction

There are two similarly named but different facilities:

- **Legacy MSL `simdgroup_matrix`** types and
  `simdgroup_multiply[_accumulate]` are the cooperative 8-by-8 matrix API.
  The current MSL specification search extract says SIMD-group matrix types
  have existed since Metal 2.3; legacy `half` and `float` forms predate Metal
  4, while the `bfloat` form is MSL 3.1 or later. Every participating lane must
  execute its operations under uniform SIMD-group control flow, and the mapping
  of elements to lanes is unspecified. Thus code must use the documented
  load/store/multiply interfaces, never assume a lane owns a matrix element.
- The Feature Set Table row **“SIMD-scoped matrix multiply operations”** says
  `Metal 4` and `Apple 7`. It must not be used as a capability floor for the
  legacy MSL API above. It denotes a newer feature-table category. Any code
  using that newer facility needs its own Metal-4/runtime gate.

The [official Metal resources page](https://developer.apple.com/metal/resources/)
identifies the Metal Shading Language Specification as definitive. Its linked
PDF did not render in the research reader, so the legacy-version/type summary
above comes from a current indexed specification extract rather than a
line-audited official PDF. Before writing a matrix kernel, verify the locally
installed current MSL specification sections **2.4 SIMD-group Matrix Data
Types** and **6.7 SIMD-group Matrix Functions**, and compile a capability probe
for its exact MSL target. This documented gap is intentional: it is safer than
inventing matrix layouts or operation support from CUDA/WGSL conventions.

## Execution shape, registers, and threadgroup memory

Query `threadExecutionWidth`, `maxTotalThreadsPerThreadgroup`, and static
threadgroup-memory length from each compiled pipeline. Apple recommends a
threadgroup count that is a multiple of `threadExecutionWidth`; the maximum is
pipeline-specific because its register and memory requirements matter. See
[Calculating threadgroup and grid sizes](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes).
Start with a narrow sweep of 1, 2, 4, and 8 SIMD-groups, not a universal block
size.

Family 9 changes old shared-memory intuition. Apple says its shader core
dynamically allocates register storage over a shader's lifetime, and its
flexible on-chip memory services register, threadgroup, tile, stack, and
buffer state through shared cache capacity. This often improves occupancy, but
large live accumulators, dequant temporaries, barriers, or staged tiles can
still force eviction or cause the GPU to cap occupancy to protect locality.
[Explore GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/)
is the primary explanation.

Use `threadgroup` memory when it enables genuine cross-thread reuse, a
reduction, or a controlled cooperative layout. Align its allocation and access
to 16 bytes. But compare it with direct `device` reads for a tile that is only
read once per lane: Apple specifically says that on family 9 it can be faster
to operate on cached device/constant buffers than copy them to threadgroup
memory first. [Learn performance best practices for Metal shaders](https://developer.apple.com/videos/play/tech-talks/111373/)
documents that caveat. Avoid divergent per-expert branches in a hot dot
product; compact/rout work earlier or dispatch bounded specialized variants.

## Layout and numerical discipline

Apple GPUs use unified system memory, with GPU cache hierarchy between shader
cores and device memory. That avoids a usual discrete-VRAM transfer but makes
bandwidth and locality central. [WWDC20's counter guidance](https://developer.apple.com/videos/play/wwdc2020/10603/)
recommends packed smaller data, vectorized buffer loads/stores, locality, and
avoiding device atomics and register spills.

For V4.1 packed weights, map adjacent lanes to adjacent words, unpack
contiguous vectors, and keep code/scale/zero metadata in a layout that avoids
cross-group scatter. First create a format oracle covering nibble order,
signedness, group boundaries, scale/zero broadcast, and tails. Only then test a
fused decode-dot kernel. It wins only when lower traffic/launch count survives
the register/cache cost of its temporary decode state.

Choose precision explicitly. BF16 storage can reduce traffic, but RMSNorm
variance, softmax max/sum, RoPE trig, and output accumulation need model-level
accuracy gates. Keep an f32 reference path and compare full logits or an
equivalent, fixed test fixture before accepting any lower-precision kernel.

## Specialization and pipeline lifecycle

Use function constants for a small bounded set of invariants: hidden/head
dimension, quant group size, scale/zero form, and perhaps expert intermediate
width. Apple shows that constants let the compiler remove unreachable paths and
dynamic branches in [Optimize GPU renderers with Metal](https://developer.apple.com/videos/play/wwdc2023/10127/).
Do not specialize token count, request ID, expert ID, or every sequence length:
variant explosion harms first-token latency and cache behavior.

Cache pipeline states using a typed key such as `(kernel, dimensions, quant
format, device family, language capability)` and warm the bounded variants at
model load. Pipeline compilation is a cold-start issue, not a steady-state
tokens/s win: Metal compiles MSL to AIR and subsequently creates
device-specific code for pipeline states. [Build GPU binaries with Metal](https://developer.apple.com/videos/play/wwdc2020/10615/)
documents that flow and binary archives. Consider persisted binary archives
only after in-process cache keys are correct and cold-start traces show a real
pipeline-creation cost.

## Measurement protocol

Use a fixed prompt and deterministic sampler; warm both model and pipeline;
report prefill and decode separately. Record median and tail encoder GPU time,
tokens/s, and kernel count. Wall time alone conflates queueing and compilation.
Metal System Trace gives the CPU/GPU overview; Xcode Metal Debugger supplies
encoder-level counters. The latter should guide a change through ALU,
Buffer Read/Write, LLC, memory bandwidth, occupancy, and shader-launch
limiters.

M3's Xcode 15 tooling adds shader-cost graphs, compute-thread heat maps,
SIMD-group execution history, occupancy-manager target, L1 eviction/bandwidth,
and L1 residency. [Discover new Metal profiling tools for M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111374/)
provides the important interpretation: low occupancy with saturated FP16 is
not an occupancy problem; low occupancy plus low ALU and a low launch limiter
may be an undersized grid; a high launch limiter plus low occupancy can be
resource backpressure. A counter is a hypothesis discriminator, not a score.

## Falsifiable kernel experiments

1. **Function-constant dense GEMM.** Compare generic vector tiling against
   bounded Qwen/V4.1 shape variants. Require a reference-logit gate and a
   repeatable warmed median/p95 GPU-time win. Reject if compile/cache cost or
   counters show no instruction/branch reduction.
2. **Direct versus staged FP4 tile.** Implement identical decode-dot arithmetic
   with direct contiguous device loads and cooperative staging. Require
   byte-exact decode tests. Retain only a cross-width GPU-time win without L1
   eviction or occupancy collapse.
3. **Legacy SIMD-group matrix GEMM.** After the MSL section/probe gate, compare
   the legacy 8-by-8 path to vector tiling for QKV/control GEMMs. Test all
   remainders and final logits. Reject on unsupported target, poor tile mapping,
   or dequant/register pressure.
4. **Fused FP4 decode and matmul.** Start from tested standalone decoder/GEMM.
   Accept fusion only if exhaustive format tests and model logits pass *and*
   lower buffer traffic/launch count becomes lower GPU time.
5. **Separate decode/prefill shape sweeps.** Sweep only execution-width
   multiples for batch-one decode and fuller prefill/compacted expert work.
   Retain a result only if it repeats across warm runs and relevant sequence
   shapes; reject thermal-state or single-shape victories.

## Source and coverage ledger

| Source | Coverage read | Constraint carried into this guide |
|---|---|---|
| [Metal Feature Set Tables](https://developer.apple.com/metal/limits/) | Current Apple-family/API feature rows | Apple-6 BF16 type floor; separate Metal-4/Apple-7 matrix-multiply row. |
| [Threadgroup and grid sizing](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes) | Complete page | Use the compiled pipeline's width and maximum, not a device-wide magic number. |
| [Explore GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/) | Complete transcript | Family-9 dynamic registers, flexible on-chip cache, and occupancy tradeoffs. |
| [Learn performance best practices for Metal shaders](https://developer.apple.com/videos/play/tech-talks/111373/) | Complete transcript | Function constants; direct buffer versus software-managed threadgroup cache is empirical on family 9. |
| [Discover new Metal profiling tools for M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111374/) | Complete transcript | Counter-led occupancy, L1, and shader-cost diagnosis. |
| [Optimize Metal apps and games with GPU counters](https://developer.apple.com/videos/play/wwdc2020/10603/) | Complete transcript | Buffer/vectorization/locality and limiter methodology. |
| [Optimize GPU renderers with Metal](https://developer.apple.com/videos/play/wwdc2023/10127/) | Complete transcript | Function-constant specialization and pipeline compilation workflow. |
| [Build GPU binaries with Metal](https://developer.apple.com/videos/play/wwdc2020/10615/) | Complete transcript | AIR-to-device pipeline compilation and binary archives. |
| [Metal resources](https://developer.apple.com/metal/resources/) | Complete page | Official MSL-specification link; linked PDF could not be read in this research environment. |

No external MLX or llama.cpp source was copied or adapted here. No GPU build,
benchmark, or implementation is implied by this research note.
