# Metal command execution and synchronization

This note describes a possible future native Metal provider. It does not
describe the present MLX provider as native Metal: MLX owns its command queues,
resource bindings, submission boundaries, and completion lifetimes.

## Decision

Keep MLX as the current correctness and experimental execution substrate.
A direct Metal path is a separate experimental provider, beginning with a small independently
verifiable operation chain rather than a V4.1 rewrite. Metal 4 APIs require
macOS 26 or later; feature availability must be checked at runtime and MLX
remains the fallback. [MTL4CommandQueue][mtl4-queue]

Metal 4 cannot optimize an opaque MLX command stream. Conversely, a direct
implementation must own the whole resource and completion lifecycle before it
can safely execute model work.

## Command submission and lifetime

A legacy [`MTLCommandBuffer`][command-buffer] is the container for encoded GPU
work. A command queue creates buffers and schedules work in enqueue order on
that queue. Create a buffer, encode commands, end the active encoder, and then
`commit`; only one encoder may be active in a legacy command buffer at a time.
Scheduled and completed handlers, status, wait methods, and async completion
interfaces are the GPU lifecycle signals.

`commit` is not completion. A Rust future ready at submission may only say a
request was accepted by Metal; it must not make results readable or recycle its
buffers. A native backend should carry an owned `InFlightSubmission` lease with
the command buffer and every request-scoped input, output, staging buffer, and
binding object. Completion resolves the request and returns the whole lease to
a bounded pool. Cancellation and command-buffer error handling need that same
single owner.

By default a Metal command buffer retains strong references to resources. If an
application selects `retainedReferences = false`, it must retain every used
resource until completion. Apple warns that early release can cause runtime
errors or erratic behaviour. The initial native implementation should keep the
default; unretained mode is an optional measured optimization, never a default.
[retained references][retained-refs]

Use a long-lived device/queue and immutable pipeline state rather than creating
queues or pipelines per token. This is performance advice, not a guarantee
about throughput. Measure CPU encode time, commit-to-scheduled queue delay,
scheduled-to-completed GPU time, and end-to-end request time separately.

## Synchronization: choose the smallest correct primitive

There are three distinct synchronization questions; collapsing them into a
generic wait obscures correctness and performance.

* **A GPU producer pass writes a resource a GPU consumer pass reads.** Use a
  fence or barrier as appropriate. A fence orders memory operations between GPU
  passes: update after the producer write and wait before the consumer read. It
  is not CPU completion and cannot justify recycling a Rust allocation.
  [MTLFence][fence]
* **Work has a dependency across command buffers or queues.** `MTLEvent`
  provides a GPU event timeline on one Metal device. `MTLSharedEvent` additionally
  provides CPU notifications and can synchronize multiple CPUs, GPUs, or
  processes. [shared-event guide][shared-event]
* **CPU preparation and GPU consumption share mutable storage.** Do not reuse a
  resource while it remains visible to the GPU. Use distinct staging instances
  in a bounded in-flight ring; this permits CPU preparation of n+1 while GPU
  consumes n. Otherwise reads are undefined or incorrect.
  [CPU/GPU synchronization sample][cpu-gpu]

Fences are not merely same-command-buffer synchronization. Apple states they
can synchronize passes submitted to different `MTLCommandQueue` and
`MTL4CommandQueue` instances, provided the producing command buffer is
committed before the consuming one. Events remain the clearer default for an
explicit cross-queue timeline and are necessary when CPU participation is
required. [MTLFence][fence]

For `MTLSharedEvent`, values are monotonic: a notification registered at value
`n` runs once another action signals a value at least `n`; a CPU signal unblocks
commands waiting for that or a lower value. It is appropriate for a genuine
GPU-to-CPU-to-GPU dependency, not a substitute for completion handlers.
[shared-event guide][shared-event]

## Metal 4 responsibility model

Metal 4 command buffers are decoupled from queues and can be encoded in
parallel. An `MTL4CommandBuffer` is begun with an `MTL4CommandAllocator`, ended,
and submitted by an `MTL4CommandQueue`; queues can commit an array of buffers.
The allocator supplies backing memory for command encoding and must not be reset
until work using it completes. Its bounded allocator ring needs the same
completion-lease discipline as mutable request buffers. [Metal 4 core API][metal4-core]
[MTL4CommandBuffer][mtl4-buffer]

Metal 4 resources are untracked, so an application must express barriers for
mutable cross-stage dependencies. This is a semantic cost, not a free
optimization. A native provider needs a small typed dependency model before
enabling Metal 4; it should reject or serialize an operation graph whose
producer/consumer ordering it cannot represent. [Metal 4 core API][metal4-core]

Reusable `MTL4ArgumentTable` bindings and allocator reuse can reduce CPU-side
binding and encoding work only when profiles show that work is material. They
do not speed a matrix multiply by themselves and are inaccessible through MLX.
The Metal 4 machine-learning encoder likewise is not a general Hugging Face
checkpoint executor: Apple presents it for model packages / Core ML and graphics
integration. [Discover Metal 4][wwdc-metal4] [Combine Metal 4 ML and graphics][wwdc-ml]

Indirect compute dispatch lets the GPU fetch grid dimensions from a buffer just
before dispatch, avoiding a CPU readback when prior GPU work produced them. It
does not fuse model layers, eliminate dispatches, or solve MoE routing. For
fixed-shape decode it is likely needless complexity; consider it only for
GPU-derived sparse or compacted work sizes. [indirect dispatch][indirect]

## Experiments before an architectural commitment

1. **Submission envelope.** Implement a tiny direct-Metal operation only after
   a native boundary exists. Measure CPU encoding, commit-to-scheduled delay,
   scheduled-to-completed time, and end-to-end latency for one and bounded N
   in-flight submissions. Gate on CPU-reference byte equality, no request
   reordering, and lease release only after completion. If encoding and commit
   are insignificant against GPU execution, do not add a queue abstraction.

2. **No accidental per-layer blocking.** Compare one decode-step submission
   with deliberately split, wait-after-each-layer work. Gate on the full-logit
   parity fixture over varied lengths and cancellation while work is in flight.
   If memory or kernel boundaries make one submission worse, preserve the
   coarser boundary and record the reason.

3. **Lifetime-lease stress.** Start with retained references, then test a
   lease-owned unretained mode under aggressive allocation/reuse. Completion
   count must equal commit count; output checksums must remain valid at
   completion; no allocation may re-enter the pool early. If retained references
   are below measurement noise, retain the safer default permanently.

4. **Preparation/compute overlap.** Double- or triple-buffer only mutable
   request staging resources. Delay CPU writers adversarially while GPU work is
   active. Gate on unchanged output, monotonic event values where used, and an
   explicit admission cap. If there is no preparation work to hide, lower depth
   should win on p99 latency.

5. **Dependency and Metal 4 A/B.** Build a producer/consumer two-pass test with
   a mutable intermediate. Compare a correct legacy fence/event construction
   with Metal 4 explicit barriers, allocator reuse, and argument-table reuse.
   Gate on randomized CPU-oracle parity, no read-before-write in GPU capture,
   and allocator reset only after completion. If binding time is below noise or
   the graph has no mutable cross-pass sharing, remain on MLX.

The near-term action is observability, not a rewrite: add provider-level timing
and in-flight ownership around MLX, exercise Qwen parity under concurrent
requests, and decide whether host-side submission is material. A future probe
crate should own no model loader or server routing and fail closed to MLX.

[command-buffer]: https://developer.apple.com/documentation/metal/mtlcommandbuffer
[retained-refs]: https://developer.apple.com/documentation/metal/mtlcommandbufferdescriptor/retainedreferences
[fence]: https://developer.apple.com/documentation/metal/mtlfence
[shared-event]: https://developer.apple.com/documentation/metal/synchronizing-events-between-a-gpu-and-the-cpu
[cpu-gpu]: https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work
[metal4-core]: https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api
[mtl4-queue]: https://developer.apple.com/documentation/metal/mtl4commandqueue
[mtl4-buffer]: https://developer.apple.com/documentation/metal/mtl4commandbuffer
[wwdc-metal4]: https://developer.apple.com/videos/play/wwdc2025/205/
[wwdc-ml]: https://developer.apple.com/videos/play/wwdc2025/262/
[indirect]: https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/dispatchthreadgroups(indirectbuffer:indirectbufferoffset:threadsperthreadgroup:)
