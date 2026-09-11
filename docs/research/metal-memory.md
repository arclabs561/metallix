# Apple Metal memory and I/O reference

This note records primary-source constraints relevant to streamed weights,
bounded working sets, and truthful memory instrumentation on one Apple-silicon
Mac. It is not a claim that every described API is available through the
installed OS, the selected M3 Max, `mlx-rs`, or MLX. Capability and performance
must be checked at runtime on the machine that runs a benchmark.

**Checked:** 2026-09-11. **Evidence:** Apple documentation claims plus local
inspection of the pinned MLX allocator source. No Metal I/O, residency-set,
heap, or no-copy implementation has been added to Metallix from this research.

## Unified memory does not erase resource policy

Apple GPUs use a unified memory model: CPU and GPU share system memory. That
does *not* make every resource equally CPU-accessible or eliminate allocation,
cache, fault, synchronization, or working-set pressure. The storage mode still
declares the permitted access pattern:

| Mode | Apple-defined access | Relevance to Metallix |
|---|---|---|
| `shared` | CPU and GPU can access system memory. | CPU-populated checkpoint staging and diagnostics. Default on Apple silicon. |
| `private` | Only the GPU can access system memory. | Candidate for GPU-produced, GPU-only scratch after parity and lifetime proof. |
| `memoryless` | GPU tile memory; textures only. | Not a weight, KV, or general buffer store. |
| `managed` | CPU accessible but GPU-backed private storage. | Intel/discrete-Mac distinction; do not select for the Mac-only Apple-silicon path. |

Apple recommends `shared` for CPU-populated or CPU/GPU-shared data and
`private` for data populated and consumed exclusively by GPU work. A private
buffer therefore cannot be inspected by the checkpoint reader after upload;
the reader must retain a shared staging representation until an explicit GPU
copy or transform produces the private resource. [Choosing a resource storage
mode for Apple GPUs](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus#overview)

`hasUnifiedMemory` only reports whether the GPU shares all of its memory with
the CPU. It is a capability observation, not a capacity guarantee, a page
pinning promise, or proof that an SSD-backed checkpoint range is resident.
[MTLDevice.hasUnifiedMemory](https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory)

## Keep allocator, device, and OS measures separate

Three scopes are useful and cannot be substituted for one another:

| Scope | Candidate signal | Boundary |
|---|---|---|
| MLX arrays | MLX active/cache/peak allocator counters | Tracks MLX allocator buffers, not ordinary Rust readers or process RSS. |
| Metal resources | `MTLDevice.currentAllocatedSize` | Metal reports bytes used by device resources; it is not an OS-memory or residency proof. |
| Working-set guidance | `MTLDevice.recommendedMaxWorkingSetSize` | Approximate allocation level below which GPU runtime performance is not affected; not a hard reservation or limit. |
| Host impact | separately collected process/VM-pressure data | Needed for Rust buffers, file cache, compressed memory, swap, and unrelated processes. |

Apple describes `currentAllocatedSize` as the total memory used by a device's
resources. [MTLDevice.currentAllocatedSize](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize)
It describes `recommendedMaxWorkingSetSize` as an approximation and recommends
keeping total resource and heap footprint below it to help preserve performance.
[MTLDevice.recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize#discussion)

The local MLX source adds a further constraint: its Metal allocator increments
active and peak values using `MTLBuffer::length()` and keeps released buffers in
a separate cache. It does not count Rust `Vec` values, checkpoint-reader
buffers, or process RSS. Snapshot fields must name that scope; a candidate
cannot report an MLX peak as “host memory.” `reset_peak_memory` changes only
MLX's process-global peak counter, not its memory/cache/wired limits. Do not
reset it from concurrent benchmark paths. The C bindings exist in `mlx-sys`,
but pinned `mlx-rs` 0.25.3 has no public safe wrapper, so any future direct
instrumentation needs an explicit dependency and a narrowly reviewed unsafe
boundary.

The operational policy should reserve headroom below the reported working-set
guidance for the OS, loader buffers, file cache, command submission, and other
applications. Arithmetic fit is insufficient: memory-pressure events,
allocation failures, or a material latency collapse fail the experiment.

## Heaps: use for proven temporary lifetimes

`MTLHeap` is a large allocation from which Metal suballocates resources. Apple
calls out faster resource creation/destruction and potential memory savings by
aliasing portions of a heap. [Memory heaps](https://developer.apple.com/documentation/metal/memory-heaps#overview)
That makes one a possible arena for temporary layer-local activations,
quantization outputs, or attention scratch whose non-overlap is demonstrated.

It does not make a large immutable-weight heap an initial design default. Such
a heap can retain capacity longer than intended and makes failure recovery and
per-tensor accounting less obvious. First use a single scratch lifetime class,
then capture `size`, `currentAllocatedSize`, `usedSize`, and
`maxAvailableSize(alignment:)`. Apple exposes those heap measures directly.
[MTLHeap.currentAllocatedSize](https://developer.apple.com/documentation/metal/mtlheap/currentallocatedsize)

The correctness gate is a numerical-parity test which deliberately exercises
overlapping lifetimes. The performance gate is a repeatable reduction in peak
device allocation or allocation cost. If neither result appears, ordinary
device/MLX allocation remains simpler and preferable.

## Residency sets are scheduling hints, not a pager

`MTLResidencySet` groups allocations which Metal may move in and out of
resident memory. A caller can add/remove allocations, `commit` pending changes,
ask Metal to prepare the set with `requestResidency`, and tell it residency is
no longer needed with `endResidency`. Apple also provides an allocation list and
`allocatedSize`. [MTLResidencySet](https://developer.apple.com/documentation/metal/mtlresidencyset)

This maps conceptually to a future expert/layer working-set scheduler: retain
the executing set, request the predicted next set, and release a preceding set
only after completion fencing demonstrates no in-flight command reads it. It
does not grant a physical-RAM guarantee, erase the need for a bounded logical
cache, or make a resource reusable before GPU completion. Residency sets do
not perform hazard tracking; Apple directs users to fences and events for that
responsibility. [MTLResidencySet](https://developer.apple.com/documentation/metal/mtlresidencyset#overview)

This is a version-gated optional optimization. `makeResidencySet(descriptor:)`
requires macOS 15.0 or newer. [MTLDevice.makeResidencySet](https://developer.apple.com/documentation/metal/mtldevice/makeresidencyset(descriptor:))
The initial loader must have a correct non-residency-set path. A later probe
must cover capability detection, set mutation/commit, events or fences,
cold-versus-warm latency distributions, and memory pressure.

## Metal I/O is a candidate beside the File reader

Metal 3 provides I/O command queues and buffers that can load file data into
GPU resources or system memory. Apple specifically positions the facility for
storage hardware and, where present, Apple-silicon unified memory. It supports
multiple command buffers, queue priority, cancellation, and shared-event
coordination with GPU work. [Resource loading](https://developer.apple.com/documentation/metal/resource-loading#overview)

An I/O command buffer can load a file range into an `MTLBuffer`; the device
creates an I/O queue and a file handle. [Resource loading](https://developer.apple.com/documentation/metal/resource-loading#overview)
`makeIOCommandQueue` is available from macOS 13.0, but can throw/fail.
[MTLDevice.makeIOCommandQueue](https://developer.apple.com/documentation/metal/mtldevice/makeiocommandqueue(descriptor:))

Metallix's current baseline is its bounded `File` `seek`/`read_exact` range
reader, not `pread`. Metal I/O must therefore be evaluated as a replacement or
additional path against that baseline, including queue/file-handle construction
and synchronization overhead. It must not be described as direct SSD-to-model
execution until its exact safetensors source-offset, destination-buffer,
alignment, cancellation, error, ownership, and Rust-FFI contracts have been
shown on a real tensor range.

The first experiment should load one selected checkpoint range, retain source
identity, compare bytes against the `seek`/`read_exact` result, and exercise
failure/cancellation. Only then benchmark warm and cold cases including setup.
If it does not win on the measured target, the existing range reader remains
the supported path.

## No-copy buffers demand a real ownership proof

`makeBuffer(bytesNoCopy:length:options:deallocator:)` wraps an existing
contiguous allocation. Apple requires a page-aligned starting pointer and a
page-aligned region; the deallocator is how an app releases the backing memory
when Metal releases its buffer. [MTLDevice.makeBuffer(bytesNoCopy:length:options:deallocator:)](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:))

An ordinary Rust `Vec<u8>` or arbitrary safetensors subslice therefore cannot
be passed casually. A sound wrapper needs a backing allocation whose actual
pointer and region satisfy the page requirements, a lifetime that outlives all
encoded and in-flight GPU uses, a correct panic-free deallocator bridge, and
validated tensor range/dtype metadata before binding. A mapped file is not
automatically eligible: its mapping and view boundaries still have to satisfy
the contract. No-copy also does not demonstrate lower physical-memory use or
better throughput. Start from a copying baseline and introduce it only when a
profile identifies the copy as material and Rust can express the lifetime.

## M3 Max is not a proxy for newer API support

M3 Max establishes the Apple-silicon target but not the installed macOS level,
`mlx-rs` coverage, or a particular API's runtime behavior. Feature selection
must use availability checks and actual creation results:

| Feature | Documented availability | Required fallback |
|---|---|---|
| Metal I/O | macOS 13.0+ | bounded `File` `seek`/`read_exact` reader |
| Residency sets | macOS 15.0+ | ordinary resource/fence ownership |
| Metal 4 command queues | macOS 26.0+ | legacy `MTLCommandQueue` path |

Metal 4's `MTL4` types are separate from legacy `MTL` types. Apple describes
incremental adoption, reusable command buffers, and reusable command
allocators, but these are future optimization candidates, not V4.1 execution
requirements. [Understanding the Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api#overview)
The queue factory itself is macOS 26.0+ and returns optional/nil.
[MTLDevice.makeMTL4CommandQueue](https://developer.apple.com/documentation/metal/mtldevice/makemtl4commandqueue())

## Five testable hypotheses

1. **Layer-boundary snapshots reveal actual GPU growth.** Record planned range
   bytes, reader bytes, MLX active/cache/peak, and Metal-device allocation when
   available after `eval` boundaries. Gate on full-logit parity and a receipt
   that distinguishes the independent resident reference from candidate arrays.
   If Metal counters are unavailable, retain MLX counters with scope marked
   incomplete rather than inventing a host-memory result.

2. **Private GPU-only scratch lowers pressure without changing results.** Move
   one ephemeral intermediate, never immutable checkpoint weights first. Gate
   on no post-creation CPU access, completion before reuse, full-logit parity,
   and measured gain. Otherwise retain shared scratch.

3. **A bounded heap reduces transient churn.** Allocate one proven
   non-overlapping scratch class from a heap. Gate on capacity/usage receipts,
   an intentional overlap stress test, and improved peak/allocation timing.
   Otherwise retain ordinary allocations.

4. **Residency prewarming reduces predicted-next-layer stalls.** On macOS 15+,
   compare explicit set request/end lifecycles with the normal path. Gate on
   events/fences, cold/warm distributions, and no memory-pressure regression.
   Otherwise use existing bounded logical admission policy.

5. **Metal I/O beats the existing range reader for selected large tensors.**
   Gate on byte-exact output, source identity, cancellation/error recovery, and
   an end-to-end latency benefit including setup. Otherwise keep serial,
   aligned `File` `seek`/`read_exact` reads.

## Source inventory and coverage

All sources below are mutable Apple documentation, read as complete rendered
page bodies on 2026-09-11; they are not historical snapshots.

| Source | Sections used |
|---|---|
| [Resource fundamentals](https://developer.apple.com/documentation/metal/resource-fundamentals) | Overview; resource, heap, storage, residency collections |
| [Apple-GPU storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus) | Overview and selection guidance |
| [Setting storage modes](https://developer.apple.com/documentation/metal/setting-resource-storage-modes) | Apple-silicon defaults and mode setup |
| [Memory heaps](https://developer.apple.com/documentation/metal/memory-heaps) | Overview and heap APIs |
| [Device working-set size](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize) | Discussion and linked counters |
| [Residency sets](https://developer.apple.com/documentation/metal/mtlresidencyset) | Lifecycle, hazards, accounting |
| [Resource loading](https://developer.apple.com/documentation/metal/resource-loading) | I/O queue, buffer, file-handle workflow |
| [No-copy buffer creation](https://developer.apple.com/documentation/metal/mtldevice/makebuffer(bytesnocopy:length:options:deallocator:)) | Alignment, lifetime/deallocator contract |
| [Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api) | Compatibility and submission model |

Gaps remain: this note does not establish `mlx-rs` bindings for Metal I/O,
residency sets, heaps, no-copy buffers, or Metal 4; it does not measure any
Metal API locally; and it does not infer a physical-memory ceiling from unified
memory or device counters.

