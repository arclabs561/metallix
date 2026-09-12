# Quantization and precision: research requirements

This is a research and validation plan, not evidence that Metallix can quantize,
load, or run a new format. Training decisions matter to serving: often the
deployed representation is constrained by how the model or cache was trained. A
smaller stored tensor helps only if format, metadata, kernel, numerical
behavior, and bytes moved agree.

## Decode versus convert

**Exact format decoding** reads a representation a checkpoint or cache already
specifies. Its contract includes bit layout, sign/exponent conventions and
special values, scale encoding and axes, group shape and ordering, packing,
padding, byte offsets, and accumulator/output type. It has a concrete oracle:
a pinned upstream implementation or independent small decoder. It does not
need a language benchmark to prove that a stored block expands as intended.

**Quantization/conversion** chooses new values, scales, clipping and rounding
for higher-precision weights or caches. It changes numerical behavior and
needs calibration plus quality evaluation. Calling both jobs “4-bit” obscures
the key difference. Generic conversion must not substitute for decoding a
published format. In particular, `expert_dtype: fp4` in Flash's config is not
permission to assume NF4, GGUF, or a vendor FP4 layout.

The target evidence is the [pinned configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/config.json),
[inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py),
and [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf).
Their revision and hashes are in the [research index](README.md#v41-source-identity).
The configuration names dynamic FP8 weights with `ue8m0` scales and 32x32
blocks plus FP4 experts; actual tensor metadata must identify the decoder. The
report separately specifies global-KV FP4. Neither fact proves an on-disk
expert representation until a tensor manifest is inspected.

## Scalar decoding gate

`deepseek::precision` now expands E2M1, E4M3FN and E8M0 scalar codes to FP32.
Tests cover every encoding, signed zeros, subnormals, NaNs, and rejection of
non-nibble E2M1 inputs. They use scalar equations and hand vectors, not an
upstream executable oracle or checkpoint slices.

Encoding authority: [OCP MX specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
sections 5.3.1, 5.3.3 and 5.4.1 (tables 2, 5, 7; read with sections 5.1–5.4).
E8M0 byte zero is 2^-127, not zero; byte 255 is NaN. E4M3FN has two NaN
codes and no infinities. E2M1 has sixteen finite codes including signed zero.
Section 5.1 explicitly leaves physical block placement unspecified.

The pinned `inference/model.py` above names these scalar formats in `Linear`.
This gate does **not** establish checkpoint nibble order, tensor offsets,
scale placement, fused arithmetic, quantization, or full-model execution.
Next: identify packed layouts from headers and upstream packing code, then
compare approved real slices before attaching these decoders to a loader.

## Source INT8 expert storage: one real header qualified

The [pinned converter](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/convert.py)
was read in full (205 lines, 2026-09-12), not executed. Its
`cast_e2m1fn_to_e4m3fn` takes a rank-two INT8 tensor, views its bytes as unsigned,
unpacks low nibble then high nibble, and doubles the input dimension. Its FP4
export branch reinterprets INT8 as `float4_e2m1fn_x2` without arithmetic.
The optional FP8 branch instead transforms weights and scales; it is not the
same representation-preserving path. The helper's "lossless" description is
not an independently qualified numerical guarantee here.

Metadata-only inspection of pinned `model-00009-of-00048.safetensors` obtained
bytes 0–7 (length prefix) and 8–258143 (258,136-byte header), both via bounded
HTTP range requests. The complete shard length is 7,389,759,032 bytes; no tensor
payload was fetched. The entire header passed range/byte-count validation and
exact tensor-name agreement with the pinned index after the dtype fix below.
For `layers.6.ffn.experts.0`, the actual metadata is:

| Projection | Weight storage / shape | Scale storage / shape | Logical weight shape from converter |
|---|---|---|---|
| `w1` | `I8 [2304,2560]` | `F8_E8M0 [2304,160]` | `[2304,5120]` |
| `w2` | `I8 [5120,1152]` | `F8_E8M0 [5120,72]` | `[5120,2304]` |
| `w3` | `I8 [2304,2560]` | `F8_E8M0 [2304,160]` | `[2304,5120]` |

The source index pairs `.weight` and `.scale` directly. The converter also
accepts the older `weight_scale_inv` spelling and renames it, without inverting
it in the FP4 branch. A suffix alone is not evidence of inverse-scale math.
Routed experts are assigned to model-parallel ranks by expert index; shared
experts follow a different branch. This inspection does not generalize their
ownership, or the Engram layout, from a single routed expert.

The real header exposed missing canonical dtype spellings in our parser:
`F8_E4M3` and `F8_E8M0`. Safetensors 0.6.2's
[Python binding](https://github.com/huggingface/safetensors/blob/v0.6.2/bindings/python/src/lib.rs)
explicitly maps these to `float8_e4m3fn` and `float8_e8m0fnu` in both directions.
The parser now accepts them alongside its previous explicit-suffix spellings.
The minimal spelling test and the full local header/index test failed before
the fix and passed afterward. The local data test is deliberately ignored in
ordinary checks; it is run explicitly with absolute `METALLIX_V41_HEADER` and
`METALLIX_V41_INDEX` paths:

```sh
cargo test -p deepseek pinned_v41_shard09_metadata -- --ignored
```

Retained source and data identities (SHA-256):

| Local artifact | SHA-256 |
|---|---|
| `artifacts/v41-convert-pinned.py` | `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76` |
| `artifacts/v41-shard09-header-pinned.json` | `139eeea4664aba161a4b4cb82a60a86a2429467601ee6584a887825f8137adfd` |
| `artifacts/v41-index-pinned.json` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| `artifacts/safetensors-0.6.2-python-lib.rs` | `651fc421fe2489f6424220c13f6c541039d890a0d38c4216d5f37ec6a310f95e` |

Receipts: `artifacts/check-v41-storage-spelling-{before,after}.log` and
`artifacts/check-v41-shard09-{before-absolute,after}.log`. A validated
INT8 weight/scale pair descriptor is described below; explicitly approved
payload-slice comparison remains a later gate. Converted raw `F4` storage,
all-shard coverage, decoded numerical parity and full V4.1 execution remain
unqualified. The current fixed-width parser still rejects packed FP4 dtype
tags; source INT8 byte storage does not require accepting them.

### Selected expert-pair metadata

`V41ExpertI8ScalePair` carries the checked relation between one source INT8
weight and its E8M0 scale. It accepts only the evidenced canonical name shape
`layers.L.ffn.experts.E.w1|w2|w3.weight`, with canonical unsigned numeric layer
and expert segments. Projection identity is an enum. It derives the exact
`.scale` mate, checks both index assignments against the caller-declared shard,
and obtains their validated ranges from the supplied header.

The weight must be nonempty rank-two INT8 `[N,P]`; logical `K = 2P` must fit
`u64` and be divisible by 32. Scale storage must be E8M0 `[N,K/32]`.
The descriptor preserves the source names, shard and byte ranges alongside
logical shape `[N,K]`. It does not require physical adjacency and does not
rescan unrelated index/header entries. Use `validate_index_shard` separately
when full-shard agreement is required.

This is metadata validation, not file authentication: the caller still binds
the header to an actual file and revision and checks layer/expert limits against
the model configuration. No payload read, nibble expansion or numerical parity
is implied. MTP, shared experts, alternate source names and converted raw-F4
headers remain outside this deliberately narrow constructor.

The explicit shard-09 metadata test now constructs descriptors for all 1,152
routed-expert pairs in that header, plus checks exact ranges and logical shapes
for expert 0's three projections. Module tests exercise names, missing or
misassigned mates, dtypes, ranks, zero dimensions, scale dimensions, grouping,
and logical-width overflow. These do not allocate tensor payloads. Local
execution receipt: `artifacts/check-v41-pair-shard09.log`.

## Next numerical join: FP4 linear runtime contract

Source: pinned V4.1
[`linear` / `Linear`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
and [`act_quant_kernel` / `fp4_gemm_kernel` / `fp4_gemm`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py).
Those functions were read in full on 2026-09-11 from the retained source;
this is source inspection, not CUDA execution or full-file reading.

The smallest next numerical reference can consume already-quantized runtime
buffers without pretending to interpret checkpoint bytes:

| Input | Logical arrangement |
|---|---|
| Activation codes | E4M3FN `[M, K]` |
| Activation scales | `[M, K/G]`, with `G` equal to 32 or 128 |
| Weight codes | E2M1x2 physical `[N, K/2]`, logical `[N, K]` |
| Weight scales | E8M0 `[N, K/32]` |
| Output | `[M, N]`, computing activation times transposed weight |

Require complete groups: `K % G == 0`. For each output element, take an
unscaled 32-term dot product, multiply its result by the corresponding
activation and weight scales, then add to the accumulated output. At group
index `g`, the activation scale index is `g / (G / 32)` and the weight scale
index is `g`. Applying scales to expanded weights before a whole-row dot is
not the same stated arithmetic order.

The existing `expand_e2m1x2_blocks32` establishes runtime lane/scale decoding,
not this GEMM. A scalar FP32 implementation of the equation would still not
establish Tensor Core reduction order or CUDA bit parity: `T.gemm` owns that
reduction. The kernel defaults to BF16 output, while its Python wrapper
allocates using `torch.get_default_dtype()`; qualify that call context and
output cast rather than infer them from storage dtypes.

Activation preparation is a separate required boundary: per row/group,
`amax = max(max(abs(x)), 1e-4)`, followed by FP32 multiplication by the
rounded FP32 reciprocal of 448, then its next power-of-two scale when
`scale_fmt` is set. Do not replace the reciprocal multiplication with division
when qualifying rounding boundaries. Divide by the computed scale,
clamp to `[-448, 448]`, and cast to E4M3FN; the scale itself is stored using
the requested scale dtype. The pinned model selects 32-element activation
groups and E8M0 scales. A reference must distinguish the computed scale from
its stored representation, and must not omit activation quantization.

`deepseek::precision::fp4_linear_runtime_f32` now implements the narrow scalar
FP32 equation over caller-supplied E4M3FN activation codes, E8M0 activation and
weight scales, and low-first E2M1x2 weights. It accepts complete G=32 or G=128
groups, checks exact buffer lengths and shape arithmetic, and validates every
result before writing the caller's output. This deliberately computes twice
to avoid allocating an unbudgeted temporary output; it is a numerical
reference, not an optimized serving kernel.

Exactly representable hand vectors check transposition, scale grouping and
packed signs. These test the stated scalar equation, not an independent CUDA
capture. Hardware FP8 conversion/reduction and BF16 output remain separate
qualification gates. No checkpoint file-to-runtime mapping is established.

Run `cargo test -p deepseek precision::linear`.

## BF16 activation preparation reference

`quantize_bf16_activations_e4m3fn` covers the pinned non-inplace activation path:
BF16 storage bits promoted exactly to FP32, complete 32- or 128-element groups,
the `1e-4` absolute-maximum floor, reciprocal multiplication, and upward
power-of-two E8M0 scales. The model selects groups of 32; 128 is a supported
kernel variant. Finite BF16 inputs bound scale exponents to -22 through 120,
so stored scales are exact and representable. Normalization uses the computed
scale; no unrounded-scale or in-place BF16 reconstruction mode is inferred.
Input/shape validation completes before either caller-owned output changes.

The private FP8 encoder enumerates finite E4M3FN values with nearest-even
rounding and preserves signed zero. This is a software reference choice.
[NVIDIA's CUDA 13.0 conversion API](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html)
documents nearest-even FP8 conversion, but that does **not** establish the
pinned TileLang cast lowering, subnormal handling, or hardware parity. Those
require an independent generated-kernel capture. This is activation preparation
for supplied-format inference, not a calibrated checkpoint converter.

Run `cargo test -p deepseek precision::activation`.

## Expert composition: preserve the casts

The pinned `Expert.forward` is not simply three FP32 matrix products around
SwiGLU. For a quantized expert in the BF16 execution context, its numerical
sequence is:

1. Quantize BF16 input into E4M3FN activations and E8M0 scales for each linear.
2. Compute `w1` and `w3`, rounding each linear output to BF16, then promote
   those results to FP32.
3. When the configured limit is positive, clamp the up branch on both sides
   but the gate branch only from above. Compute `SiLU(gate) * up` in FP32.
4. Multiply the selected routing weight into that intermediate, then cast to
   BF16 **before** the down-projection's activation quantization and `w2`.
5. Round the down-projection output to BF16. MoE accumulates routed outputs
   in FP32, adds the unweighted shared expert, and casts the sum to input dtype.

Moving routing weights after `w2` is not equivalent across these rounding and
quantization boundaries. Clamping the gate symmetrically is also a different
function. A test should distinguish both changes, not only check a zero or
uniform-weight example.

Source: pinned `model.py` `Expert`, `MoE`, and its BF16 main context; pinned
`kernel.py` `fp4_gemm_kernel` and `fp8_gemm_kernel` copy their FP32 accumulator
into BF16 shared output. Their wrappers allocate using the global default
dtype but do not override the kernel's BF16 default. This qualifies the BF16
context, not arbitrary default dtypes or actual hardware rounding behavior.

The shared expert needs a separate numerical representation: routed experts
explicitly select FP4, whereas the shared expert inherits the FP8 default.
FP8 weight scales cover `[ceil(N/G), K/G]` blocks, unlike FP4's per-output-row
`[N,K/32]` scales. The pinned model uses `G=32`. For each K block the FP8
kernel forms an unscaled dot, multiplies activation scale then weight scale,
and accumulates in FP32; the weight scale index uses `output_row / G`.
The FP8 reference tests output rows across that boundary.
Reusing the FP4 scale indexing or substituting an unquantized shared expert
would not establish full MoE agreement.

The test-only `precision::expert_composition_tests` now joins the existing
activation and FP4 linear references with software nearest-even BF16 casts.
Five reduced 32-wide cases distinguish route-after-`w2`, route-after-hidden-BF16,
omitted first-linear BF16 casts, a symmetric gate clamp, and a missing up-branch
lower clamp. Hand-derived projection and final values anchor the routing and
linear-cast cases. These are synthetic scalar composition
checks, not an executable model adapter, independent upstream execution,
hardware parity, or full MoE coverage.

Run `cargo test -p deepseek expert_composition_tests`.

## FP8 shared-expert linear reference

`fp8_linear_runtime_f32` implements the pinned FP8 activation/weight equation
over caller-supplied E4M3FN codes and E8M0 scales. Weight scales are shared
across output rows in 32- or 128-row groups, not one scale row per output.
Complete K groups are required. Each unscaled FP32 group dot is multiplied by
its activation scale and then weight scale before FP32 accumulation. Exact
length, shape, nonfinite-code and numerical-overflow checks finish before
output writes. The output remains FP32; this is not a hardware kernel.

A test-only composition also executes a synthetic FP8 shared expert and adds
its output once to a supplied-weight FP4 routed branch, with software BF16
boundaries. Independent hand values are 256 from the shared branch and 72
from the routed branch, giving 328 after the final BF16 cast. This does not
yet exercise router selection, multiple routed experts, actual weights, or
a complete block. It joins numerical representations without claiming full
MoE or upstream execution parity.

Run `cargo test -p deepseek precision`.

## Text routing into multiple experts

`flash_sqrt_softplus_routes` consumes one supplied row of gate logits and
correction biases. It applies the pinned gate temperature, sqrt-softplus
scores (softplus threshold 20), biased Top-K selection, unbiased gathered
weights, optional normalization only for K greater than one with `1e-20`, and
the route scale. Source: pinned `model.py` `Gate.forward`, lines 807–827.
The selected weights are normalized in biased-selection order before results
are presented in ascending expert ID, matching the MoE execution loop.

The helper bounds width to 4,096 experts, rejects nonfinite/intermediate
overflow and ambiguous cutoff ties, and does not invent a portable PyTorch
tie order. Within-selected-set ties and scalar reductions are not hardware
bit-parity guarantees. Gate projection and vision bias selection remain
unimplemented by this function.

`routing::flash_bf16_gate_routes` now supplies that projection for one BF16
hidden row and a row-major BF16 gate matrix. It promotes both operands to
FP32, checks each scalar product and accumulation, and feeds the logits into
the same routing helper. The explicit work bound is 4,194,304 matrix elements.
The pinned source's `Gate.forward` explicitly calls `.float()` on both
operands; the captured shard-09 header records `layers.6.ffn.gate.weight` as
BF16 `[384, 5120]`. The source's module-local FP8 `default_dtype` selects
`Linear` storage, not Torch's global default used by the gate. This software
dot reference does not establish hardware reduction parity or load weights.

Cancellation fixtures distinguish FP32 arithmetic from premature BF16 rounding:
`1 + 2^-8 - 1` preserves an accumulation residual, and
`(1 + 2^-7)^2 - (1 + 2^-6)` preserves a product residual. Each residual changes
the selected expert. The qualification test was checked against two temporary
production mutations (BF16 rounding of each product, then each accumulation);
both failed by selecting expert 1 instead of expert 0. Both mutations were
removed. This establishes sensitivity to those two rounding errors, not
general hardware equivalence.

The reduced fixture uses width/intermediate width 32, three candidate experts,
Top-2 selection, gate temperature 1, route scale 1, and SwiGLU limit 4, with
synthetic weights. These are test settings, not the released Flash dimensions
or its route scale. The reduced composition selects two distinct FP4 experts while correction
bias excludes the highest raw-logit expert. Their down projections give
hand-derived outputs 104 and 288, and the unweighted FP8 shared expert gives
256; final BF16 output is 648. The same 32-wide BF16 hidden row now feeds
the synthetic `[3, 32]` gate matrix and every expert; the gate produces logits
`[0, 1, 3]`. This joins gate projection, routing, and quantized experts, not a
complete block or pretrained-model execution.

## Serving-side choices

Weight-only post-training quantization (PTQ) commonly stores low-bit weights
while preserving higher-precision activations. GPTQ uses calibration activations
and approximate second-order information to minimize layer reconstruction
error, compensating remaining weights as it commits quantized columns. Its
low-batch gains come from reduced memory movement and bespoke CUDA
matrix-vector kernels, not a universal matmul speedup. [GPTQ](https://arxiv.org/abs/2210.17323)
therefore gives a useful constraint: an appealing encoding without suitable
packing and a kernel can reduce capacity while adding decode cost.

AWQ uses activation observations to identify salient channels and applies an
equivalent scaling transformation before weight-only quantization. It is a
quality-oriented conversion recipe, not a file format or runtime mixed
precision; the method intentionally avoids a small separately high-precision
weight set. [AWQ](https://arxiv.org/abs/2306.00978) makes the deployment
dependency clear: calibration data and workload distribution matter. Neither
GPTQ nor AWQ should rewrite Flash weights before supplied-artifact decoding and
a text-only reference fixture exist.

Activation-aware mixed precision has another bottleneck profile. Prefill can
be compute-bound, so lower-precision activations help only with an efficient
compatible GEMM. Low-batch decode is often weight-traffic dominated, so
weight-only formats can help while activations remain BF16/FP16. Outlier
channels, residual paths, norms, logits, softmax reductions, routing scores,
and accumulators often need wider precision. “W4A8” or “W8A8” is a complete
data-layout/kernel proposal, not two labels: it needs saturation, layer-error,
and end-to-end checks.

KV quantization is an independent representation problem. KIVI observed, for
the decoder-only models it studied, that key-cache outliers favored per-channel
grouping while value caches favored per-token grouping; it retained a
full-precision residual cache. [KIVI](https://arxiv.org/abs/2402.02750) is a
reason to measure key/value distributions separately, not an acceptance test
for Flash. CED, shared CSA2 global KV, and the separate SWA cache make a
KIVI-style two-bit path a new quality-changing experiment.

The Flash report instead specifies global KV FP4 E2M1 with one E4M3 scale per
16 channels, after RoPE, with no second-level global scale. It keeps SWA KV in
FP8 because of sensitivity, and introduces QAT during post-training for the
main FP4 KV cache. A faithful runtime must decode and reproduce this cache path
before comparing an alternate KV quantizer. Its reported 890 bytes/token is
accelerator-resident global-KV accounting, not a general cache formula,
host-memory result, or SSD-traffic claim.

## Training precision, scales, and metadata

PTQ starts after training and chooses calibration inputs, statistic scope,
clipping, and rounding. QAT exposes training or post-training to simulated or
actual quantization so weights adapt; a serving runtime cannot recreate it
without training authority and a quality program. Flash cache QAT is direct
evidence against presuming generic post-hoc four-bit conversion is equivalent.

The [FP8 interchange-format paper](https://arxiv.org/abs/2209.05433) explains
why scale handling is first-class. E4M3 trades range for precision; E5M2
trades precision for range. Its training recipe uses higher-precision products
with scaling around casts, generally E4M3 in forward weights/activations and
E5M2 for gradients while retaining appropriate wider state. It also shows that
scaling, calibration, and whether residual paths are quantized materially
change PTQ behavior. That informs Metal test design; it does not prove a Metal
FP8 instruction path.

Per-tensor scales are cheap but let one outlier degrade a tensor. Per-channel
or per-group scales reduce this, at the cost of scale loads, address work,
alignment and metadata. For packed FP4, two values normally share a byte before
alignment; true bytes are not `elements / 2`. Count `ceil(elements / 2)`, every
scale, zero-point when present, group-tail padding, tensor alignment, and the
header/index records needed to find them. For offloaded experts and Engram,
separately record disk reads, host staging, Metal uploads, resident compressed
bytes, transient expanded bytes, scale/metadata bytes, and cache-miss
duplication.

Rounding and accumulation are format contracts. A decoder must state
round-to-nearest-even, stochastic rounding, truncation, saturation, denormal
and NaN behavior where applicable. A GEMM must state conversion and
accumulator/output precision. FP4-to-f32 is a useful reference oracle, not a
production FP4 kernel. Likewise `ue8m0` must be decoded from supplied format
evidence, never silently treated as E4M3 because both occupy one byte.

Packing is likewise an executable contract, not a serialization afterthought.
The loader needs an explicit mapping from logical tensor coordinate to packed
byte/nibble and its scale group, including whether rows, columns, or tiles are
the contiguous group axis. A kernel needs the same mapping without an
unmeasured gather/dequantization pass dominating decode. Its block shape may
be chosen for calibration accuracy, Tensor Core tile assumptions, DMA
coalescing, or a source checkpoint's layout; these are different constraints.
Persist a layout descriptor beside the tensor-range plan and make it prove
that all values, scale groups and padding are covered exactly once.

Calibration is also an input contract. Record tokenization, prompts or corpus,
sequence-length distribution, number of samples, selected layers, statistic
(max, percentile, MSE, activation norm, Hessian approximation), clipping rule,
and seed. Reusing a convenient short-prompt corpus can hide long-context,
routing, vision-token, or rare-expert failures. A quality experiment should
therefore include both representative task metrics and failure-sensitive
numerical diagnostics: saturation rates, scale histograms, per-layer output
error, top-k route overlap, cache-attention error, and divergence by context
length. These measurements identify whether a degradation came from an
encoding, its scales, the kernel, or the calibration distribution.

## Hardware boundary

The current target is an M3 Max in Apple's Apple9 GPU family. Apple documents
that `MTLGPUFamilyApple9` corresponds to A17, M3 and M4, and feature support is
queried from `MTLDevice`; its [feature tables](https://developer.apple.com/metal/capabilities/)
are capability limits, not claims of CUDA Tensor Core or NVFP4 support. Compile
and measure a Metal kernel on this target before claiming FP8/FP4 acceleration.

CUDA documents are useful vocabulary but a separate platform result. Current
[TensorRT quantization documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
describes NVFP4 E2M1 with per-block scaling, while [Transformer Engine's
FP8/FP4 guide](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.8/user-guide/examples/fp8_primer.html)
describes NVIDIA scaling recipes. Neither establishes identical storage,
rounding, operations, or throughput in MLX/Metal. CUDA measurements belong in
a CUDA comparison lane, never a Metal benchmark table.

## Validation gates and alternatives

1. **Identity/manifest:** pin config, source and tensor-index revisions;
   inspect tensor metadata and reject unsupported tags.
2. **Bit-exact decoder:** hand vectors cover nibble order, group boundary, tail
   padding, scale, zero/sign, saturation and special-value behavior; compare to
   the pinned reference where it defines the representation.
3. **Range integrity:** verify offsets, shapes, dtype and metadata accounting;
   carry scales with their values and reject truncation/overlap.
4. **Numerical path:** compare an operation, layer, then text-only forward;
   report max/mean error, routing/index agreement and token/logit agreement.
   For KV, include prefill, decode, and cache-resumption boundaries.
5. **Quality-changing branch:** only after faithful decoding passes, test a
   named GPTQ/AWQ or KV proposal using fixed calibration corpus, prompts,
   sampler, seeds, task suite, and an explicit quality budget.
6. **Capacity/speed:** measure cold/warm reads, staging/upload, peak
   resident/transient memory, compressed/expanded cache bytes, TTFT and
   inter-token latency under matched model revision and device conditions.

Keep four alternatives separate: exact Flash decoding (required); a later
weight-only PTQ experiment; an activation-aware W8A8/W4A8 kernel experiment;
and a KV grouping experiment. None becomes a generic “quantization” checkbox.

## Source and reading ledger

| Source | Identity | Coverage | Role |
| --- | --- | --- | --- |
| [V4.1 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf) | pinned revision; hash in index | Full body §§1–6 and appendices A–C previously recorded; §§2.2–2.4 and §3.2 rechecked | Target cache/QAT |
| [GPTQ](https://arxiv.org/abs/2210.17323) | arXiv v2 | Full PDF captured; §§1–4 and timing appendix read in detail | Weight-only PTQ |
| [AWQ](https://arxiv.org/abs/2306.00978) | arXiv | Full PDF captured; abstract and method/result excerpts reviewed, not full-body reading | Activation-aware PTQ |
| [KIVI](https://arxiv.org/abs/2402.02750) | arXiv | Full PDF captured; abstract and cache-layout excerpts reviewed, not full-body reading | KV alternative |
| [FP8 formats](https://arxiv.org/abs/2209.05433) | arXiv | Full PDF captured; §§2–5 and conclusion excerpts reviewed, not full-body reading | FP8/scaling/training |
| [Apple GPU family](https://developer.apple.com/documentation/metal/mtlgpufamily) | mutable, checked 2026-09-11 | Relevant family/capability sections | Apple9 boundary |
| [TensorRT quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html) | mutable, checked 2026-09-11 | NVFP4/FP8 sections | CUDA-only comparison |

The five primary PDFs are ignored local captures named `artifacts/quant-reference-*.pdf`.
No weights, GPU work, conversion, build, or host configuration change was performed for this note.
