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
