# Changelog

User-visible changes and implementation milestones. Unreleased work is not a
published release; reference operators do not imply full-model support.

## Unreleased

### Added

- Reduced V4.1 CPU source-forward harness with an explicit shape/schedule
  manifest, hash-checked source loading and independent quantized-kernel
  replacements. Synthetic prefill/decode captures include final logits and
  candidate-filter checks; this is not native Rust full-model execution or
  upstream GPU parity.
- Bounded FP32 linear reference with atomic output writes and no BF16
  narrowing. Compressor fixture tests now execute BF16/FP32 projection
  matrices before native pooling and normalization.
- Model-local scalar compressor pooling and partial-group state with separate
  BF16 ratio-one and FP32 gated inputs. Learned projection results are supplied
  by the caller; cache publication and scheduling remain separate.
- Native compressor prefill and singleton completion joined with index-key
  preparation, Metal selection and mathematical attention in a batch-one
  composition test, retaining the premature-latent-rotation counterexample.
- Text prompts for `mx gen --prompt`, with decoded generated text in its JSON
  report. Raw `--input-ids` remains available for numerical diagnostics.
  Prompts are encoded as plain text, without a chat template.
- Inline constraints through `--json-schema-inline`, alongside schema files.
- Bounded V4.1 Engram hash-state and preprojected BF16 residual-gate references,
  checked against captures from pinned upstream code. Gate tests cover
  per-copy normalization, shared values, signed-zero masking, multiplication
  order, and failure without partial output writes.
- `just check`, `just check-metal`, and `just check-fixtures` development
  commands. The Engram fixture integrity checker also runs in the canonical
  quality gate.
- Source-captured V4.1 compressor and two-block sequencing tests, using
  explicit projection/sublayer stubs. Compressor checks cover partial groups,
  sequential decode, reset and exact BF16 output; block checks cover the
  attention-to-FFN and inter-block pre-mix handoff.
- Offline particle ancestry tests for absorbing EOS, resampling, effective
  sample size and weighted final selection through the categorical sampler.
  These do not provide model-backed SMC or physical cache branching.
- Pinned V4.1 compressed-cache publication traces covering indexer visibility
  before mutation, post-write cache reads, partial groups and shared consumers.
  These check source control flow with explicit numerical stubs, not a native
  Rust attention implementation.
- Library FP4 activation reconstruction through
  `deepseek::precision::requantize_bf16_activations_e2m1`, with typed
  compressed-KV E4M3 and indexer E8M0 modes. The former test-only implementation
  now serves the oracle and composition tests through the public API, using
  one fallible staging allocation and no diagnostic trace arrays.
  E2M1 rounding remains a software assumption, not qualified GPU behavior.
- Compressed-only attention composition tests joining rotary, FP4
  reconstruction and mathematical sparse attention with supplied latents
  and indices. Intermediate BF16 values and final attention are checked
  against an independent CPU oracle.
- Metal-feature composition of FP4-prepared index operands, GPU index scores,
  final selection and compressed-vector attention, checked against closed-form
  scores and output. That test supplies post-projection index operands.
- Synthetic-weight index-key projection and RMSNorm joined with rotary,
  FP4 preparation and Metal selection. A numerical ordering check detects
  reading compressed latents after attention has rotated them.
- Synthetic BF16/FP8 index-query projection and signed BF16 head-weight projection
  joined with rotary, FP4 reconstruction and Metal scoring. Closed-form checks
  distinguish omitted rotation, early quantization and omitted head scaling;
  FP8 intermediate checks expose skipped activation quantization and shared
  output-block scales. Checkpoint loading and GPU GEMM parity remain open.

### Fixed

- New Metal composition tests serialize process-global MLX device use,
  matching the library tests' guard convention.
- Qwen checkpoint discovery accepts Hugging Face cache shard symlinks that
  resolve to regular files, while rejecting broken and non-file targets.
- Generation tokenization disables saved tokenizer padding and truncation,
  and checks tokenizer IDs against the model's vocabulary width.

### Current limits

- Full DeepSeek-V4.1-Flash generation is not implemented. Engram table decoding,
  real `wkv` projection and Metal execution remain separate qualification work.
- The new Engram references are scalar correctness tools, not performance
  improvements or evidence of larger-than-memory serving.
