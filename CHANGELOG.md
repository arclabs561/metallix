# Changelog

User-visible changes and implementation milestones. Unreleased work is not a
published release; reference operators do not imply full-model support.

## Unreleased

### Added

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
- Test-only V4.1 FP4 activation reconstruction with separate compressed-KV
  E4M3 and indexer E8M0 scales, compared with an independent CPU oracle.
  E2M1 rounding remains a software assumption, not qualified GPU behavior.
- Compressed-only attention composition tests joining rotary, FP4
  reconstruction and mathematical sparse attention with supplied latents
  and indices. Intermediate BF16 values and final attention are checked
  against an independent CPU oracle.
- Metal-feature composition of FP4-prepared index operands, GPU index scores,
  final selection and compressed-vector attention, checked against closed-form
  scores and output. Index projections remain supplied, not integrated.

### Fixed

- Qwen checkpoint discovery accepts Hugging Face cache shard symlinks that
  resolve to regular files, while rejecting broken and non-file targets.
- Generation tokenization disables saved tokenizer padding and truncation,
  and checks tokenizer IDs against the model's vocabulary width.

### Current limits

- Full DeepSeek-V4.1-Flash generation is not implemented. Engram table decoding,
  real `wkv` projection and Metal execution remain separate qualification work.
- The new Engram references are scalar correctness tools, not performance
  improvements or evidence of larger-than-memory serving.
