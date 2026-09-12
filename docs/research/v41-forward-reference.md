# Reduced V4.1 CPU reference

This development harness connects the pinned source text graph to independent
CPU numerical kernels. Synthetic weights exercise implementation mechanics;
they do not produce a useful pretrained model. A successful capture is not
Rust agreement, upstream GPU parity, or checkpoint support.

## Run it

With the pinned source files already retained under `artifacts/`:

```sh
uv run scripts/v41_forward_manifest.py --require-execution-ready
uv run scripts/v41-forward-reference.py --output artifacts/v41-forward-reference.json
uv run --with torch==2.13.0 --with numpy==2.5.3 scripts/test_v41_cpu_kernels.py
```

The manifest command validates shapes and the call schedule without allocating
tensors. The capture command runs the source graph with synthetic weights and
writes encoded parameters, intermediate outputs, caches and final logits.
The runner pins its Python dependencies and rejects source-hash mismatches.
Neither command downloads model weights.

The default `just check` runs the dependency-free manifest and source-loader
tests. Numerical kernel tests require Torch and are run explicitly above.
The loader's source-body comparison is skipped if retained source artifacts
are absent; that skip is not source verification.

## Boundaries and provenance

The source loader verifies SHA-256 before executing retained `model.py` and
`engram.py`. It replaces the six kernel imports and supplies the real Engram
classes; unused vision imports are excluded. Model class and function bodies
remain unchanged, including the final head and token-hash history.

- [Pinned model source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py)
- [Pinned numerical kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/kernel.py)
- [Pinned Engram source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/engram.py)

The CPU backend retains quantization encodings, scale placement and explicit
BF16 boundaries. Torch CPU dot products are not the original GPU reduction
sequence. Captures must identify the numerical backend, source revision,
synthetic manifest and initialized tensor encodings.

## Why this fixture is not just smaller defaults

The source defaults disable Engram and candidate filtering. The manifest
instead enables both, with window-only attention, two compression ratios,
source/consumer cache sharing and nontrivial HC and expert routing.

Candidate block selection always includes the newest block. Keeping just one
block would remove score-driven choice. Asking for more final indices than
retained candidates can also admit a masked position: the pinned indexer's
final check tests causal position, not the candidate mask. The reduced
schedule therefore retains two single-position blocks and selects one final
index, with calls exposing more than two positions.

FP4 has distinct roles: compressed KV uses groups of 16 with E4M3 scales;
index query/key reconstruction uses groups of 32 with E8M0 scales; packed
expert weights use their separate group-32 representation. One generic
“4-bit” setting would erase these distinctions.

## Acceptance remains separate

The capture must expose actual mechanism coverage and complete final logits
before it can become an oracle fixture. Integer decisions and encoded storage
need exact checks; floating comparisons require a policy fixed before running
the Rust candidate. Retained source files and generated captures are local
artifacts, not fetched automatically by the ordinary quality gate.

HC kernel calls are observed without changing the source graph: ten calls per
forward record their inputs and pre/post/combination coefficients. Finite
logits, candidate-mask checks and these recorded boundaries are a
source-execution milestone, not acceptance of the full Rust comparison gate.
