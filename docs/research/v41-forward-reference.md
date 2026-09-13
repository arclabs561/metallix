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

## First Rust consumer: the output head

```sh
uv run scripts/v41-forward-reference.py --output artifacts/v41-forward-reference.json \
  --head-fixture-output fixtures/deepseek-v41/forward-head-reference.json
cargo test -p deepseek --test forward_head
```

The small checked-in subset contains exact BF16 final-normalization outputs,
FP32 head weights and FP32 logits from all three source calls. Rust executes
`fp32_linear_reference` using the last prefill position, then both decode
positions. Negative controls select the wrong prefill position or reverse
vocabulary rows; neither may satisfy the comparison.

The tolerance is fixed before candidate execution: the sum of two FP32
dot-product roundoff bounds, each `gamma(2K) * sum(abs(x*w))`, with
`gamma(n) = n*u/(1-n*u)` and `u = 2^-24`. The bound is evaluated in FP64 with
a guard for that evaluation's own roundoff. This is input-dependent absolute
error, not a blanket relative tolerance near cancellation. Finite normal
arithmetic without underflow/overflow is required. The test qualifies this
head boundary only.

The tail-composition test now starts earlier, from the captured final-block
residual copies and returned pre-mix coefficients. Native HC collapse and
RMSNorm must match the source BF16 storage exactly at all seven positions;
the native normalized last row then feeds the output head under the same
unchanged dot-product bound. The collapsed reference is observed by a pre-hook
on the source's final norm, not recomputed by the fixture exporter. Controls
reject passing through one residual copy and omitting normalization.

Final-block execution and derivation of its pre-mix coefficients still come
from the source graph; this qualifies the native tail, not the full Rust model.

## Native MoE sublayer

```sh
uv run scripts/v41-forward-reference.py --output artifacts/v41-forward-reference.json \
  --head-fixture-output fixtures/deepseek-v41/forward-head-reference.json \
  --moe-fixture-output fixtures/deepseek-v41/forward-moe-reference.json
cargo test -p deepseek --test forward_moe
```

The MoE subset retains layer 4's source input, actual gate decisions and output,
plus its encoded gate, four FP4 routed experts and one FP8 shared expert.
`deepseek::moe::MoEReference` computes routing and executes the selected experts
from those weights. All seven captured positions must match the final BF16
output exactly. Selected expert IDs must match exactly; route weights are
compared by ID with a fixed absolute tolerance of `2^-20`. That tolerance is
a diagnostic policy, not a proved bound on transcendental implementations.

Execution preserves the source's BF16 projection boundaries and applies route
weights after SwiGLU but before the hidden BF16 cast and W2. Routed expert
outputs accumulate in ascending expert-ID order in FP32; the unweighted shared
expert contributes once before the final BF16 cast. A source-fixture negative
control zeros the shared output projection and must fail output agreement.
Separate analytic tests cover nonzero clamps and route-weight placement; this
source manifest has its SwiGLU clamp disabled.

Two bounded property tests add coverage beyond the source trace: flipping the
up branch's sign must flip SwiGLU's sign, and renaming two routed experts
together with their gate rows and biases must preserve their combined output.
Each runs 64 generated cases with shrinking. The permutation property is
deliberately limited to two selected experts; it does not assume arbitrary
FP32 summation order is invariant. Run the properties and analytic tests with
`cargo test -p deepseek moe:: --lib`.

The API is a bounded scalar diagnostic over runtime-encoded weight views, not
a checkpoint loader, scheduler, or optimized serving path. MoE input still
comes from the source block. Attention, cache ownership and complete block
composition remain necessary before native full-graph parity can be claimed.
