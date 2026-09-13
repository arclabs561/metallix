# Mathematical ideas: corrections and experiment gates

This ledger preserves useful proposals even when their first formulation is
wrong or incomplete. Checked 2026-09-13. It distinguishes inspected source,
real-arithmetic derivation, and untested performance hypotheses. It is not a
full-paper review or evidence that these methods run in Metallix.

## Recurrence: fix the convention before optimizing

Use value-by-key state $S_t\in\mathbb R^{V\times K}$ here, keys/queries in
$\mathbb R^K$, and values in $\mathbb R^V$. This is the transpose of the
FLA-oriented convention in [architecture landscape](architecture-landscape.md).
Never transpose the state without also reversing multiplication order.

For scalar Gated DeltaNet retention, the corrected recurrence is

$$
D_t=\alpha_t S_{t-1},\qquad
u_t=\beta_t(v_t-D_t k_t),\qquad
S_t=D_t+u_t k_t^\top,\qquad y_t=S_t q_t.
$$

**Rejected formulation:** predicting with the undecayed state gives a different
update. With scalar $S=2$, $\alpha=1/2$, $\beta=1$, $k=v=1$, the corrected
update is $1$; predicting before decay produces $0$. This counterexample is
derived algebraically, not an executed model test.

The inspected MLX `compute_g` uses
$\alpha=\exp(-\exp(A_{\log})\text{softplus}(a+b))$.
The expression without its outer exponential is the log-retention, not the
retention. In real arithmetic retention is positive and at most one; floating
evaluation can underflow to zero. The recurrent kernel decays before prediction.
Source: [MLX gated_delta.py, compute_g and kernel update, pinned revision](https://github.com/ml-explore/mlx-lm/blob/dcbcf786c0cf56f9a12fabe9468c887781431ae2/mlx_lm/models/gated_delta.py#L15-L109).

Expanding the corrected update gives the local algebraic identity

$$
S_t=S_{t-1}\alpha_t(I-\beta_t k_tk_t^\top)+\beta_t v_tk_t^\top.
$$

With per-key retention vector $g_t$, KDA instead has
$D_{t}=S_{t-1}\text{Diag}(g_{t})$. The right-multiplied transition is
$\text{Diag}(g_{t})(I-\beta_{t} k_{t}k_{t}^\top)$; the two factors generally
do not commute. The [pinned FLA naive KDA recurrence](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/kda/naive.py)
uses log-gates and the transposed state convention, with separate query and
value heads. A vector-gated kernel is therefore not by itself a complete KDA
adapter. Gate parameterization, normalization, head mapping, and precision
remain source-specific.

## Low-rank update journals: valid identity, unproven speedup

### Ordered chunk composition and stability

The [Gated Delta Networks paper, v3, Eq. 10 and §3.3](https://arxiv.org/pdf/2412.06464v3)
and [Kimi Linear, v2, §3](https://arxiv.org/html/2510.26692v2#S3)
give source grounding beyond the naive recurrences. The targeted reading covers
GDN §§2–3.4 and Appendix B.2, and Kimi Linear §3 and its chunk derivation and
appendix pseudocode; their reported hardware speedups were not reproduced.

For the value-by-key convention above, write each token as the affine map
$F_{i}(S)=SA_{i}+B_{i}$. Two successive tokens compose as

$$
F_2(F_1(S))=SA_1A_2+B_1A_2+B_2.
$$

Composition is associative, but not commutative. In the transposed key-by-value
convention the chronological product is instead $A_{2}A_{1}S$: explicit ordering
avoids an ambiguous product symbol. The chunked/WY construction summarizes
these ordered transitions and propagates a terminal state. It does not permit
reordering the input stream. The [pinned FLA GDN chunk reference](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/gated_delta_rule/naive.py)
and [KDA chunk reference](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/fla/ops/kda/naive.py)
make the triangular dependencies and terminal-state update inspectable.

For a unit-norm key and $0\le\beta\le1$, the eigenvalues of
$I-\beta kk^\top$ are $1-\beta$ along $k$ and $1$ in orthogonal directions.
Its spectral norm is therefore at most one. Scalar or coordinate retention
in $[0,1]$ preserves that nonexpansive bound; a uniformly smaller retention
bound gives contraction of the homogeneous state transition. This does not
bound the injected values or prove small accumulated rounding error. The
reference functions accept inputs outside this stability regime: callers must
not infer normalization or gate validation from a successful call.

Transitions can also be singular: $\beta=1$ removes the component along a
unit key. Consequently, inverse-update rollback cannot generally recover the
previous state. Snapshot/replay is the straightforward baseline. A compact
journal needs its initial-state identity, original ordered update parameters,
gate domain and rounding policy—not merely a list of subtractions.

The [FLA KDA tests](https://github.com/fla-org/flash-linear-attention/blob/516143e31fce09925e6c39ac37148444bad176c4/tests/ops/test_kda.py)
compare outputs, terminal states and gradients, with tolerance-based numerical
checks. Transfer the coverage, not one tolerance number: test chunk-boundary
splits, grouped value heads, gate extremes, cancellation, and branch replay.
Training additionally requires gradient checks; forward agreement alone is
insufficient. No recurrent adapter or journal was implemented by this reading.

### Scalar journal derivation

For the recurrence above, define $d_0=1$ and
$d_t=\prod_{i=1}^{t}\alpha_i$. If every $d_j$ is nonzero, substitution gives

$$
S_t=d_t\left[S_0+\sum_{j=1}^{t}(u_j/d_j)k_j^\top\right].
$$

Proof: divide $S_{t}=\alpha_{t} S_{t-1}+u_{t}k_{t}^\top$ by $d_t$,
use $d_{t}=\alpha_{t} d_{t-1}$, and telescope. The coefficients $u_j$ still
depend on the evolving state; the identity does not make their formation
independent or parallel.

**Retained hypothesis:** store a dense base plus a bounded low-rank journal to
defer writes and support speculative branches. **Rejected conclusion:** this
identity alone establishes either numerical equivalence or faster decode.

- Tiny cumulative retention makes division ill-conditioned or overflowing.
  Log-domain storage of $d_t$ alone does not fix the scaled update vectors.
- Evaluating the dense base against a key/query still costs a dense read;
  the journal adds low-rank work. Rank-$R$ compaction costs $O(VKR)$, or
  $O(VK)$ arithmetic per appended update when amortized over $R$ updates.
  Its traffic may nevertheless benefit from batching; measure it.
- Reassociation changes floating rounding. Compare long-horizon state and
  outputs, including small gates, cancellation, and repeated compactions.
- Rollback must restore retention metadata and the journal length. The base
  must remain immutable or copy-on-write during speculation; compaction across
  an uncommitted branch invalidates simple truncation.

Next gate: an adapter-private experiment against the ordered FP32 recurrence,
with periodic rebasing, explicit error limits, branch/rollback tests, and
measured base reads, journal reads, compaction traffic, and accepted tokens/s.
Compare against a simpler multi-token kernel that loads/stores state once per
candidate block. No shared runtime journal abstraction is justified yet.

### Vector gates and a scale plane

Let $d_t=d_{t-1}\odot g_t$ and write
$S_{t}=B_{t}\text{Diag}(d_{t})$. Componentwise nonzero $d_t$ gives

$$
B_t=B_{t-1}+u_t(k_t\oslash d_t)^\top.
$$

This follows by right-multiplying the recurrence by
$\text{Diag}(d_{t})^{-1}$. It inherits the conditioning problems above.
Arbitrary per-coordinate scales cannot simply be folded into a shared
block-quantization scale, particularly a restricted power-of-two scale format.
Retain low-bit state plus scale-plane storage as a quality experiment requiring
specified scale granularity, requantization, rebasing, and error tests—not as
an exact format substitution.

## Chunking and Metal execution

For $Q:[C,K]$, $V:[C,V]$, and state $S:[V,K]$, the proposed bare product
`QV` is dimensionally invalid. A state-mediated contribution is $QS^\top$;
within-chunk contributions additionally require the correct causal factors.
The notation $(I+L)^{-1}$ is only a schematic unit-triangular solve when $L$
is strictly lower triangular. It does not specify a complete WY algorithm or
justify computing a generic inverse. Exact factor definitions, ordering,
conditioning, and terminal-state propagation need a pinned algorithm review.

Keeping a tile in shader-private storage does not guarantee physical register
residency or survival across dispatches. Likewise, halving stored state bytes
does not establish a twofold latency improvement. Preserve the exact reduction
tree and narrowing boundary before comparing occupancy, spills, bandwidth,
and GPU time. See the source-qualified [Metal kernel guide](metal-kernels.md).

## Other proposals retained with their limits

| Proposal or claim | Verdict and evidence needed |
|---|---|
| Dense Transformer decode is $O(L)$ | Incomplete: at fixed width, one-token dense attention scans context of length $n$, giving an $O(nL)$ attention term. Projection/FFN costs are separate. |
| 36 layers × 48 heads × 128² FP32 state | Arithmetic gives 108 MiB; one full read and write is 216 MiB. Geometry and actual memory traffic remain assumptions, not a measured model result. |
| 6B active Q4 parameters at 400 GB/s | Raw 3 GB divided by assumed effective bandwidth gives 7.5 ms. This ignores scales, reuse, other precision, caches and other operators; it does not prove MoE dominates latency. |
| Cost-biased speculative proposals | Exact target sampling requires the actual changed proposal probability in acceptance and residual correction. A verifier alone is insufficient; see [serving efficiency](serving-efficiency.md). |
| Batch speculative tokens by expert | Potential reuse requires overlapping routes and preserved causality. Count rejected tokens, draft work, unique weight traffic and dispatch overhead; compare against an ordinary batched verifier. |
| Sort sparse attention accesses | Real-arithmetic invariance requires intact key/value/position pairs and masks; floating softmax reductions can change. V4.1 already sorts selected compact IDs. Scratch gathering adds traffic; see [index domains](v41-compressor.md). |
| Shared expert plus low-rank residual | Exact only if weights admit the representation at the retained rank. Otherwise a training/compression change with a quality gate, not a lossless serving trick. |
| Morton-order expert dispatch | Retained hypothesis; requires measured locality gains without unacceptable routing, sorting or staging cost. |
| Page-resident sparse weights | Retained systems experiment; demand reads, prefetch, cancellation, resource lifetime, storage bandwidth and working-set tails need measurement. See [host memory](host-memory.md). |
| Unified memory removes data movement | It can remove a discrete CPU/GPU copy boundary, not memory traffic, synchronization or SSD bandwidth limits. See [Metal memory](metal-memory.md). |
| Mamba-3 complex/trapezoidal/MIMO details and M3/M5 performance ranking | Not validated in this pass. Preserve as research leads, not architecture or performance facts; primary-paper/API coverage remains required. |

## Application order and coverage boundary

### DeepSeek: paper mathematics versus executable optimization

The [pinned V4.1 report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/DeepSeek_V41_Tech_Report.pdf)
was rechecked at §§2.2–2.4.1, §2.4.4 and §§3.2–3.2.2 for the following
implementation obligations. Extracted formulas were checked against the local
PDF text where extraction duplicated or mangled notation. This is a targeted
reread, not a new claim of full-report coverage.

**Candidate-only work is distinct from masked full-range work.** Section 2.3.2
bounds later indexers by scoring only a fixed-size candidate pool; the initial
Full layer still scans the context. In contrast, the readable
[reference `Indexer.forward`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L525)
forms full-range scores before applying the candidate mask. Porting that
sequence preserves a reference, but not the bounded arithmetic cost.

A candidate-only implementation needs an explicit mapping from compact
candidate slots back to shared-KV positions. Preserve causal lengths, offsets,
duplicates policy, score rounding and the final position sort. BF16-rounded
ties can change Top-K membership even if FP32 score errors look small. Check
intermediate query/head-weight/score boundaries, not just the final IDs.
The [selection qualification](../experiments/v41-candidates.md#cpu-final-selection)
also records the degenerate case where fewer finite scores than requested
positions allow masked but causally reachable entries into the source result.
Do not silently substitute a different sentinel policy in the optimized path.

**The FP4 range argument has hypotheses.** For a $D$-component vector $x$,
RMSNorm weights $w$ with $\lVert w\rVert_\infty\le M$, and positive epsilon,
the normalized vector $z$ obeys

$$
\lVert z\rVert_2^2
=\frac{\sum_i w_i^2x_i^2}{\lVert x\rVert_2^2/D+\epsilon}
\le M^2D.
$$

An orthogonal RoPE rotation preserves the real-arithmetic norm, hence every
rotated component has magnitude at most $M\sqrt D$. Section 2.4.4 applies
approximately $M=1$, $D=512$ to motivate dropping the second-level global
scale: about 22.6 is well below the format's representable maximum 2688.
This is a range argument, not a quantization-error guarantee. Weight bounds,
finite-precision normalization/rotation, and actual scale selection still need
tests. Indexer G32/E8M0 and main-KV G16/E4M3 are different formats; a backend's
generic “FP4 supported” flag cannot identify either contract.

**Single-pass mHC is a trained dependency change.** Section 2.4.1 shifts the
input-mixing coefficient to the preceding block. In its notation the input
is $A_{l-1}X_{l}$, not $A_{l}X_{l}$. That enables tile-local mixing before the new
coefficient reduction completes. Its ideal activation-traffic count excludes
other traffic; halving that count does not establish a twofold layer speedup.
The shift is part of V4.1, not permission to change arbitrary checkpoints.

**Bounded replay is approximate state reconstruction.** Sections 2.2 and 3.2.2
explicitly truncate local attention during a short replay. Replayed prefix
tokens reuse global KV without overwriting it; uncached suffix tokens produce
new state. Their values can depend on the cache-hit boundary. Keep this as a
separately named quality/performance mode after exact full-forward acceptance,
not an invisible implementation of exact prefix-cache restoration.

### Transfer tests, not CUDA assumptions

[DeepGEMM's mHC tests](https://github.com/deepseek-ai/DeepGEMM/blob/66081d4c9c7d7c44f13fea402e5b622aa0f409c2/tests/test_mega_mhc.py)
and its [API implementation](https://github.com/deepseek-ai/DeepGEMM/blob/66081d4c9c7d7c44f13fea402e5b622aa0f409c2/csrc/apis/mega_mhc.hpp)
provide useful contracts for a later Metal kernel. Shifted FP8 output retains
a rounded BF16 intermediate even when BF16 is not a requested output. Tests
check repeatability after different-sized invocations, repeated identical
tokens, per-stream scratch isolation, and warmup before graph capture.
Reference comparisons use tolerances while repeated invocation checks use
bitwise equality: those are different guarantees. The SM100 implementation
and CUDA synchronization primitives are not portable to Metal unchanged.

[DeepSelect's tests](https://github.com/deepseek-ai/DeepSelect/blob/0f03b68748b304863fdf0181a11458d04ae533a9/tests/test.py)
check valid range, unique indices, gathered-value identity, requested ordering,
and that the minimum selected value is no smaller than the maximum unselected
value. This is a useful tie-tolerant membership oracle. Add model-specific
mask, offset and sentinel checks separately; a generic Top-K test cannot
establish V4.1's selection semantics. These upstream tests were inspected,
not executed on this Mac.

DeepSeek remains first: source-observed query, index-key and candidate-mask
boundaries; native reindexing; producer/cache ownership; complete reduced graph;
then budgeted checkpoint execution and performance work. Recurrent journals
are a separate experiment, not a shortcut to that graph.

This pass checks recurrence algebra, dimensions, a source gate formula, journal
conditioning/cost/rollback obligations, and the listed systems hypotheses.
It does not validate every external architecture claim, re-read all underlying
papers, or measure proposed speedups. Subsequent work should update each verdict
with a pinned source, executed test, or disconfirming result, retaining the
reason a proposal was corrected or deferred.
