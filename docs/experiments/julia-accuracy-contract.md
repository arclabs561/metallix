# Julia full-stack numerical-fidelity experiment

Status: protocol frozen before new native calibration/acceptance outputs.
Starting implementation: `5964f59`. This experiment does not change the enabled
`1e-5` full-hidden source gate or qualify a checkpoint, ordinary typed request,
or Metal implementation.

## Hypothesis

An independently expressed F64 calculation using the exact existing F32 weight
values can distinguish source-runtime rounding from native implementation error
and support a numerical-fidelity contract for the bounded 22-layer composition.
Failure is informative: source/native agreement alone is not mathematical
accuracy, and a small decision-score error cannot excuse an incorrect encoder.

## Reference and provenance

Use the pinned Julia/ModernBERT topology, deterministic synthetic weights, and
the existing eight-position resource limit. Generate weights in F32 once, then
cast those exact values to F64. Do not regenerate weights with F64 arithmetic.
Express matrix products, normalization, rotary frequencies/trigonometry,
activations, and attention softmax in F64. The source's `.double()` path is not
this reference because some source operators explicitly retain F32 arithmetic.
Record the distinction between ideal F64 rotary coefficients and the source's
F32 rotary construction; use promoted source coefficients as a diagnostic if
needed to attribute that difference.

Record source revisions/hashes, reference implementation hash, exact weight
hash, case-manifest hash, Torch build/backend identity, and native commit.
The F64 calculation is a higher-precision engineering reference, not a proof
of exact real arithmetic. Cross-check its topology against existing operator
oracles and intentional defects before treating its output as authoritative.

## Splits and measurements

All five previously inspected full-prefill cases are legacy diagnostics.
Freeze new calibration and acceptance cases in a checked-in manifest before
generating their native outputs. Use differing lengths, token order, sparse
padding, qtypes, marker order, seeded random inputs, and repeated-token near
ties. Keep padding-perturbation pairs in the same split. Separate input
identities do not establish generalization to new weights or real checkpoints.

Frozen manifest: `fixtures/julia-1/accuracy-cases.json`, SHA-256
`e41b492e7ec8e0b0545515eda40fae63d4b1f87ea8fb54545acb277911f62b82`.
It contains eight calibration and eight acceptance cases. Native receipts must
match the exact expected case set for the selected split, with no missing,
duplicate, extra, or mixed-split records. Bind results to manifest, parameter,
source and reference identities; reject wrong shapes, nonfinite values, and
missing boundaries before calculating errors. Broadcasting or silently omitted
comparisons must never create an acceptance result.

First measure source F32 against the F64 reference on calibration inputs.
Retain embedding, each of 22 block outputs, final normalization, and raw valid
marker scores. Report maximum absolute error, p99, and normalized RMS error per
case and boundary, with explicit absolute errors near zero. Do not replace
per-case failures with an average. Compute probability comparisons using one
common F64 softmax over valid options.

## Prospective acceptance rule

The following is an engineering non-inferiority budget, not an analytic bound
for the nonlinear network. Its factor and floors are fixed before new native
results are read; failure does not authorize increasing them.

For each hidden boundary `l`, let `s_l` be the largest reference RMS over
calibration cases, floored at one. With `u = 2^-24`, define:

```text
e_l = max over calibration elements of
      abs(source_f32 - reference_f64) / (s_l + abs(reference_f64))
b_l = max(8*u, 2*e_l)
element passes iff abs(candidate - reference) <= b_l*(s_l + abs(reference))
```

The factor two permits at most a twofold calibration rounding envelope; the
eight-unit-roundoff floor avoids a zero budget from exact or nearly exact
calibration cases. Both choices are explicit policy. Freeze the resulting
`s_l` and `b_l` before opening acceptance outputs. Apply the same contract to
source F32 and native F32 on acceptance cases. A source failure indicates an
inadequate calibration envelope; it is not a native pass or a reason to retune
using acceptance data. A native failure requires diagnosis or a new versioned
experiment with fresh acceptance data.

Decision scores retain the already used `1e-5` absolute budget against the F64
reference. Valid-option probability maximum error must be at most `5e-6`.
Softmax is one-half Lipschitz from logit infinity norm to probability infinity
norm, so this probability target follows from that score budget. Require exact
invalid-marker masking and finite outputs. Require winner agreement when the
reference top-two margin exceeds `2e-5`; otherwise report the winner as
uncertified by this budget, including exact ties. Report total variation too.
Do not describe numerical probability fidelity as probability calibration or
task accuracy.

## Falsifiers and promotion

Check that omitted final normalization, a wrong weight mapping, inverted masks,
and wrong rotary coefficients breach the intended boundary or invariant. Keep
the existing 66-token single-block local-window controls; an eight-token full
stack cannot qualify composed long-context behavior or distinguish every
global/local schedule mutation. Require padding isolation and marker-order
controls in addition to numerical closeness.

Only after independent-reference checks, calibration freeze, held-out acceptance,
and defect sensitivity pass may a reviewed follow-up replace the original
full-stack source-comparison gate. Preserve the old result as a diagnostic.
Do not change thresholds, fixtures, or runtime arithmetic during acceptance.

## Results and conclusion

Calibration stopped this experiment before held-out execution. The frozen
manifest is `fixtures/julia-1/accuracy-cases.json`, SHA-256
`e41b492e7ec8e0b0545515eda40fae63d4b1f87ea8fb54545acb277911f62b82`;
it has eight calibration and eight held-out cases. The generated F32 weight
identity is `db22ef523c79b55a019a8e62f8b096157035af0945d380a9bbe6bab5586cdf68`.

The owner-local receipt
`.agents/receipts/julia/accuracy-calibration-native-f64.json` records source,
F64 and native calibration boundaries, source/build provenance and defect
controls. Seven calibration cases fit every frozen hidden boundary; `cal_len7`
exceeds 23 boundaries from layer 0 through final normalization. At layer 0 its
worst normalized error is about 2.003 times the pre-registered boundary. Raw
score and probability checks remain below their prospective budgets, but do
not qualify the encoder. The held-out split was deliberately not generated or
read for native acceptance.

The reference properties pass: constant unit-weight normalization is zero;
nonconstant unit-weight normalization has zero mean; RoPE preserves the position-zero vector and norm; masked padding
does not change visible hidden states or valid scores; and marker permutation
permutes scores. Deliberate omitted-final-norm, wrong-layer mapping, inverted
mask and wrong-RoPE-theta controls all breach the hidden defect check.

Next diagnosis starts at layer 0 for `cal_len7`: capture actual source and
native raw QKV, attention output before `Wo`, and post-`Wo` residual. Expand
to MLP intermediates only if that attention trace matches. The strict original
F32 source gate remains enabled and failing; no numerical runtime change or
acceptance threshold was made.

The reference uses the pinned Transformers revision
`08810b1e278938278c50153ee1edfd7a20a759da` and its ModernBERT source SHA-256
`83875f54a029339c62a8f5061801873d41e134b3e9abb8308b8e9b0f9f57b5dc`, plus
the pinned Julia source revision and hash recorded in the Julia decision
contract. Reproduce the opt-in diagnostic after writing the calibration export
to an absolute owner-local path:

```sh
JULIA_DIAGNOSTIC_OUTPUT=/absolute/path/accuracy-native-calibration.json \
  cargo test -p julia write_accuracy_calibration_outputs -- --ignored
uv run scripts/test_julia_accuracy_reference.py
uv run scripts/julia_accuracy_reference.py \
  --native-output /absolute/path/accuracy-native-calibration.json \
  --check-defects --check-properties \
  --write-report /absolute/path/accuracy-calibration-native-f64.json > /dev/null
```

### Rejected QK-reduction experiment

The `cal_len7` layer-0 trace located the excess before softmax: native raw QKV
was within `3.8147e-6` of source-derived F32, while attended values before
`Wo` differed by `7.4387e-5`. An isolated runtime trial accumulated only the
64-term QK dot product in F64, cast the completed dot to F32, and retained the
existing F32 score scaling, RoPE, masking, softmax, value reduction, and all
weights. The trace improved source-derived maximum errors for logits from
`1.0681e-4` to `9.1553e-5`, probabilities from `1.5169e-5` to `1.0133e-5`,
and attended values from `7.4387e-5` to `3.7193e-5`.

The frozen all-calibration result nevertheless failed. It reduced `cal_len7`
layer 0 from `2.00335` to `1.45272` times its frozen boundary, but left ten
`cal_len7` boundaries failing and introduced failures at `cal_len5` layer 21
(`1.02518`) and final normalization (`1.00866`). Scores and probabilities
passed. The trial was reverted; baseline and trial traces remain owner-local
at `.agents/receipts/julia/cal-len7-layer0-native-baseline-f32dot.json` and
`.agents/receipts/julia/cal-len7-layer0-native-f64dot.json`, and the frozen
all-calibration trial report is
`.agents/receipts/julia/accuracy-calibration-native-f64dot.json`. The next
diagnosis is same-input replay that separates QKV/RoPE/score construction from
reduction behavior; it must not stack another precision change.


Raw QKV and attended outputs above are actual pinned source-module captures.
Source logits and probabilities are explicit reconstructions from captured QKV,
not internal CPU FlashAttention observations. The original baseline/trial native
trace files store flat attention arrays in query/head/key order despite their
original `attention_shape` metadata; the reported comparisons transpose them
explicitly. New native trace schema 2 stores nested head/query/key arrays and
names that layout. Preserve this distinction when replaying historical traces.


Reproduce the retained trace without evaluating held-out inputs:

```sh
uv run scripts/julia_accuracy_reference.py --trace-case cal_len7 \
  --trace-output /absolute/path/cal-len7-layer0-source-f64.json
JULIA_DIAGNOSTIC_OUTPUT=/absolute/path/cal-len7-layer0-native-v2.json \
  cargo test -p julia write_cal_len7_layer0_trace -- --ignored
```

### Same-input layer-zero replay

The schema-2 `cal_len7` replay captures native post-RoPE Q and K in addition to
QKV and scores, so each score comparison uses the same recorded inputs. Its
owner-local report is `.agents/receipts/julia/cal-len7-layer0-replay.json`.
Native scalar-F32 replay from those captured rotated vectors is bit-exact to
the native scores. Against the explicit source-QKV reconstruction, raw QKV
differs by at most `3.8147e-6`; applying the source F32 rotation and tensor
score to native rather than source QKV changes scores by `9.1553e-5`; native
captured rotation rather than source rotation on that same native QKV changes
scores by `3.0518e-5`; and native scalar reduction rather than tensor reduction
on those same captured rotated vectors changes scores by `6.1035e-5`.

These maximum errors are non-additive. QKV propagation is the largest measured
contribution, followed by scalar reduction and then native rotation. The
source-tensor reconstruction from its captured rotated vectors and the native
scalar-F32 replay from its captured rotated vectors are both zero-error
identities. The source tensor result is an explicit F32 reconstruction, not a
claim about the opaque CPU FlashAttention kernel or mathematical truth. F32
versus ideal-F64 rotary coefficients on the same native QKV changes scores by
only `5.2076e-6`. The prior isolated F64 QK accumulator trial failed the
frozen all-calibration contract. These diagnostic comparisons locate error
sources; they do not independently establish an acceptable runtime change.

That common-QKV scalar replay is now complete: source rotation versus native
rotation differs by `3.0518e-5` under the same scalar-F32 reducer, exactly the
same maximum as under the tensor reducer. The source rotation's tensor versus
scalar reduction difference is `4.5776e-5`; native rotation's scalar versus
tensor reduction difference is `6.1035e-5`. Thus rotation is independently
material, but changing coefficient precision alone remains unsupported. A
bounded runtime experiment was a balanced F32 QK reduction: it targets the
measured reduction seam without changing native rotation or introducing F64
casts. Its predeclared falsifier was failure to move the same trace toward the
source reconstruction or to pass all eight frozen calibration hidden, score,
and probability boundaries without refitting.

### Rejected balanced-F32 reduction experiment

The balanced-F32 QK reduction trial retained F32 products and the
existing F32 score scale, replacing only the 64-term serial addition with a
fixed pairwise tree. It made the legacy strict source test pass, and its native
balanced replay is exact, but it failed the frozen calibration contract. Six of
eight calibration cases passed rather than the serial baseline's seven: it
introduced a `cal_len5` final-normalization failure (`1.04277` times its frozen
boundary), while `cal_len7` still failed layer 0 (`1.05789`) and layers 15–21
plus final normalization. Its worst ratio was `1.39199` at `cal_len7` layer 21;
scores and probabilities passed. The runtime trial was therefore reverted
automatically. Owner-local artifacts are
`.agents/receipts/julia/cal-len7-layer0-native-balanced-f32.json`,
`.agents/receipts/julia/cal-len7-layer0-replay-balanced-f32.json`, and
`.agents/receipts/julia/accuracy-calibration-native-balanced-f32.json`.

### QKV projection/input ablation

The replay report can extend the existing `cal_len7` trace with a same-weight
F64 Wqkv projection ablation. It binds SHA-256 identities for both trace inputs
and the saved baseline calibration report, requires the frozen manifest and
generated F32-weight identities, validates `[7, 384]` embeddings and `[7,
1152]` QKV tensors, and fails unless projecting the saved ideal-F64 embedding
reproduces the saved ideal QKV exactly. Reproduce it with:

```sh
uv run scripts/julia_accuracy_reference.py \
  --replay-native /absolute/path/cal-len7-layer0-native-replay.json \
  --replay-source /absolute/path/cal-len7-layer0-source-f64-v2.json \
  --replay-calibration-report /absolute/path/accuracy-calibration-native-f64.json \
  --replay-output /absolute/path/cal-len7-layer0-replay-projection.json
```

For the saved serial baseline, native QKV versus F64 projection of its captured
embedding is `5.8637e-6`; source QKV versus F64 projection of its captured
source-F32 embedding is `5.7967e-6`; and F64 projection of native versus source
embeddings is `3.3297e-6`. Source-F32 embedding versus ideal-F64 embedding
propagates by `8.5707e-7`. The values are non-additive, but projection
accumulation exceeds upstream input-normalization propagation and occurs at a
comparable scale in source. This does not justify a native-only projection
precision patch. Any future QKV change must lower the native projection gap
without increasing the upstream gap or breaking source parity, then pass all
eight frozen calibration hidden, score, and probability boundaries unchanged.
