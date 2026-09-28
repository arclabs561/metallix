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
