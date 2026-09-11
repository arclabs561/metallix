# Uncertainty and confidence signals

This note distinguishes observables from claims about answer correctness.
Selected-token logprobs and grammar feasible mass are available through
`mx gen --logprobs`. Entropy/varentropy, semantic uncertainty, calibration,
abstention and hidden-state probes are not implemented yet.

## Probability namespaces

For a raw model distribution `p(token | prefix)`, the selected-token logprob is
`log p(emitted_token | prefix)`. Its sum is the log-likelihood of one *teacher-
forced model token path*. It is not the probability of the stochastic serving
policy's outcome when temperature, top-p/top-k, a grammar mask, tool
intervention, or retry policy differs. Mean negative logprob reduces length
dependence but still measures how typical the text is to the model, not whether
the answer is true. Reasoning length, rare wording, tokenization, and fixed
formatting can all confound it.

Full token entropy is `H(p) = -sum p_i log p_i`; varentropy is the variance of
`-log p_i` under that same full distribution. They describe local next-token
ambiguity. A selected token alone cannot recover either statistic: compute an
exact log-sum-exp and first/second surprisal moments while examining the whole
vocabulary distribution. Aggregate answer-span mean, maxima, and quantiles;
do not quietly turn them into a correctness probability.

With an active constraint, preserve a separate allowed set `A(prefix)`, raw
feasible mass `alpha = sum(i in A) p_i`, and constrained distribution
`p_A(i) = p_i / alpha` for allowed `i`. A constrained selected-token logprob is
the raw logprob minus `log alpha`; it therefore includes grammar tightness, not
just model preference. Record raw and constrained telemetry separately, and
separate answer tokens from forced delimiters. A score calibrated under one
model revision, tokenizer/template, schema/grammar, decoding settings, or tool
pipeline is not automatically valid under another.

## Signals with different costs

Semantic entropy clusters multiple sampled completions by meaning, then measures
entropy over meaning classes. It removes harmless paraphrase variation, but
needs several stochastic forwards and a semantic-equivalence oracle (NLI,
execution, or a judge). It detects disagreement or confabulation risk, not
correctness in general. Ensemble semantic entropy additionally needs several
models; it can expose a single model's consistent wrong mode but is not a
single-Mac first-path technique.

Verbalized confidence and P(True) are an evaluator task, not a free property of
the generated answer. They need an extra prompt/forward and ground-truth
calibration. Abstention is a policy decision over a calibrated error score plus
the application's false-accept and false-reject costs.

Hidden-state probes can make a cheap inference-time score only after offline
training. Capturing activations/readback is a model-specific execution cost;
the probe predicts its chosen target (for example semantic entropy or labelled
error), not truth by definition.

## Calibration gate

Freeze the checkpoint, tokenizer/chat template, prompt distribution, decoding
parameters, constraint/intervention policy, and correctness oracle. Split by
prompt/task/template family before fitting; train the calibrator, choose an
abstention threshold on validation, and report one untouched test. Report
Brier/NLL and reliability or ECE for probability estimates; report AUROC plus
risk--coverage/AURC and fixed-FPR false-accept rate for abstention. For
structured output, schema validity and task correctness are distinct labels.

## Near-term boundary

The next low-cost diagnostic would extend the existing selected-token logprob
and `allowed_log_mass` trace with raw entropy/varentropy, span kind, sampling
policy, and approximation markers. `allowed_log_mass` already represents raw
`log alpha` when constrained; the other proposed additions are not implemented. A
tiled vocabulary projection can accumulate the moments without another model
forward, although it still adds reductions/readback. First evaluate these on a
small fixed labelled fixture and held-out split. Do not display a user-facing
confidence or abstain automatically until that gate succeeds.

Semantic sampling, P(True), ensembles, and activation probes are later
experiments with their own cost and quality gates.

## Primary sources and coverage

- [Jiang, Araki, Ding and Neubig, *How Can We Know When Language Models Know?*, arXiv:2012.00955](https://arxiv.org/abs/2012.00955): abstract, calibration framing, and experiment summary read. It finds raw generative QA probabilities miscalibrated; its reported code is [lm-calibration](https://github.com/jzbjyb/lm-calibration).
- [Kadavath et al., *Language Models (Mostly) Know What They Know*, arXiv:2207.05221](https://arxiv.org/abs/2207.05221): abstract, self-evaluation, and OOD-calibration sections read. P(True) is format- and evaluation-dependent; zero-shot and OOD calibration remain caveats.
- [Kuhn, Gal and Farquhar, *Semantic Uncertainty*, arXiv:2302.09664](https://arxiv.org/abs/2302.09664): abstract read. It introduces meaning-level entropy; no full-paper claim here.
- [Kossen et al., *Semantic Entropy Probes*, arXiv:2406.15927](https://arxiv.org/html/2406.15927): method and evaluation sections read. It uses hidden-state probes trained on semantic-entropy targets; public [MIT-licensed code](https://github.com/OATML/semantic-entropy-probes) was verified.
- [Eusebi et al., *Reading Calibrated Uncertainty from Language Model Trajectories*, arXiv:2605.22864v2](https://arxiv.org/abs/2605.22864v2): method and experimental split/probe sections read. Its layer-trajectory probe is relevant prior art, not a Metal result.
- [Wei et al., *Ensemble-Based Uncertainty Estimation for Code Correctness Estimation*, arXiv:2603.27098v2](https://arxiv.org/abs/2603.27098v2): method and selective-generation evaluation sections read. It is code- and ensemble-specific; no code release was verified here.
- [Zhang et al., *Direct Confidence Alignment*, arXiv:2512.11998v1](https://arxiv.org/abs/2512.11998v1): method and limitations read. It aligns verbalized to token confidence, not ground-truth correctness, and reports model-dependent results.

The coverage labels above are intentionally selective; they do not assert full
paper or appendix reading. Upstream benchmark results are not predictions for
Apple Silicon or Metallix.
