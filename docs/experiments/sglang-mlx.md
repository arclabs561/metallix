# SGLang MLX smoke trial

## Purpose

Exercise SGLang's current Apple path as a serving-semantics reference, not as
a V4.1 compatibility claim. The trial used upstream SGLang, its non-CUDA
project configuration with the `all_mps` extra, and `Qwen/Qwen3-0.6B` on one
Mac.

## What worked

- The server exposed `/v1/models`, OpenAI-compatible chat completions, and
  Server-Sent Events on loopback.
- Two simultaneous short chat requests both returned HTTP 200 in about 0.96
  seconds. This confirms request concurrency reaches the runtime, but is not a
  throughput benchmark.
- The process reported `MlxModelRunner`, so the trial exercised the MLX path.

## Observed limits

- A listening socket appeared before the model was ready to answer a health
  probe. A server needs distinct loading and ready states.
- The runtime reported `torch_native` attention rather than custom Metal paged
  attention for this configuration. API compatibility must not be treated as
  proof of the kernel path Metallix needs.
- `/metrics` returned HTTP 404 by default. Metrics and benchmark evidence need
  to be explicit parts of Metallix's service contract.
- A short exact-output prompt still entered Qwen reasoning mode. Model template
  and reasoning controls are adapter responsibilities and require conformance
  tests, not generic request assumptions.

## Implication for Metallix

SGLang is valuable as an API and scheduler oracle, but this trial does not
establish it as a V4.1 Apple execution substrate. Metallix should retain its
own readiness, capability, and measurement contracts while comparing any
future V4.1 backend against the same benchmark profiles.
