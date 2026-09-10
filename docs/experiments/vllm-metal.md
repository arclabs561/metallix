# vLLM-Metal smoke trial

## Purpose

Exercise the closest current Apple serving reference on one Mac. The trial used
vLLM-Metal 0.28.0 and `Qwen/Qwen3-0.6B` on loopback.

## What worked

- The Metal plug-in activated, selected the MLX path, and reported paged
  attention plus chunked prefill with an 8,192-token batch budget.
- OpenAI-compatible chat completions and `/v1/models` responded normally.
- `/metrics` was enabled and exposed scheduler occupancy, success reasons, and
  prefix-cache query/hit counters.
- Two simultaneous short requests both returned HTTP 200 in about 1.54
  seconds. This is a concurrency smoke test, not a throughput benchmark.

## Install and startup observations

- The published release wheels require native arm64 Python 3.12 and a modern
  macOS release. They installed successfully in a dedicated environment.
- Invoking the repository installer from outside its checkout failed because it
  resolved its release pin relative to the current directory. Installing the
  declared release wheels directly was reproducible.
- Plug-in import and server startup are noticeable phases. A server must report
  loading, warming, and ready separately rather than treating process start or
  socket bind as readiness.

## Implication for Metallix

vLLM-Metal is the strongest service and observability oracle tested so far. It
does not establish V4.1 Flash support: that requires a qualified model adapter
and measured handling of sparse experts and Engram data. Metallix should copy
the *contract* of explicit queue, cache, and request-outcome metrics while
keeping its V4.1 execution and SSD-tier policy independent.
