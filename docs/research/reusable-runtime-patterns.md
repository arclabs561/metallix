# Reusable runtime patterns

This is a source-backed shortlist of implementation patterns to adopt only when
they survive Metallix's V4.1 parity and benchmark gates.

| Pattern | Source | Adopt in Metallix | Boundary |
| --- | --- | --- | --- |
| Separate prefill and decode compilation, then warm the decode buckets used by the scheduler. | [tinygrad model](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/llm/model.py) | Yes, after a correct one-request path. Capture compile/warm status in model readiness. | V4.1 Metal adapter. |
| Chunked prefill with a bounded token budget. | [vLLM-Metal configuration](https://docs.vllm.ai/projects/vllm-metal/en/stable/configuration/) | Yes. The scheduler owns the budget; the adapter reports safe batch capacity. | Control plane plus adapter capacity contract. |
| Explicit paged KV and prefix-cache query/hit metrics. | [vLLM-Metal](https://docs.vllm.ai/projects/vllm-metal/en/stable/) | Yes. Also record resident bytes, evictions, and SSD bytes per generated token. | Logical policy in engine; physical pages in adapter. |
| Prefix checkpoints that are invalidated correctly after a divergent prompt. | [tinygrad prefix logic](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/llm/model.py) | Yes, but as multi-request, trust-scoped paged KV rather than a model-global token list. | Engine cache identity and adapter page handles. |
| Stream routing that keeps reasoning and tool-call delimiters intact across token chunks. | [tinygrad server](https://github.com/tinygrad/tinygrad/blob/master/tinygrad/llm/serve.py) | Yes. Test partial delimiters and client cancellation. | Server protocol layer. |
| Routing-aware expert placement and prefetch. | [tinygrad MoE traffic issue](https://github.com/tinygrad/tinygrad/issues/17316) | Yes, only after exact routing and cold/warm SSD traces exist. | V4.1 adapter and SSD pager. |

## Explicit non-adoptions

- Do not copy tinygrad's serialized, single-model server or its model-global
  strict-prefix cache. It has no continuous batching, paged multi-tenant KV,
  fairness, or admission control.
- Do not copy AMD kernel code or extrapolate its speed figures to Apple Metal.
  The reported Qwen speedups are hardware- and GGUF-format-specific.
- Do not use tinygrad's draft DeepSeek V4 Flash pull request as a dependency;
  it is unmerged, diverged from main, and does not qualify V4.1 or Apple
  behavior.
