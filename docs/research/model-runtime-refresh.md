# Model and runtime refresh

Reviewed 2026-09-19 using primary publisher pages through Firecrawl. These are
integration candidates, not claims of native Metallix support.

| Candidate | Relevance | Next gate |
| --- | --- | --- |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Open-weight hybrid GatedDeltaNet/GatedAttention model; materially different from the existing dense Qwen3 control | Verify checkpoint layout and hybrid state semantics before an adapter decision |
| [vLLM-Metal](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md) | Publisher lists Qwen3.8 support; useful local serving comparator | Run a matched prompt/tool workload in an independently owned process; qualify cache behavior separately |
| [Whallm](https://github.com/yanun0323/Whallm) | Publisher documents DeepSeek-V4.1, SSD offload, Responses, tools, and Codex integration | Inspect layout/protocol choices and reproduce local latency, process memory, and task success before adopting performance claims |
| [Qwen3.8-Flash-Next](https://qwen.ai/blog?id=qwen3.8-flash-next) | New model architecture with a separate N-gram component | Track official checkpoint availability and hardware requirements; no assumed compatibility |

The immediate path remains native Qwen3 as a small control for chat templates,
tool round trips, and serving measurements, alongside source-grounded DeepSeek
forward integration. A successful protocol test on the control does not qualify
a larger model or establish coding quality.

## Performance decisions

Keep model loading, prefill, decode, request wall time, first visible token, and
process memory separate. Retained weights remove repeated loading but do not
imply faster kernels. Use the [chat performance ledger](../experiments/chat-performance.md)
for measured results. Compare task completion as well as tokens per second.

Before expanding native model support, compare the cost of an adapter against
using an existing local runtime behind the same client contract. Keep native
kernel work justified by an observed bottleneck, numerical parity, and a
repeatable before/after measurement.

## Codex qualification

The current [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
uses the Responses wire API. The experimental Metallix endpoint implements a
text/function subset, but its 512-token budget and lack of custom grammar tools
prevent calling it a general Codex backend. Qualify longer-context execution,
actual client event consumption, tool execution/results, cancellation, and
multi-turn task completion before enabling a personal profile.
