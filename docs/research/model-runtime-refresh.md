# Model and runtime refresh

Reviewed 2026-09-20 using primary publisher pages through Firecrawl. These are
integration candidates, not claims of native Metallix support.

| Candidate | Relevance | Next gate |
| --- | --- | --- |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Dense, non-thinking Qwen3 with 36 layers and 32 query / 8 KV heads; matches the existing architecture family | Pinned `cdbee75f17c01a7cc42f958dc650907174af0554` passes the full-logit reference and all 12 bounded native read-tool trials; longer-context and coding qualification remain separate gates |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Open-weight hybrid GatedDeltaNet/GatedAttention model; materially different from the existing dense Qwen3 control | Verify checkpoint layout and hybrid state semantics before an adapter decision |
| [vLLM-Metal](https://docs.vllm.ai/projects/vllm-metal/en/latest/supported_models/) | Publisher explicitly lists Qwen3.5/3.6/3.8 hybrid SDPA + GDN support; prefix caching remains experimental | Run a matched prompt/tool workload in an independently owned process; qualify recurrent-state restoration separately |
| [Whallm](https://github.com/yanun0323/Whallm) | Publisher documents DeepSeek-V4.1, SSD offload, Responses, tools, and Codex integration | Inspect layout/protocol choices and reproduce local latency, process memory, and task success before adopting performance claims |
| [Qwen3.8-Flash-Next](https://qwen.ai/blog?id=qwen3.8-flash-next) | Publisher describes GDN + Qwen Sparse Attention with a separate N-gram component | Treat QSA/indexer and N-gram integration as a separate architecture track; no assumed compatibility with dense Qwen3 or the 27B hybrid |

## Newly salient Hub candidates

Pinned revisions below are live Hub metadata observed on 2026-09-20. License
labels are publisher-declared Hub metadata, not a full legal audit; memory,
speed, and tool claims are publisher claims until reproduced.

| Candidate | Pin and publisher-declared license | Load shape and native fit | Priority |
| --- | --- | --- | --- |
| [Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | `3f926b415992eaa2ae9dd7b573706494d6bbf787`; Apache-2.0 | MLX safetensors with custom `prism_hadamard_qwen35` format: 64-layer Qwen3.8-derived hybrid linear/full attention, ternary weights plus FP16 group scales. The card also points to GGUF for llama.cpp. It adds a custom codec/kernel problem to the existing hybrid path. | P4: compression reference after hybrid correctness, not a first loader |
| [Xing4.0-29B-A4B](https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B) | `baae3c3e813cad5f888f1f485cfff659c89076c5`; Apache-2.0 | 40-layer custom-code Transformers model; publisher describes 29B total/4B active with mHC, MLA, and MTP. Its repository contains custom configuration, modeling, and tokenization code plus 41 safetensor shards. | P5: separate custom MoE/MLA/MTP adapter, not dense-Qwen-compatible |
| [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | `12a3808a956f869c767195e9266b59c4d21d92e2`; Apache-2.0 | Standard `LlamaForCausalLM`: 42 layers, GQA 16 query/2 KV heads, 131072 context. Canonical weights are BF16 safetensors; publisher links MLX 4-bit and GGUF derivatives and describes XML-to-OpenAI tool-call conversion in SGLang. | P2: independent compact GQA/tool control; validate template and tool parser locally |
| [Edge0-35B-A3B-preview](https://huggingface.co/Edge0/Edge0-35B-A3B-preview) | `e21098e7faa916a00f5493795f12f20497d032bb`; Apache-2.0 | MLX 4-bit `qwen3_5_moe`, 40-layer GQA 16 query/2 KV heads, based on Qwen3.6-35B-A3B, with separate LoRA and pre-router tensors. Publisher claims under 3 GiB active memory and 15 tok/s, and labels it preview/not agent-optimized. | P3: inspect sparse/offload scheduling only after dense and hybrid controls |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`; MIT | Custom `glm5_next` Transformers FP8 conditional-generation model, 45 layers/64 query/64 KV heads. Publisher describes 320B total/18B active with sparse-plus-linear attention and mHC; its Hub task is image-text-to-text. | P5: large custom multimodal hybrid; outside the immediate text-native path |

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
text/function subset. Resident admission defaults to 2048 tokens and can be
explicitly raised to 16384 with a sufficient logical K/V budget. This ceiling
and the lack of custom grammar tools do not establish general Codex support.
A controlled Codex CLI run passed three fresh read-command/final-answer trials
at the expanded limit. Broader coding tasks, custom tools, cancellation, and
ordinary profile configuration still need qualification; see the
[progress record](../progress.md).
