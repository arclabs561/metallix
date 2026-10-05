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

## October prioritization

Re-checked 2026-10-01 against live Hub API metadata (trending, all-time and
30-day likes, and MLX-tagged listings) and each candidate's `config.json`.
Download counts are cumulative and favor older models; trending reflects the
last week. Priority weighs a concrete local consumer, fit on one Mac, and how
much each model reuses or extends what Metallix already qualifies.

| Model | Architecture (config) | Checkpoint / license | Reuse and new work | Priority |
| --- | --- | --- | --- | --- |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `qwen3_5`, 64 layers: 48 linear-attention (GatedDeltaNet) + 16 full; GQA 24/4 | 55.6 GB BF16, Apache-2.0; most-liked trending open model (6.9M downloads) | New recurrent-state layer and cache semantics; dense, fits in memory at BF16 | **P1**: the most-used modern local model, and the hybrid-state design every newer Qwen variant builds on |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | `gemma4`, 60 layers: 50 sliding (window 1024) + 10 full; GQA 32/16 | 62.5 GB, Apache-2.0; 9.8M downloads | Sliding/global KV and a second independent tokenizer/template; no recurrent state | **P2**: a large-user-base dense model that exercises the attention-window path without new state types |
| [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | Plain `LlamaForCausalLM`, 42 layers, GQA 16/2 | 5.0 GB BF16, Apache-2.0; 0.9M downloads in weeks | Runs on the Qwen3 dense decoder without Q/K norms and with an untied `lm_head`; XML tool calls | **P2**: pinned `f97400052a43` matches the CPU float32 transformers oracle (48/48 argmax, max abs logit error 1.3e-4) and serves chat and tool calls through `mx serve`; tool-call quality is not yet qualified |
| [MiMo-V2.6-Flash](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL) | `mimo_v2`, 48 layers, 256 experts / top-8, sliding window 128, FP8 128x128 blocks | 177.7 GB FP8, MIT | Shares FP8 block decoding, MoE routing and windowed attention with DeepSeek; larger than RAM like DeepSeek | **P3**: the natural second larger-than-RAM MoE once DeepSeek's offload path works |
| [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `qwen4_exp`, 48 layers (36 linear + 12 full), 512 experts / top-10 | 360 GB, non-standard license | Combines Qwen3.8's hybrid state with sparse attention and very wide MoE | **P4**: after Qwen3.8-27B; license needs review |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | `glm5_next`, 45 layers (34 linear + 11 DeepSeek-style sparse attention), 288 experts, FP8 | 328 GB, MIT | Reuses DeepSeek sparse-attention work plus linear attention | **P4**: valuable only after both DeepSeek and Qwen3.8 hybrids work |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | `kimi_k3`, 93 layers, 896 experts | compressed-tensors; very large | Exceeds a single Mac's SSD-streaming budget for useful speed | Deferred |
| [Ternary-Bonsai-2-27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) | Qwen3.8-27B layout with ternary + Hadamard weights | 2-bit MLX, Apache-2.0; top of trending | A codec on top of Qwen3.8-27B | Follows Qwen3.8-27B; reuse its hybrid adapter |

Two architecture families dominate current demand: hybrid linear/full
attention (Qwen3.8, GLM-5.3, Qwen3.8-Flash-Next, Bonsai) and large FP8 MoE
(DeepSeek, MiMo, GLM-5.3). DeepSeek work covers the second family's FP8 and
routing pieces; recurrent hybrid state is the main missing capability, which
is why Qwen3.8-27B ranks first among new models. Gemma 4 and MiniCPM5 are the
lowest-cost checks that adapters generalize beyond the Qwen control.

## Performance decisions

### September 30 implementation check

A fresh check found an important qualification boundary in
[MLX-LM at `a9bd8af`](https://github.com/ml-explore/mlx-lm/blob/a9bd8af5c02118882af735cef60705d2efce9fd0/mlx_lm/models/deepseek_v41.py):
its `deepseek_v41` adapter declares the target's 5120-wide, 40-layer,
384-expert geometry, but `ModelArgs.__post_init__` rejects nonempty
`engram_layer_ids` because Engram is unimplemented. The adapter name therefore
does not establish support for the pinned complete checkpoint.

[Whallm at `eba1f5b`](https://github.com/yanun0323/Whallm/tree/eba1f5b6b9f7f0b53fac45b13fd916767191a4a6/runtime/deepseek_v4_ssd/deepseek_v41)
has a separate V4.1 implementation, including Engram, and its
[tests](https://github.com/yanun0323/Whallm/blob/eba1f5b6b9f7f0b53fac45b13fd916767191a4a6/runtime/tests/test_deepseek_v41.py)
exercise selected SSD Engram rows and reduced model loading. This source
inspection did not execute those tests or reproduce checkpoint quality,
latency or memory use. It remains a concrete upstream comparison candidate.

Its [route recorder](https://github.com/yanun0323/Whallm/blob/eba1f5b6b9f7f0b53fac45b13fd916767191a4a6/runtime/deepseek_v4_ssd/route_trace.py)
separates prefill histograms from exact per-layer decode routes. Histograms
cannot recover chronological accesses for an exact LRU replay. Any imported
trace must retain checkpoint identity, phase and execution ordering; the
recorder alone does not supply source-compatible routing evidence. Obtaining
real routes requires actual forward execution or a separately qualified
capture, not metadata alone.

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
