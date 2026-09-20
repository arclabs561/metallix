# Hugging Face trending landscape

Observed 2026-09-20. This is an architecture and integration survey, not a model
quality ranking or a claim of native Metallix support.

## Coverage

The [trending model listing](https://huggingface.co/models?sort=trending)
exposes pages `p=0` through `p=99`. All 100 pages were captured: 30 entries each,
3,000 entries and 3,000 distinct repository IDs. Page 99 has no pagination link
beyond 99. This covers the navigable trending list, not the complete Hub catalog.
The snapshot spans 16:44:29–16:45:48 UTC. Three temporary rate-limit responses
were retried successfully; the final page receipts are complete.

The ignored local evidence is `artifacts/hf-trending-2026-09-20/`: individual
page JSON, URL, retrieval time, HTML hash, embedded listing metadata, pagination
links, a combined model inventory, and aggregate summary. Ranking can change
during collection. Every listing was inspected structurally; representative
model cards/configurations were researched separately. This is not a deep read
or license audit of 3,000 model cards.

| Declared task | Repositories |
| --- | ---: |
| Text generation | 842 |
| Image + text to text | 557 |
| Text to image | 140 |
| Text to speech | 88 |
| Automatic speech recognition | 87 |
| Text to video | 70 |
| Image to image | 67 |
| Any to any | 54 |
| Time-series forecasting | 18 |
| Robotics | 17 |
| No task supplied in the listing | 584 |

These are exact values of the listing's `pipeline_tag`, not manually inferred
architectures. Other task categories account for the remaining entries. Missing
metadata is not evidence of absent capability. By repository-name substring
only, 606 entries mention GGUF and 93 mention MLX; those are discovery hints,
not verified format or runtime-support counts. Derived checkpoints, quantized
exports, adapters, and originals all count as separate repositories here.

## What changes the engineering priorities

The modern landscape is broader than decoder-only chat. It contains hybrid
attention/recurrent graphs, sparse experts, ultra-low-bit encodings, encoders,
codec pipelines, iterative denoisers, and action/numeric outputs. A tokenizer,
KV cache, or temperature setting is not a universal model interface.

An MLX-tagged artifact may still require a new architecture and custom kernels.
Bonsai's ternary artifact is a concrete example: its MLX format does not make it
a drop-in dense-Qwen checkpoint. Likewise, a small active-parameter count does
not imply that a sparse model's complete weights fit in memory. The detailed
text/hybrid triage and immutable config pins are in the
[model/runtime refresh](model-runtime-refresh.md).

## Representative non-text contracts

License labels below distinguish card metadata from inspected license text.
Actual artifact and dependency terms must be retained before redistribution.
No model weights were acquired for this survey.

| Representative | Contract that a reusable adapter must preserve | Integration judgment |
| --- | --- | --- |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Text-to-phoneme preprocessing, voice conditioning, duration/alignment and vocoder; 24 kHz audio. Card labels Apache-2.0; its listed weight files did not include a separate LICENSE. | Small useful speech vertical with [an existing MLX conversion](https://huggingface.co/mlx-community/Kokoro-82M-bf16). Verify preprocessing and voice provenance as well as tensors. |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | Audio frontend at 16 kHz, mel features, encoder-decoder state, temperature fallback and timestamps. | Clear speech-recognition reference with [an MLX artifact](https://huggingface.co/mlx-community/whisper-large-v3-mlx); larger than Kokoro and requires segment/state semantics. HF card labels Apache-2.0; retain the precise artifact terms. |
| [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) | Separate symbolic/semantic sampling and VAE/ODE audio synthesis, not a single token loop. | Relevant to composite sampling, but its [weight license is CC BY-NC 4.0](https://huggingface.co/m-a-p/YuE2-3B/raw/main/LICENSE), distinct from code terms. |
| [Tencent AuK](https://huggingface.co/tencent/AuK) | Diffusion speech generation/editing, auxiliary text encoder and audio VAE. | Multi-stage target after a simpler audio contract; [MIT text explicitly includes code and weights](https://huggingface.co/tencent/AuK/raw/main/LICENSE). |
| [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) | A small deterministic image encoder producing class, patch, and register features with explicit patch geometry. | Strong compute/vision test case; gated DINOv3-specific terms precede acquisition. No sampling loop is needed. |
| [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | Prompt/image conditioning, DiT, VAE, flow-matching schedule and image artifacts. | Useful future generative-image vertical, not a text adapter extension. Qwen Research License, not an assumed Apache license from the publisher name. |
| [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5-Diffusers) | Video/audio latents, denoising scheduler, upsampling and modality-specific output timing. | Substantial later target. Gated LTX community terms require inspection for the intended use. |
| [SAM 3.1](https://huggingface.co/facebook/sam3.1) | Text/point/box prompts, masks and multi-object video-tracking state. | Custom state/output contract; gated, nonstandard license. A generic decoder cache would be the wrong abstraction. |
| [Chronos-2](https://huggingface.co/amazon/chronos-2) | A 120M numeric forecaster with patched sequences, covariates/grouping and quantile outputs. HF metadata declares Apache-2.0. | Attractive small non-text compute consumer; validates numeric shapes and units independently of token generation. |
| [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch) | Patched numeric forecasting with quantile outputs. | Architecturally useful contrast, but the [non-commercial license](https://huggingface.co/google/timesfm-3.0-pytorch/resolve/main/LICENSE) rules out treating it as a permissive serving default. |
| [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) | Retrieval prompts, pooling, embedding dimensions and re-normalization. | Useful embeddings contract; gated Gemma terms. It remains a text-input task even though its output is not text. |
| [SmolVLA](https://huggingface.co/lerobot/smolvla_base) | Images, language and numeric robot state; flow-matching action chunks and state/action normalization. HF metadata declares Apache-2.0. | Demonstrates why model state, units, and sampling schedules must be task-specific; later than a bounded encoder or forecaster. |

These examples support several first non-text choices rather than a premature
universal API: Chronos-2 for small numeric inference, Kokoro for useful audio,
or DINOv3 for a small vision encoder after access/terms are resolved. The first
consumer should be selected for its checkable reference, workload value, and
measured resource cost. Existing Qwen and DeepSeek completion remains ahead of
adding a large new generative pipeline.

## Uncensored and NSFW-oriented variants

The supplied [full-text search](https://huggingface.co/search/full-text?q=uncensored)
matches model cards, datasets, and Spaces, including older releases; it is not
a newest-model ranking. The relevant engineering distinction is base
architecture, artifact encoding, and adaptation provenance. An "uncensored",
"abliterated", or NSFW label is a publisher description of behavior, not a
kernel format or a verified quality property.

| Variant | Verified artifact/provenance observations | Adapter implications |
| --- | --- | --- |
| [Huihui Qwen3.8-27B GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | Qwen3.8 base and Apache-2.0 metadata; GGUF variants, optional MTP, multimodal projector. Pin `3d5cf9efbd7cd22926e648a7c1cbaa001fc29f54`. | Hybrid Qwen3.8 execution, GGUF codec and optional projector/draft support are separate gates. |
| [HauhauCS Aggressive MTP](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF) | Qwen3.8/Apache-2.0 metadata; GGUF, projector, draft sidecar. Card requires a supplied llama.cpp patch. Pin `993a5971fda8f30dd1b7eb2654792ba4415c7460`. | Reproduce the patched draft semantics before attributing any advertised speed to a standard MTP implementation. |
| [DeepSeek-V4.1 Flash uncensored FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8) | DeepSeek base/MIT metadata; 48 safetensor shards. Publisher describes weight-level alteration and a multi-GPU deployment. Pin `d61c59ea5e514e25d305b5850e8a432f7a9969f2`. | Shares the independently planned DeepSeek architecture work; no native full-checkpoint or Apple-memory qualification follows from the variant label. |
| [DavidAU Qwen3.8 Fable/Cold Fusion](https://huggingface.co/DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF) | Card describes a modified Qwen3.8-derived base and multi-stage merge/training; Apache-2.0 metadata, GGUF/MTP/projectors. Pin `c02caef111a8acf987947f35e1e288aa5450e184`. | Track the complete base/merge chain and evaluate quality independently; the publisher's method names do not define an execution architecture. |
| [HiDream uncensored](https://huggingface.co/e-n-v-y/hidream-uncensored) | HiDream-I1-Dev fine-tune/MIT metadata; FP8 diffusion safetensors. Pin `5f1188ae804841b80399a24e61000c5726b9fd6a`; the card does not specify the modification method. | Belongs to the planned image-generation adapter lane, with its own conditioning, scheduler and VAE requirements. |

These licenses are card/Hub declarations, not a completed audit of every base,
merge, projector, or bundled code dependency. Treat provenance and behavioral
evaluation separately from loader compatibility. A compatible fine-tune should
usually reuse its base adapter after configuration and artifact validation,
without adding a new runtime architecture solely for its behavior label.

## Consequences for Metallix

The [adapter/config proposal](../model-adapters.md) separates shared lifecycle,
loading/provenance, admission, profiling and sampling policy from model-owned
preprocessing, graphs, caches and output contracts. Compute functionality can
expose MLX deliberately without making every task look like chat.

SMC needs a specified target/proposal and coherent state ancestry; diffusion
schedules, categorical token sampling and forecast quantiles are distinct
objects. LoRA/training additionally require parameter/adapter identity,
gradients, optimizer state, deterministic resume and export semantics. These
requirements belong in the design now, with implementation ordered behind
existing-model correctness and profiling.

Performance opportunities should follow the contracts: eliminate avoidable
host synchronization, use compiled/fused shape-stable graphs where supported,
qualify low-bit codecs independently, retain reusable prefixes/encoder outputs,
and measure conditional expert residency. Compare complete task cost and
quality with current specialized runtimes. A trending position, model-card
throughput number or successful load establishes none of those results.
