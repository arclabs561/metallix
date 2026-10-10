# Gemma 4

The `gemma` crate runs the dense text path of Google's Gemma 4 on Metal:
interleaved sliding-window and full-attention decoder layers with a cached
decoder. It is an adapter-owned crate like `qwen`, `deepseek` and `julia`:
Gemma shares no tensors, cache layout or quirks with Qwen3, and keeping it
separate lets each adapter change its graph without touching the other's
qualified path. `mx serve` loads it as the registry kind `gemma4`; see
[serving](#serving).

## Checkpoints

| Checkpoint | Revision | Text layout | Status |
| --- | --- | --- | --- |
| [gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | 48 layers (40 sliding, 8 full), hidden 3840, GQA 16/8 sliding and 16/1 full | Development checkpoint; parity below |
| [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | `842da3794eaa0b77d5f08bae87a17459d91ff475` | 60 layers (50 sliding, 10 full), hidden 5376, GQA 32/16 sliding and 32/4 full | Final qualification |
| [gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | 35 layers (28 sliding, 7 full), hidden 1536, GQA 8/1, last 20 layers share K/V, 256-wide per-layer input | Parity below |
| [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | `ee0ef6023621cff504d758262d4e04895a5af4a2` | 42 layers (35 sliding, 7 full), hidden 2560, GQA 8/2, last 18 layers share K/V, 256-wide per-layer input | Parity below |

Both are Apache-2.0 and not gated (live Hub metadata, 2026-10-05). They share
the layer pattern (five sliding layers, then one full), a 1024-token window,
K-as-V on full layers, a 512-wide global head and the same tokenizer and chat
template bytes. The 12B declares `model_type` `gemma4_unified` and the 31B
`gemma4`; transformers v5.18.0 computes the same text forward for both.
E2B and E4B (`gemma4`, also Apache-2.0) add a per-layer input to every layer,
let their last 20 (E2B) or 18 (E4B) layers attend over an earlier layer's K/V
instead of their own, use a 512-token window and full layers with their own
`v_proj`, and on E2B double the MLP width in the sharing layers. They share
one chat template, which differs from the 12B one only in not pre-filling an
empty thought channel when thinking is off. The adapter refuses 26B-A4B,
whose mixture-of-experts block it does not implement yet.

## Numerics

From `modeling_gemma4.py`, `modeling_gemma4_unified.py` and
`modeling_rope_utils.py` at transformers v5.18.0
(`a906d3c4b65095f2308b6a6a193e934d03b8eb5d`):

| Quirk | Gemma 4 | Qwen3 adapter |
| --- | --- | --- |
| Embedding | times `sqrt(hidden_size)` cast to the weight dtype (62.0 for 3840 in bf16, 61.97 in f32) | unscaled |
| `RMSNorm` | `x * w` | `x * w` (Gemma 2/3 used `1 + w`) |
| Norm placement | input norm, then norms on the attention output and on the MLP output before each residual add, plus the pre-MLP norm | pre-norm only |
| Layer output | whole layer output, residual included, times the stored `layer_scalar` | none |
| Q/K/V norms | per-head Q and K norm with weights; V norm without a weight | Q and K norm |
| Attention scale | 1 | `1/sqrt(head_dim)` |
| Full layers | head 512 (sliding 256), own K/V head count, no `v_proj`: the raw K projection is V | one geometry |
| `RoPE` | sliding: standard, theta 10000; full: "proportional", theta 1e6, exponents `2i/512` for the first 64 of 256 pairs, pairs `(i, i+256)`, other pairs unrotated | standard |
| Sliding mask | query `q` sees keys `q - 1024 < k <= q` | causal |
| Logits | tied embedding, then `30 * tanh(logits / 30)` | tied, uncapped |
| MLP | `down(gelu_tanh(gate(x)) * up(x))` | SiLU |
| Per-layer input (E2B, E4B) | `(rms_norm(embeds @ P / sqrt(H)) + table[ids] * sqrt(256)) / sqrt(2)`, one 256-wide slice per layer; each layer adds `norm(proj(gelu_tanh(gate(x)) * slice))` after its MLP, before `layer_scalar` | none |
| K/V sharing (E2B, E4B) | a sharing layer projects only Q and attends over the K/V that the last earlier layer of its kind attended over in the same step | none |

Proportional `RoPE` maps onto MLX `fast::rope` over the whole head with
explicit wavelengths (`theta^(2i/512)` for rotating pairs, infinity for the
rest), the same construction as MLX-LM's `ProportionalRoPE`.

## Cache and memory

Sliding layers keep only the 1023 positions before the next query; full layers
keep every position. The retained K/V after `n` tokens is therefore
`sliding_layers * 2 * kv_heads * 256 * min(n, 1023) + full_layers * 2 *
global_kv_heads * 512 * n` elements. In bf16 the sliding layers cost 320 KiB
per token up to the window (12B) and the full layers 16 KiB per token after
it; for 31B the figures are 800 KiB and 80 KiB.

| Retained K/V at 8192 tokens, bf16 | Windowed | Every position kept |
| --- | --- | --- |
| 12B | 448 MiB | 2688 MiB |
| 31B | 1439 MiB | 7040 MiB |
| E2B | 54 MiB | 144 MiB |
| E4B | 148 MiB | 448 MiB |

`Gemma4TextConfig::retained_kv_elements` computes this. A synthetic test checks
it against the bytes the executor holds, and the 12B f32 parity run held
exactly the predicted 720,371,712 bytes after 1524 tokens.

These are logical bytes. A trimmed sliding cache is an MLX view of the buffer
its last append built, so right after a long prefill that buffer still holds
up to 1023 + 2048 positions per sliding layer; the next append copies only
the kept rows. Each append concatenates new K/V onto the cache, so a decode
step copies the retained K/V once. That cost is bounded for sliding layers by the window and
grows linearly for full layers. Prompts longer than 2048 tokens are prefilled
in 2048-token chunks so the unfused attention for 256- and 512-wide heads never
materializes a full `[heads, prompt, prompt]` score matrix.

## Reference and parity

`scripts/gemma4-reference.py` loads the checkpoint with transformers
`AutoModelForCausalLM` (CPU, eager attention), prefills three prompts with
the source K/V cache and decodes six greedy tokens one at a time, writing
every step's full-vocabulary logits:

- `raw_short`: "The capital of France is" with `<bos>` and no template.
- `chat_turn`: one templated user message, 20 tokens.
- `long_window`: a templated 1518-token prompt, so prefill and every decode
  step cross the 1024-token window. The source continues it with "The survey
  team conducted 40", which is correct (the prompt has 40 entries).

It also records the source chat-template rendering and token IDs for six
conversations, including thinking, a tool declaration and a tool call with
its response. A second, bfloat16 capture fed the same decode tokens measures
the source's own rounding noise; `--fixture` combines both captures into
[`fixtures/gemma-4-12b/reference.json`](../fixtures/gemma-4-12b/reference.json),
which keeps each step's top-8 logits and drops the full-vocabulary sidecars.
The float32 capture needs about 48 GB of memory and refuses checkpoints whose
float32 copy would be much larger; capture larger models in bfloat16.

```sh
uv run scripts/gemma4-reference.py --model <12B dir> --output <f32 dir>
uv run scripts/gemma4-reference.py --model <12B dir> --output <bf16 dir> \
  --dtype bfloat16 --decode-from <f32 dir>/manifest.json \
  --fixture fixtures/gemma-4-12b/reference.json
METALLIX_GEMMA4_MODEL=<12B dir> METALLIX_GEMMA4_LOGITS_DIR=<f32 dir> \
  cargo test -p gemma --features metal -- --ignored --nocapture
```

`METALLIX_GEMMA4_LOGITS_DIR` is optional; without it the parity test compares
the reference's top-8 logits. `METALLIX_GEMMA4_PRECISION=bf16` runs the bf16
path.

12B, 21 steps (three prefills and 18 cached decode steps) plus one prefill
split across `extend`, full vocabulary, on an Apple-Silicon Mac (128 GiB),
transformers 5.18.0 and torch 2.13.0 on the CPU:

| Compared with the float32 source | Worst max abs logit difference | Argmax agreement |
| --- | --- | --- |
| Metallix f32 | 7.5e-4 | 22 of 22 |
| Metallix bf16 | 1.65 | 21 of 22 |
| Source bf16, same tokens | 2.78 | 18 of 21 |

E2B and E4B, same three prompts (the long one, 1514 tokens, crosses their
512-token window), the same bounds:

| Compared with the float32 source | E2B | E4B |
| --- | --- | --- |
| Metallix f32, worst / argmax | 2.1e-4, 22 of 22 | 7.8e-4, 22 of 22 |
| Metallix bf16, worst / argmax | 0.95, 22 of 22 | 0.98, 22 of 22 |
| Source bf16, worst / argmax | 1.58, 21 of 21 | 1.36, 21 of 21 |

[`fixtures/gemma-4-e2b/reference.json`](../fixtures/gemma-4-e2b/reference.json)
is the E2B fixture; pass it as `METALLIX_GEMMA4_REFERENCE` with the E2B
checkpoint. A 1514-token float32 prefill is one GPU command buffer of about
0.9 s on an idle machine; under heavy GPU load macOS has killed it
(`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`), so run the f32
parity on a quiet machine.

[`fixtures/gemma-4-tiny/ple-reference.json`](../fixtures/gemma-4-tiny/ple-reference.json),
from `scripts/gemma4-tiny-reference.py`, is a tiny random-weight model with
every E2B/E4B feature, run through transformers' own `Gemma4ForCausalLM`; the
always-on test holds Metallix within 1e-4 of its float64 logits. The source's
own float32 output lands up to 4.1e-5 from float64 on it.

Logits are soft-capped to [-30, 30]. Every argmax flip, Metallix's and the
source's, sits where the float32 top-2 margin (0.009 to 0.60) is smaller than
the source's own bf16 distance at that step. The parity test therefore holds
f32 to 0.01 with every argmax equal, and holds bf16 at each step to twice the
source's bf16 distance there, with the argmax required to match wherever the
float32 margin exceeds that distance. The factor of two was set from these
same comparisons (worst observed ratio 1.57); no held-out prompt informed it.

The template renders identically with minijinja 2.24 and Python-method
compatibility, the server's template engine, when `bos_token` is in the
context; [`fixtures/gemma-4-12b/chat-template.jinja`](../fixtures/gemma-4-12b/chat-template.jinja)
is the 12B template, byte-identical to the 31B one, so that check runs without
the checkpoint. The tokenizer reproduces the source token IDs. Its
post-processor adds no special tokens, so a prompt rendered without
`bos_token` starts at `<|turn>` whichever way it is encoded.

## Serving

A registry entry `{"id": "gemma", "kind": "gemma4", "path": "<checkpoint>"}`
serves generation through the same protocol code as Qwen (`/v1/responses`,
`/v1/chat/completions`, `/v1/messages`); it does not serve `/v1/decisions`. Weights and K/V are bf16. The session checks the
configuration and the K/V plan (sliding layers counted at their window)
against `--context-tokens` and `--kv-budget-mib` before loading any weights.

The checkpoint's chat format supplies the rest: the template, with
`bos_token` in its context; the stop set `<eos>`, `<turn|>` and
`<|tool_response>`, the last ending a turn that called tools; the
checkpoint's suppressed tokens (the 12B lists `<audio|>` and `<image|>`, the
31B none); tool calls in Gemma's own syntax,
`<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`; and reasoning in
`<|channel>thought ...<channel|>`, which is stripped from the answer even
with thinking off, since the model writes an empty channel after tool
results.

An ignored server test checks one 12B session end to end over
`/v1/chat/completions`: a greedy answer, the same text streamed, and a tool
call whose arguments equal the transformers float32 greedy call for the same
prompt and tool. Thinking mode and the other two protocols are not exercised
against the checkpoint.

## Follow-ups

- Prompt-prefix reuse across turns and GPU-side token picks, which the Qwen
  session has: every Gemma turn prefills its whole prompt and reads each
  step's full logit row.
- Replace per-step concatenation with preallocated capacity, as the Qwen
  resident cache does, before long-context decode measurements.
