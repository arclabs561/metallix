# GGUF checkpoints

Many Hub repositories publish a model only as GGUF files, the container
llama.cpp reads. Metallix reads GGUF directly: the `checkpoint` crate parses
the file, `blockfloat::gguf` names and decodes its tensor encodings, and a
model adapter maps the tensors onto the same graph it runs for the Hugging
Face checkpoint.

## What loads

| Adapter | GGUF architecture | Encodings | Entry point |
| --- | --- | --- | --- |
| `qwen35` (dense) | `qwen35` | F32, F16, BF16, `Q8_0` | `Qwen35Weights::load_gguf` |

A file with any other encoding (`Q4_K`, `Q6_K`, `IQ4_XS`, ...) is refused by
name before weights are allocated. Mixture-of-experts files (`qwen35moe`) are
refused too. A file named `Q4_K_M` mixes several encodings (`Q4_K`, `Q5_K`,
`Q6_K`, `Q8_0`, F32), so it needs all of them.

## Values are the file's

`Q8_0` stores 32 signed bytes `q` and one binary16 scale `d` per block; a
value is `d * q`. MLX's affine quantization stores `scale * code + bias` per
group, so a block becomes 8-bit codes `q + 128`, scale `d` and bias `-128 d`,
in groups of 32. Every product and sum there is exact in `f32`, and the bias
is exact in binary16 unless it overflows, in which case the tensor's
parameters are kept in `f32`. The loaded weights therefore equal the file's
values bit for bit; only the matrix arithmetic differs from llama.cpp's.
MLX's own GGUF loader keeps `Q8_0` the same way but expands other encodings
to F16 and BF16 to F16, so Metallix uses its own reader.

F16 and BF16 tensors keep their dtype. Activations are BF16 at checkpoint
precision.

## What the converter changed

llama.cpp's converter (`conversion/qwen.py` at
[7fe450e](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/conversion/qwen.py))
renames tensors and changes some values. The `qwen35` loader maps them back:

| GGUF | Stored as | Loader |
| --- | --- | --- |
| `*norm.weight` (not `ssm_norm`) | `1 + w` | used as is (the Hugging Face path computes `1 + w` itself) |
| `ssm_a` | `-exp(A_log)` | decay rate `exp(A_log)` is `-ssm_a`, no round trip through `log` |
| `ssm_conv1d.weight` | `[channels, K]` | transposed to `[K, channels]` |
| value-head tensors when value heads outnumber key heads | heads tiled (`v0` of every key head, then `v1`, ...) | put back in grouped order (key head `k` serves value heads `k r .. (k + 1) r`) by moving whole rows or column blocks, which is exact for every encoding |

Configuration comes from the `qwen35.*` keys (layer `i` is full attention
when `(i + 1) % full_attention_interval == 0`, as in
[`src/models/qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/src/models/qwen35.cpp)).

## Tokenizer and chat template

A GGUF file embeds the vocabulary, merges and chat template, but the
pre-tokenizer split rule is only a name (`tokenizer.ggml.pre`) for code inside
llama.cpp. `ChatFormat::from_gguf` therefore takes the Hugging Face checkpoint
the file was converted from (the file usually names it in
`general.base_model.0.repo_url`, without a revision), uses its
`tokenizer.json` only if every token spelling and merge equals the file's,
and renders the template the file embeds, ending turns at the file's
`tokenizer.ggml.eos_token_id`. Quantizers sometimes edit the template, so it
may differ from the base repository's.

## Verification

`scripts/gguf-oracle.py MODEL.gguf OUT` runs llama.cpp on the same file: for
three fixed prompts it writes every position's logits (`llama-results`) and 64
greedy tokens (`llama-server`). The opt-in test
`crates/models/qwen35/tests/gguf_checkpoint.rs` then compares, with bounds
fixed before the first run:

- tokenization: the chat format built from the file encodes each prompt to
  llama.cpp's token IDs;
- teacher-forced logits at every position: top-1 agreement at least 98%,
  mean KL(llama.cpp || metallix) at most 1e-3 and maximum at most 1e-2, and
  top-1 differs only where llama.cpp's own top-1/top-2 margin is at most 0.10;
- chunked prefill against token-by-token decode at the last position;
- greedy decode, diverging only at a near-tie;
- resident weight bytes (`Q8_0` at 9 bits per weight) and peak memory during
  load;
- every `Q8_0` matrix, dequantized by MLX, bit-equal to the reference
  decoder.


Measured 2026-10-09 on an M3 Max (unsloth Qwen3.5 GGUFs, `Q8_0`, llama.cpp
b11146 as the oracle, 1,178 positions per model):

| Model | Top-1 agreement | Mean KL | Max KL | `Q8_0` values bit-equal |
| --- | --- | --- | --- | --- |
| Qwen3.5-0.8B | 98.73% | 4.6e-4 | 3.4e-3 | 751,894,528 of 751,894,528 |
| Qwen3.5-4B | 98.47% | 3.8e-4 | 4.2e-3 | 4,204,789,760 of 4,204,789,760 |

Parity is not yet a pass: three of the bounds fixed before the first run
are missed, and they stay as declared.

- Top-1 differs at a llama.cpp margin above 0.10 at one position on each
  model (0.1071 on 0.8B, 0.1003 on 4B).
- On the 0.8B dialogue prompt, chunked prefill and token-by-token decode
  differ by KL 1.61e-3 at the last position, against a bound of 1e-3 (the
  top-1 token is the same).

Two explanations are ruled out. The reference is not the noise: llama.cpp
run against itself on the same file (whole-prompt batches against one token
per batch) agrees at every position on 0.8B and differs once on 4B, at
margin 0.0009. The prefill bound is not below Metallix's own BF16 noise: the
Hugging Face BF16 checkpoint of the 0.8B model, loaded without any GGUF
code, stays under 1e-3 on all three prompts (dialogue 6.9e-4). Weights are
bit-exact, so the differences come from the arithmetic Metallix runs on
them; the cause is under investigation.

A copy of the 0.8B file with each layer's `ssm_alpha` and `ssm_beta`
exchanged drops to 44% agreement and mean KL 1.6, so the comparison detects
a wrong tensor mapping.
