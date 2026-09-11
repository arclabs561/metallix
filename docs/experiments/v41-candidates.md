# V4.1 candidate-block qualification

The first executable V4.1-specific diagnostic is a CPU implementation of
`select_candidate_blocks` from the [official inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L583).
It checks the first-stage candidate mask, not index-score generation, the
second-stage Top-K, sparse attention, or a V4.1 decoder.

For each query, the reference takes each block's maximum score, pins the block
containing the newest reachable compressed position, selects the highest
scoring blocks, and drops blocks whose score remains negative infinity. Its
output covers entire blocks: future positions inside a selected partial block
can have `true` bits. The caller's causal score mask remains necessary.

`fixtures/deepseek-v41/candidate-block-reference.json` contains 11 synthetic
cases captured by executing the exact hash-pinned upstream helper with CPU
float32 Torch. No model weights are downloaded or used. The Rust fixture test
requires exact boolean masks, including newest-block pinning, partial blocks,
zero K, no reachable positions, and K exceeding the available blocks.

The diagnostic deliberately rejects NaN, positive infinity, unmasked future
scores, and finite score ties across the Top-K cutoff. The upstream helper
delegates tie selection to `torch.topk`; matching one CPU choice would not
establish a portable GPU tie policy. Ties entirely within the selected or
unselected set do not make the mask ambiguous.

```sh
mkdir -p artifacts
curl --fail --location --max-time 30 \
  --output artifacts/v41-reference-model.py \
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py
uv run scripts/v41-candidate-reference.py \
  --source artifacts/v41-reference-model.py > artifacts/v41-candidate-reference.json
cargo test -p deepseek csa2
```

The capture script checks the complete source SHA-256, extracts only the named
function, and does not import the CUDA/Triton model module. The fixture records
the source revision, hash, symbol, dtype, and Torch version. See
[third-party notices](../../THIRD_PARTY_NOTICES.md) for the upstream MIT license.

This is one operator-level parity gate. It does not satisfy the text-forward
gate for downloading the full V4.1 checkpoint, and it is not a performance or
model-quality result. Next are index-score/second-stage selection qualification
and a Metal implementation with the same numerical and masking checks.
