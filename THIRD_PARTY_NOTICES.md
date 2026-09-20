# Third-party notices

`fixtures/qwen3-0.6b/chat-template.jinja` reproduces the `chat_template`
field from Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca` for local compatibility tests.
It is provided under the Apache License 2.0; a copy is in
`third_party/LICENSE-APACHE-2.0.txt`. The
[source and license](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE)
identify the applicable notice: Copyright 2024 Alibaba Cloud.

`fixtures/qwen3-0.6b/embedding-reference.json` and
`fixtures/qwen3-0.6b/forward-reference.json` are small numerical observations
generated locally from Qwen/Qwen3-0.6B. They record input IDs, selected
embedding, hidden-state, and logit values plus hashes of the local config and
checkpoint bytes. They do not contain a copied Qwen config, tokenizer payload,
checkpoint, or full-logit sidecar. The compact Qwen3-0.6B configuration literal
in `crates/models/qwen/src/forward.rs` is a parser-test layout, not a tracked
upstream config file. These Qwen-derived compatibility records are covered by
the Apache-2.0 notice and license copy above.

The candidate-block selection and FP32 index-score diagnostics in
`crates/models/deepseek/src/csa2.rs`, `indexer.rs`, and `selection.rs` follow
`select_candidate_blocks` and the score-reduction expressions in
`Indexer.forward` from DeepSeek's official V4.1 Flash inference implementation,
revision `dba1be0a40aa45a94ad051997016db3960a90277`.
The fixture capture script executes those hash-pinned operations and the final
Top-K selection expressions from `Indexer.forward` on synthetic data.
The rotary-tail diagnostic in `crates/models/deepseek/src/rotary.rs` and its
synthetic fixture follow `apply_rotary_emb` from that same revision;
`scripts/v41-rotary-reference.py` executes only that hash-pinned helper.
The frequency generator in the same Rust module follows `precompute_freqs_cis`;
`scripts/v41-rope-reference.py` captures frequencies and composed rotations
from those pinned helpers. The mathematical sparse-attention reference in
`crates/models/deepseek/src/attention.rs` follows the gather, duplicate-slot,
and denominator-only sink semantics of `inference/kernel.py:sparse_attn_kernel`
at the same revision. Its independent fixture generator does not execute the
upstream kernel or reproduce its BF16 intermediate rounding.
The scalar output composition in `crates/models/deepseek/src/attention/output.rs`
follows the inverse-rotary, grouped `wo_a`, and `wo_b` ordering in
`Attention.forward`. `scripts/v41-output-reference.py` independently captures
its grouped BF16 einsum expression on synthetic CPU tensors after checking
source identity; it does not execute the complete model or quantized kernels.
The compressed-token hash reference in `crates/models/deepseek/src/engram.rs`
follows `inference/engram.py:NgramHashState.forward` at that revision.
`scripts/v41-engram-reference.py` executes only that hash-checked method with
explicit synthetic tensors, excluding tokenizer normalization, RNG setup and
embedding-table access.
`scripts/v41-engram-gate-reference.py` captures the hash-checked
`Engram.forward` residual gate with explicit embedding/projection stubs;
it does not execute real table lookups or projection weights.
The scalar BF16 reference in `crates/models/deepseek/src/engram/gate.rs`
follows that residual expression with supplied preprojected tensors.
This is distinct from the bounded scalar selected-row lookup in
`crates/models/deepseek/src/engram/embedding.rs` and the source-forward
`scripts/v41_layer3_engram_capture.py` export. The latter writes
`fixtures/deepseek-v41/layer3-engram-reference.json`, which contains synthetic
FP8 embedding and WKV parameters and observations from the pinned source
graph, not a released checkpoint or copied source file. The layer-two FFN
export and `fixtures/deepseek-v41/layer2-ffn-reference.json` likewise record a
synthetic source-forward boundary. The layer-one ratio-two owner export and
`fixtures/deepseek-v41/layer1-ratio2-owner-reference.json` capture the same
synthetic graph's compressed KV and direct index publication. The separate
`scripts/v41_layer1_attention_capture.py` export and
`fixtures/deepseek-v41/layer1-attention-reference.json` retain the same
synthetic graph's layer-one attention projections, local window, owned
compressed KV/indices, sparse attention and output boundaries. They use the
same pinned source and locally generated parameters; no checkpoint weights
are included. The exporter checks source identity and exact tensor storage.
The separate
`scripts/v41_layer2_attention_capture.py` export and
`fixtures/deepseek-v41/layer2-attention-reference.json` retain exact observed
layer-two attention boundaries, including its borrowed layer-one KV/index
publication, local window state, and FFN handoff. The exporter verifies pinned
source/helper hashes, raw tensor storage and finite values; it performs no
attention arithmetic. `scripts/v41_layer2_hc_capture.py` and
`fixtures/deepseek-v41/layer2-hc-reference.json` retain the same synthetic
source graph's layer-one residual/pre into layer-two HC and attention boundaries,
including the historical attention/FFN handoffs. These are reduced numerical
captures, not full-model, native-attention, or serving artifacts.
The compressor and block composition captures and tests follow
`Compressor.forward`, `Block.forward`, the block HC helpers and `RMSNorm.forward`
at the same revision, with explicitly stubbed projections or sublayers.
The bounded scalar implementation in `crates/models/deepseek/src/compressor.rs`
follows that compressor pooling and normalization order with caller-supplied
projection results; it does not execute checkpoint projection weights.
The compressed-publication capture executes `Attention._compress_kv` and
`Attention._compress_topk_idxs` at that revision, with explicit compressor,
indexer, rotary and quantization stubs. Its Rust tests check captured control
flow, not the upstream numerical kernels.
The scalar FP4 activation arithmetic in
`crates/models/deepseek/src/precision/fp4_activation.rs` follows `fp4_quant_kernel` and
`fast_round_scale` in the same revision's `inference/kernel.py`. Its independent
CPU oracle uses explicit software E2M1 rounding and Torch CPU casts; it does
not execute the upstream TileLang kernel.
The compressed-attention composition oracle joins independently expressed
rotary, FP4 reconstruction and mathematical sparse attention in CPU Torch.
It follows the same pinned ordering with supplied latents and indices, not
the complete upstream attention implementation.

`scripts/v41_source_loader.py` loads the same revision's retained, hash-checked
text graph and Engram source. `scripts/v41-forward-reference.py` executes that
graph with synthetic encoded weights and independent CPU replacements in
`scripts/v41_cpu_kernels.py`. Those replacements follow the quantization,
scaled GEMM, sparse-attention and HC/Sinkhorn boundaries in `inference/kernel.py`.
They preserve explicit numerical representations but do not execute TileLang
or establish GPU reduction parity. The reduced manifest and receipts record
the synthetic configuration separately from the released checkpoint.
All tracked `fixtures/deepseek-v41/*.json` are reduced synthetic-capture
outputs and carry the pinned source revision in their `source` object. The
retained upstream source inputs live under ignored `artifacts/` during local
capture and are not committed to this repository.

`crates/models/deepseek/src/moe.rs` composes the same revision's `Gate`,
`Expert.forward` and `MoE.forward` ordering through the scalar numerical
references. Its encoded synthetic-weight oracle is extracted from actual
source-forward hooks by `scripts/v41-forward-reference.py`.

`crates/models/deepseek/src/ffn.rs` follows the same revision's `Block.forward`
FFN ordering: derive fresh HC coefficients, collapse with the incoming
attention pre-mix, normalize, execute the MoE, then apply the fresh HC post-mix.
The capture retains the source block's HC inputs, coefficients and residuals
to diagnose numerical differences; this is not a full-block parity claim.

`crates/models/deepseek/src/attention/layer.rs` composes the same revision's
`Attention.forward` query, window-cache, sparse-attention and output-projection
ordering. Compressed numerical KV and the consumer's selected indices remain
caller-supplied. `scripts/v41_attention_capture.py` extracts observed layer-four
boundaries from the source-forward capture; it does not recompute expected
intermediates or establish native indexer, compressor or full-model parity.

[Source and license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/LICENSE)

```text
MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
