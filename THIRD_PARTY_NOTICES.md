# Third-party notices

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

`crates/models/deepseek/src/moe.rs` composes the same revision's `Gate`,
`Expert.forward` and `MoE.forward` ordering through the scalar numerical
references. Its encoded synthetic-weight oracle is extracted from actual
source-forward hooks by `scripts/v41-forward-reference.py`.

`crates/models/deepseek/src/ffn.rs` follows the same revision's `Block.forward`
FFN ordering: derive fresh HC coefficients, collapse with the incoming
attention pre-mix, normalize, execute the MoE, then apply the fresh HC post-mix.
The capture retains the source block's HC inputs, coefficients and residuals
to diagnose numerical differences; this is not a full-block parity claim.

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
