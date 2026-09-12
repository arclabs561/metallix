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
