# Julia-1 native decision prerequisite

Status: source contract, published-header validation, real-tokenizer sequence
parity, and a native CPU decision head are implemented. Full Julia execution
still requires native ModernBERT and checkpoint qualification.

## Reproduction identity

The public Hugging Face model API reported the following source revision on
2026-09-27:

- repository: `SupersonicLabs/Julia-1`
- revision: `a85b127321d580d65176c89ced8273f305745d85`
- checkpoint: `model.safetensors` (144,292,870 F32 parameters reported by the
  API)
- published checkpoint SHA-256: `df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72`
- bounded header pin: 17,568 bytes, 170 F32 tensors, header SHA-256
  `228dd499df2a19d94285cf64b61f28e1beafec9f9a45984fbb878f0c43c34e19`,
  and canonical tensor-name/type/shape SHA-256
  `6e986ce6b9b8ae653a33f89bf021a3670e313a353ff1298fe902d99438b44d48`
- architecture metadata: `JuliaDecisionModel`, `head_layers = 2`, `n_act = 2`,
  `dropout = 0.1`

All source claims below are pinned to that revision, rather than `main`:

- [model.py](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/julia/model.py)
- [data.py](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/julia/data.py)
- [encoder/config.json](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/encoder/config.json)
- [inference.py](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/julia/inference.py)

The pin identifies source and metadata only. It does not establish a local
checkpoint payload or executable native runtime. Tokenizer sequence parity is
qualified separately below. The header
was fetched through exact HTTP byte ranges only after the host returned `206`
and matching `Content-Range`; no tensor payload was read.
The pinned project declares Python 3.11+, PyTorch 2.6+, Transformers 5.0.x,
and Safetensors 0.5+; its encoder configuration records Transformers 5.0.0.

## Artifact and encoder boundary

An artifact loader must require the top-level `config.json`,
`julia_config.json`, `encoder/config.json`, `tokenizer/`, and one
`model.safetensors` before attempting execution. The source loader creates an
`AutoModel` from `encoder/config.json` with `trust_remote_code=False`, then
strictly loads the complete Julia state dictionary. A native loader must not
silently accept missing, extra, or differently shaped tensors.

The local inspector now validates a complete published-header fingerprint at
this pin: its declared file length, raw header hash, tensor count, and canonical
tensor-name/type/shape hash. It also directly checks the Julia-owned named
tensors, including `act_head.0.weight` shape `[256, 388]`. This establishes
artifact-header identity, not executable operator parity.

The encoder is ModernBERT, used as an encoder (`last_hidden_state`), not as the
declared `ModernBertForMaskedLM` output head. Its execution-relevant published
configuration is:

| field | value |
| --- | --- |
| hidden width | 384 |
| layers | 22 |
| attention heads | 6 |
| intermediate width | 1,152 |
| maximum positions | 8,192 |
| attention schedule | full attention every third layer, otherwise sliding attention |
| local window | 128 |
| position scheme | `sans_pos` with RoPE theta 160,000 |
| token IDs | pad 0, CLS 1, mask 4, SEP 1 |

This makes ModernBERT's global/local bidirectional attention, RoPE behavior,
padding semantics, and tokenizer behavior prerequisites. A generic decoder
prefill, Qwen execution graph, causal mask, or reusable decoder KV cache cannot
serve this contract.

## Request encoding contract

`data.sequence` is the source of the model-input serialization. For a validated
row, it performs these steps exactly:

1. Validation requires a string, dictionary, or list `state`; a string
   `question`; 2--20 nonempty string options; and type `choice`, `score`, or
   `noul`. A `noul` request has exactly two options. A supplied target is an
   in-range integer and supplied teacher logits are finite and option-aligned.
2. `state` is retained when it is a string; a dictionary or list is encoded by
   Python `json.dumps(..., ensure_ascii=False)`. The default type is `choice`.
   The accepted type-to-ID map is `choice: 0`, `score: 1`, `noul: 2`.
3. Before tokenization, every literal tokenizer mask-token string in `state`,
   `question`, and options becomes one space. Strict mode instead rejects any
   such input. Text is tokenized with `add_special_tokens=False`.
4. The question text is `"{type} question: {question}"`. Each option text is
   prefixed with one ASCII space and receives a leading mask-token ID. Options
   are normally limited to 48 tokenizer IDs before that marker.
5. With defaults `max_length = 8192` and `head_length = 256`, the option block
   receives its budget first. If its marked options leave fewer than 16 head
   tokens, each marked option is truncated to
   `max(4, (head_length - 16) // option_count)` tokens. Strict mode rejects any
   loss of question or option tokens.
6. IDs are `[CLS] + question[:max(8, budget)] + [SEP]`, then each marked option
   in caller order, then `[SEP]`, then state IDs truncated to leave a final
   `[SEP]`. Marker positions are the offsets of the leading mask IDs. Strict
   mode rejects a state that would be truncated; normal mode reports only this
   state truncation as `truncated`.
7. A batch pads IDs with pad ID to the next multiple of eight, creates a boolean
   attention mask for real IDs, pads marker positions with zero, and uses a
   boolean marker mask to distinguish those padding positions. It carries
   `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, and `qtype`.

For `noul`, validation requires exactly two supplied options ordered
`[false_description, true_description]`. The marker scores retain that order;
the source does not convert this request into generated `true`/`false` tokens.
Keep supplied descriptions intact: the pinned author-reported [CPU ablation](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/metrics/typed-cpu-20260926.json)
reports 483/600 Boolean decisions correct with original criteria versus 391/600
after replacing them with literal labels. These are upstream measurements, not
Metallix results. Our encoding regression preserves descriptive options and
rejects missing or invalid options.

The stdlib fixtures isolate marker positions, marker masks, qtype,
round-up-to-eight padding, option permutation, `noul` ordering and strict
overflow behavior using a fixed tokenizer stub.

A separate real-tokenizer fixture now checks exact sequence outputs against
the inspected pinned source function, covering choice, `noul`, Unicode,
reserved-marker sanitation and strict rejection, and context truncation.
The tokenizer asset SHA-256 is
`609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f`;
`data.py` SHA-256 is
`e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e`.
`scripts/julia-tokenizer-parity.py` requires local assets and
`tokenizers==0.23.2`; it checks hashes before executing only the audited
`QTYPES` and `sequence` AST nodes. It neither imports the remote module nor
loads model weights. This qualifies these encoding vectors, not the model's
numerical forward pass or arbitrary input coverage.

```sh
uv run scripts/julia-tokenizer-parity.py --source /path/to/pinned/data.py --tokenizer /path/to/pinned/tokenizer.json
python3 scripts/test_julia_tokenizer_fixture.py
```

The settings at this same source revision differ by entry point:

| Entry point | Maximum length | Head budget | Strict encoding |
| --- | --- | --- | --- |
| `data.sequence` defaults | 8192 | 256 | false |
| Published inference policy | 8192 | 512 | true |
| Typed reproduction script | 1024 | 512 | true |

The [policy](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/inference-policy.json)
and [reproduction script](https://huggingface.co/SupersonicLabs/Julia-1/blob/a85b127321d580d65176c89ced8273f305745d85/scripts/reproduce_typed.py)
are distinct configurations at one pin, not evidence of architectural drift.
Future runtime receipts must identify the selected settings explicitly.

## Forward and score contract

For a batch of width 384, the source forward path is:

```text
hidden = ModernBERT(input_ids, attention_mask).last_hidden_state
hidden = hidden + type_emb[qtype][:, None, :]
hidden = two pre-norm TransformerEncoderLayer blocks(hidden, padding_mask)
marker_hidden = gather(hidden, marker_pos)
score = Linear(384, 1)(GELU(Linear(384, 384)(LayerNorm(marker_hidden))))
score[not marker_mask] = -10000
```

`type_emb` has three rows of width 384. Each decision-head transformer layer
uses six heads, width 1,536 feed-forward layers, batch-first layout, pre-norm,
and the PyTorch constructor defaults not overridden by the source: ReLU in its
feed-forward activation and LayerNorm epsilon `1e-5`. It is bidirectional with
only the padding mask; it has no causal mask. Eval-mode parity must also match
the configured `0.1` dropout being disabled.

The scorer's first LayerNorm also uses the PyTorch default epsilon `1e-5`; the
intermediate activation is GELU. Scores are converted to F32 after squeezing
the final singleton dimension. `forward(..., return_actions=False)` returns
these raw, masked scores. The registered three-element `temperature` buffer is
not read by that forward path. The published `predict` helper applies ordinary
softmax across valid returned option scores; its display-only rounding is not a
model operator and must not be used for native decision probabilities.

`return_actions=True`, sparse marker-only head execution, training dropout,
checkpointing, CUDA autocast, INT8 exports, router backends, and presentation
rounding are outside a first native decision-forward fixture.

## Implementable next gate

The pure-stdlib `scripts/julia_encoding_contract.py` and nine hand-derived
fixtures in `scripts/test_julia_encoding_contract.py` now cover serialization,
mask sanitation, option permutation, Boolean order, strict loss, and padding.
They do not qualify the published tokenizer or execute the model.

The isolated artifact inspector now checks top-level/Julia/ModernBERT metadata
and the complete pinned F32 header. The separate published-tokenizer sequence
gate now passes six source-oracle cases, including option permutation.

The synthetic F32 decision-head fixture now compares the hash-gated pinned
`JuliaDecisionModel` class with a separately spelled-out numerical reference:
type embedding, two pre-norm attention/FFN blocks, marker gather and scorer.
Five cases cover ordinary scoring, nonuniform padding perturbation, an unmasked
padding negative control, marker permutation, and an invalid marker. All use
CPU PyTorch 2.13.0, deterministic synthetic parameters, and fixed `1e-5`
absolute/relative tolerances. Fixture metadata and case inputs are checked too.
The CPU-only `julia` crate now implements this head in Rust from caller-supplied
F32 weights and encoder hidden states. Its fixture test checks structured
parameter names, shapes and generation ordinals, then executes all five cases
under the same fixed tolerance. Malformed tensors, missing unmasked keys,
out-of-range markers (including masked ones), nonfinite inputs and overflowing
normalization statistics are rejected. A checked operation budget includes
both attention reductions, all head affine projections and scorer work.
This qualifies native CPU head math, not checkpoint weights, ModernBERT,
Metal execution or model-level quality.

The optional source gate requires an inspected copy of pinned `julia/model.py`
at `.agents/receipts/julia/model.py`, SHA-256
`ef2ba82fe20cdf0db7bb887e9ef075476ed08b985ce9a95be0de3e26246ecc81`.
The runner checks this hash before extracting only the audited class; it does
not import the remote module, download weights, or execute ModernBERT.

```sh
uv run scripts/julia_head_reference.py
uv run scripts/test_julia_head_reference.py
```

The native head gate runs with `cargo test -p julia` and the workspace check.
Next, qualify a separate native ModernBERT forward under an explicit
checkpoint-resource budget. Native support
requires an actual pinned checkpoint and independent source-runtime numerical
comparison before server or registry integration.


## Native head placement

The native head lives in `crates/models/julia`, beside the existing
model crates. A test-only experiment would prove the arithmetic but leave no
reusable head for a later encoder. Putting it in the shared engine or importing
DeepSeek numerical policy would couple unrelated model contracts. A bounded
CPU head with local affine, normalization, attention and activation operations
keeps this step model-specific without introducing a shared tensor framework.
The first gate is the synthetic pinned-source fixture; encoder execution and
checkpoint loading remain separately qualified boundaries.
