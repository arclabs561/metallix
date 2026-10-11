// The dense decoder in crates/models/qwen, which runs qwen3, llama, qwen2 and
// bidirectional_pplx_qwen3 checkpoints. Numbers come from qwen3.facts.json.

#import "lib.typ": *

#show: setup
#let f = json("qwen3.facts.json")
#let reference = f.variants.find(v => v.model_type == f.model_type)

#diagram(
  spacing: (8mm, 5mm),
  edge-stroke: 0.6pt,
  block((0, 0), [Token IDs], name: <tokens>),
  block(
    (0, 1),
    [*Token embedding*\ #f.vocab_size #sym.times #f.hidden_size],
    role: "embed",
    name: <embed>,
  ),
  rmsnorm((0, 2), name: <norm1>),
  gqa-attention(
    (0, 3),
    f,
    extras: (
      if reference.qk_norm [per-head RMSNorm on queries and keys],
      [#reference.attention mask],
    ).filter(x => x != none),
    name: <attention>,
  ),
  add((0, 4), name: <add1>),
  rmsnorm((0, 5), name: <norm2>),
  swiglu((0, 6), width: f.intermediate_size, name: <mlp>),
  add((0, 7), name: <add2>),
  repeated(
    (<norm1>, <attention>, <add1>, <norm2>, <mlp>, <add2>, (-1, 1.6)),
    [#sym.times #f.layers layers],
  ),
  rmsnorm((0, 8), name: <final>),
  block(
    (0, 9),
    [*Output projection*\ #f.hidden_size #sym.arrow #f.vocab_size#if f.tied_output_embedding [, tied to the embedding]],
    role: "embed",
    name: <head>,
  ),
  block((0, 10), [Logits], name: <logits>),

  edge(<tokens>, <embed>, flow),
  edge(<embed>, <norm1>, flow),
  edge(<norm1>, <attention>, flow),
  edge(<attention>, <add1>, flow),
  edge(<add1>, <norm2>, flow),
  edge(<norm2>, <mlp>, flow),
  edge(<mlp>, <add2>, flow),
  edge(<add2>, <final>, flow),
  edge(<final>, <head>, flow),
  edge(<head>, <logits>, flow),
  // Residual paths skip each sublayer.
  edge((0, 1.6), (-1, 1.6), (-1, 4), <add1>, flow, corner-radius: 2pt),
  edge((0, 4.5), (-1, 4.5), (-1, 7), <add2>, flow, corner-radius: 2pt),
)

#v(6pt)
#set text(size: 8pt)
#table(
  columns: 4,
  stroke: 0.4pt,
  inset: 4pt,
  table.header([`model_type`], [Q/K RMSNorm], [Q/K/V bias], [Mask]),
  ..f
    .variants
    .map(v => (
      raw(v.model_type),
      if v.qk_norm [yes] else [no],
      if v.qkv_bias [yes] else [no],
      v.attention,
    ))
    .flatten(),
)
#v(2pt)
Reference: #raw(f.reference.repo) at #raw(f.reference.revision.slice(0, 7)):
#f.layers layers, width #f.hidden_size,
#f.kv_bytes_per_token_bf16 bytes of BF16 K/V cache per token.
