// The hybrid decoder in crates/models/qwen35 (qwen3_5 and qwen3_5_moe text).
// Numbers come from qwen35.facts.json: one dense and one mixture-of-experts
// reference.

#import "lib.typ": *

#show: setup
#let f = json("qwen35.facts.json")
#let kinds = (linear: rgb("#8e6cc9"), full: rgb("#d9654a"))

#diagram(
  spacing: (6mm, 5mm),
  edge-stroke: 0.6pt,
  block((0, 0), [Token IDs], name: <tokens>),
  block((-1.6, 0), [Vision encoder\ and merger], loaded: false, name: <vision>),
  block((0, 1), [*Token embedding*], role: "embed", name: <embed>),
  rmsnorm((0, 2), name: <norm1>),
  gated-deltanet((-0.8, 3), name: <linear>),
  gated-attention((0.8, 3), name: <full>),
  add((0, 4), name: <add1>),
  rmsnorm((0, 5), name: <norm2>),
  swiglu((-0.8, 6), name: <dense>),
  moe((0.8, 6), name: <experts>),
  add((0, 7), name: <add2>),
  repeated(
    (<norm1>, <linear>, <full>, <add1>, <norm2>, <dense>, <experts>, <add2>, (-1.9, 1.6)),
    [#sym.times each decoder layer],
  ),
  rmsnorm((0, 8), name: <final>),
  block((0, 9), [*Output projection*], role: "embed", name: <head>),
  block((-1.6, 9), [Multi-token\ prediction head], loaded: false, name: <mtp>),
  block((0, 10), [Logits], name: <logits>),

  edge(<tokens>, <embed>, flow),
  edge(<vision>, <embed>, flow, dash: "dashed"),
  edge(<embed>, <norm1>, flow),
  edge(<norm1>, <linear>, flow, label: text(size: 7pt)[linear layers], label-side: right),
  edge(<norm1>, <full>, flow, label: text(size: 7pt)[full layers], label-side: left),
  edge(<linear>, <add1>, flow),
  edge(<full>, <add1>, flow),
  edge(<add1>, <norm2>, flow),
  edge(<norm2>, <dense>, flow, label: text(size: 7pt)[`qwen3_5`], label-side: right),
  edge(<norm2>, <experts>, flow, label: text(size: 7pt)[`qwen3_5_moe`], label-side: left),
  edge(<dense>, <add2>, flow),
  edge(<experts>, <add2>, flow),
  edge(<add2>, <final>, flow),
  edge(<final>, <head>, flow),
  edge(<head>, <logits>, flow),
  edge((0, 7.5), (-1.6, 7.5), <mtp>, flow, dash: "dashed"),
  // Residual paths skip each sublayer.
  edge((0, 1.6), (-1.75, 1.6), (-1.75, 4), <add1>, flow, corner-radius: 2pt),
  edge((0, 4.5), (-1.75, 4.5), (-1.75, 7), <add2>, flow, corner-radius: 2pt),
)

#v(6pt)
#set text(size: 8pt)
#let row(label, value) = (label, ..f.models.map(value))
#let count(m, kind) = m.layers.filter(k => k == kind).len()
#table(
  columns: 1 + f.models.len(),
  stroke: 0.4pt,
  inset: 4pt,
  align: (left, ..f.models.map(_ => left)),
  table.header(
    [],
    ..f.models.map(m => [#raw(m.repo) at #raw(m.revision.slice(0, 7))]),
  ),
  ..row([Layers], m => [#m.layers.len(): #count(m, "linear") linear, #count(m, "full") full]),
  ..row([Layer order], m => layer-strip(m.layers, kinds)),
  ..row([Width, vocabulary], m => [#m.hidden_size, #m.vocab_size]),
  ..row(
    [Full attention],
    m => [#m.attention_heads query, #m.kv_heads K/V heads of #m.head_dim, rotary on #m.rotary_dim],
  ),
  ..row(
    [Gated DeltaNet],
    m => [#m.linear_key_heads key heads of #m.linear_key_head_dim, #m.linear_value_heads value heads of #m.linear_value_head_dim, conv width #m.conv_kernel],
  ),
  ..row(
    [Feed-forward],
    m => if m.mlp.kind == "dense" [SwiGLU, inner width #m.mlp.intermediate_size] else [#m.mlp.experts experts, top #m.mlp.top_k, inner width #m.mlp.expert_intermediate_size, shared #m.mlp.shared_intermediate_size],
  ),
  ..row([Output projection], m => if m.tied_output_embedding [tied to the embedding] else [separate]),
  ..row([Prediction head], m => [#m.mtp_layers layer, not run]),
)
#v(2pt)
Layer order: #box(width: 4pt, height: 8pt, fill: kinds.linear, stroke: 0.3pt) Gated DeltaNet,
#box(width: 4pt, height: 8pt, fill: kinds.full, stroke: 0.3pt) gated full attention.
Dashed: in the checkpoint, not run by metallix.
