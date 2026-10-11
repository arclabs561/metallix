// The dense Gemma 4 text decoder in crates/models/gemma. Numbers come from
// gemma4.facts.json: a plain dense reference and E2B, which adds per-layer
// inputs and K/V sharing.

#import "lib.typ": *

#show: setup
#let f = json("gemma4.facts.json")
#let kinds = (
  sliding: rgb("#e8a33d"),
  full: rgb("#d9654a"),
  sliding_shared: rgb("#f3d4a3"),
  full_shared: rgb("#efb3a4"),
)
#let note(body) = text(size: 7pt, body)

#diagram(
  spacing: (6mm, 5mm),
  edge-stroke: 0.6pt,
  block((0, 0), [Token IDs], name: <tokens>),
  block((-1.5, 0), [Vision and audio\ encoders], loaded: false, name: <media>),
  block((0, 1), [*Token embedding*\ #sym.times $sqrt(H)$], role: "embed", name: <embed>),
  block(
    (1.5, 1),
    [*Per-layer inputs*\ table row + norm(projected embedding)],
    role: "embed",
    optional: true,
    name: <ple>,
  ),
  rmsnorm((0, 2), name: <norm1>),
  block(
    (0, 3),
    [*Grouped-query attention*\ sliding window or full; per-head Q/K RMSNorm,\ V RMSNorm without weight; softmax scale 1],
    role: "attention",
    name: <attention>,
  ),
  rmsnorm((0, 4), name: <post1>),
  add((0, 5), name: <add1>),
  rmsnorm((0, 6), name: <norm2>),
  geglu((0, 7), name: <mlp>),
  rmsnorm((0, 8), name: <post2>),
  add((0, 9), name: <add2>),
  block(
    (1.5, 9),
    [*Per-layer input*\ norm(proj(GELU(gate _x_) #sym.dot.c row))],
    role: "mlp",
    optional: true,
    name: <ple-add>,
  ),
  block((0, 10), [#sym.times `layer_scalar`], role: "norm", name: <scalar>),
  repeated(
    (<norm1>, <attention>, <post1>, <add1>, <norm2>, <mlp>, <post2>, <add2>, <ple-add>, <scalar>, (-1.5, 1.6)),
    [#sym.times each decoder layer],
  ),
  rmsnorm((0, 11), name: <final>),
  block((0, 12), [*Output projection*\ tied to the embedding], role: "embed", name: <head>),
  block((0, 13), [Soft cap: _c_ tanh(logits / _c_)], name: <cap>),

  edge(<tokens>, <embed>, flow),
  edge(<media>, <embed>, flow, dash: "dashed"),
  edge(<tokens>, <ple>, flow, dash: "dotted"),
  edge(<embed>, <norm1>, flow),
  edge(<norm1>, <attention>, flow),
  edge(<attention>, <post1>, flow),
  edge(<post1>, <add1>, flow),
  edge(<add1>, <norm2>, flow),
  edge(<norm2>, <mlp>, flow),
  edge(<mlp>, <post2>, flow),
  edge(<post2>, <add2>, flow),
  edge(<embed>, <ple>, flow, dash: "dotted"),
  edge(<ple>, <ple-add>, flow, dash: "dotted", label: note[this layer's slice], label-side: left),
  edge(<add2>, <ple-add>, flow, dash: "dotted"),
  edge(<ple-add>, <scalar>, flow, dash: "dotted", label: note[added], label-side: left),
  edge(<add2>, <scalar>, flow),
  edge(<scalar>, <final>, flow),
  edge(<final>, <head>, flow),
  edge(<head>, <cap>, flow),
  // Residual paths skip each sublayer.
  edge((0, 1.6), (-1.5, 1.6), (-1.5, 5), <add1>, flow, corner-radius: 2pt),
  edge((0, 5.5), (-1.5, 5.5), (-1.5, 9), <add2>, flow, corner-radius: 2pt),
)

#v(6pt)
#set text(size: 8pt)
#let row(label, value) = (label, ..f.models.map(value))
#let count(m, kind) = m.layers.filter(k => k.starts-with(kind)).len()
#let shared(m) = m.layers.filter(k => k.ends-with("_shared")).len()
#let heads(a) = [#a.heads query, #a.kv_heads K/V #if a.kv_heads == 1 [head] else [heads] of #a.head_dim]
#table(
  columns: 1 + f.models.len(),
  stroke: 0.4pt,
  inset: 4pt,
  table.header([], ..f.models.map(m => [#raw(m.repo) at #raw(m.revision.slice(0, 7))])),
  ..row([Layers], m => [#m.layers.len(): #count(m, "sliding") sliding, #count(m, "full") full]),
  ..row([Layer order], m => layer-strip(m.layers, kinds)),
  ..row([Width, vocabulary], m => [#m.hidden_size, #m.vocab_size]),
  ..row([Sliding layers], m => [#heads(m.sliding), window #m.sliding_window, rotary on all]),
  ..row(
    [Full layers],
    m => [#heads(m.full), rotary on #(2 * m.full.rotated_pairs) of #m.full.head_dim#if m.full.value_from_key [, V is the K projection]],
  ),
  ..row(
    [K/V sharing],
    m => if shared(m) == 0 [none] else [last #shared(m) layers reuse an earlier layer's K/V],
  ),
  ..row(
    [MLP inner width],
    m => [#m.intermediate_size#if m.shared_intermediate_size != none [, #m.shared_intermediate_size in K/V-sharing layers]],
  ),
  ..row(
    [Per-layer input],
    m => if m.per_layer_input_width == none [none] else [width #m.per_layer_input_width per layer],
  ),
  ..row([Logit soft cap], m => if m.final_logit_softcapping == none [none] else [#m.final_logit_softcapping]),
)
#v(2pt)
Layer order:
#for (kind, label) in (
  ("sliding", [sliding]),
  ("full", [full]),
  ("sliding_shared", [sliding, shared K/V]),
  ("full_shared", [full, shared K/V]),
) [#box(width: 4pt, height: 8pt, fill: kinds.at(kind), stroke: 0.3pt) #label#h(6pt)]
\ Dashed: in the checkpoint, not run by metallix. Dotted: E2B and E4B only.
