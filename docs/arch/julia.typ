// Julia-1 in crates/models/julia: a ModernBERT encoder and a decision head
// that scores each option of a typed question, on the CPU. Numbers come
// from julia.facts.json, written from the crate's pinned shape constants.

#import "lib.typ": *

#show: setup
#let f = json("julia.facts.json")
#let kinds = (global: rgb("#d9654a"), local: rgb("#e8a33d"))
#let globals = f.encoder_layers.filter(k => k == "global").len()

#diagram(
  spacing: (6mm, 5mm),
  edge-stroke: 0.6pt,
  block((0, 0), [Typed request: state and questions], name: <request>),
  block((0, 1), [*Serialized question*\ token IDs and option markers, at most #f.max_positions], name: <tokens>),
  block((0, 2), [*Token embedding*\ #f.vocab_size #sym.times #f.width, then LayerNorm], role: "embed", name: <embed>),
  layernorm((0, 3), name: <norm1>),
  block(
    (0, 4),
    [*Self-attention*\ #f.attention_heads heads of #f.head_width, rotary positions;\ global, or #sym.plus.minus#f.local_window positions],
    role: "attention",
    name: <attention>,
  ),
  add((0, 5), name: <add1>),
  layernorm((0, 6), name: <norm2>),
  block((0, 7), [*GeGLU MLP*\ inner width #f.encoder_ffn_width], role: "mlp", name: <mlp>),
  add((0, 8), name: <add2>),
  repeated(
    (<norm1>, <attention>, <add1>, <norm2>, <mlp>, <add2>, (-1.4, 2.6), (1.9, 2.6)),
    [#sym.times #f.encoder_layers.len() encoder layers],
  ),
  layernorm((0, 9), name: <final>),
  block((0, 10), [\+ question-type embedding], role: "embed", name: <qtype>),
  block(
    (0, 11),
    [*#f.head_layers decision-head layers*\ pre-norm attention, #f.attention_heads heads;\ ReLU feed-forward, inner width #f.head_ffn_width],
    role: "attention",
    name: <head>,
  ),
  block(
    (0, 12),
    [*Scorer*, at each option marker\ LayerNorm, linear, GELU, linear to one score],
    role: "mlp",
    name: <scorer>,
  ),
  block((0, 13), [Softmax over options, then the question type's readout], name: <answer>),

  edge(<request>, <tokens>, flow),
  edge(<tokens>, <embed>, flow),
  edge(<embed>, <norm1>, flow),
  edge(<norm1>, <attention>, flow, label: text(size: 7pt)[identity in layer 0], label-side: left),
  edge(<attention>, <add1>, flow),
  edge(<add1>, <norm2>, flow),
  edge(<norm2>, <mlp>, flow),
  edge(<mlp>, <add2>, flow),
  edge(<add2>, <final>, flow),
  edge(<final>, <qtype>, flow),
  edge(<qtype>, <head>, flow),
  edge(<head>, <scorer>, flow),
  edge(<scorer>, <answer>, flow),
  // Residual paths skip each sublayer.
  edge((0, 2.6), (-1.4, 2.6), (-1.4, 5), <add1>, flow, corner-radius: 2pt),
  edge((0, 5.5), (-1.4, 5.5), (-1.4, 8), <add2>, flow, corner-radius: 2pt),
)

#v(6pt)
#set text(size: 8pt)
Encoder layer order: #layer-strip(f.encoder_layers, kinds)
#h(4pt) #box(width: 4pt, height: 8pt, fill: kinds.global, stroke: 0.3pt) global (#globals layers)
#h(4pt) #box(width: 4pt, height: 8pt, fill: kinds.local, stroke: 0.3pt) local window
\ Reference: #raw(f.reference.repo) at #raw(f.reference.revision.slice(0, 7)). Every step runs in F32 on the CPU.
