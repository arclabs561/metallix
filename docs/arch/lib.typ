// Shared blocks for the architecture diagrams in this directory.
//
// Each <kind>.typ composes these blocks and reads its numbers from
// <kind>.facts.json, which a test in the model's crate writes from that
// crate's own config parsing. Render with scripts/check_arch_diagrams.py
// --write; the check renders again and compares, so the committed SVG
// matches its source.

#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node

// Fill per block role. Text stays black on a white page so the SVG reads on
// dark backgrounds too.
#let fills = (
  io: rgb("#f2f2f2"),
  embed: rgb("#e3ecf7"),
  norm: rgb("#f7f3e3"),
  attention: rgb("#fbe4dc"),
  linear: rgb("#ece4f7"),
  mlp: rgb("#e2f2e4"),
  layer: rgb("#fafafa"),
)

/// Page and text settings every diagram uses. Fonts come from the typst
/// binary (render with --ignore-system-fonts), so output is the same on
/// every machine with the pinned typst.
#let setup(body) = {
  set page(width: auto, height: auto, margin: 8pt, fill: white)
  set text(font: "New Computer Modern", size: 9pt)
  body
}

/// A block of the model. `role` picks the fill from `fills`. `loaded: false`
/// draws the block dashed: the checkpoint has it but metallix does not run
/// it. `optional: true` draws it dotted: only some checkpoints of the
/// architecture have it. The caption says which.
#let block(pos, label, role: "io", name: none, loaded: true, optional: false, width: auto) = node(
  pos,
  label,
  name: name,
  fill: fills.at(role),
  stroke: if not loaded { (thickness: 0.6pt, dash: "dashed") } else if optional {
    (thickness: 0.6pt, dash: "dotted")
  } else { 0.6pt },
  corner-radius: 2pt,
  inset: 5pt,
  width: width,
)

/// A residual addition.
#let add(pos, name: none) = node(
  pos,
  $plus$,
  name: name,
  shape: circle,
  stroke: 0.6pt,
  inset: 1.5pt,
)

/// RMS normalization.
#let rmsnorm(pos, name: none) = block(pos, [RMSNorm], role: "norm", name: name)

/// Grouped-query self-attention over `f` (a facts dictionary with
/// `attention_heads`, `kv_heads` and `head_dim`). `extras` lists per-variant
/// details such as Q/K normalization or the mask.
#let gqa-attention(pos, f, extras: (), name: none) = block(
  pos,
  {
    [*Grouped-query attention*]
    linebreak()
    [#f.attention_heads query heads, #f.kv_heads key/value heads]
    linebreak()
    [head width #f.head_dim, rotary positions]
    for extra in extras {
      linebreak()
      extra
    }
  },
  role: "attention",
  name: name,
)

/// The gated MLP: `down(silu(gate(x)) * up(x))`, with inner width `width`
/// when given.
#let swiglu(pos, width: none, name: none) = block(
  pos,
  {
    [*SwiGLU MLP*]
    linebreak()
    [down(SiLU(gate _x_) #sym.dot.c up _x_)]
    if width != none {
      linebreak()
      [inner width #width]
    }
  },
  role: "mlp",
  name: name,
)

/// A box drawn behind `members` (node names) with `label` at its top right,
/// for a block repeated once per layer.
#let repeated(members, label, name: none) = node(
  enclose: members,
  align(top + right, text(size: 8pt, label)),
  name: name,
  fill: fills.layer,
  stroke: (thickness: 0.6pt, paint: luma(120)),
  corner-radius: 4pt,
  inset: 8pt,
  layer: -1,
)

/// The arrow style shared by every data-flow edge.
#let flow = "-|>"

/// `GatedDeltaNet` linear attention: a short causal convolution, then the
/// gated delta rule over a fixed-size state per value head, then a gated
/// output norm. Its state does not grow with the sequence.
#let gated-deltanet(pos, name: none) = block(
  pos,
  [*Gated DeltaNet*\ causal conv, then the gated delta\ rule on a fixed state per value head;\ output RMSNorm #sym.dot.c SiLU(_z_)],
  role: "linear",
  name: name,
)

/// Full attention with a sigmoid output gate, per-head Q/K RMSNorm and
/// rotary embedding on part of each head.
#let gated-attention(pos, name: none) = block(
  pos,
  [*Gated full attention*\ grouped-query, K/V cache;\ per-head Q/K RMSNorm, partial rotary;\ output #sym.dot.c sigmoid(gate)],
  role: "attention",
  name: name,
)

/// Routed experts plus one shared expert scaled by a sigmoid gate.
#let moe(pos, name: none) = block(
  pos,
  [*Mixture of experts*\ router softmax, keep top _k_, renormalize;\ #sym.sum _w#sub[e]_ SwiGLU#sub[_e_] + sigmoid(gate) #sym.dot.c shared SwiGLU],
  role: "mlp",
  name: name,
)

/// One square per decoder layer, filled by kind: `kinds` maps each value in
/// `layers` to a fill.
#let layer-strip(layers, kinds) = box(
  stack(
    dir: ltr,
    spacing: 1pt,
    ..layers.map(kind => box(width: 4pt, height: 8pt, fill: kinds.at(kind), stroke: 0.3pt)),
  ),
)

/// The gated MLP with tanh-approximated GELU: `down(gelu(gate(x)) * up(x))`.
#let geglu(pos, name: none) = block(
  pos,
  [*GeGLU MLP*\ down(GELU#sub[tanh]\(gate _x_) #sym.dot.c up _x_)],
  role: "mlp",
  name: name,
)

/// Layer normalization (mean and variance), as in Whisper-style encoders.
#let layernorm(pos, name: none) = block(pos, [LayerNorm], role: "norm", name: name)

/// The ungated two-layer MLP: `fc2(gelu(fc1(x)))`.
#let gelu-mlp(pos, width: none, name: none) = block(
  pos,
  {
    [*GELU MLP*]
    linebreak()
    [fc#sub[2]\(GELU(fc#sub[1] _x_))]
    if width != none {
      linebreak()
      [inner width #width]
    }
  },
  role: "mlp",
  name: name,
)

/// A step of a pipeline that is not a network layer: scheduling, packing,
/// sampling.
#let step(pos, label, name: none) = block(pos, label, role: "io", name: name)
