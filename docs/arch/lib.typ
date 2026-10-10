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
/// it, and the caption says so.
#let block(pos, label, role: "io", name: none, loaded: true, width: auto) = node(
  pos,
  label,
  name: name,
  fill: fills.at(role),
  stroke: if loaded { 0.6pt } else { (thickness: 0.6pt, dash: "dashed") },
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

/// The gated MLP: `down(silu(gate(x)) * up(x))`.
#let swiglu(pos, name: none) = block(
  pos,
  [*SwiGLU MLP*\ down(SiLU(gate _x_) #sym.dot.c up _x_)],
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
