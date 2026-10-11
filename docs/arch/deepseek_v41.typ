// DeepSeek-V4.1 Flash text model in crates/models/deepseek, as its request
// schedule runs it. Numbers come from deepseek_v41.facts.json. mx serve does
// not load this adapter yet.

#import "lib.typ": *

#show: setup
#let f = json("deepseek_v41.facts.json")
#let a = f.attention
#let m = f.moe
#let note(body) = text(size: 7pt, body)
#let roles = (
  startup: (rgb("#9e9e9e"), [startup, window only]),
  window: (rgb("#cfcfcf"), [window only]),
  ratio2_owner: (rgb("#3f6fb5"), [ratio 2, owns K/V and indexer]),
  ratio2_consumer: (rgb("#a9c1e6"), [ratio 2, reuses its owner's]),
  ratio1_owner: (rgb("#b5523f"), [ratio 1, owns K/V and candidate blocks]),
  ratio1_indexer: (rgb("#e08a6f"), [ratio 1, re-indexes within the blocks]),
  ratio1_consumer: (rgb("#f2c6b8"), [ratio 1, reuses its owner's]),
)
#let fills-by-role = (:)
#for (key, value) in roles { fills-by-role.insert(key, value.at(0)) }
#let hc = f.hyper_connection_copies

#diagram(
  spacing: (6mm, 5mm),
  edge-stroke: 0.6pt,
  block((0, 0), [Token IDs], name: <tokens>),
  block((-1.6, 0), [Vision encoder], loaded: false, name: <vision>),
  block((0, 1), [*Token embedding*, #f.vocab_size #sym.times #f.width\ repeated into #hc residual copies], role: "embed", name: <embed>),
  block(
    (1.6, 2),
    [*Engram*\ hashed #(f.engram_max_ngram_size)-gram embedding lookup,\ gated into the residual],
    role: "embed",
    optional: true,
    name: <engram>,
  ),
  block((0, 2), [Hyper-connection pre-mix:\ #hc copies #sym.arrow one input], role: "norm", name: <pre1>),
  rmsnorm((0, 3), name: <norm1>),
  block(
    (0, 4),
    [*Sparse attention*, #a.heads heads of #a.head_dim\ low-rank queries (rank #a.q_lora_rank), rotary on #a.rope_head_dim;\ keys: the last #a.window_size tokens; compressed layers add the\ top #a.index_topk compressed K/V entries an indexer of #a.index_heads heads picks;\ output in #a.o_groups low-rank groups (rank #a.o_lora_rank)],
    role: "attention",
    name: <attention>,
  ),
  block((0, 5), [Hyper-connection post-mix:\ output written to all #hc copies], role: "norm", name: <post1>),
  block((0, 6), [Hyper-connection pre-mix], role: "norm", name: <pre2>),
  rmsnorm((0, 7), name: <norm2>),
  block(
    (0, 8),
    [*Mixture of experts*\ #m.score_func router, top #m.activated_experts of #m.routed_experts routed experts\ plus #m.shared_experts shared; SwiGLU, inner width #m.expert_width],
    role: "mlp",
    name: <moe>,
  ),
  block((0, 9), [Hyper-connection post-mix], role: "norm", name: <post2>),
  repeated(
    (<engram>, <pre1>, <norm1>, <attention>, <post1>, <pre2>, <norm2>, <moe>, <post2>, (-2, 1.6)),
    [#sym.times #f.layers.len() layers],
  ),
  block((0, 10), [Hyper-connection collapse, RMSNorm], role: "norm", name: <final>),
  block((0, 11), [*Output projection*, last position only], role: "embed", name: <head>),
  block((-1.6, 11), [Multi-token prediction\ and draft heads], loaded: false, name: <mtp>),

  edge(<tokens>, <embed>, flow),
  edge(<vision>, <embed>, flow, dash: "dashed"),
  edge(<embed>, <pre1>, flow),
  edge(<engram>, <pre1>, flow, dash: "dotted", label: note[layers #f.engram_layers.map(str).join(", ")], label-side: right),
  edge(<pre1>, <norm1>, flow),
  edge(<norm1>, <attention>, flow),
  edge(<attention>, <post1>, flow),
  edge(<post1>, <pre2>, flow),
  edge(<pre2>, <norm2>, flow),
  edge(<norm2>, <moe>, flow),
  edge(<moe>, <post2>, flow),
  edge(<post2>, <final>, flow),
  edge(<final>, <head>, flow),
  edge((0, 9.5), (-1.6, 9.5), <mtp>, flow, dash: "dashed"),
  // The residual copies bypass each sublayer.
  edge((0, 1.6), (-1.8, 1.6), (-1.8, 5), <post1>, flow, corner-radius: 2pt),
  edge((0, 5.5), (-1.8, 5.5), (-1.8, 9), <post2>, flow, corner-radius: 2pt),
)

#v(6pt)
#set text(size: 8pt)
Layer order: #layer-strip(f.layers, fills-by-role)
#v(2pt)
#grid(
  columns: 2,
  column-gutter: 12pt,
  row-gutter: 3pt,
  ..roles.values().map(value => [#box(width: 4pt, height: 8pt, fill: value.at(0), stroke: 0.3pt) #value.at(1)]),
)
#v(2pt)
A reusing layer attends over its owner's compressed K/V and indices from the same step.
Candidate blocks hold #a.candidate_block_size positions; an owner keeps the top #a.candidate_topk_blocks.
\ Reference: #raw(f.reference.repo) at #raw(f.reference.revision.slice(0, 7)).
Dashed: in the checkpoint, not run by metallix. Dotted: only the listed layers.
