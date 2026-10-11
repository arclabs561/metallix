// Qwen3-ASR in crates/models/qwen3-asr: an audio encoder whose output rows
// replace audio placeholder tokens in the prompt of the Qwen3 decoder
// (qwen3.typ). Numbers come from qwen3_asr.facts.json.

#import "lib.typ": *

#show: setup
#let f = json("qwen3_asr.facts.json")
#let e = f.encoder
#let d = f.decoder

#diagram(
  spacing: (6mm, 5mm),
  edge-stroke: 0.6pt,
  block((0, 0), [16 kHz mono audio], name: <audio>),
  block((0, 1), [*Log-mel features*\ #e.mel_bins bins, cut into #e.chunk_frames\-frame chunks], role: "embed", name: <mel>),
  block(
    (0, 2),
    [*3 convolutions*\ 3#sym.times\3, stride 2, GELU, #e.conv_channels channels;\ linear to width #e.width],
    role: "embed",
    name: <conv>,
  ),
  block((0, 3), [\+ sinusoid positions, restarting in each chunk\ (#e.frames_per_chunk frames per chunk)], role: "embed", name: <pos>),
  layernorm((0, 4), name: <norm1>),
  block(
    (0, 5),
    [*Windowed self-attention*\ #e.heads heads; windows of #e.window_frames frames],
    role: "attention",
    name: <attention>,
  ),
  add((0, 6), name: <add1>),
  layernorm((0, 7), name: <norm2>),
  gelu-mlp((0, 8), width: e.ffn_width, name: <mlp>),
  add((0, 9), name: <add2>),
  repeated(
    (<norm1>, <attention>, <add1>, <norm2>, <mlp>, <add2>, (-1.4, 3.6), (1.4, 3.6)),
    [#sym.times #e.layers encoder layers],
  ),
  layernorm((0, 10), name: <final>),
  block((0, 11), [*Projection*\ two layers with GELU, to width #e.output_width], role: "embed", name: <proj>),
  block((2, 11), [Prompt token IDs\ with one `<|audio_pad|>` per frame], name: <prompt>),
  block((2, 12), [*Token embedding*], role: "embed", name: <embed>),
  block((2, 13), [Audio rows replace the\ `<|audio_pad|>` embeddings], name: <splice>),
  block(
    (2, 14),
    [*Qwen3 decoder* (as drawn for `qwen3`)\ #d.layers layers of width #d.hidden_size, #d.attention_heads query and #d.kv_heads K/V heads of #d.head_dim,\ MLP inner width #d.intermediate_size#if d.tied_output_embedding [, output tied to the embedding]],
    role: "attention",
    name: <decoder>,
  ),
  block((2, 15), [Transcript tokens], name: <out>),

  edge(<audio>, <mel>, flow),
  edge(<mel>, <conv>, flow),
  edge(<conv>, <pos>, flow),
  edge(<pos>, <norm1>, flow),
  edge(<norm1>, <attention>, flow),
  edge(<attention>, <add1>, flow),
  edge(<add1>, <norm2>, flow),
  edge(<norm2>, <mlp>, flow),
  edge(<mlp>, <add2>, flow),
  edge(<add2>, <final>, flow),
  edge(<final>, <proj>, flow),
  edge(<proj>, (0, 13), <splice>, flow),
  edge(<prompt>, <embed>, flow),
  edge(<embed>, <splice>, flow),
  edge(<splice>, <decoder>, flow),
  edge(<decoder>, <out>, flow),
  // Residual paths skip each sublayer.
  edge((0, 3.6), (-1.4, 3.6), (-1.4, 6), <add1>, flow, corner-radius: 2pt),
  edge((0, 6.5), (-1.4, 6.5), (-1.4, 9), <add2>, flow, corner-radius: 2pt),
)

#v(4pt)
#set text(size: 8pt)
Reference: #raw(f.reference.repo) at #raw(f.reference.revision.slice(0, 7)).
Attention never crosses a window boundary.
