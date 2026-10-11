// FLUX.2 [klein] text-to-image in crates/models/flux2: a Qwen3 text encoder,
// a rectified-flow transformer run once per scheduler step, and a VAE
// decoder. Numbers come from flux2.facts.json.

#import "lib.typ": *

#show: setup
#let f = json("flux2.facts.json")
#let t = f.transformer
#let te = f.text_encoder
#let note(body) = text(size: 7pt, body)

#diagram(
  spacing: (6mm, 5mm),
  edge-stroke: 0.6pt,
  step((0, 0), [Prompt, chat-rendered and padded], name: <prompt>),
  block(
    (0, 1),
    [*Qwen3 text encoder* (as drawn for `qwen3`)\ #te.layers layers of width #te.hidden_size, residual streams of\ #(t.text_channels / te.hidden_size) layers concatenated to #t.text_channels per token],
    role: "attention",
    name: <text>,
  ),
  step((2, 1), [Gaussian noise latents\ one token per #f.pixels_per_token#sym.times#f.pixels_per_token pixels, #t.latent_channels channels], name: <noise>),
  block((0, 2), [Linear to width #t.width], role: "embed", name: <text-in>),
  block((2, 2), [Linear to width #t.width], role: "embed", name: <image-in>),
  block((3.2, 3), [*Timestep embedding*\ sinusoid of #t.timestep_channels, MLP;\ shift, scale, gate per block], role: "norm", name: <time>),
  block(
    (1, 3),
    [*#t.double_blocks double-stream blocks*\ text and image keep their own weights and\ attend jointly; #t.heads heads of #t.head_dim, rotary on\ #t.rope_axes.len() axes of #t.rope_axes.at(0) lanes; MLP hidden #t.mlp_hidden],
    role: "attention",
    name: <double>,
  ),
  block(
    (1, 4),
    [*#t.single_blocks single-stream blocks*\ one stream of text then image tokens;\ attention and MLP in parallel],
    role: "attention",
    name: <single>,
  ),
  block((1, 5), [Image tokens: norm, linear to #t.latent_channels channels\ (the velocity)], role: "embed", name: <out>),
  step((1, 6), [Flow-match Euler step on the latents], name: <euler>),
  step((1, 7), [Latent grid: denormalize, split 2#sym.times\2 patches], name: <unpack>),
  block((1, 8), [*VAE decoder*\ resnet up-blocks with a mid attention block], role: "mlp", name: <vae>),
  step((1, 9), [RGB image], name: <image>),

  edge(<prompt>, <text>, flow),
  edge(<text>, <text-in>, flow),
  edge(<noise>, <image-in>, flow),
  edge(<text-in>, <double>, flow),
  edge(<image-in>, <double>, flow),
  edge(<time>, <double>, flow, dash: "dotted"),
  edge(<time>, <single>, flow, dash: "dotted", label: note[modulates], label-side: left),
  edge(<double>, <single>, flow),
  edge(<single>, <out>, flow),
  edge(<out>, <euler>, flow),
  edge(<euler>, (2.4, 6), (2.4, 1.5), (2, 1.5), <image-in>, flow, corner-radius: 2pt, label: note[next step], label-pos: 0.3, label-side: right),
  edge(<euler>, <unpack>, flow, label: note[after the last step], label-side: left),
  edge(<unpack>, <vae>, flow),
  edge(<vae>, <image>, flow),
)

#v(4pt)
#set text(size: 8pt)
Reference: #raw(f.reference.repo) at #raw(f.reference.revision.slice(0, 7)).
The text encoder runs once per prompt; the transformer runs once per scheduler step.
