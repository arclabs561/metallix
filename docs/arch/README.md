# Model architecture diagrams

One diagram per model architecture metallix runs. Each draws the blocks the
crate executes, with numbers from one reference checkpoint. A block the
checkpoint has but metallix does not load is drawn dashed, and its caption
says so.

## Dense decoder (`qwen3`, `llama`, `qwen2`, `bidirectional_pplx_qwen3`)

![Qwen3 dense decoder: embedding, 28 pre-norm layers of grouped-query attention and SwiGLU MLP, final norm and output projection](qwen3.svg)

The dense decoder in `crates/models/qwen`, drawn at Qwen3-0.6B. Each layer
normalizes its input, applies grouped-query attention with rotary positions,
adds the result to the residual stream, then does the same with a SwiGLU MLP.
The table lists how the four model types this decoder accepts differ: Qwen3
normalizes each query and key head, Qwen2 adds a bias to the query, key and
value projections, and pplx-embed attends in both directions.

## How the diagrams stay correct

- `<kind>.typ` lays out the blocks from [`lib.typ`](lib.typ) and reads every
  number from `<kind>.facts.json`.
- A test in the model's crate parses the reference `config.json` under
  [`configs/`](configs) with that crate's own config code and writes
  `<kind>.facts.json`. It fails when the committed file differs, so a parsing
  change shows up as a stale diagram.
- `scripts/check_arch_diagrams.py` renders each `.typ` with typst 0.15.1 and
  fails when the committed `.svg` differs. It also fails when a crate under
  `crates/models` has no diagram.
- A server test maps every model adapter to its diagram; a new adapter does
  not compile until it names one.

## Adding or updating a diagram

1. Add the reference `config.json` under `configs/`, from a pinned revision,
   and a test in the model's crate like `crates/models/qwen/tests/arch_facts.rs`.
2. Run that test with `UPDATE_ARCH_FACTS=1` to write `<kind>.facts.json`.
3. Write `<kind>.typ` from the blocks in `lib.typ`, adding a block there when
   another architecture will reuse it.
4. Run `scripts/check_arch_diagrams.py --write` to render `<kind>.svg`, then
   add a section to this page with the image and its caption.
