<p align="center">
  <img src="docs/assets/metallix.png" alt="" width="160" />
</p>
<h1 align="center">metallix</h1>

A local inference engine for Apple Silicon, built around Rust and Metal.

Generate schema-constrained JSON, inspect token probabilities, and compare
resident versus layer-streamed inference with `mx`. Qwen3-0.6B runs today
through MLX; DeepSeek-V4.1-Flash is the main target, still in development.

## Make it generate a record

With a local Qwen3-0.6B checkpoint and the [built CLI](#build), run from the
repository root:

```sh
mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 29 \
  --json-schema fixtures/constraints/record.json --preview
```

The local run produced this value in `constraint.output`:

```json
{"status":"ready","count":1}
```

The schema constrains tokens during generation. `--preview` shows decoded JSON
and a short diagnostic summary on stderr; stdout remains the full JSON report.
Successful completion is independently validated. Running out of tokens before
completion reports `incomplete` and exits nonzero.

## Build

Requires Rust 1.87+, Apple Silicon, CMake and the Xcode Metal toolchain.
From the repository root:

```sh
cargo build -p server --release --all-features
export PATH="$PWD/target/release:$PATH"
mx --help
```

The PATH change lasts for this shell. `mx` and `metallix` are native executables
with the same CLI; neither needs a shell alias.

For the examples below, point `MODEL` at an already-downloaded Qwen3-0.6B
safetensors checkpoint with `config.json` and `tokenizer.json`:

```sh
MODEL=/path/to/Qwen3-0.6B
```

Inputs are currently raw token IDs, not text prompts or chat templates.

## Explore with mx

**Check your setup without loading weights.** Run a small Metal graph, then
inspect the checkpoint's headers:

```sh
mx smoke-qwen-metal
mx inspect-qwen-checkpoint --model "$MODEL"
```

**Make a draw reproducible.** Choose a temperature and seed; inspect the selected
tokens and their probabilities:

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 4 \
  --sample --temperature 0.7 --seed 42 --logprobs --preview
```

The local control returned `13,21927,11,1879`; repeating the run preserved IDs
and scores. Replay requires the same model execution and sampling policy, not
just the same seed. Without sampling flags, selection is greedy.

**Keep the schema, change the sampling policy.** Combine the same flags with
the record schema. With `jq`, extract the object and the first token's scores:

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 29 \
  --json-schema fixtures/constraints/record.json \
  --sample --temperature 0.7 --seed 42 --logprobs \
  | jq '{output: .constraint.output, first_token: .logprobs.tokens[0]}'
```

Scores distinguish raw-model, grammar-conditioned and actual sampling-policy
log probabilities, plus the grammar-allowed log mass. These are natural-log
quantities, not answer-confidence scores.

**Check that KV reuse preserves the result.** Compare each cached step with a
full forward and show timing/memory diagnostics:

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 4 \
  --verify-cache --verbose --preview
```

Verification adds reference work outside the reported generation timings;
use this for correctness checks, not speed comparisons.

**Stream the weights a layer at a time.** Keep the JSON constraint while
selecting explicit weight/staging and KV budgets:

```sh
mx gen --model "$MODEL" --input-ids 9707,11,1879 --max-tokens 29 \
  --memory-mode streamed --max-weight-bytes 81798144 --max-kv-bytes 7340032 \
  --json-schema fixtures/constraints/record.json --preview --verbose
```

In streamed mode, `--verbose` adds phase profiles for loading/conversion,
execution/readback and tiled projection. These measure host wall-clock time,
not individual GPU kernels. The budgets are logical, not a process-memory cap
or evidence of beyond-RAM serving.

See [the generation guide](DEVELOPMENT.md) for full reports, benchmark recipes
and the meanings of individual timing fields.

## Build features

| Feature | Enables |
|---|---|
| Default | Configuration and checkpoint inspection; no Metal execution |
| `metal` | Qwen execution and V4.1 operator diagnostics on Apple Silicon |
| `structured-output` with `metal` | Qwen JSON Schema constrained generation |

## What is qualified

[V4.1 operator checks](docs/experiments/v41-candidates.md) cover candidate
masks, final selection, index scores, and rotary tails against pinned official
expressions on synthetic inputs. They do not establish full-model or BF16/FP4
execution parity.

[Qwen experiments](docs/experiments/qwen-metal.md) record independent CPU
logit comparisons and measured decode changes.
[Streamed loading checks](docs/experiments/loader-qualification.md) include
teacher-forced cached prefill and appends compared with resident controls.
Streamed generation reuses the same constraints and sampling path. Its logical
budgets are not process-memory ceilings or proof of larger-than-RAM serving.

## Development

The [developer guide](DEVELOPMENT.md) covers profiling, benchmarks and
diagnostic commands. The [research reference](docs/research/README.md) tracks
source versions, reading coverage, implementation status and next tests.
The [architecture](docs/architecture.md) records the serving contract.

Checks additionally require uv, Node.js, and Ruff:

```sh
uv run scripts/check.py
uv run scripts/check.py --metal
```

These run formatting, tests, strict Clippy, rustdoc, and Python/Node harness
checks without downloading model weights. Run them sequentially.

## Limitations

Generation is a single FP32 Qwen sequence. Selection defaults to greedy;
seeded temperature sampling is opt-in, with or without a JSON schema.
Resident mode allows at most `min(model context, 512)` total prompt-plus-generated tokens;
streamed mode allows at most 32 total and separately checks weight/staging
and retained-KV budgets.
There is no canonical text-prompt/chat-template pipeline, V4.1 decoder,
HTTP serving, continuous batching, execution-backed paged KV, quantization
conversion, or tuning workflow yet. Beyond-RAM execution remains a goal,
not a demonstrated capability. V4.1 weight download is gated on its own small
text-forward numerical fixture.

## License

MIT.
