<p align="center">
  <img src="docs/assets/metallix.png" alt="" width="160" />
</p>
<h1 align="center">metallix</h1>

Metallix qualifies model layouts and builds Metal execution paths for one
Apple-Silicon Mac. Its first target is DeepSeek-V4.1-Flash.

Qwen3-0.6B is the working control: resident or streamed generation, KV reuse, and
JSON Schema constraints run on Metal through MLX. DeepSeek-V4.1 currently has
layout inspection and synthetic operator qualification, not a decoder.
An OpenAI-compatible local service is intended; there is no HTTP server yet.

## Build and try it

From the repository root, with Rust 1.87+, Apple Silicon, CMake, and a working
Xcode Metal toolchain:

```sh
cargo build -p server --release --all-features
target/release/mx --help
```

Both `mx` and `metallix` are native executables with the same CLI; no shell
alias is required. Nothing is installed on your PATH by these commands.

With an already-downloaded Qwen3-0.6B safetensors checkpoint, including its
`config.json` and `tokenizer.json`, replace the model path below:

```sh
target/release/mx gen --model /path/to/Qwen3-0.6B \
  --input-ids 9707,11,1879 --max-tokens 64 --verify-cache \
  --json-schema fixtures/constraints/record.json --logprobs --preview
```

This uses raw prompt token IDs, not a text/chat prompt. In the qualified local
run, the generated JSON was:

```json
{"status":"ready","count":1}
```

Stdout contains a JSON diagnostic report, including `constraint.output`,
`constraint.status: "validated"`, token IDs, timings, and requested scores.
The object above is the generated value, not the entire stdout report.
A token limit reached before grammar completion reports `incomplete` and exits
nonzero; only completed, independently validated JSON counts as success.

`--preview` adds a readable stderr view without changing stdout. Score colors
describe token likelihood, not answer correctness; non-TTY output and
`NO_COLOR=1` disable color. `--logprobs` includes original-model probabilities
and, with a schema, grammar-conditioned scores. `--debug` / `--verbose` add
metadata-only phase diagnostics. These options are off by default.
See [the generation guide](DEVELOPMENT.md) for limits and score semantics.

`--verify-cache` compares each cached result with a full-prefix Metal
recomputation outside the measured decode regions. It is a correctness check,
not an independent CPU reference or a serving benchmark.

For layer-streamed generation, use `--max-tokens 16 --memory-mode streamed
--max-weight-bytes 81798144 --max-kv-bytes 7340032` in the example above.
Omit `--verify-cache` to avoid loading the resident verification model.

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

Generation is a single FP32 Qwen sequence with greedy selection. Resident mode
allows `min(model context, 512)` input plus generated tokens; streamed mode
allows at most 32 and separately checks weight/staging and retained-KV budgets.
There is no canonical text-prompt/chat-template pipeline, V4.1 decoder,
HTTP serving, continuous batching, execution-backed paged KV, quantization
conversion, or tuning workflow yet. Beyond-RAM execution remains a goal,
not a demonstrated capability. V4.1 weight download is gated on its own small
text-forward numerical fixture.

## License

MIT.
