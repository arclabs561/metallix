# Command-line examples

These examples use the `mx` binary built with all features (see the
[README](../README.md#build)). `MODEL` is a local Qwen3 checkpoint directory
containing `config.json`, `tokenizer.json` and safetensors shards. `mx` and
`metallix` are the same program. `mx --help` and `mx <command> --help` list
every option.

Chat, the read-only workspace agent and the Responses endpoint are covered in
the [developer guide](../DEVELOPMENT.md#experimental-chat-agent-and-responses-control);
typed decisions in [typed decisions](typed-decisions.md); embeddings and
reranking in [embeddings](embeddings.md).

## Build features

| Feature | Enables |
| --- | --- |
| Default | Configuration and checkpoint inspection, the reduced DeepSeek runner; no Metal execution |
| `metal` | Qwen execution, serving, decisions, embeddings and V4.1 Metal diagnostics on Apple Silicon |
| `structured-output` with `metal` | JSON Schema constrained generation and verified schedule candidates |

## Complete a prompt

`mx gen` (short for `generate-qwen-metal`) encodes `--prompt` with the local
`tokenizer.json` and prints a JSON receipt on stdout. `generated_text` holds
the decoded output; `--preview` writes a bounded summary to stderr. The prompt
is plain text unless `--chat-template` renders it as one non-thinking user
message with the checkpoint's template.

```sh
mx gen --model "$MODEL" --prompt "The capital of France is" --max-tokens 8 \
  --verify-cache --preview
```

`--verify-cache` checks each cached decode step against a complete uncached
forward, outside the reported timings. Preview excerpt from a local run with
Qwen3-0.6B:

```text
finish_reason: length
output_text:  Paris. The capital of Italy is Rome
cache_checks: 8/8 passed
```

## Generate JSON that follows a schema

The schema constrains token selection. Successful output is then validated
independently against the schema and reported as `constraint.verification`.
A token budget that ends before the grammar completes reports `incomplete` and
exits nonzero.

```sh
mx gen --model "$MODEL" --prompt "Return only the status." --max-tokens 8 \
  --json-schema-inline '{"type":"string","enum":["ready","waiting"]}' --preview
```

Excerpt from a local run:

```json
{"finish_reason":"grammar_complete","generated_text":"\"waiting\"","constraint":{"status":"validated","output":"waiting"}}
```

`--verify-schedule` with `--schedule-requirements` retries sampled candidates
until one satisfies explicit interval requirements; see
[verified schedule candidates](candidate-control.md).

## Sampling and diagnostics

Sampling is opt-in and reproducible from a seed. Replay requires the same
model execution and sampling policy, not only the same seed.

```sh
mx gen --model "$MODEL" --prompt "The capital of France is" --max-tokens 4 \
  --sample --temperature 0.7 --seed 42 --logprobs
```

`--logprobs` adds selected-token natural-log probabilities; they are not
answer-confidence scores. `--input-ids 9707,11,1879` replaces `--prompt` with
raw token IDs for forward, cache and probability diagnostics. With resident
weights, `--verify-cache` also forks the KV state for each candidate decode and
compares child logits with the parent (`branch_verification` in the receipt).
`--memory-mode streamed` reads one layer at a time under explicit weight and
KV budgets, limited to 32 total tokens. The [developer guide](../DEVELOPMENT.md)
covers profiling, benchmarks and the full receipt fields.

## DeepSeek V4.1 artifacts

`mx fetch` acquires the ordinary Hugging Face layout through the `hf` CLI,
using the pinned revision in [`config/artifacts/`](../config/artifacts/).
Weights are included by default and require `--yes`; `--metadata-only` skips
them and `--dry-run` prints the plan without downloading.

```sh
mx fetch deepseek /path/to/deepseek --dry-run
mx fetch deepseek /path/to/deepseek --metadata-only
mx fetch deepseek /path/to/deepseek --yes
```

The inspect commands validate the config, tokenizer and template, index and
shard headers without loading tensor payloads:

```sh
mx inspect deepseek artifact /path/to/deepseek
mx inspect deepseek index /path/to/model.safetensors.index.json
mx inspect deepseek shard /path/to/model-00001-of-00018.safetensors
mx inspect-v41-embedding-row /path/to/shard.safetensors --row 0
```

The last command decodes one real embedding row. The older `inspect-v41-*`
spellings remain available.

## Reduced DeepSeek runner

A fixed five-block reduced model with synthetic weights runs without Metal or
a checkpoint download. It has no text tokenizer and does not run the released
checkpoint.

```sh
python3 scripts/export_v41_reduced_artifact.py \
  --source fixtures/deepseek-v41/reduced-runner-reference.json \
  --output /tmp/metallix-reduced-artifact.json
mx run-deepseek-reduced --artifact /tmp/metallix-reduced-artifact.json \
  --input-ids 0,1,2,3,4,5,6 --prefill-tokens 5
mx run-deepseek-reduced --artifact /tmp/metallix-reduced-artifact.json \
  --input-ids 0,1,2,3,4 --generation-max-new-tokens 2
```

The first replays a fixed prefill partition (`--prefill-tokens 4` selects the
alternate qualified schedule) and prints final-token logit bits per call. The
second generates greedy IDs (`generated_ids`, `stop_reason`); it is scalar-only.
The exporter refuses to overwrite an existing file. Artifacts are limited to
64 MiB encoded and 32 MiB of decoded tensors.

With the `metal` feature, `--score-execution metal-bf16`,
`--key-preparation-execution metal-prefp4` (or the older
`--key-rotary-execution metal-fp32`) and `--head-execution metal-fp32` move
individual stages to Metal and report `backend: "mixed-cpu-metal"`. Other
arithmetic stays on the CPU. These are placement diagnostics checked against
the scalar path's source-derived bounds, not speed claims. See the
[reduced V4.1 reference](research/v41-forward-reference.md) for the
qualification record.
