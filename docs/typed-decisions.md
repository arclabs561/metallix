# Typed decisions

`mx decide` is a local direct-option-scoring surface for a Qwen checkpoint.
It accepts one state and a bounded map of typed questions.  It returns a
probability for each permitted option without generating answer tokens.

The supported question forms are:

| Type | Request criteria | Result selection |
| --- | --- | --- |
| `choice` | caller option IDs mapped to descriptions | selected caller ID |
| `score` | ordered criterion descriptions | probability-weighted zero-based expected index |
| `noul` | optional `false` and `true` descriptions | `noul` probability of `true` |

All requests have 2 through 16 options.  Choice IDs are ordered
lexicographically, score options retain their supplied order, and `noul` always
orders `false`, then `true`.  A receipt carries those option orders and a full
probability vector.  Its `usage.output_tokens` is zero because the model scores
the final option logits directly.  Its calibration object records status
`uncalibrated` and method `temperature_scaled_option_softmax`: these values are
not a validated error probability and must not drive an automated threshold as
though they were calibrated.

## Usage

Save a request as `decision.json`:

```json
{
  "state": {"color": "blue"},
  "questions": {
    "color": {
      "type": "choice",
      "instructions": "Select the color stated in the input.",
      "criteria": {"blue": "Blue", "red": "Red"}
    }
  }
}
```

```sh
mx decide --model /path/to/local/Qwen3 --request decision.json
uv run scripts/qualify-decisions.py --binary target/debug/mx --model /path/to/local/Qwen3 --execute --replay
```

The second command runs three built-in qualification cases. Add
`--jevbench-jsonl /path/to/public.jsonl` to supply public JevBench tasks.
Omitting `--execute` prints the projected requests without model execution.
Each question receives fresh KV state; the model weights remain loaded within
one invocation. Checkpoint weights are not fingerprinted in the receipt;
configuration, tokenizer, template, and request bytes are fingerprinted.

## Serving decisions over HTTP

`mx serve` answers the same requests at `POST /v1/decisions`, for Qwen and for
Julia-1 (`mx decide-julia`), with the target model named in the body:

```json
{"models": [
  {"id": "julia-1", "kind": "julia", "path": "/path/to/Julia-1"},
  {"id": "qwen", "kind": "qwen", "path": "/path/to/local/Qwen3"}
]}
```

```sh
mx serve --registry models.json
curl -s localhost:8321/v1/decisions -d '{"model": "julia-1", "state": {"color": "blue"},
  "questions": {"color": {"type": "choice", "instructions": "Select the color stated in the input.",
  "criteria": {"blue": "Blue", "red": "Red"}}}}'
```

The response is the receipt the matching command prints, labelled with the
registered ID instead of a directory; its `request_sha256` covers the HTTP body,
which includes `model`. An opt-in test finds the served and command receipts
identical apart from timing, that label and that hash. `GET /v1/models` lists
each model's capabilities.

- Qwen decisions over HTTP always use temperature 1, the `mx decide` default;
  the body has no temperature field.
- The two models keep their own option semantics: Qwen orders choice IDs
  lexicographically and applies the temperature; Julia keeps caller order and a
  plain softmax.
- Each model admits one request at a time, so a Julia decision runs while a
  Qwen generation is in progress. Admission is checked after the request is
  read, because the body names the model. As a result `/healthz` and
  `/v1/models` answer while a model is busy, and a malformed body sent to a
  busy server gets 400 rather than 503.
- Errors: an unknown model returns 404, a model without the capability returns
  400 `unsupported_capability`, a busy model returns 503 `server_busy`, and a
  request the model rejects returns 400 with its message.

## Flattened hierarchies

A `choice` can represent a hierarchy by using full leaf paths as option IDs:

```json
{
  "state": "I was charged twice and want the duplicate payment returned.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the most specific matching support category.",
      "criteria": {
        "billing/invoice": "Questions about an invoice or receipt",
        "billing/refund": "Requests to return a payment",
        "technical/login": "Problems signing in"
      }
    }
  }
}
```

This remains one flat scoring pass, with at most 16 mutually exclusive leaves.
The caller can sum `billing/invoice` and `billing/refund` probabilities to get
mass for `billing`; ancestor and descendant labels should not both be competing
options. The path separator has no runtime routing semantics. Include an
explicit other/none leaf if the categories do not cover the intended inputs.
A normalized distribution remains conditional on the supplied options and is
not calibrated confidence. Overlapping categories need a separate multilabel
formulation; multiple `noul` questions do not enforce a joint distribution.

A decomposed hierarchy instead runs decisions at successive nodes. Leaf mass
then depends on the product of conditional branch probabilities; greedy routing
alone does not produce a complete leaf distribution. No tree traversal or
parent aggregation is built into `mx decide`.

## Qualification

`scripts/qualify-decisions.py` is a local, dry-run-first bridge.  It validates
an operator-supplied JevBench-style JSONL file, projects each record into an
`mx decide` request, and verifies receipts independently when a caller supplies
them to its pure helpers.  It checks finite normalized distributions,
complete option mappings, deterministic argmax selections, Boolean
probabilities, and zero generated answer tokens.  It does not download a model, use
network access, fit a calibration transform, or claim an official benchmark
score.

The bridge accepts the canonical JevBench task fields (`id`, `state`, typed
`question`, ordered `labels`, and `expected`).  Scores map the task's ordered
labels onto zero-based indexes.  A `noul` task maps only unambiguous
`false`/`true` or `no`/`yes` labels; any other binary labeling is rejected rather
than guessed.  Local summaries keep accuracy, expected-label probability,
coverage, abstentions, and failures separate.

JevBench’s current board release is v1.4.2.2; its scoring algorithm remains
v1.4.2. This freshness check does not change our earlier pinned public input.
The release and canonical task contract are external, versioned inputs:

- [JevBench v1.4.2.2 board and unchanged scorer](https://github.com/fstandhartinger/jevbench/blob/fd54ea7dc02bbe29c6ac8f6e015a54cdcff26805/README.md)
- [Pinned JevBench task contract](https://raw.githubusercontent.com/fstandhartinger/jevbench/d06ee95988/jevbench/tasks.py)
- [Jev Decision Index 0.2.1 data and methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json)

Only public-split tasks are accepted by this bridge.  A missing expected label
or a provenance exclusion remains covered in the local run but is excluded from
the local accuracy denominator.  Those sources use their own pinned data, availability handling, metrics, and
reporting rules.  A Metallix run is only a local adapter qualification until it
is executed through the upstream harness with its pinned version and split.
Julia-1 remains unsupported here because it needs a separate encoder and
decision-head adapter; this Qwen path is direct decoder option scoring.

## Local qualification evidence

A Qwen3-0.6B checkpoint at revision
`c1899de289a04d12100db370d81485cdf75e47ca`, temperature 1, and a 2048-token
context completed the pinned public `original.jsonl` tasks: 72/72 valid
receipts, zero execution failures, and 35/72 correct (48.6%). Three built-in
cases also reproduced identical decision content on replay, excluding timing.
The full public run was not replayed. This measures this prompt and adapter,
not an upstream official score or confidence calibration.

Public task input: [JevBench original split at d06ee95988](https://github.com/fstandhartinger/jevbench/blob/d06ee95988/datasets/public/original.jsonl),
SHA-256 `5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180`.

On the same public tasks and unchanged decision prompt, the existing local
Qwen3-4B-Instruct-2507 revision
`cdbee75f17c01a7cc42f958dc650907174af0554` completed 72/72 valid receipts with
65/72 correct (90.3%). Its context remained 2048 tokens; the logical KV budget
was 1024 MiB. These checkpoints differ in size and instruction tuning; this is
not a controlled estimate of size alone.

| Checkpoint | Choice | Boolean (`noul`) | Score | Overall |
| --- | --- | --- | --- | --- |
| Qwen3-0.6B | 15/36 | 12/24 | 8/12 | 35/72 |
| Qwen3-4B-Instruct-2507 | 34/36 | 19/24 | 12/12 | 65/72 |

Saved receipts can be revalidated and summarized without inference:

```sh
uv run scripts/qualify-decisions.py --summarize-receipt /path/to/qualification.json
```

The summary verifies the dataset hash before using its labels and recomputing
results. It separates coverage, per-type/per-family accuracy, multiclass Brier
score, negative log likelihood, and process/load/render/prefill timing. Zero
probability on the expected answer means infinite NLL; missing data is separate.
The Brier means were 0.8934 and 0.1468, respectively, and NLL means were 2.6408
and 0.8597. These are probability-quality measurements, not proof of calibration.
Process times include checkpoint loading for each task and are not resident
serving latency.
