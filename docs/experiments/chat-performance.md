# Chat serving performance ledger

`scripts/benchmark-openai.mjs` establishes the client-observed baseline for a
local server that implements the streamed Responses API. It is a harness, not
a result: do not fill this document with estimated model performance.

Run it against the exact local endpoint after the server is started by its
owner. The default excludes one warmup and records three measured requests.

```sh
node scripts/benchmark-openai.mjs \
  --url http://127.0.0.1:8321 --api responses \
  --model metallix-qwen3 --cache-condition warm \
  --prompt 'Reply with the single word: ready.'
```

Save the JSON receipt under an ignored or reviewed artifact path. Compare only
receipts with identical endpoint, model, prompt hash, parameters, cache
condition, and concurrency. The receipt records its full identity and sample
standard deviation; it intentionally records no remote RSS claim.

The `responses` mode times `POST /v1/responses` with a fixed input, temperature
zero, a fixed output cap, and SSE enabled. TTFT is measured when the first
nonempty `response.output_text.delta` arrives. Completion wall time ends after
a valid `response.completed` event. Tool-call argument events are excluded from
output text bytes. Wire `response_bytes`, decoded text UTF-8 bytes, and
server-reported output tokens are different quantities and must not be
substituted for one another.

Any HTTP failure, malformed JSON event, server error event, invalid usage,
multiple terminal events, or stream without a Responses terminal event fails
the run and emits no receipt. `response.incomplete` is a valid terminal event
with a distinct `status: "incomplete"` receipt and a nonzero exit; it is not
silently counted as a successful completion. A `[DONE]` marker may follow a
completion event for compatibility but cannot itself make a Responses stream
valid.

For the native serving baseline, use the local server's configured endpoint:

```sh
node scripts/benchmark-openai.mjs \
  --url http://127.0.0.1:8321 --api responses \
  --model metallix-qwen3 --cache-condition warm \
  --prompt 'Reply with exactly OK.'
```

## External comparison candidates

[Qwen3.8](https://huggingface.co/Qwen/Qwen3.8-27B),
[vLLM-Metal's supported-model matrix](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md),
and [Whallm](https://github.com/yanun0323/Whallm) are useful candidate sources
for a matched comparison. Their publisher claims and reported measurements are
not Metallix qualification or performance evidence. Reproduce the same
workload and record a local receipt before drawing a runtime comparison.

## Initial native resident baseline

The raw receipts are retained under ignored `artifacts/chat-baseline/`. This
baseline used Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`,
the native binary built from `a8c4251` plus session changes, greedy sampling,
and three independent fresh CLI processes compared with one excluded warmup and
three serial warm `POST /v1/responses` requests. The host was an Apple M3 Max
with 128 GiB unified memory on macOS 26.6.2, built with rustc 1.98.1, and was
not isolated from ordinary machine activity.

The output gate passed before timing was reported: all CLI turns returned the
same text and generated token IDs, and every warm Responses turn returned the
same output-text hash and three reported output tokens. The text reached EOS
after three tokens. The `--max-tokens 32` control therefore confirms the same
EOS-limited path, not a 32-token decode benchmark.

| Metric | Fresh CLI, three processes | Warm Responses, three requests |
| --- | ---: | ---: |
| Session load median / sample stdev | 531.03 / 10.50 ms | resident setup excluded |
| Prefill median / sample stdev | 37.27 / 1.02 ms | 23.18 / 0.28 ms |
| Decode total median / sample stdev | 18.63 / 0.34 ms | 17.76 / 0.45 ms |
| TTFT median / sample stdev | 37.69 / 1.02 ms | 24.49 / 0.40 ms |
| Completion wall median / sample stdev | not separately timed | 42.70 / 0.86 ms |
| Peak RSS median | 658,276,352 bytes | not claimed by the HTTP client |

The fresh process's median 531.03 ms session load dominates its startup path.
The warm request's measured model work is instead roughly 23 ms prefill plus
18 ms decode for this three-token response. This is a residency and request
setup comparison, not evidence of a faster GPU kernel. The next useful
optimization measurement needs a prompt that reliably consumes a longer token
budget, with this short EOS control retained beside it.

## 32-token capped decode control

The fixed prompt `Write a detailed explanation of how a bicycle works.` reached
the 32-token cap in every CLI and warm Responses trial. All CLI text hashes and
token IDs matched, and every Responses text hash matched the CLI output. The
Responses terminal state was intentionally `incomplete` with reason
`max_output_tokens`; this is an expected capped control, not a successful
completion. The checked-in pre-optimization HTTP receipt has null usage and
phase metrics because the benchmark harness initially discarded the terminal
event's embedded fields. It must not be backfilled or estimated. The corrected
harness retains those fields for the post-change matched rerun; the existing
32 emitted text-delta count remains distinct from reported token usage.

| Metric | Fresh CLI, three processes | Warm Responses, three requests |
| --- | ---: | ---: |
| Session load median / sample stdev | 503.93 / 196.02 ms | resident setup excluded |
| Prefill median / sample stdev | 40.57 / 2.46 ms | unavailable in pre-fix receipt |
| Decode total median / sample stdev | 281.55 / 2.07 ms | unavailable in pre-fix receipt |
| TTFT median / sample stdev | receipt-local 40.80 / 2.45 ms | 24.57 / 0.73 ms |
| Completion wall median / sample stdev | not separately timed | 304.92 / 2.87 ms |
| Inter-text-delta latency median / sample stdev | decode receipt is authoritative | 9.04 / 0.07 ms |
| Peak RSS median | 665,370,624 bytes | not claimed by the HTTP client |

This is the pre-optimization decode baseline. Re-run the matched CLI/process
and serial warm Responses recipe after a binary change, writing to a new
artifact directory, and compare only the same prompt, cap, checkpoint revision,
server model ID, and request shape.

## GPU argmax comparison

The GPU-argmax build passed the same output gate against the pre-change control:
the 32 CLI token IDs and output-text hash are identical across revisions, and
all warm Responses outputs share that hash. Both sides generated 32 tokens and
ended with the expected capped incomplete terminal state.

The matched three-request wall-time comparison does not show a performance
win: pre-change median completion wall time was 304.92 ms and post-change was
307.16 ms. Pre-change median inter-text-delta latency was 9.04 ms and
post-change was 9.08 ms. The post-change server-reported decode total was
280.70 ms (sample standard deviation 0.87 ms), while the pre-change CLI
decode-receipt median was 281.55 ms (2.07 ms); these are different request
surfaces and do not prove a kernel improvement.

Ten additional post-change serial warm requests had median completion wall time
300.57 ms and median inter-text-delta latency 8.92 ms, with 1.31 ms and
0.03 ms sample standard deviations respectively. They characterize the
post-change server's short-run variance, but lack a matching ten-request
pre-change control and therefore cannot convert the three-run result into a
speedup claim. The GPU-argmax route was removed from the delivered implementation:
its performance case is unproven. The experiment patch and optimized chat source
are retained alongside the ignored receipts for future profiling.

One bounded agent smoke check also called `list_files` and returned the first
filename accurately. It establishes a single tool-call round trip only; it is
not an agent task-success or performance result.

Use this baseline before an optimization, then repeat the identical workload
after one change. For serving work, retain a tool-call workload beside this
plain-text control and report task completion separately from latency.
