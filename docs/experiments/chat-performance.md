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

## Resident qualification runner

`scripts/qualify-chat.py` is the repeatable longer-context gate. It defaults to
a dry run and never starts a server or downloads a model. An operator supplies
the binary, checkpoint, endpoint, model ID, expected checkpoint revision, and a
new or empty receipt directory. It first submits an intentionally over-budget
request to obtain the server's prompt-token count, then fails if the calibrated
prompt plus the requested cap would overflow the configured context; it does
not estimate tokens from bytes or words.

The live run records three or more independent fresh CLI processes and serial
Responses requests to an already resident server; it does not issue a discarded
HTTP warmup. It requires the long CLI token IDs and text hash to be stable,
requires the long HTTP hash to match the CLI hash, and proves a resident-server
reset with matching short HTTP controls before and after the long workload.
Capped Responses requests are expected to end
`response.incomplete`, with the reported output-token count required to equal
the cap. Fresh CLI maximum RSS is reported separately from remote server
memory.

Pass `--sample-pid` only for the operator-owned server PID. It invokes macOS
`sample` for five seconds during the long HTTP workload and stores the CPU-stack
artifact beside the receipt; it is not a system-wide or GPU trace. A timeout
kills the owned CLI or harness process group and retains stdout and stderr
artifacts for diagnosis.

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

## Bounded CPU sample

An OS CPU sampler observed the owned native server for five seconds at a 1 ms
interval while 30 serial, capped 32-token Responses requests ran. The raw CPU
sample and request receipt are ignored artifacts under `artifacts/chat-profile/`.
All 30 requests produced the same output-text hash and the expected capped
incomplete terminal state.

The main server thread contributed 4,220 observations. `ChatSession::generate`
appeared in 3,432 observations and its forward append path in 3,313. The largest
leaf was MLX's `Event::wait` through an IOSurface shared event and IOKit: 2,493
observations, or about 59% of the sampled main thread and 73% of generation-path
observations. The remaining evaluated path included MLX graph evaluation, Metal
command encoding, dispatch, resource binding, and minor allocator activity.
Tiny HTTP worker threads were predominantly condition waits and socket receives;
they are transport-idle stacks, not decoder work.

This is a host CPU sample, so the event-wait stack shows synchronization with
an asynchronous Metal submission, not a GPU-kernel attribution. It rules out
another host-side greedy-selection micro-optimization as the next evidence-led
step. A future GPU investigation needs an explicitly process-isolated Metal
trace or a separately qualified MLX measurement; the existing streamed
diagnostic's verbose phase profiles are host wall-clock accounting and are not
GPU-kernel timings.

## 2,048-token context qualification

At `405b682`, the default native chat, agent, and serving context is 2,048
tokens. The resident plan reported `context_tokens: 2048` and
`planned_kv_bytes: 469,762,048`; this is a plan for retained K/V storage, not a
process-memory ceiling. The host was not isolated.

An error-only calibration constructed a 1,983-token prompt, then paired it with
a 64-token output budget for a total of 2,047. Three fresh CLI runs all kept the
same text and token IDs. Their median prefill was 271.64 ms (sample standard
deviation 1.59 ms), decode total 882.01 ms (1.73 ms), and time to first token
273.94 ms (1.55 ms). Their process RSS observations varied substantially across
fresh processes, so they are retained in the ignored receipt rather than used
as a resident-memory claim. Their maximum-RSS medians / sample standard
deviations were 986,300,416 / 7,910,724 bytes for short-before, 658,833,408 /
189,050,422 for long, and 670,400,512 / 183,707,543 for short-after. The
non-monotonic medians and large long/after variation reinforce that these are
fresh-process observations, not evidence about resident context memory.

Three serial warm Responses requests on the same long input emitted stable
output hashes and each ended at the expected 64-token output cap. Their median
TTFT was 270.31 ms (16.48 ms), completion wall time 1,094.14 ms (52.14 ms),
and inter-text-delta latency 13.07 ms (0.55 ms). These are client-observed
request timings, not kernel timings.

The 32-token short control had stable CLI text and IDs before and after the
long runs. A post-long HTTP short control likewise produced stable hashes that
matched the CLI short output, so no long-input output state leaked into the
next request. One bounded agent run invoked `read_file` on an isolated note
with filler content and returned its trailing codeword. This is a single
tool-round-trip qualification, not a general agent-quality claim.

## Reproducible resident qualifier

The maintained qualifier was run against source state `405b682` plus the
working diff, Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, and an already resident native
server. The host was not isolated. Its error-only calibration reported 1,983
prompt tokens; with a 64-token cap, the long workload total was 2,047 under the
2,048-token limit.

All three fresh CLI runs had identical generated IDs and text. All three long
Responses requests produced that same text hash, emitted 64 reported output
tokens, and ended at the expected `response.incomplete` cap. The short HTTP
control hash was identical before and after the long sequence. These are output
and state-recovery gates, not an assessment of response quality.

| Metric | Fresh CLI, three processes | Resident Responses, three serial requests |
| --- | ---: | ---: |
| Session load median / sample stdev | 471.56 / 8.30 ms | resident setup excluded |
| Prefill median / sample stdev | 271.98 / 1.40 ms | server metric retained in raw receipt |
| Decode total median / sample stdev | 902.02 / 17.02 ms | server metric retained in raw receipt |
| Time to first token median / sample stdev | 274.40 / 1.54 ms | 269.30 / 17.34 ms |
| Completion wall median / sample stdev | not separately timed | 1,110.05 / 54.89 ms |
| Inter-text-delta latency median / sample stdev | decode receipt is authoritative | 13.34 / 0.58 ms |
| Fresh-process peak RSS median / sample stdev | 974,209,024 / 9,452,252 bytes | not claimed by the HTTP client |

The qualifier's optional five-second CPU sample ran during the long HTTP
sequence. That sequence's measured wall time was 3,409.37 ms, so the sample
also had up to 1,590.63 ms after the requests completed; it is not a pure decode
profile. Sampling can also perturb those timed requests. Keep this receipt out
of unsampled before/after latency comparisons, and use its CPU stacks only to
locate a later, separately qualified investigation. The separate live HTTP
smoke passed JSON, streamed text, output-limit, post-header failure, invalid
request, function round-trip, and disconnect-then-next-request checks.

## Transport deadline regression check

Five serial, warm Responses requests with one excluded warmup used the same
32-token cap, prompt hash, output-text hash, and expected capped incomplete
terminal state before and after the transport deadline change. The client saw
median completion wall time move from 286.508 ms (sample standard deviation
1.239 ms) to 277.508 ms (1.183 ms); median inter-text-delta latency moved from
8.483 ms (0.024 ms) to 8.213 ms (0.023 ms). The requests occurred at different
times on non-isolated hardware, so this is a regression check with preserved
output, not an attributed speedup or a transport-performance claim.

The final timeout receipt also exercised stalled headers and stalled bodies:
each received HTTP 408 after about five seconds, and the next health request
returned 200. It observed immediate best-effort rejection for duplicate length
and transfer encoding (400), expectation handling (417), body cap (413), and
header cap (431). These are bounded live cases; general error-reply behavior
still needs broader protocol qualification.

## Read-tool task qualification

`scripts/qualify-agent.py` ran three fresh-process trials for each of three
synthetic tasks against the `2e07b39` release and Qwen3-0.6B snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`. The runner retained its binary and
workspace hashes, model revision, stdout, stderr, exit status, and task checks
under ignored `artifacts/agent-multitask/qualified-baseline/`.

| Task | Passing trials | Observed result |
| --- | ---: | --- |
| Literal search | 3/3 | Used `search_file` and returned the expected color |
| Two-file chain | 0/3 | Read the first file and repeated its pointer without finding the answer |
| Long-file read | 3/3 | Read the file and returned its final marker |

All nine processes exited successfully, but the task-level gate failed. Tool
name logs and answer matching are bounded evidence for these synthetic tasks;
they do not prove correct tool arguments or general coding capability. The
current control model needs stronger multi-step qualification before it can
support a Codex-readiness claim. Timing here includes model loading and is not
a resident-inference benchmark.

The follow-through diagnostic at `b8b4b8a` repeated these three tasks through
Responses, once without system instructions and once with a general read-only
workspace policy to continue until the original request was answered or a
concrete blocker was found. Both chain runs again stopped after the pointer;
both search and long-read controls returned the expected answers. The six
traces are retained under ignored `artifacts/agent-followthrough/`. This was
a quality diagnostic with synthetic tool results, not a timing comparison.
It did not justify changing the agent prompt or claiming improved completion.

### Structured execution and phase costs

The `465e95f` implementation was qualified with three fresh processes per task
on the same Qwen3-0.6B revision and M3 Max host, using 256 output tokens per
turn, at most six turns, and a 2048-token context. The host was not isolated.
Receipts are retained under ignored `artifacts/agent-structured/qualified-final/`;
the release binary SHA-256 is
`e7346e3c961e41bfdda2b9a03ce58149b5a186448fc38470b79a9762295dbb8d`.
Generated-text hashes and generated-token counts matched across all three trials
of each task. These are an agent baseline, not a before/after speedup result.

| Task | Passing trials | Total prefill median / sample stdev | Total decode median / sample stdev | Generated tokens |
| --- | ---: | ---: | ---: | ---: |
| Literal search | 3/3 | 97.42 / 5.90 ms | 336.77 / 1.42 ms | 36 |
| Two-file pointer chain | 0/3 | 95.51 / 0.40 ms | 320.06 / 5.28 ms | 34 |
| Long-file read | 3/3 | 289.34 / 2.86 ms | 480.29 / 6.03 ms | 42 |
| Two-file factual join | 3/3 | 111.10 / 0.83 ms | 1234.30 / 12.33 ms | 124 |

All tasks ended after two model turns. Session load was approximately
509–514 ms by task median and occurred once per process; it is excluded from
the phase totals above. The factual join read both required files and then
searched each, adding two successful tool calls. The qualifier requires exact
names, paths, and argument hashes for the required ordered reads, but reports
successful extra calls separately. An earlier exact-list gate falsely rejected
that completed task; its raw receipts remain preserved.

The pointer chain still only read the first file and repeated its instruction,
so the overall qualification exits nonzero. A shorter or faster failed answer
is not an optimization. The join's long generated tool-call text and the
long-file task's prefill cost are useful next measurement targets; reducing
either must preserve the required evidence and answer.

## Streaming disconnect recovery baseline

This baseline used release binary `2e07b39`, before the new budget change, on
an Apple M3 Max with 128 GiB unified memory running macOS 26.6.2. The host was
not isolated. The resident `metallix-qwen3` server at `127.0.0.1:8321` used the
default 2,048-token context limit; the Responses API default output budget is
128 tokens, while this probe explicitly used 64 tokens for its long workload
and 32 for its short recovery control.

Three serial trials sent the same prompt (SHA-256
`d5abe68ebfe8211f97636c9ecdf4ed03efa468da4bf9f27de192a0791d59e762`), waited
for the first visible text delta, then closed the connection with RST. The
first deltas arrived at 58.685, 24.180, and 25.045 ms. A following health check
returned in 10.130, 9.372, and 9.372 ms (median 9.372 ms; sample standard
deviation 0.438 ms). Each health check was followed by a fresh, capped 32-token
control. All three controls reported 22 prompt tokens and 32 completion tokens,
ended with expected `incomplete` status, emitted 39 events and 152 UTF-8 output
bytes, and had text hash
`9a2afc71ae542cc8d1dbc4b156d2fcd8e01d65dc6f2bfea6408e170c9aac9d6b`. Their
total times were 313.325, 314.231, and 310.996 ms (median 313.325 ms; sample
standard deviation 1.669 ms).

Three serial uninterrupted 64-token controls used the same prompt and each
reported 22 prompt tokens and 64 completion tokens, expected capped
`incomplete` status, 71 events, 320 UTF-8 output bytes, and text hash
`33a1a362600ec209d5f17a8af03a311a423d313bd11a212dd871b01d04b12466`. Their
first-delta times were 24.396, 24.057, and 24.895 ms; total times were 606.650,
604.340, and 605.125 ms (median 605.125 ms; sample standard deviation 1.175
ms). The cap deliberately makes these controls incomplete, so their terminal
status is a workload gate rather than a failure.

The close-to-health interval is an upper bound across request-disconnect and
recovery handling. Because the socket is reset after the first delta, it does
not identify the decode phase in flight, establish GPU cancellation timing, or
infer how many tokens were generated after disconnect. Repeat the identical
three-trial protocol on a newly built resident server before comparing a budget
change; retain the caps, model ID, prompt hash, and output identity gates.

### Budget-change comparison

The matched three-trial rerun used the resident server built from `2e07b39`
plus the budget working diff. It preserved the origin, `metallix-qwen3` model
ID, prompt hash, 64/32 caps, 22 prompt tokens, expected capped `incomplete`
status, event and UTF-8-byte counts, and both output-text hashes from the
baseline. Its recovery controls likewise reported 32 completion tokens and its
uninterrupted controls reported 64.

| Client-observed metric | Before budget diff | After budget diff |
| --- | ---: | ---: |
| RST close-to-health median / sample stdev | 9.372 / 0.438 ms | 8.368 / 0.393 ms |
| Capped 32-token recovery-control wall median / sample stdev | 313.325 / 1.669 ms | 283.527 / 5.215 ms |
| Capped 64-token uninterrupted-control wall median / sample stdev | 605.125 / 1.175 ms | 542.235 / 2.663 ms |

The rerun occurred later on the same non-isolated host, with only three
observations per side. The lower client wall times are a compatible regression
check with preserved output and recovery behavior, not an attributed speedup
from the budget change. The raw after-change receipt remains at
`artifacts/cancellation/recovery-after-budget/receipt.json`.

Use this baseline before an optimization, then repeat the identical workload
after one change. For serving work, retain a tool-call workload beside this
plain-text control and report task completion separately from latency.
